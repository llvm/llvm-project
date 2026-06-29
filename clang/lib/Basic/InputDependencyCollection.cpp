//===--- InputDependencyCollection.h - Searching for Resource----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// User-provided filters include/exclude profile instrumentation in certain
// functions or files.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/InputDependencyCollection.h"
#include "clang/Basic/FileManager.h"
#include "llvm/Support/Path.h"
#include <filesystem>

using namespace llvm;

namespace clang {

PatternFilter::PatternFilter(std::string Pattern)
    : Input(std::move(Pattern)), SearchRoot(""), PatternRoot(""), Pattern(""),
      PatternRegex(), RootHandling(RootPatternScanType::None), Exported(false) {
}

bool PatternFilter::InputDependencyCheck(StringRef Input, StringRef FoundDir,
                                         StringRef Filename,
                                         FileEntryRef Fileref) const {
  // TODO: in the future, more sophisticated checking for different
  // performance/security reasons may be necessary. Keep these arguments here to
  // allow minimal disruption of all call locations in the future.
  (void)Input;
  (void)FoundDir;
  (void)Fileref;
  SmallVector<StringRef, 1> Matches;
  if (PatternRegex.match(Filename)) {
    // it matches the raw filename, so we know for sure that any `../` or `./`
    // were appropriately resolved
    if (PatternRegex.match(Input, &Matches)) {
      // IF the input is ALSO a match, then we need to use this as a
      // discriminator, but we cannot scueed solely based on just this match,
      // because `..` and `./` and other filesystem-specific entities may have
      // corrupted the input. Therefore, if and only if it matches, we check if
      // it's properly anchored to the start of this string using Matches, and
      // if not? we blow it up as illegal.
      if (Matches[0].data() != Input.data()) {
        // REJECT since the #depend Input is a match but it's NOT
        // front-anchored, which is illegal/bad
        return false;
      }
    }
    return true;
  }

  return false;
}

bool PatternFilter::SimpleInputDependencyCheck(StringRef Filename) const {
  // otherwise, commit to doing a regex search against whatever we have
  if (PatternRegex.match(Filename)) {
    // it matches and we can find it
    return true;
  }

  return false;
}

PatternFilter InputDependencyCollection::ComputeFilter(std::string Pattern,
                                                       bool Exported) {
  static constexpr const char *RecursiveReplacement = ".*";
  static constexpr const std::size_t RecursiveReplacementSize = 2;
  static constexpr const char *Replacement = "[^/\\]*";
  static constexpr const std::size_t ReplacementSize = 6;
  PatternFilter Computed(std::move(Pattern));
  // Technically, this is a very conservative estimate, since this is only in
  // the most harmless of cases.
  Computed.Pattern.reserve(Computed.Input.size());
  const std::size_t InputSize = Computed.Input.size();
  std::optional<std::size_t> LastRootSeparator = std::nullopt;
  std::optional<std::size_t> LastStar = std::nullopt;
  std::optional<std::size_t> LastStarStar = std::nullopt;
  for (std::size_t I = 0; I < InputSize; ++I) {
    const char CharVal = Computed.Input[I];
    switch (CharVal) {
    case '*':
      if (I < InputSize && Computed.Input[I + 1] == '*') {
        // completely unrestricted: replace with `.*`
        ++I;
        Computed.RootHandling = static_cast<RootPatternScanType>(
            static_cast<unsigned int>(RootPatternScanType::RecursiveDirectory) |
            static_cast<unsigned int>(Computed.RootHandling));
        Computed.Pattern.append(RecursiveReplacement, RecursiveReplacementSize);
        LastStarStar = I;
      } else {
        // regular non-path-delimited changers: `[^\\/]*
        Computed.RootHandling = static_cast<RootPatternScanType>(
            static_cast<unsigned int>(
                LastStar
                    ? (*LastStar < LastRootSeparator
                           ? RootPatternScanType::DirectoryAndRecursiveDirectory
                           : RootPatternScanType::Directory)
                    : RootPatternScanType::Directory) |
            static_cast<unsigned int>(Computed.RootHandling));
        Computed.Pattern.append(Replacement, ReplacementSize);
        LastStar = I;
      }
      break;
    case ')':
    case '(':
    case '[':
    case ']':
    case '{':
    case '}':
    case '^':
    case '$':
    case '.':
    case '+':
    case '?':
    case '|':
      // cases where the character must be escaped
      Computed.Pattern.push_back('\\');
      Computed.Pattern.push_back(CharVal);
      break;
    case '/':
      if (Computed.RootHandling == RootPatternScanType::None)
        LastRootSeparator = I;
      Computed.Pattern.append("[/\\]", 4);
      break;
    case '\\':
      if (Computed.RootHandling == RootPatternScanType::None)
        LastRootSeparator = I;
      Computed.Pattern.append("[/\\]", 4);
      break;
    default:
      Computed.Pattern.push_back(CharVal);
      break;
    }
  }
  Computed.Pattern.push_back('$');
  Computed.PatternRoot.append(
      Computed.Input.cbegin(),
      Computed.Input.cbegin() +
          LastRootSeparator.value_or(static_cast<std::size_t>(0)));
  // If LastSeperator == 0 and the Input's size is Non-Zero Could be a directory
  // OR a file we're relying on... could be a bit strange to work with!
  // nevertheless, we'll treat it as a file, no reason to use `stat` and other
  // temporary checks to try and determine whether or not something is a file
  // versus a directory here.
  Computed.PatternRegex = Regex(Computed.Pattern, Regex::NoFlags);
  return Computed;
}

PatternFilter &
InputDependencyCollection::Add(std::string Pattern, bool IsAngled,
                               bool Exported, FileManager &FM,
                               const std::vector<std::string> &SearchEntries,
                               OptionalFileEntryRef LookupFrom) {
  PatternFilters.push_back(ComputeFilter(std::move(Pattern), Exported));
  PatternFilter &Filter = PatternFilters.back();
  if (llvm::sys::path::is_absolute(Filter.PatternRoot)) {
    // Absolute paths are basically already anchored, no searching to do
    Filter.SearchRoot = Filter.PatternRoot;
    return Filter;
  }
  // Find a plausible search root among the entries, if possible, to anchor this
  // to a given entry
  SmallString<256> Buffer;
  llvm::vfs::FileSystem &VFS = FM.getVirtualFileSystem();
  auto TryDetermineSearchRoot = [&](StringRef SearchEntry) -> bool {
    if (!Filter.PatternRoot.empty() &&
        SearchEntry.contains(Filter.PatternRoot)) {
      // the entry is contained within: approve the search entry as the search
      // root
      Filter.SearchRoot.assign(SearchEntry.begin(), SearchEntry.end());
      return true;
    }
    Buffer.assign(SearchEntry.begin(), SearchEntry.end());
    if (Filter.RootHandling == RootPatternScanType::None) {
      llvm::sys::path::append(Buffer, Filter.Input);
      if (VFS.exists(Buffer)) {
        // Then the search root is just the entry itself, and doesn't need the
        // pattern root Also modify the pattern root to be the *whole* found
        // file
        Filter.SearchRoot.assign(SearchEntry.begin(), SearchEntry.end());
        Filter.PatternRoot.assign(Buffer.begin(), Buffer.end());
        return true;
      }
    } else {
      llvm::sys::path::append(Buffer, Filter.PatternRoot);
      if (VFS.exists(Buffer)) {
        // if the entry can prefix the pattern root and is a proper location,
        // it is the search root
        Filter.SearchRoot.assign(Buffer.begin(), Buffer.end());
        return true;
      }
    }
    return false;
  };
  if (!IsAngled && LookupFrom) {
    // quote search; including the optional file entry as a root search location
    // too
    StringRef LookupDirName = LookupFrom->getDir().getName();
    if (TryDetermineSearchRoot(LookupDirName)) {
      return Filter;
    }
  }
  for (StringRef SearchEntry : SearchEntries) {
    if (TryDetermineSearchRoot(SearchEntry)) {
      return Filter;
    }
  }
  // if we're here, then we just need to make the SearchRoot identical to the
  // Pattern's root.
  Filter.SearchRoot = Filter.PatternRoot;
  return Filter;
}

bool InputDependencyCollection::InputDependencyCheck(
    StringRef Input, StringRef EmbedDir, StringRef Filename,
    FileEntryRef Fileref) const {
  for (const auto &Filter : PatternFilters) {
    if (Filter.InputDependencyCheck(Input, EmbedDir, Filename, Fileref)) {
      return true;
    }
  }
  return false;
}

bool InputDependencyCollection::SimpleInputDependencyCheck(
    StringRef Filename) const {
  for (const auto &Filter : PatternFilters) {
    if (Filter.SimpleInputDependencyCheck(Filename)) {
      return true;
    }
  }
  return false;
}

} // namespace clang
