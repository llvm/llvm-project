//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IncludeSorter.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include <algorithm>
#include <optional>

namespace clang::tidy {
namespace utils {

static StringRef removeFirstSuffix(StringRef Str,
                                   ArrayRef<const char *> Suffixes) {
  for (StringRef Suffix : Suffixes) {
    if (Str.consume_back(Suffix))
      return Str;
  }
  return Str;
}

static StringRef makeCanonicalName(StringRef Str,
                                   IncludeSorter::IncludeStyle Style) {
  // The list of suffixes to remove from source file names to get the
  // "canonical" file names.
  // E.g. tools/sort_includes.cc and tools/sort_includes_test.cc
  // would both canonicalize to tools/sort_includes and tools/sort_includes.h
  // (once canonicalized) will match as being the main include file associated
  // with the source files.
  if (Style == IncludeSorter::IS_LLVM) {
    return removeFirstSuffix(
        removeFirstSuffix(Str, {".cc", ".cpp", ".c", ".h", ".hpp"}), {"Test"});
  }
  if (Style == IncludeSorter::IS_Google_ObjC) {
    StringRef Canonical =
        removeFirstSuffix(removeFirstSuffix(Str, {".cc", ".cpp", ".c", ".h",
                                                  ".hpp", ".mm", ".m"}),
                          {"_unittest", "_regtest", "_test", "Test"});

    // Objective-C categories have a `+suffix` format, but should be grouped
    // with the file they are a category of.
    size_t StartIndex = Canonical.find_last_of('/');
    if (StartIndex == StringRef::npos) {
      StartIndex = 0;
    }
    return Canonical.substr(0, Canonical.find_first_of('+', StartIndex));
  }
  return removeFirstSuffix(
      removeFirstSuffix(Str, {".cc", ".cpp", ".c", ".h", ".hpp"}),
      {"_unittest", "_regtest", "_test"});
}

// Scan to the end of the line and return the offset of the next line.
static size_t findNextLine(const char *Text) {
  size_t EOLIndex = std::strcspn(Text, "\n");
  return Text[EOLIndex] == '\0' ? EOLIndex : EOLIndex + 1;
}

static IncludeSorter::IncludeKinds
determineIncludeKind(StringRef CanonicalFile, StringRef IncludeFile,
                     bool IsAngled, IncludeSorter::IncludeStyle Style) {
  // Compute the two "canonical" forms of the include's filename sans extension.
  // The first form is the include's filename without ".h" or "-inl.h" at the
  // end. The second form is the first form with "/public/" in the file path
  // replaced by "/internal/".
  if (IsAngled) {
    // If the system include (<foo>) ends with ".h", then it is a normal C-style
    // include. Otherwise assume it is a C++-style extensionless include.
    return IncludeFile.ends_with(".h") ? IncludeSorter::IK_CSystemInclude
                                       : IncludeSorter::IK_CXXSystemInclude;
  }
  StringRef CanonicalInclude = makeCanonicalName(IncludeFile, Style);
  if (CanonicalFile.ends_with(CanonicalInclude) ||
      CanonicalInclude.ends_with(CanonicalFile)) {
    return IncludeSorter::IK_MainTUInclude;
  }
  if ((Style == IncludeSorter::IS_Google) ||
      (Style == IncludeSorter::IS_Google_ObjC)) {
    std::pair<StringRef, StringRef> Parts = CanonicalInclude.split("/public/");
    StringRef FileCopy = CanonicalFile;
    if (FileCopy.consume_front(Parts.first) &&
        FileCopy.consume_back(Parts.second)) {
      // Determine the kind of this inclusion.
      if (FileCopy == "/internal/" || FileCopy == "/proto/") {
        return IncludeSorter::IK_MainTUInclude;
      }
    }
  }
  if (Style == IncludeSorter::IS_Google_ObjC) {
    if (IncludeFile.ends_with(".generated.h") ||
        IncludeFile.ends_with(".proto.h") ||
        IncludeFile.ends_with(".pbobjc.h")) {
      return IncludeSorter::IK_GeneratedInclude;
    }
  }
  return IncludeSorter::IK_NonSystemInclude;
}

static int compareHeaders(StringRef LHS, StringRef RHS,
                          IncludeSorter::IncludeStyle Style) {
  if (Style == IncludeSorter::IncludeStyle::IS_Google_ObjC) {
    const std::pair<const char *, const char *> &Mismatch =
        std::mismatch(LHS.begin(), LHS.end(), RHS.begin(), RHS.end());
    if ((Mismatch.first != LHS.end()) && (Mismatch.second != RHS.end())) {
      if ((*Mismatch.first == '.') && (*Mismatch.second == '+')) {
        return -1;
      }
      if ((*Mismatch.first == '+') && (*Mismatch.second == '.')) {
        return 1;
      }
    }
  }
  return LHS.compare(RHS);
}

IncludeSorter::IncludeSorter(const SourceManager *SourceMgr,
                             const FileID FileID, StringRef FileName,
                             IncludeStyle Style)
    : SourceMgr(SourceMgr), Style(Style), CurrentFileID(FileID),
      CanonicalFile(makeCanonicalName(FileName, Style)) {}

void IncludeSorter::addInclude(StringRef FileName, bool IsAngled,
                               SourceLocation HashLocation,
                               SourceLocation EndLocation) {
  int Offset = findNextLine(SourceMgr->getCharacterData(EndLocation));

  // Record the relevant location information for this inclusion directive.
  auto &IncludeLocation = IncludeLocations[FileName];
  IncludeLocation.push_back(
      SourceRange(HashLocation, EndLocation.getLocWithOffset(Offset)));
  SourceLocations.push_back(IncludeLocation.back());

  // Stop if this inclusion is a duplicate.
  if (IncludeLocation.size() > 1)
    return;

  // Add the included file's name to the appropriate bucket.
  IncludeKinds Kind =
      determineIncludeKind(CanonicalFile, FileName, IsAngled, Style);
  if (Kind != IK_InvalidInclude)
    IncludeBucket[Kind].push_back(FileName.str());
}

std::optional<FixItHint>
IncludeSorter::createIncludeInsertion(StringRef FileName, bool IsAngled) {
  std::string IncludeStmt;
  if (Style == IncludeStyle::IS_Google_ObjC) {
    IncludeStmt = IsAngled
                      ? llvm::Twine("#import <" + FileName + ">\n").str()
                      : llvm::Twine("#import \"" + FileName + "\"\n").str();
  } else {
    IncludeStmt = IsAngled
                      ? llvm::Twine("#include <" + FileName + ">\n").str()
                      : llvm::Twine("#include \"" + FileName + "\"\n").str();
  }
  if (SourceLocations.empty()) {
    // If there are no includes in this file, add it in the first line.
    // FIXME: insert after the file comment or the header guard, if present.
    IncludeStmt.append("\n");
    return FixItHint::CreateInsertion(
        SourceMgr->getLocForStartOfFile(CurrentFileID), IncludeStmt);
  }

  auto IncludeKind =
      determineIncludeKind(CanonicalFile, FileName, IsAngled, Style);

  if (!IncludeBucket[IncludeKind].empty()) {
    for (const std::string &IncludeEntry : IncludeBucket[IncludeKind]) {
      if (compareHeaders(FileName, IncludeEntry, Style) < 0) {
        const auto &Location = IncludeLocations[IncludeEntry][0];
        return FixItHint::CreateInsertion(Location.getBegin(), IncludeStmt);
      }
      if (FileName == IncludeEntry) {
        return std::nullopt;
      }
    }
    // FileName comes after all include entries in bucket, insert it after
    // last.
    const std::string &LastInclude = IncludeBucket[IncludeKind].back();
    SourceRange LastIncludeLocation = IncludeLocations[LastInclude].back();
    return FixItHint::CreateInsertion(LastIncludeLocation.getEnd(),
                                      IncludeStmt);
  }
  // Find the non-empty include bucket to be sorted directly above
  // 'IncludeKind'. If such a bucket exists, we'll want to sort the include
  // after that bucket. If no such bucket exists, find the first non-empty
  // include bucket in the file. In that case, we'll want to sort the include
  // before that bucket.
  IncludeKinds NonEmptyKind = IK_InvalidInclude;
  for (int I = IK_InvalidInclude - 1; I >= 0; --I) {
    if (!IncludeBucket[I].empty()) {
      NonEmptyKind = static_cast<IncludeKinds>(I);
      if (NonEmptyKind < IncludeKind)
        break;
    }
  }
  if (NonEmptyKind == IK_InvalidInclude) {
    return std::nullopt;
  }

  if (NonEmptyKind < IncludeKind) {
    // Create a block after.
    const std::string &LastInclude = IncludeBucket[NonEmptyKind].back();
    SourceRange LastIncludeLocation = IncludeLocations[LastInclude].back();
    IncludeStmt = '\n' + IncludeStmt;
    return FixItHint::CreateInsertion(LastIncludeLocation.getEnd(),
                                      IncludeStmt);
  }
  // Create a block before.
  const std::string &FirstInclude = IncludeBucket[NonEmptyKind][0];
  SourceRange FirstIncludeLocation = IncludeLocations[FirstInclude].back();
  IncludeStmt.append("\n");
  return FixItHint::CreateInsertion(FirstIncludeLocation.getBegin(),
                                    IncludeStmt);
}

} // namespace utils

llvm::ArrayRef<std::pair<utils::IncludeSorter::IncludeStyle, StringRef>>
OptionEnumMapping<utils::IncludeSorter::IncludeStyle>::getEnumMapping() {
  static constexpr std::pair<utils::IncludeSorter::IncludeStyle, StringRef>
      Mapping[] = {{utils::IncludeSorter::IS_LLVM, "llvm"},
                   {utils::IncludeSorter::IS_Google, "google"},
                   {utils::IncludeSorter::IS_Google_ObjC, "google-objc"}};
  return {Mapping};
}
} // namespace clang::tidy
