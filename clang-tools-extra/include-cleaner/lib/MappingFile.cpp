//===--- MappingFile.cpp - IWYU mapping file support ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-include-cleaner/MappingFile.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <string>
#include <vector>

namespace clang::include_cleaner {

void MappingFile::merge(MappingFile Other) {
  for (auto &E : Other.IncludeMappings)
    IncludeMappings[E.getKey()] = std::move(E.getValue());
  for (auto &E : Other.SymbolMappings)
    SymbolMappings[E.getKey()] = std::move(E.getValue());
  IncludeRegexPatterns.insert(
      IncludeRegexPatterns.end(),
      std::make_move_iterator(Other.IncludeRegexPatterns.begin()),
      std::make_move_iterator(Other.IncludeRegexPatterns.end()));
}

namespace {

// Strip surrounding <> or "" delimiters, yielding the bare path.
static std::string stripDelimiters(llvm::StringRef S) {
  S = S.trim();
  if ((S.starts_with("<") && S.ends_with(">")) ||
      (S.starts_with("\"") && S.ends_with("\"")))
    return S.substr(1, S.size() - 2).str();
  return S.str();
}

// Ensure a header spelling has angle brackets or quotes.
static std::string ensureQuoted(llvm::StringRef S) {
  S = S.trim();
  if (S.starts_with("<") || S.starts_with("\""))
    return S.str();
  return "<" + S.str() + ">";
}

struct ParseResult {
  MappingFile Mapping;
  std::vector<std::string> Refs;
};

// The four fields common to "include" and "symbol" mapping entries.
struct EntryFields {
  std::string From;
  std::string FromVisibility;
  std::string To;
  std::string ToVisibility;
};

llvm::Expected<ParseResult> parseOneFile(llvm::StringRef FilePath);

// Parses YAML content and returns a ParseResult containing the mapping data
// and raw (unresolved) ref paths. The caller resolves refs against a base dir.
llvm::Expected<ParseResult> parseContent(llvm::StringRef Content) {
  // Capture YAML diagnostics instead of printing to stderr.
  std::string DiagStr;
  llvm::raw_string_ostream DiagOS(DiagStr);
  llvm::SourceMgr SM;
  SM.setDiagHandler(
      [](const llvm::SMDiagnostic &D, void *Ctx) {
        auto *OS = static_cast<llvm::raw_string_ostream *>(Ctx);
        D.print("", *OS, false);
      },
      &DiagOS);

  llvm::yaml::Stream YAMLStream(Content, SM);
  ParseResult PR;

  // Returns the four scalar fields of an "include" or "symbol" entry, or
  // std::nullopt if the node is not a sequence of exactly 4 scalars.
  // Always drains the full sequence so the YAML stream stays consistent.
  auto ParseMappingFields =
      [](llvm::yaml::Node *N) -> std::optional<EntryFields> {
    auto *Seq = llvm::dyn_cast<llvm::yaml::SequenceNode>(N);
    if (!Seq)
      return std::nullopt;
    EntryFields E;
    std::string *Fields[] = {&E.From, &E.FromVisibility, &E.To,
                             &E.ToVisibility};
    int Idx = 0;
    bool Invalid = false;
    for (llvm::yaml::Node &Item : *Seq) {
      auto *S = llvm::dyn_cast<llvm::yaml::ScalarNode>(&Item);
      if (!S) {
        Invalid = true;
      } else if (Idx < 4) {
        llvm::SmallString<64> St;
        *Fields[Idx] = S->getValue(St).str();
      }
      ++Idx;
    }
    if (Invalid || Idx != 4)
      return std::nullopt;
    return E;
  };

  for (llvm::yaml::document_iterator DI = YAMLStream.begin(),
                                     DE = YAMLStream.end();
       DI != DE; ++DI) {
    llvm::yaml::Node *Root = DI->getRoot();
    if (!Root)
      break;

    auto *TopSeq = llvm::dyn_cast<llvm::yaml::SequenceNode>(Root);
    if (!TopSeq)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "expected a top-level sequence");

    for (llvm::yaml::Node &Item : *TopSeq) {
      auto *MapNode = llvm::dyn_cast<llvm::yaml::MappingNode>(&Item);
      if (!MapNode)
        continue;

      // Iterate ALL key-value pairs — partial iteration leaves the YAML
      // stream in a mid-parse state that yaml::skip() cannot recover from.
      for (llvm::yaml::KeyValueNode &KV : *MapNode) {
        auto *K = llvm::dyn_cast<llvm::yaml::ScalarNode>(KV.getKey());
        llvm::yaml::Node *EntryVal = KV.getValue();
        if (!K || !EntryVal)
          continue;

        llvm::SmallString<16> KeyStorage;
        llvm::StringRef EntryType = K->getValue(KeyStorage);

        if (EntryType == "ref") {
          auto *ValScalar = llvm::dyn_cast<llvm::yaml::ScalarNode>(EntryVal);
          if (!ValScalar)
            return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                           "'ref' value must be a string");
          llvm::SmallString<256> ValStorage;
          PR.Refs.push_back(ValScalar->getValue(ValStorage).str());
          continue;
        }

        // { "include": [from, from_vis, to, to_vis] }
        if (EntryType == "include") {
          std::optional<EntryFields> E = ParseMappingFields(EntryVal);
          if (!E)
            continue;
          // Regex patterns: IWYU uses '@' as a prefix, e.g. "@<foo/.*>".
          if (llvm::StringRef(E->From).trim().starts_with("@")) {
            llvm::StringRef Pat = llvm::StringRef(E->From).trim().drop_front(1);
            std::string RawPat = stripDelimiters(Pat);
            if (!RawPat.empty())
              PR.Mapping.IncludeRegexPatterns.push_back(
                  {RawPat, ensureQuoted(E->To)});
            continue;
          }
          std::string Key = stripDelimiters(E->From);
          if (Key.empty())
            continue;
          PR.Mapping.IncludeMappings[Key] = ensureQuoted(E->To);
          continue;
        }

        // { "symbol": [name, sym_vis, header, hdr_vis] }
        if (EntryType == "symbol") {
          std::optional<EntryFields> E = ParseMappingFields(EntryVal);
          if (!E)
            continue;
          if (E->From.empty())
            continue;
          PR.Mapping.SymbolMappings[E->From] = ensureQuoted(E->To);
          continue;
        }
      }
    }
  }

  if (YAMLStream.failed())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   DiagStr.empty() ? "invalid YAML" : DiagStr);

  return PR;
}

llvm::Expected<ParseResult> parseOneFile(llvm::StringRef FilePath) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> MBOrErr =
      llvm::MemoryBuffer::getFile(FilePath);
  if (!MBOrErr)
    return llvm::createFileError(FilePath, MBOrErr.getError());

  llvm::Expected<ParseResult> PR = parseContent((*MBOrErr)->getBuffer());
  if (!PR)
    return llvm::createFileError(FilePath, PR.takeError());

  // Resolve raw ref paths relative to this file's directory.
  llvm::StringRef Dir = llvm::sys::path::parent_path(FilePath);
  for (auto &Ref : PR->Refs) {
    if (!llvm::sys::path::is_absolute(Ref)) {
      llvm::SmallString<256> Resolved(Dir);
      llvm::sys::path::append(Resolved, Ref);
      Ref = std::string(Resolved);
    }
  }
  return PR;
}

} // namespace

llvm::Expected<MappingFile>
parseMappingFiles(llvm::ArrayRef<std::string> Paths) {
  MappingFile Result;
  std::vector<std::string> Queue(Paths.begin(), Paths.end());
  llvm::StringSet<> Visited;
  while (!Queue.empty()) {
    std::string Path = std::move(Queue.back());
    Queue.pop_back();
    if (!Visited.insert(Path).second)
      continue;
    llvm::Expected<ParseResult> PR = parseOneFile(Path);
    if (!PR)
      return PR.takeError();
    Result.merge(std::move(PR->Mapping));
    for (auto &Ref : PR->Refs)
      Queue.push_back(std::move(Ref));
  }
  return Result;
}

} // namespace clang::include_cleaner
