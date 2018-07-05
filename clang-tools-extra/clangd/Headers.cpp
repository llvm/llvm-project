//===--- Headers.cpp - Include headers ---------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Headers.h"
#include "Compiler.h"
#include "Logger.h"
#include "SourceCode.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/HeaderSearch.h"
#include "llvm/Support/Path.h"

namespace clang {
namespace clangd {
namespace {

class RecordHeaders : public PPCallbacks {
public:
  RecordHeaders(const SourceManager &SM, IncludeStructure *Out)
      : SM(SM), Out(Out) {}

  // Record existing #includes - both written and resolved paths. Only #includes
  // in the main file are collected.
  void InclusionDirective(SourceLocation HashLoc, const Token & /*IncludeTok*/,
                          llvm::StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange, const FileEntry *File,
                          llvm::StringRef /*SearchPath*/,
                          llvm::StringRef /*RelativePath*/,
                          const Module * /*Imported*/,
                          SrcMgr::CharacteristicKind /*FileType*/) override {
    if (SM.isInMainFile(HashLoc))
      Out->MainFileIncludes.push_back({
          halfOpenToRange(SM, FilenameRange),
          (IsAngled ? "<" + FileName + ">" : "\"" + FileName + "\"").str(),
          File ? File->tryGetRealPathName() : "",
      });
    if (File) {
      auto *IncludingFileEntry = SM.getFileEntryForID(SM.getFileID(HashLoc));
      if (!IncludingFileEntry) {
        assert(SM.getBufferName(HashLoc).startswith("<") &&
               "Expected #include location to be a file or <built-in>");
        // Treat as if included from the main file.
        IncludingFileEntry = SM.getFileEntryForID(SM.getMainFileID());
      }
      Out->recordInclude(IncludingFileEntry->getName(), File->getName(),
                         File->tryGetRealPathName());
    }
  }

private:
  const SourceManager &SM;
  IncludeStructure *Out;
};

} // namespace

bool isLiteralInclude(llvm::StringRef Include) {
  return Include.startswith("<") || Include.startswith("\"");
}

bool HeaderFile::valid() const {
  return (Verbatim && isLiteralInclude(File)) ||
         (!Verbatim && llvm::sys::path::is_absolute(File));
}

std::unique_ptr<PPCallbacks>
collectIncludeStructureCallback(const SourceManager &SM,
                                IncludeStructure *Out) {
  return llvm::make_unique<RecordHeaders>(SM, Out);
}

void IncludeStructure::recordInclude(llvm::StringRef IncludingName,
                                     llvm::StringRef IncludedName,
                                     llvm::StringRef IncludedRealName) {
  auto Child = fileIndex(IncludedName);
  if (!IncludedRealName.empty() && RealPathNames[Child].empty())
    RealPathNames[Child] = IncludedRealName;
  auto Parent = fileIndex(IncludingName);
  IncludeChildren[Parent].push_back(Child);
}

unsigned IncludeStructure::fileIndex(llvm::StringRef Name) {
  auto R = NameToIndex.try_emplace(Name, RealPathNames.size());
  if (R.second)
    RealPathNames.emplace_back();
  return R.first->getValue();
}

llvm::StringMap<unsigned>
IncludeStructure::includeDepth(llvm::StringRef Root) const {
  // Include depth 0 is the main file only.
  llvm::StringMap<unsigned> Result;
  Result[Root] = 0;
  std::vector<unsigned> CurrentLevel;
  llvm::DenseSet<unsigned> Seen;
  auto It = NameToIndex.find(Root);
  if (It != NameToIndex.end()) {
    CurrentLevel.push_back(It->second);
    Seen.insert(It->second);
  }

  // Each round of BFS traversal finds the next depth level.
  std::vector<unsigned> PreviousLevel;
  for (unsigned Level = 1; !CurrentLevel.empty(); ++Level) {
    PreviousLevel.clear();
    PreviousLevel.swap(CurrentLevel);
    for (const auto &Parent : PreviousLevel) {
      for (const auto &Child : IncludeChildren.lookup(Parent)) {
        if (Seen.insert(Child).second) {
          CurrentLevel.push_back(Child);
          const auto &Name = RealPathNames[Child];
          // Can't include files if we don't have their real path.
          if (!Name.empty())
            Result[Name] = Level;
        }
      }
    }
  }
  return Result;
}

/// FIXME(ioeric): we might not want to insert an absolute include path if the
/// path is not shortened.
bool IncludeInserter::shouldInsertInclude(
    const HeaderFile &DeclaringHeader, const HeaderFile &InsertedHeader) const {
  assert(DeclaringHeader.valid() && InsertedHeader.valid());
  if (FileName == DeclaringHeader.File || FileName == InsertedHeader.File)
    return false;
  llvm::StringSet<> IncludedHeaders;
  for (const auto &Inc : Inclusions) {
    IncludedHeaders.insert(Inc.Written);
    if (!Inc.Resolved.empty())
      IncludedHeaders.insert(Inc.Resolved);
  }
  auto Included = [&](llvm::StringRef Header) {
    return IncludedHeaders.find(Header) != IncludedHeaders.end();
  };
  return !Included(DeclaringHeader.File) && !Included(InsertedHeader.File);
}

std::string
IncludeInserter::calculateIncludePath(const HeaderFile &DeclaringHeader,
                                      const HeaderFile &InsertedHeader) const {
  assert(DeclaringHeader.valid() && InsertedHeader.valid());
  if (InsertedHeader.Verbatim)
    return InsertedHeader.File;
  bool IsSystem = false;
  std::string Suggested = HeaderSearchInfo.suggestPathToFileForDiagnostics(
      InsertedHeader.File, BuildDir, &IsSystem);
  if (IsSystem)
    Suggested = "<" + Suggested + ">";
  else
    Suggested = "\"" + Suggested + "\"";
  return Suggested;
}

Optional<TextEdit> IncludeInserter::insert(StringRef VerbatimHeader) const {
  Optional<TextEdit> Edit = None;
  if (auto Insertion = Inserter.insert(VerbatimHeader.trim("\"<>"),
                                       VerbatimHeader.startswith("<")))
    Edit = replacementToEdit(Code, *Insertion);
  return Edit;
}

} // namespace clangd
} // namespace clang
