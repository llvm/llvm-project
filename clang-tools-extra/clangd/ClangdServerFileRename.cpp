//===--- ClangdServerFileRename.cpp - File rename handling ------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangdServer.h"
#include "Compiler.h"
#include "Diagnostics.h"
#include "FileRename.h"
#include "Format.h"
#include "ParsedAST.h"
#include "URI.h"
#include "index/Background.h"
#include "support/Logger.h"
#include "support/Path.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"

namespace clang {
namespace clangd {

void ClangdServer::prepareFileRename(
    llvm::ArrayRef<std::pair<Path, Path>> Renames, Callback<WorkspaceEdit> CB) {
  if (!BackgroundIdx)
    return CB(error("file rename requires background indexing"));
  if (!WorkspaceRoot)
    return CB(error("file rename requires a workspace root"));

  std::vector<std::pair<Path, Path>> OwnedRenames(Renames.begin(),
                                                  Renames.end());
  WorkScheduler->run(
      "FileRename", /*Path=*/"",
      [this, Renames = std::move(OwnedRenames), CB = std::move(CB)]() mutable {
        auto FS = getHeaderFS().view(std::nullopt);
        auto Mappings = expandFileRenames(Renames, *WorkspaceRoot, *FS);
        if (!Mappings)
          return CB(Mappings.takeError());
        if (Mappings->empty()) {
          WorkspaceEdit Result;
          Result.documentChanges.emplace();
          return CB(std::move(Result));
        }

        auto MatchesRename = [&](PathRef Candidate) {
          if (llvm::any_of(*Mappings, [&](const auto &Mapping) {
                return pathEqual(Mapping.OldPath, Candidate);
              }))
            return true;
          auto CandidateStatus = FS->status(Candidate);
          return CandidateStatus &&
                 llvm::any_of(*Mappings, [&](const auto &Mapping) {
                   return Mapping.OldIdentity == CandidateStatus->getUniqueID();
                 });
        };

        // Looking up commands triggers project discovery and queues the
        // complete compilation database for background indexing.
        for (const auto &Mapping : *Mappings)
          (void)CDB.getCompileCommand(Mapping.OldPath);
        if (!BackgroundIdx->blockUntilIdle(/*TimeoutSeconds=*/30))
          return CB(error("background index did not become ready"));

        auto Graph = BackgroundIdx->includeGraphSnapshot();
        if (!Graph)
          return CB(Graph.takeError());
        if (Graph->empty())
          return CB(error("background include graph is empty"));

        llvm::StringSet<> Candidates;
        for (const auto &Node : *Graph) {
          if (!pathEqual(Node.File, *WorkspaceRoot) &&
              !pathStartsWith(*WorkspaceRoot, Node.File))
            continue;
          // Open files are parsed from their current drafts below. Their
          // persisted graph nodes are intentionally allowed to be stale.
          if (DraftMgr.getDraft(Node.File))
            continue;
          bool Relevant = MatchesRename(Node.File);
          if (!Relevant)
            Relevant = llvm::any_of(Node.DirectIncludes, MatchesRename);
          if (!Relevant)
            continue;
          auto Buffer = FS->getBufferForFile(Node.File);
          if (!Buffer)
            return CB(error("cannot validate indexed file {0}: {1}", Node.File,
                            Buffer.getError().message()));
          if (digest(Buffer->get()->getBuffer()) != Node.Digest)
            return CB(
                error("background include graph is stale for {0}", Node.File));
          Candidates.insert(Node.File);
        }
        for (const Path &OpenFile : DraftMgr.getActiveFiles())
          if (pathEqual(OpenFile, *WorkspaceRoot) ||
              pathStartsWith(*WorkspaceRoot, OpenFile))
            Candidates.insert(OpenFile);

        std::vector<llvm::StringRef> CandidateFiles;
        CandidateFiles.reserve(Candidates.size());
        for (llvm::StringRef File : Candidates.keys())
          CandidateFiles.push_back(File);
        llvm::sort(CandidateFiles);
        WorkspaceEdit Result;
        Result.documentChanges.emplace();
        for (llvm::StringRef File : CandidateFiles) {
          std::string Contents;
          std::optional<int64_t> Version;
          auto Draft = DraftMgr.getDraft(File);
          if (Draft) {
            Contents = *Draft->Contents;
            if (!Draft->Version.empty()) {
              int64_t ParsedVersion;
              if (!llvm::to_integer(Draft->Version, ParsedVersion, 10))
                return CB(error("open file {0} has invalid version {1}", File,
                                Draft->Version));
              Version = ParsedVersion;
            }
          } else {
            auto Buffer = FS->getBufferForFile(File);
            if (!Buffer)
              return CB(error("cannot read includer {0}: {1}", File,
                              Buffer.getError().message()));
            Contents = Buffer->get()->getBuffer().str();
          }

          auto Command = CDB.getCompileCommand(File);
          if (!Command)
            return CB(
                error("no compilation command is available for {0}", File));
          ParseInputs Inputs{std::move(*Command), &getHeaderFS(),
                             std::move(Contents)};
          Inputs.Index = Index;
          Inputs.FeatureModules = FeatureModules;
          Inputs.ModulesManager = ModulesManager;
          Inputs.Opts.ImportInsertions = ImportInsertions;
          adjustParseInputs(Inputs, File);
          StoreDiags Diags;
          auto CI = buildCompilerInvocation(Inputs, Diags);
          if (!CI)
            return CB(error("cannot build compiler invocation for {0}", File));
          auto AST = ParsedAST::build(File, Inputs, std::move(CI), Diags.take(),
                                      /*Preamble=*/nullptr);
          if (!AST)
            return CB(error("cannot parse includer {0}", File));

          auto Style = getFormatStyleForFile(File, Inputs.Contents, TFS, false);
          auto Edits = renameIncludeDirectives(
              File, Inputs.Contents, AST->getIncludeStructure(),
              AST->getPreprocessor().getHeaderSearchInfo(),
              Inputs.CompileCommand.Directory, *Mappings, Style, *FS);
          if (!Edits)
            return CB(Edits.takeError());
          if (Edits->empty())
            continue;
          TextDocumentEdit &Change = Result.documentChanges->emplace_back();
          Change.textDocument = VersionedTextDocumentIdentifier{
              {URIForFile::canonicalize(File, File)}, Version};
          Change.edits = std::move(*Edits);
        }
        CB(std::move(Result));
      });
}

void ClangdServer::didRenameFiles(
    llvm::ArrayRef<std::pair<Path, Path>> Renames) {
  if (BackgroundIdx)
    if (auto Err = BackgroundIdx->invalidateAfterFileRenames(Renames))
      elog("Failed to invalidate background index after file rename: {0}",
           std::move(Err));

  struct MovedDraft {
    Path OldPath;
    Path NewPath;
    DraftStore::Draft Draft;
  };
  std::vector<MovedDraft> MovedDrafts;
  for (const Path &OpenFile : DraftMgr.getActiveFiles()) {
    auto NewPath = mapPathAfterRenames(OpenFile, Renames);
    if (!NewPath) {
      elog("Failed to map open file after rename: {0}", NewPath.takeError());
      return;
    }
    if (OpenFile == *NewPath)
      continue;
    if (DraftMgr.getDraft(*NewPath)) {
      elog("Cannot migrate open file {0}: destination {1} is already open",
           OpenFile, *NewPath);
      return;
    }
    auto Draft = DraftMgr.getDraft(OpenFile);
    assert(Draft && "active draft disappeared");
    MovedDrafts.push_back({OpenFile, std::move(*NewPath), std::move(*Draft)});
  }
  for (const MovedDraft &Moved : MovedDrafts)
    removeDocument(Moved.OldPath);
  for (const MovedDraft &Moved : MovedDrafts)
    addDocument(Moved.NewPath, *Moved.Draft.Contents, Moved.Draft.Version,
                WantDiagnostics::Auto);
  reparseOpenFilesIfNeeded([&](PathRef File) {
    return llvm::none_of(MovedDrafts, [&](const MovedDraft &Moved) {
      return pathEqual(File, Moved.NewPath);
    });
  });
}

} // namespace clangd
} // namespace clang
