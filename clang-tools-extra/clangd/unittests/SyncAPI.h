//===--- SyncAPI.h - Sync version of ClangdServer's API ----------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains synchronous versions of ClangdServer's async API. We
// deliberately don't expose the sync API outside tests to encourage using the
// async versions in clangd code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_SYNCAPI_H
#define LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_SYNCAPI_H

#include "ClangdServer.h"
#include "index/Index.h"

namespace clang {
namespace clangd {

// Calls addDocument and then blockUntilIdleForTest.
void runAddDocument(ClangdServer &Server, PathRef File, StringRef Contents,
                    StringRef Version = "null",
                    WantDiagnostics WantDiags = WantDiagnostics::Auto,
                    bool ForceRebuild = false);

llvm::Expected<CodeCompleteResult>
runCodeComplete(ClangdServer &Server, PathRef File, Position Pos,
                clangd::CodeCompleteOptions Opts);

llvm::Expected<SignatureHelp> runSignatureHelp(ClangdServer &Server,
                                               PathRef File, Position Pos);

llvm::Expected<std::vector<LocatedSymbol>>
runLocateSymbolAt(ClangdServer &Server, PathRef File, Position Pos);

llvm::Expected<std::vector<DocumentHighlight>>
runFindDocumentHighlights(ClangdServer &Server, PathRef File, Position Pos);

llvm::Expected<FileEdits> runRename(ClangdServer &Server, PathRef File,
                                    Position Pos, StringRef NewName,
                                    const clangd::RenameOptions &RenameOpts);

llvm::Expected<tooling::Replacements>
runFormatFile(ClangdServer &Server, PathRef File, StringRef Code);

std::string runDumpAST(ClangdServer &Server, PathRef File);

SymbolSlab runFuzzyFind(const SymbolIndex &Index, StringRef Query);
SymbolSlab runFuzzyFind(const SymbolIndex &Index, const FuzzyFindRequest &Req);
RefSlab getRefs(const SymbolIndex &Index, SymbolID ID);

llvm::Expected<std::vector<SelectionRange>>
runSemanticRanges(ClangdServer &Server, PathRef File,
                  const std::vector<Position> &Pos);

llvm::Expected<llvm::Optional<clangd::Path>>
runSwitchHeaderSource(ClangdServer &Server, PathRef File);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_SYNCAPI_H
