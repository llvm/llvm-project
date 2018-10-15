//===--- SyncAPI.h - Sync version of ClangdServer's API ----------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
                    WantDiagnostics WantDiags = WantDiagnostics::Auto);

llvm::Expected<CodeCompleteResult>
runCodeComplete(ClangdServer &Server, PathRef File, Position Pos,
                clangd::CodeCompleteOptions Opts);

llvm::Expected<SignatureHelp> runSignatureHelp(ClangdServer &Server,
                                               PathRef File, Position Pos);

llvm::Expected<std::vector<Location>>
runFindDefinitions(ClangdServer &Server, PathRef File, Position Pos);

llvm::Expected<std::vector<DocumentHighlight>>
runFindDocumentHighlights(ClangdServer &Server, PathRef File, Position Pos);

llvm::Expected<std::vector<tooling::Replacement>>
runRename(ClangdServer &Server, PathRef File, Position Pos, StringRef NewName);

std::string runDumpAST(ClangdServer &Server, PathRef File);

llvm::Expected<std::vector<SymbolInformation>>
runWorkspaceSymbols(ClangdServer &Server, StringRef Query, int Limit);

llvm::Expected<std::vector<SymbolInformation>>
runDocumentSymbols(ClangdServer &Server, PathRef File);

SymbolSlab runFuzzyFind(const SymbolIndex &Index, StringRef Query);
SymbolSlab runFuzzyFind(const SymbolIndex &Index, const FuzzyFindRequest &Req);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_SYNCAPI_H
