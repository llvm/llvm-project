//===--- ClangdLSPServer.cpp - LSP server ------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include "ClangdLSPServer.h"
#include "JSONRPCDispatcher.h"

using namespace clang::clangd;
using namespace clang;

class ClangdLSPServer::LSPDiagnosticsConsumer : public DiagnosticsConsumer {
public:
  LSPDiagnosticsConsumer(ClangdLSPServer &Server) : Server(Server) {}

  virtual void onDiagnosticsReady(PathRef File,
                                  std::vector<DiagWithFixIts> Diagnostics) {
    Server.consumeDiagnostics(File, Diagnostics);
  }

private:
  ClangdLSPServer &Server;
};

ClangdLSPServer::ClangdLSPServer(JSONOutput &Out, bool RunSynchronously)
    : Out(Out),
      Server(llvm::make_unique<DirectoryBasedGlobalCompilationDatabase>(),
             llvm::make_unique<LSPDiagnosticsConsumer>(*this),
             RunSynchronously) {}

void ClangdLSPServer::openDocument(StringRef File, StringRef Contents) {
  Server.addDocument(File, Contents);
}

void ClangdLSPServer::closeDocument(StringRef File) {
  Server.removeDocument(File);
}

std::vector<CompletionItem> ClangdLSPServer::codeComplete(PathRef File,
                                                          Position Pos) {
  return Server.codeComplete(File, Pos);
}

std::vector<clang::tooling::Replacement>
ClangdLSPServer::getFixIts(StringRef File, const clangd::Diagnostic &D) {
  std::lock_guard<std::mutex> Lock(FixItsMutex);
  auto DiagToFixItsIter = FixItsMap.find(File);
  if (DiagToFixItsIter == FixItsMap.end())
    return {};

  const auto &DiagToFixItsMap = DiagToFixItsIter->second;
  auto FixItsIter = DiagToFixItsMap.find(D);
  if (FixItsIter == DiagToFixItsMap.end())
    return {};

  return FixItsIter->second;
}

std::string ClangdLSPServer::getDocument(PathRef File) {
  return Server.getDocument(File);
}

void ClangdLSPServer::consumeDiagnostics(
    PathRef File, std::vector<DiagWithFixIts> Diagnostics) {
  std::string DiagnosticsJSON;

  DiagnosticToReplacementMap LocalFixIts; // Temporary storage
  for (auto &DiagWithFixes : Diagnostics) {
    auto Diag = DiagWithFixes.Diag;
    DiagnosticsJSON +=
        R"({"range":)" + Range::unparse(Diag.range) +
        R"(,"severity":)" + std::to_string(Diag.severity) +
        R"(,"message":")" + llvm::yaml::escape(Diag.message) +
        R"("},)";

    // We convert to Replacements to become independent of the SourceManager.
    auto &FixItsForDiagnostic = LocalFixIts[Diag];
    std::copy(DiagWithFixes.FixIts.begin(), DiagWithFixes.FixIts.end(),
              std::back_inserter(FixItsForDiagnostic));
  }

  // Cache FixIts
  {
    // FIXME(ibiryukov): should be deleted when documents are removed
    std::lock_guard<std::mutex> Lock(FixItsMutex);
    FixItsMap[File] = LocalFixIts;
  }

  // Publish diagnostics.
  if (!DiagnosticsJSON.empty())
    DiagnosticsJSON.pop_back(); // Drop trailing comma.
  Out.writeMessage(
      R"({"jsonrpc":"2.0","method":"textDocument/publishDiagnostics","params":{"uri":")" +
      URI::fromFile(File).uri + R"(","diagnostics":[)" + DiagnosticsJSON +
      R"(]}})");
}
