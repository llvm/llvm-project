//===--- ClangdLSPServer.h - LSP server --------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGDLSPSERVER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGDLSPSERVER_H

#include "ClangdServer.h"
#include "Path.h"
#include "Protocol.h"
#include "clang/Tooling/Core/Replacement.h"

namespace clang {
namespace clangd {

class JSONOutput;

/// This class serves as an intermediate layer of LSP server implementation,
/// glueing the JSON LSP protocol layer and ClangdServer together. It doesn't
/// directly handle input from LSP client.
/// Most methods are synchronous and return their result directly, but
/// diagnostics are provided asynchronously when ready via
/// JSONOutput::writeMessage.
class ClangdLSPServer {
public:
  ClangdLSPServer(JSONOutput &Out, bool RunSynchronously);

  /// Update the document text for \p File with \p Contents, schedule update of
  /// diagnostics. Out.writeMessage will called to push diagnostics to LSP
  /// client asynchronously when they are ready.
  void openDocument(PathRef File, StringRef Contents);
  /// Stop tracking the document for \p File.
  void closeDocument(PathRef File);

  /// Run code completion synchronously.
  std::vector<CompletionItem> codeComplete(PathRef File, Position Pos);

  /// Get the fixes associated with a certain diagnostic in a specified file as
  /// replacements.
  ///
  /// This function is thread-safe. It returns a copy to avoid handing out
  /// references to unguarded data.
  std::vector<clang::tooling::Replacement>
  getFixIts(StringRef File, const clangd::Diagnostic &D);

  /// Get the current document contents stored for \p File.
  /// FIXME(ibiryukov): This function is here to allow implementation of
  /// formatCode from ProtocolHandlers.cpp. We should move formatCode to
  /// ClangdServer class and remove this function from public interface.
  std::string getDocument(PathRef File);

private:
  class LSPDiagnosticsConsumer;

  /// Function that will be called on a separate thread when diagnostics are
  /// ready. Sends the Dianostics to LSP client via Out.writeMessage and caches
  /// corresponding fixits in the FixItsMap.
  void consumeDiagnostics(PathRef File,
                          std::vector<DiagWithFixIts> Diagnostics);

  JSONOutput &Out;

  std::mutex FixItsMutex;
  typedef std::map<clangd::Diagnostic, std::vector<clang::tooling::Replacement>>
      DiagnosticToReplacementMap;
  /// Caches FixIts per file and diagnostics
  llvm::StringMap<DiagnosticToReplacementMap> FixItsMap;
  // Server must be the last member of the class to allow its destructor to exit
  // the worker thread that may otherwise run an async callback on partially
  // destructed instance of ClangdLSPServer.
  ClangdServer Server;
};

} // namespace clangd
} // namespace clang

#endif
