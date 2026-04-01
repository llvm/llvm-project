//===- AIIRServer.h - AIIR General Language Server --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIB_AIIR_TOOLS_AIIRLSPSERVER_SERVER_H_
#define LIB_AIIR_TOOLS_AIIRLSPSERVER_SERVER_H_

#include "Protocol.h"
#include "aiir/Support/LLVM.h"
#include "aiir/Tools/aiir-lsp-server/AiirLspRegistryFunction.h"
#include "llvm/Support/Error.h"
#include <memory>
#include <optional>

namespace aiir {
class DialectRegistry;

namespace lsp {
using llvm::lsp::CodeAction;
using llvm::lsp::CodeActionContext;
using llvm::lsp::CompletionList;
using llvm::lsp::Diagnostic;
using llvm::lsp::DocumentSymbol;
using llvm::lsp::Hover;
using llvm::lsp::Location;
using llvm::lsp::AIIRConvertBytecodeResult;
using llvm::lsp::Position;
using llvm::lsp::Range;
using llvm::lsp::URIForFile;

/// This class implements all of the AIIR related functionality necessary for a
/// language server. This class allows for keeping the AIIR specific logic
/// separate from the logic that involves LSP server/client communication.
class AIIRServer {
public:
  /// Construct a new server with the given dialect registry function.
  AIIRServer(DialectRegistryFn registry_fn);
  ~AIIRServer();

  /// Add or update the document, with the provided `version`, at the given URI.
  /// Any diagnostics emitted for this document should be added to
  /// `diagnostics`.
  void addOrUpdateDocument(const URIForFile &uri, StringRef contents,
                           int64_t version,
                           std::vector<Diagnostic> &diagnostics);

  /// Remove the document with the given uri. Returns the version of the removed
  /// document, or std::nullopt if the uri did not have a corresponding document
  /// within the server.
  std::optional<int64_t> removeDocument(const URIForFile &uri);

  /// Return the locations of the object pointed at by the given position.
  void getLocationsOf(const URIForFile &uri, const Position &defPos,
                      std::vector<Location> &locations);

  /// Find all references of the object pointed at by the given position.
  void findReferencesOf(const URIForFile &uri, const Position &pos,
                        std::vector<Location> &references);

  /// Find a hover description for the given hover position, or std::nullopt if
  /// one couldn't be found.
  std::optional<Hover> findHover(const URIForFile &uri,
                                 const Position &hoverPos);

  /// Find all of the document symbols within the given file.
  void findDocumentSymbols(const URIForFile &uri,
                           std::vector<DocumentSymbol> &symbols);

  /// Get the code completion list for the position within the given file.
  CompletionList getCodeCompletion(const URIForFile &uri,
                                   const Position &completePos);

  /// Get the set of code actions within the file.
  void getCodeActions(const URIForFile &uri, const Range &pos,
                      const CodeActionContext &context,
                      std::vector<CodeAction> &actions);

  /// Convert the given bytecode file to the textual format.
  llvm::Expected<AIIRConvertBytecodeResult>
  convertFromBytecode(const URIForFile &uri);

  /// Convert the given textual file to the bytecode format.
  llvm::Expected<AIIRConvertBytecodeResult>
  convertToBytecode(const URIForFile &uri);

private:
  struct Impl;

  std::unique_ptr<Impl> impl;
};

} // namespace lsp
} // namespace aiir

#endif // LIB_AIIR_TOOLS_AIIRLSPSERVER_SERVER_H_
