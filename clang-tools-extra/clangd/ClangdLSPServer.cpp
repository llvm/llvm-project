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
#include "ProtocolHandlers.h"

using namespace clang::clangd;
using namespace clang;

namespace {

std::string
replacementsToEdits(StringRef Code,
                    const std::vector<tooling::Replacement> &Replacements) {
  // Turn the replacements into the format specified by the Language Server
  // Protocol. Fuse them into one big JSON array.
  std::string Edits;
  for (auto &R : Replacements) {
    Range ReplacementRange = {
        offsetToPosition(Code, R.getOffset()),
        offsetToPosition(Code, R.getOffset() + R.getLength())};
    TextEdit TE = {ReplacementRange, R.getReplacementText()};
    Edits += TextEdit::unparse(TE);
    Edits += ',';
  }
  if (!Edits.empty())
    Edits.pop_back();

  return Edits;
}

} // namespace

ClangdLSPServer::LSPDiagnosticsConsumer::LSPDiagnosticsConsumer(
    ClangdLSPServer &Server)
    : Server(Server) {}

void ClangdLSPServer::LSPDiagnosticsConsumer::onDiagnosticsReady(
    PathRef File, Tagged<std::vector<DiagWithFixIts>> Diagnostics) {
  Server.consumeDiagnostics(File, Diagnostics.Value);
}

class ClangdLSPServer::LSPProtocolCallbacks : public ProtocolCallbacks {
public:
  LSPProtocolCallbacks(ClangdLSPServer &LangServer) : LangServer(LangServer) {}

  void onInitialize(StringRef ID, JSONOutput &Out) override;
  void onShutdown(JSONOutput &Out) override;
  void onDocumentDidOpen(DidOpenTextDocumentParams Params,
                         JSONOutput &Out) override;
  void onDocumentDidChange(DidChangeTextDocumentParams Params,
                           JSONOutput &Out) override;
  void onDocumentDidClose(DidCloseTextDocumentParams Params,
                          JSONOutput &Out) override;
  void onDocumentOnTypeFormatting(DocumentOnTypeFormattingParams Params,
                                  StringRef ID, JSONOutput &Out) override;
  void onDocumentRangeFormatting(DocumentRangeFormattingParams Params,
                                 StringRef ID, JSONOutput &Out) override;
  void onDocumentFormatting(DocumentFormattingParams Params, StringRef ID,
                            JSONOutput &Out) override;
  void onCodeAction(CodeActionParams Params, StringRef ID,
                    JSONOutput &Out) override;
  void onCompletion(TextDocumentPositionParams Params, StringRef ID,
                    JSONOutput &Out) override;

private:
  ClangdLSPServer &LangServer;
};

void ClangdLSPServer::LSPProtocolCallbacks::onInitialize(StringRef ID,
                                                         JSONOutput &Out) {
  Out.writeMessage(
      R"({"jsonrpc":"2.0","id":)" + ID +
      R"(,"result":{"capabilities":{
          "textDocumentSync": 1,
          "documentFormattingProvider": true,
          "documentRangeFormattingProvider": true,
          "documentOnTypeFormattingProvider": {"firstTriggerCharacter":"}","moreTriggerCharacter":[]},
          "codeActionProvider": true,
          "completionProvider": {"resolveProvider": false, "triggerCharacters": [".",">"]}
        }}})");
}

void ClangdLSPServer::LSPProtocolCallbacks::onShutdown(JSONOutput &Out) {
  LangServer.IsDone = true;
}

void ClangdLSPServer::LSPProtocolCallbacks::onDocumentDidOpen(
    DidOpenTextDocumentParams Params, JSONOutput &Out) {
  LangServer.Server.addDocument(Params.textDocument.uri.file,
                                Params.textDocument.text);
}

void ClangdLSPServer::LSPProtocolCallbacks::onDocumentDidChange(
    DidChangeTextDocumentParams Params, JSONOutput &Out) {
  // We only support full syncing right now.
  LangServer.Server.addDocument(Params.textDocument.uri.file,
                                Params.contentChanges[0].text);
}

void ClangdLSPServer::LSPProtocolCallbacks::onDocumentDidClose(
    DidCloseTextDocumentParams Params, JSONOutput &Out) {
  LangServer.Server.removeDocument(Params.textDocument.uri.file);
}

void ClangdLSPServer::LSPProtocolCallbacks::onDocumentOnTypeFormatting(
    DocumentOnTypeFormattingParams Params, StringRef ID, JSONOutput &Out) {
  auto File = Params.textDocument.uri.file;
  std::string Code = LangServer.Server.getDocument(File);
  std::string Edits = replacementsToEdits(
      Code, LangServer.Server.formatOnType(File, Params.position));

  Out.writeMessage(R"({"jsonrpc":"2.0","id":)" + ID.str() +
                   R"(,"result":[)" + Edits + R"(]})");
}

void ClangdLSPServer::LSPProtocolCallbacks::onDocumentRangeFormatting(
    DocumentRangeFormattingParams Params, StringRef ID, JSONOutput &Out) {
  auto File = Params.textDocument.uri.file;
  std::string Code = LangServer.Server.getDocument(File);
  std::string Edits = replacementsToEdits(
      Code, LangServer.Server.formatRange(File, Params.range));

  Out.writeMessage(R"({"jsonrpc":"2.0","id":)" + ID.str() +
                   R"(,"result":[)" + Edits + R"(]})");
}

void ClangdLSPServer::LSPProtocolCallbacks::onDocumentFormatting(
    DocumentFormattingParams Params, StringRef ID, JSONOutput &Out) {
  auto File = Params.textDocument.uri.file;
  std::string Code = LangServer.Server.getDocument(File);
  std::string Edits =
      replacementsToEdits(Code, LangServer.Server.formatFile(File));

  Out.writeMessage(R"({"jsonrpc":"2.0","id":)" + ID.str() +
                   R"(,"result":[)" + Edits + R"(]})");
}

void ClangdLSPServer::LSPProtocolCallbacks::onCodeAction(
    CodeActionParams Params, StringRef ID, JSONOutput &Out) {
  // We provide a code action for each diagnostic at the requested location
  // which has FixIts available.
  std::string Code =
      LangServer.Server.getDocument(Params.textDocument.uri.file);
  std::string Commands;
  for (Diagnostic &D : Params.context.diagnostics) {
    std::vector<clang::tooling::Replacement> Fixes =
        LangServer.getFixIts(Params.textDocument.uri.file, D);
    std::string Edits = replacementsToEdits(Code, Fixes);

    if (!Edits.empty())
      Commands +=
          R"({"title":"Apply FixIt ')" + llvm::yaml::escape(D.message) +
          R"('", "command": "clangd.applyFix", "arguments": [")" +
          llvm::yaml::escape(Params.textDocument.uri.uri) +
          R"(", [)" + Edits +
          R"(]]},)";
  }
  if (!Commands.empty())
    Commands.pop_back();

  Out.writeMessage(
      R"({"jsonrpc":"2.0","id":)" + ID.str() +
      R"(, "result": [)" + Commands +
      R"(]})");
}

void ClangdLSPServer::LSPProtocolCallbacks::onCompletion(
    TextDocumentPositionParams Params, StringRef ID, JSONOutput &Out) {

  auto Items = LangServer.Server.codeComplete(
      Params.textDocument.uri.file,
      Position{Params.position.line, Params.position.character}).Value;

  std::string Completions;
  for (const auto &Item : Items) {
    Completions += CompletionItem::unparse(Item);
    Completions += ",";
  }
  if (!Completions.empty())
    Completions.pop_back();
  Out.writeMessage(
      R"({"jsonrpc":"2.0","id":)" + ID.str() +
      R"(,"result":[)" + Completions + R"(]})");
}

ClangdLSPServer::ClangdLSPServer(JSONOutput &Out, bool RunSynchronously)
    : Out(Out), DiagConsumer(*this),
      Server(CDB, DiagConsumer, FSProvider, RunSynchronously) {}

void ClangdLSPServer::run(std::istream &In) {
  assert(!IsDone && "Run was called before");

  // Set up JSONRPCDispatcher.
  LSPProtocolCallbacks Callbacks(*this);
  JSONRPCDispatcher Dispatcher(llvm::make_unique<Handler>(Out));
  regiterCallbackHandlers(Dispatcher, Out, Callbacks);

  // Run the Language Server loop.
  runLanguageServerLoop(In, Out, Dispatcher, IsDone);

  // Make sure IsDone is set to true after this method exits to ensure assertion
  // at the start of the method fires if it's ever executed again.
  IsDone = true;
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
