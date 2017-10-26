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

void ClangdLSPServer::onInitialize(Ctx C, InitializeParams &Params) {
  C.reply(
      R"({"capabilities":{
          "textDocumentSync": 1,
          "documentFormattingProvider": true,
          "documentRangeFormattingProvider": true,
          "documentOnTypeFormattingProvider": {"firstTriggerCharacter":"}","moreTriggerCharacter":[]},
          "codeActionProvider": true,
          "completionProvider": {"resolveProvider": false, "triggerCharacters": [".",">",":"]},
          "signatureHelpProvider": {"triggerCharacters": ["(",","]},
          "definitionProvider": true
        }})");
  if (Params.rootUri && !Params.rootUri->file.empty())
    Server.setRootPath(Params.rootUri->file);
  else if (Params.rootPath && !Params.rootPath->empty())
    Server.setRootPath(*Params.rootPath);
}

void ClangdLSPServer::onShutdown(Ctx C, ShutdownParams &Params) {
  // Do essentially nothing, just say we're ready to exit.
  ShutdownRequestReceived = true;
  C.reply("null");
}

void ClangdLSPServer::onExit(Ctx C, ExitParams &Params) { IsDone = true; }

void ClangdLSPServer::onDocumentDidOpen(Ctx C,
                                        DidOpenTextDocumentParams &Params) {
  if (Params.metadata && !Params.metadata->extraFlags.empty())
    CDB.setExtraFlagsForFile(Params.textDocument.uri.file,
                             std::move(Params.metadata->extraFlags));
  Server.addDocument(Params.textDocument.uri.file, Params.textDocument.text);
}

void ClangdLSPServer::onDocumentDidChange(Ctx C,
                                          DidChangeTextDocumentParams &Params) {
  if (Params.contentChanges.size() != 1)
    return C.replyError(-32602, "can only apply one change at a time");
  // We only support full syncing right now.
  Server.addDocument(Params.textDocument.uri.file,
                     Params.contentChanges[0].text);
}

void ClangdLSPServer::onFileEvent(Ctx C, DidChangeWatchedFilesParams &Params) {
  Server.onFileEvent(Params);
}

void ClangdLSPServer::onDocumentDidClose(Ctx C,
                                         DidCloseTextDocumentParams &Params) {
  Server.removeDocument(Params.textDocument.uri.file);
}

void ClangdLSPServer::onDocumentOnTypeFormatting(
    Ctx C, DocumentOnTypeFormattingParams &Params) {
  auto File = Params.textDocument.uri.file;
  std::string Code = Server.getDocument(File);
  std::string Edits =
      replacementsToEdits(Code, Server.formatOnType(File, Params.position));
  C.reply("[" + Edits + "]");
}

void ClangdLSPServer::onDocumentRangeFormatting(
    Ctx C, DocumentRangeFormattingParams &Params) {
  auto File = Params.textDocument.uri.file;
  std::string Code = Server.getDocument(File);
  std::string Edits =
      replacementsToEdits(Code, Server.formatRange(File, Params.range));
  C.reply("[" + Edits + "]");
}

void ClangdLSPServer::onDocumentFormatting(Ctx C,
                                           DocumentFormattingParams &Params) {
  auto File = Params.textDocument.uri.file;
  std::string Code = Server.getDocument(File);
  std::string Edits = replacementsToEdits(Code, Server.formatFile(File));
  C.reply("[" + Edits + "]");
}

void ClangdLSPServer::onCodeAction(Ctx C, CodeActionParams &Params) {
  // We provide a code action for each diagnostic at the requested location
  // which has FixIts available.
  std::string Code = Server.getDocument(Params.textDocument.uri.file);
  std::string Commands;
  for (Diagnostic &D : Params.context.diagnostics) {
    std::vector<clang::tooling::Replacement> Fixes =
        getFixIts(Params.textDocument.uri.file, D);
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
  C.reply("[" + Commands + "]");
}

void ClangdLSPServer::onCompletion(Ctx C, TextDocumentPositionParams &Params) {
  auto Items = Server
                   .codeComplete(Params.textDocument.uri.file,
                                 Position{Params.position.line,
                                          Params.position.character})
                   .get() // FIXME(ibiryukov): This could be made async if we
                          // had an API that would allow to attach callbacks to
                          // futures returned by ClangdServer.
                   .Value;

  std::string Completions;
  for (const auto &Item : Items) {
    Completions += CompletionItem::unparse(Item);
    Completions += ",";
  }
  if (!Completions.empty())
    Completions.pop_back();
  C.reply("[" + Completions + "]");
}

void ClangdLSPServer::onSignatureHelp(Ctx C,
                                      TextDocumentPositionParams &Params) {
  C.reply(SignatureHelp::unparse(
      Server
          .signatureHelp(
              Params.textDocument.uri.file,
              Position{Params.position.line, Params.position.character})
          .Value));
}

void ClangdLSPServer::onGoToDefinition(Ctx C,
                                       TextDocumentPositionParams &Params) {
  auto Items = Server
                   .findDefinitions(Params.textDocument.uri.file,
                                    Position{Params.position.line,
                                             Params.position.character})
                   .Value;

  std::string Locations;
  for (const auto &Item : Items) {
    Locations += Location::unparse(Item);
    Locations += ",";
  }
  if (!Locations.empty())
    Locations.pop_back();
  C.reply("[" + Locations + "]");
}

void ClangdLSPServer::onSwitchSourceHeader(Ctx C,
                                           TextDocumentIdentifier &Params) {
  llvm::Optional<Path> Result = Server.switchSourceHeader(Params.uri.file);
  std::string ResultUri;
  C.reply(Result ? URI::unparse(URI::fromFile(*Result)) : R"("")");
}

ClangdLSPServer::ClangdLSPServer(JSONOutput &Out, unsigned AsyncThreadsCount,
                                 bool SnippetCompletions,
                                 llvm::Optional<StringRef> ResourceDir,
                                 llvm::Optional<Path> CompileCommandsDir)
    : Out(Out), CDB(/*Logger=*/Out, std::move(CompileCommandsDir)),
      Server(CDB, /*DiagConsumer=*/*this, FSProvider, AsyncThreadsCount,
             clangd::CodeCompleteOptions(
                 /*EnableSnippetsAndCodePatterns=*/SnippetCompletions),
             /*Logger=*/Out, ResourceDir) {}

bool ClangdLSPServer::run(std::istream &In) {
  assert(!IsDone && "Run was called before");

  // Set up JSONRPCDispatcher.
  JSONRPCDispatcher Dispatcher(
      [](RequestContext Ctx, llvm::yaml::MappingNode *Params) {
        Ctx.replyError(-32601, "method not found");
      });
  registerCallbackHandlers(Dispatcher, Out, /*Callbacks=*/*this);

  // Run the Language Server loop.
  runLanguageServerLoop(In, Out, Dispatcher, IsDone);

  // Make sure IsDone is set to true after this method exits to ensure assertion
  // at the start of the method fires if it's ever executed again.
  IsDone = true;

  return ShutdownRequestReceived;
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

void ClangdLSPServer::onDiagnosticsReady(
    PathRef File, Tagged<std::vector<DiagWithFixIts>> Diagnostics) {
  std::string DiagnosticsJSON;

  DiagnosticToReplacementMap LocalFixIts; // Temporary storage
  for (auto &DiagWithFixes : Diagnostics.Value) {
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
