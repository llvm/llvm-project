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
#include "SourceCode.h"
#include "URI.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"

using namespace clang::clangd;
using namespace clang;

namespace {

/// \brief Supports a test URI scheme with relaxed constraints for lit tests.
/// The path in a test URI will be combined with a platform-specific fake
/// directory to form an absolute path. For example, test:///a.cpp is resolved
/// C:\clangd-test\a.cpp on Windows and /clangd-test/a.cpp on Unix.
class TestScheme : public URIScheme {
public:
  llvm::Expected<std::string>
  getAbsolutePath(llvm::StringRef /*Authority*/, llvm::StringRef Body,
                  llvm::StringRef /*HintPath*/) const override {
    using namespace llvm::sys;
    // Still require "/" in body to mimic file scheme, as we want lengths of an
    // equivalent URI in both schemes to be the same.
    if (!Body.startswith("/"))
      return llvm::make_error<llvm::StringError>(
          "Expect URI body to be an absolute path starting with '/': " + Body,
          llvm::inconvertibleErrorCode());
    Body = Body.ltrim('/');
#ifdef LLVM_ON_WIN32
    constexpr char TestDir[] = "C:\\clangd-test";
#else
    constexpr char TestDir[] = "/clangd-test";
#endif
    llvm::SmallVector<char, 16> Path(Body.begin(), Body.end());
    path::native(Path);
    auto Err = fs::make_absolute(TestDir, Path);
    if (Err)
      llvm_unreachable("Failed to make absolute path in test scheme.");
    return std::string(Path.begin(), Path.end());
  }

  llvm::Expected<URI>
  uriFromAbsolutePath(llvm::StringRef AbsolutePath) const override {
    llvm_unreachable("Clangd must never create a test URI.");
  }
};

static URISchemeRegistry::Add<TestScheme>
    X("test", "Test scheme for clangd lit tests.");

TextEdit replacementToEdit(StringRef Code, const tooling::Replacement &R) {
  Range ReplacementRange = {
      offsetToPosition(Code, R.getOffset()),
      offsetToPosition(Code, R.getOffset() + R.getLength())};
  return {ReplacementRange, R.getReplacementText()};
}

std::vector<TextEdit>
replacementsToEdits(StringRef Code,
                    const std::vector<tooling::Replacement> &Replacements) {
  // Turn the replacements into the format specified by the Language Server
  // Protocol. Fuse them into one big JSON array.
  std::vector<TextEdit> Edits;
  for (const auto &R : Replacements)
    Edits.push_back(replacementToEdit(Code, R));
  return Edits;
}

std::vector<TextEdit> replacementsToEdits(StringRef Code,
                                          const tooling::Replacements &Repls) {
  std::vector<TextEdit> Edits;
  for (const auto &R : Repls)
    Edits.push_back(replacementToEdit(Code, R));
  return Edits;
}

} // namespace

void ClangdLSPServer::onInitialize(InitializeParams &Params) {
  if (Params.rootUri && *Params.rootUri)
    Server.setRootPath(Params.rootUri->file());
  else if (Params.rootPath && !Params.rootPath->empty())
    Server.setRootPath(*Params.rootPath);

  CCOpts.EnableSnippets =
      Params.capabilities.textDocument.completion.completionItem.snippetSupport;

  reply(json::obj{
      {{"capabilities",
        json::obj{
            {"textDocumentSync", 1},
            {"documentFormattingProvider", true},
            {"documentRangeFormattingProvider", true},
            {"documentOnTypeFormattingProvider",
             json::obj{
                 {"firstTriggerCharacter", "}"},
                 {"moreTriggerCharacter", {}},
             }},
            {"codeActionProvider", true},
            {"completionProvider",
             json::obj{
                 {"resolveProvider", false},
                 {"triggerCharacters", {".", ">", ":"}},
             }},
            {"signatureHelpProvider",
             json::obj{
                 {"triggerCharacters", {"(", ","}},
             }},
            {"definitionProvider", true},
            {"documentHighlightProvider", true},
            {"hoverProvider", true},
            {"renameProvider", true},
            {"executeCommandProvider",
             json::obj{
                 {"commands",
                  {ExecuteCommandParams::CLANGD_APPLY_FIX_COMMAND,
                   ExecuteCommandParams::CLANGD_INSERT_HEADER_INCLUDE}},
             }},
        }}}});
}

void ClangdLSPServer::onShutdown(ShutdownParams &Params) {
  // Do essentially nothing, just say we're ready to exit.
  ShutdownRequestReceived = true;
  reply(nullptr);
}

void ClangdLSPServer::onExit(ExitParams &Params) { IsDone = true; }

void ClangdLSPServer::onDocumentDidOpen(DidOpenTextDocumentParams &Params) {
  if (Params.metadata && !Params.metadata->extraFlags.empty())
    CDB.setExtraFlagsForFile(Params.textDocument.uri.file(),
                             std::move(Params.metadata->extraFlags));
  Server.addDocument(Params.textDocument.uri.file(), Params.textDocument.text,
                     WantDiagnostics::Yes);
}

void ClangdLSPServer::onDocumentDidChange(DidChangeTextDocumentParams &Params) {
  if (Params.contentChanges.size() != 1)
    return replyError(ErrorCode::InvalidParams,
                      "can only apply one change at a time");
  auto WantDiags = WantDiagnostics::Auto;
  if (Params.wantDiagnostics.hasValue())
    WantDiags = Params.wantDiagnostics.getValue() ? WantDiagnostics::Yes
                                                  : WantDiagnostics::No;
  // We only support full syncing right now.
  Server.addDocument(Params.textDocument.uri.file(),
                     Params.contentChanges[0].text, WantDiags);
}

void ClangdLSPServer::onFileEvent(DidChangeWatchedFilesParams &Params) {
  Server.onFileEvent(Params);
}

void ClangdLSPServer::onCommand(ExecuteCommandParams &Params) {
  auto ApplyEdit = [](WorkspaceEdit WE) {
    ApplyWorkspaceEditParams Edit;
    Edit.edit = std::move(WE);
    // We don't need the response so id == 1 is OK.
    // Ideally, we would wait for the response and if there is no error, we
    // would reply success/failure to the original RPC.
    call("workspace/applyEdit", Edit);
  };
  if (Params.command == ExecuteCommandParams::CLANGD_APPLY_FIX_COMMAND &&
      Params.workspaceEdit) {
    // The flow for "apply-fix" :
    // 1. We publish a diagnostic, including fixits
    // 2. The user clicks on the diagnostic, the editor asks us for code actions
    // 3. We send code actions, with the fixit embedded as context
    // 4. The user selects the fixit, the editor asks us to apply it
    // 5. We unwrap the changes and send them back to the editor
    // 6. The editor applies the changes (applyEdit), and sends us a reply (but
    // we ignore it)

    reply("Fix applied.");
    ApplyEdit(*Params.workspaceEdit);
  } else if (Params.command ==
             ExecuteCommandParams::CLANGD_INSERT_HEADER_INCLUDE) {
    auto &FileURI = Params.includeInsertion->textDocument.uri;
    auto Code = Server.getDocument(FileURI.file());
    if (!Code)
      return replyError(ErrorCode::InvalidParams,
                        ("command " +
                         ExecuteCommandParams::CLANGD_INSERT_HEADER_INCLUDE +
                         " called on non-added file " + FileURI.file())
                            .str());
    llvm::StringRef DeclaringHeader = Params.includeInsertion->declaringHeader;
    if (DeclaringHeader.empty())
      return replyError(
          ErrorCode::InvalidParams,
          "declaringHeader must be provided for include insertion.");
    llvm::StringRef PreferredHeader = Params.includeInsertion->preferredHeader;
    auto Replaces = Server.insertInclude(
        FileURI.file(), *Code, DeclaringHeader,
        PreferredHeader.empty() ? DeclaringHeader : PreferredHeader);
    if (!Replaces) {
      std::string ErrMsg =
          ("Failed to generate include insertion edits for adding " +
           DeclaringHeader + " (" + PreferredHeader + ") into " +
           FileURI.file())
              .str();
      log(ErrMsg + ":" + llvm::toString(Replaces.takeError()));
      replyError(ErrorCode::InternalError, ErrMsg);
      return;
    }
    auto Edits = replacementsToEdits(*Code, *Replaces);
    WorkspaceEdit WE;
    WE.changes = {{FileURI.uri(), Edits}};

    reply(("Inserted header " + DeclaringHeader + " (" + PreferredHeader + ")")
              .str());
    ApplyEdit(std::move(WE));
  } else {
    // We should not get here because ExecuteCommandParams would not have
    // parsed in the first place and this handler should not be called. But if
    // more commands are added, this will be here has a safe guard.
    replyError(
        ErrorCode::InvalidParams,
        llvm::formatv("Unsupported command \"{0}\".", Params.command).str());
  }
}

void ClangdLSPServer::onRename(RenameParams &Params) {
  Path File = Params.textDocument.uri.file();
  llvm::Optional<std::string> Code = Server.getDocument(File);
  if (!Code)
    return replyError(ErrorCode::InvalidParams,
                      "onRename called for non-added file");

  Server.rename(
      File, Params.position, Params.newName,
      [File, Code,
       Params](llvm::Expected<std::vector<tooling::Replacement>> Replacements) {
        if (!Replacements)
          return replyError(ErrorCode::InternalError,
                            llvm::toString(Replacements.takeError()));

        std::vector<TextEdit> Edits = replacementsToEdits(*Code, *Replacements);
        WorkspaceEdit WE;
        WE.changes = {{Params.textDocument.uri.uri(), Edits}};
        reply(WE);
      });
}

void ClangdLSPServer::onDocumentDidClose(DidCloseTextDocumentParams &Params) {
  Server.removeDocument(Params.textDocument.uri.file());
}

void ClangdLSPServer::onDocumentOnTypeFormatting(
    DocumentOnTypeFormattingParams &Params) {
  auto File = Params.textDocument.uri.file();
  auto Code = Server.getDocument(File);
  if (!Code)
    return replyError(ErrorCode::InvalidParams,
                      "onDocumentOnTypeFormatting called for non-added file");

  auto ReplacementsOrError = Server.formatOnType(*Code, File, Params.position);
  if (ReplacementsOrError)
    reply(json::ary(replacementsToEdits(*Code, ReplacementsOrError.get())));
  else
    replyError(ErrorCode::UnknownErrorCode,
               llvm::toString(ReplacementsOrError.takeError()));
}

void ClangdLSPServer::onDocumentRangeFormatting(
    DocumentRangeFormattingParams &Params) {
  auto File = Params.textDocument.uri.file();
  auto Code = Server.getDocument(File);
  if (!Code)
    return replyError(ErrorCode::InvalidParams,
                      "onDocumentRangeFormatting called for non-added file");

  auto ReplacementsOrError = Server.formatRange(*Code, File, Params.range);
  if (ReplacementsOrError)
    reply(json::ary(replacementsToEdits(*Code, ReplacementsOrError.get())));
  else
    replyError(ErrorCode::UnknownErrorCode,
               llvm::toString(ReplacementsOrError.takeError()));
}

void ClangdLSPServer::onDocumentFormatting(DocumentFormattingParams &Params) {
  auto File = Params.textDocument.uri.file();
  auto Code = Server.getDocument(File);
  if (!Code)
    return replyError(ErrorCode::InvalidParams,
                      "onDocumentFormatting called for non-added file");

  auto ReplacementsOrError = Server.formatFile(*Code, File);
  if (ReplacementsOrError)
    reply(json::ary(replacementsToEdits(*Code, ReplacementsOrError.get())));
  else
    replyError(ErrorCode::UnknownErrorCode,
               llvm::toString(ReplacementsOrError.takeError()));
}

void ClangdLSPServer::onCodeAction(CodeActionParams &Params) {
  // We provide a code action for each diagnostic at the requested location
  // which has FixIts available.
  auto Code = Server.getDocument(Params.textDocument.uri.file());
  if (!Code)
    return replyError(ErrorCode::InvalidParams,
                      "onCodeAction called for non-added file");

  json::ary Commands;
  for (Diagnostic &D : Params.context.diagnostics) {
    auto Edits = getFixIts(Params.textDocument.uri.file(), D);
    if (!Edits.empty()) {
      WorkspaceEdit WE;
      WE.changes = {{Params.textDocument.uri.uri(), std::move(Edits)}};
      Commands.push_back(json::obj{
          {"title", llvm::formatv("Apply FixIt {0}", D.message)},
          {"command", ExecuteCommandParams::CLANGD_APPLY_FIX_COMMAND},
          {"arguments", {WE}},
      });
    }
  }
  reply(std::move(Commands));
}

void ClangdLSPServer::onCompletion(TextDocumentPositionParams &Params) {
  Server.codeComplete(Params.textDocument.uri.file(), Params.position, CCOpts,
                      [](Tagged<CompletionList> List) { reply(List.Value); });
}

void ClangdLSPServer::onSignatureHelp(TextDocumentPositionParams &Params) {
  Server.signatureHelp(Params.textDocument.uri.file(), Params.position,
                       [](llvm::Expected<Tagged<SignatureHelp>> SignatureHelp) {
                         if (!SignatureHelp)
                           return replyError(
                               ErrorCode::InvalidParams,
                               llvm::toString(SignatureHelp.takeError()));
                         reply(SignatureHelp->Value);
                       });
}

void ClangdLSPServer::onGoToDefinition(TextDocumentPositionParams &Params) {
  Server.findDefinitions(
      Params.textDocument.uri.file(), Params.position,
      [](llvm::Expected<Tagged<std::vector<Location>>> Items) {
        if (!Items)
          return replyError(ErrorCode::InvalidParams,
                            llvm::toString(Items.takeError()));
        reply(json::ary(Items->Value));
      });
}

void ClangdLSPServer::onSwitchSourceHeader(TextDocumentIdentifier &Params) {
  llvm::Optional<Path> Result = Server.switchSourceHeader(Params.uri.file());
  reply(Result ? URI::createFile(*Result).toString() : "");
}

void ClangdLSPServer::onDocumentHighlight(TextDocumentPositionParams &Params) {
  Server.findDocumentHighlights(
      Params.textDocument.uri.file(), Params.position,
      [](llvm::Expected<Tagged<std::vector<DocumentHighlight>>> Highlights) {
        if (!Highlights)
          return replyError(ErrorCode::InternalError,
                            llvm::toString(Highlights.takeError()));
        reply(json::ary(Highlights->Value));
      });
}

void ClangdLSPServer::onHover(TextDocumentPositionParams &Params) {
  Server.findHover(Params.textDocument.uri.file(), Params.position,
                   [](llvm::Expected<Tagged<Hover>> H) {
                     if (!H) {
                       replyError(ErrorCode::InternalError,
                                  llvm::toString(H.takeError()));
                       return;
                     }

                     reply(H->Value);
                   });
}

// FIXME: This function needs to be properly tested.
void ClangdLSPServer::onChangeConfiguration(
    DidChangeConfigurationParams &Params) {
  ClangdConfigurationParamsChange &Settings = Params.settings;

  // Compilation database change.
  if (Settings.compilationDatabasePath.hasValue()) {
    CDB.setCompileCommandsDir(Settings.compilationDatabasePath.getValue());
    Server.reparseOpenedFiles();
  }
}

ClangdLSPServer::ClangdLSPServer(JSONOutput &Out, unsigned AsyncThreadsCount,
                                 bool StorePreamblesInMemory,
                                 const clangd::CodeCompleteOptions &CCOpts,
                                 llvm::Optional<StringRef> ResourceDir,
                                 llvm::Optional<Path> CompileCommandsDir,
                                 bool BuildDynamicSymbolIndex,
                                 SymbolIndex *StaticIdx)
    : Out(Out), CDB(std::move(CompileCommandsDir)), CCOpts(CCOpts),
      Server(CDB, /*DiagConsumer=*/*this, FSProvider, AsyncThreadsCount,
             StorePreamblesInMemory, BuildDynamicSymbolIndex, StaticIdx,
             ResourceDir, /*UpdateDebounce=*/std::chrono::milliseconds(500)) {}

bool ClangdLSPServer::run(std::istream &In, JSONStreamStyle InputStyle) {
  assert(!IsDone && "Run was called before");

  // Set up JSONRPCDispatcher.
  JSONRPCDispatcher Dispatcher([](const json::Expr &Params) {
    replyError(ErrorCode::MethodNotFound, "method not found");
  });
  registerCallbackHandlers(Dispatcher, Out, /*Callbacks=*/*this);

  // Run the Language Server loop.
  runLanguageServerLoop(In, Out, InputStyle, Dispatcher, IsDone);

  // Make sure IsDone is set to true after this method exits to ensure assertion
  // at the start of the method fires if it's ever executed again.
  IsDone = true;

  return ShutdownRequestReceived;
}

std::vector<TextEdit> ClangdLSPServer::getFixIts(StringRef File,
                                                 const clangd::Diagnostic &D) {
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
  json::ary DiagnosticsJSON;

  DiagnosticToReplacementMap LocalFixIts; // Temporary storage
  for (auto &DiagWithFixes : Diagnostics.Value) {
    auto Diag = DiagWithFixes.Diag;
    DiagnosticsJSON.push_back(json::obj{
        {"range", Diag.range},
        {"severity", Diag.severity},
        {"message", Diag.message},
    });
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
  Out.writeMessage(json::obj{
      {"jsonrpc", "2.0"},
      {"method", "textDocument/publishDiagnostics"},
      {"params",
       json::obj{
           {"uri", URIForFile{File}},
           {"diagnostics", std::move(DiagnosticsJSON)},
       }},
  });
}
