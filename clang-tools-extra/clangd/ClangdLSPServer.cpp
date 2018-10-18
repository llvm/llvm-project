//===--- ClangdLSPServer.cpp - LSP server ------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ClangdLSPServer.h"
#include "Diagnostics.h"
#include "SourceCode.h"
#include "Trace.h"
#include "URI.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace clang::clangd;
using namespace clang;
using namespace llvm;

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
#ifdef _WIN32
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

SymbolKindBitset defaultSymbolKinds() {
  SymbolKindBitset Defaults;
  for (size_t I = SymbolKindMin; I <= static_cast<size_t>(SymbolKind::Array);
       ++I)
    Defaults.set(I);
  return Defaults;
}

CompletionItemKindBitset defaultCompletionItemKinds() {
  CompletionItemKindBitset Defaults;
  for (size_t I = CompletionItemKindMin;
       I <= static_cast<size_t>(CompletionItemKind::Reference); ++I)
    Defaults.set(I);
  return Defaults;
}

} // namespace

// MessageHandler dispatches incoming LSP messages.
// It handles cross-cutting concerns:
//  - serializes/deserializes protocol objects to JSON
//  - logging of inbound messages
//  - cancellation handling
//  - basic call tracing
// MessageHandler ensures that initialize() is called before any other handler.
class ClangdLSPServer::MessageHandler : public Transport::MessageHandler {
public:
  MessageHandler(ClangdLSPServer &Server) : Server(Server) {}

  bool onNotify(StringRef Method, json::Value Params) override {
    log("<-- {0}", Method);
    if (Method == "exit")
      return false;
    if (!Server.Server)
      elog("Notification {0} before initialization", Method);
    else if (Method == "$/cancelRequest")
      onCancel(std::move(Params));
    else if (auto Handler = Notifications.lookup(Method))
      Handler(std::move(Params));
    else
      log("unhandled notification {0}", Method);
    return true;
  }

  bool onCall(StringRef Method, json::Value Params, json::Value ID) override {
    log("<-- {0}({1})", Method, ID);
    if (!Server.Server && Method != "initialize") {
      elog("Call {0} before initialization.", Method);
      Server.reply(ID, make_error<LSPError>("server not initialized",
                                            ErrorCode::ServerNotInitialized));
    } else if (auto Handler = Calls.lookup(Method))
      Handler(std::move(Params), std::move(ID));
    else
      Server.reply(ID, llvm::make_error<LSPError>("method not found",
                                                  ErrorCode::MethodNotFound));
    return true;
  }

  bool onReply(json::Value ID, Expected<json::Value> Result) override {
    // We ignore replies, just log them.
    if (Result)
      log("<-- reply({0})", ID);
    else
      log("<-- reply({0}) error: {1}", ID, llvm::toString(Result.takeError()));
    return true;
  }

  // Bind an LSP method name to a call.
  template <typename Param, typename Reply>
  void bind(const char *Method,
            void (ClangdLSPServer::*Handler)(const Param &, Callback<Reply>)) {
    Calls[Method] = [Method, Handler, this](json::Value RawParams,
                                            json::Value ID) {
      Param P;
      if (!fromJSON(RawParams, P)) {
        elog("Failed to decode {0} request.", Method);
        Server.reply(ID, make_error<LSPError>("failed to decode request",
                                              ErrorCode::InvalidRequest));
        return;
      }
      trace::Span Tracer(Method);
      SPAN_ATTACH(Tracer, "Params", RawParams);
      auto *Trace = Tracer.Args; // We attach reply from another thread.
      // Calls can be canceled by the client. Add cancellation context.
      WithContext WithCancel(cancelableRequestContext(ID));
      // FIXME: this function should assert it's called exactly once.
      (Server.*Handler)(P, [this, ID, Trace](llvm::Expected<Reply> Result) {
        if (Result) {
          if (Trace)
            (*Trace)["Reply"] = *Result;
          Server.reply(ID, json::Value(std::move(*Result)));
        } else {
          auto Err = Result.takeError();
          if (Trace)
            (*Trace)["Error"] = llvm::to_string(Err);
          Server.reply(ID, std::move(Err));
        }
      });
    };
  }

  // Bind an LSP method name to a notification.
  template <typename Param>
  void bind(const char *Method,
            void (ClangdLSPServer::*Handler)(const Param &)) {
    Notifications[Method] = [Method, Handler, this](json::Value RawParams) {
      Param P;
      if (!fromJSON(RawParams, P)) {
        elog("Failed to decode {0} request.", Method);
        return;
      }
      trace::Span Tracer(Method);
      SPAN_ATTACH(Tracer, "Params", RawParams);
      (Server.*Handler)(P);
    };
  }

private:
  llvm::StringMap<std::function<void(json::Value)>> Notifications;
  llvm::StringMap<std::function<void(json::Value, json::Value)>> Calls;

  // Method calls may be cancelled by ID, so keep track of their state.
  // This needs a mutex: handlers may finish on a different thread, and that's
  // when we clean up entries in the map.
  mutable std::mutex RequestCancelersMutex;
  llvm::StringMap<std::pair<Canceler, /*Cookie*/ unsigned>> RequestCancelers;
  unsigned NextRequestCookie = 0; // To disambiguate reused IDs, see below.
  void onCancel(const llvm::json::Value &Params) {
    const json::Value *ID = nullptr;
    if (auto *O = Params.getAsObject())
      ID = O->get("id");
    if (!ID) {
      elog("Bad cancellation request: {0}", Params);
      return;
    }
    auto StrID = llvm::to_string(*ID);
    std::lock_guard<std::mutex> Lock(RequestCancelersMutex);
    auto It = RequestCancelers.find(StrID);
    if (It != RequestCancelers.end())
      It->second.first(); // Invoke the canceler.
  }
  // We run cancelable requests in a context that does two things:
  //  - allows cancellation using RequestCancelers[ID]
  //  - cleans up the entry in RequestCancelers when it's no longer needed
  // If a client reuses an ID, the last wins and the first cannot be canceled.
  Context cancelableRequestContext(const json::Value &ID) {
    auto Task = cancelableTask();
    auto StrID = llvm::to_string(ID);  // JSON-serialize ID for map key.
    auto Cookie = NextRequestCookie++; // No lock, only called on main thread.
    {
      std::lock_guard<std::mutex> Lock(RequestCancelersMutex);
      RequestCancelers[StrID] = {std::move(Task.second), Cookie};
    }
    // When the request ends, we can clean up the entry we just added.
    // The cookie lets us check that it hasn't been overwritten due to ID
    // reuse.
    return Task.first.derive(make_scope_exit([this, StrID, Cookie] {
      std::lock_guard<std::mutex> Lock(RequestCancelersMutex);
      auto It = RequestCancelers.find(StrID);
      if (It != RequestCancelers.end() && It->second.second == Cookie)
        RequestCancelers.erase(It);
    }));
  }

  ClangdLSPServer &Server;
};

// call(), notify(), and reply() wrap the Transport, adding logging and locking.
void ClangdLSPServer::call(StringRef Method, json::Value Params) {
  auto ID = NextCallID++;
  log("--> {0}({1})", Method, ID);
  // We currently don't handle responses, so no need to store ID anywhere.
  std::lock_guard<std::mutex> Lock(TranspWriter);
  Transp.call(Method, std::move(Params), ID);
}

void ClangdLSPServer::notify(StringRef Method, json::Value Params) {
  log("--> {0}", Method);
  std::lock_guard<std::mutex> Lock(TranspWriter);
  Transp.notify(Method, std::move(Params));
}

void ClangdLSPServer::reply(llvm::json::Value ID,
                            llvm::Expected<llvm::json::Value> Result) {
  if (Result) {
    log("--> reply({0})", ID);
    std::lock_guard<std::mutex> Lock(TranspWriter);
    Transp.reply(std::move(ID), std::move(Result));
  } else {
    Error Err = Result.takeError();
    log("--> reply({0}) error: {1}", ID, Err);
    std::lock_guard<std::mutex> Lock(TranspWriter);
    Transp.reply(std::move(ID), std::move(Err));
  }
}

void ClangdLSPServer::onInitialize(const InitializeParams &Params,
                                   Callback<json::Value> Reply) {
  if (Server)
    return Reply(make_error<LSPError>("server already initialized",
                                      ErrorCode::InvalidRequest));
  Server.emplace(CDB.getCDB(), FSProvider,
                 static_cast<DiagnosticsConsumer &>(*this), ClangdServerOpts);
  if (Params.initializationOptions) {
    const ClangdInitializationOptions &Opts = *Params.initializationOptions;

    // Explicit compilation database path.
    if (Opts.compilationDatabasePath.hasValue()) {
      CDB.setCompileCommandsDir(Opts.compilationDatabasePath.getValue());
    }

    applyConfiguration(Opts.ParamsChange);
  }

  if (Params.rootUri && *Params.rootUri)
    Server->setRootPath(Params.rootUri->file());
  else if (Params.rootPath && !Params.rootPath->empty())
    Server->setRootPath(*Params.rootPath);

  CCOpts.EnableSnippets = Params.capabilities.CompletionSnippets;
  DiagOpts.EmbedFixesInDiagnostics = Params.capabilities.DiagnosticFixes;
  DiagOpts.SendDiagnosticCategory = Params.capabilities.DiagnosticCategory;
  if (Params.capabilities.WorkspaceSymbolKinds)
    SupportedSymbolKinds |= *Params.capabilities.WorkspaceSymbolKinds;
  if (Params.capabilities.CompletionItemKinds)
    SupportedCompletionItemKinds |= *Params.capabilities.CompletionItemKinds;
  SupportsCodeAction = Params.capabilities.CodeActionStructure;

  Reply(json::Object{
      {{"capabilities",
        json::Object{
            {"textDocumentSync", (int)TextDocumentSyncKind::Incremental},
            {"documentFormattingProvider", true},
            {"documentRangeFormattingProvider", true},
            {"documentOnTypeFormattingProvider",
             json::Object{
                 {"firstTriggerCharacter", "}"},
                 {"moreTriggerCharacter", {}},
             }},
            {"codeActionProvider", true},
            {"completionProvider",
             json::Object{
                 {"resolveProvider", false},
                 {"triggerCharacters", {".", ">", ":"}},
             }},
            {"signatureHelpProvider",
             json::Object{
                 {"triggerCharacters", {"(", ","}},
             }},
            {"definitionProvider", true},
            {"documentHighlightProvider", true},
            {"hoverProvider", true},
            {"renameProvider", true},
            {"documentSymbolProvider", true},
            {"workspaceSymbolProvider", true},
            {"referencesProvider", true},
            {"executeCommandProvider",
             json::Object{
                 {"commands", {ExecuteCommandParams::CLANGD_APPLY_FIX_COMMAND}},
             }},
        }}}});
}

void ClangdLSPServer::onShutdown(const ShutdownParams &Params,
                                 Callback<std::nullptr_t> Reply) {
  // Do essentially nothing, just say we're ready to exit.
  ShutdownRequestReceived = true;
  Reply(nullptr);
}

void ClangdLSPServer::onDocumentDidOpen(
    const DidOpenTextDocumentParams &Params) {
  PathRef File = Params.textDocument.uri.file();
  if (Params.metadata && !Params.metadata->extraFlags.empty())
    CDB.setExtraFlagsForFile(File, std::move(Params.metadata->extraFlags));

  const std::string &Contents = Params.textDocument.text;

  DraftMgr.addDraft(File, Contents);
  Server->addDocument(File, Contents, WantDiagnostics::Yes);
}

void ClangdLSPServer::onDocumentDidChange(
    const DidChangeTextDocumentParams &Params) {
  auto WantDiags = WantDiagnostics::Auto;
  if (Params.wantDiagnostics.hasValue())
    WantDiags = Params.wantDiagnostics.getValue() ? WantDiagnostics::Yes
                                                  : WantDiagnostics::No;

  PathRef File = Params.textDocument.uri.file();
  llvm::Expected<std::string> Contents =
      DraftMgr.updateDraft(File, Params.contentChanges);
  if (!Contents) {
    // If this fails, we are most likely going to be not in sync anymore with
    // the client.  It is better to remove the draft and let further operations
    // fail rather than giving wrong results.
    DraftMgr.removeDraft(File);
    Server->removeDocument(File);
    CDB.invalidate(File);
    elog("Failed to update {0}: {1}", File, Contents.takeError());
    return;
  }

  Server->addDocument(File, *Contents, WantDiags);
}

void ClangdLSPServer::onFileEvent(const DidChangeWatchedFilesParams &Params) {
  Server->onFileEvent(Params);
}

void ClangdLSPServer::onCommand(const ExecuteCommandParams &Params,
                                Callback<json::Value> Reply) {
  auto ApplyEdit = [&](WorkspaceEdit WE) {
    ApplyWorkspaceEditParams Edit;
    Edit.edit = std::move(WE);
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

    Reply("Fix applied.");
    ApplyEdit(*Params.workspaceEdit);
  } else {
    // We should not get here because ExecuteCommandParams would not have
    // parsed in the first place and this handler should not be called. But if
    // more commands are added, this will be here has a safe guard.
    Reply(make_error<LSPError>(
        llvm::formatv("Unsupported command \"{0}\".", Params.command).str(),
        ErrorCode::InvalidParams));
  }
}

void ClangdLSPServer::onWorkspaceSymbol(
    const WorkspaceSymbolParams &Params,
    Callback<std::vector<SymbolInformation>> Reply) {
  Server->workspaceSymbols(
      Params.query, CCOpts.Limit,
      Bind(
          [this](decltype(Reply) Reply,
                 llvm::Expected<std::vector<SymbolInformation>> Items) {
            if (!Items)
              return Reply(Items.takeError());
            for (auto &Sym : *Items)
              Sym.kind = adjustKindToCapability(Sym.kind, SupportedSymbolKinds);

            Reply(std::move(*Items));
          },
          std::move(Reply)));
}

void ClangdLSPServer::onRename(const RenameParams &Params,
                               Callback<WorkspaceEdit> Reply) {
  Path File = Params.textDocument.uri.file();
  llvm::Optional<std::string> Code = DraftMgr.getDraft(File);
  if (!Code)
    return Reply(make_error<LSPError>("onRename called for non-added file",
                                      ErrorCode::InvalidParams));

  Server->rename(
      File, Params.position, Params.newName,
      Bind(
          [File, Code, Params](
              decltype(Reply) Reply,
              llvm::Expected<std::vector<tooling::Replacement>> Replacements) {
            if (!Replacements)
              return Reply(Replacements.takeError());

            // Turn the replacements into the format specified by the Language
            // Server Protocol. Fuse them into one big JSON array.
            std::vector<TextEdit> Edits;
            for (const auto &R : *Replacements)
              Edits.push_back(replacementToEdit(*Code, R));
            WorkspaceEdit WE;
            WE.changes = {{Params.textDocument.uri.uri(), Edits}};
            Reply(WE);
          },
          std::move(Reply)));
}

void ClangdLSPServer::onDocumentDidClose(
    const DidCloseTextDocumentParams &Params) {
  PathRef File = Params.textDocument.uri.file();
  DraftMgr.removeDraft(File);
  Server->removeDocument(File);
  CDB.invalidate(File);
}

void ClangdLSPServer::onDocumentOnTypeFormatting(
    const DocumentOnTypeFormattingParams &Params,
    Callback<std::vector<TextEdit>> Reply) {
  auto File = Params.textDocument.uri.file();
  auto Code = DraftMgr.getDraft(File);
  if (!Code)
    return Reply(make_error<LSPError>(
        "onDocumentOnTypeFormatting called for non-added file",
        ErrorCode::InvalidParams));

  auto ReplacementsOrError = Server->formatOnType(*Code, File, Params.position);
  if (ReplacementsOrError)
    Reply(replacementsToEdits(*Code, ReplacementsOrError.get()));
  else
    Reply(ReplacementsOrError.takeError());
}

void ClangdLSPServer::onDocumentRangeFormatting(
    const DocumentRangeFormattingParams &Params,
    Callback<std::vector<TextEdit>> Reply) {
  auto File = Params.textDocument.uri.file();
  auto Code = DraftMgr.getDraft(File);
  if (!Code)
    return Reply(make_error<LSPError>(
        "onDocumentRangeFormatting called for non-added file",
        ErrorCode::InvalidParams));

  auto ReplacementsOrError = Server->formatRange(*Code, File, Params.range);
  if (ReplacementsOrError)
    Reply(replacementsToEdits(*Code, ReplacementsOrError.get()));
  else
    Reply(ReplacementsOrError.takeError());
}

void ClangdLSPServer::onDocumentFormatting(
    const DocumentFormattingParams &Params,
    Callback<std::vector<TextEdit>> Reply) {
  auto File = Params.textDocument.uri.file();
  auto Code = DraftMgr.getDraft(File);
  if (!Code)
    return Reply(
        make_error<LSPError>("onDocumentFormatting called for non-added file",
                             ErrorCode::InvalidParams));

  auto ReplacementsOrError = Server->formatFile(*Code, File);
  if (ReplacementsOrError)
    Reply(replacementsToEdits(*Code, ReplacementsOrError.get()));
  else
    Reply(ReplacementsOrError.takeError());
}

void ClangdLSPServer::onDocumentSymbol(
    const DocumentSymbolParams &Params,
    Callback<std::vector<SymbolInformation>> Reply) {
  Server->documentSymbols(
      Params.textDocument.uri.file(),
      Bind(
          [this](decltype(Reply) Reply,
                 llvm::Expected<std::vector<SymbolInformation>> Items) {
            if (!Items)
              return Reply(Items.takeError());
            for (auto &Sym : *Items)
              Sym.kind = adjustKindToCapability(Sym.kind, SupportedSymbolKinds);
            Reply(std::move(*Items));
          },
          std::move(Reply)));
}

static Optional<Command> asCommand(const CodeAction &Action) {
  Command Cmd;
  if (Action.command && Action.edit)
    return llvm::None; // Not representable. (We never emit these anyway).
  if (Action.command) {
    Cmd = *Action.command;
  } else if (Action.edit) {
    Cmd.command = Command::CLANGD_APPLY_FIX_COMMAND;
    Cmd.workspaceEdit = *Action.edit;
  } else {
    return llvm::None;
  }
  Cmd.title = Action.title;
  if (Action.kind && *Action.kind == CodeAction::QUICKFIX_KIND)
    Cmd.title = "Apply fix: " + Cmd.title;
  return Cmd;
}

void ClangdLSPServer::onCodeAction(const CodeActionParams &Params,
                                   Callback<json::Value> Reply) {
  auto Code = DraftMgr.getDraft(Params.textDocument.uri.file());
  if (!Code)
    return Reply(make_error<LSPError>("onCodeAction called for non-added file",
                                      ErrorCode::InvalidParams));
  // We provide a code action for Fixes on the specified diagnostics.
  std::vector<CodeAction> Actions;
  for (const Diagnostic &D : Params.context.diagnostics) {
    for (auto &F : getFixes(Params.textDocument.uri.file(), D)) {
      Actions.emplace_back();
      Actions.back().title = F.Message;
      Actions.back().kind = CodeAction::QUICKFIX_KIND;
      Actions.back().diagnostics = {D};
      Actions.back().edit.emplace();
      Actions.back().edit->changes.emplace();
      (*Actions.back().edit->changes)[Params.textDocument.uri.uri()] = {
          F.Edits.begin(), F.Edits.end()};
    }
  }

  if (SupportsCodeAction)
    Reply(json::Array(Actions));
  else {
    std::vector<Command> Commands;
    for (const auto &Action : Actions)
      if (auto Command = asCommand(Action))
        Commands.push_back(std::move(*Command));
    Reply(json::Array(Commands));
  }
}

void ClangdLSPServer::onCompletion(const TextDocumentPositionParams &Params,
                                   Callback<CompletionList> Reply) {
  Server->codeComplete(Params.textDocument.uri.file(), Params.position, CCOpts,
                       Bind(
                           [this](decltype(Reply) Reply,
                                  llvm::Expected<CodeCompleteResult> List) {
                             if (!List)
                               return Reply(List.takeError());
                             CompletionList LSPList;
                             LSPList.isIncomplete = List->HasMore;
                             for (const auto &R : List->Completions) {
                               CompletionItem C = R.render(CCOpts);
                               C.kind = adjustKindToCapability(
                                   C.kind, SupportedCompletionItemKinds);
                               LSPList.items.push_back(std::move(C));
                             }
                             return Reply(std::move(LSPList));
                           },
                           std::move(Reply)));
}

void ClangdLSPServer::onSignatureHelp(const TextDocumentPositionParams &Params,
                                      Callback<SignatureHelp> Reply) {
  Server->signatureHelp(Params.textDocument.uri.file(), Params.position,
                        std::move(Reply));
}

void ClangdLSPServer::onGoToDefinition(const TextDocumentPositionParams &Params,
                                       Callback<std::vector<Location>> Reply) {
  Server->findDefinitions(Params.textDocument.uri.file(), Params.position,
                          std::move(Reply));
}

void ClangdLSPServer::onSwitchSourceHeader(const TextDocumentIdentifier &Params,
                                           Callback<std::string> Reply) {
  llvm::Optional<Path> Result = Server->switchSourceHeader(Params.uri.file());
  Reply(Result ? URI::createFile(*Result).toString() : "");
}

void ClangdLSPServer::onDocumentHighlight(
    const TextDocumentPositionParams &Params,
    Callback<std::vector<DocumentHighlight>> Reply) {
  Server->findDocumentHighlights(Params.textDocument.uri.file(),
                                 Params.position, std::move(Reply));
}

void ClangdLSPServer::onHover(const TextDocumentPositionParams &Params,
                              Callback<llvm::Optional<Hover>> Reply) {
  Server->findHover(Params.textDocument.uri.file(), Params.position,
                    std::move(Reply));
}

void ClangdLSPServer::applyConfiguration(
    const ClangdConfigurationParamsChange &Params) {
  // Per-file update to the compilation database.
  if (Params.compilationDatabaseChanges) {
    const auto &CompileCommandUpdates = *Params.compilationDatabaseChanges;
    bool ShouldReparseOpenFiles = false;
    for (auto &Entry : CompileCommandUpdates) {
      /// The opened files need to be reparsed only when some existing
      /// entries are changed.
      PathRef File = Entry.first;
      if (!CDB.setCompilationCommandForFile(
              File, tooling::CompileCommand(
                        std::move(Entry.second.workingDirectory), File,
                        std::move(Entry.second.compilationCommand),
                        /*Output=*/"")))
        ShouldReparseOpenFiles = true;
    }
    if (ShouldReparseOpenFiles)
      reparseOpenedFiles();
  }
}

// FIXME: This function needs to be properly tested.
void ClangdLSPServer::onChangeConfiguration(
    const DidChangeConfigurationParams &Params) {
  applyConfiguration(Params.settings);
}

void ClangdLSPServer::onReference(const ReferenceParams &Params,
                                  Callback<std::vector<Location>> Reply) {
  Server->findReferences(Params.textDocument.uri.file(), Params.position,
                         std::move(Reply));
}

ClangdLSPServer::ClangdLSPServer(class Transport &Transp,
                                 const clangd::CodeCompleteOptions &CCOpts,
                                 llvm::Optional<Path> CompileCommandsDir,
                                 bool ShouldUseInMemoryCDB,
                                 const ClangdServer::Options &Opts)
    : Transp(Transp), MsgHandler(new MessageHandler(*this)),
      CDB(ShouldUseInMemoryCDB ? CompilationDB::makeInMemory()
                               : CompilationDB::makeDirectoryBased(
                                     std::move(CompileCommandsDir))),
      CCOpts(CCOpts), SupportedSymbolKinds(defaultSymbolKinds()),
      SupportedCompletionItemKinds(defaultCompletionItemKinds()),
      ClangdServerOpts(Opts) {
  // clang-format off
  MsgHandler->bind("initialize", &ClangdLSPServer::onInitialize);
  MsgHandler->bind("shutdown", &ClangdLSPServer::onShutdown);
  MsgHandler->bind("textDocument/rangeFormatting", &ClangdLSPServer::onDocumentRangeFormatting);
  MsgHandler->bind("textDocument/onTypeFormatting", &ClangdLSPServer::onDocumentOnTypeFormatting);
  MsgHandler->bind("textDocument/formatting", &ClangdLSPServer::onDocumentFormatting);
  MsgHandler->bind("textDocument/codeAction", &ClangdLSPServer::onCodeAction);
  MsgHandler->bind("textDocument/completion", &ClangdLSPServer::onCompletion);
  MsgHandler->bind("textDocument/signatureHelp", &ClangdLSPServer::onSignatureHelp);
  MsgHandler->bind("textDocument/definition", &ClangdLSPServer::onGoToDefinition);
  MsgHandler->bind("textDocument/references", &ClangdLSPServer::onReference);
  MsgHandler->bind("textDocument/switchSourceHeader", &ClangdLSPServer::onSwitchSourceHeader);
  MsgHandler->bind("textDocument/rename", &ClangdLSPServer::onRename);
  MsgHandler->bind("textDocument/hover", &ClangdLSPServer::onHover);
  MsgHandler->bind("textDocument/documentSymbol", &ClangdLSPServer::onDocumentSymbol);
  MsgHandler->bind("workspace/executeCommand", &ClangdLSPServer::onCommand);
  MsgHandler->bind("textDocument/documentHighlight", &ClangdLSPServer::onDocumentHighlight);
  MsgHandler->bind("workspace/symbol", &ClangdLSPServer::onWorkspaceSymbol);
  MsgHandler->bind("textDocument/didOpen", &ClangdLSPServer::onDocumentDidOpen);
  MsgHandler->bind("textDocument/didClose", &ClangdLSPServer::onDocumentDidClose);
  MsgHandler->bind("textDocument/didChange", &ClangdLSPServer::onDocumentDidChange);
  MsgHandler->bind("workspace/didChangeWatchedFiles", &ClangdLSPServer::onFileEvent);
  MsgHandler->bind("workspace/didChangeConfiguration", &ClangdLSPServer::onChangeConfiguration);
  // clang-format on
}

ClangdLSPServer::~ClangdLSPServer() = default;

bool ClangdLSPServer::run() {
  // Run the Language Server loop.
  bool CleanExit = true;
  if (auto Err = Transp.loop(*MsgHandler)) {
    elog("Transport error: {0}", std::move(Err));
    CleanExit = false;
  }

  // Destroy ClangdServer to ensure all worker threads finish.
  Server.reset();
  return CleanExit && ShutdownRequestReceived;
}

std::vector<Fix> ClangdLSPServer::getFixes(StringRef File,
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

void ClangdLSPServer::onDiagnosticsReady(PathRef File,
                                         std::vector<Diag> Diagnostics) {
  json::Array DiagnosticsJSON;

  DiagnosticToReplacementMap LocalFixIts; // Temporary storage
  for (auto &Diag : Diagnostics) {
    toLSPDiags(Diag, [&](clangd::Diagnostic Diag, llvm::ArrayRef<Fix> Fixes) {
      json::Object LSPDiag({
          {"range", Diag.range},
          {"severity", Diag.severity},
          {"message", Diag.message},
      });
      // LSP extension: embed the fixes in the diagnostic.
      if (DiagOpts.EmbedFixesInDiagnostics && !Fixes.empty()) {
        json::Array ClangdFixes;
        for (const auto &Fix : Fixes) {
          WorkspaceEdit WE;
          URIForFile URI{File};
          WE.changes = {{URI.uri(), std::vector<TextEdit>(Fix.Edits.begin(),
                                                          Fix.Edits.end())}};
          ClangdFixes.push_back(
              json::Object{{"edit", toJSON(WE)}, {"title", Fix.Message}});
        }
        LSPDiag["clangd_fixes"] = std::move(ClangdFixes);
      }
      if (DiagOpts.SendDiagnosticCategory && !Diag.category.empty())
        LSPDiag["category"] = Diag.category;
      DiagnosticsJSON.push_back(std::move(LSPDiag));

      auto &FixItsForDiagnostic = LocalFixIts[Diag];
      llvm::copy(Fixes, std::back_inserter(FixItsForDiagnostic));
    });
  }

  // Cache FixIts
  {
    // FIXME(ibiryukov): should be deleted when documents are removed
    std::lock_guard<std::mutex> Lock(FixItsMutex);
    FixItsMap[File] = LocalFixIts;
  }

  // Publish diagnostics.
  notify("textDocument/publishDiagnostics",
         json::Object{
             {"uri", URIForFile{File}},
             {"diagnostics", std::move(DiagnosticsJSON)},
         });
}

void ClangdLSPServer::reparseOpenedFiles() {
  for (const Path &FilePath : DraftMgr.getActiveFiles())
    Server->addDocument(FilePath, *DraftMgr.getDraft(FilePath),
                        WantDiagnostics::Auto);
}

ClangdLSPServer::CompilationDB ClangdLSPServer::CompilationDB::makeInMemory() {
  return CompilationDB(llvm::make_unique<InMemoryCompilationDb>(), nullptr,
                       /*IsDirectoryBased=*/false);
}

ClangdLSPServer::CompilationDB
ClangdLSPServer::CompilationDB::makeDirectoryBased(
    llvm::Optional<Path> CompileCommandsDir) {
  auto CDB = llvm::make_unique<DirectoryBasedGlobalCompilationDatabase>(
      std::move(CompileCommandsDir));
  auto CachingCDB = llvm::make_unique<CachingCompilationDb>(*CDB);
  return CompilationDB(std::move(CDB), std::move(CachingCDB),
                       /*IsDirectoryBased=*/true);
}

void ClangdLSPServer::CompilationDB::invalidate(PathRef File) {
  if (!IsDirectoryBased)
    static_cast<InMemoryCompilationDb *>(CDB.get())->invalidate(File);
  else
    CachingCDB->invalidate(File);
}

bool ClangdLSPServer::CompilationDB::setCompilationCommandForFile(
    PathRef File, tooling::CompileCommand CompilationCommand) {
  if (IsDirectoryBased) {
    elog("Trying to set compile command for {0} while using directory-based "
         "compilation database",
         File);
    return false;
  }
  return static_cast<InMemoryCompilationDb *>(CDB.get())
      ->setCompilationCommandForFile(File, std::move(CompilationCommand));
}

void ClangdLSPServer::CompilationDB::setExtraFlagsForFile(
    PathRef File, std::vector<std::string> ExtraFlags) {
  if (!IsDirectoryBased) {
    elog("Trying to set extra flags for {0} while using in-memory compilation "
         "database",
         File);
    return;
  }
  static_cast<DirectoryBasedGlobalCompilationDatabase *>(CDB.get())
      ->setExtraFlagsForFile(File, std::move(ExtraFlags));
  CachingCDB->invalidate(File);
}

void ClangdLSPServer::CompilationDB::setCompileCommandsDir(Path P) {
  if (!IsDirectoryBased) {
    elog("Trying to set compile commands dir while using in-memory compilation "
         "database");
    return;
  }
  static_cast<DirectoryBasedGlobalCompilationDatabase *>(CDB.get())
      ->setCompileCommandsDir(P);
  CachingCDB->clear();
}

GlobalCompilationDatabase &ClangdLSPServer::CompilationDB::getCDB() {
  if (CachingCDB)
    return *CachingCDB;
  return *CDB;
}
