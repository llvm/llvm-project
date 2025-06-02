//===--- ClangdLSPServer.cpp - LSP server ------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangdLSPServer.h"
#include "ClangdServer.h"
#include "CodeComplete.h"
#include "CompileCommands.h"
#include "Diagnostics.h"
#include "Feature.h"
#include "GlobalCompilationDatabase.h"
#include "LSPBinder.h"
#include "ModulesBuilder.h"
#include "Protocol.h"
#include "SemanticHighlighting.h"
#include "SourceCode.h"
#include "TUScheduler.h"
#include "URI.h"
#include "refactor/Tweak.h"
#include "support/Cancellation.h"
#include "support/Context.h"
#include "support/MemoryTree.h"
#include "support/Trace.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA1.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace clang {
namespace clangd {

namespace {
// Tracks end-to-end latency of high level lsp calls. Measurements are in
// seconds.
constexpr trace::Metric LSPLatency("lsp_latency", trace::Metric::Distribution,
                                   "method_name");

// LSP defines file versions as numbers that increase.
// ClangdServer treats them as opaque and therefore uses strings instead.
std::string encodeVersion(std::optional<int64_t> LSPVersion) {
  return LSPVersion ? llvm::to_string(*LSPVersion) : "";
}
std::optional<int64_t> decodeVersion(llvm::StringRef Encoded) {
  int64_t Result;
  if (llvm::to_integer(Encoded, Result, 10))
    return Result;
  if (!Encoded.empty()) // Empty can be e.g. diagnostics on close.
    elog("unexpected non-numeric version {0}", Encoded);
  return std::nullopt;
}

const llvm::StringLiteral ApplyFixCommand = "clangd.applyFix";
const llvm::StringLiteral ApplyTweakCommand = "clangd.applyTweak";
const llvm::StringLiteral ApplyRenameCommand = "clangd.applyRename";

CodeAction toCodeAction(const ClangdServer::CodeActionResult::Rename &R,
                        const URIForFile &File) {
  CodeAction CA;
  CA.title = R.FixMessage;
  CA.kind = std::string(CodeAction::REFACTOR_KIND);
  CA.command.emplace();
  CA.command->title = R.FixMessage;
  CA.command->command = std::string(ApplyRenameCommand);
  RenameParams Params;
  Params.textDocument = TextDocumentIdentifier{File};
  Params.position = R.Diag.Range.start;
  Params.newName = R.NewName;
  CA.command->argument = Params;
  return CA;
}

/// Transforms a tweak into a code action that would apply it if executed.
/// EXPECTS: T.prepare() was called and returned true.
CodeAction toCodeAction(const ClangdServer::TweakRef &T, const URIForFile &File,
                        Range Selection) {
  CodeAction CA;
  CA.title = T.Title;
  CA.kind = T.Kind.str();
  // This tweak may have an expensive second stage, we only run it if the user
  // actually chooses it in the UI. We reply with a command that would run the
  // corresponding tweak.
  // FIXME: for some tweaks, computing the edits is cheap and we could send them
  //        directly.
  CA.command.emplace();
  CA.command->title = T.Title;
  CA.command->command = std::string(ApplyTweakCommand);
  TweakArgs Args;
  Args.file = File;
  Args.tweakID = T.ID;
  Args.selection = Selection;
  CA.command->argument = std::move(Args);
  return CA;
}

/// Convert from Fix to LSP CodeAction.
CodeAction toCodeAction(const Fix &F, const URIForFile &File,
                        const std::optional<int64_t> &Version,
                        bool SupportsDocumentChanges,
                        bool SupportChangeAnnotation) {
  CodeAction Action;
  Action.title = F.Message;
  Action.kind = std::string(CodeAction::QUICKFIX_KIND);
  Action.edit.emplace();
  if (!SupportsDocumentChanges) {
    Action.edit->changes.emplace();
    auto &Changes = (*Action.edit->changes)[File.uri()];
    for (const auto &E : F.Edits)
      Changes.push_back({E.range, E.newText, /*annotationId=*/""});
  } else {
    Action.edit->documentChanges.emplace();
    TextDocumentEdit &Edit = Action.edit->documentChanges->emplace_back();
    Edit.textDocument = VersionedTextDocumentIdentifier{{File}, Version};
    for (const auto &E : F.Edits)
      Edit.edits.push_back(
          {E.range, E.newText,
           SupportChangeAnnotation ? E.annotationId : ""});
    if (SupportChangeAnnotation) {
      for (const auto &[AID, Annotation]: F.Annotations)
        Action.edit->changeAnnotations[AID] = Annotation;
    }
  }
  return Action;
}

void adjustSymbolKinds(llvm::MutableArrayRef<DocumentSymbol> Syms,
                       SymbolKindBitset Kinds) {
  for (auto &S : Syms) {
    S.kind = adjustKindToCapability(S.kind, Kinds);
    adjustSymbolKinds(S.children, Kinds);
  }
}

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

// Makes sure edits in \p FE are applicable to latest file contents reported by
// editor. If not generates an error message containing information about files
// that needs to be saved.
llvm::Error validateEdits(const ClangdServer &Server, const FileEdits &FE) {
  size_t InvalidFileCount = 0;
  llvm::StringRef LastInvalidFile;
  for (const auto &It : FE) {
    if (auto Draft = Server.getDraft(It.first())) {
      // If the file is open in user's editor, make sure the version we
      // saw and current version are compatible as this is the text that
      // will be replaced by editors.
      if (!It.second.canApplyTo(*Draft)) {
        ++InvalidFileCount;
        LastInvalidFile = It.first();
      }
    }
  }
  if (!InvalidFileCount)
    return llvm::Error::success();
  if (InvalidFileCount == 1)
    return error("File must be saved first: {0}", LastInvalidFile);
  return error("Files must be saved first: {0} (and {1} others)",
               LastInvalidFile, InvalidFileCount - 1);
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

  bool onNotify(llvm::StringRef Method, llvm::json::Value Params) override {
    trace::Span Tracer(Method, LSPLatency);
    SPAN_ATTACH(Tracer, "Params", Params);
    WithContext HandlerContext(handlerContext());
    log("<-- {0}", Method);
    if (Method == "exit")
      return false;
    auto Handler = Server.Handlers.NotificationHandlers.find(Method);
    if (Handler != Server.Handlers.NotificationHandlers.end()) {
      Handler->second(std::move(Params));
      Server.maybeExportMemoryProfile();
      Server.maybeCleanupMemory();
    } else if (!Server.Server) {
      elog("Notification {0} before initialization", Method);
    } else if (Method == "$/cancelRequest") {
      onCancel(std::move(Params));
    } else {
      log("unhandled notification {0}", Method);
    }
    return true;
  }

  bool onCall(llvm::StringRef Method, llvm::json::Value Params,
              llvm::json::Value ID) override {
    WithContext HandlerContext(handlerContext());
    // Calls can be canceled by the client. Add cancellation context.
    WithContext WithCancel(cancelableRequestContext(ID));
    trace::Span Tracer(Method, LSPLatency);
    SPAN_ATTACH(Tracer, "Params", Params);
    ReplyOnce Reply(ID, Method, &Server, Tracer.Args);
    log("<-- {0}({1})", Method, ID);
    auto Handler = Server.Handlers.MethodHandlers.find(Method);
    if (Handler != Server.Handlers.MethodHandlers.end()) {
      Handler->second(std::move(Params), std::move(Reply));
    } else if (!Server.Server) {
      elog("Call {0} before initialization.", Method);
      Reply(llvm::make_error<LSPError>("server not initialized",
                                       ErrorCode::ServerNotInitialized));
    } else {
      Reply(llvm::make_error<LSPError>("method not found",
                                       ErrorCode::MethodNotFound));
    }
    return true;
  }

  bool onReply(llvm::json::Value ID,
               llvm::Expected<llvm::json::Value> Result) override {
    WithContext HandlerContext(handlerContext());

    Callback<llvm::json::Value> ReplyHandler = nullptr;
    if (auto IntID = ID.getAsInteger()) {
      std::lock_guard<std::mutex> Mutex(CallMutex);
      // Find a corresponding callback for the request ID;
      for (size_t Index = 0; Index < ReplyCallbacks.size(); ++Index) {
        if (ReplyCallbacks[Index].first == *IntID) {
          ReplyHandler = std::move(ReplyCallbacks[Index].second);
          ReplyCallbacks.erase(ReplyCallbacks.begin() +
                               Index); // remove the entry
          break;
        }
      }
    }

    if (!ReplyHandler) {
      // No callback being found, use a default log callback.
      ReplyHandler = [&ID](llvm::Expected<llvm::json::Value> Result) {
        elog("received a reply with ID {0}, but there was no such call", ID);
        if (!Result)
          llvm::consumeError(Result.takeError());
      };
    }

    // Log and run the reply handler.
    if (Result) {
      log("<-- reply({0})", ID);
      ReplyHandler(std::move(Result));
    } else {
      auto Err = Result.takeError();
      log("<-- reply({0}) error: {1}", ID, Err);
      ReplyHandler(std::move(Err));
    }
    return true;
  }

  // Bind a reply callback to a request. The callback will be invoked when
  // clangd receives the reply from the LSP client.
  // Return a call id of the request.
  llvm::json::Value bindReply(Callback<llvm::json::Value> Reply) {
    std::optional<std::pair<int, Callback<llvm::json::Value>>> OldestCB;
    int ID;
    {
      std::lock_guard<std::mutex> Mutex(CallMutex);
      ID = NextCallID++;
      ReplyCallbacks.emplace_back(ID, std::move(Reply));

      // If the queue overflows, we assume that the client didn't reply the
      // oldest request, and run the corresponding callback which replies an
      // error to the client.
      if (ReplyCallbacks.size() > MaxReplayCallbacks) {
        elog("more than {0} outstanding LSP calls, forgetting about {1}",
             MaxReplayCallbacks, ReplyCallbacks.front().first);
        OldestCB = std::move(ReplyCallbacks.front());
        ReplyCallbacks.pop_front();
      }
    }
    if (OldestCB)
      OldestCB->second(
          error("failed to receive a client reply for request ({0})",
                OldestCB->first));
    return ID;
  }

private:
  // Function object to reply to an LSP call.
  // Each instance must be called exactly once, otherwise:
  //  - the bug is logged, and (in debug mode) an assert will fire
  //  - if there was no reply, an error reply is sent
  //  - if there were multiple replies, only the first is sent
  class ReplyOnce {
    std::atomic<bool> Replied = {false};
    std::chrono::steady_clock::time_point Start;
    llvm::json::Value ID;
    std::string Method;
    ClangdLSPServer *Server; // Null when moved-from.
    llvm::json::Object *TraceArgs;

  public:
    ReplyOnce(const llvm::json::Value &ID, llvm::StringRef Method,
              ClangdLSPServer *Server, llvm::json::Object *TraceArgs)
        : Start(std::chrono::steady_clock::now()), ID(ID), Method(Method),
          Server(Server), TraceArgs(TraceArgs) {
      assert(Server);
    }
    ReplyOnce(ReplyOnce &&Other)
        : Replied(Other.Replied.load()), Start(Other.Start),
          ID(std::move(Other.ID)), Method(std::move(Other.Method)),
          Server(Other.Server), TraceArgs(Other.TraceArgs) {
      Other.Server = nullptr;
    }
    ReplyOnce &operator=(ReplyOnce &&) = delete;
    ReplyOnce(const ReplyOnce &) = delete;
    ReplyOnce &operator=(const ReplyOnce &) = delete;

    ~ReplyOnce() {
      // There's one legitimate reason to never reply to a request: clangd's
      // request handler send a call to the client (e.g. applyEdit) and the
      // client never replied. In this case, the ReplyOnce is owned by
      // ClangdLSPServer's reply callback table and is destroyed along with the
      // server. We don't attempt to send a reply in this case, there's little
      // to be gained from doing so.
      if (Server && !Server->IsBeingDestroyed && !Replied) {
        elog("No reply to message {0}({1})", Method, ID);
        assert(false && "must reply to all calls!");
        (*this)(llvm::make_error<LSPError>("server failed to reply",
                                           ErrorCode::InternalError));
      }
    }

    void operator()(llvm::Expected<llvm::json::Value> Reply) {
      assert(Server && "moved-from!");
      if (Replied.exchange(true)) {
        elog("Replied twice to message {0}({1})", Method, ID);
        assert(false && "must reply to each call only once!");
        return;
      }
      auto Duration = std::chrono::steady_clock::now() - Start;
      if (Reply) {
        log("--> reply:{0}({1}) {2:ms}", Method, ID, Duration);
        if (TraceArgs)
          (*TraceArgs)["Reply"] = *Reply;
        std::lock_guard<std::mutex> Lock(Server->TranspWriter);
        Server->Transp.reply(std::move(ID), std::move(Reply));
      } else {
        llvm::Error Err = Reply.takeError();
        log("--> reply:{0}({1}) {2:ms}, error: {3}", Method, ID, Duration, Err);
        if (TraceArgs)
          (*TraceArgs)["Error"] = llvm::to_string(Err);
        std::lock_guard<std::mutex> Lock(Server->TranspWriter);
        Server->Transp.reply(std::move(ID), std::move(Err));
      }
    }
  };

  // Method calls may be cancelled by ID, so keep track of their state.
  // This needs a mutex: handlers may finish on a different thread, and that's
  // when we clean up entries in the map.
  mutable std::mutex RequestCancelersMutex;
  llvm::StringMap<std::pair<Canceler, /*Cookie*/ unsigned>> RequestCancelers;
  unsigned NextRequestCookie = 0; // To disambiguate reused IDs, see below.
  void onCancel(const llvm::json::Value &Params) {
    const llvm::json::Value *ID = nullptr;
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

  Context handlerContext() const {
    return Context::current().derive(
        kCurrentOffsetEncoding,
        Server.Opts.Encoding.value_or(OffsetEncoding::UTF16));
  }

  // We run cancelable requests in a context that does two things:
  //  - allows cancellation using RequestCancelers[ID]
  //  - cleans up the entry in RequestCancelers when it's no longer needed
  // If a client reuses an ID, the last wins and the first cannot be canceled.
  Context cancelableRequestContext(const llvm::json::Value &ID) {
    auto Task = cancelableTask(
        /*Reason=*/static_cast<int>(ErrorCode::RequestCancelled));
    auto StrID = llvm::to_string(ID);  // JSON-serialize ID for map key.
    auto Cookie = NextRequestCookie++; // No lock, only called on main thread.
    {
      std::lock_guard<std::mutex> Lock(RequestCancelersMutex);
      RequestCancelers[StrID] = {std::move(Task.second), Cookie};
    }
    // When the request ends, we can clean up the entry we just added.
    // The cookie lets us check that it hasn't been overwritten due to ID
    // reuse.
    return Task.first.derive(llvm::make_scope_exit([this, StrID, Cookie] {
      std::lock_guard<std::mutex> Lock(RequestCancelersMutex);
      auto It = RequestCancelers.find(StrID);
      if (It != RequestCancelers.end() && It->second.second == Cookie)
        RequestCancelers.erase(It);
    }));
  }

  // The maximum number of callbacks held in clangd.
  //
  // We bound the maximum size to the pending map to prevent memory leakage
  // for cases where LSP clients don't reply for the request.
  // This has to go after RequestCancellers and RequestCancellersMutex since it
  // can contain a callback that has a cancelable context.
  static constexpr int MaxReplayCallbacks = 100;
  mutable std::mutex CallMutex;
  int NextCallID = 0; /* GUARDED_BY(CallMutex) */
  std::deque<std::pair</*RequestID*/ int,
                       /*ReplyHandler*/ Callback<llvm::json::Value>>>
      ReplyCallbacks; /* GUARDED_BY(CallMutex) */

  ClangdLSPServer &Server;
};
constexpr int ClangdLSPServer::MessageHandler::MaxReplayCallbacks;

// call(), notify(), and reply() wrap the Transport, adding logging and locking.
void ClangdLSPServer::callMethod(StringRef Method, llvm::json::Value Params,
                                 Callback<llvm::json::Value> CB) {
  auto ID = MsgHandler->bindReply(std::move(CB));
  log("--> {0}({1})", Method, ID);
  std::lock_guard<std::mutex> Lock(TranspWriter);
  Transp.call(Method, std::move(Params), ID);
}

void ClangdLSPServer::notify(llvm::StringRef Method, llvm::json::Value Params) {
  log("--> {0}", Method);
  maybeCleanupMemory();
  std::lock_guard<std::mutex> Lock(TranspWriter);
  Transp.notify(Method, std::move(Params));
}

static std::vector<llvm::StringRef> semanticTokenTypes() {
  std::vector<llvm::StringRef> Types;
  for (unsigned I = 0; I <= static_cast<unsigned>(HighlightingKind::LastKind);
       ++I)
    Types.push_back(toSemanticTokenType(static_cast<HighlightingKind>(I)));
  return Types;
}

static std::vector<llvm::StringRef> semanticTokenModifiers() {
  std::vector<llvm::StringRef> Modifiers;
  for (unsigned I = 0;
       I <= static_cast<unsigned>(HighlightingModifier::LastModifier); ++I)
    Modifiers.push_back(
        toSemanticTokenModifier(static_cast<HighlightingModifier>(I)));
  return Modifiers;
}

void ClangdLSPServer::onInitialize(const InitializeParams &Params,
                                   Callback<llvm::json::Value> Reply) {
  // Determine character encoding first as it affects constructed ClangdServer.
  if (Params.capabilities.offsetEncoding && !Opts.Encoding) {
    Opts.Encoding = OffsetEncoding::UTF16; // fallback
    for (OffsetEncoding Supported : *Params.capabilities.offsetEncoding)
      if (Supported != OffsetEncoding::UnsupportedEncoding) {
        Opts.Encoding = Supported;
        break;
      }
  }

  if (Params.capabilities.TheiaSemanticHighlighting &&
      !Params.capabilities.SemanticTokens) {
    elog("Client requested legacy semanticHighlights notification, which is "
         "no longer supported. Migrate to standard semanticTokens request");
  }

  if (Params.rootUri && *Params.rootUri)
    Opts.WorkspaceRoot = std::string(Params.rootUri->file());
  else if (Params.rootPath && !Params.rootPath->empty())
    Opts.WorkspaceRoot = *Params.rootPath;
  if (Server)
    return Reply(llvm::make_error<LSPError>("server already initialized",
                                            ErrorCode::InvalidRequest));

  Opts.CodeComplete.EnableSnippets = Params.capabilities.CompletionSnippets;
  Opts.CodeComplete.IncludeFixIts = Params.capabilities.CompletionFixes;
  if (!Opts.CodeComplete.BundleOverloads)
    Opts.CodeComplete.BundleOverloads = Params.capabilities.HasSignatureHelp;
  Opts.CodeComplete.DocumentationFormat =
      Params.capabilities.CompletionDocumentationFormat;
  Opts.SignatureHelpDocumentationFormat =
      Params.capabilities.SignatureHelpDocumentationFormat;
  DiagOpts.EmbedFixesInDiagnostics = Params.capabilities.DiagnosticFixes;
  DiagOpts.SendDiagnosticCategory = Params.capabilities.DiagnosticCategory;
  DiagOpts.EmitRelatedLocations =
      Params.capabilities.DiagnosticRelatedInformation;
  if (Params.capabilities.WorkspaceSymbolKinds)
    SupportedSymbolKinds |= *Params.capabilities.WorkspaceSymbolKinds;
  if (Params.capabilities.CompletionItemKinds)
    SupportedCompletionItemKinds |= *Params.capabilities.CompletionItemKinds;
  SupportsCompletionLabelDetails = Params.capabilities.CompletionLabelDetail;
  SupportsCodeAction = Params.capabilities.CodeActionStructure;
  SupportsHierarchicalDocumentSymbol =
      Params.capabilities.HierarchicalDocumentSymbol;
  SupportsReferenceContainer = Params.capabilities.ReferenceContainer;
  SupportFileStatus = Params.initializationOptions.FileStatus;
  SupportsDocumentChanges = Params.capabilities.DocumentChanges;
  SupportsChangeAnnotation = Params.capabilities.ChangeAnnotation;
  HoverContentFormat = Params.capabilities.HoverContentFormat;
  Opts.LineFoldingOnly = Params.capabilities.LineFoldingOnly;
  SupportsOffsetsInSignatureHelp = Params.capabilities.OffsetsInSignatureHelp;
  if (Params.capabilities.WorkDoneProgress)
    BackgroundIndexProgressState = BackgroundIndexProgress::Empty;
  BackgroundIndexSkipCreate = Params.capabilities.ImplicitProgressCreation;
  Opts.ImplicitCancellation = !Params.capabilities.CancelsStaleRequests;
  Opts.PublishInactiveRegions = Params.capabilities.InactiveRegions;

  if (Opts.UseDirBasedCDB) {
    DirectoryBasedGlobalCompilationDatabase::Options CDBOpts(TFS);
    if (const auto &Dir = Params.initializationOptions.compilationDatabasePath)
      CDBOpts.CompileCommandsDir = Dir;
    CDBOpts.ContextProvider = Opts.ContextProvider;
    BaseCDB =
        std::make_unique<DirectoryBasedGlobalCompilationDatabase>(CDBOpts);
  }
  auto Mangler = CommandMangler::detect();
  Mangler.SystemIncludeExtractor =
      getSystemIncludeExtractor(llvm::ArrayRef(Opts.QueryDriverGlobs));
  if (Opts.ResourceDir)
    Mangler.ResourceDir = *Opts.ResourceDir;
  CDB.emplace(BaseCDB.get(), Params.initializationOptions.fallbackFlags,
              std::move(Mangler));

  if (Opts.EnableExperimentalModulesSupport) {
    ModulesManager.emplace(*CDB);
    Opts.ModulesManager = &*ModulesManager;
  }

  {
    // Switch caller's context with LSPServer's background context. Since we
    // rather want to propagate information from LSPServer's context into the
    // Server, CDB, etc.
    WithContext MainContext(BackgroundContext.clone());
    std::optional<WithContextValue> WithOffsetEncoding;
    if (Opts.Encoding)
      WithOffsetEncoding.emplace(kCurrentOffsetEncoding, *Opts.Encoding);
    Server.emplace(*CDB, TFS, Opts,
                   static_cast<ClangdServer::Callbacks *>(this));
  }

  llvm::json::Object ServerCaps{
      {"textDocumentSync",
       llvm::json::Object{
           {"openClose", true},
           {"change", (int)TextDocumentSyncKind::Incremental},
           {"save", true},
       }},
      {"documentFormattingProvider", true},
      {"documentRangeFormattingProvider",
       llvm::json::Object{
           {"rangesSupport", true},
       }},
      {"documentOnTypeFormattingProvider",
       llvm::json::Object{
           {"firstTriggerCharacter", "\n"},
           {"moreTriggerCharacter", {}},
       }},
      {"completionProvider",
       llvm::json::Object{
           // We don't set `(` etc as allCommitCharacters as they interact
           // poorly with snippet results.
           // See https://github.com/clangd/vscode-clangd/issues/357
           // Hopefully we can use them one day without this side-effect:
           //     https://github.com/microsoft/vscode/issues/42544
           {"resolveProvider", false},
           // We do extra checks, e.g. that > is part of ->.
           {"triggerCharacters", {".", "<", ">", ":", "\"", "/", "*"}},
       }},
      {"semanticTokensProvider",
       llvm::json::Object{
           {"full", llvm::json::Object{{"delta", true}}},
           {"range", false},
           {"legend",
            llvm::json::Object{{"tokenTypes", semanticTokenTypes()},
                               {"tokenModifiers", semanticTokenModifiers()}}},
       }},
      {"signatureHelpProvider",
       llvm::json::Object{
           {"triggerCharacters", {"(", ")", "{", "}", "<", ">", ","}},
       }},
      {"declarationProvider", true},
      {"definitionProvider", true},
      {"implementationProvider", true},
      {"typeDefinitionProvider", true},
      {"documentHighlightProvider", true},
      {"documentLinkProvider",
       llvm::json::Object{
           {"resolveProvider", false},
       }},
      {"hoverProvider", true},
      {"selectionRangeProvider", true},
      {"documentSymbolProvider", true},
      {"workspaceSymbolProvider", true},
      {"referencesProvider", true},
      {"astProvider", true}, // clangd extension
      {"typeHierarchyProvider", true},
      // Unfortunately our extension made use of the same capability name as the
      // standard. Advertise this capability to tell clients that implement our
      // extension we really have support for the standardized one as well.
      {"standardTypeHierarchyProvider", true}, // clangd extension
      {"memoryUsageProvider", true},           // clangd extension
      {"compilationDatabase",                  // clangd extension
       llvm::json::Object{{"automaticReload", true}}},
      {"inactiveRegionsProvider", true}, // clangd extension
      {"callHierarchyProvider", true},
      {"clangdInlayHintsProvider", true},
      {"inlayHintProvider", true},
      {"foldingRangeProvider", true},
  };

  {
    LSPBinder Binder(Handlers, *this);
    bindMethods(Binder, Params.capabilities);
    if (Opts.FeatureModules)
      for (auto &Mod : *Opts.FeatureModules)
        Mod.initializeLSP(Binder, Params.rawCapabilities, ServerCaps);
  }

  // Per LSP, renameProvider can be either boolean or RenameOptions.
  // RenameOptions will be specified if the client states it supports prepare.
  ServerCaps["renameProvider"] =
      Params.capabilities.RenamePrepareSupport
          ? llvm::json::Object{{"prepareProvider", true}}
          : llvm::json::Value(true);

  // Per LSP, codeActionProvider can be either boolean or CodeActionOptions.
  // CodeActionOptions is only valid if the client supports action literal
  // via textDocument.codeAction.codeActionLiteralSupport.
  ServerCaps["codeActionProvider"] =
      Params.capabilities.CodeActionStructure
          ? llvm::json::Object{{"codeActionKinds",
                                {CodeAction::QUICKFIX_KIND,
                                 CodeAction::REFACTOR_KIND,
                                 CodeAction::INFO_KIND}}}
          : llvm::json::Value(true);

  std::vector<llvm::StringRef> Commands;
  for (llvm::StringRef Command : Handlers.CommandHandlers.keys())
    Commands.push_back(Command);
  llvm::sort(Commands);
  ServerCaps["executeCommandProvider"] =
      llvm::json::Object{{"commands", Commands}};

  llvm::json::Object Result{
      {{"serverInfo",
        llvm::json::Object{
            {"name", "clangd"},
            {"version", llvm::formatv("{0} {1} {2}", versionString(),
                                      featureString(), platformString())}}},
       {"capabilities", std::move(ServerCaps)}}};
  if (Opts.Encoding)
    Result["offsetEncoding"] = *Opts.Encoding;
  Reply(std::move(Result));

  // Apply settings after we're fully initialized.
  // This can start background indexing and in turn trigger LSP notifications.
  applyConfiguration(Params.initializationOptions.ConfigSettings);
}

void ClangdLSPServer::onInitialized(const InitializedParams &Params) {}

void ClangdLSPServer::onShutdown(const NoParams &,
                                 Callback<std::nullptr_t> Reply) {
  // Do essentially nothing, just say we're ready to exit.
  ShutdownRequestReceived = true;
  Reply(nullptr);
}

// sync is a clangd extension: it blocks until all background work completes.
// It blocks the calling thread, so no messages are processed until it returns!
void ClangdLSPServer::onSync(const NoParams &, Callback<std::nullptr_t> Reply) {
  if (Server->blockUntilIdleForTest(/*TimeoutSeconds=*/60))
    Reply(nullptr);
  else
    Reply(error("Not idle after a minute"));
}

void ClangdLSPServer::onDocumentDidOpen(
    const DidOpenTextDocumentParams &Params) {
  PathRef File = Params.textDocument.uri.file();

  const std::string &Contents = Params.textDocument.text;

  Server->addDocument(File, Contents,
                      encodeVersion(Params.textDocument.version),
                      WantDiagnostics::Yes);
}

void ClangdLSPServer::onDocumentDidChange(
    const DidChangeTextDocumentParams &Params) {
  auto WantDiags = WantDiagnostics::Auto;
  if (Params.wantDiagnostics)
    WantDiags =
        *Params.wantDiagnostics ? WantDiagnostics::Yes : WantDiagnostics::No;

  PathRef File = Params.textDocument.uri.file();
  auto Code = Server->getDraft(File);
  if (!Code) {
    log("Trying to incrementally change non-added document: {0}", File);
    return;
  }
  std::string NewCode(*Code);
  for (const auto &Change : Params.contentChanges) {
    if (auto Err = applyChange(NewCode, Change)) {
      // If this fails, we are most likely going to be not in sync anymore with
      // the client.  It is better to remove the draft and let further
      // operations fail rather than giving wrong results.
      Server->removeDocument(File);
      elog("Failed to update {0}: {1}", File, std::move(Err));
      return;
    }
  }
  Server->addDocument(File, NewCode, encodeVersion(Params.textDocument.version),
                      WantDiags, Params.forceRebuild);
}

void ClangdLSPServer::onDocumentDidSave(
    const DidSaveTextDocumentParams &Params) {
  Server->reparseOpenFilesIfNeeded([](llvm::StringRef) { return true; });
}

void ClangdLSPServer::onFileEvent(const DidChangeWatchedFilesParams &Params) {
  // We could also reparse all open files here. However:
  //  - this could be frequent, and revalidating all the preambles isn't free
  //  - this is useful e.g. when switching git branches, but we're likely to see
  //    fresh headers but still have the old-branch main-file content
  Server->onFileEvent(Params);
  // FIXME: observe config files, immediately expire time-based caches, reparse:
  //  - compile_commands.json and compile_flags.txt
  //  - .clang_format and .clang-tidy
  //  - .clangd and clangd/config.yaml
}

void ClangdLSPServer::onCommand(const ExecuteCommandParams &Params,
                                Callback<llvm::json::Value> Reply) {
  auto It = Handlers.CommandHandlers.find(Params.command);
  if (It == Handlers.CommandHandlers.end()) {
    return Reply(llvm::make_error<LSPError>(
        llvm::formatv("Unsupported command \"{0}\".", Params.command).str(),
        ErrorCode::InvalidParams));
  }
  It->second(Params.argument, std::move(Reply));
}

void ClangdLSPServer::onCommandApplyEdit(const WorkspaceEdit &WE,
                                         Callback<llvm::json::Value> Reply) {
  // The flow for "apply-fix" :
  // 1. We publish a diagnostic, including fixits
  // 2. The user clicks on the diagnostic, the editor asks us for code actions
  // 3. We send code actions, with the fixit embedded as context
  // 4. The user selects the fixit, the editor asks us to apply it
  // 5. We unwrap the changes and send them back to the editor
  // 6. The editor applies the changes (applyEdit), and sends us a reply
  // 7. We unwrap the reply and send a reply to the editor.
  applyEdit(WE, "Fix applied.", std::move(Reply));
}

void ClangdLSPServer::onCommandApplyTweak(const TweakArgs &Args,
                                          Callback<llvm::json::Value> Reply) {
  auto Action = [this, Reply = std::move(Reply)](
                    llvm::Expected<Tweak::Effect> R) mutable {
    if (!R)
      return Reply(R.takeError());

    assert(R->ShowMessage || (!R->ApplyEdits.empty() && "tweak has no effect"));

    if (R->ShowMessage) {
      ShowMessageParams Msg;
      Msg.message = *R->ShowMessage;
      Msg.type = MessageType::Info;
      ShowMessage(Msg);
    }
    // When no edit is specified, make sure we Reply().
    if (R->ApplyEdits.empty())
      return Reply("Tweak applied.");

    if (auto Err = validateEdits(*Server, R->ApplyEdits))
      return Reply(std::move(Err));

    WorkspaceEdit WE;
    // FIXME: use documentChanges when SupportDocumentChanges is true.
    WE.changes.emplace();
    for (const auto &It : R->ApplyEdits) {
      (*WE.changes)[URI::createFile(It.first()).toString()] =
          It.second.asTextEdits();
    }
    // ApplyEdit will take care of calling Reply().
    return applyEdit(std::move(WE), "Tweak applied.", std::move(Reply));
  };
  Server->applyTweak(Args.file.file(), Args.selection, Args.tweakID,
                     std::move(Action));
}

void ClangdLSPServer::onCommandApplyRename(const RenameParams &R,
                                           Callback<llvm::json::Value> Reply) {
  onRename(R, [this, Reply = std::move(Reply)](
                  llvm::Expected<WorkspaceEdit> Edit) mutable {
    if (!Edit)
      Reply(Edit.takeError());
    applyEdit(std::move(*Edit), "Rename applied.", std::move(Reply));
  });
}

void ClangdLSPServer::applyEdit(WorkspaceEdit WE, llvm::json::Value Success,
                                Callback<llvm::json::Value> Reply) {
  ApplyWorkspaceEditParams Edit;
  Edit.edit = std::move(WE);
  ApplyWorkspaceEdit(
      Edit, [Reply = std::move(Reply), SuccessMessage = std::move(Success)](
                llvm::Expected<ApplyWorkspaceEditResponse> Response) mutable {
        if (!Response)
          return Reply(Response.takeError());
        if (!Response->applied) {
          std::string Reason = Response->failureReason
                                   ? *Response->failureReason
                                   : "unknown reason";
          return Reply(error("edits were not applied: {0}", Reason));
        }
        return Reply(SuccessMessage);
      });
}

void ClangdLSPServer::onWorkspaceSymbol(
    const WorkspaceSymbolParams &Params,
    Callback<std::vector<SymbolInformation>> Reply) {
  Server->workspaceSymbols(
      Params.query, Params.limit.value_or(Opts.CodeComplete.Limit),
      [Reply = std::move(Reply),
       this](llvm::Expected<std::vector<SymbolInformation>> Items) mutable {
        if (!Items)
          return Reply(Items.takeError());
        for (auto &Sym : *Items)
          Sym.kind = adjustKindToCapability(Sym.kind, SupportedSymbolKinds);

        Reply(std::move(*Items));
      });
}

void ClangdLSPServer::onPrepareRename(const TextDocumentPositionParams &Params,
                                      Callback<PrepareRenameResult> Reply) {
  Server->prepareRename(
      Params.textDocument.uri.file(), Params.position, /*NewName*/ std::nullopt,
      Opts.Rename,
      [Reply = std::move(Reply)](llvm::Expected<RenameResult> Result) mutable {
        if (!Result)
          return Reply(Result.takeError());
        PrepareRenameResult PrepareResult;
        PrepareResult.range = Result->Target;
        PrepareResult.placeholder = Result->Placeholder;
        return Reply(std::move(PrepareResult));
      });
}

void ClangdLSPServer::onRename(const RenameParams &Params,
                               Callback<WorkspaceEdit> Reply) {
  Path File = std::string(Params.textDocument.uri.file());
  if (!Server->getDraft(File))
    return Reply(llvm::make_error<LSPError>(
        "onRename called for non-added file", ErrorCode::InvalidParams));
  Server->rename(File, Params.position, Params.newName, Opts.Rename,
                 [File, Params, Reply = std::move(Reply),
                  this](llvm::Expected<RenameResult> R) mutable {
                   if (!R)
                     return Reply(R.takeError());
                   if (auto Err = validateEdits(*Server, R->GlobalChanges))
                     return Reply(std::move(Err));
                   WorkspaceEdit Result;
                   // FIXME: use documentChanges if SupportDocumentChanges is
                   // true.
                   Result.changes.emplace();
                   for (const auto &Rep : R->GlobalChanges) {
                     (*Result
                           .changes)[URI::createFile(Rep.first()).toString()] =
                         Rep.second.asTextEdits();
                   }
                   Reply(Result);
                 });
}

void ClangdLSPServer::onDocumentDidClose(
    const DidCloseTextDocumentParams &Params) {
  PathRef File = Params.textDocument.uri.file();
  Server->removeDocument(File);

  {
    std::lock_guard<std::mutex> Lock(DiagRefMutex);
    DiagRefMap.erase(File);
  }
  {
    std::lock_guard<std::mutex> HLock(SemanticTokensMutex);
    LastSemanticTokens.erase(File);
  }
  // clangd will not send updates for this file anymore, so we empty out the
  // list of diagnostics shown on the client (e.g. in the "Problems" pane of
  // VSCode). Note that this cannot race with actual diagnostics responses
  // because removeDocument() guarantees no diagnostic callbacks will be
  // executed after it returns.
  PublishDiagnosticsParams Notification;
  Notification.uri = URIForFile::canonicalize(File, /*TUPath=*/File);
  PublishDiagnostics(Notification);
}

void ClangdLSPServer::onDocumentOnTypeFormatting(
    const DocumentOnTypeFormattingParams &Params,
    Callback<std::vector<TextEdit>> Reply) {
  auto File = Params.textDocument.uri.file();
  Server->formatOnType(File, Params.position, Params.ch, std::move(Reply));
}

void ClangdLSPServer::onDocumentRangeFormatting(
    const DocumentRangeFormattingParams &Params,
    Callback<std::vector<TextEdit>> Reply) {
  onDocumentRangesFormatting(
      DocumentRangesFormattingParams{Params.textDocument, {Params.range}},
      std::move(Reply));
}

void ClangdLSPServer::onDocumentRangesFormatting(
    const DocumentRangesFormattingParams &Params,
    Callback<std::vector<TextEdit>> Reply) {
  auto File = Params.textDocument.uri.file();
  auto Code = Server->getDraft(File);
  Server->formatFile(File, Params.ranges,
                     [Code = std::move(Code), Reply = std::move(Reply)](
                         llvm::Expected<tooling::Replacements> Result) mutable {
                       if (Result)
                         Reply(replacementsToEdits(*Code, Result.get()));
                       else
                         Reply(Result.takeError());
                     });
}

void ClangdLSPServer::onDocumentFormatting(
    const DocumentFormattingParams &Params,
    Callback<std::vector<TextEdit>> Reply) {
  auto File = Params.textDocument.uri.file();
  auto Code = Server->getDraft(File);
  Server->formatFile(File,
                     /*Rngs=*/{},
                     [Code = std::move(Code), Reply = std::move(Reply)](
                         llvm::Expected<tooling::Replacements> Result) mutable {
                       if (Result)
                         Reply(replacementsToEdits(*Code, Result.get()));
                       else
                         Reply(Result.takeError());
                     });
}

/// The functions constructs a flattened view of the DocumentSymbol hierarchy.
/// Used by the clients that do not support the hierarchical view.
static std::vector<SymbolInformation>
flattenSymbolHierarchy(llvm::ArrayRef<DocumentSymbol> Symbols,
                       const URIForFile &FileURI) {
  std::vector<SymbolInformation> Results;
  std::function<void(const DocumentSymbol &, llvm::StringRef)> Process =
      [&](const DocumentSymbol &S, std::optional<llvm::StringRef> ParentName) {
        SymbolInformation SI;
        SI.containerName = std::string(ParentName ? "" : *ParentName);
        SI.name = S.name;
        SI.kind = S.kind;
        SI.location.range = S.range;
        SI.location.uri = FileURI;

        Results.push_back(std::move(SI));
        std::string FullName =
            !ParentName ? S.name : (ParentName->str() + "::" + S.name);
        for (auto &C : S.children)
          Process(C, /*ParentName=*/FullName);
      };
  for (auto &S : Symbols)
    Process(S, /*ParentName=*/"");
  return Results;
}

void ClangdLSPServer::onDocumentSymbol(const DocumentSymbolParams &Params,
                                       Callback<llvm::json::Value> Reply) {
  URIForFile FileURI = Params.textDocument.uri;
  Server->documentSymbols(
      Params.textDocument.uri.file(),
      [this, FileURI, Reply = std::move(Reply)](
          llvm::Expected<std::vector<DocumentSymbol>> Items) mutable {
        if (!Items)
          return Reply(Items.takeError());
        adjustSymbolKinds(*Items, SupportedSymbolKinds);
        if (SupportsHierarchicalDocumentSymbol)
          return Reply(std::move(*Items));
        return Reply(flattenSymbolHierarchy(*Items, FileURI));
      });
}

void ClangdLSPServer::onFoldingRange(
    const FoldingRangeParams &Params,
    Callback<std::vector<FoldingRange>> Reply) {
  Server->foldingRanges(Params.textDocument.uri.file(), std::move(Reply));
}

static std::optional<Command> asCommand(const CodeAction &Action) {
  Command Cmd;
  if (Action.command && Action.edit)
    return std::nullopt; // Not representable. (We never emit these anyway).
  if (Action.command) {
    Cmd = *Action.command;
  } else if (Action.edit) {
    Cmd.command = std::string(ApplyFixCommand);
    Cmd.argument = *Action.edit;
  } else {
    return std::nullopt;
  }
  Cmd.title = Action.title;
  if (Action.kind && *Action.kind == CodeAction::QUICKFIX_KIND)
    Cmd.title = "Apply fix: " + Cmd.title;
  return Cmd;
}

void ClangdLSPServer::onCodeAction(const CodeActionParams &Params,
                                   Callback<llvm::json::Value> Reply) {
  URIForFile File = Params.textDocument.uri;
  std::map<ClangdServer::DiagRef, clangd::Diagnostic> ToLSPDiags;
  ClangdServer::CodeActionInputs Inputs;

  for (const auto& LSPDiag : Params.context.diagnostics) {
    if (auto DiagRef = getDiagRef(File.file(), LSPDiag)) {
      ToLSPDiags[*DiagRef] = LSPDiag;
      Inputs.Diagnostics.push_back(*DiagRef);
    }
  }
  Inputs.File = File.file();
  Inputs.Selection = Params.range;
  Inputs.RequestedActionKinds = Params.context.only;
  Inputs.TweakFilter = [this](const Tweak &T) {
    return Opts.TweakFilter(T);
  };
  auto CB = [this,
             Reply = std::move(Reply),
             ToLSPDiags = std::move(ToLSPDiags), File,
             Selection = Params.range](
                llvm::Expected<ClangdServer::CodeActionResult> Fixits) mutable {
    if (!Fixits)
      return Reply(Fixits.takeError());
    std::vector<CodeAction> CAs;
    auto Version = decodeVersion(Fixits->Version);
    for (const auto &QF : Fixits->QuickFixes) {
      CAs.push_back(toCodeAction(QF.F, File, Version, SupportsDocumentChanges,
                                 SupportsChangeAnnotation));
      if (auto It = ToLSPDiags.find(QF.Diag);
          It != ToLSPDiags.end()) {
        CAs.back().diagnostics = {It->second};
      }
    }

    for (const auto &R : Fixits->Renames)
      CAs.push_back(toCodeAction(R, File));

    for (const auto &TR : Fixits->TweakRefs)
      CAs.push_back(toCodeAction(TR, File, Selection));

    // If there's exactly one quick-fix, call it "preferred".
    // We never consider refactorings etc as preferred.
    CodeAction *OnlyFix = nullptr;
    for (auto &Action : CAs) {
      if (Action.kind && *Action.kind == CodeAction::QUICKFIX_KIND) {
        if (OnlyFix) {
          OnlyFix = nullptr;
          break;
        }
        OnlyFix = &Action;
      }
    }
    if (OnlyFix) {
      OnlyFix->isPreferred = true;
      if (ToLSPDiags.size() == 1 &&
          ToLSPDiags.begin()->second.range == Selection)
        OnlyFix->diagnostics = {ToLSPDiags.begin()->second};
    }

    if (SupportsCodeAction)
      return Reply(llvm::json::Array(CAs));
    std::vector<Command> Commands;
    for (const auto &Action : CAs) {
      if (auto Command = asCommand(Action))
        Commands.push_back(std::move(*Command));
    }
    return Reply(llvm::json::Array(Commands));
  };
  Server->codeAction(Inputs, std::move(CB));
}

void ClangdLSPServer::onCompletion(const CompletionParams &Params,
                                   Callback<CompletionList> Reply) {
  if (!shouldRunCompletion(Params)) {
    // Clients sometimes auto-trigger completions in undesired places (e.g.
    // 'a >^ '), we return empty results in those cases.
    vlog("ignored auto-triggered completion, preceding char did not match");
    return Reply(CompletionList());
  }
  auto Opts = this->Opts.CodeComplete;
  if (Params.limit && *Params.limit >= 0)
    Opts.Limit = *Params.limit;
  Server->codeComplete(Params.textDocument.uri.file(), Params.position, Opts,
                       [Reply = std::move(Reply), Opts,
                        this](llvm::Expected<CodeCompleteResult> List) mutable {
                         if (!List)
                           return Reply(List.takeError());
                         CompletionList LSPList;
                         LSPList.isIncomplete = List->HasMore;
                         for (const auto &R : List->Completions) {
                           CompletionItem C = R.render(Opts);
                           C.kind = adjustKindToCapability(
                               C.kind, SupportedCompletionItemKinds);
                           if (!SupportsCompletionLabelDetails)
                             removeCompletionLabelDetails(C);
                           LSPList.items.push_back(std::move(C));
                         }
                         return Reply(std::move(LSPList));
                       });
}

void ClangdLSPServer::onSignatureHelp(const TextDocumentPositionParams &Params,
                                      Callback<SignatureHelp> Reply) {
  Server->signatureHelp(Params.textDocument.uri.file(), Params.position,
                        Opts.SignatureHelpDocumentationFormat,
                        [Reply = std::move(Reply), this](
                            llvm::Expected<SignatureHelp> Signature) mutable {
                          if (!Signature)
                            return Reply(Signature.takeError());
                          if (SupportsOffsetsInSignatureHelp)
                            return Reply(std::move(*Signature));
                          // Strip out the offsets from signature help for
                          // clients that only support string labels.
                          for (auto &SigInfo : Signature->signatures) {
                            for (auto &Param : SigInfo.parameters)
                              Param.labelOffsets.reset();
                          }
                          return Reply(std::move(*Signature));
                        });
}

// Go to definition has a toggle function: if def and decl are distinct, then
// the first press gives you the def, the second gives you the matching def.
// getToggle() returns the counterpart location that under the cursor.
//
// We return the toggled location alone (ignoring other symbols) to encourage
// editors to "bounce" quickly between locations, without showing a menu.
static Location *getToggle(const TextDocumentPositionParams &Point,
                           LocatedSymbol &Sym) {
  // Toggle only makes sense with two distinct locations.
  if (!Sym.Definition || *Sym.Definition == Sym.PreferredDeclaration)
    return nullptr;
  if (Sym.Definition->uri.file() == Point.textDocument.uri.file() &&
      Sym.Definition->range.contains(Point.position))
    return &Sym.PreferredDeclaration;
  if (Sym.PreferredDeclaration.uri.file() == Point.textDocument.uri.file() &&
      Sym.PreferredDeclaration.range.contains(Point.position))
    return &*Sym.Definition;
  return nullptr;
}

void ClangdLSPServer::onGoToDefinition(const TextDocumentPositionParams &Params,
                                       Callback<std::vector<Location>> Reply) {
  Server->locateSymbolAt(
      Params.textDocument.uri.file(), Params.position,
      [Params, Reply = std::move(Reply)](
          llvm::Expected<std::vector<LocatedSymbol>> Symbols) mutable {
        if (!Symbols)
          return Reply(Symbols.takeError());
        std::vector<Location> Defs;
        for (auto &S : *Symbols) {
          if (Location *Toggle = getToggle(Params, S))
            return Reply(std::vector<Location>{std::move(*Toggle)});
          Defs.push_back(S.Definition.value_or(S.PreferredDeclaration));
        }
        Reply(std::move(Defs));
      });
}

void ClangdLSPServer::onGoToDeclaration(
    const TextDocumentPositionParams &Params,
    Callback<std::vector<Location>> Reply) {
  Server->locateSymbolAt(
      Params.textDocument.uri.file(), Params.position,
      [Params, Reply = std::move(Reply)](
          llvm::Expected<std::vector<LocatedSymbol>> Symbols) mutable {
        if (!Symbols)
          return Reply(Symbols.takeError());
        std::vector<Location> Decls;
        for (auto &S : *Symbols) {
          if (Location *Toggle = getToggle(Params, S))
            return Reply(std::vector<Location>{std::move(*Toggle)});
          Decls.push_back(std::move(S.PreferredDeclaration));
        }
        Reply(std::move(Decls));
      });
}

void ClangdLSPServer::onSwitchSourceHeader(
    const TextDocumentIdentifier &Params,
    Callback<std::optional<URIForFile>> Reply) {
  Server->switchSourceHeader(
      Params.uri.file(),
      [Reply = std::move(Reply),
       Params](llvm::Expected<std::optional<clangd::Path>> Path) mutable {
        if (!Path)
          return Reply(Path.takeError());
        if (*Path)
          return Reply(URIForFile::canonicalize(**Path, Params.uri.file()));
        return Reply(std::nullopt);
      });
}

void ClangdLSPServer::onDocumentHighlight(
    const TextDocumentPositionParams &Params,
    Callback<std::vector<DocumentHighlight>> Reply) {
  Server->findDocumentHighlights(Params.textDocument.uri.file(),
                                 Params.position, std::move(Reply));
}

void ClangdLSPServer::onHover(const TextDocumentPositionParams &Params,
                              Callback<std::optional<Hover>> Reply) {
  Server->findHover(Params.textDocument.uri.file(), Params.position,
                    [Reply = std::move(Reply),
                     this](llvm::Expected<std::optional<HoverInfo>> H) mutable {
                      if (!H)
                        return Reply(H.takeError());
                      if (!*H)
                        return Reply(std::nullopt);

                      Hover R;
                      R.contents.kind = HoverContentFormat;
                      R.range = (*H)->SymRange;
                      switch (HoverContentFormat) {
                      case MarkupKind::PlainText:
                        R.contents.value = (*H)->present().asPlainText();
                        return Reply(std::move(R));
                      case MarkupKind::Markdown:
                        R.contents.value = (*H)->present().asMarkdown();
                        return Reply(std::move(R));
                      };
                      llvm_unreachable("unhandled MarkupKind");
                    });
}

// Our extension has a different representation on the wire than the standard.
// https://clangd.llvm.org/extensions#type-hierarchy
llvm::json::Value serializeTHIForExtension(TypeHierarchyItem THI) {
  llvm::json::Object Result{{
      {"name", std::move(THI.name)},
      {"kind", static_cast<int>(THI.kind)},
      {"uri", std::move(THI.uri)},
      {"range", THI.range},
      {"selectionRange", THI.selectionRange},
      {"data", std::move(THI.data)},
  }};
  if (THI.deprecated)
    Result["deprecated"] = THI.deprecated;
  if (THI.detail)
    Result["detail"] = std::move(*THI.detail);

  if (THI.parents) {
    llvm::json::Array Parents;
    for (auto &Parent : *THI.parents)
      Parents.emplace_back(serializeTHIForExtension(std::move(Parent)));
    Result["parents"] = std::move(Parents);
  }

  if (THI.children) {
    llvm::json::Array Children;
    for (auto &child : *THI.children)
      Children.emplace_back(serializeTHIForExtension(std::move(child)));
    Result["children"] = std::move(Children);
  }
  return Result;
}

void ClangdLSPServer::onTypeHierarchy(const TypeHierarchyPrepareParams &Params,
                                      Callback<llvm::json::Value> Reply) {
  auto Serialize =
      [Reply = std::move(Reply)](
          llvm::Expected<std::vector<TypeHierarchyItem>> Resp) mutable {
        if (!Resp) {
          Reply(Resp.takeError());
          return;
        }
        if (Resp->empty()) {
          Reply(nullptr);
          return;
        }
        Reply(serializeTHIForExtension(std::move(Resp->front())));
      };
  Server->typeHierarchy(Params.textDocument.uri.file(), Params.position,
                        Params.resolve, Params.direction, std::move(Serialize));
}

void ClangdLSPServer::onResolveTypeHierarchy(
    const ResolveTypeHierarchyItemParams &Params,
    Callback<llvm::json::Value> Reply) {
  auto Serialize =
      [Reply = std::move(Reply)](
          llvm::Expected<std::optional<TypeHierarchyItem>> Resp) mutable {
        if (!Resp) {
          Reply(Resp.takeError());
          return;
        }
        if (!*Resp) {
          Reply(std::move(*Resp));
          return;
        }
        Reply(serializeTHIForExtension(std::move(**Resp)));
      };
  Server->resolveTypeHierarchy(Params.item, Params.resolve, Params.direction,
                               std::move(Serialize));
}

void ClangdLSPServer::onPrepareTypeHierarchy(
    const TypeHierarchyPrepareParams &Params,
    Callback<std::vector<TypeHierarchyItem>> Reply) {
  Server->typeHierarchy(Params.textDocument.uri.file(), Params.position,
                        Params.resolve, Params.direction, std::move(Reply));
}

void ClangdLSPServer::onSuperTypes(
    const ResolveTypeHierarchyItemParams &Params,
    Callback<std::optional<std::vector<TypeHierarchyItem>>> Reply) {
  Server->superTypes(Params.item, std::move(Reply));
}

void ClangdLSPServer::onSubTypes(
    const ResolveTypeHierarchyItemParams &Params,
    Callback<std::vector<TypeHierarchyItem>> Reply) {
  Server->subTypes(Params.item, std::move(Reply));
}

void ClangdLSPServer::onPrepareCallHierarchy(
    const CallHierarchyPrepareParams &Params,
    Callback<std::vector<CallHierarchyItem>> Reply) {
  Server->prepareCallHierarchy(Params.textDocument.uri.file(), Params.position,
                               std::move(Reply));
}

void ClangdLSPServer::onCallHierarchyIncomingCalls(
    const CallHierarchyIncomingCallsParams &Params,
    Callback<std::vector<CallHierarchyIncomingCall>> Reply) {
  Server->incomingCalls(Params.item, std::move(Reply));
}

void ClangdLSPServer::onClangdInlayHints(const InlayHintsParams &Params,
                                         Callback<llvm::json::Value> Reply) {
  // Our extension has a different representation on the wire than the standard.
  // We have a "range" property and "kind" is represented as a string, not as an
  // enum value.
  // https://clangd.llvm.org/extensions#inlay-hints
  auto Serialize = [Reply = std::move(Reply)](
                       llvm::Expected<std::vector<InlayHint>> Hints) mutable {
    if (!Hints) {
      Reply(Hints.takeError());
      return;
    }
    llvm::json::Array Result;
    Result.reserve(Hints->size());
    for (auto &Hint : *Hints) {
      Result.emplace_back(llvm::json::Object{
          {"kind", llvm::to_string(Hint.kind)},
          {"range", Hint.range},
          {"position", Hint.position},
          // Extension doesn't have paddingLeft/Right so adjust the label
          // accordingly.
          {"label",
           ((Hint.paddingLeft ? " " : "") + llvm::StringRef(Hint.joinLabels()) +
            (Hint.paddingRight ? " " : ""))
               .str()},
      });
    }
    Reply(std::move(Result));
  };
  Server->inlayHints(Params.textDocument.uri.file(), Params.range,
                     std::move(Serialize));
}

void ClangdLSPServer::onInlayHint(const InlayHintsParams &Params,
                                  Callback<std::vector<InlayHint>> Reply) {
  Server->inlayHints(Params.textDocument.uri.file(), Params.range,
                     std::move(Reply));
}

void ClangdLSPServer::onCallHierarchyOutgoingCalls(
    const CallHierarchyOutgoingCallsParams &Params,
    Callback<std::vector<CallHierarchyOutgoingCall>> Reply) {
  Server->outgoingCalls(Params.item, std::move(Reply));
}

void ClangdLSPServer::applyConfiguration(
    const ConfigurationSettings &Settings) {
  // Per-file update to the compilation database.
  llvm::StringSet<> ModifiedFiles;
  for (auto &[File, Command] : Settings.compilationDatabaseChanges) {
    auto Cmd =
        tooling::CompileCommand(std::move(Command.workingDirectory), File,
                                std::move(Command.compilationCommand),
                                /*Output=*/"");
    if (CDB->setCompileCommand(File, std::move(Cmd))) {
      ModifiedFiles.insert(File);
    }
  }

  Server->reparseOpenFilesIfNeeded(
      [&](llvm::StringRef File) { return ModifiedFiles.count(File) != 0; });
}

void ClangdLSPServer::maybeExportMemoryProfile() {
  if (!trace::enabled() || !ShouldProfile())
    return;

  static constexpr trace::Metric MemoryUsage(
      "memory_usage", trace::Metric::Value, "component_name");
  trace::Span Tracer("ProfileBrief");
  MemoryTree MT;
  profile(MT);
  record(MT, "clangd_lsp_server", MemoryUsage);
}

void ClangdLSPServer::maybeCleanupMemory() {
  if (!Opts.MemoryCleanup || !ShouldCleanupMemory())
    return;
  Opts.MemoryCleanup();
}

// FIXME: This function needs to be properly tested.
void ClangdLSPServer::onChangeConfiguration(
    const DidChangeConfigurationParams &Params) {
  applyConfiguration(Params.settings);
}

void ClangdLSPServer::onReference(
    const ReferenceParams &Params,
    Callback<std::vector<ReferenceLocation>> Reply) {
  Server->findReferences(Params.textDocument.uri.file(), Params.position,
                         Opts.ReferencesLimit, SupportsReferenceContainer,
                         [Reply = std::move(Reply),
                          IncludeDecl(Params.context.includeDeclaration)](
                             llvm::Expected<ReferencesResult> Refs) mutable {
                           if (!Refs)
                             return Reply(Refs.takeError());
                           // Filter out declarations if the client asked.
                           std::vector<ReferenceLocation> Result;
                           Result.reserve(Refs->References.size());
                           for (auto &Ref : Refs->References) {
                             bool IsDecl =
                                 Ref.Attributes & ReferencesResult::Declaration;
                             if (IncludeDecl || !IsDecl)
                               Result.push_back(std::move(Ref.Loc));
                           }
                           return Reply(std::move(Result));
                         });
}

void ClangdLSPServer::onGoToType(const TextDocumentPositionParams &Params,
                                 Callback<std::vector<Location>> Reply) {
  Server->findType(
      Params.textDocument.uri.file(), Params.position,
      [Reply = std::move(Reply)](
          llvm::Expected<std::vector<LocatedSymbol>> Types) mutable {
        if (!Types)
          return Reply(Types.takeError());
        std::vector<Location> Response;
        for (const LocatedSymbol &Sym : *Types)
          Response.push_back(Sym.Definition.value_or(Sym.PreferredDeclaration));
        return Reply(std::move(Response));
      });
}

void ClangdLSPServer::onGoToImplementation(
    const TextDocumentPositionParams &Params,
    Callback<std::vector<Location>> Reply) {
  Server->findImplementations(
      Params.textDocument.uri.file(), Params.position,
      [Reply = std::move(Reply)](
          llvm::Expected<std::vector<LocatedSymbol>> Overrides) mutable {
        if (!Overrides)
          return Reply(Overrides.takeError());
        std::vector<Location> Impls;
        for (const LocatedSymbol &Sym : *Overrides)
          Impls.push_back(Sym.Definition.value_or(Sym.PreferredDeclaration));
        return Reply(std::move(Impls));
      });
}

void ClangdLSPServer::onSymbolInfo(const TextDocumentPositionParams &Params,
                                   Callback<std::vector<SymbolDetails>> Reply) {
  Server->symbolInfo(Params.textDocument.uri.file(), Params.position,
                     std::move(Reply));
}

void ClangdLSPServer::onSelectionRange(
    const SelectionRangeParams &Params,
    Callback<std::vector<SelectionRange>> Reply) {
  Server->semanticRanges(
      Params.textDocument.uri.file(), Params.positions,
      [Reply = std::move(Reply)](
          llvm::Expected<std::vector<SelectionRange>> Ranges) mutable {
        if (!Ranges)
          return Reply(Ranges.takeError());
        return Reply(std::move(*Ranges));
      });
}

void ClangdLSPServer::onDocumentLink(
    const DocumentLinkParams &Params,
    Callback<std::vector<DocumentLink>> Reply) {

  // TODO(forster): This currently resolves all targets eagerly. This is slow,
  // because it blocks on the preamble/AST being built. We could respond to the
  // request faster by using string matching or the lexer to find the includes
  // and resolving the targets lazily.
  Server->documentLinks(
      Params.textDocument.uri.file(),
      [Reply = std::move(Reply)](
          llvm::Expected<std::vector<DocumentLink>> Links) mutable {
        if (!Links) {
          return Reply(Links.takeError());
        }
        return Reply(std::move(Links));
      });
}

// Increment a numeric string: "" -> 1 -> 2 -> ... -> 9 -> 10 -> 11 ...
static void increment(std::string &S) {
  for (char &C : llvm::reverse(S)) {
    if (C != '9') {
      ++C;
      return;
    }
    C = '0';
  }
  S.insert(S.begin(), '1');
}

void ClangdLSPServer::onSemanticTokens(const SemanticTokensParams &Params,
                                       Callback<SemanticTokens> CB) {
  auto File = Params.textDocument.uri.file();
  Server->semanticHighlights(
      Params.textDocument.uri.file(),
      [this, File(File.str()), CB(std::move(CB)), Code(Server->getDraft(File))](
          llvm::Expected<std::vector<HighlightingToken>> HT) mutable {
        if (!HT)
          return CB(HT.takeError());
        SemanticTokens Result;
        Result.tokens = toSemanticTokens(*HT, *Code);
        {
          std::lock_guard<std::mutex> Lock(SemanticTokensMutex);
          auto &Last = LastSemanticTokens[File];

          Last.tokens = Result.tokens;
          increment(Last.resultId);
          Result.resultId = Last.resultId;
        }
        CB(std::move(Result));
      });
}

void ClangdLSPServer::onSemanticTokensDelta(
    const SemanticTokensDeltaParams &Params,
    Callback<SemanticTokensOrDelta> CB) {
  auto File = Params.textDocument.uri.file();
  Server->semanticHighlights(
      Params.textDocument.uri.file(),
      [this, PrevResultID(Params.previousResultId), File(File.str()),
       CB(std::move(CB)), Code(Server->getDraft(File))](
          llvm::Expected<std::vector<HighlightingToken>> HT) mutable {
        if (!HT)
          return CB(HT.takeError());
        std::vector<SemanticToken> Toks = toSemanticTokens(*HT, *Code);

        SemanticTokensOrDelta Result;
        {
          std::lock_guard<std::mutex> Lock(SemanticTokensMutex);
          auto &Last = LastSemanticTokens[File];

          if (PrevResultID == Last.resultId) {
            Result.edits = diffTokens(Last.tokens, Toks);
          } else {
            vlog("semanticTokens/full/delta: wanted edits vs {0} but last "
                 "result had ID {1}. Returning full token list.",
                 PrevResultID, Last.resultId);
            Result.tokens = Toks;
          }

          Last.tokens = std::move(Toks);
          increment(Last.resultId);
          Result.resultId = Last.resultId;
        }

        CB(std::move(Result));
      });
}

void ClangdLSPServer::onMemoryUsage(const NoParams &,
                                    Callback<MemoryTree> Reply) {
  llvm::BumpPtrAllocator DetailAlloc;
  MemoryTree MT(&DetailAlloc);
  profile(MT);
  Reply(std::move(MT));
}

void ClangdLSPServer::onAST(const ASTParams &Params,
                            Callback<std::optional<ASTNode>> CB) {
  Server->getAST(Params.textDocument.uri.file(), Params.range, std::move(CB));
}

ClangdLSPServer::ClangdLSPServer(Transport &Transp, const ThreadsafeFS &TFS,
                                 const ClangdLSPServer::Options &Opts)
    : ShouldProfile(/*Period=*/std::chrono::minutes(5),
                    /*Delay=*/std::chrono::minutes(1)),
      ShouldCleanupMemory(/*Period=*/std::chrono::minutes(1),
                          /*Delay=*/std::chrono::minutes(1)),
      BackgroundContext(Context::current().clone()), Transp(Transp),
      MsgHandler(new MessageHandler(*this)), TFS(TFS),
      SupportedSymbolKinds(defaultSymbolKinds()),
      SupportedCompletionItemKinds(defaultCompletionItemKinds()), Opts(Opts) {
  if (Opts.ConfigProvider) {
    assert(!Opts.ContextProvider &&
           "Only one of ConfigProvider and ContextProvider allowed!");
    this->Opts.ContextProvider = ClangdServer::createConfiguredContextProvider(
        Opts.ConfigProvider, this);
  }
  LSPBinder Bind(this->Handlers, *this);
  Bind.method("initialize", this, &ClangdLSPServer::onInitialize);
}

void ClangdLSPServer::bindMethods(LSPBinder &Bind,
                                  const ClientCapabilities &Caps) {
  // clang-format off
  Bind.notification("initialized", this, &ClangdLSPServer::onInitialized);
  Bind.method("shutdown", this, &ClangdLSPServer::onShutdown);
  Bind.method("sync", this, &ClangdLSPServer::onSync);
  Bind.method("textDocument/rangeFormatting", this, &ClangdLSPServer::onDocumentRangeFormatting);
  Bind.method("textDocument/rangesFormatting", this, &ClangdLSPServer::onDocumentRangesFormatting);
  Bind.method("textDocument/onTypeFormatting", this, &ClangdLSPServer::onDocumentOnTypeFormatting);
  Bind.method("textDocument/formatting", this, &ClangdLSPServer::onDocumentFormatting);
  Bind.method("textDocument/codeAction", this, &ClangdLSPServer::onCodeAction);
  Bind.method("textDocument/completion", this, &ClangdLSPServer::onCompletion);
  Bind.method("textDocument/signatureHelp", this, &ClangdLSPServer::onSignatureHelp);
  Bind.method("textDocument/definition", this, &ClangdLSPServer::onGoToDefinition);
  Bind.method("textDocument/declaration", this, &ClangdLSPServer::onGoToDeclaration);
  Bind.method("textDocument/typeDefinition", this, &ClangdLSPServer::onGoToType);
  Bind.method("textDocument/implementation", this, &ClangdLSPServer::onGoToImplementation);
  Bind.method("textDocument/references", this, &ClangdLSPServer::onReference);
  Bind.method("textDocument/switchSourceHeader", this, &ClangdLSPServer::onSwitchSourceHeader);
  Bind.method("textDocument/prepareRename", this, &ClangdLSPServer::onPrepareRename);
  Bind.method("textDocument/rename", this, &ClangdLSPServer::onRename);
  Bind.method("textDocument/hover", this, &ClangdLSPServer::onHover);
  Bind.method("textDocument/documentSymbol", this, &ClangdLSPServer::onDocumentSymbol);
  Bind.method("workspace/executeCommand", this, &ClangdLSPServer::onCommand);
  Bind.method("textDocument/documentHighlight", this, &ClangdLSPServer::onDocumentHighlight);
  Bind.method("workspace/symbol", this, &ClangdLSPServer::onWorkspaceSymbol);
  Bind.method("textDocument/ast", this, &ClangdLSPServer::onAST);
  Bind.notification("textDocument/didOpen", this, &ClangdLSPServer::onDocumentDidOpen);
  Bind.notification("textDocument/didClose", this, &ClangdLSPServer::onDocumentDidClose);
  Bind.notification("textDocument/didChange", this, &ClangdLSPServer::onDocumentDidChange);
  Bind.notification("textDocument/didSave", this, &ClangdLSPServer::onDocumentDidSave);
  Bind.notification("workspace/didChangeWatchedFiles", this, &ClangdLSPServer::onFileEvent);
  Bind.notification("workspace/didChangeConfiguration", this, &ClangdLSPServer::onChangeConfiguration);
  Bind.method("textDocument/symbolInfo", this, &ClangdLSPServer::onSymbolInfo);
  Bind.method("textDocument/typeHierarchy", this, &ClangdLSPServer::onTypeHierarchy);
  Bind.method("typeHierarchy/resolve", this, &ClangdLSPServer::onResolveTypeHierarchy);
  Bind.method("textDocument/prepareTypeHierarchy", this, &ClangdLSPServer::onPrepareTypeHierarchy);
  Bind.method("typeHierarchy/supertypes", this, &ClangdLSPServer::onSuperTypes);
  Bind.method("typeHierarchy/subtypes", this, &ClangdLSPServer::onSubTypes);
  Bind.method("textDocument/prepareCallHierarchy", this, &ClangdLSPServer::onPrepareCallHierarchy);
  Bind.method("callHierarchy/incomingCalls", this, &ClangdLSPServer::onCallHierarchyIncomingCalls);
  if (Opts.EnableOutgoingCalls)
    Bind.method("callHierarchy/outgoingCalls", this, &ClangdLSPServer::onCallHierarchyOutgoingCalls);
  Bind.method("textDocument/selectionRange", this, &ClangdLSPServer::onSelectionRange);
  Bind.method("textDocument/documentLink", this, &ClangdLSPServer::onDocumentLink);
  Bind.method("textDocument/semanticTokens/full", this, &ClangdLSPServer::onSemanticTokens);
  Bind.method("textDocument/semanticTokens/full/delta", this, &ClangdLSPServer::onSemanticTokensDelta);
  Bind.method("clangd/inlayHints", this, &ClangdLSPServer::onClangdInlayHints);
  Bind.method("textDocument/inlayHint", this, &ClangdLSPServer::onInlayHint);
  Bind.method("$/memoryUsage", this, &ClangdLSPServer::onMemoryUsage);
  Bind.method("textDocument/foldingRange", this, &ClangdLSPServer::onFoldingRange);
  Bind.command(ApplyFixCommand, this, &ClangdLSPServer::onCommandApplyEdit);
  Bind.command(ApplyTweakCommand, this, &ClangdLSPServer::onCommandApplyTweak);
  Bind.command(ApplyRenameCommand, this, &ClangdLSPServer::onCommandApplyRename);

  ApplyWorkspaceEdit = Bind.outgoingMethod("workspace/applyEdit");
  PublishDiagnostics = Bind.outgoingNotification("textDocument/publishDiagnostics");
  if (Caps.InactiveRegions)
    PublishInactiveRegions = Bind.outgoingNotification("textDocument/inactiveRegions");
  ShowMessage = Bind.outgoingNotification("window/showMessage");
  NotifyFileStatus = Bind.outgoingNotification("textDocument/clangd.fileStatus");
  CreateWorkDoneProgress = Bind.outgoingMethod("window/workDoneProgress/create");
  BeginWorkDoneProgress = Bind.outgoingNotification("$/progress");
  ReportWorkDoneProgress = Bind.outgoingNotification("$/progress");
  EndWorkDoneProgress = Bind.outgoingNotification("$/progress");
  if(Caps.SemanticTokenRefreshSupport)
    SemanticTokensRefresh = Bind.outgoingMethod("workspace/semanticTokens/refresh");
  // clang-format on
}

ClangdLSPServer::~ClangdLSPServer() {
  IsBeingDestroyed = true;
  // Explicitly destroy ClangdServer first, blocking on threads it owns.
  // This ensures they don't access any other members.
  Server.reset();
}

bool ClangdLSPServer::run() {
  // Run the Language Server loop.
  bool CleanExit = true;
  if (auto Err = Transp.loop(*MsgHandler)) {
    elog("Transport error: {0}", std::move(Err));
    CleanExit = false;
  }

  return CleanExit && ShutdownRequestReceived;
}

void ClangdLSPServer::profile(MemoryTree &MT) const {
  if (Server)
    Server->profile(MT.child("clangd_server"));
}

std::optional<ClangdServer::DiagRef>
ClangdLSPServer::getDiagRef(StringRef File, const clangd::Diagnostic &D) {
  std::lock_guard<std::mutex> Lock(DiagRefMutex);
  auto DiagToDiagRefIter = DiagRefMap.find(File);
  if (DiagToDiagRefIter == DiagRefMap.end())
    return std::nullopt;

  const auto &DiagToDiagRefMap = DiagToDiagRefIter->second;
  auto FixItsIter = DiagToDiagRefMap.find(toDiagKey(D));
  if (FixItsIter == DiagToDiagRefMap.end())
    return std::nullopt;

  return FixItsIter->second;
}

// A completion request is sent when the user types '>' or ':', but we only
// want to trigger on '->' and '::'. We check the preceding text to make
// sure it matches what we expected.
// Running the lexer here would be more robust (e.g. we can detect comments
// and avoid triggering completion there), but we choose to err on the side
// of simplicity here.
bool ClangdLSPServer::shouldRunCompletion(
    const CompletionParams &Params) const {
  if (Params.context.triggerKind != CompletionTriggerKind::TriggerCharacter)
    return true;
  auto Code = Server->getDraft(Params.textDocument.uri.file());
  if (!Code)
    return true; // completion code will log the error for untracked doc.
  auto Offset = positionToOffset(*Code, Params.position,
                                 /*AllowColumnsBeyondLineLength=*/false);
  if (!Offset) {
    vlog("could not convert position '{0}' to offset for file '{1}'",
         Params.position, Params.textDocument.uri.file());
    return true;
  }
  return allowImplicitCompletion(*Code, *Offset);
}

void ClangdLSPServer::onDiagnosticsReady(PathRef File, llvm::StringRef Version,
                                         llvm::ArrayRef<Diag> Diagnostics) {
  PublishDiagnosticsParams Notification;
  Notification.version = decodeVersion(Version);
  Notification.uri = URIForFile::canonicalize(File, /*TUPath=*/File);
  DiagnosticToDiagRefMap LocalDiagMap; // Temporary storage
  for (auto &Diag : Diagnostics) {
    toLSPDiags(Diag, Notification.uri, DiagOpts,
               [&](clangd::Diagnostic LSPDiag, llvm::ArrayRef<Fix> Fixes) {
                 if (DiagOpts.EmbedFixesInDiagnostics) {
                   std::vector<CodeAction> CodeActions;
                   for (const auto &Fix : Fixes)
                     CodeActions.push_back(toCodeAction(
                         Fix, Notification.uri, Notification.version,
                         SupportsDocumentChanges, SupportsChangeAnnotation));
                   LSPDiag.codeActions.emplace(std::move(CodeActions));
                   if (LSPDiag.codeActions->size() == 1)
                     LSPDiag.codeActions->front().isPreferred = true;
                 }
                 LocalDiagMap[toDiagKey(LSPDiag)] = {Diag.Range, Diag.Message};
                 Notification.diagnostics.push_back(std::move(LSPDiag));
               });
  }

  // Cache DiagRefMap
  {
    std::lock_guard<std::mutex> Lock(DiagRefMutex);
    DiagRefMap[File] = LocalDiagMap;
  }

  // Send a notification to the LSP client.
  PublishDiagnostics(Notification);
}

void ClangdLSPServer::onInactiveRegionsReady(
    PathRef File, std::vector<Range> InactiveRegions) {
  InactiveRegionsParams Notification;
  Notification.TextDocument = {URIForFile::canonicalize(File, /*TUPath=*/File)};
  Notification.InactiveRegions = std::move(InactiveRegions);

  PublishInactiveRegions(Notification);
}

void ClangdLSPServer::onBackgroundIndexProgress(
    const BackgroundQueue::Stats &Stats) {
  static const char ProgressToken[] = "backgroundIndexProgress";

  // The background index did some work, maybe we need to cleanup
  maybeCleanupMemory();

  std::lock_guard<std::mutex> Lock(BackgroundIndexProgressMutex);

  auto NotifyProgress = [this](const BackgroundQueue::Stats &Stats) {
    if (BackgroundIndexProgressState != BackgroundIndexProgress::Live) {
      WorkDoneProgressBegin Begin;
      Begin.percentage = true;
      Begin.title = "indexing";
      BeginWorkDoneProgress({ProgressToken, std::move(Begin)});
      BackgroundIndexProgressState = BackgroundIndexProgress::Live;
    }

    if (Stats.Completed < Stats.Enqueued) {
      assert(Stats.Enqueued > Stats.LastIdle);
      WorkDoneProgressReport Report;
      Report.percentage = 100 * (Stats.Completed - Stats.LastIdle) /
                          (Stats.Enqueued - Stats.LastIdle);
      Report.message =
          llvm::formatv("{0}/{1}", Stats.Completed - Stats.LastIdle,
                        Stats.Enqueued - Stats.LastIdle);
      ReportWorkDoneProgress({ProgressToken, std::move(Report)});
    } else {
      assert(Stats.Completed == Stats.Enqueued);
      EndWorkDoneProgress({ProgressToken, WorkDoneProgressEnd()});
      BackgroundIndexProgressState = BackgroundIndexProgress::Empty;
    }
  };

  switch (BackgroundIndexProgressState) {
  case BackgroundIndexProgress::Unsupported:
    return;
  case BackgroundIndexProgress::Creating:
    // Cache this update for when the progress bar is available.
    PendingBackgroundIndexProgress = Stats;
    return;
  case BackgroundIndexProgress::Empty: {
    if (BackgroundIndexSkipCreate) {
      NotifyProgress(Stats);
      break;
    }
    // Cache this update for when the progress bar is available.
    PendingBackgroundIndexProgress = Stats;
    BackgroundIndexProgressState = BackgroundIndexProgress::Creating;
    WorkDoneProgressCreateParams CreateRequest;
    CreateRequest.token = ProgressToken;
    CreateWorkDoneProgress(
        CreateRequest,
        [this, NotifyProgress](llvm::Expected<std::nullptr_t> E) {
          std::lock_guard<std::mutex> Lock(BackgroundIndexProgressMutex);
          if (E) {
            NotifyProgress(this->PendingBackgroundIndexProgress);
          } else {
            elog("Failed to create background index progress bar: {0}",
                 E.takeError());
            // give up forever rather than thrashing about
            BackgroundIndexProgressState = BackgroundIndexProgress::Unsupported;
          }
        });
    break;
  }
  case BackgroundIndexProgress::Live:
    NotifyProgress(Stats);
    break;
  }
}

void ClangdLSPServer::onFileUpdated(PathRef File, const TUStatus &Status) {
  if (!SupportFileStatus)
    return;
  // FIXME: we don't emit "BuildingFile" and `RunningAction`, as these
  // two statuses are running faster in practice, which leads the UI constantly
  // changing, and doesn't provide much value. We may want to emit status at a
  // reasonable time interval (e.g. 0.5s).
  if (Status.PreambleActivity == PreambleAction::Idle &&
      (Status.ASTActivity.K == ASTAction::Building ||
       Status.ASTActivity.K == ASTAction::RunningAction))
    return;
  NotifyFileStatus(Status.render(File));
}

void ClangdLSPServer::onSemanticsMaybeChanged(PathRef File) {
  if (SemanticTokensRefresh) {
    SemanticTokensRefresh(NoParams{}, [](llvm::Expected<std::nullptr_t> E) {
      if (E)
        return;
      elog("Failed to refresh semantic tokens: {0}", E.takeError());
    });
  }
}

} // namespace clangd
} // namespace clang
