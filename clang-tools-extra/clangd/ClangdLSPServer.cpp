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
#include "JSONRPCDispatcher.h"
#include "SourceCode.h"
#include "URI.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"

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

void ClangdLSPServer::onInitialize(InitializeParams &Params) {
  if (Params.initializationOptions)
    applyConfiguration(*Params.initializationOptions);

  if (Params.rootUri && *Params.rootUri)
    Server->setRootPath(Params.rootUri->file());
  else if (Params.rootPath && !Params.rootPath->empty())
    Server->setRootPath(*Params.rootPath);

  CCOpts.EnableSnippets =
      Params.capabilities.textDocument.completion.completionItem.snippetSupport;
  DiagOpts.EmbedFixesInDiagnostics =
      Params.capabilities.textDocument.publishDiagnostics.clangdFixSupport;
  DiagOpts.SendDiagnosticCategory =
      Params.capabilities.textDocument.publishDiagnostics.categorySupport;

  if (Params.capabilities.workspace && Params.capabilities.workspace->symbol &&
      Params.capabilities.workspace->symbol->symbolKind &&
      Params.capabilities.workspace->symbol->symbolKind->valueSet) {
    for (SymbolKind Kind :
         *Params.capabilities.workspace->symbol->symbolKind->valueSet) {
      SupportedSymbolKinds.set(static_cast<size_t>(Kind));
    }
  }

  if (Params.capabilities.textDocument.completion.completionItemKind &&
      Params.capabilities.textDocument.completion.completionItemKind->valueSet)
    for (CompletionItemKind Kind : *Params.capabilities.textDocument.completion
                                        .completionItemKind->valueSet)
      SupportedCompletionItemKinds.set(static_cast<size_t>(Kind));

  reply(json::Object{
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

void ClangdLSPServer::onShutdown(ShutdownParams &Params) {
  // Do essentially nothing, just say we're ready to exit.
  ShutdownRequestReceived = true;
  reply(nullptr);
}

void ClangdLSPServer::onExit(ExitParams &Params) { IsDone = true; }

void ClangdLSPServer::onDocumentDidOpen(DidOpenTextDocumentParams &Params) {
  PathRef File = Params.textDocument.uri.file();
  if (Params.metadata && !Params.metadata->extraFlags.empty())
    CDB.setExtraFlagsForFile(File, std::move(Params.metadata->extraFlags));

  std::string &Contents = Params.textDocument.text;

  DraftMgr.addDraft(File, Contents);
  Server->addDocument(File, Contents, WantDiagnostics::Yes);
}

void ClangdLSPServer::onDocumentDidChange(DidChangeTextDocumentParams &Params) {
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

void ClangdLSPServer::onFileEvent(DidChangeWatchedFilesParams &Params) {
  Server->onFileEvent(Params);
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
  } else {
    // We should not get here because ExecuteCommandParams would not have
    // parsed in the first place and this handler should not be called. But if
    // more commands are added, this will be here has a safe guard.
    replyError(
        ErrorCode::InvalidParams,
        llvm::formatv("Unsupported command \"{0}\".", Params.command).str());
  }
}

void ClangdLSPServer::onWorkspaceSymbol(WorkspaceSymbolParams &Params) {
  Server->workspaceSymbols(
      Params.query, CCOpts.Limit,
      [this](llvm::Expected<std::vector<SymbolInformation>> Items) {
        if (!Items)
          return replyError(ErrorCode::InternalError,
                            llvm::toString(Items.takeError()));
        for (auto &Sym : *Items)
          Sym.kind = adjustKindToCapability(Sym.kind, SupportedSymbolKinds);

        reply(json::Array(*Items));
      });
}

void ClangdLSPServer::onRename(RenameParams &Params) {
  Path File = Params.textDocument.uri.file();
  llvm::Optional<std::string> Code = DraftMgr.getDraft(File);
  if (!Code)
    return replyError(ErrorCode::InvalidParams,
                      "onRename called for non-added file");

  Server->rename(
      File, Params.position, Params.newName,
      [File, Code,
       Params](llvm::Expected<std::vector<tooling::Replacement>> Replacements) {
        if (!Replacements)
          return replyError(ErrorCode::InternalError,
                            llvm::toString(Replacements.takeError()));

        // Turn the replacements into the format specified by the Language
        // Server Protocol. Fuse them into one big JSON array.
        std::vector<TextEdit> Edits;
        for (const auto &R : *Replacements)
          Edits.push_back(replacementToEdit(*Code, R));
        WorkspaceEdit WE;
        WE.changes = {{Params.textDocument.uri.uri(), Edits}};
        reply(WE);
      });
}

void ClangdLSPServer::onDocumentDidClose(DidCloseTextDocumentParams &Params) {
  PathRef File = Params.textDocument.uri.file();
  DraftMgr.removeDraft(File);
  Server->removeDocument(File);
  CDB.invalidate(File);
}

void ClangdLSPServer::onDocumentOnTypeFormatting(
    DocumentOnTypeFormattingParams &Params) {
  auto File = Params.textDocument.uri.file();
  auto Code = DraftMgr.getDraft(File);
  if (!Code)
    return replyError(ErrorCode::InvalidParams,
                      "onDocumentOnTypeFormatting called for non-added file");

  auto ReplacementsOrError = Server->formatOnType(*Code, File, Params.position);
  if (ReplacementsOrError)
    reply(json::Array(replacementsToEdits(*Code, ReplacementsOrError.get())));
  else
    replyError(ErrorCode::UnknownErrorCode,
               llvm::toString(ReplacementsOrError.takeError()));
}

void ClangdLSPServer::onDocumentRangeFormatting(
    DocumentRangeFormattingParams &Params) {
  auto File = Params.textDocument.uri.file();
  auto Code = DraftMgr.getDraft(File);
  if (!Code)
    return replyError(ErrorCode::InvalidParams,
                      "onDocumentRangeFormatting called for non-added file");

  auto ReplacementsOrError = Server->formatRange(*Code, File, Params.range);
  if (ReplacementsOrError)
    reply(json::Array(replacementsToEdits(*Code, ReplacementsOrError.get())));
  else
    replyError(ErrorCode::UnknownErrorCode,
               llvm::toString(ReplacementsOrError.takeError()));
}

void ClangdLSPServer::onDocumentFormatting(DocumentFormattingParams &Params) {
  auto File = Params.textDocument.uri.file();
  auto Code = DraftMgr.getDraft(File);
  if (!Code)
    return replyError(ErrorCode::InvalidParams,
                      "onDocumentFormatting called for non-added file");

  auto ReplacementsOrError = Server->formatFile(*Code, File);
  if (ReplacementsOrError)
    reply(json::Array(replacementsToEdits(*Code, ReplacementsOrError.get())));
  else
    replyError(ErrorCode::UnknownErrorCode,
               llvm::toString(ReplacementsOrError.takeError()));
}

void ClangdLSPServer::onDocumentSymbol(DocumentSymbolParams &Params) {
  Server->documentSymbols(
      Params.textDocument.uri.file(),
      [this](llvm::Expected<std::vector<SymbolInformation>> Items) {
        if (!Items)
          return replyError(ErrorCode::InvalidParams,
                            llvm::toString(Items.takeError()));
        for (auto &Sym : *Items)
          Sym.kind = adjustKindToCapability(Sym.kind, SupportedSymbolKinds);
        reply(json::Array(*Items));
      });
}

void ClangdLSPServer::onCodeAction(CodeActionParams &Params) {
  // We provide a code action for each diagnostic at the requested location
  // which has FixIts available.
  auto Code = DraftMgr.getDraft(Params.textDocument.uri.file());
  if (!Code)
    return replyError(ErrorCode::InvalidParams,
                      "onCodeAction called for non-added file");

  json::Array Commands;
  for (Diagnostic &D : Params.context.diagnostics) {
    for (auto &F : getFixes(Params.textDocument.uri.file(), D)) {
      WorkspaceEdit WE;
      std::vector<TextEdit> Edits(F.Edits.begin(), F.Edits.end());
      WE.changes = {{Params.textDocument.uri.uri(), std::move(Edits)}};
      Commands.push_back(json::Object{
          {"title", llvm::formatv("Apply fix: {0}", F.Message)},
          {"command", ExecuteCommandParams::CLANGD_APPLY_FIX_COMMAND},
          {"arguments", {WE}},
      });
    }
  }
  reply(std::move(Commands));
}

void ClangdLSPServer::onCompletion(TextDocumentPositionParams &Params) {
  Server->codeComplete(Params.textDocument.uri.file(), Params.position, CCOpts,
                       [this](llvm::Expected<CodeCompleteResult> List) {
                         if (!List)
                           return replyError(List.takeError());
                         CompletionList LSPList;
                         LSPList.isIncomplete = List->HasMore;
                         for (const auto &R : List->Completions) {
                           CompletionItem C = R.render(CCOpts);
                           C.kind = adjustKindToCapability(
                               C.kind, SupportedCompletionItemKinds);
                           LSPList.items.push_back(std::move(C));
                         }
                         return reply(std::move(LSPList));
                       });
}

void ClangdLSPServer::onSignatureHelp(TextDocumentPositionParams &Params) {
  Server->signatureHelp(Params.textDocument.uri.file(), Params.position,
                        [](llvm::Expected<SignatureHelp> SignatureHelp) {
                          if (!SignatureHelp)
                            return replyError(
                                ErrorCode::InvalidParams,
                                llvm::toString(SignatureHelp.takeError()));
                          reply(*SignatureHelp);
                        });
}

void ClangdLSPServer::onGoToDefinition(TextDocumentPositionParams &Params) {
  Server->findDefinitions(Params.textDocument.uri.file(), Params.position,
                          [](llvm::Expected<std::vector<Location>> Items) {
                            if (!Items)
                              return replyError(
                                  ErrorCode::InvalidParams,
                                  llvm::toString(Items.takeError()));
                            reply(json::Array(*Items));
                          });
}

void ClangdLSPServer::onSwitchSourceHeader(TextDocumentIdentifier &Params) {
  llvm::Optional<Path> Result = Server->switchSourceHeader(Params.uri.file());
  reply(Result ? URI::createFile(*Result).toString() : "");
}

void ClangdLSPServer::onDocumentHighlight(TextDocumentPositionParams &Params) {
  Server->findDocumentHighlights(
      Params.textDocument.uri.file(), Params.position,
      [](llvm::Expected<std::vector<DocumentHighlight>> Highlights) {
        if (!Highlights)
          return replyError(ErrorCode::InternalError,
                            llvm::toString(Highlights.takeError()));
        reply(json::Array(*Highlights));
      });
}

void ClangdLSPServer::onHover(TextDocumentPositionParams &Params) {
  Server->findHover(Params.textDocument.uri.file(), Params.position,
                    [](llvm::Expected<llvm::Optional<Hover>> H) {
                      if (!H) {
                        replyError(ErrorCode::InternalError,
                                   llvm::toString(H.takeError()));
                        return;
                      }

                      reply(*H);
                    });
}

void ClangdLSPServer::applyConfiguration(
    const ClangdConfigurationParamsChange &Settings) {
  // Compilation database change.
  if (Settings.compilationDatabasePath.hasValue()) {
    CDB.setCompileCommandsDir(Settings.compilationDatabasePath.getValue());

    reparseOpenedFiles();
  }

  // Update to the compilation database.
  if (Settings.compilationDatabaseChanges) {
    const auto &CompileCommandUpdates = *Settings.compilationDatabaseChanges;
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
    DidChangeConfigurationParams &Params) {
  applyConfiguration(Params.settings);
}

void ClangdLSPServer::onReference(ReferenceParams &Params) {
  Server->findReferences(Params.textDocument.uri.file(), Params.position,
                         [](llvm::Expected<std::vector<Location>> Locations) {
                           if (!Locations)
                             return replyError(
                                 ErrorCode::InternalError,
                                 llvm::toString(Locations.takeError()));
                           reply(llvm::json::Array(*Locations));
                         });
}

ClangdLSPServer::ClangdLSPServer(JSONOutput &Out,
                                 const clangd::CodeCompleteOptions &CCOpts,
                                 llvm::Optional<Path> CompileCommandsDir,
                                 bool ShouldUseInMemoryCDB,
                                 const ClangdServer::Options &Opts)
    : Out(Out), CDB(ShouldUseInMemoryCDB ? CompilationDB::makeInMemory()
                                         : CompilationDB::makeDirectoryBased(
                                               std::move(CompileCommandsDir))),
      CCOpts(CCOpts), SupportedSymbolKinds(defaultSymbolKinds()),
      SupportedCompletionItemKinds(defaultCompletionItemKinds()),
      Server(new ClangdServer(CDB.getCDB(), FSProvider, /*DiagConsumer=*/*this,
                              Opts)) {}

bool ClangdLSPServer::run(std::FILE *In, JSONStreamStyle InputStyle) {
  assert(!IsDone && "Run was called before");
  assert(Server);

  // Set up JSONRPCDispatcher.
  JSONRPCDispatcher Dispatcher([](const json::Value &Params) {
    replyError(ErrorCode::MethodNotFound, "method not found");
  });
  registerCallbackHandlers(Dispatcher, /*Callbacks=*/*this);

  // Run the Language Server loop.
  runLanguageServerLoop(In, Out, InputStyle, Dispatcher, IsDone);

  // Make sure IsDone is set to true after this method exits to ensure assertion
  // at the start of the method fires if it's ever executed again.
  IsDone = true;
  // Destroy ClangdServer to ensure all worker threads finish.
  Server.reset();

  return ShutdownRequestReceived;
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
      std::copy(Fixes.begin(), Fixes.end(),
                std::back_inserter(FixItsForDiagnostic));
    });
  }

  // Cache FixIts
  {
    // FIXME(ibiryukov): should be deleted when documents are removed
    std::lock_guard<std::mutex> Lock(FixItsMutex);
    FixItsMap[File] = LocalFixIts;
  }

  // Publish diagnostics.
  Out.writeMessage(json::Object{
      {"jsonrpc", "2.0"},
      {"method", "textDocument/publishDiagnostics"},
      {"params",
       json::Object{
           {"uri", URIForFile{File}},
           {"diagnostics", std::move(DiagnosticsJSON)},
       }},
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
