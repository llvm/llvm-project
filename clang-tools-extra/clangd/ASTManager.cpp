//===--- ASTManager.cpp - Clang AST manager -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ASTManager.h"
#include "JSONRPCDispatcher.h"
#include "Protocol.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"
#include <mutex>
#include <thread>
using namespace clang;
using namespace clangd;

/// Retrieve a copy of the contents of every file in the store, for feeding into
/// ASTUnit.
static std::vector<ASTUnit::RemappedFile>
getRemappedFiles(const DocumentStore &Docs) {
  // FIXME: Use VFS instead. This would allow us to get rid of the chdir below.
  std::vector<ASTUnit::RemappedFile> RemappedFiles;
  for (const auto &P : Docs.getAllDocuments()) {
    StringRef FileName = P.first;
    RemappedFiles.push_back(ASTUnit::RemappedFile(
        FileName,
        llvm::MemoryBuffer::getMemBufferCopy(P.second, FileName).release()));
  }
  return RemappedFiles;
}

/// Convert from clang diagnostic level to LSP severity.
static int getSeverity(DiagnosticsEngine::Level L) {
  switch (L) {
  case DiagnosticsEngine::Remark:
    return 4;
  case DiagnosticsEngine::Note:
    return 3;
  case DiagnosticsEngine::Warning:
    return 2;
  case DiagnosticsEngine::Fatal:
  case DiagnosticsEngine::Error:
    return 1;
  case DiagnosticsEngine::Ignored:
    return 0;
  }
  llvm_unreachable("Unknown diagnostic level!");
}

ASTManager::ASTManager(JSONOutput &Output, DocumentStore &Store,
                       bool RunSynchronously)
    : Output(Output), Store(Store), RunSynchronously(RunSynchronously),
      PCHs(std::make_shared<PCHContainerOperations>()),
      ClangWorker([this]() { runWorker(); }) {}

void ASTManager::runWorker() {
  while (true) {
    std::string File;

    {
      std::unique_lock<std::mutex> Lock(RequestLock);
      // Check if there's another request pending. We keep parsing until
      // our one-element queue is empty.
      ClangRequestCV.wait(Lock,
                          [this] { return !RequestQueue.empty() || Done; });

      if (RequestQueue.empty() && Done)
        return;

      File = std::move(RequestQueue.back());
      RequestQueue.pop_back();
    } // unlock.

    parseFileAndPublishDiagnostics(File);
  }
}

void ASTManager::parseFileAndPublishDiagnostics(StringRef File) {
  DiagnosticToReplacementMap LocalFixIts; // Temporary storage
  std::string Diagnostics;
  {
    std::lock_guard<std::mutex> ASTGuard(ASTLock);
    auto &Unit = ASTs[File]; // Only one thread can access this at a time.

    if (!Unit) {
      Unit = createASTUnitForFile(File, this->Store);
    } else {
      // Do a reparse if this wasn't the first parse.
      // FIXME: This might have the wrong working directory if it changed in the
      // meantime.
      Unit->Reparse(PCHs, getRemappedFiles(this->Store));
    }

    if (!Unit)
      return;

    // Send the diagnotics to the editor.
    // FIXME: If the diagnostic comes from a different file, do we want to
    // show them all? Right now we drop everything not coming from the
    // main file.
    for (ASTUnit::stored_diag_iterator D = Unit->stored_diag_begin(),
                                       DEnd = Unit->stored_diag_end();
         D != DEnd; ++D) {
      if (!D->getLocation().isValid() ||
          !D->getLocation().getManager().isInMainFile(D->getLocation()))
        continue;
      Position P;
      P.line = D->getLocation().getSpellingLineNumber() - 1;
      P.character = D->getLocation().getSpellingColumnNumber();
      Range R = {P, P};
      Diagnostics +=
          R"({"range":)" + Range::unparse(R) +
          R"(,"severity":)" + std::to_string(getSeverity(D->getLevel())) +
          R"(,"message":")" + llvm::yaml::escape(D->getMessage()) +
          R"("},)";

      // We convert to Replacements to become independent of the SourceManager.
      clangd::Diagnostic Diag = {R, getSeverity(D->getLevel()),
                                 D->getMessage()};
      auto &FixItsForDiagnostic = LocalFixIts[Diag];
      for (const FixItHint &Fix : D->getFixIts()) {
        FixItsForDiagnostic.push_back(clang::tooling::Replacement(
            Unit->getSourceManager(), Fix.RemoveRange, Fix.CodeToInsert));
      }
    }
  } // unlock ASTLock

  // Put FixIts into place.
  {
    std::lock_guard<std::mutex> Guard(FixItLock);
    FixIts = std::move(LocalFixIts);
  }

  if (!Diagnostics.empty())
    Diagnostics.pop_back(); // Drop trailing comma.
  Output.writeMessage(
      R"({"jsonrpc":"2.0","method":"textDocument/publishDiagnostics","params":{"uri":")" +
      URI::fromFile(File).uri + R"(","diagnostics":[)" + Diagnostics + R"(]}})");
}

ASTManager::~ASTManager() {
  {
    std::lock_guard<std::mutex> Guard(RequestLock);
    // Wake up the clang worker thread, then exit.
    Done = true;
    ClangRequestCV.notify_one();
  }
  ClangWorker.join();
}

void ASTManager::onDocumentAdd(StringRef File) {
  if (RunSynchronously) {
    parseFileAndPublishDiagnostics(File);
    return;
  }
  std::lock_guard<std::mutex> Guard(RequestLock);
  // Currently we discard all pending requests and just enqueue the latest one.
  RequestQueue.clear();
  RequestQueue.push_back(File);
  ClangRequestCV.notify_one();
}

tooling::CompilationDatabase *
ASTManager::getOrCreateCompilationDatabaseForFile(StringRef File) {
  auto &I = CompilationDatabases[File];
  if (I)
    return I.get();

  std::string Error;
  I = tooling::CompilationDatabase::autoDetectFromSource(File, Error);
  Output.log("Failed to load compilation database: " + Twine(Error) + "\n");
  return I.get();
}

std::unique_ptr<clang::ASTUnit>
ASTManager::createASTUnitForFile(StringRef File, const DocumentStore &Docs) {
  tooling::CompilationDatabase *CDB =
      getOrCreateCompilationDatabaseForFile(File);

  std::vector<tooling::CompileCommand> Commands;

  if (CDB) {
    Commands = CDB->getCompileCommands(File);
    // chdir. This is thread hostile.
    if (!Commands.empty())
      llvm::sys::fs::set_current_path(Commands.front().Directory);
  }
  if (Commands.empty()) {
    // Add a fake command line if we know nothing.
    Commands.push_back(tooling::CompileCommand(
        llvm::sys::path::parent_path(File), llvm::sys::path::filename(File),
        {"clang", "-fsyntax-only", File.str()}, ""));
  }

  // Inject the resource dir.
  // FIXME: Don't overwrite it if it's already there.
  static int Dummy; // Just an address in this process.
  std::string ResourceDir =
      CompilerInvocation::GetResourcesPath("clangd", (void *)&Dummy);
  Commands.front().CommandLine.push_back("-resource-dir=" + ResourceDir);

  IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
      CompilerInstance::createDiagnostics(new DiagnosticOptions);

  std::vector<const char *> ArgStrs;
  for (const auto &S : Commands.front().CommandLine)
    ArgStrs.push_back(S.c_str());

  auto ArgP = &*ArgStrs.begin();
  return std::unique_ptr<clang::ASTUnit>(ASTUnit::LoadFromCommandLine(
      ArgP, ArgP + ArgStrs.size(), PCHs, Diags, ResourceDir,
      /*OnlyLocalDecls=*/false, /*CaptureDiagnostics=*/true,
      getRemappedFiles(Docs),
      /*RemappedFilesKeepOriginalName=*/true,
      /*PrecompilePreambleAfterNParses=*/1, /*TUKind=*/TU_Complete,
      /*CacheCodeCompletionResults=*/true,
      /*IncludeBriefCommentsInCodeCompletion=*/true,
      /*AllowPCHWithCompilerErrors=*/true));
}

std::vector<clang::tooling::Replacement>
ASTManager::getFixIts(const clangd::Diagnostic &D) {
  std::lock_guard<std::mutex> Guard(FixItLock);
  auto I = FixIts.find(D);
  if (I != FixIts.end())
    return I->second;
  return {};
}

namespace {

class CompletionItemsCollector : public CodeCompleteConsumer {
  std::vector<CompletionItem> *Items;
  std::shared_ptr<clang::GlobalCodeCompletionAllocator> Allocator;
  CodeCompletionTUInfo CCTUInfo;

public:
  CompletionItemsCollector(std::vector<CompletionItem> *Items,
                           const CodeCompleteOptions &CodeCompleteOpts)
      : CodeCompleteConsumer(CodeCompleteOpts, /*OutputIsBinary=*/false),
        Items(Items),
        Allocator(std::make_shared<clang::GlobalCodeCompletionAllocator>()),
        CCTUInfo(Allocator) {}

  void ProcessCodeCompleteResults(Sema &S, CodeCompletionContext Context,
                                  CodeCompletionResult *Results,
                                  unsigned NumResults) override {
    for (unsigned I = 0; I != NumResults; ++I) {
      CodeCompletionString *CCS = Results[I].CreateCodeCompletionString(
          S, Context, *Allocator, CCTUInfo,
          CodeCompleteOpts.IncludeBriefComments);
      if (CCS) {
        CompletionItem Item;
        assert(CCS->getTypedText());
        Item.label = CCS->getTypedText();
        if (CCS->getBriefComment())
          Item.documentation = CCS->getBriefComment();
        Items->push_back(std::move(Item));
      }
    }
  }

  GlobalCodeCompletionAllocator &getAllocator() override { return *Allocator; }

  CodeCompletionTUInfo &getCodeCompletionTUInfo() override { return CCTUInfo; }
};

} // namespace

std::vector<CompletionItem>
ASTManager::codeComplete(StringRef File, unsigned Line, unsigned Column) {
  CodeCompleteOptions CCO;
  CCO.IncludeBriefComments = 1;
  // This is where code completion stores dirty buffers. Need to free after
  // completion.
  SmallVector<const llvm::MemoryBuffer *, 4> OwnedBuffers;
  SmallVector<StoredDiagnostic, 4> StoredDiagnostics;
  IntrusiveRefCntPtr<DiagnosticsEngine> DiagEngine(
      new DiagnosticsEngine(new DiagnosticIDs, new DiagnosticOptions));
  std::vector<CompletionItem> Items;
  CompletionItemsCollector Collector(&Items, CCO);
  std::lock_guard<std::mutex> Guard(ASTLock);
  auto &Unit = ASTs[File];
  if (!Unit)
    Unit = createASTUnitForFile(File, this->Store);
  if (!Unit)
    return {};
  IntrusiveRefCntPtr<SourceManager> SourceMgr(
      new SourceManager(*DiagEngine, Unit->getFileManager()));
  // CodeComplete seems to require fresh LangOptions.
  LangOptions LangOpts = Unit->getLangOpts();
  // The language server protocol uses zero-based line and column numbers.
  // The clang code completion uses one-based numbers.
  Unit->CodeComplete(File, Line + 1, Column + 1, getRemappedFiles(this->Store),
                     CCO.IncludeMacros, CCO.IncludeCodePatterns,
                     CCO.IncludeBriefComments, Collector, PCHs, *DiagEngine,
                     LangOpts, *SourceMgr, Unit->getFileManager(),
                     StoredDiagnostics, OwnedBuffers);
  for (const llvm::MemoryBuffer *Buffer : OwnedBuffers)
    delete Buffer;
  return Items;
}
