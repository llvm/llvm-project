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

void DocData::setAST(std::unique_ptr<ASTUnit> AST) {
  this->AST = std::move(AST);
}

ASTUnit *DocData::getAST() const { return AST.get(); }

void DocData::cacheFixIts(DiagnosticToReplacementMap FixIts) {
  this->FixIts = std::move(FixIts);
}

std::vector<clang::tooling::Replacement>
DocData::getFixIts(const clangd::Diagnostic &D) const {
  auto it = FixIts.find(D);
  if (it != FixIts.end())
    return it->second;
  return {};
}

ASTManagerRequest::ASTManagerRequest(ASTManagerRequestType Type,
                                     std::string File,
                                     DocVersion Version)
    : Type(Type), File(File), Version(Version) {}

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

static CompletionItemKind getKind(CXCursorKind K) {
  switch (K) {
  case CXCursor_MacroInstantiation:
  case CXCursor_MacroDefinition:
    return CompletionItemKind::Text;
  case CXCursor_CXXMethod:
    return CompletionItemKind::Method;
  case CXCursor_FunctionDecl:
  case CXCursor_FunctionTemplate:
    return CompletionItemKind::Function;
  case CXCursor_Constructor:
  case CXCursor_Destructor:
    return CompletionItemKind::Constructor;
  case CXCursor_FieldDecl:
    return CompletionItemKind::Field;
  case CXCursor_VarDecl:
  case CXCursor_ParmDecl:
    return CompletionItemKind::Variable;
  case CXCursor_ClassDecl:
  case CXCursor_StructDecl:
  case CXCursor_UnionDecl:
  case CXCursor_ClassTemplate:
  case CXCursor_ClassTemplatePartialSpecialization:
    return CompletionItemKind::Class;
  case CXCursor_Namespace:
  case CXCursor_NamespaceAlias:
  case CXCursor_NamespaceRef:
    return CompletionItemKind::Module;
  case CXCursor_EnumConstantDecl:
    return CompletionItemKind::Value;
  case CXCursor_EnumDecl:
    return CompletionItemKind::Enum;
  case CXCursor_TypeAliasDecl:
  case CXCursor_TypeAliasTemplateDecl:
  case CXCursor_TypedefDecl:
  case CXCursor_MemberRef:
  case CXCursor_TypeRef:
    return CompletionItemKind::Reference;
  default:
    return CompletionItemKind::Missing;
  }
}

ASTManager::ASTManager(JSONOutput &Output, DocumentStore &Store,
                       bool RunSynchronously)
    : Output(Output), Store(Store), RunSynchronously(RunSynchronously),
      PCHs(std::make_shared<PCHContainerOperations>()),
      ClangWorker([this]() { runWorker(); }) {}

void ASTManager::runWorker() {
  while (true) {
    ASTManagerRequest Request;

    // Pick request from the queue
    {
      std::unique_lock<std::mutex> Lock(RequestLock);
      // Wait for more requests.
      ClangRequestCV.wait(Lock,
                          [this] { return !RequestQueue.empty() || Done; });
      if (Done)
        return;
      assert(!RequestQueue.empty() && "RequestQueue was empty");

      Request = std::move(RequestQueue.back());
      RequestQueue.pop_back();

      // Skip outdated requests
      if (Request.Version != DocVersions.find(Request.File)->second) {
        Output.log("Version for " + Twine(Request.File) +
                   " in request is outdated, skipping request\n");
        continue;
      }
    } // unlock RequestLock

    handleRequest(Request.Type, Request.File);
  }
}

void ASTManager::queueOrRun(ASTManagerRequestType RequestType, StringRef File) {
  if (RunSynchronously) {
    handleRequest(RequestType, File);
    return;
  }

  std::lock_guard<std::mutex> Guard(RequestLock);
  // We increment the version of the added document immediately and schedule
  // the requested operation to be run on a worker thread
  DocVersion version = ++DocVersions[File];
  RequestQueue.push_back(ASTManagerRequest(RequestType, File, version));
  ClangRequestCV.notify_one();
}

void ASTManager::handleRequest(ASTManagerRequestType RequestType,
                               StringRef File) {
  switch (RequestType) {
  case ASTManagerRequestType::ParseAndPublishDiagnostics:
    parseFileAndPublishDiagnostics(File);
    break;
  case ASTManagerRequestType::RemoveDocData: {
    std::lock_guard<std::mutex> Lock(ClangObjectLock);
    auto DocDataIt = DocDatas.find(File);
    // We could get the remove request before parsing for the document is
    // started, just do nothing in that case, parsing request will be discarded
    // because it has a lower version value
    if (DocDataIt == DocDatas.end())
      return;
    DocDatas.erase(DocDataIt);
    break;
  } // unlock ClangObjectLock
  }
}

void ASTManager::parseFileAndPublishDiagnostics(StringRef File) {
  std::unique_lock<std::mutex> ClangObjectLockGuard(ClangObjectLock);

  auto &DocData = DocDatas[File];
  ASTUnit *Unit = DocData.getAST();
  if (!Unit) {
    auto newAST = createASTUnitForFile(File, this->Store);
    Unit = newAST.get();

    DocData.setAST(std::move(newAST));
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
  std::string Diagnostics;
  DocData::DiagnosticToReplacementMap LocalFixIts; // Temporary storage
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
    clangd::Diagnostic Diag = {R, getSeverity(D->getLevel()), D->getMessage()};
    auto &FixItsForDiagnostic = LocalFixIts[Diag];
    for (const FixItHint &Fix : D->getFixIts()) {
      FixItsForDiagnostic.push_back(clang::tooling::Replacement(
          Unit->getSourceManager(), Fix.RemoveRange, Fix.CodeToInsert));
    }
  }

  // Put FixIts into place.
  DocData.cacheFixIts(std::move(LocalFixIts));

  ClangObjectLockGuard.unlock();
  // No accesses to clang objects are allowed after this point.

  // Publish diagnostics.
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
  } // unlock DocDataLock
  ClangWorker.join();
}

void ASTManager::onDocumentAdd(StringRef File) {
  queueOrRun(ASTManagerRequestType::ParseAndPublishDiagnostics, File);
}

void ASTManager::onDocumentRemove(StringRef File) {
  queueOrRun(ASTManagerRequestType::RemoveDocData, File);
}

tooling::CompilationDatabase *
ASTManager::getOrCreateCompilationDatabaseForFile(StringRef File) {
  namespace path = llvm::sys::path;

  assert((path::is_absolute(File, path::Style::posix) ||
          path::is_absolute(File, path::Style::windows)) &&
         "path must be absolute");

  for (auto Path = path::parent_path(File); !Path.empty();
       Path = path::parent_path(Path)) {

    auto CachedIt = CompilationDatabases.find(Path);
    if (CachedIt != CompilationDatabases.end())
      return CachedIt->second.get();
    std::string Error;
    auto CDB = tooling::CompilationDatabase::loadFromDirectory(Path, Error);
    if (!CDB) {
      if (!Error.empty()) {
        Output.log("Error when trying to load compilation database from " +
                   Twine(Path) + ": " + Twine(Error) + "\n");
      }
      continue;
    }

    // TODO(ibiryukov): Invalidate cached compilation databases on changes
    auto result = CDB.get();
    CompilationDatabases.insert(std::make_pair(Path, std::move(CDB)));
    return result;
  }

  Output.log("Failed to find compilation database for " + Twine(File) + "\n");
  return nullptr;
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
ASTManager::getFixIts(StringRef File, const clangd::Diagnostic &D) {
  // TODO(ibiryukov): the FixIts should be available immediately
  // even when parsing is being run on a worker thread
  std::lock_guard<std::mutex> Guard(ClangObjectLock);
  return DocDatas[File].getFixIts(D);
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
      CodeCompletionResult &Result = Results[I];
      CodeCompletionString *CCS = Result.CreateCodeCompletionString(
          S, Context, *Allocator, CCTUInfo,
          CodeCompleteOpts.IncludeBriefComments);
      if (CCS) {
        CompletionItem Item;
        assert(CCS->getTypedText());
        Item.label = CCS->getTypedText();
        Item.kind = getKind(Result.CursorKind);
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

  std::lock_guard<std::mutex> Guard(ClangObjectLock);
  auto &DocData = DocDatas[File];
  auto Unit = DocData.getAST();
  if (!Unit) {
    auto newAST = createASTUnitForFile(File, this->Store);
    Unit = newAST.get();
    DocData.setAST(std::move(newAST));
  }
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
