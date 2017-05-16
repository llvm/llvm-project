//===--- ClangdUnit.cpp -----------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include "ClangdUnit.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Tooling/CompilationDatabase.h"

using namespace clang::clangd;
using namespace clang;

ClangdUnit::ClangdUnit(PathRef FileName, StringRef Contents,
                       std::shared_ptr<PCHContainerOperations> PCHs,
                       std::vector<tooling::CompileCommand> Commands)
    : FileName(FileName), PCHs(PCHs) {
  assert(!Commands.empty() && "No compile commands provided");

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

  ASTUnit::RemappedFile RemappedSource(
      FileName,
      llvm::MemoryBuffer::getMemBufferCopy(Contents, FileName).release());

  auto ArgP = &*ArgStrs.begin();
  Unit = std::unique_ptr<ASTUnit>(ASTUnit::LoadFromCommandLine(
      ArgP, ArgP + ArgStrs.size(), PCHs, Diags, ResourceDir,
      /*OnlyLocalDecls=*/false, /*CaptureDiagnostics=*/true, RemappedSource,
      /*RemappedFilesKeepOriginalName=*/true,
      /*PrecompilePreambleAfterNParses=*/1, /*TUKind=*/TU_Complete,
      /*CacheCodeCompletionResults=*/true,
      /*IncludeBriefCommentsInCodeCompletion=*/true,
      /*AllowPCHWithCompilerErrors=*/true));
}

void ClangdUnit::reparse(StringRef Contents) {
  // Do a reparse if this wasn't the first parse.
  // FIXME: This might have the wrong working directory if it changed in the
  // meantime.
  ASTUnit::RemappedFile RemappedSource(
      FileName,
      llvm::MemoryBuffer::getMemBufferCopy(Contents, FileName).release());

  Unit->Reparse(PCHs, RemappedSource);
}

namespace {

CompletionItemKind getKind(CXCursorKind K) {
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

std::vector<CompletionItem> ClangdUnit::codeComplete(StringRef Contents,
                                                     Position Pos) {
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

  ASTUnit::RemappedFile RemappedSource(
      FileName,
      llvm::MemoryBuffer::getMemBufferCopy(Contents, FileName).release());

  IntrusiveRefCntPtr<SourceManager> SourceMgr(
      new SourceManager(*DiagEngine, Unit->getFileManager()));
  // CodeComplete seems to require fresh LangOptions.
  LangOptions LangOpts = Unit->getLangOpts();
  // The language server protocol uses zero-based line and column numbers.
  // The clang code completion uses one-based numbers.
  Unit->CodeComplete(FileName, Pos.line + 1, Pos.character + 1, RemappedSource,
                     CCO.IncludeMacros, CCO.IncludeCodePatterns,
                     CCO.IncludeBriefComments, Collector, PCHs, *DiagEngine,
                     LangOpts, *SourceMgr, Unit->getFileManager(),
                     StoredDiagnostics, OwnedBuffers);
  for (const llvm::MemoryBuffer *Buffer : OwnedBuffers)
    delete Buffer;
  return Items;
}

namespace {
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
} // namespace

std::vector<DiagWithFixIts> ClangdUnit::getLocalDiagnostics() const {
  std::vector<DiagWithFixIts> Result;
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
    clangd::Diagnostic Diag = {R, getSeverity(D->getLevel()), D->getMessage()};

    llvm::SmallVector<tooling::Replacement, 1> FixItsForDiagnostic;
    for (const FixItHint &Fix : D->getFixIts()) {
      FixItsForDiagnostic.push_back(clang::tooling::Replacement(
          Unit->getSourceManager(), Fix.RemoveRange, Fix.CodeToInsert));
    }
    Result.push_back({Diag, std::move(FixItsForDiagnostic)});
  }
  return Result;
}
