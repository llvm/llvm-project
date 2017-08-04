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
#include "clang/Frontend/Utils.h"
#include "clang/Index/IndexingAction.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/Support/Format.h"

#include <algorithm>

using namespace clang::clangd;
using namespace clang;

ClangdUnit::ClangdUnit(PathRef FileName, StringRef Contents,
                       StringRef ResourceDir,
                       std::shared_ptr<PCHContainerOperations> PCHs,
                       std::vector<tooling::CompileCommand> Commands,
                       IntrusiveRefCntPtr<vfs::FileSystem> VFS)
    : FileName(FileName), PCHs(PCHs) {
  assert(!Commands.empty() && "No compile commands provided");

  // Inject the resource dir.
  // FIXME: Don't overwrite it if it's already there.
  Commands.front().CommandLine.push_back("-resource-dir=" +
                                         std::string(ResourceDir));

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
      /*PrecompilePreambleAfterNParses=*/1, /*TUKind=*/TU_Prefix,
      /*CacheCodeCompletionResults=*/true,
      /*IncludeBriefCommentsInCodeCompletion=*/true,
      /*AllowPCHWithCompilerErrors=*/true,
      /*SkipFunctionBodies=*/false,
      /*SingleFileParse=*/false,
      /*UserFilesAreVolatile=*/false, /*ForSerialization=*/false,
      /*ModuleFormat=*/llvm::None,
      /*ErrAST=*/nullptr, VFS));
  assert(Unit && "Unit wasn't created");
}

void ClangdUnit::reparse(StringRef Contents,
                         IntrusiveRefCntPtr<vfs::FileSystem> VFS) {
  // Do a reparse if this wasn't the first parse.
  // FIXME: This might have the wrong working directory if it changed in the
  // meantime.
  ASTUnit::RemappedFile RemappedSource(
      FileName,
      llvm::MemoryBuffer::getMemBufferCopy(Contents, FileName).release());

  Unit->Reparse(PCHs, RemappedSource, VFS);
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
        for (CodeCompletionString::Chunk C : *CCS) {
          switch (C.Kind) {
          case CodeCompletionString::CK_ResultType:
            Item.detail = C.Text;
            break;
          case CodeCompletionString::CK_Optional:
            break;
          default:
            Item.label += C.Text;
            break;
          }
        }
        assert(CCS->getTypedText());
        Item.kind = getKind(Result.CursorKind);
        // Priority is a 16-bit integer, hence at most 5 digits.
        // Since identifiers with higher priority need to come first,
        // we subtract the priority from 99999.
        // For example, the sort text of the identifier 'a' with priority 35
        // is 99964a.
        assert(CCS->getPriority() < 99999 && "Expecting code completion result "
                                             "priority to have at most "
                                             "5-digits");
        llvm::raw_string_ostream(Item.sortText) << llvm::format(
            "%05d%s", 99999 - CCS->getPriority(), CCS->getTypedText());
        Item.insertText = Item.filterText = CCS->getTypedText();
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
ClangdUnit::codeComplete(StringRef Contents, Position Pos,
                         IntrusiveRefCntPtr<vfs::FileSystem> VFS) {
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

  IntrusiveRefCntPtr<FileManager> FileMgr(
      new FileManager(Unit->getFileSystemOpts(), VFS));
  IntrusiveRefCntPtr<SourceManager> SourceMgr(
      new SourceManager(*DiagEngine, *FileMgr));
  // CodeComplete seems to require fresh LangOptions.
  LangOptions LangOpts = Unit->getLangOpts();
  // The language server protocol uses zero-based line and column numbers.
  // The clang code completion uses one-based numbers.
  Unit->CodeComplete(FileName, Pos.line + 1, Pos.character + 1, RemappedSource,
                     CCO.IncludeMacros, CCO.IncludeCodePatterns,
                     CCO.IncludeBriefComments, Collector, PCHs, *DiagEngine,
                     LangOpts, *SourceMgr, *FileMgr, StoredDiagnostics,
                     OwnedBuffers);
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

void ClangdUnit::dumpAST(llvm::raw_ostream &OS) const {
  Unit->getASTContext().getTranslationUnitDecl()->dump(OS, true);
}

namespace {
/// Finds declarations locations that a given source location refers to.
class DeclarationLocationsFinder : public index::IndexDataConsumer {
  std::vector<Location> DeclarationLocations;
  const SourceLocation &SearchedLocation;
  ASTUnit &Unit;
public:
  DeclarationLocationsFinder(raw_ostream &OS,
      const SourceLocation &SearchedLocation, ASTUnit &Unit) :
      SearchedLocation(SearchedLocation), Unit(Unit) {}

  std::vector<Location> takeLocations() {
    // Don't keep the same location multiple times.
    // This can happen when nodes in the AST are visited twice.
    std::sort(DeclarationLocations.begin(), DeclarationLocations.end());
    auto last =
        std::unique(DeclarationLocations.begin(), DeclarationLocations.end());
    DeclarationLocations.erase(last, DeclarationLocations.end());
    return std::move(DeclarationLocations);
  }

  bool handleDeclOccurence(const Decl* D, index::SymbolRoleSet Roles,
      ArrayRef<index::SymbolRelation> Relations, FileID FID, unsigned Offset,
      index::IndexDataConsumer::ASTNodeInfo ASTNode) override
      {
    if (isSearchedLocation(FID, Offset)) {
      addDeclarationLocation(D->getSourceRange());
    }
    return true;
  }

private:
  bool isSearchedLocation(FileID FID, unsigned Offset) const {
    const SourceManager &SourceMgr = Unit.getSourceManager();
    return SourceMgr.getFileOffset(SearchedLocation) == Offset
        && SourceMgr.getFileID(SearchedLocation) == FID;
  }

  void addDeclarationLocation(const SourceRange& ValSourceRange) {
    const SourceManager& SourceMgr = Unit.getSourceManager();
    const LangOptions& LangOpts = Unit.getLangOpts();
    SourceLocation LocStart = ValSourceRange.getBegin();
    SourceLocation LocEnd = Lexer::getLocForEndOfToken(ValSourceRange.getEnd(),
        0, SourceMgr, LangOpts);
    Position Begin;
    Begin.line = SourceMgr.getSpellingLineNumber(LocStart) - 1;
    Begin.character = SourceMgr.getSpellingColumnNumber(LocStart) - 1;
    Position End;
    End.line = SourceMgr.getSpellingLineNumber(LocEnd) - 1;
    End.character = SourceMgr.getSpellingColumnNumber(LocEnd) - 1;
    Range R = {Begin, End};
    Location L;
    L.uri = URI::fromFile(
        SourceMgr.getFilename(SourceMgr.getSpellingLoc(LocStart)));
    L.range = R;
    DeclarationLocations.push_back(L);
  }

  void finish() override {
    // Also handle possible macro at the searched location.
    Token Result;
    if (!Lexer::getRawToken(SearchedLocation, Result, Unit.getSourceManager(),
        Unit.getASTContext().getLangOpts(), false)) {
      if (Result.is(tok::raw_identifier)) {
        Unit.getPreprocessor().LookUpIdentifierInfo(Result);
      }
      IdentifierInfo* IdentifierInfo = Result.getIdentifierInfo();
      if (IdentifierInfo && IdentifierInfo->hadMacroDefinition()) {
        std::pair<FileID, unsigned int> DecLoc =
            Unit.getSourceManager().getDecomposedExpansionLoc(SearchedLocation);
        // Get the definition just before the searched location so that a macro
        // referenced in a '#undef MACRO' can still be found.
        SourceLocation BeforeSearchedLocation = Unit.getLocation(
            Unit.getSourceManager().getFileEntryForID(DecLoc.first),
            DecLoc.second - 1);
        MacroDefinition MacroDef =
            Unit.getPreprocessor().getMacroDefinitionAtLoc(IdentifierInfo,
                BeforeSearchedLocation);
        MacroInfo* MacroInf = MacroDef.getMacroInfo();
        if (MacroInf) {
          addDeclarationLocation(
              SourceRange(MacroInf->getDefinitionLoc(),
                  MacroInf->getDefinitionEndLoc()));
        }
      }
    }
  }
};
} // namespace

std::vector<Location> ClangdUnit::findDefinitions(Position Pos) {
  const FileEntry *FE = Unit->getFileManager().getFile(Unit->getMainFileName());
  if (!FE)
    return {};

  SourceLocation SourceLocationBeg = getBeginningOfIdentifier(Pos, FE);

  auto DeclLocationsFinder = std::make_shared<DeclarationLocationsFinder>(
      llvm::errs(), SourceLocationBeg, *Unit);
  index::IndexingOptions IndexOpts;
  IndexOpts.SystemSymbolFilter =
      index::IndexingOptions::SystemSymbolFilterKind::All;
  IndexOpts.IndexFunctionLocals = true;
  index::indexASTUnit(*Unit, DeclLocationsFinder, IndexOpts);

  return DeclLocationsFinder->takeLocations();
}

SourceLocation ClangdUnit::getBeginningOfIdentifier(const Position &Pos,
    const FileEntry *FE) const {
  // The language server protocol uses zero-based line and column numbers.
  // Clang uses one-based numbers.
  SourceLocation InputLocation = Unit->getLocation(FE, Pos.line + 1,
      Pos.character + 1);

  if (Pos.character == 0) {
    return InputLocation;
  }

  // This handle cases where the position is in the middle of a token or right
  // after the end of a token. In theory we could just use GetBeginningOfToken
  // to find the start of the token at the input position, but this doesn't
  // work when right after the end, i.e. foo|.
  // So try to go back by one and see if we're still inside the an identifier
  // token. If so, Take the beginning of this token.
  // (It should be the same identifier because you can't have two adjacent
  // identifiers without another token in between.)
  SourceLocation PeekBeforeLocation = Unit->getLocation(FE, Pos.line + 1,
      Pos.character);
  const SourceManager &SourceMgr = Unit->getSourceManager();
  Token Result;
  if (Lexer::getRawToken(PeekBeforeLocation, Result, SourceMgr,
                         Unit->getASTContext().getLangOpts(), false)) {
    // getRawToken failed, just use InputLocation.
    return InputLocation;
  }

  if (Result.is(tok::raw_identifier)) {
    return Lexer::GetBeginningOfToken(PeekBeforeLocation, SourceMgr,
        Unit->getASTContext().getLangOpts());
  }

  return InputLocation;
}
