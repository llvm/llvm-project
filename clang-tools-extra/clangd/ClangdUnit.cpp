//===--- ClangdUnit.cpp -----------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include "ClangdUnit.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/Utils.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Index/IndexingAction.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Sema/Sema.h"
#include "clang/Serialization/ASTWriter.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/Format.h"

#include <algorithm>

using namespace clang::clangd;
using namespace clang;

namespace {

class DeclTrackingASTConsumer : public ASTConsumer {
public:
  DeclTrackingASTConsumer(std::vector<const Decl *> &TopLevelDecls)
      : TopLevelDecls(TopLevelDecls) {}

  bool HandleTopLevelDecl(DeclGroupRef DG) override {
    for (const Decl *D : DG) {
      // ObjCMethodDecl are not actually top-level decls.
      if (isa<ObjCMethodDecl>(D))
        continue;

      TopLevelDecls.push_back(D);
    }
    return true;
  }

private:
  std::vector<const Decl *> &TopLevelDecls;
};

class ClangdFrontendAction : public SyntaxOnlyAction {
public:
  std::vector<const Decl *> takeTopLevelDecls() {
    return std::move(TopLevelDecls);
  }

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    return llvm::make_unique<DeclTrackingASTConsumer>(/*ref*/ TopLevelDecls);
  }

private:
  std::vector<const Decl *> TopLevelDecls;
};

class ClangdUnitPreambleCallbacks : public PreambleCallbacks {
public:
  std::vector<serialization::DeclID> takeTopLevelDeclIDs() {
    return std::move(TopLevelDeclIDs);
  }

  void AfterPCHEmitted(ASTWriter &Writer) override {
    TopLevelDeclIDs.reserve(TopLevelDecls.size());
    for (Decl *D : TopLevelDecls) {
      // Invalid top-level decls may not have been serialized.
      if (D->isInvalidDecl())
        continue;
      TopLevelDeclIDs.push_back(Writer.getDeclID(D));
    }
  }

  void HandleTopLevelDecl(DeclGroupRef DG) override {
    for (Decl *D : DG) {
      if (isa<ObjCMethodDecl>(D))
        continue;
      TopLevelDecls.push_back(D);
    }
  }

private:
  std::vector<Decl *> TopLevelDecls;
  std::vector<serialization::DeclID> TopLevelDeclIDs;
};

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

llvm::Optional<DiagWithFixIts> toClangdDiag(StoredDiagnostic D) {
  auto Location = D.getLocation();
  if (!Location.isValid() || !Location.getManager().isInMainFile(Location))
    return llvm::None;

  Position P;
  P.line = Location.getSpellingLineNumber() - 1;
  P.character = Location.getSpellingColumnNumber();
  Range R = {P, P};
  clangd::Diagnostic Diag = {R, getSeverity(D.getLevel()), D.getMessage()};

  llvm::SmallVector<tooling::Replacement, 1> FixItsForDiagnostic;
  for (const FixItHint &Fix : D.getFixIts()) {
    FixItsForDiagnostic.push_back(clang::tooling::Replacement(
        Location.getManager(), Fix.RemoveRange, Fix.CodeToInsert));
  }
  return DiagWithFixIts{Diag, std::move(FixItsForDiagnostic)};
}

class StoreDiagsConsumer : public DiagnosticConsumer {
public:
  StoreDiagsConsumer(std::vector<DiagWithFixIts> &Output) : Output(Output) {}

  void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                        const clang::Diagnostic &Info) override {
    DiagnosticConsumer::HandleDiagnostic(DiagLevel, Info);

    if (auto convertedDiag = toClangdDiag(StoredDiagnostic(DiagLevel, Info)))
      Output.push_back(std::move(*convertedDiag));
  }

private:
  std::vector<DiagWithFixIts> &Output;
};

class EmptyDiagsConsumer : public DiagnosticConsumer {
public:
  void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                        const clang::Diagnostic &Info) override {}
};

std::unique_ptr<CompilerInvocation>
createCompilerInvocation(ArrayRef<const char *> ArgList,
                         IntrusiveRefCntPtr<DiagnosticsEngine> Diags,
                         IntrusiveRefCntPtr<vfs::FileSystem> VFS) {
  auto CI = createInvocationFromCommandLine(ArgList, std::move(Diags),
                                            std::move(VFS));
  // We rely on CompilerInstance to manage the resource (i.e. free them on
  // EndSourceFile), but that won't happen if DisableFree is set to true.
  // Since createInvocationFromCommandLine sets it to true, we have to override
  // it.
  CI->getFrontendOpts().DisableFree = false;
  return CI;
}

/// Creates a CompilerInstance from \p CI, with main buffer overriden to \p
/// Buffer and arguments to read the PCH from \p Preamble, if \p Preamble is not
/// null. Note that vfs::FileSystem inside returned instance may differ from \p
/// VFS if additional file remapping were set in command-line arguments.
/// On some errors, returns null. When non-null value is returned, it's expected
/// to be consumed by the FrontendAction as it will have a pointer to the \p
/// Buffer that will only be deleted if BeginSourceFile is called.
std::unique_ptr<CompilerInstance>
prepareCompilerInstance(std::unique_ptr<clang::CompilerInvocation> CI,
                        const PrecompiledPreamble *Preamble,
                        std::unique_ptr<llvm::MemoryBuffer> Buffer,
                        std::shared_ptr<PCHContainerOperations> PCHs,
                        IntrusiveRefCntPtr<vfs::FileSystem> VFS,
                        DiagnosticConsumer &DiagsClient) {
  assert(VFS && "VFS is null");
  assert(!CI->getPreprocessorOpts().RetainRemappedFileBuffers &&
         "Setting RetainRemappedFileBuffers to true will cause a memory leak "
         "of ContentsBuffer");

  // NOTE: we use Buffer.get() when adding remapped files, so we have to make
  // sure it will be released if no error is emitted.
  if (Preamble) {
    Preamble->AddImplicitPreamble(*CI, Buffer.get());
  } else {
    CI->getPreprocessorOpts().addRemappedFile(
        CI->getFrontendOpts().Inputs[0].getFile(), Buffer.get());
  }

  auto Clang = llvm::make_unique<CompilerInstance>(PCHs);
  Clang->setInvocation(std::move(CI));
  Clang->createDiagnostics(&DiagsClient, false);

  if (auto VFSWithRemapping = createVFSFromCompilerInvocation(
          Clang->getInvocation(), Clang->getDiagnostics(), VFS))
    VFS = VFSWithRemapping;
  Clang->setVirtualFileSystem(VFS);

  Clang->setTarget(TargetInfo::CreateTargetInfo(
      Clang->getDiagnostics(), Clang->getInvocation().TargetOpts));
  if (!Clang->hasTarget())
    return nullptr;

  // RemappedFileBuffers will handle the lifetime of the Buffer pointer,
  // release it.
  Buffer.release();
  return Clang;
}

} // namespace

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

  Command = std::move(Commands.front());
  reparse(Contents, VFS);
}

void ClangdUnit::reparse(StringRef Contents,
                         IntrusiveRefCntPtr<vfs::FileSystem> VFS) {
  std::vector<const char *> ArgStrs;
  for (const auto &S : Command.CommandLine)
    ArgStrs.push_back(S.c_str());

  VFS->setCurrentWorkingDirectory(Command.Directory);

  std::unique_ptr<CompilerInvocation> CI;
  {
    // FIXME(ibiryukov): store diagnostics from CommandLine when we start
    // reporting them.
    EmptyDiagsConsumer CommandLineDiagsConsumer;
    IntrusiveRefCntPtr<DiagnosticsEngine> CommandLineDiagsEngine =
        CompilerInstance::createDiagnostics(new DiagnosticOptions,
                                            &CommandLineDiagsConsumer, false);
    CI = createCompilerInvocation(ArgStrs, CommandLineDiagsEngine, VFS);
  }
  assert(CI && "Couldn't create CompilerInvocation");

  std::unique_ptr<llvm::MemoryBuffer> ContentsBuffer =
      llvm::MemoryBuffer::getMemBufferCopy(Contents, FileName);

  // Rebuild the preamble if it is missing or can not be reused.
  auto Bounds =
      ComputePreambleBounds(*CI->getLangOpts(), ContentsBuffer.get(), 0);
  if (!Preamble || !Preamble->Preamble.CanReuse(*CI, ContentsBuffer.get(),
                                                Bounds, VFS.get())) {
    std::vector<DiagWithFixIts> PreambleDiags;
    StoreDiagsConsumer PreambleDiagnosticsConsumer(/*ref*/ PreambleDiags);
    IntrusiveRefCntPtr<DiagnosticsEngine> PreambleDiagsEngine =
        CompilerInstance::createDiagnostics(
            &CI->getDiagnosticOpts(), &PreambleDiagnosticsConsumer, false);
    ClangdUnitPreambleCallbacks SerializedDeclsCollector;
    auto BuiltPreamble = PrecompiledPreamble::Build(
        *CI, ContentsBuffer.get(), Bounds, *PreambleDiagsEngine, VFS, PCHs,
        SerializedDeclsCollector);
    if (BuiltPreamble)
      Preamble = PreambleData(std::move(*BuiltPreamble),
                              SerializedDeclsCollector.takeTopLevelDeclIDs(),
                              std::move(PreambleDiags));
  }
  Unit = ParsedAST::Build(
      std::move(CI), Preamble ? &Preamble->Preamble : nullptr,
      Preamble ? llvm::makeArrayRef(Preamble->TopLevelDeclIDs) : llvm::None,
      std::move(ContentsBuffer), PCHs, VFS);
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
  std::vector<const char *> ArgStrs;
  for (const auto &S : Command.CommandLine)
    ArgStrs.push_back(S.c_str());

  VFS->setCurrentWorkingDirectory(Command.Directory);

  std::unique_ptr<CompilerInvocation> CI;
  EmptyDiagsConsumer DummyDiagsConsumer;
  {
    IntrusiveRefCntPtr<DiagnosticsEngine> CommandLineDiagsEngine =
        CompilerInstance::createDiagnostics(new DiagnosticOptions,
                                            &DummyDiagsConsumer, false);
    CI = createCompilerInvocation(ArgStrs, CommandLineDiagsEngine, VFS);
  }
  assert(CI && "Couldn't create CompilerInvocation");

  std::unique_ptr<llvm::MemoryBuffer> ContentsBuffer =
      llvm::MemoryBuffer::getMemBufferCopy(Contents, FileName);

  // Attempt to reuse the PCH from precompiled preamble, if it was built.
  const PrecompiledPreamble *PreambleForCompletion = nullptr;
  if (Preamble) {
    auto Bounds =
        ComputePreambleBounds(*CI->getLangOpts(), ContentsBuffer.get(), 0);
    if (Preamble->Preamble.CanReuse(*CI, ContentsBuffer.get(), Bounds,
                                    VFS.get()))
      PreambleForCompletion = &Preamble->Preamble;
  }

  auto Clang = prepareCompilerInstance(std::move(CI), PreambleForCompletion,
                                       std::move(ContentsBuffer), PCHs, VFS,
                                       DummyDiagsConsumer);
  auto &DiagOpts = Clang->getDiagnosticOpts();
  DiagOpts.IgnoreWarnings = true;

  auto &FrontendOpts = Clang->getFrontendOpts();
  FrontendOpts.SkipFunctionBodies = true;

  FrontendOpts.CodeCompleteOpts.IncludeGlobals = true;
  // we don't handle code patterns properly yet, disable them.
  FrontendOpts.CodeCompleteOpts.IncludeCodePatterns = false;
  FrontendOpts.CodeCompleteOpts.IncludeMacros = true;
  FrontendOpts.CodeCompleteOpts.IncludeBriefComments = true;

  FrontendOpts.CodeCompletionAt.FileName = FileName;
  FrontendOpts.CodeCompletionAt.Line = Pos.line + 1;
  FrontendOpts.CodeCompletionAt.Column = Pos.character + 1;

  std::vector<CompletionItem> Items;
  Clang->setCodeCompletionConsumer(
      new CompletionItemsCollector(&Items, FrontendOpts.CodeCompleteOpts));

  SyntaxOnlyAction Action;
  if (!Action.BeginSourceFile(*Clang, Clang->getFrontendOpts().Inputs[0])) {
    // FIXME(ibiryukov): log errors
    return Items;
  }
  if (!Action.Execute()) {
    // FIXME(ibiryukov): log errors
  }
  Action.EndSourceFile();

  return Items;
}

std::vector<DiagWithFixIts> ClangdUnit::getLocalDiagnostics() const {
  if (!Unit)
    return {}; // Parsing failed.

  std::vector<DiagWithFixIts> Result;
  auto PreambleDiagsSize = Preamble ? Preamble->Diags.size() : 0;
  const auto &Diags = Unit->getDiagnostics();
  Result.reserve(PreambleDiagsSize + Diags.size());

  if (Preamble)
    Result.insert(Result.end(), Preamble->Diags.begin(), Preamble->Diags.end());
  Result.insert(Result.end(), Diags.begin(), Diags.end());
  return Result;
}

void ClangdUnit::dumpAST(llvm::raw_ostream &OS) const {
  if (!Unit) {
    OS << "<no-ast-in-clang>";
    return; // Parsing failed.
  }
  Unit->getASTContext().getTranslationUnitDecl()->dump(OS, true);
}

llvm::Optional<ClangdUnit::ParsedAST>
ClangdUnit::ParsedAST::Build(std::unique_ptr<clang::CompilerInvocation> CI,
                             const PrecompiledPreamble *Preamble,
                             ArrayRef<serialization::DeclID> PreambleDeclIDs,
                             std::unique_ptr<llvm::MemoryBuffer> Buffer,
                             std::shared_ptr<PCHContainerOperations> PCHs,
                             IntrusiveRefCntPtr<vfs::FileSystem> VFS) {

  std::vector<DiagWithFixIts> ASTDiags;
  StoreDiagsConsumer UnitDiagsConsumer(/*ref*/ ASTDiags);

  auto Clang =
      prepareCompilerInstance(std::move(CI), Preamble, std::move(Buffer), PCHs,
                              VFS, /*ref*/ UnitDiagsConsumer);

  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<CompilerInstance> CICleanup(
      Clang.get());

  auto Action = llvm::make_unique<ClangdFrontendAction>();
  if (!Action->BeginSourceFile(*Clang, Clang->getFrontendOpts().Inputs[0])) {
    // FIXME(ibiryukov): log error
    return llvm::None;
  }
  if (!Action->Execute()) {
    // FIXME(ibiryukov): log error
  }

  // UnitDiagsConsumer is local, we can not store it in CompilerInstance that
  // has a longer lifetime.
  Clang->getDiagnostics().setClient(new EmptyDiagsConsumer);

  std::vector<const Decl *> ParsedDecls = Action->takeTopLevelDecls();
  std::vector<serialization::DeclID> PendingDecls;
  if (Preamble) {
    PendingDecls.reserve(PreambleDeclIDs.size());
    PendingDecls.insert(PendingDecls.begin(), PreambleDeclIDs.begin(),
                        PreambleDeclIDs.end());
  }

  return ParsedAST(std::move(Clang), std::move(Action), std::move(ParsedDecls),
                   std::move(PendingDecls), std::move(ASTDiags));
}

namespace {

SourceLocation getMacroArgExpandedLocation(const SourceManager &Mgr,
                                           const FileEntry *FE,
                                           unsigned Offset) {
  SourceLocation FileLoc = Mgr.translateFileLineCol(FE, 1, 1);
  return Mgr.getMacroArgExpandedLocation(FileLoc.getLocWithOffset(Offset));
}

SourceLocation getMacroArgExpandedLocation(const SourceManager &Mgr,
                                           const FileEntry *FE, Position Pos) {
  SourceLocation InputLoc =
      Mgr.translateFileLineCol(FE, Pos.line + 1, Pos.character + 1);
  return Mgr.getMacroArgExpandedLocation(InputLoc);
}

/// Finds declarations locations that a given source location refers to.
class DeclarationLocationsFinder : public index::IndexDataConsumer {
  std::vector<Location> DeclarationLocations;
  const SourceLocation &SearchedLocation;
  const ASTContext &AST;
  Preprocessor &PP;

public:
  DeclarationLocationsFinder(raw_ostream &OS,
                             const SourceLocation &SearchedLocation,
                             ASTContext &AST, Preprocessor &PP)
      : SearchedLocation(SearchedLocation), AST(AST), PP(PP) {}

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
    const SourceManager &SourceMgr = AST.getSourceManager();
    return SourceMgr.getFileOffset(SearchedLocation) == Offset &&
           SourceMgr.getFileID(SearchedLocation) == FID;
  }

  void addDeclarationLocation(const SourceRange &ValSourceRange) {
    const SourceManager &SourceMgr = AST.getSourceManager();
    const LangOptions &LangOpts = AST.getLangOpts();
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
    if (!Lexer::getRawToken(SearchedLocation, Result, AST.getSourceManager(),
                            AST.getLangOpts(), false)) {
      if (Result.is(tok::raw_identifier)) {
        PP.LookUpIdentifierInfo(Result);
      }
      IdentifierInfo *IdentifierInfo = Result.getIdentifierInfo();
      if (IdentifierInfo && IdentifierInfo->hadMacroDefinition()) {
        std::pair<FileID, unsigned int> DecLoc =
            AST.getSourceManager().getDecomposedExpansionLoc(SearchedLocation);
        // Get the definition just before the searched location so that a macro
        // referenced in a '#undef MACRO' can still be found.
        SourceLocation BeforeSearchedLocation = getMacroArgExpandedLocation(
            AST.getSourceManager(),
            AST.getSourceManager().getFileEntryForID(DecLoc.first),
            DecLoc.second - 1);
        MacroDefinition MacroDef =
            PP.getMacroDefinitionAtLoc(IdentifierInfo, BeforeSearchedLocation);
        MacroInfo *MacroInf = MacroDef.getMacroInfo();
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
  if (!Unit)
    return {}; // Parsing failed.

  const SourceManager &SourceMgr = Unit->getASTContext().getSourceManager();
  const FileEntry *FE = SourceMgr.getFileEntryForID(SourceMgr.getMainFileID());
  if (!FE)
    return {};

  SourceLocation SourceLocationBeg = getBeginningOfIdentifier(Pos, FE);

  auto DeclLocationsFinder = std::make_shared<DeclarationLocationsFinder>(
      llvm::errs(), SourceLocationBeg, Unit->getASTContext(),
      Unit->getPreprocessor());
  index::IndexingOptions IndexOpts;
  IndexOpts.SystemSymbolFilter =
      index::IndexingOptions::SystemSymbolFilterKind::All;
  IndexOpts.IndexFunctionLocals = true;

  indexTopLevelDecls(Unit->getASTContext(), Unit->getTopLevelDecls(),
                     DeclLocationsFinder, IndexOpts);

  return DeclLocationsFinder->takeLocations();
}

SourceLocation ClangdUnit::getBeginningOfIdentifier(const Position &Pos,
                                                    const FileEntry *FE) const {

  // The language server protocol uses zero-based line and column numbers.
  // Clang uses one-based numbers.

  const ASTContext &AST = Unit->getASTContext();
  const SourceManager &SourceMgr = AST.getSourceManager();

  SourceLocation InputLocation =
      getMacroArgExpandedLocation(SourceMgr, FE, Pos);
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
  SourceLocation PeekBeforeLocation = getMacroArgExpandedLocation(
      SourceMgr, FE, Position{Pos.line, Pos.character - 1});
  Token Result;
  if (Lexer::getRawToken(PeekBeforeLocation, Result, SourceMgr,
                         AST.getLangOpts(), false)) {
    // getRawToken failed, just use InputLocation.
    return InputLocation;
  }

  if (Result.is(tok::raw_identifier)) {
    return Lexer::GetBeginningOfToken(PeekBeforeLocation, SourceMgr,
                                      Unit->getASTContext().getLangOpts());
  }

  return InputLocation;
}

void ClangdUnit::ParsedAST::ensurePreambleDeclsDeserialized() {
  if (PendingTopLevelDecls.empty())
    return;

  std::vector<const Decl *> Resolved;
  Resolved.reserve(PendingTopLevelDecls.size());

  ExternalASTSource &Source = *getASTContext().getExternalSource();
  for (serialization::DeclID TopLevelDecl : PendingTopLevelDecls) {
    // Resolve the declaration ID to an actual declaration, possibly
    // deserializing the declaration in the process.
    if (Decl *D = Source.GetExternalDecl(TopLevelDecl))
      Resolved.push_back(D);
  }

  TopLevelDecls.reserve(TopLevelDecls.size() + PendingTopLevelDecls.size());
  TopLevelDecls.insert(TopLevelDecls.begin(), Resolved.begin(), Resolved.end());

  PendingTopLevelDecls.clear();
}

ClangdUnit::ParsedAST::ParsedAST(ParsedAST &&Other) = default;

ClangdUnit::ParsedAST &ClangdUnit::ParsedAST::
operator=(ParsedAST &&Other) = default;

ClangdUnit::ParsedAST::~ParsedAST() {
  if (Action) {
    Action->EndSourceFile();
  }
}

ASTContext &ClangdUnit::ParsedAST::getASTContext() {
  return Clang->getASTContext();
}

const ASTContext &ClangdUnit::ParsedAST::getASTContext() const {
  return Clang->getASTContext();
}

Preprocessor &ClangdUnit::ParsedAST::getPreprocessor() {
  return Clang->getPreprocessor();
}

const Preprocessor &ClangdUnit::ParsedAST::getPreprocessor() const {
  return Clang->getPreprocessor();
}

ArrayRef<const Decl *> ClangdUnit::ParsedAST::getTopLevelDecls() {
  ensurePreambleDeclsDeserialized();
  return TopLevelDecls;
}

const std::vector<DiagWithFixIts> &
ClangdUnit::ParsedAST::getDiagnostics() const {
  return Diags;
}

ClangdUnit::ParsedAST::ParsedAST(
    std::unique_ptr<CompilerInstance> Clang,
    std::unique_ptr<FrontendAction> Action,
    std::vector<const Decl *> TopLevelDecls,
    std::vector<serialization::DeclID> PendingTopLevelDecls,
    std::vector<DiagWithFixIts> Diags)
    : Clang(std::move(Clang)), Action(std::move(Action)),
      Diags(std::move(Diags)), TopLevelDecls(std::move(TopLevelDecls)),
      PendingTopLevelDecls(std::move(PendingTopLevelDecls)) {
  assert(this->Clang);
  assert(this->Action);
}

ClangdUnit::PreambleData::PreambleData(
    PrecompiledPreamble Preamble,
    std::vector<serialization::DeclID> TopLevelDeclIDs,
    std::vector<DiagWithFixIts> Diags)
    : Preamble(std::move(Preamble)),
      TopLevelDeclIDs(std::move(TopLevelDeclIDs)), Diags(std::move(Diags)) {}
