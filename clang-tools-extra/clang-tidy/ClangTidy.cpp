//===--- tools/extra/clang-tidy/ClangTidy.cpp - Clang tidy tool -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
///  \file This file implements a clang-tidy tool.
///
///  This tool uses the Clang Tooling infrastructure, see
///    http://clang.llvm.org/docs/HowToSetupToolingForLLVM.html
///  for details on setting it up with LLVM source tree.
///
//===----------------------------------------------------------------------===//

#include "ClangTidy.h"
#include "ClangTidyDiagnosticConsumer.h"
#include "ClangTidyModuleRegistry.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Frontend/FixItRewriter.h"
#include "clang/Rewrite/Frontend/FrontendActions.h"
#include "clang/StaticAnalyzer/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Tooling/Refactoring.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include <vector>

using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm;

namespace clang {
namespace tidy {
namespace {

/// \brief A combined ASTConsumer that forwards calls to two different
/// consumers.
///
/// FIXME: This currently forwards just enough methods for the static analyzer
/// and the \c MatchFinder's consumer to work; expand this to all methods of
/// ASTConsumer and put it into a common location.
class CombiningASTConsumer : public ASTConsumer {
public:
  CombiningASTConsumer(ASTConsumer *Consumer1, ASTConsumer *Consumer2)
      : Consumer1(Consumer1), Consumer2(Consumer2) {}

  virtual void Initialize(ASTContext &Context) LLVM_OVERRIDE {
    Consumer1->Initialize(Context);
    Consumer2->Initialize(Context);
  }
  virtual bool HandleTopLevelDecl(DeclGroupRef D) LLVM_OVERRIDE {
    return Consumer1->HandleTopLevelDecl(D) && Consumer2->HandleTopLevelDecl(D);
  }
  virtual void HandleTopLevelDeclInObjCContainer(DeclGroupRef D) LLVM_OVERRIDE {
    Consumer1->HandleTopLevelDeclInObjCContainer(D);
    Consumer2->HandleTopLevelDeclInObjCContainer(D);
  }
  virtual void HandleTranslationUnit(ASTContext &Context) LLVM_OVERRIDE {
    Consumer1->HandleTranslationUnit(Context);
    Consumer2->HandleTranslationUnit(Context);
  }

private:
  llvm::OwningPtr<ASTConsumer> Consumer1;
  llvm::OwningPtr<ASTConsumer> Consumer2;
};

/// \brief Action that runs clang-tidy and static analyzer checks.
///
/// FIXME: Note that this inherits from \c AnalysisAction as this is the only
/// way we can currently get to AnalysisAction::CreateASTConsumer. Ideally
/// we'd want to build a more generic way to use \c FrontendAction based
/// checkers in clang-tidy, but that needs some preparation work first.
class ClangTidyAction : public ento::AnalysisAction {
public:
  ClangTidyAction(StringRef CheckRegexString,
                  SmallVectorImpl<ClangTidyCheck *> &Checks,
                  ClangTidyContext &Context, MatchFinder &Finder)
      : CheckRegexString(CheckRegexString), Checks(Checks), Context(Context),
        Finder(Finder) {}

private:
  clang::ASTConsumer *CreateASTConsumer(clang::CompilerInstance &Compiler,
                                        StringRef File) LLVM_OVERRIDE {
    AnalyzerOptionsRef Options = Compiler.getAnalyzerOpts();
    llvm::Regex CheckRegex(CheckRegexString);

    // Always add all core checkers if any other static analyzer checks are
    // enabled. This is currently necessary, as other path sensitive checks rely
    // on the core checkers.
    if (CheckRegex.match("clang-analyzer-"))
      Options->CheckersControlList.push_back(std::make_pair("core", true));

// Run our regex against all possible static analyzer checkers.
// Note that debug checkers print values / run programs to visualize the CFG
// and are thus not applicable to clang-tidy in general.
#define GET_CHECKERS
#define CHECKER(FULLNAME, CLASS, DESCFILE, HELPTEXT, GROUPINDEX, HIDDEN)       \
  if (!StringRef(FULLNAME).startswith("core") &&                               \
      !StringRef(FULLNAME).startswith("debug") &&                              \
      CheckRegex.match("clang-analyzer-" FULLNAME))                            \
    Options->CheckersControlList.push_back(std::make_pair(FULLNAME, true));
#include "../../../lib/StaticAnalyzer/Checkers/Checkers.inc"
#undef CHECKER
#undef GET_CHECKERS

    Options->AnalysisStoreOpt = RegionStoreModel;
    Options->AnalysisDiagOpt = PD_TEXT;
    Options->AnalyzeNestedBlocks = true;
    Options->eagerlyAssumeBinOpBifurcation = true;
    return new CombiningASTConsumer(
        Finder.newASTConsumer(),
        ento::AnalysisAction::CreateASTConsumer(Compiler, File));
  }

  virtual bool BeginSourceFileAction(CompilerInstance &Compiler,
                                     llvm::StringRef Filename) LLVM_OVERRIDE {
    if (!ento::AnalysisAction::BeginSourceFileAction(Compiler, Filename))
      return false;
    Context.setSourceManager(&Compiler.getSourceManager());
    for (SmallVectorImpl<ClangTidyCheck *>::iterator I = Checks.begin(),
                                                     E = Checks.end();
         I != E; ++I)
      (*I)->registerPPCallbacks(Compiler);
    return true;
  }

  std::string CheckRegexString;
  SmallVectorImpl<ClangTidyCheck *> &Checks;
  ClangTidyContext &Context;
  MatchFinder &Finder;
};

class ClangTidyActionFactory : public FrontendActionFactory {
public:
  ClangTidyActionFactory(StringRef CheckRegexString, ClangTidyContext &Context)
      : CheckRegexString(CheckRegexString), Context(Context) {
    ClangTidyCheckFactories CheckFactories;
    for (ClangTidyModuleRegistry::iterator I = ClangTidyModuleRegistry::begin(),
                                           E = ClangTidyModuleRegistry::end();
         I != E; ++I) {
      OwningPtr<ClangTidyModule> Module(I->instantiate());
      Module->addCheckFactories(CheckFactories);
    }

    SmallVector<ClangTidyCheck *, 16> Checks;
    CheckFactories.createChecks(CheckRegexString, Checks);

    for (SmallVectorImpl<ClangTidyCheck *>::iterator I = Checks.begin(),
                                                     E = Checks.end();
         I != E; ++I) {
      (*I)->setContext(&Context);
      (*I)->registerMatchers(&Finder);
    }
  }

  virtual FrontendAction *create() {
    return new ClangTidyAction(CheckRegexString, Checks, Context, Finder);
  }

private:
  std::string CheckRegexString;
  SmallVector<ClangTidyCheck *, 8> Checks;
  ClangTidyContext &Context;
  MatchFinder Finder;
};

} // namespace

ClangTidyMessage::ClangTidyMessage(StringRef Message) : Message(Message) {}

ClangTidyMessage::ClangTidyMessage(StringRef Message,
                                   const SourceManager &Sources,
                                   SourceLocation Loc)
    : Message(Message) {
  FilePath = Sources.getFilename(Loc);
  FileOffset = Sources.getFileOffset(Loc);
}

ClangTidyError::ClangTidyError(const ClangTidyMessage &Message)
    : Message(Message) {}

DiagnosticBuilder ClangTidyContext::Diag(SourceLocation Loc,
                                         StringRef Message) {
  return DiagEngine->Report(
      Loc, DiagEngine->getCustomDiagID(DiagnosticsEngine::Warning, Message));
}

void ClangTidyContext::setDiagnosticsEngine(DiagnosticsEngine *Engine) {
  DiagEngine = Engine;
}

void ClangTidyContext::setSourceManager(SourceManager *SourceMgr) {
  DiagEngine->setSourceManager(SourceMgr);
}

/// \brief Store a \c ClangTidyError.
void ClangTidyContext::storeError(const ClangTidyError &Error) {
  Errors->push_back(Error);
}

void ClangTidyCheck::run(const ast_matchers::MatchFinder::MatchResult &Result) {
  Context->setSourceManager(Result.SourceManager);
  check(Result);
}

FrontendActionFactory *createClangTidyActionFactory(StringRef CheckRegexString,
                                                    ClangTidyContext &Context) {
  return new ClangTidyActionFactory(CheckRegexString, Context);
}

void runClangTidy(StringRef CheckRegexString,
                  const tooling::CompilationDatabase &Compilations,
                  ArrayRef<std::string> Ranges,
                  SmallVectorImpl<ClangTidyError> *Errors) {
  // FIXME: Ranges are currently full files. Support selecting specific
  // (line-)ranges.
  ClangTool Tool(Compilations, Ranges);
  clang::tidy::ClangTidyContext Context(Errors);
  ClangTidyDiagnosticConsumer DiagConsumer(Context);

  Tool.setDiagnosticConsumer(&DiagConsumer);
  Tool.run(createClangTidyActionFactory(CheckRegexString, Context));
}

static void reportDiagnostic(const ClangTidyMessage &Message,
                             SourceManager &SourceMgr,
                             DiagnosticsEngine::Level Level,
                             DiagnosticsEngine &Diags) {
  SourceLocation Loc;
  if (!Message.FilePath.empty()) {
    const FileEntry *File =
        SourceMgr.getFileManager().getFile(Message.FilePath);
    FileID ID = SourceMgr.createFileID(File, SourceLocation(), SrcMgr::C_User);
    Loc = SourceMgr.getLocForStartOfFile(ID);
    Loc = Loc.getLocWithOffset(Message.FileOffset);
  }
  Diags.Report(Loc, Diags.getCustomDiagID(Level, Message.Message));
}

void handleErrors(SmallVectorImpl<ClangTidyError> &Errors, bool Fix) {
  FileManager Files((FileSystemOptions()));
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  DiagnosticConsumer *DiagPrinter =
      new TextDiagnosticPrinter(llvm::outs(), &*DiagOpts);
  DiagnosticsEngine Diags(IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs),
                          &*DiagOpts, DiagPrinter);
  DiagPrinter->BeginSourceFile(LangOptions());
  SourceManager SourceMgr(Diags, Files);
  Rewriter Rewrite(SourceMgr, LangOptions());
  for (SmallVectorImpl<ClangTidyError>::iterator I = Errors.begin(),
                                                 E = Errors.end();
       I != E; ++I) {
    reportDiagnostic(I->Message, SourceMgr, DiagnosticsEngine::Warning, Diags);
    for (unsigned i = 0, e = I->Notes.size(); i != e; ++i) {
      reportDiagnostic(I->Notes[i], SourceMgr, DiagnosticsEngine::Note, Diags);
    }
    tooling::applyAllReplacements(I->Fix, Rewrite);
  }
  // FIXME: Run clang-format on changes.
  if (Fix)
    Rewrite.overwriteChangedFiles();
}

} // namespace tidy
} // namespace clang
