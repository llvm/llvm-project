#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Analysis/CallGraph.h"
#include "llvm/Support/CommandLine.h"
#include "clang/AST/ParentMapContext.h"

using namespace clang;
using namespace clang::tooling;
using namespace llvm;

// Apply a custom category to all command-line options so that they are the
// only ones displayed.
static cl::OptionCategory MyToolCategory("my-tool options");

// CommonOptionsParser declares HelpMessage with a description of the common
// command-line options related to the compilation database and input files.
// It's nice to have this help message in all tools.
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

// A help message for this specific tool can be added afterwards.
static cl::extrahelp MoreHelp("\nMore help text...\n");

// class tbtASTDumpAction : public ASTFrontendAction {
// protected:
//   std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
//                                                  StringRef InFile) override;
// public:
//   void ExecuteAction() override {
//     llvm::errs() << "tbtASTDumpAction::ExecuteAction\n";
//     llvm::errs() << "  " << getCurrentFile() << "\n";
//     // get declarations in ast
//     ASTContext &Context = getCompilerInstance().getASTContext();
//     TranslationUnitDecl *TUD = Context.getTranslationUnitDecl();
//     for (Decl *D : TUD->decls()) {
//       llvm::errs() << "  " << D->getDeclKindName() << "\n";
//     }
//   }
// };

// std::unique_ptr<ASTConsumer>
// tbtASTDumpAction::CreateASTConsumer(CompilerInstance &CI, StringRef InFile) {
//   const FrontendOptions &Opts = CI.getFrontendOpts();
//   return CreateASTDumper(nullptr /*Dump to stdout.*/, Opts.ASTDumpFilter,
//                          Opts.ASTDumpDecls, Opts.ASTDumpAll,
//                          Opts.ASTDumpLookups, Opts.ASTDumpDeclTypes,
//                          Opts.ASTDumpFormat);
// }

void requireTrue(bool condition, std::string message) {
  if (!condition) {
    llvm::errs() << "requireTrue failed: " << message << "\n";
    exit(1);
  }
}

class HasContextVisitor {
protected:
  ASTContext *Context;

  std::string getLocation(const SourceLocation &loc) {
    PresumedLoc PLoc = Context->getSourceManager().getPresumedLoc(loc);
    if (PLoc.isInvalid())
      return "<invalid>";

    std::string filename = PLoc.getFilename();
    std::string line = std::to_string(PLoc.getLine());
    std::string column = std::to_string(PLoc.getColumn());
    return filename + ":" + line + ":" + column;
  }

  void printStmtLocation(const Stmt &s) {
    llvm::errs() << "    beg: " << getLocation(s.getBeginLoc()) << "\n";
    llvm::errs() << "    end: " << getLocation(s.getEndLoc()) << "\n";
  }

public:
  explicit HasContextVisitor(ASTContext *Context) : Context(Context) {}
};

/**
 * Visit all DeclRefExprs and print their parents.
 */
class VarVisitor : public RecursiveASTVisitor<VarVisitor>, public HasContextVisitor {
public:
  explicit VarVisitor(ASTContext *Context) : HasContextVisitor(Context) {
  }

  void visitParents(const Stmt &base) {
    const Stmt *s = &base;
    llvm::errs() << "    parents:\n";
    while (true) {
      const auto &parents = Context->getParents(*s);
      requireTrue(parents.size() == 1, "parent size is not 1");

      const Stmt *parent = parents.begin()->get<Stmt>();
      requireTrue(parent != nullptr, "parent is null");

      llvm::errs() << "      " << parent->getStmtClassName() << "\n";
      if (isa<CompoundStmt>(parent)) {
        break;
      }

      s = parent;
    }
  }

  bool VisitStmt(Stmt *s) {
    // DeclRefExpr
    if (DeclRefExpr *dre = dyn_cast<DeclRefExpr>(s)) {
      llvm::errs() << "  DeclRefExpr: " << dre->getDecl()->getQualifiedNameAsString() << "\n";
      printStmtLocation(*s);
      visitParents(*s);
    }
    return true;
  }
};

/**
 * Visit all FunctionDecls and print their CFGs.
 */
class FunctionDeclVisitor : public RecursiveASTVisitor<FunctionDeclVisitor>, public HasContextVisitor {
public:
  explicit FunctionDeclVisitor(ASTContext *Context)
    : HasContextVisitor(Context) {}

  bool VisitFunctionDecl(FunctionDecl *D) {
    FullSourceLoc FullLocation = Context->getFullLoc(D->getBeginLoc());
    requireTrue(FullLocation.hasManager(), "no source manager!");
    if (FullLocation.isInvalid())
      return true;

    llvm::errs() << "------ FunctionDecl: " << D->getQualifiedNameAsString() << " at "
                  << FullLocation.getSpellingLineNumber() << ":"
                  << FullLocation.getSpellingColumnNumber() << "\n";

    if (!D->hasBody())
      return true;

    // show call graph
    // TranslationUnitDecl *TUD = Context->getTranslationUnitDecl();
    // CallGraph CG;
    // CG.addToCallGraph(TUD);
    // CG.viewGraph();

    llvm::errs() << "--------- CFG dump: " << D->getQualifiedNameAsString() << "\n";
    // build CFG
    auto cfg = CFG::buildCFG(D, D->getBody() , &D->getASTContext(), CFG::BuildOptions());
    cfg->dump(D->getASTContext().getLangOpts(), true);
    // cfg->viewCFG(D->getASTContext().getLangOpts());

    // traverse each block
    llvm::errs() << "--------- Block traversal: " << D->getQualifiedNameAsString() << "\n";
    for (auto BI = cfg->begin(); BI != cfg->end(); ++BI) {
      const CFGBlock &B = **BI;
      // print block ID
      llvm::errs() << "Block " << B.getBlockID();
      if (&B == &cfg->getEntry()) {
        llvm::errs() << " (Entry)";
      } else if (&B == &cfg->getExit()) {
        llvm::errs() << " (Exit)";
      }
      llvm::errs() << ":\n";

      // traverse & print block contents
      for (auto EI = B.begin(); EI != B.end(); ++EI) {
        const CFGElement &E = *EI;
        llvm::errs() << "  ";
        E.dump();
        if (std::optional<CFGStmt> CS = E.getAs<CFGStmt>()) {
          // CS->getStmt()->dump();
          // printStmtLocation(*CS->getStmt());
        }
      }

      // print block terminator
      if (B.getTerminator().isValid()) {
        const CFGTerminator &T = B.getTerminator();
        if (T.getStmt()) {
          const Stmt &S = *T.getStmt();
          llvm::errs() << "  T: <" << S.getStmtClassName() << ">\n";
          // printStmtLocation(S);
        }
      }

      // print predecessors
      llvm::errs() << "  Preds:";
      for (auto PI = B.pred_begin(); PI != B.pred_end(); ++PI) {
        const CFGBlock *Pred = *PI;
        llvm::errs() << " B" << Pred->getBlockID();
      }
      llvm::errs() << "\n";

      // print successors
      llvm::errs() << "  Succs:";
      for (auto SI = B.succ_begin(); SI != B.succ_end(); ++SI) {
        const CFGBlock *Succ = *SI;
        llvm::errs() << " B" << Succ->getBlockID();
      }
      llvm::errs() << "\n";
    }

    return true;
  }
};

class FindNamedClassConsumer : public clang::ASTConsumer {
public:
  explicit FindNamedClassConsumer(ASTContext *Context) {}

  virtual void HandleTranslationUnit(clang::ASTContext &Context) {
    auto *TUD = Context.getTranslationUnitDecl();
    llvm::errs() << "\n--- TranslationUnitDecl Dump ---\n";
    TUD->dump();

    // call different visitors
    llvm::errs() << "\n--- FunctionDeclVisitor ---\n";
    FunctionDeclVisitor(&Context).TraverseDecl(TUD);

    llvm::errs() << "\n--- VarVisitor ---\n";
    VarVisitor(&Context).TraverseDecl(TUD);
  }
};

class FindNamedClassAction : public clang::ASTFrontendAction {
public:
  virtual std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
    clang::CompilerInstance &Compiler, llvm::StringRef InFile) {
    return std::make_unique<FindNamedClassConsumer>(&Compiler.getASTContext());
  }
};

std::unique_ptr<CompilationDatabase> getCompilationDatabase(std::string buildPath) {
  llvm::errs() << "Getting compilation database from: " << buildPath << "\n";
  std::string errorMsg;
  std::unique_ptr<CompilationDatabase> cb =
    CompilationDatabase::autoDetectFromDirectory(buildPath, errorMsg);
  if (!cb) {
    llvm::errs() << "Error while trying to load a compilation database:\n"
                 << errorMsg << "Running without flags.\n";
    exit(1);
  }
  return cb;
}

int main(int argc, const char **argv) {
  std::string buildPath = argv[1];
  std::unique_ptr<CompilationDatabase> cb = getCompilationDatabase(buildPath);

  const auto &allFiles = cb->getAllFiles();

  llvm::errs() << "All files:\n";
  for (auto &file : allFiles) {
    llvm::errs() << "  " << file << "\n";
  }

  ClangTool Tool(*cb, allFiles);
  return Tool.run(newFrontendActionFactory<FindNamedClassAction>().get());
}