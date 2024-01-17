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


class FindNamedClassVisitor
  : public RecursiveASTVisitor<FindNamedClassVisitor> {
public:
  explicit FindNamedClassVisitor(ASTContext *Context)
    : Context(Context) {}

  bool VisitFunctionDecl(FunctionDecl *D) {
      FullSourceLoc FullLocation = Context->getFullLoc(D->getBeginLoc());
      if (FullLocation.isValid())
        llvm::outs() << "Found declaration " << D->getQualifiedNameAsString() << " at "
                     << FullLocation.getSpellingLineNumber() << ":"
                     << FullLocation.getSpellingColumnNumber() << "\n";
      D->dump();
      if (D->hasBody()) {
        TranslationUnitDecl *TUD = Context->getTranslationUnitDecl();

        if (D->getQualifiedNameAsString().find("globalF") != std::string::npos) {
          CallGraph CG;
          CG.addToCallGraph(TUD);
          CG.viewGraph();
        }

        CFG::BuildOptions cfgBuildOptions;
        auto cfg = CFG::buildCFG(D, D->getBody() , &D->getASTContext(), cfgBuildOptions);
        cfg->dump(D->getASTContext().getLangOpts(), true);
      }

    return true;
  }

  bool VisitCXXRecordDecl(CXXRecordDecl *D) {
    FullSourceLoc FullLocation = Context->getFullLoc(D->getBeginLoc());
    if (FullLocation.isValid())
      llvm::outs() << "Class decl " << D->getQualifiedNameAsString() << " at "
                    << FullLocation.getSpellingLineNumber() << ":"
                    << FullLocation.getSpellingColumnNumber() << "\n";
    return true;
  }

private:
  ASTContext *Context;
};

class FindNamedClassConsumer : public clang::ASTConsumer {
public:
  explicit FindNamedClassConsumer(ASTContext *Context)
    : Visitor(Context) {}

  virtual void HandleTranslationUnit(clang::ASTContext &Context) {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }
private:
  FindNamedClassVisitor Visitor;
};

class FindNamedClassAction : public clang::ASTFrontendAction {
public:
  virtual std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
    clang::CompilerInstance &Compiler, llvm::StringRef InFile) {
    llvm::outs() << "CreateASTConsumer\n";
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
    llvm::outs() << "  " << file << "\n";
  }

  ClangTool Tool(*cb, allFiles);
  return Tool.run(newFrontendActionFactory<FindNamedClassAction>().get());
}