#include "FunctionInfo.h"
#include "VarFinder.h"
#include "utils.h"

#include "clang/Tooling/CompilationDatabase.h"

class FunctionAccumulator : public RecursiveASTVisitor<FunctionAccumulator> {
  public:
    bool VisitFunctionDecl(FunctionDecl *D) {

        FunctionInfo *fi = FunctionInfo::fromDecl(D);
        if (fi == nullptr)
            return true;

        functionsInFile[fi->file].insert(fi);

        return true;
    }
};

std::unique_ptr<CompilationDatabase>
getCompilationDatabase(fs::path buildPath) {
    llvm::errs() << "Getting compilation database from: " << buildPath << "\n";
    std::string errorMsg;
    std::unique_ptr<CompilationDatabase> cb =
        CompilationDatabase::autoDetectFromDirectory(buildPath.string(),
                                                     errorMsg);
    if (!cb) {
        llvm::errs() << "Error while trying to load a compilation database:\n"
                     << errorMsg << "Running without flags.\n";
        exit(1);
    }
    return cb;
}

int main(int argc, const char **argv) {
    BUILD_PATH = fs::canonical(fs::absolute(argv[1]));
    std::unique_ptr<CompilationDatabase> cb =
        getCompilationDatabase(BUILD_PATH);

    const auto &allFiles = cb->getAllFiles();

    llvm::errs() << "All files:\n";
    for (auto &file : allFiles) {
        llvm::errs() << "  " << file << "\n";
    }

    ClangTool Tool(*cb, allFiles);
    std::vector<std::unique_ptr<ASTUnit>> ASTs;
    Tool.buildASTs(ASTs);

    for (auto &AST : ASTs) {
        ASTContext &Context = AST->getASTContext();

        auto *TUD = Context.getTranslationUnitDecl();
        llvm::errs() << "\n--- TranslationUnitDecl Dump ---\n";
        TUD->dump();

        llvm::errs() << "\n--- FunctionAccumulator ---\n";
        FunctionAccumulator fpv;
        fpv.TraverseDecl(TUD);
    }

    // traverse functionInFile
    for (const auto &[file, functions] : functionsInFile) {
        llvm::errs() << "File: " << file << "\n";
        for (const auto *fi : functions) {
            llvm::errs() << "  " << fi->name << "\n";
        }
    }

    llvm::errs() << "\n--- FindVarVisitor ---\n";
    VarLocation targetLoc(
        "/home/thebesttv/vul/llvm-project/graph-generation/test4.cpp", 2, 7);
    FindVarVisitor::findVar(functionsInFile, targetLoc);

    targetLoc.line = 2;
    targetLoc.column = 14;
    FindVarVisitor::findVar(functionsInFile, targetLoc);

    targetLoc.line = 23;
    targetLoc.column = 7;
    FindVarVisitor::findVar(functionsInFile, targetLoc);

    targetLoc.line = 23;
    targetLoc.column = 11;
    FindVarVisitor::findVar(functionsInFile, targetLoc);

    targetLoc.line = 23;
    targetLoc.column = 15;
    FindVarVisitor::findVar(functionsInFile, targetLoc);
}