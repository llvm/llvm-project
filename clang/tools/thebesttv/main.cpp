#include "FunctionInfo.h"
#include "VarFinder.h"
#include "utils.h"

#include "clang/Tooling/CompilationDatabase.h"

class FunctionAccumulator : public RecursiveASTVisitor<FunctionAccumulator> {
  private:
    fif &functionsInFile;

  public:
    explicit FunctionAccumulator(fif &functionsInFile)
        : functionsInFile(functionsInFile) {}

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

struct VarLocResult {
    const FunctionInfo *fi;
    const CFGBlock *block;
    const int id;

    VarLocResult() : fi(nullptr), block(nullptr), id(-1) {}
    VarLocResult(const FunctionInfo *fi, const CFGBlock *block)
        : fi(fi), block(block), id(block->getBlockID()) {}

    bool isValid() const { return fi != nullptr; }
};

VarLocResult locateVariable(const fif &functionsInFile, const std::string &file,
                            int line, int column) {
    FindVarVisitor visitor;

    for (const FunctionInfo *fi : functionsInFile.at(file)) {
        // function is defined later than targetLoc
        if (fi->line > line)
            continue;

        ASTContext *Context = &fi->D->getASTContext();
        for (const auto &[stmt, block] : fi->stmtBlockPairs) {
            const std::string var =
                visitor.findVarInStmt(Context, stmt, file, line, column);
            if (!var.empty()) {
                int id = block->getBlockID();
                llvm::errs()
                    << "Found var '" << var << "' in " << fi->name << "() at "
                    << line << ":" << column << " in block " << id << "\n";
                return VarLocResult(fi, block);
            }
        }
    }
    return VarLocResult();
}

void findPathBetween(const fif &functionsInFile, //
                     const std::string &fileFrom, int lineFrom, int columnFrom,
                     const std::string &fileTo, int lineTo, int columnTo) {
    VarLocResult from =
        locateVariable(functionsInFile, fileFrom, lineFrom, columnFrom);
    VarLocResult to = locateVariable(functionsInFile, fileTo, lineTo, columnTo);

    if (!from.isValid() || !to.isValid()) {
        llvm::errs() << "Invalid variable location!\n";
        return;
    }

    requireTrue(from.fi == to.fi, "different functions!");

    const FunctionInfo *fi = from.fi;
    int u = from.id;
    int v = to.id;

    fi->bg->dij(from.block);
    llvm::errs() << "Dis from " << u << " to " << v << ": " << fi->bg->g.d[v]
                 << "\n  path:";
    for (int x : fi->bg->g.trace(v)) {
        llvm::errs() << " " << x;
    }
    llvm::errs() << "\n";

    fi->bg->dij(to.block);
    llvm::errs() << "Dis from " << v << " to " << u << ": " << fi->bg->g.d[u]
                 << "\n  path:";
    for (int x : fi->bg->g.trace(u)) {
        llvm::errs() << " " << x;
    }
    llvm::errs() << "\n";
}

int main(int argc, const char **argv) {
    BUILD_PATH = fs::canonical(fs::absolute(argv[1]));
    std::unique_ptr<CompilationDatabase> cb =
        getCompilationDatabase(BUILD_PATH);

    const auto &allFiles = cb->getAllFiles();
    llvm::errs() << "All files:\n";
    for (auto &file : allFiles)
        llvm::errs() << "  " << file << "\n";

    ClangTool Tool(*cb, allFiles);
    std::vector<std::unique_ptr<ASTUnit>> ASTs;
    Tool.buildASTs(ASTs);

    llvm::errs() << "\n--- TranslationUnitDecl Dump ---\n";
    fif functionsInFile;
    for (auto &AST : ASTs) {
        ASTContext &Context = AST->getASTContext();
        auto *TUD = Context.getTranslationUnitDecl();
        TUD->dump();
        FunctionAccumulator(functionsInFile).TraverseDecl(TUD);
    }

    llvm::errs() << "\n--- All functions from all files ---\n";
    // traverse functionInFile
    for (const auto &[file, functions] : functionsInFile) {
        llvm::errs() << "File: " << file << "\n";
        for (const auto *fi : functions) {
            llvm::errs() << "  Fun: " << fi->name << "() at " << fi->line << ":"
                         << fi->column << "\n";
        }
    }

    llvm::errs() << "\n--- FindVarVisitor ---\n";
    std::string root_dir = "/home/thebesttv/vul/llvm-project/graph-generation/";
    std::string file1 = root_dir + "test4.cpp";
    std::string file2 = root_dir + "test5.cpp";

    locateVariable(functionsInFile, file1, 2, 7);
    locateVariable(functionsInFile, file1, 2, 14);
    locateVariable(functionsInFile, file1, 15, 10);

    locateVariable(functionsInFile, file2, 5, 7);
    locateVariable(functionsInFile, file2, 5, 11);
    locateVariable(functionsInFile, file2, 5, 15);

    findPathBetween(functionsInFile, file1, 3, 19, file1, 15, 10);
}