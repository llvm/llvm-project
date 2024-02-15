#include "CallGraph.h"
#include "FunctionInfo.h"
#include "ICFG.h"
#include "VarFinder.h"
#include "utils.h"
#include <fstream>
#include <iostream>
#include <unistd.h>

#include "llvm/Support/Program.h"

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

        // search all CFG stmts in function for matching variable
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

VarLocResult locateVariable(const std::string &signature, int line,
                            int column) {

    int fid = Global.getIdOfFunction(signature);
    if (fid == -1) {
        llvm::errs() << "Function not found: " << signature << "\n";
        return VarLocResult();
    }

    const NamedLocation &loc = *Global.functionLocations[fid];

    ClangTool Tool(*Global.cb, {loc.file});
    DiagnosticConsumer DC = IgnoringDiagConsumer();
    Tool.setDiagnosticConsumer(&DC);

    std::vector<std::unique_ptr<ASTUnit>> ASTs;
    Tool.buildASTs(ASTs);

    fif functionsInFile;
    for (auto &AST : ASTs) {
        ASTContext &Context = AST->getASTContext();
        auto *TUD = Context.getTranslationUnitDecl();
        if (TUD->isUnavailable())
            continue;
        FunctionAccumulator(functionsInFile).TraverseDecl(TUD);
    }

    return locateVariable(functionsInFile, loc.file, line, column);
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

void printCloc(const std::vector<std::string> &allFiles) {
    // save all files to "compile_files.txt" under build path
    fs::path resultFiles = Global.buildPath / "compile_files.txt";
    std::ofstream ofs(resultFiles);
    if (!ofs.is_open()) {
        llvm::errs() << "Error: cannot open file " << resultFiles << "\n";
        exit(1);
    }
    for (auto &file : allFiles)
        ofs << file << "\n";
    ofs.close();

    // run cloc on all files
    if (ErrorOr<std::string> P = sys::findProgramByName("cloc")) {
        std::string programPath = *P;
        std::vector<StringRef> args;
        args.push_back("cloc");
        args.push_back("--list-file");
        args.push_back(resultFiles.c_str()); // don't use .string() here
        std::string errorMsg;
        if (sys::ExecuteAndWait(programPath, args, std::nullopt, {}, 0, 0,
                                &errorMsg)) {
            llvm::errs() << "Error: " << errorMsg << "\n";
        }
    }
}

int main(int argc, const char **argv) {
    if (argc != 2) {
        llvm::errs() << "Usage: " << argv[0] << " <build-path>\n";
        return 1;
    }

    Global.buildPath = fs::canonical(fs::absolute(argv[1]));
    Global.cb = getCompilationDatabase(Global.buildPath);

    // print all files in compilation database
    const auto &allFiles = Global.cb->getAllFiles();
    llvm::errs() << "All files (" << allFiles.size() << "):\n";
    for (auto &file : allFiles)
        llvm::errs() << "  " << file << "\n";

    llvm::errs() << "\n--- Building ATS from files ---\n";
    ClangTool Tool(*Global.cb, allFiles);
    DiagnosticConsumer DC = IgnoringDiagConsumer();
    Tool.setDiagnosticConsumer(&DC);

    // 生成所有函数的调用图
    Tool.run(newFrontendActionFactory<GenWholeProgramCallGraphAction>().get());

    printCloc(allFiles);

    {
        llvm::errs() << "--- ICFG ---\n";
        llvm::errs() << "  n: " << Global.icfg.n << "\n";
        int m = 0;
        for (const auto &edges : Global.icfg.G) {
            m += edges.size();
        }
        llvm::errs() << "  m: " << m << "\n";
    }

    std::string source = "IOPriorityPanel_new(IOPriority)";
    std::string target = "Vector_new(const ObjectClass *, _Bool, int)";

    locateVariable(source, 23, 11);
    locateVariable(target, 31, 10);

    while (true) {
        std::string methodName;
        llvm::errs() << "> ";
        std::getline(std::cin, methodName);
        if (!std::cin)
            break;
        if (methodName.find('(') == std::string::npos)
            methodName += "(";

        for (const auto &[caller, callees] : Global.callGraph) {
            if (caller.find(methodName) != 0)
                continue;
            llvm::errs() << caller << "\n";
            for (const auto &callee : callees) {
                llvm::errs() << "  " << callee << "\n";
            }
        }
    }

    return 0;

    /*
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
    */
}