#include "FunctionInfo.h"
#include "GenICFG.h"
#include "ICFG.h"
#include "PathFinder.h"
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
    const int fid, bid;

    VarLocResult() : fid(-1), bid(-1) {}
    VarLocResult(const FunctionInfo *fi, const CFGBlock *block)
        : fid(Global.getIdOfFunction(fi->signature)), bid(block->getBlockID()) {
    }

    bool isValid() const { return fid != -1; }
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
                llvm::errs() << "Found var '" << var << "' in " << fi->signature
                             << " at " << line << ":" << column << " in block "
                             << id << "\n";
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

void dumpIcfgNode(int u) {
    auto [fid, bid] = Global.icfg.functionBlockOfNodeId[u];
    requireTrue(fid != -1);

    const NamedLocation &loc = *Global.functionLocations[fid];

    llvm::errs() << ">> Node " << u << " is in " << loc.name << " at block "
                 << bid << "\n";

    ClangTool Tool(*Global.cb, {loc.file});
    DiagnosticConsumer DC = IgnoringDiagConsumer();
    Tool.setDiagnosticConsumer(&DC);

    std::vector<std::unique_ptr<ASTUnit>> ASTs;
    Tool.buildASTs(ASTs);

    requireTrue(ASTs.size() == 1);
    std::unique_ptr<ASTUnit> AST = std::move(ASTs[0]);

    fif functionsInFile;
    ASTContext &Context = AST->getASTContext();
    auto *TUD = Context.getTranslationUnitDecl();
    requireTrue(!TUD->isUnavailable());
    FunctionAccumulator(functionsInFile).TraverseDecl(TUD);

    for (const FunctionInfo *fi : functionsInFile.at(loc.file)) {
        if (fi->signature != Global.functionLocations[fid]->name)
            continue;
        for (auto BI = fi->cfg->begin(); BI != fi->cfg->end(); ++BI) {
            const CFGBlock &B = **BI;
            if (B.getBlockID() != bid)
                continue;

            // B.dump(fi->cfg, Context.getLangOpts(), true);

            std::vector<const Stmt *> allStmts;
            std::set<const Stmt *> isChild;

            // iterate over all elements to find stmts & record children
            for (auto EI = B.begin(); EI != B.end(); ++EI) {
                const CFGElement &E = *EI;
                if (std::optional<CFGStmt> CS = E.getAs<CFGStmt>()) {
                    const Stmt *S = CS->getStmt();
                    allStmts.push_back(S);

                    // iterate over childern
                    for (const Stmt *child : S->children()) {
                        if (child != nullptr)
                            isChild.insert(child);
                    }
                }
            }

            // print all non-child stmts
            for (const Stmt *S : allStmts) {
                if (isChild.find(S) != isChild.end())
                    continue;
                // S is not child of any stmt in this CFGBlock
                auto bLoc =
                    Location::fromSourceLocation(Context, S->getBeginLoc());
                auto eLoc =
                    Location::fromSourceLocation(Context, S->getEndLoc());
                llvm::errs()
                    << "  Stmt " << bLoc->line << ":" << bLoc->column << " "
                    << eLoc->line << ":" << eLoc->column << "\n";
                // S->dumpColor();
            }

            return;
        }
    }
}

void findPathBetween(const VarLocResult &from, const VarLocResult &to) {
    if (!from.isValid() || !to.isValid()) {
        llvm::errs() << "Invalid variable location!\n";
        return;
    }

    ICFG &icfg = Global.icfg;
    int u = icfg.getNodeId(from.fid, from.bid);
    int v = icfg.getNodeId(to.fid, to.bid);

    llvm::errs() << "u: " << u << ", v: " << v << "\n";

    ICFGPathFinder pFinder(icfg);
    pFinder.search(u, v, 3);

    for (const auto &path : pFinder.results) {
        llvm::errs() << "> p:";
        for (int x : path) {
            llvm::errs() << " " << x;
        }
        llvm::errs() << "\n";
        for (int x : path)
            dumpIcfgNode(x);
        llvm::errs() << "\n";
    }
}

/**
 * 生成全程序调用图
 */
void generateICFG(const std::vector<std::string> &allFiles) {
    llvm::errs() << "\n--- Generating whole program call graph ---\n";
    ClangTool Tool(*Global.cb, allFiles);
    DiagnosticConsumer DC = IgnoringDiagConsumer();
    Tool.setDiagnosticConsumer(&DC);

    Tool.run(newFrontendActionFactory<GenICFGAction>().get());
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

    generateICFG(allFiles);
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

    findPathBetween(locateVariable("main()", 23, 9),
                    locateVariable("useAlias(const A &)", 19, 12));

    // std::string source = "IOPriorityPanel_new(IOPriority)";
    // std::string target = "Vector_new(const ObjectClass *, _Bool, int)";

    // findPathBetween(locateVariable(source, 23, 11),
    //                 locateVariable(target, 31, 10));

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
}