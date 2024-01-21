#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Analysis/CallGraph.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"

#include <filesystem>
#include <queue>

using namespace clang;
using namespace clang::tooling;
using namespace llvm;
namespace fs = std::filesystem;

/*****************************************************************
 * Global Variables
 *****************************************************************/

fs::path BUILD_PATH;

struct FunctionInfo;
std::map<std::string, std::set<const FunctionInfo *>> functionsInFile;

/*****************************************************************
 * Main body
 *****************************************************************/

void requireTrue(bool condition, std::string message = "") {
    if (!condition) {
        llvm::errs() << "requireTrue failed: " << message << "\n";
        exit(1);
    }
}

struct Graph {
    int n; // 0-indexed
    std::vector<std::vector<int>> G;
    std::vector<int> d;

    const int INF = 0x3f3f3f3f;

    struct Node {
        int u, d;
        bool operator<(const Node &b) const { return d > b.d; }
    };

    Graph(int n) : n(n) {
        G.resize(n);
        d.resize(n);
    }

    void addEdge(int u, int v) { G[u].push_back(v); }

    void dij(int s) {
        std::fill(d.begin(), d.end(), INF);
        d[s] = 0;

        std::priority_queue<Node> q;
        q.push({s, 0});

        while (!q.empty()) {
            Node p = q.top();
            q.pop();
            int u = p.u;
            if (p.d != d[u])
                continue;
            for (int v : G[u]) {
                if (d[v] > d[u] + 1) {
                    d[v] = d[u] + 1;
                    q.push({v, d[v]});
                }
            }
        }
    }
};

struct BlockGraph {
    const ASTContext &Context;
    const CFG *cfg;
    Graph g;

    std::map<const Stmt *, int> blockIdOfStmt;

    BlockGraph(const ASTContext &Context, const CFG *cfg)
        : Context(Context), cfg(cfg), g(cfg->size()) {
        for (auto BI = cfg->begin(); BI != cfg->end(); ++BI) {
            const CFGBlock &B = **BI;

            llvm::errs() << "Block " << B.getBlockID() << ":\n";

            // map stmts to block ids
            for (auto EI = B.begin(); EI != B.end(); ++EI) {
                const CFGElement &E = *EI;
                if (std::optional<CFGStmt> CS = E.getAs<CFGStmt>()) {
                    const Stmt &S = *CS->getStmt();
                    blockIdOfStmt[&S] = B.getBlockID();

                    llvm::errs() << "  " << S.getStmtClassName() << " ("
                                 << S.getID(Context) << ")\n";
                }
            }

            llvm::errs() << "  successors:";
            // add edges
            for (auto SI = B.succ_begin(); SI != B.succ_end(); ++SI) {
                const CFGBlock *Succ = *SI;
                g.addEdge(B.getBlockID(), Succ->getBlockID());

                llvm::errs() << " " << Succ->getBlockID();
            }
            llvm::errs() << "\n";
        }
    }

    int getBlockId(const Stmt &s) {
        auto it = blockIdOfStmt.find(&s);
        if (it == blockIdOfStmt.end())
            return -1;
        return it->second;
    }
};

struct FunctionInfo {
    const FunctionDecl *D;
    std::string name;
    std::string file;
    int line;
    int column;

    const CFG *cfg;
    BlockGraph *bg;

    static FunctionInfo *fromDecl(FunctionDecl *D) {
        // ensure that the function has a body
        if (!D->hasBody())
            return nullptr;

        // get location
        FullSourceLoc FullLocation =
            D->getASTContext().getFullLoc(D->getBeginLoc());
        if (FullLocation.isInvalid() || !FullLocation.hasManager())
            return nullptr;

        std::string name = D->getQualifiedNameAsString();
        int line = FullLocation.getSpellingLineNumber();
        int column = FullLocation.getSpellingColumnNumber();
        StringRef file = FullLocation.getFileEntry()->tryGetRealPathName();
        requireTrue(!file.empty());

        // build CFG
        const std::unique_ptr<CFG> cfg = CFG::buildCFG(
            D, D->getBody(), &D->getASTContext(), CFG::BuildOptions());

        // build graph for each CFGBlock
        BlockGraph *bg = new BlockGraph(D->getASTContext(), cfg.get());

        FunctionInfo *fi = new FunctionInfo();
        fi->D = D;
        fi->name = name;
        fi->file = file;
        fi->line = line;
        fi->column = column;
        fi->cfg = cfg.get();
        fi->bg = bg;
        return fi;
    }
};

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
        const ASTContext &Context = AST->getASTContext();

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
}