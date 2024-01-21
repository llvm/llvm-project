#ifndef FUNCTIONINFO_H
#define FUNCTIONINFO_H

#include "utils.h"

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

#endif