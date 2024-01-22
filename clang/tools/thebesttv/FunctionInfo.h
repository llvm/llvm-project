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
    const CFG *cfg;
    Graph g;

    BlockGraph(const CFG *cfg) : g(cfg->size()) {
        for (auto BI = cfg->begin(); BI != cfg->end(); ++BI) {
            const CFGBlock &B = **BI;
            // add edges to successors
            for (auto SI = B.succ_begin(); SI != B.succ_end(); ++SI) {
                const CFGBlock *Succ = *SI;
                g.addEdge(B.getBlockID(), Succ->getBlockID());
            }
        }
    }

    void dij(const CFGBlock *s) { g.dij(s->getBlockID()); }
};

struct FunctionInfo {
    const FunctionDecl *D;
    std::string name;
    std::string file;
    int line;
    int column;

    const CFG *cfg;
    std::vector<std::pair<const Stmt *, const CFGBlock *>> stmtBlockPairs;
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
        CFG *cfg = CFG::buildCFG(D, D->getBody(), &D->getASTContext(),
                                 CFG::BuildOptions())
                       .release();

        // build graph for each CFGBlock
        BlockGraph *bg = new BlockGraph(cfg);

        FunctionInfo *fi = new FunctionInfo();
        fi->D = D;
        fi->name = name;
        fi->file = file;
        fi->line = line;
        fi->column = column;
        fi->cfg = cfg;
        fi->buildStmtBlockPairs();
        fi->bg = bg;
        return fi;
    }

  private:
    void buildStmtBlockPairs() {
        for (auto BI = cfg->begin(); BI != cfg->end(); ++BI) {
            const CFGBlock &B = **BI;
            for (auto EI = B.begin(); EI != B.end(); ++EI) {
                const CFGElement &E = *EI;
                if (std::optional<CFGStmt> CS = E.getAs<CFGStmt>()) {
                    const Stmt *S = CS->getStmt();
                    stmtBlockPairs.push_back({S, &B});
                }
            }
        }
    }
};

#endif