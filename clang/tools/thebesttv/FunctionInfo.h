#ifndef FUNCTIONINFO_H
#define FUNCTIONINFO_H

#include "utils.h"

struct Graph {
    int n; // 0-indexed
    std::vector<std::set<int>> G;
    std::vector<int> d, fa;

    const int INF = 0x3f3f3f3f;

    struct Node {
        int u, d, fa;
        bool operator<(const Node &b) const { return d > b.d; }
    };

    Graph(int n) : n(n), G(n), d(n), fa(n) {}

    void addEdge(int u, int v) { G[u].insert(v); }

    void dij(int s) {
        std::fill(d.begin(), d.end(), INF);
        d[s] = 0;

        std::fill(fa.begin(), fa.end(), -1);

        std::priority_queue<Node> q;
        q.push({s, 0, -1});

        while (!q.empty()) {
            Node p = q.top();
            q.pop();
            int u = p.u;
            if (p.d != d[u])
                continue;

            fa[u] = p.fa;

            for (int v : G[u]) {
                if (d[v] > d[u] + 1) {
                    d[v] = d[u] + 1;
                    q.push({v, d[v], u});
                }
            }
        }
    }

    std::vector<int> trace(int u) {
        std::vector<int> path;
        while (u != -1) {
            path.push_back(u);
            u = fa[u];
        }
        std::reverse(path.begin(), path.end());
        return path;
    }
};

struct DfsTraverse {
    DfsTraverse(const Graph &G) : G(G) {}

    const Graph &G;

    int source;
    int target;
    int maxDistance;
    std::vector<int> path;
    std::vector<bool> visiting;
    std::set<std::vector<int>> results;

    void search(int source, int target, int maxDistance) {
        this->source = source;
        this->target = target;
        this->maxDistance = maxDistance;

        this->path.clear();
        this->path.push_back(source);

        this->visiting.resize(G.n, false);

        dfs(source, 0);
    }

    void dfs(int u, int d) {
        if (d > maxDistance || (d == maxDistance && u != target))
            return;

        if (u == target) {
            // llvm::errs() << "found:";
            // for (int v : path) {
            //     llvm::errs() << " " << v;
            // }
            // llvm::errs() << "\n";

            results.insert(path);
            return;
        }

        for (int v : G.G[u]) {
            if (visiting[v])
                continue;

            visiting[v] = true;
            this->path.push_back(v);
            dfs(v, d + 1);
            this->path.pop_back();
            visiting[v] = false;
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
                // Successor may be null in case of optimized-out edges. See:
                // https://clang.llvm.org/doxygen/classclang_1_1CFGBlock.html#details
                if (!Succ)
                    continue;
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

    CallGraph *cg;

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
        const FileEntry *fileEntry = FullLocation.getFileEntry();
        if (!fileEntry)
            return nullptr;
        StringRef file = fileEntry->tryGetRealPathName();
        requireTrue(!file.empty());

        // build CFG
        CFG *cfg = CFG::buildCFG(D, D->getBody(), &D->getASTContext(),
                                 CFG::BuildOptions())
                       .release();
        // CFG may be null (may be because the function is in STL, e.g.
        // "std::destroy_at")
        if (!cfg)
            return nullptr;

        // build graph for each CFGBlock
        BlockGraph *bg = new BlockGraph(cfg);

        CallGraph *cg = new CallGraph();
        cg->addToCallGraph(D->getASTContext().getTranslationUnitDecl());

        FunctionInfo *fi = new FunctionInfo();
        fi->D = D;
        fi->name = name;
        fi->file = file;
        fi->line = line;
        fi->column = column;
        fi->cfg = cfg;
        fi->buildStmtBlockPairs();
        fi->bg = bg;
        fi->cg = cg;
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