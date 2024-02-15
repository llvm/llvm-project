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
    DfsTraverse(const ICFG &icfg) : icfg(icfg) {}

    const ICFG &icfg;

    int source;
    int target;
    int maxCallDepth;
    std::vector<int> path;
    std::vector<bool> visiting;
    std::set<std::vector<int>> results;

    std::stack<int> callStack; // 部分平衡的括号匹配
    std::set<int> callSites;

    void search(int source, int target, int maxCallDepth) {
        this->source = source;
        this->target = target;
        this->maxCallDepth = maxCallDepth;

        path.clear();
        path.push_back(source);

        visiting.resize(icfg.n);
        std::fill(visiting.begin(), visiting.end(), false);

        while (!callStack.empty())
            callStack.pop();
        callSites.clear();

        dfs(source);
    }

    void dfs(int u) {
        if (u == target) {
            llvm::errs() << "found:";
            for (int v : path) {
                llvm::errs() << " " << v;
            }
            llvm::errs() << "\n";

            results.insert(path);
            return;
        }

        for (const auto &e : icfg.G[u]) {
            int v = e.target;
            if (visiting[v])
                continue;

            if (e.type == ICFG::Edge::Type::INTRA_PROC) {
                visiting[v] = true;
                path.push_back(v);
                dfs(v);
                path.pop_back();
                visiting[v] = false;
            } else {
                std::stack<int> oldCallStack = callStack;
                std::set<int> oldCallSites = callSites;

                if (e.type == ICFG::Edge::Type::CALL_EDGE) { // 左括号
                    callStack.push(e.callSiteId);
                } else { // 右括号
                    if (!callStack.empty()) {
                        if (callStack.top() != e.callSiteId) {
                            continue;
                        } else {
                            callStack.pop();
                        }
                    }
                }
                callSites.insert(e.callSiteId);

                if (callSites.size() <= maxCallDepth) {
                    visiting[v] = true;
                    path.push_back(v);
                    dfs(v);
                    path.pop_back();
                    visiting[v] = false;
                }

                callSites = oldCallSites;
                callStack = oldCallStack;
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
    std::string signature;
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
        std::unique_ptr<Location> pLoc =
            Location::fromSourceLocation(D->getASTContext(), D->getBeginLoc());
        if (!pLoc)
            return nullptr;

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

        FunctionInfo *fi = new FunctionInfo();
        fi->D = D;
        fi->signature = getFullSignature(D);
        fi->file = pLoc->file;
        fi->line = pLoc->line;
        fi->column = pLoc->column;
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