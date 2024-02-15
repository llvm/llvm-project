#ifndef FUNCTIONINFO_H
#define FUNCTIONINFO_H

#include "utils.h"

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

struct FunctionInfo {
    const FunctionDecl *D;
    std::string signature;
    std::string file;
    int line;
    int column;

    const CFG *cfg;
    std::vector<std::pair<const Stmt *, const CFGBlock *>> stmtBlockPairs;

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

        FunctionInfo *fi = new FunctionInfo();
        fi->D = D;
        fi->signature = getFullSignature(D);
        fi->file = pLoc->file;
        fi->line = pLoc->line;
        fi->column = pLoc->column;
        fi->cfg = cfg;
        fi->buildStmtBlockPairs();
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