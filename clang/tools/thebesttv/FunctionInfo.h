#ifndef FUNCTIONINFO_H
#define FUNCTIONINFO_H

#include "utils.h"

struct FunctionInfo {
    const FunctionDecl *D;
    std::string signature;
    std::string file;
    int line;
    int column;

    std::unique_ptr<CFG> cfg;
    std::vector<std::pair<const Stmt *, const CFGBlock *>> stmtBlockPairs;

    static std::unique_ptr<FunctionInfo> fromDecl(FunctionDecl *D) {
        // ensure that the function has a body
        if (!D->hasBody())
            return nullptr;

        // get location
        std::unique_ptr<Location> pLoc =
            Location::fromSourceLocation(D->getASTContext(), D->getBeginLoc());
        if (!pLoc)
            return nullptr;

        // build CFG

        auto cfg = CFG::buildCFG(D, D->getBody(), &D->getASTContext(),
                                 CFG::BuildOptions());
        // CFG may be null (may be because the function is in STL, e.g.
        // "std::destroy_at")
        if (!cfg)
            return nullptr;

        auto fi = std::make_unique<FunctionInfo>();
        fi->D = D;
        fi->signature = getFullSignature(D);
        fi->file = pLoc->file;
        fi->line = pLoc->line;
        fi->column = pLoc->column;
        fi->cfg = std::move(cfg);
        fi->buildStmtBlockPairs();
        return fi;
    }

  private:
    void buildStmtBlockPairs() {
        // cfg->begin() & cfg->end() 返回的 Block ID 是递增的
        // CFG 中的 succ 的 ID 更小，对应也就是先访问 succ 再 pred
        for (auto BI = cfg->begin(); BI != cfg->end(); ++BI) {
            const CFGBlock &B = **BI;

            // swith 的 case 语句等
            if (B.getLabel() != nullptr) {
                stmtBlockPairs.push_back({B.getLabel(), &B});
            }

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