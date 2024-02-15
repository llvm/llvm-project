#ifndef FUNCTIONINFO_H
#define FUNCTIONINFO_H

#include "utils.h"

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