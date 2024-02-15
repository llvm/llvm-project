#ifndef VARFINDER_H
#define VARFINDER_H

#include "FunctionInfo.h"
#include "utils.h"

/**
 * Visit all DeclRefExprs and print their parents.
 */
class FindVarVisitor : public RecursiveASTVisitor<FindVarVisitor> {
  private:
    ASTContext *Context;
    Location targetLoc;
    std::string found;

  public:
    FindVarVisitor() {}

    const std::string &findVarInStmt(ASTContext *Context, const Stmt *S, //
                                     std::string file, int line, int column) {
        this->Context = Context;
        this->targetLoc = Location{file, line, column};
        this->found.clear();
        TraverseStmt(const_cast<Stmt *>(S));
        return found;
    }

    bool findMatch(const Stmt *S, const NamedDecl *decl,
                   const SourceLocation &loc) {
        std::unique_ptr<Location> pLoc =
            Location::fromSourceLocation(*Context, loc);
        if (!pLoc)
            return false;

        bool match = (*pLoc) == targetLoc;
        if (match)
            found = decl->getNameAsString();

        return match;
    }

    bool VisitDeclStmt(DeclStmt *ds) {
        for (Decl *d : ds->decls()) {
            if (VarDecl *vd = dyn_cast<VarDecl>(d)) {
                if (findMatch(ds, vd, vd->getLocation()))
                    return false;
            }
        }
        return true;
    }

    bool VisitDeclRefExpr(DeclRefExpr *dre) {
        if (findMatch(dre, dre->getDecl(), dre->getBeginLoc()))
            return false;
        return true;
    }

    bool VisitMemberExpr(MemberExpr *me) {
        if (findMatch(me, me->getMemberDecl(), me->getMemberLoc()))
            return false;
        return true;
    }
};

#endif