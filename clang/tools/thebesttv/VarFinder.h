#ifndef VARFINDER_H
#define VARFINDER_H

#include "FunctionInfo.h"
#include "utils.h"

struct VarLocation {
    std::string file; // absolute path
    int line;
    int column;

    VarLocation() : VarLocation("", -1, -1) {}

    VarLocation(std::string file, int line, int column)
        : file(file), line(line), column(column) {}

    VarLocation(const FullSourceLoc &fullLoc) {
        requireTrue(fullLoc.hasManager(), "no source manager!");
        requireTrue(fullLoc.isValid(), "invalid location!");

        file = fullLoc.getFileEntry()->tryGetRealPathName();
        line = fullLoc.getLineNumber();
        column = fullLoc.getColumnNumber();
        requireTrue(!file.empty(), "empty file path!");
    }

    bool operator==(const VarLocation &other) const {
        return file == other.file && line == other.line &&
               column == other.column;
    }
};

/**
 * Visit all DeclRefExprs and print their parents.
 */
class FindVarVisitor : public RecursiveASTVisitor<FindVarVisitor> {
  private:
    ASTContext *Context;
    VarLocation targetLoc;
    bool found = false;

  public:
    FindVarVisitor() {}

    bool findVarInStmt(ASTContext *Context, const Stmt *S,
                       VarLocation targetLoc) {
        this->Context = Context;
        this->targetLoc = targetLoc;
        this->found = false;
        TraverseStmt(const_cast<Stmt *>(S));
        return found;
    }

    bool findMatch(const Stmt *S, const NamedDecl *decl,
                   const SourceLocation &loc) {
        FullSourceLoc FullLocation = Context->getFullLoc(loc);
        VarLocation varLoc(FullLocation);
        bool match = varLoc == targetLoc;
        if (match)
            found = true;

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
};

#endif