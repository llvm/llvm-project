#ifndef VARFINDER_H
#define VARFINDER_H

#include "FunctionInfo.h"
#include "utils.h"

struct VarLocation {
    std::string file; // absolute path
    int line;
    int column;

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

    std::vector<const Stmt *> resultStmtChain;

    explicit FindVarVisitor(ASTContext *Context, VarLocation targetLoc)
        : Context(Context), targetLoc(targetLoc) {}

  public:
    template <typename NodeT> void visitParentsRecursively(const NodeT &base) {
        const DynTypedNodeList &parents = Context->getParents(base);
        requireTrue(parents.size() == 1, "parent size is not 1");

        if (const Stmt *parent = parents.begin()->get<Stmt>()) {
            llvm::errs() << "    " << parent->getStmtClassName() << "\n";
            resultStmtChain.push_back(parent);
            if (isa<CompoundStmt>(parent))
                return;
            visitParentsRecursively(*parent);
        } else if (const Decl *parent = parents.begin()->get<Decl>()) {
            llvm::errs() << "    " << parent->getDeclKindName() << "\n";
            visitParentsRecursively(*parent);
        } else {
            llvm::errs() << "    unknown parent\n";
            exit(1);
        }
    }

    bool findMatch(const Stmt *S, const NamedDecl *decl,
                   const SourceLocation &loc) {
        FullSourceLoc FullLocation = Context->getFullLoc(loc);
        VarLocation varLoc(FullLocation);
        bool match = varLoc == targetLoc;
        if (match) {
            const auto &var = decl->getQualifiedNameAsString();
            llvm::errs() << "Found VarDecl: " << var << "\n";
            llvm::errs() << "  at " << varLoc.file << ":" << varLoc.line << ":"
                         << varLoc.column << "\n";
            llvm::errs() << "  parents:\n";

            requireTrue(resultStmtChain.empty());
            resultStmtChain.push_back(S);
            visitParentsRecursively(*S);
        }

        return match;
    }

    bool VisitDeclStmt(DeclStmt *ds) {
        for (Decl *d : ds->decls()) {
            if (VarDecl *vd = dyn_cast<VarDecl>(d)) {
                findMatch(ds, vd, vd->getLocation());
            }
        }
        return true;
    }

    bool VisitDeclRefExpr(DeclRefExpr *dre) {
        findMatch(dre, dre->getDecl(), dre->getBeginLoc());
        return true;
    }

    static std::vector<const Stmt *> findVar(
        std::map<std::string, std::set<const FunctionInfo *>> functionsInFile,
        VarLocation targetLoc) {
        for (const FunctionInfo *fi : functionsInFile[targetLoc.file]) {
            llvm::errs() << "------ FunctionDecl: " << fi->name << " at "
                         << fi->line << ":" << fi->column << "\n";
            ASTContext *Context = &fi->D->getASTContext();
            FindVarVisitor fv(Context, targetLoc);
            TranslationUnitDecl *TUD = Context->getTranslationUnitDecl();
            fv.TraverseDecl(TUD);
            if (!fv.resultStmtChain.empty())
                return fv.resultStmtChain;
        }
        requireTrue(false, "no match found!");
    }
};

#endif