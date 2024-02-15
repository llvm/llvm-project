#ifndef HAS_CONTEXT_VISITOR_H
#define HAS_CONTEXT_VISITOR_H

#include "utils.h"

class HasContextVisitor {
  protected:
    ASTContext *Context;

    std::string getLocation(const SourceLocation &loc) {
        PresumedLoc PLoc = Context->getSourceManager().getPresumedLoc(loc);
        if (PLoc.isInvalid())
            return "<invalid>";

        std::string filename = PLoc.getFilename();
        std::string line = std::to_string(PLoc.getLine());
        std::string column = std::to_string(PLoc.getColumn());
        return filename + ":" + line + ":" + column;
    }

    void printStmtLocation(const Stmt &s) {
        llvm::errs() << "    beg: " << getLocation(s.getBeginLoc()) << "\n";
        llvm::errs() << "    end: " << getLocation(s.getEndLoc()) << "\n";
    }

  public:
    explicit HasContextVisitor(ASTContext *Context) : Context(Context) {}
};

/**
 * Visit all DeclRefExprs and print their parents.
 */
class VarVisitor : public RecursiveASTVisitor<VarVisitor>,
                   public HasContextVisitor {
  public:
    explicit VarVisitor(ASTContext *Context) : HasContextVisitor(Context) {}

    void visitParents(const Stmt &base) {
        const Stmt *s = &base;
        llvm::errs() << "    parents:\n";
        while (true) {
            const DynTypedNodeList &parents = Context->getParents(*s);
            requireTrue(parents.size() == 1, "parent size is not 1");

            const Stmt *parent = parents.begin()->get<Stmt>();
            requireTrue(parent != nullptr, "parent is null");

            llvm::errs() << "      " << parent->getStmtClassName() << "\n";
            if (isa<CompoundStmt>(parent)) {
                break;
            }

            s = parent;
        }
    }

    bool VisitStmt(Stmt *s) {
        // DeclRefExpr
        if (DeclRefExpr *dre = dyn_cast<DeclRefExpr>(s)) {
            llvm::errs() << "  DeclRefExpr: "
                         << dre->getDecl()->getQualifiedNameAsString() << "\n";
            printStmtLocation(*s);
            visitParents(*s);
        }
        return true;
    }
};

/**
 * Visit all FunctionDecls and print their CFGs.
 */
class FunctionDeclVisitor : public RecursiveASTVisitor<FunctionDeclVisitor>,
                            public HasContextVisitor {
  public:
    explicit FunctionDeclVisitor(ASTContext *Context)
        : HasContextVisitor(Context) {}

    bool VisitFunctionDecl(FunctionDecl *D) {
        std::unique_ptr<Location> pLoc =
            Location::fromSourceLocation(*Context, D->getBeginLoc());

        llvm::errs() << "------ FunctionDecl: " << D->getQualifiedNameAsString()
                     << " at " << pLoc->line << ":" << pLoc->column << "\n";

        if (!D->hasBody())
            return true;

        // show call graph
        // TranslationUnitDecl *TUD = Context->getTranslationUnitDecl();
        // CallGraph CG;
        // CG.addToCallGraph(TUD);
        // CG.viewGraph();

        llvm::errs() << "--------- CFG dump: " << D->getQualifiedNameAsString()
                     << "\n";
        // build CFG
        auto cfg = CFG::buildCFG(D, D->getBody(), &D->getASTContext(),
                                 CFG::BuildOptions());
        cfg->dump(D->getASTContext().getLangOpts(), true);
        // cfg->viewCFG(D->getASTContext().getLangOpts());

        int n = cfg->size(); // num of blocks
        llvm::errs() << "Num of blocks: " << n << "\n";

        // traverse each block
        llvm::errs() << "--------- Block traversal: "
                     << D->getQualifiedNameAsString() << "\n";
        for (auto BI = cfg->begin(); BI != cfg->end(); ++BI) {
            const CFGBlock &B = **BI;
            // print block ID
            llvm::errs() << "Block " << B.getBlockID();
            if (&B == &cfg->getEntry()) {
                llvm::errs() << " (Entry)";
            } else if (&B == &cfg->getExit()) {
                llvm::errs() << " (Exit)";
            }
            llvm::errs() << ":\n";

            // traverse & print block contents
            for (auto EI = B.begin(); EI != B.end(); ++EI) {
                const CFGElement &E = *EI;
                llvm::errs() << "  ";
                E.dump();
                if (std::optional<CFGStmt> CS = E.getAs<CFGStmt>()) {
                    // CS->getStmt()->dump();
                    // printStmtLocation(*CS->getStmt());
                }
            }

            // print block terminator
            if (B.getTerminator().isValid()) {
                const CFGTerminator &T = B.getTerminator();
                if (T.getStmt()) {
                    const Stmt &S = *T.getStmt();
                    llvm::errs() << "  T: <" << S.getStmtClassName() << ">\n";
                    // printStmtLocation(S);
                }
            }

            // print predecessors
            llvm::errs() << "  Preds:";
            for (auto PI = B.pred_begin(); PI != B.pred_end(); ++PI) {
                const CFGBlock *Pred = *PI;
                llvm::errs() << " B" << Pred->getBlockID();
            }
            llvm::errs() << "\n";

            // print successors
            llvm::errs() << "  Succs:";
            for (auto SI = B.succ_begin(); SI != B.succ_end(); ++SI) {
                const CFGBlock *Succ = *SI;
                llvm::errs() << " B" << Succ->getBlockID();
            }
            llvm::errs() << "\n";
        }

        return true;
    }
};

#endif