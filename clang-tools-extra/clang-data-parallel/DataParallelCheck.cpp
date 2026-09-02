#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "llvm/Support/CommandLine.h"
#include <iostream>
#include <unordered_set>

using namespace clang;
using namespace clang::tooling;

static llvm::cl::OptionCategory MyToolCategory("clang-data-parallel options");

static llvm::cl::opt<bool> AnalyzeDataParallelism("analyze-data-parallelism",
    llvm::cl::desc("Enable data parallelism analysis"),
    llvm::cl::cat(MyToolCategory));

class DataParallelVisitor : public RecursiveASTVisitor<DataParallelVisitor> {
public:
    DataParallelVisitor(ASTContext &Ctx) : Context(&Ctx) {}

    bool VisitFunctionDecl(FunctionDecl *FD) {
        if (!AnalyzeDataParallelism)
            return true;

        // Skip non-user-defined functions
        if (!isUserCode(FD) || !FD->hasBody() || FD->getBuiltinID() != 0 ||
            FD->isInlined() || FD->getStorageClass() == SC_Extern) {
            return true;
        }

        functionName = FD->getNameInfo().getName().getAsString();
        hasDataParallelism = false;
        foundDependentAccess = false;

        TraverseStmt(FD->getBody());

        if (hasDataParallelism && !foundDependentAccess) {
            llvm::outs() << "Function '" << functionName
                         << "' has data parallelism: Independent loop iterations detected at line "
                         << loopLine << "\n";
        } else {
            llvm::outs() << "Function '" << functionName << "' does not have data parallelism\n";
        }

        return true;
    }

    bool VisitForStmt(ForStmt *FS) {
        if (!AnalyzeDataParallelism) return true;

        // Reset state for each loop
        hasDataParallelism = false;
        foundDependentAccess = false;
        loopVar = nullptr;
        hasValidAssignment = false;

        // Check loop variable declaration
        if (auto *InitStmt = dyn_cast_or_null<DeclStmt>(FS->getInit())) {
            if (auto *VD = dyn_cast_or_null<VarDecl>(InitStmt->getSingleDecl())) {
                loopVar = VD;

                // Skip if loop has conditionals or is not simple
                if (hasConditional(FS->getBody()) || !isSimpleLoop(FS)) {
                    return true;
                }

                TraverseStmt(FS->getBody());

                if (hasValidAssignment && !foundDependentAccess) {
                    hasDataParallelism = true;
                    unsigned Line = Context->getSourceManager().getPresumedLineNumber(FS->getBeginLoc());
                    loopLine = std::to_string(Line);
                }
            }
        }

        return true;
    }

    bool VisitBinaryOperator(BinaryOperator *BO) {
        if (!AnalyzeDataParallelism || !loopVar) return true;

        if (BO->getOpcode() == BO_Assign) {
            // Check LHS for array access with loop variable
            if (auto *ASE = dyn_cast<ArraySubscriptExpr>(BO->getLHS()->IgnoreImpCasts())) {
                if (auto *Idx = dyn_cast<DeclRefExpr>(ASE->getIdx()->IgnoreImpCasts())) {
                    if (Idx->getDecl() == loopVar) {
                        hasValidAssignment = true;
                        // Check RHS for dependent accesses
                        if (HasDependentAccess(BO->getRHS())) {
                            foundDependentAccess = true;
                        }
                    }
                }
            }
            // Check if LHS itself introduces a dependency (e.g., data[i] = data[i-1] + data[i])
            if (HasDependentAccess(BO->getLHS())) {
                foundDependentAccess = true;
            }
        }

        return true;
    }

private:
    ASTContext *Context;
    VarDecl *loopVar = nullptr;
    std::string functionName;
    std::string loopLine;
    bool hasValidAssignment = false;
    bool hasDataParallelism = false;
    bool foundDependentAccess = false;

    // Check if the declaration is in user code
    bool isUserCode(const Decl *D) {
        SourceManager &SM = Context->getSourceManager();
        SourceLocation Loc = D->getLocation();
        if (!Loc.isValid() || SM.isInSystemHeader(Loc)) {
            return false;
        }
        return SM.isInMainFile(Loc);
    }

    // Check for conditional statements in loop body
    bool hasConditional(Stmt *S) {
        class ConditionalVisitor : public RecursiveASTVisitor<ConditionalVisitor> {
        public:
            bool VisitIfStmt(IfStmt *) { found = true; return false; }
            bool VisitSwitchStmt(SwitchStmt *) { found = true; return false; }
            bool found = false;
        };

        ConditionalVisitor CV;
        CV.TraverseStmt(S);
        return CV.found;
    }

    // Verify if loop is simple (e.g., i++ or i+=1)
    bool isSimpleLoop(ForStmt *FS) {
        if (!FS->getInc()) return false;

        // Check increment (e.g., i++ or i+=1)
        if (auto *UO = dyn_cast<UnaryOperator>(FS->getInc())) {
            if (UO->getOpcode() != UO_PostInc && UO->getOpcode() != UO_PreInc) {
                return false;
            }
            if (auto *DRE = dyn_cast<DeclRefExpr>(UO->getSubExpr()->IgnoreImpCasts())) {
                return DRE->getDecl() == loopVar;
            }
        } else if (auto *BO = dyn_cast<BinaryOperator>(FS->getInc())) {
            if (BO->getOpcode() != BO_Assign && BO->getOpcode() != BO_AddAssign) {
                return false;
            }
            if (auto *DRE = dyn_cast<DeclRefExpr>(BO->getLHS()->IgnoreImpCasts())) {
                if (DRE->getDecl() != loopVar) return false;
                if (BO->getOpcode() == BO_AddAssign) {
                    if (auto *IL = dyn_cast<IntegerLiteral>(BO->getRHS())) {
                        return IL->getValue().getSExtValue() == 1;
                    }
                }
            }
        }

        return true;
    }

    // Check for dependent array accesses (e.g., data[i-1], data[i+1])
    bool HasDependentAccess(Expr *E) {
        class DependencyVisitor : public RecursiveASTVisitor<DependencyVisitor> {
        public:
            DependencyVisitor(VarDecl *LV) : loopVar(LV) {}

            bool VisitArraySubscriptExpr(ArraySubscriptExpr *ASE) {
                Expr *Idx = ASE->getIdx()->IgnoreImpCasts();
                // Check if index involves loop variable with offset
                if (auto *BO = dyn_cast<BinaryOperator>(Idx)) {
                    if (BO->getOpcode() == BO_Add || BO->getOpcode() == BO_Sub) {
                        Expr *LHS = BO->getLHS()->IgnoreImpCasts();
                        Expr *RHS = BO->getRHS()->IgnoreImpCasts();
                        if (auto *DRE = dyn_cast<DeclRefExpr>(LHS)) {
                            if (DRE->getDecl() == loopVar) {
                                if (auto *IL = dyn_cast<IntegerLiteral>(RHS)) {
                                    int offset = IL->getValue().getSExtValue();
                                    if (offset != 0) {
                                        found = true;
                                        return false;
                                    }
                                }
                            }
                        }
                    }
                }
                // Check if index is directly the loop variable (no dependency)
                if (auto *DRE = dyn_cast<DeclRefExpr>(Idx)) {
                    if (DRE->getDecl() == loopVar) {
                        return true; // Not a dependency
                    }
                }
                // Any other index expression could be complex, conservatively assume dependency
                found = true;
                return false;
            }

            bool found = false;
            VarDecl *loopVar;
        };

        DependencyVisitor DV(loopVar);
        DV.TraverseStmt(E);
        return DV.found;
    }
};

// AST Consumer
class DataParallelConsumer : public ASTConsumer {
public:
    DataParallelConsumer(ASTContext &Ctx) : Visitor(Ctx) {}

    void HandleTranslationUnit(ASTContext &Ctx) override {
        Visitor.TraverseDecl(Ctx.getTranslationUnitDecl());
    }

private:
    DataParallelVisitor Visitor;
};

// Frontend Action
class DataParallelAction : public ASTFrontendAction {
public:
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override {
        return std::make_unique<DataParallelConsumer>(CI.getASTContext());
    }
};

// Main
int main(int argc, const char **argv) {
    auto ExpectedParser = CommonOptionsParser::create(argc, argv, MyToolCategory);
    if (!ExpectedParser) {
        llvm::errs() << ExpectedParser.takeError();
        return 1;
    }
    ClangTool Tool(ExpectedParser->getCompilations(), ExpectedParser->getSourcePathList());

    // Run the tool, relying on CommonOptionsParser for error diagnostics
    return Tool.run(newFrontendActionFactory<DataParallelAction>().get());
}