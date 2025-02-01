#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_DANGLING_REFERENCE_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_DANGLING_REFERENCE_H
#include "clang/AST/DeclBase.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
namespace clang {
class DanglingReferenceReporter {
public:
  DanglingReferenceReporter() = default;
  virtual ~DanglingReferenceReporter() = default;

  virtual void ReportReturnLocalVar(const Expr *RetExpr,
                                    const Decl *LocalDecl) {}
  virtual void ReportReturnTemporaryExpr(const Expr *TemporaryExpr) {}
};

void runDanglingReferenceAnalysis(const DeclContext &dc, const CFG &cfg,
                                  AnalysisDeclContext &ac,
                                  DanglingReferenceReporter *reporter);

} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_DANGLING_REFERENCE_H
