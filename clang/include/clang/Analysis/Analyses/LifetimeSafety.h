#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIME_SAFETY_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIME_SAFETY_H
#include "clang/AST/DeclBase.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
namespace clang {

void runLifetimeAnalysis(const DeclContext &DC, const CFG &Cfg,
                         AnalysisDeclContext &AC);

} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIME_SAFETY_H
