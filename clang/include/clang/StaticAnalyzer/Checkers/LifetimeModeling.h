#ifndef LLVM_CLANG_INCLUDE_STATICANALYZER_CHECKERS_LIFETIMEMODELING_H
#define LLVM_CLANG_INCLUDE_STATICANALYZER_CHECKERS_LIFETIMEMODELING_H

#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SymbolManager.h"
#include <vector>

namespace clang {
namespace ento {
namespace lifetimemodeling {

std::vector<const MemRegion *> getLifetimeSourceSet(ProgramStateRef, SVal);
bool isDeallocated(ProgramStateRef, const MemRegion *);
ProgramStateRef removeDeadBindings(ProgramStateRef, SymbolReaper &);

} // namespace lifetimemodeling
} // namespace ento
} // namespace clang

#endif // LLVM_CLANG_INCLUDE_STATICANALYZER_CHECKERS_LIFETIMEMODELING_H
