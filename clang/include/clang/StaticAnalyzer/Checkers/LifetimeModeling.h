#ifndef LLVM_CLANG_INCLUDE_STATICANALYZER_CHECKERS_LIFETIMEMODELING_H
#define LLVM_CLANG_INCLUDE_STATICANALYZER_CHECKERS_LIFETIMEMODELING_H

#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include <vector>

namespace clang::ento::lifetime_modeling {

/// Returns the set of lifetime sources bound to \p Source that are dangling
/// stack regions.
const std::vector<const MemRegion *>
checkReturnedBorrower(SVal Source, ProgramStateRef State, CheckerContext &C);

/// Returns true if the SVal key is present in the map.
bool isBoundToLifetimeSource(SVal Val, ProgramStateRef State);
} // namespace clang::ento::lifetime_modeling

#endif // LLVM_CLANG_INCLUDE_STATICANALYZER_CHECKERS_LIFETIMEMODELING_H
