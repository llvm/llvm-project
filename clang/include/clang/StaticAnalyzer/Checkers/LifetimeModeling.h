#ifndef LLVM_CLANG_INCLUDE_STATICANALYZER_CHECKERS_LIFETIMEMODELING_H
#define LLVM_CLANG_INCLUDE_STATICANALYZER_CHECKERS_LIFETIMEMODELING_H

#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include <vector>

namespace clang::ento::lifetime_modeling {
// FIXME: Once #205951 lands this function might needs to be removed.
std::vector<const MemRegion *> getLifetimeSourceSet(ProgramStateRef, SVal);

/// Returns true if the SVal key is present in the map.
bool isBoundToLifetimeSourceSet(ProgramStateRef State, SVal Val);

// FIXME: Once PR #205951 lands this function might needs to be removed.
bool isDeallocated(ProgramStateRef, const MemRegion *);

} // namespace clang::ento::lifetime_modeling

#endif // LLVM_CLANG_INCLUDE_STATICANALYZER_CHECKERS_LIFETIMEMODELING_H
