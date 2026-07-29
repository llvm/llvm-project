#ifndef LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_LIFETIMEMODELING_H
#define LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_LIFETIMEMODELING_H

#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include <vector>

namespace clang::ento::lifetime_modeling {
/// Returns the set of lifetime sources bound to \p Source that are dangling
/// stack regions.
std::vector<const MemRegion *>
getDanglingRegionsAfterReturn(SVal Source, ProgramStateRef State,
                              CheckerContext &C);

/// Returns true if the underlying MemRegion is deallocated.
bool isDeallocated(ProgramStateRef State, const MemRegion *Region);

/// Returns true if \p Val is a key in the LifetimeBoundMap.
bool isBoundToLifetimeSource(ProgramStateRef State, SVal Val);

/// Returns the descriptive name of the memory region or a placeholder if a
/// descriptive name cannot be constructed for it.
std::string getRegionName(const MemRegion *Reg);
} // namespace clang::ento::lifetime_modeling

#endif // LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_LIFETIMEMODELING_H
