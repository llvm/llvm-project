#ifndef LLVM_CLANG_INCLUDE_STATICANALYZER_CHECKERS_LIFETIMEMODELING_H
#define LLVM_CLANG_INCLUDE_STATICANALYZER_CHECKERS_LIFETIMEMODELING_H

#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include <vector>

namespace clang::ento::lifetimemodeling {
/// Returns true if the lifetime of a region has ended.
bool isDeallocated(ProgramStateRef State, const MemRegion *Region);

/// Returns the set of of lifetime sources bound to \p Source that are dangling stack regions.
const std::vector<const MemRegion *> checkReturnedBorrower(SVal Source, ProgramStateRef State, CheckerContext &C);

/// Writes the lifetime sources bound to Source to OS.
void dumpLifetimeSources(ProgramStateRef State, SVal Source, raw_ostream &OS);

} // namespace clang::ento::lifetimemodeling

#endif // LLVM_CLANG_INCLUDE_STATICANALYZER_CHECKERS_LIFETIMEMODELING_H
