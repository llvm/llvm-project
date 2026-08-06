//===- BoundsChecking.h - Bounds checking related APIs ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This header declares 'checkBounds', a function that compares memory offsets
//  (that may be symbolic) and uses heuristical workarounds to provide more
//  accurate results than directly calling evalBinOp or assumeInBound.
//
//  As of now, this logic only supports the needs of `security.ArrayBound`, but
//  in the future it will be generalized and applied in all checkers that
//  perform bounds checking (to bring them out of `alpha` stage).
//
//  TODO: This header should be extended by other utilities (e.g. message
//  formatting tools) that are relevant for multiple bounds checking checkers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CHECKERS_BOUNDSCHECKING_H
#define LLVM_CLANG_STATICANALYZER_CHECKERS_BOUNDSCHECKING_H
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include <optional>

namespace clang::ento::bounds {

struct CheckFlags {
  unsigned CheckUnderflow : 1;
  unsigned OffsetObviouslyNonnegative : 1;
  unsigned AcceptPastTheEnd : 1;
};

class CheckResult;

/// Checks the validity of accessing a memory region with extent \p Extent at
/// offset \p Offset. The \p Flags influence the semantics of the check, in
/// particular if `AcceptPastTheEnd` is true, then Offset == Extent is also
/// accepted as valid.
CheckResult checkBounds(ProgramStateRef State, SValBuilder &SVB, NonLoc Offset,
                        std::optional<NonLoc> Extent, CheckFlags Flags);

class CheckResult {
public:
  /// When true, the bounds check noticed that the value of an unsigned
  /// expression is constrained to negative values (because the analyzer
  /// skipped the modeling of a cast expression). This execution path must be
  /// discarded because it does not represent a real possibility.
  /// FIXME: This hack is currently needed to filter out many ugly false
  /// positives; but it should be removed when we fix cast modeling.
  bool isCorruptedState() const { return IsCorruptedState; }

  /// When true, the checked offset may be in bounds.
  /// As an exceptional case, this is also true for idiomatic expressions that
  /// define a past-the-end pointer (and do not dereference it).
  bool mayBeInBounds() const { return static_cast<bool>(InBoundsState); }

  /// When true, the checked offset may be negative.
  bool mayUnderflow() const { return MayUnderflow; }
  /// When true, the checked offset may be >= the extent of the region.
  /// As an exceptional case, this is also false for idiomatic expressions that
  /// define a past-the-end pointer (and do not dereference it).
  bool mayOverflow() const { return ExtentIfMayOverflow.has_value(); }
  /// When true, the checked offset may be out of bounds.
  bool mayBeInvalid() const { return MayUnderflow || ExtentIfMayOverflow; }

  /// Returns the offset of the accessed location from the beginning of the
  /// accessd region.
  NonLoc getOffset() const { return Offset; }

  /// Returns the extent of the accessed region if it is relevant (because the
  /// offset may overflow it), otherwise returns std::nullopt.
  std::optional<NonLoc> getExtentIfMayOverflow() const {
    return ExtentIfMayOverflow;
  }

  /// Returns the program state that should be used for continuing the analysis
  /// after this bounds check. This returns null if mayBeInBounds() is false, in
  /// that case the state before the check should be used in the error node.
  /// Note that we also have a valid state in the exception case when the
  /// 'access' calculates the past-the-end pointer without dereferencing it.
  ProgramStateRef getInBoundsState() const { return InBoundsState; }

  friend CheckResult checkBounds(ProgramStateRef State, SValBuilder &SVB,
                                 NonLoc Offset, std::optional<NonLoc> Extent,
                                 CheckFlags Flags);

private:
  // Offset of the accessed location, measured from the start of the region.
  // TODO: As of now, the offset and the extent are always measured in bytes,
  // but we will probably need to allow other size units in the future.
  const NonLoc Offset;

  explicit CheckResult(NonLoc Offs) : Offset(Offs) {}

  bool IsCorruptedState = false;
  bool MayUnderflow = false;
  std::optional<NonLoc> ExtentIfMayOverflow = std::nullopt;
  ProgramStateRef InBoundsState = nullptr;
};

} // namespace clang::ento::bounds

#endif // LLVM_CLANG_STATICANALYZER_CHECKERS_BOUNDSCHECKING_H
