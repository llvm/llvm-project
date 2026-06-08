//===- BoundsChecking.h - Bounds checking related APIs ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines APIs for performing a bounds check (i.e. comparing a
//  symbolic Offset value to zero and a symbolic Extent value) and composing
//  descriptions that explain its results.
//
//  This is intended as a replacement for `ProgramState::assumeInBound` to
//  avoid its incorrect logic and compensate for deficiencies of other parts of
//  the analyzer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_BOUNDSCHECKING_H
#define LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_BOUNDSCHECKING_H
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "llvm/Support/FormatVariadic.h"
#include <optional>

namespace clang {
namespace ento {

/// If `E` is an array subscript expression with a base that is "clean" (= not
/// modified by pointer arithmetic = the beginning of a memory region), return
/// it as a pointer to ArraySubscriptExpr; otherwise return nullptr.
/// This helper function is used by two separate heuristics that are only valid
/// in these "clean" cases.
const ArraySubscriptExpr *getAsCleanArraySubscriptExpr(const Expr *E,
                                                       const CheckerContext &C);

class SizeUnit {
  QualType AsType;
  int64_t AsCharUnits;

  SizeUnit() : AsType(), AsCharUnits(1) {}

public:
  SizeUnit(QualType T, const ASTContext &ACtx)
      : AsType(T), AsCharUnits(ACtx.getTypeSizeInChars(T).getQuantity()) {
    assert(!T.isNull());
  }

  static SizeUnit bytes() { return SizeUnit(); }

  bool isBytes() const { return AsType.isNull(); }

  /// If `E` is a "clean" array subscript expression, return the type of the
  /// accessed element; otherwise return 'Bytes' because that's the best (or
  /// least bad) option for the assumption messages that use this.
  static SizeUnit forExpr(const Expr *E, const CheckerContext &C) {
    const auto *ASE = getAsCleanArraySubscriptExpr(E, C);
    if (!ASE)
      return bytes();

    return SizeUnit(ASE->getType(), C.getASTContext());
  }

  /// Return the element type that is "natural" for reporting out-of-bounds
  /// memory access to 'Location'.
  /// FIXME: It is unfortunate that this heuristic differs from the heuristic
  /// used for reporting assumption (`SizeUnit::forExpr`).
  static SizeUnit forSVal(SVal Location, const ASTContext &ACtx) {
    const auto *EReg = Location.getAsRegion()->getAs<ElementRegion>();
    assert(EReg && "this checker only handles element access");
    return SizeUnit(EReg->getElementType(), ACtx);
  }

  int64_t asCharUnits() const { return AsCharUnits; }

  std::string asExtentDesc(bool ForceBytes) const {
    if (ForceBytes || isBytes())
      return "the extent of";
    return llvm::formatv("the number of '{0}' elements in",
                         AsType.getAsString());
  }

  std::string asElementName(bool ForceBytes) const {
    if (ForceBytes || isBytes())
      return "byte";
    return llvm::formatv("'{0}' element", AsType.getAsString());
  }
};

struct Messages {
  std::string Short, Full;
};

enum class BadOffsetKind { Negative, Overflowing, Indeterminate };

constexpr llvm::StringLiteral Adjectives[] = {"a negative", "an overflowing",
                                              "a negative or overflowing"};
inline StringRef asAdjective(BadOffsetKind Problem) {
  return Adjectives[static_cast<int>(Problem)];
}

constexpr llvm::StringLiteral Prepositions[] = {"preceding", "after the end of",
                                                "around"};
inline StringRef asPreposition(BadOffsetKind Problem) {
  return Prepositions[static_cast<int>(Problem)];
}

struct CheckFlags {
  bool CheckUnderflow;
  bool OffsetObviouslyNonnegative;
  bool AcceptPastTheEnd;
};

class BoundsCheckResult {
public:
  enum class Kind { Underflow, Overflow, TaintBug, Paradox, Valid };

private:
  Kind K = Kind::Valid;
  bool AssumedNonNegative = false;
  bool AssumedUpperBound = false;
  const NonLoc Offset;
  std::optional<NonLoc> Extent;
  ProgramStateRef State = nullptr;

  BoundsCheckResult(NonLoc Offs, std::optional<NonLoc> E)
      : Offset(Offs), Extent(E) {}

  void recordNonNegativeAssumption() { AssumedNonNegative = true; }

  void recordUpperBoundAssumption() { AssumedUpperBound = true; }

  void finalize(Kind K_, ProgramStateRef S) {
    K = K_;
    State = S;
  }

public:
  friend BoundsCheckResult checkBounds(ProgramStateRef State, SValBuilder &SVB,
                                       NonLoc Offset,
                                       std::optional<NonLoc> Extent,
                                       CheckFlags Flags);

  bool assumedNonNegative() const { return AssumedNonNegative; }

  bool hasAssumption() const { return AssumedNonNegative || AssumedUpperBound; }

  ProgramStateRef getState() const { return State; }

  Kind getKind() const { return K; }

  std::optional<BadOffsetKind> getBadOffsetKind() const {
    switch (K) {
    case Kind::Underflow:
      return BadOffsetKind::Negative;
    case Kind::Overflow:
      return assumedNonNegative() ? BadOffsetKind::Indeterminate
                                  : BadOffsetKind::Overflowing;
    default:
      return std::nullopt;
    }
  }

  std::string getMessage(PathSensitiveBugReport &BR, StringRef RegName,
                         SizeUnit SU) const;

private:
  /// Return true if information about the value of `Sym` can put constraints
  /// on some symbol which is interesting within the bug report `BR`.
  /// In particular, this returns true when `Sym` is interesting within `BR`;
  /// but it also returns true if `Sym` is an expression that contains integer
  /// constants and a single symbolic operand which is interesting (in `BR`).
  /// We need to use this instead of plain `BR.isInteresting()` because if we
  /// are analyzing code like
  ///   int array[10];
  ///   int f(int arg) {
  ///     return array[arg] && array[arg + 10];
  ///   }
  /// then the byte offsets are `arg * 4` and `(arg + 10) * 4`, which are not
  /// sub-expressions of each other (but `getSimplifiedOffsets` is smart enough
  /// to detect this out of bounds access).
  static bool providesInformationAboutInteresting(SymbolRef Sym,
                                                  PathSensitiveBugReport &BR);
  static bool providesInformationAboutInteresting(SVal SV,
                                                  PathSensitiveBugReport &BR) {
    return providesInformationAboutInteresting(SV.getAsSymbol(), BR);
  }
};

BoundsCheckResult checkBounds(ProgramStateRef State, SValBuilder &SVB,
                              NonLoc Offset, std::optional<NonLoc> Extent,
                              CheckFlags Flags);

std::string getRegionName(const MemSpaceRegion *Space, const SubRegion *Region);

Messages getTaintMsgs(std::string RegName, const char *OffsetName,
                      bool AlsoMentionUnderflow);

Messages getNonTaintMsgs(std::string RegName, SizeUnit SU, NonLoc Offset,
                         std::optional<NonLoc> Extent, BadOffsetKind Problem);

} // namespace ento
} // namespace clang

#endif // LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_BOUNDSCHECKING_H
