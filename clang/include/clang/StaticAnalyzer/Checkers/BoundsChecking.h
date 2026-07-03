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
//  This fulfills a similar role as `ProgramState::assumeInBound`, but uses
//  more accurate logic and heuristic workarounds to account for the quirks of
//  signed/unsigned conversions and the lack of cast modeling in the analyzer.
//
//  As of now, this logic only supports the needs of `security.ArrayBound`, but
//  in the future it will be generalized and applied in all checkers that
//  perform bounds checking (to bring them out of `alpha` stage).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CHECKERS_BOUNDSCHECKING_H
#define LLVM_CLANG_STATICANALYZER_CHECKERS_BOUNDSCHECKING_H
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "llvm/Support/FormatVariadic.h"
#include <optional>

namespace clang::ento::bounds {

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
    return ASE ? SizeUnit(ASE->getType(), C.getASTContext()) : bytes();
  }

  /// Return the element type that is "natural" for reporting out-of-bounds
  /// memory access to 'Location'.
  /// FIXME: It is unfortunate that this heuristic differs from the heuristic
  /// used for reporting assumption (`SizeUnit::forExpr`).
  static SizeUnit forSVal(SVal Location, const ASTContext &ACtx) {
    const auto *TVR = Location.getAsRegion()->getAs<TypedValueRegion>();
    return TVR ? SizeUnit(TVR->getValueType(), ACtx) : bytes();
  }

  int64_t asCharUnits() const { return AsCharUnits; }

  std::string asExtentDesc() const {
    if (isBytes())
      return "the extent of";
    return llvm::formatv("the number of '{0}' elements in",
                         AsType.getAsString());
  }

  std::string asElementName() const {
    if (isBytes())
      return "byte";
    return llvm::formatv("'{0}' element", AsType.getAsString());
  }

  std::string getOffsetName() const {
    return isBytes() ? "byte offset" : "index";
  }

  /// Try to divide `Val1` and `Val2` (in place) by `this->asCharUnits()` and
  /// return true if it can be performed without remainder. The values \p Val1
  /// and \p Val2 may be nullopt and in that case the corresponding division is
  /// considered to be successful.
  bool tryConvertValuesFromBytes(std::optional<int64_t> &Val1,
                                 std::optional<int64_t> &Val2) const;
};

struct Messages {
  std::string Short;
  std::string Full;
};

struct CheckFlags {
  unsigned CheckUnderflow : 1;
  unsigned OffsetObviouslyNonnegative : 1;
  unsigned AcceptPastTheEnd : 1;
};

class CheckResult;

class CheckInfo {
protected:
  // Changed to true if we see that underflow was not ruled out by the previous
  // knowledge about the offset.
  bool UnderflowFeasible = false;
  // The offset from the beginning of the accessed region in CharUnits.
  const NonLoc Offset;
  // The extent of the accessed region in CharUnits; or `nullopt` if the extent
  // is irrelevant because overflow was ruled out by previous knowledge about
  // the offset and extent.
  std::optional<NonLoc> Extent = std::nullopt;

public:
  bool hasAssumption() const { return UnderflowFeasible || Extent; }

  friend CheckResult checkBounds(ProgramStateRef State, SValBuilder &SVB,
                                 NonLoc Offset, std::optional<NonLoc> Extent,
                                 CheckFlags Flags);

protected:
  CheckInfo(NonLoc Offs) : Offset(Offs) {}

  void recordUnderflowFeasible() { UnderflowFeasible = true; }
  void recordRelevantExtent(NonLoc E) { Extent = E; }
  void discardExtentInformation() { Extent = std::nullopt; }
};

class CheckResult : public CheckInfo {
public:
  enum class Kind { Valid, Invalid, TaintBug, CorruptedState };

private:
  Kind K = Kind::Valid;
  ProgramStateRef State = nullptr;

  CheckResult(CheckInfo CI, Kind K_, ProgramStateRef S)
      : CheckInfo(CI), K(K_), State(S) {}

public:
  friend CheckResult checkBounds(ProgramStateRef State, SValBuilder &SVB,
                                 NonLoc Offset, std::optional<NonLoc> Extent,
                                 CheckFlags Flags);

  ProgramStateRef getState() const { return State; }

  Kind getKind() const { return K; }

  Messages getTaintMsgs(std::string RegName, const char *OffsetName);

  Messages getNonTaintMsgs(std::string RegName, SizeUnit SU);

  std::string getAssumptionMsg(PathSensitiveBugReport &BR, StringRef RegName,
                               SizeUnit SU) const;

  using InterestingVec = SmallVector<NonLoc, 2>;
  InterestingVec getInteresting() const {
    InterestingVec Res = {Offset};
    if (Extent)
      Res.push_back(*Extent);
    return Res;
  }

protected:
  /// Return true if information about the symbol behind `SV` can constrain
  /// some symbol which is interesting within the bug report `BR`.
  /// In particular, this returns true when `SV` is interesting within `BR`;
  /// but it also returns true if `SV` is an expression that contains integer
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
  static bool providesInformationAboutInteresting(SVal SV,
                                                  PathSensitiveBugReport &BR);
  const char *offsetAdjective() const {
    return UnderflowFeasible
               ? (Extent ? "a negative or overflowing" : "a negative")
               : (Extent ? "an overflowing" : "a valid");
  }
  const char *offsetPreposition() const {
    return UnderflowFeasible ? (Extent ? "around" : "preceding")
                             : (Extent ? "after the end of" : "in");
  }
};

CheckResult checkBounds(ProgramStateRef State, SValBuilder &SVB, NonLoc Offset,
                        std::optional<NonLoc> Extent, CheckFlags Flags);

// FIXME: This utility probably should become a method of `MemRegion`.
std::string getRegionName(const MemSpaceRegion *Space, const SubRegion *Region);

} // namespace clang::ento::bounds

#endif // LLVM_CLANG_STATICANALYZER_CHECKERS_BOUNDSCHECKING_H
