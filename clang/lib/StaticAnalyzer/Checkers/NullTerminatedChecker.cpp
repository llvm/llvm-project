//===- NullTerminatedChecker.cpp - Check null_terminated params -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This defines NullTerminatedChecker, which checks for arguments treated as
// buffers that are expected to be null-terminated (ends with a zero-valued
// element). A constant-size array is considered null-terminated if any of its
// elements may be zero on the current path.
//
// Parameters are marked as expecting null-terminated buffers using:
//   __attribute__((annotate("null_terminated")))
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Attr.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace clang;
using namespace ento;

namespace {
class NullTerminatedChecker : public Checker<check::PreCall> {
public:
  // TODO: region-store-max-binding-fanout defaults to 128, meaning a single
  // bind only covers that many elements. The 1024 option here is only truly
  // respected when the array is built by separate bind operations, e.g.,
  // the case of straight-line writes:
  //
  // int a[500];
  // a[0] = val;
  // a[1] = val;
  // ...
  // a[499] = val;
  int MaxArraySize = 1024;

  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;

private:
  const BugType BT{this, "Array not null-terminated", "API"};

  /// Return true if the parameter has annotate("null_terminated").
  static bool isNullTerminatedParam(const ParmVarDecl *Param);

  /// Return true if any element in [0, \p ArraySize) can be zero.
  bool mayContainZeroElement(ProgramStateRef State, SValBuilder &SVB,
                             QualType EltTy, uint64_t ArraySize,
                             const TypedValueRegion *Arr) const;
};

/// Return true if we can't prove \p Val is non-zero on the current path.
bool mayBeZero(ProgramStateRef State, SVal Val) {
  // Unknown or undefined: can't reason either way.
  auto DV = Val.getAs<DefinedSVal>();
  if (!DV)
    return true;

  // Try the fast lookup first (much cheaper than assuming condition for
  // concrete values).
  ConditionTruthVal IsZero = State->isNull(*DV);
  if (IsZero.isConstrainedFalse())
    return false;
  if (IsZero.isConstrainedTrue())
    return true;

  // For an atomic symbol, the solver won't do any better than the preceding
  // check, so we cannot prove anything further (hence it may be zero).
  SymbolRef Sym = DV->getAsSymbol(/*IncludeBaseRegion=*/true);
  if (Sym && isa<SymbolData>(Sym))
    return true;

  // Worst case: ask the solver if the value can be zero.
  assert(!DV->isConstant() &&
         "Constants should have been handled by the fast path");
  return static_cast<bool>(State->assume(*DV, /*Assumption=*/false));
}

/// Load element \p Idx of the array \p Arr from the store. This is relatively
/// expensive since we are essentially asking the analyzer to work out a value
/// (rather than just read a preexisting binding), so this should be a last
/// resort call.
SVal loadElement(ProgramStateRef State, SValBuilder &SVB, QualType EltTy,
                 uint64_t Idx, const TypedValueRegion *Arr) {
  SVal EltAddr =
      State->getLValue(EltTy, SVB.makeArrayIndex(Idx), loc::MemRegionVal(Arr));
  if (auto EltLoc = EltAddr.getAs<Loc>())
    return State->getSVal(*EltLoc);
  return UnknownVal();
}

/// Map the direct bindings of a memory cluster onto the elements of one array,
/// looking for an element that may be zero.
class ElementBindingScanner : public StoreManager::ClusterBindingsHandler {
  ProgramStateRef State;
  ASTContext &Ctx;
  /// The region the offset of the array is relative to.
  const MemRegion *OffsetRegion;
  /// Offset of the array within \c OffsetRegion, in bits.
  uint64_t ArrOffset;
  uint64_t EltBits;

  /// Elements that have a direct binding.
  llvm::SmallBitVector Covered;
  /// A binding overlaps an element without us knowing which part of it.
  bool Imprecise = false;
  /// An element that has a direct binding may be zero.
  bool FoundPossibleZero = false;

public:
  ElementBindingScanner(ProgramStateRef State, ASTContext &Ctx,
                        RegionOffset ArrOffset, uint64_t EltBits,
                        uint64_t ArraySize)
      : State(State), Ctx(Ctx), OffsetRegion(ArrOffset.getRegion()),
        ArrOffset(ArrOffset.getOffset()), EltBits(EltBits), Covered(ArraySize) {
  }

  bool foundPossibleZero() const { return FoundPossibleZero; }
  bool isImprecise() const { return Imprecise; }
  bool hasElementWithoutBinding() const { return !Covered.all(); }
  bool hasBinding(uint64_t Idx) const { return Covered[Idx]; }

  /// Check a binding to see if we can reason about elements in the array.
  /// Return true if we should keep checking/iterating over the rest of the
  /// bindings in the cluster.
  bool handleBinding(StoreManager &, Store, const MemRegion *Region,
                     std::optional<uint64_t> BitOffset,
                     StoreManager::BindingKind Kind, SVal Val) override {
    // Skip default bindings: we can't tell which elements this applies to,
    // which will end up getting loaded later on, so safe to skip.
    if (Kind == StoreManager::BindingKind::Default)
      return true;

    // Skip symbolic offsets: we can't map the location to an index. But the
    // write would drop existing overlapping concrete bindings anyway, which
    // would then load as unknown, so skipping is sound. Bailing entirely
    // (returning false) would skip bindings for sibling objects (e.g., writes
    // to `s.b[i]` should not make `s.a` unknown), so we skip.
    if (!BitOffset)
      return true;

    // Skip bindings not measured from the array's own offset because the
    // offsets aren't comparable.
    if (Region != OffsetRegion)
      return true;

    // The binding belongs to another object within the cluster, so there's
    // nothing to learn here.
    if (*BitOffset < ArrOffset)
      return true; // Precedes the array
    uint64_t Rel = *BitOffset - ArrOffset;
    uint64_t Idx = Rel / EltBits;
    if (Idx >= Covered.size())
      return true; // Follows the array

    // Bail when the offset is either not element-aligned, or when the width
    // doesn't match element size (e.g., writing a char into an int array on
    // x86).
    if (Rel % EltBits != 0 || getBindingWidth(Val) != EltBits) {
      Imprecise = true;
      return false;
    }

    Covered.set(Idx);
    if (mayBeZero(State, Val)) {
      FoundPossibleZero = true;
      return false;
    }
    return true;
  }

private:
  /// Return the number of bits the value of a binding occupies, if known.
  std::optional<uint64_t> getBindingWidth(SVal Val) const {
    QualType T = Val.getType(Ctx);
    if (T.isNull() || T->isFunctionType() || T->isIncompleteType())
      return std::nullopt;
    return Ctx.getTypeSize(T);
  }
};
} // namespace

bool NullTerminatedChecker::isNullTerminatedParam(const ParmVarDecl *Param) {
  return llvm::any_of(Param->specific_attrs<AnnotateAttr>(),
                      [](const AnnotateAttr *Ann) {
                        return Ann->getAnnotation() == "null_terminated";
                      });
}

bool NullTerminatedChecker::mayContainZeroElement(
    ProgramStateRef State, SValBuilder &SVB, QualType EltTy, uint64_t ArraySize,
    const TypedValueRegion *Arr) const {
  ASTContext &Ctx = State->getStateManager().getContext();

  // Bindings are keyed by an offset from the base region of the cluster, so we
  // need the offset of the array itself to map them onto its elements.
  RegionOffset Offset = Arr->getAsOffset();
  if (!Offset.getRegion() || Offset.hasSymbolicOffset())
    return true;
  uint64_t EltBits = Ctx.getTypeSize(EltTy);
  if (EltBits == 0)
    return true;

  ElementBindingScanner Scanner(State, Ctx, Offset, EltBits, ArraySize);
  State->getStateManager().getStoreManager().iterClusterBindings(
      State->getStore(), Offset.getRegion()->getBaseRegion(), Scanner);

  if (Scanner.foundPossibleZero() || Scanner.isImprecise())
    return true;
  if (!Scanner.hasElementWithoutBinding())
    return false;

  // The remaining elements are either covered by a default binding or
  // uninitialized. Iterate backwards, starting at the end of the array, since
  // terminators are usually closer to the end.
  for (uint64_t I = ArraySize; I-- > 0;) {
    if (Scanner.hasBinding(I))
      continue;
    if (mayBeZero(State, loadElement(State, SVB, EltTy, I, Arr)))
      return true;
  }
  return false;
}

void NullTerminatedChecker::checkPreCall(const CallEvent &Call,
                                         CheckerContext &C) const {
  const auto *FD = dyn_cast_or_null<FunctionDecl>(Call.getDecl());
  if (!FD)
    return;

  ProgramStateRef State = C.getState();
  SValBuilder &SVB = C.getSValBuilder();
  ASTContext &Ctx = C.getASTContext();

  unsigned NumParams = FD->getNumParams();
  unsigned NumArgs = Call.getNumArgs();

  // The call to min handles the case when |NumParams| != |NumArgs|.
  for (unsigned I = 0, N = std::min(NumParams, NumArgs); I < N; ++I) {
    const ParmVarDecl *Param = FD->getParamDecl(I);
    if (!isNullTerminatedParam(Param))
      continue;

    SVal ArgVal = Call.getArgSVal(I);
    const MemRegion *R = ArgVal.getAsRegion();
    if (!R)
      continue;

    // Strip ElementRegion wrappers (array-to-pointer decay produces
    // &Element{Array, 0}).
    R = R->StripCasts();
    if (const auto *ER = dyn_cast<ElementRegion>(R))
      R = ER->getSuperRegion();

    const auto *TVR = dyn_cast<TypedValueRegion>(R);
    if (!TVR)
      continue;

    // Constant-size array (skips C99 FAMs).
    const auto *CAT = Ctx.getAsConstantArrayType(TVR->getValueType());
    if (!CAT)
      continue; // TODO: Handle VLAs.

    uint64_t ArraySize = CAT->getSize().getZExtValue();

    // A zero-length array cannot hold a terminator.
    if (ArraySize == 0 || ArraySize > static_cast<uint64_t>(MaxArraySize))
      continue;

    // The analyzer can't reason about pre-C99 FAMs. The -fstrict-flex-arrays
    // level says what kind of trailing array should be considered a FAM, which
    // we should honor, but at its lowest (default) level, the semantics are:
    // "consider any trailing array to be a FAM," which would miss cases like:
    // `struct {int n; int sigs[3]; };` so we should clamp to at least level 1.
    using FAMKind = LangOptions::StrictFlexArraysLevelKind;
    FAMKind StrictFlexArraysLevel =
        std::max(Ctx.getLangOpts().getStrictFlexArraysLevel(),
                 FAMKind::OneZeroOrIncomplete);
    const FieldDecl *Field = nullptr;
    if (const auto *FR = dyn_cast<FieldRegion>(TVR))
      Field = FR->getDecl();
    if (Decl::isFlexibleArrayMemberLike(
            Ctx, Field, TVR->getValueType(), StrictFlexArraysLevel,
            /*IgnoreTemplateOrMacroSubstitution=*/true))
      continue;

    if (mayContainZeroElement(State, SVB, CAT->getElementType(), ArraySize,
                              TVR))
      continue;

    if (ExplodedNode *N = C.generateNonFatalErrorNode(State)) {
      SmallString<128> Msg;
      llvm::raw_svector_ostream OS(Msg);
      OS << "array argument is not null-terminated; parameter "
         << Param->getName() << " expects a null-terminated array";
      auto Report = std::make_unique<PathSensitiveBugReport>(BT, Msg, N);
      Report->addRange(Call.getArgSourceRange(I));
      if (const Expr *ArgE = Call.getArgExpr(I))
        bugreporter::trackExpressionValue(N, ArgE, *Report);
      C.emitReport(std::move(Report));
    }
  }
}

void ento::registerNullTerminatedChecker(CheckerManager &Mgr) {
  auto *Checker = Mgr.registerChecker<NullTerminatedChecker>();
  Checker->MaxArraySize =
      Mgr.getAnalyzerOptions().getCheckerIntegerOption(Checker, "MaxArraySize");
  if (Checker->MaxArraySize < 0) {
    Mgr.reportInvalidCheckerOptionValue(Checker, "MaxArraySize",
                                        "a non-negative value");
    Checker->MaxArraySize = 0;
  }
}

bool ento::shouldRegisterNullTerminatedChecker(const CheckerManager &Mgr) {
  return true;
}
