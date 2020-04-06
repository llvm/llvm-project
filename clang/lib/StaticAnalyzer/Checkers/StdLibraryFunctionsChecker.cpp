//=== StdLibraryFunctionsChecker.cpp - Model standard functions -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This checker improves modeling of a few simple library functions.
//
// This checker provides a specification format - `Summary' - and
// contains descriptions of some library functions in this format. Each
// specification contains a list of branches for splitting the program state
// upon call, and range constraints on argument and return-value symbols that
// are satisfied on each branch. This spec can be expanded to include more
// items, like external effects of the function.
//
// The main difference between this approach and the body farms technique is
// in more explicit control over how many branches are produced. For example,
// consider standard C function `ispunct(int x)', which returns a non-zero value
// iff `x' is a punctuation character, that is, when `x' is in range
//   ['!', '/']   [':', '@']  U  ['[', '\`']  U  ['{', '~'].
// `Summary' provides only two branches for this function. However,
// any attempt to describe this range with if-statements in the body farm
// would result in many more branches. Because each branch needs to be analyzed
// independently, this significantly reduces performance. Additionally,
// once we consider a branch on which `x' is in range, say, ['!', '/'],
// we assume that such branch is an important separate path through the program,
// which may lead to false positives because considering this particular path
// was not consciously intended, and therefore it might have been unreachable.
//
// This checker uses eval::Call for modeling pure functions (functions without
// side effets), for which their `Summary' is a precise model. This avoids
// unnecessary invalidation passes. Conflicts with other checkers are unlikely
// because if the function has no other effects, other checkers would probably
// never want to improve upon the modeling done by this checker.
//
// Non-pure functions, for which only partial improvement over the default
// behavior is expected, are modeled via check::PostCall, non-intrusively.
//
// The following standard C functions are currently supported:
//
//   fgetc      getline   isdigit   isupper
//   fread      isalnum   isgraph   isxdigit
//   fwrite     isalpha   islower   read
//   getc       isascii   isprint   write
//   getchar    isblank   ispunct
//   getdelim   iscntrl   isspace
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerHelpers.h"

using namespace clang;
using namespace clang::ento;

namespace {
class StdLibraryFunctionsChecker
    : public Checker<check::PreCall, check::PostCall, eval::Call> {
  /// Below is a series of typedefs necessary to define function specs.
  /// We avoid nesting types here because each additional qualifier
  /// would need to be repeated in every function spec.
  struct Summary;

  /// Specify how much the analyzer engine should entrust modeling this function
  /// to us. If he doesn't, he performs additional invalidations.
  enum InvalidationKind { NoEvalCall, EvalCallAsPure };

  // The universal integral type to use in value range descriptions.
  // Unsigned to make sure overflows are well-defined.
  typedef uint64_t RangeInt;

  /// Normally, describes a single range constraint, eg. {{0, 1}, {3, 4}} is
  /// a non-negative integer, which less than 5 and not equal to 2. For
  /// `ComparesToArgument', holds information about how exactly to compare to
  /// the argument.
  typedef std::vector<std::pair<RangeInt, RangeInt>> IntRangeVector;

  /// A reference to an argument or return value by its number.
  /// ArgNo in CallExpr and CallEvent is defined as Unsigned, but
  /// obviously uint32_t should be enough for all practical purposes.
  typedef uint32_t ArgNo;
  static const ArgNo Ret;

  class ValueConstraint;

  // Pointer to the ValueConstraint. We need a copyable, polymorphic and
  // default initialize able type (vector needs that). A raw pointer was good,
  // however, we cannot default initialize that. unique_ptr makes the Summary
  // class non-copyable, therefore not an option. Releasing the copyability
  // requirement would render the initialization of the Summary map infeasible.
  using ValueConstraintPtr = std::shared_ptr<ValueConstraint>;

  /// Polymorphic base class that represents a constraint on a given argument
  /// (or return value) of a function. Derived classes implement different kind
  /// of constraints, e.g range constraints or correlation between two
  /// arguments.
  class ValueConstraint {
  public:
    ValueConstraint(ArgNo ArgN) : ArgN(ArgN) {}
    virtual ~ValueConstraint() {}
    /// Apply the effects of the constraint on the given program state. If null
    /// is returned then the constraint is not feasible.
    virtual ProgramStateRef apply(ProgramStateRef State, const CallEvent &Call,
                                  const Summary &Summary) const = 0;
    virtual ValueConstraintPtr negate() const {
      llvm_unreachable("Not implemented");
    };
    ArgNo getArgNo() const { return ArgN; }

  protected:
    ArgNo ArgN; // Argument to which we apply the constraint.
  };

  /// Given a range, should the argument stay inside or outside this range?
  enum RangeKind { OutOfRange, WithinRange };

  /// Encapsulates a single range on a single symbol within a branch.
  class RangeConstraint : public ValueConstraint {
    RangeKind Kind;      // Kind of range definition.
    IntRangeVector Args; // Polymorphic arguments.

  public:
    RangeConstraint(ArgNo ArgN, RangeKind Kind, const IntRangeVector &Args)
        : ValueConstraint(ArgN), Kind(Kind), Args(Args) {}

    const IntRangeVector &getRanges() const {
      return Args;
    }

  private:
    ProgramStateRef applyAsOutOfRange(ProgramStateRef State,
                                      const CallEvent &Call,
                                      const Summary &Summary) const;
    ProgramStateRef applyAsWithinRange(ProgramStateRef State,
                                       const CallEvent &Call,
                                       const Summary &Summary) const;
  public:
    ProgramStateRef apply(ProgramStateRef State, const CallEvent &Call,
                          const Summary &Summary) const override {
      switch (Kind) {
      case OutOfRange:
        return applyAsOutOfRange(State, Call, Summary);
      case WithinRange:
        return applyAsWithinRange(State, Call, Summary);
      }
      llvm_unreachable("Unknown range kind!");
    }

    ValueConstraintPtr negate() const override {
      RangeConstraint Tmp(*this);
      switch (Kind) {
      case OutOfRange:
        Tmp.Kind = WithinRange;
        break;
      case WithinRange:
        Tmp.Kind = OutOfRange;
        break;
      }
      return std::make_shared<RangeConstraint>(Tmp);
    }
  };

  class ComparisonConstraint : public ValueConstraint {
    BinaryOperator::Opcode Opcode;
    ArgNo OtherArgN;

  public:
    ComparisonConstraint(ArgNo ArgN, BinaryOperator::Opcode Opcode,
                         ArgNo OtherArgN)
        : ValueConstraint(ArgN), Opcode(Opcode), OtherArgN(OtherArgN) {}
    ArgNo getOtherArgNo() const { return OtherArgN; }
    BinaryOperator::Opcode getOpcode() const { return Opcode; }
    ProgramStateRef apply(ProgramStateRef State, const CallEvent &Call,
                          const Summary &Summary) const override;
  };

  class NotNullConstraint : public ValueConstraint {
    using ValueConstraint::ValueConstraint;
    // This variable has a role when we negate the constraint.
    bool CannotBeNull = true;

  public:
    ProgramStateRef apply(ProgramStateRef State, const CallEvent &Call,
                          const Summary &Summary) const override {
      SVal V = getArgSVal(Call, getArgNo());
      if (V.isUndef())
        return State;

      DefinedOrUnknownSVal L = V.castAs<DefinedOrUnknownSVal>();
      if (!L.getAs<Loc>())
        return State;

      return State->assume(L, CannotBeNull);
    }

    ValueConstraintPtr negate() const override {
      NotNullConstraint Tmp(*this);
      Tmp.CannotBeNull = !this->CannotBeNull;
      return std::make_shared<NotNullConstraint>(Tmp);
    }
  };

  /// The complete list of constraints that defines a single branch.
  typedef std::vector<ValueConstraintPtr> ConstraintSet;

  using ArgTypes = std::vector<QualType>;
  using Cases = std::vector<ConstraintSet>;

  /// Includes information about
  ///   * function prototype (which is necessary to
  ///     ensure we're modeling the right function and casting values properly),
  ///   * approach to invalidation,
  ///   * a list of branches - a list of list of ranges -
  ///     A branch represents a path in the exploded graph of a function (which
  ///     is a tree). So, a branch is a series of assumptions. In other words,
  ///     branches represent split states and additional assumptions on top of
  ///     the splitting assumption.
  ///     For example, consider the branches in `isalpha(x)`
  ///       Branch 1)
  ///         x is in range ['A', 'Z'] or in ['a', 'z']
  ///         then the return value is not 0. (I.e. out-of-range [0, 0])
  ///       Branch 2)
  ///         x is out-of-range ['A', 'Z'] and out-of-range ['a', 'z']
  ///         then the return value is 0.
  ///   * a list of argument constraints, that must be true on every branch.
  ///     If these constraints are not satisfied that means a fatal error
  ///     usually resulting in undefined behaviour.
  struct Summary {
    const ArgTypes ArgTys;
    const QualType RetTy;
    const InvalidationKind InvalidationKd;
    Cases CaseConstraints;
    ConstraintSet ArgConstraints;

    Summary(ArgTypes ArgTys, QualType RetTy, InvalidationKind InvalidationKd)
        : ArgTys(ArgTys), RetTy(RetTy), InvalidationKd(InvalidationKd) {}

    Summary &Case(ConstraintSet&& CS) {
      CaseConstraints.push_back(std::move(CS));
      return *this;
    }
    Summary &ArgConstraint(ValueConstraintPtr VC) {
      ArgConstraints.push_back(VC);
      return *this;
    }

  private:
    static void assertTypeSuitableForSummary(QualType T) {
      assert(!T->isVoidType() &&
             "We should have had no significant void types in the spec");
      assert(T.isCanonical() &&
             "We should only have canonical types in the spec");
    }

  public:
    QualType getArgType(ArgNo ArgN) const {
      QualType T = (ArgN == Ret) ? RetTy : ArgTys[ArgN];
      assertTypeSuitableForSummary(T);
      return T;
    }

    /// Try our best to figure out if the call expression is the call of
    /// *the* library function to which this specification applies.
    bool matchesCall(const FunctionDecl *FD) const;
  };

  // The same function (as in, function identifier) may have different
  // summaries assigned to it, with different argument and return value types.
  // We call these "variants" of the function. This can be useful for handling
  // C++ function overloads, and also it can be used when the same function
  // may have different definitions on different platforms.
  typedef std::vector<Summary> Summaries;

  // The map of all functions supported by the checker. It is initialized
  // lazily, and it doesn't change after initialization.
  mutable llvm::StringMap<Summaries> FunctionSummaryMap;

  mutable std::unique_ptr<BugType> BT_InvalidArg;

  // Auxiliary functions to support ArgNo within all structures
  // in a unified manner.
  static QualType getArgType(const Summary &Summary, ArgNo ArgN) {
    return Summary.getArgType(ArgN);
  }
  static QualType getArgType(const CallEvent &Call, ArgNo ArgN) {
    return ArgN == Ret ? Call.getResultType().getCanonicalType()
                       : Call.getArgExpr(ArgN)->getType().getCanonicalType();
  }
  static QualType getArgType(const CallExpr *CE, ArgNo ArgN) {
    return ArgN == Ret ? CE->getType().getCanonicalType()
                       : CE->getArg(ArgN)->getType().getCanonicalType();
  }
  static SVal getArgSVal(const CallEvent &Call, ArgNo ArgN) {
    return ArgN == Ret ? Call.getReturnValue() : Call.getArgSVal(ArgN);
  }

public:
  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  bool evalCall(const CallEvent &Call, CheckerContext &C) const;

  enum CheckKind {
    CK_StdCLibraryFunctionArgsChecker,
    CK_StdCLibraryFunctionsTesterChecker,
    CK_NumCheckKinds
  };
  DefaultBool ChecksEnabled[CK_NumCheckKinds];
  CheckerNameRef CheckNames[CK_NumCheckKinds];

private:
  Optional<Summary> findFunctionSummary(const FunctionDecl *FD,
                                        CheckerContext &C) const;
  Optional<Summary> findFunctionSummary(const CallEvent &Call,
                                        CheckerContext &C) const;

  void initFunctionSummaries(CheckerContext &C) const;

  void reportBug(const CallEvent &Call, ExplodedNode *N,
                 CheckerContext &C) const {
    if (!ChecksEnabled[CK_StdCLibraryFunctionArgsChecker])
      return;
    // TODO Add detailed diagnostic.
    StringRef Msg = "Function argument constraint is not satisfied";
    if (!BT_InvalidArg)
      BT_InvalidArg = std::make_unique<BugType>(
          CheckNames[CK_StdCLibraryFunctionArgsChecker],
          "Unsatisfied argument constraints", categories::LogicError);
    auto R = std::make_unique<PathSensitiveBugReport>(*BT_InvalidArg, Msg, N);
    bugreporter::trackExpressionValue(N, Call.getArgExpr(0), *R);
    C.emitReport(std::move(R));
  }
};

const StdLibraryFunctionsChecker::ArgNo StdLibraryFunctionsChecker::Ret =
    std::numeric_limits<ArgNo>::max();

} // end of anonymous namespace

ProgramStateRef StdLibraryFunctionsChecker::RangeConstraint::applyAsOutOfRange(
    ProgramStateRef State, const CallEvent &Call,
    const Summary &Summary) const {

  ProgramStateManager &Mgr = State->getStateManager();
  SValBuilder &SVB = Mgr.getSValBuilder();
  BasicValueFactory &BVF = SVB.getBasicValueFactory();
  ConstraintManager &CM = Mgr.getConstraintManager();
  QualType T = getArgType(Summary, getArgNo());
  SVal V = getArgSVal(Call, getArgNo());

  if (auto N = V.getAs<NonLoc>()) {
    const IntRangeVector &R = getRanges();
    size_t E = R.size();
    for (size_t I = 0; I != E; ++I) {
      const llvm::APSInt &Min = BVF.getValue(R[I].first, T);
      const llvm::APSInt &Max = BVF.getValue(R[I].second, T);
      assert(Min <= Max);
      State = CM.assumeInclusiveRange(State, *N, Min, Max, false);
      if (!State)
        break;
    }
  }

  return State;
}

ProgramStateRef StdLibraryFunctionsChecker::RangeConstraint::applyAsWithinRange(
    ProgramStateRef State, const CallEvent &Call,
    const Summary &Summary) const {

  ProgramStateManager &Mgr = State->getStateManager();
  SValBuilder &SVB = Mgr.getSValBuilder();
  BasicValueFactory &BVF = SVB.getBasicValueFactory();
  ConstraintManager &CM = Mgr.getConstraintManager();
  QualType T = getArgType(Summary, getArgNo());
  SVal V = getArgSVal(Call, getArgNo());

  // "WithinRange R" is treated as "outside [T_MIN, T_MAX] \ R".
  // We cut off [T_MIN, min(R) - 1] and [max(R) + 1, T_MAX] if necessary,
  // and then cut away all holes in R one by one.
  //
  // E.g. consider a range list R as [A, B] and [C, D]
  // -------+--------+------------------+------------+----------->
  //        A        B                  C            D
  // Then we assume that the value is not in [-inf, A - 1],
  // then not in [D + 1, +inf], then not in [B + 1, C - 1]
  if (auto N = V.getAs<NonLoc>()) {
    const IntRangeVector &R = getRanges();
    size_t E = R.size();

    const llvm::APSInt &MinusInf = BVF.getMinValue(T);
    const llvm::APSInt &PlusInf = BVF.getMaxValue(T);

    const llvm::APSInt &Left = BVF.getValue(R[0].first - 1ULL, T);
    if (Left != PlusInf) {
      assert(MinusInf <= Left);
      State = CM.assumeInclusiveRange(State, *N, MinusInf, Left, false);
      if (!State)
        return nullptr;
    }

    const llvm::APSInt &Right = BVF.getValue(R[E - 1].second + 1ULL, T);
    if (Right != MinusInf) {
      assert(Right <= PlusInf);
      State = CM.assumeInclusiveRange(State, *N, Right, PlusInf, false);
      if (!State)
        return nullptr;
    }

    for (size_t I = 1; I != E; ++I) {
      const llvm::APSInt &Min = BVF.getValue(R[I - 1].second + 1ULL, T);
      const llvm::APSInt &Max = BVF.getValue(R[I].first - 1ULL, T);
      if (Min <= Max) {
        State = CM.assumeInclusiveRange(State, *N, Min, Max, false);
        if (!State)
          return nullptr;
      }
    }
  }

  return State;
}

ProgramStateRef StdLibraryFunctionsChecker::ComparisonConstraint::apply(
    ProgramStateRef State, const CallEvent &Call,
    const Summary &Summary) const {

  ProgramStateManager &Mgr = State->getStateManager();
  SValBuilder &SVB = Mgr.getSValBuilder();
  QualType CondT = SVB.getConditionType();
  QualType T = getArgType(Summary, getArgNo());
  SVal V = getArgSVal(Call, getArgNo());

  BinaryOperator::Opcode Op = getOpcode();
  ArgNo OtherArg = getOtherArgNo();
  SVal OtherV = getArgSVal(Call, OtherArg);
  QualType OtherT = getArgType(Call, OtherArg);
  // Note: we avoid integral promotion for comparison.
  OtherV = SVB.evalCast(OtherV, T, OtherT);
  if (auto CompV = SVB.evalBinOp(State, Op, V, OtherV, CondT)
                       .getAs<DefinedOrUnknownSVal>())
    State = State->assume(*CompV, true);
  return State;
}

void StdLibraryFunctionsChecker::checkPreCall(const CallEvent &Call,
                                              CheckerContext &C) const {
  Optional<Summary> FoundSummary = findFunctionSummary(Call, C);
  if (!FoundSummary)
    return;

  const Summary &Summary = *FoundSummary;
  ProgramStateRef State = C.getState();

  ProgramStateRef NewState = State;
  for (const ValueConstraintPtr& VC : Summary.ArgConstraints) {
    ProgramStateRef SuccessSt = VC->apply(NewState, Call, Summary);
    ProgramStateRef FailureSt = VC->negate()->apply(NewState, Call, Summary);
    // The argument constraint is not satisfied.
    if (FailureSt && !SuccessSt) {
      if (ExplodedNode *N = C.generateErrorNode(NewState))
        reportBug(Call, N, C);
      break;
    } else {
      // We will apply the constraint even if we cannot reason about the
      // argument. This means both SuccessSt and FailureSt can be true. If we
      // weren't applying the constraint that would mean that symbolic
      // execution continues on a code whose behaviour is undefined.
      assert(SuccessSt);
      NewState = SuccessSt;
    }
  }
  if (NewState && NewState != State)
    C.addTransition(NewState);
}

void StdLibraryFunctionsChecker::checkPostCall(const CallEvent &Call,
                                               CheckerContext &C) const {
  Optional<Summary> FoundSummary = findFunctionSummary(Call, C);
  if (!FoundSummary)
    return;

  // Now apply the constraints.
  const Summary &Summary = *FoundSummary;
  ProgramStateRef State = C.getState();

  // Apply case/branch specifications.
  for (const auto &VRS : Summary.CaseConstraints) {
    ProgramStateRef NewState = State;
    for (const auto &VR: VRS) {
      NewState = VR->apply(NewState, Call, Summary);
      if (!NewState)
        break;
    }

    if (NewState && NewState != State)
      C.addTransition(NewState);
  }
}

bool StdLibraryFunctionsChecker::evalCall(const CallEvent &Call,
                                          CheckerContext &C) const {
  Optional<Summary> FoundSummary = findFunctionSummary(Call, C);
  if (!FoundSummary)
    return false;

  const Summary &Summary = *FoundSummary;
  switch (Summary.InvalidationKd) {
  case EvalCallAsPure: {
    ProgramStateRef State = C.getState();
    const LocationContext *LC = C.getLocationContext();
    const auto *CE = cast_or_null<CallExpr>(Call.getOriginExpr());
    SVal V = C.getSValBuilder().conjureSymbolVal(
        CE, LC, CE->getType().getCanonicalType(), C.blockCount());
    State = State->BindExpr(CE, LC, V);
    C.addTransition(State);
    return true;
  }
  case NoEvalCall:
    // Summary tells us to avoid performing eval::Call. The function is possibly
    // evaluated by another checker, or evaluated conservatively.
    return false;
  }
  llvm_unreachable("Unknown invalidation kind!");
}

bool StdLibraryFunctionsChecker::Summary::matchesCall(
    const FunctionDecl *FD) const {
  // Check number of arguments:
  if (FD->param_size() != ArgTys.size())
    return false;

  // Check return type if relevant:
  if (!RetTy.isNull() && RetTy != FD->getReturnType().getCanonicalType())
    return false;

  // Check argument types when relevant:
  for (size_t I = 0, E = ArgTys.size(); I != E; ++I) {
    QualType FormalT = ArgTys[I];
    // Null type marks irrelevant arguments.
    if (FormalT.isNull())
      continue;

    assertTypeSuitableForSummary(FormalT);

    QualType ActualT = FD->getParamDecl(I)->getType().getCanonicalType();
    if (ActualT != FormalT)
      return false;
  }

  return true;
}

Optional<StdLibraryFunctionsChecker::Summary>
StdLibraryFunctionsChecker::findFunctionSummary(const FunctionDecl *FD,
                                                CheckerContext &C) const {
  if (!FD)
    return None;

  initFunctionSummaries(C);

  IdentifierInfo *II = FD->getIdentifier();
  if (!II)
    return None;
  StringRef Name = II->getName();
  if (Name.empty() || !C.isCLibraryFunction(FD, Name))
    return None;

  auto FSMI = FunctionSummaryMap.find(Name);
  if (FSMI == FunctionSummaryMap.end())
    return None;

  // Verify that function signature matches the spec in advance.
  // Otherwise we might be modeling the wrong function.
  // Strict checking is important because we will be conducting
  // very integral-type-sensitive operations on arguments and
  // return values.
  const Summaries &SpecVariants = FSMI->second;
  for (const Summary &Spec : SpecVariants)
    if (Spec.matchesCall(FD))
      return Spec;

  return None;
}

Optional<StdLibraryFunctionsChecker::Summary>
StdLibraryFunctionsChecker::findFunctionSummary(const CallEvent &Call,
                                                CheckerContext &C) const {
  const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(Call.getDecl());
  if (!FD)
    return None;
  return findFunctionSummary(FD, C);
}

void StdLibraryFunctionsChecker::initFunctionSummaries(
    CheckerContext &C) const {
  if (!FunctionSummaryMap.empty())
    return;

  SValBuilder &SVB = C.getSValBuilder();
  BasicValueFactory &BVF = SVB.getBasicValueFactory();
  const ASTContext &ACtx = BVF.getContext();

  // These types are useful for writing specifications quickly,
  // New specifications should probably introduce more types.
  // Some types are hard to obtain from the AST, eg. "ssize_t".
  // In such cases it should be possible to provide multiple variants
  // of function summary for common cases (eg. ssize_t could be int or long
  // or long long, so three summary variants would be enough).
  // Of course, function variants are also useful for C++ overloads.
  const QualType
      Irrelevant{}; // A placeholder, whenever we do not care about the type.
  const QualType IntTy = ACtx.IntTy;
  const QualType LongTy = ACtx.LongTy;
  const QualType LongLongTy = ACtx.LongLongTy;
  const QualType SizeTy = ACtx.getSizeType();
  const QualType VoidPtrTy = ACtx.VoidPtrTy; // void *
  const QualType VoidPtrRestrictTy =
      ACtx.getRestrictType(VoidPtrTy); // void *restrict
  const QualType ConstVoidPtrTy =
      ACtx.getPointerType(ACtx.VoidTy.withConst()); // const void *
  const QualType ConstCharPtrTy =
      ACtx.getPointerType(ACtx.CharTy.withConst()); // const char *
  const QualType ConstVoidPtrRestrictTy =
      ACtx.getRestrictType(ConstVoidPtrTy); // const void *restrict

  const RangeInt IntMax = BVF.getMaxValue(IntTy).getLimitedValue();
  const RangeInt LongMax = BVF.getMaxValue(LongTy).getLimitedValue();
  const RangeInt LongLongMax = BVF.getMaxValue(LongLongTy).getLimitedValue();

  // Set UCharRangeMax to min of int or uchar maximum value.
  // The C standard states that the arguments of functions like isalpha must
  // be representable as an unsigned char. Their type is 'int', so the max
  // value of the argument should be min(UCharMax, IntMax). This just happen
  // to be true for commonly used and well tested instruction set
  // architectures, but not for others.
  const RangeInt UCharRangeMax =
      std::min(BVF.getMaxValue(ACtx.UnsignedCharTy).getLimitedValue(), IntMax);

  // The platform dependent value of EOF.
  // Try our best to parse this from the Preprocessor, otherwise fallback to -1.
  const auto EOFv = [&C]() -> RangeInt {
    if (const llvm::Optional<int> OptInt =
            tryExpandAsInteger("EOF", C.getPreprocessor()))
      return *OptInt;
    return -1;
  }();

  // We are finally ready to define specifications for all supported functions.
  //
  // The signature needs to have the correct number of arguments.
  // However, we insert `Irrelevant' when the type is insignificant.
  //
  // Argument ranges should always cover all variants. If return value
  // is completely unknown, omit it from the respective range set.
  //
  // All types in the spec need to be canonical.
  //
  // Every item in the list of range sets represents a particular
  // execution path the analyzer would need to explore once
  // the call is modeled - a new program state is constructed
  // for every range set, and each range line in the range set
  // corresponds to a specific constraint within this state.
  //
  // Upon comparing to another argument, the other argument is casted
  // to the current argument's type. This avoids proper promotion but
  // seems useful. For example, read() receives size_t argument,
  // and its return value, which is of type ssize_t, cannot be greater
  // than this argument. If we made a promotion, and the size argument
  // is equal to, say, 10, then we'd impose a range of [0, 10] on the
  // return value, however the correct range is [-1, 10].
  //
  // Please update the list of functions in the header after editing!
  //

  // Below are helpers functions to create the summaries.
  auto ArgumentCondition = [](ArgNo ArgN, RangeKind Kind,
                              IntRangeVector Ranges) {
    return std::make_shared<RangeConstraint>(ArgN, Kind, Ranges);
  };
  struct {
    auto operator()(RangeKind Kind, IntRangeVector Ranges) {
      return std::make_shared<RangeConstraint>(Ret, Kind, Ranges);
    }
    auto operator()(BinaryOperator::Opcode Op, ArgNo OtherArgN) {
      return std::make_shared<ComparisonConstraint>(Ret, Op, OtherArgN);
    }
  } ReturnValueCondition;
  auto Range = [](RangeInt b, RangeInt e) {
    return IntRangeVector{std::pair<RangeInt, RangeInt>{b, e}};
  };
  auto SingleValue = [](RangeInt v) {
    return IntRangeVector{std::pair<RangeInt, RangeInt>{v, v}};
  };
  auto LessThanOrEq = BO_LE;
  auto NotNull = [&](ArgNo ArgN) {
    return std::make_shared<NotNullConstraint>(ArgN);
  };

  using RetType = QualType;
  // Templates for summaries that are reused by many functions.
  auto Getc = [&]() {
    return Summary(ArgTypes{Irrelevant}, RetType{IntTy}, NoEvalCall)
        .Case({ReturnValueCondition(WithinRange,
                                    {{EOFv, EOFv}, {0, UCharRangeMax}})});
  };
  auto Read = [&](RetType R, RangeInt Max) {
    return Summary(ArgTypes{Irrelevant, Irrelevant, SizeTy}, RetType{R},
                   NoEvalCall)
        .Case({ReturnValueCondition(LessThanOrEq, ArgNo(2)),
               ReturnValueCondition(WithinRange, Range(-1, Max))});
  };
  auto Fread = [&]() {
    return Summary(ArgTypes{VoidPtrRestrictTy, Irrelevant, SizeTy, Irrelevant},
                   RetType{SizeTy}, NoEvalCall)
        .Case({
            ReturnValueCondition(LessThanOrEq, ArgNo(2)),
        })
        .ArgConstraint(NotNull(ArgNo(0)));
  };
  auto Fwrite = [&]() {
    return Summary(
               ArgTypes{ConstVoidPtrRestrictTy, Irrelevant, SizeTy, Irrelevant},
               RetType{SizeTy}, NoEvalCall)
        .Case({
            ReturnValueCondition(LessThanOrEq, ArgNo(2)),
        })
        .ArgConstraint(NotNull(ArgNo(0)));
  };
  auto Getline = [&](RetType R, RangeInt Max) {
    return Summary(ArgTypes{Irrelevant, Irrelevant, Irrelevant}, RetType{R},
                   NoEvalCall)
        .Case({ReturnValueCondition(WithinRange, {{-1, -1}, {1, Max}})});
  };

  FunctionSummaryMap = {
      // The isascii() family of functions.
      // The behavior is undefined if the value of the argument is not
      // representable as unsigned char or is not equal to EOF. See e.g. C99
      // 7.4.1.2 The isalpha function (p: 181-182).
      {
          "isalnum",
          Summaries{
              Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
                  // Boils down to isupper() or islower() or isdigit().
                  .Case(
                      {ArgumentCondition(0U, WithinRange,
                                         {{'0', '9'}, {'A', 'Z'}, {'a', 'z'}}),
                       ReturnValueCondition(OutOfRange, SingleValue(0))})
                  // The locale-specific range.
                  // No post-condition. We are completely unaware of
                  // locale-specific return values.
                  .Case({ArgumentCondition(0U, WithinRange,
                                           {{128, UCharRangeMax}})})
                  .Case({ArgumentCondition(0U, OutOfRange,
                                           {{'0', '9'},
                                            {'A', 'Z'},
                                            {'a', 'z'},
                                            {128, UCharRangeMax}}),
                         ReturnValueCondition(WithinRange, SingleValue(0))})
                  .ArgConstraint(ArgumentCondition(
                      0U, WithinRange, {{EOFv, EOFv}, {0, UCharRangeMax}}))},
      },
      {
          "isalpha",
          Summaries{
              Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
                  .Case({ArgumentCondition(0U, WithinRange,
                                           {{'A', 'Z'}, {'a', 'z'}}),
                         ReturnValueCondition(OutOfRange, SingleValue(0))})
                  // The locale-specific range.
                  .Case({ArgumentCondition(0U, WithinRange,
                                           {{128, UCharRangeMax}})})
                  .Case({ArgumentCondition(
                             0U, OutOfRange,
                             {{'A', 'Z'}, {'a', 'z'}, {128, UCharRangeMax}}),
                         ReturnValueCondition(WithinRange, SingleValue(0))})},
      },
      {
          "isascii",
          Summaries{
              Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
                  .Case({ArgumentCondition(0U, WithinRange, Range(0, 127)),
                         ReturnValueCondition(OutOfRange, SingleValue(0))})
                  .Case({ArgumentCondition(0U, OutOfRange, Range(0, 127)),
                         ReturnValueCondition(WithinRange, SingleValue(0))})},
      },
      {
          "isblank",
          Summaries{
              Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
                  .Case({ArgumentCondition(0U, WithinRange,
                                           {{'\t', '\t'}, {' ', ' '}}),
                         ReturnValueCondition(OutOfRange, SingleValue(0))})
                  .Case({ArgumentCondition(0U, OutOfRange,
                                           {{'\t', '\t'}, {' ', ' '}}),
                         ReturnValueCondition(WithinRange, SingleValue(0))})},
      },
      {
          "iscntrl",
          Summaries{
              Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
                  .Case({ArgumentCondition(0U, WithinRange,
                                           {{0, 32}, {127, 127}}),
                         ReturnValueCondition(OutOfRange, SingleValue(0))})
                  .Case(
                      {ArgumentCondition(0U, OutOfRange, {{0, 32}, {127, 127}}),
                       ReturnValueCondition(WithinRange, SingleValue(0))})},
      },
      {
          "isdigit",
          Summaries{
              Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
                  .Case({ArgumentCondition(0U, WithinRange, Range('0', '9')),
                         ReturnValueCondition(OutOfRange, SingleValue(0))})
                  .Case({ArgumentCondition(0U, OutOfRange, Range('0', '9')),
                         ReturnValueCondition(WithinRange, SingleValue(0))})},
      },
      {
          "isgraph",
          Summaries{
              Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
                  .Case({ArgumentCondition(0U, WithinRange, Range(33, 126)),
                         ReturnValueCondition(OutOfRange, SingleValue(0))})
                  .Case({ArgumentCondition(0U, OutOfRange, Range(33, 126)),
                         ReturnValueCondition(WithinRange, SingleValue(0))})},
      },
      {
          "islower",
          Summaries{
              Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
                  // Is certainly lowercase.
                  .Case({ArgumentCondition(0U, WithinRange, Range('a', 'z')),
                         ReturnValueCondition(OutOfRange, SingleValue(0))})
                  // Is ascii but not lowercase.
                  .Case({ArgumentCondition(0U, WithinRange, Range(0, 127)),
                         ArgumentCondition(0U, OutOfRange, Range('a', 'z')),
                         ReturnValueCondition(WithinRange, SingleValue(0))})
                  // The locale-specific range.
                  .Case({ArgumentCondition(0U, WithinRange,
                                           {{128, UCharRangeMax}})})
                  // Is not an unsigned char.
                  .Case({ArgumentCondition(0U, OutOfRange,
                                           Range(0, UCharRangeMax)),
                         ReturnValueCondition(WithinRange, SingleValue(0))})},
      },
      {
          "isprint",
          Summaries{
              Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
                  .Case({ArgumentCondition(0U, WithinRange, Range(32, 126)),
                         ReturnValueCondition(OutOfRange, SingleValue(0))})
                  .Case({ArgumentCondition(0U, OutOfRange, Range(32, 126)),
                         ReturnValueCondition(WithinRange, SingleValue(0))})},
      },
      {
          "ispunct",
          Summaries{
              Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
                  .Case({ArgumentCondition(
                             0U, WithinRange,
                             {{'!', '/'}, {':', '@'}, {'[', '`'}, {'{', '~'}}),
                         ReturnValueCondition(OutOfRange, SingleValue(0))})
                  .Case({ArgumentCondition(
                             0U, OutOfRange,
                             {{'!', '/'}, {':', '@'}, {'[', '`'}, {'{', '~'}}),
                         ReturnValueCondition(WithinRange, SingleValue(0))})},
      },
      {
          "isspace",
          Summaries{
              Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
                  // Space, '\f', '\n', '\r', '\t', '\v'.
                  .Case({ArgumentCondition(0U, WithinRange,
                                           {{9, 13}, {' ', ' '}}),
                         ReturnValueCondition(OutOfRange, SingleValue(0))})
                  // The locale-specific range.
                  .Case({ArgumentCondition(0U, WithinRange,
                                           {{128, UCharRangeMax}})})
                  .Case({ArgumentCondition(
                             0U, OutOfRange,
                             {{9, 13}, {' ', ' '}, {128, UCharRangeMax}}),
                         ReturnValueCondition(WithinRange, SingleValue(0))})},
      },
      {
          "isupper",
          Summaries{
              Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
                  // Is certainly uppercase.
                  .Case({ArgumentCondition(0U, WithinRange, Range('A', 'Z')),
                         ReturnValueCondition(OutOfRange, SingleValue(0))})
                  // The locale-specific range.
                  .Case({ArgumentCondition(0U, WithinRange,
                                           {{128, UCharRangeMax}})})
                  // Other.
                  .Case({ArgumentCondition(0U, OutOfRange,
                                           {{'A', 'Z'}, {128, UCharRangeMax}}),
                         ReturnValueCondition(WithinRange, SingleValue(0))})},
      },
      {
          "isxdigit",
          Summaries{
              Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
                  .Case(
                      {ArgumentCondition(0U, WithinRange,
                                         {{'0', '9'}, {'A', 'F'}, {'a', 'f'}}),
                       ReturnValueCondition(OutOfRange, SingleValue(0))})
                  .Case(
                      {ArgumentCondition(0U, OutOfRange,
                                         {{'0', '9'}, {'A', 'F'}, {'a', 'f'}}),
                       ReturnValueCondition(WithinRange, SingleValue(0))})},
      },

      // The getc() family of functions that returns either a char or an EOF.
      {"getc", Summaries{Getc()}},
      {"fgetc", Summaries{Getc()}},
      {"getchar",
       Summaries{Summary(ArgTypes{}, RetType{IntTy}, NoEvalCall)
                     .Case({ReturnValueCondition(
                         WithinRange, {{EOFv, EOFv}, {0, UCharRangeMax}})})}},

      // read()-like functions that never return more than buffer size.
      // We are not sure how ssize_t is defined on every platform, so we
      // provide three variants that should cover common cases.
      {"read", Summaries{Read(IntTy, IntMax), Read(LongTy, LongMax),
                         Read(LongLongTy, LongLongMax)}},
      {"write", Summaries{Read(IntTy, IntMax), Read(LongTy, LongMax),
                          Read(LongLongTy, LongLongMax)}},
      {"fread", Summaries{Fread()}},
      {"fwrite", Summaries{Fwrite()}},
      // getline()-like functions either fail or read at least the delimiter.
      {"getline", Summaries{Getline(IntTy, IntMax), Getline(LongTy, LongMax),
                            Getline(LongLongTy, LongLongMax)}},
      {"getdelim", Summaries{Getline(IntTy, IntMax), Getline(LongTy, LongMax),
                             Getline(LongLongTy, LongLongMax)}},
  };

  // Functions for testing.
  if (ChecksEnabled[CK_StdCLibraryFunctionsTesterChecker]) {
    llvm::StringMap<Summaries> TestFunctionSummaryMap = {
        {"__two_constrained_args",
         Summaries{
             Summary(ArgTypes{IntTy, IntTy}, RetType{IntTy}, EvalCallAsPure)
                 .ArgConstraint(
                     ArgumentCondition(0U, WithinRange, SingleValue(1)))
                 .ArgConstraint(
                     ArgumentCondition(1U, WithinRange, SingleValue(1)))}},
        {"__arg_constrained_twice",
         Summaries{Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
                       .ArgConstraint(
                           ArgumentCondition(0U, OutOfRange, SingleValue(1)))
                       .ArgConstraint(
                           ArgumentCondition(0U, OutOfRange, SingleValue(2)))}},
        {"__defaultparam", Summaries{Summary(ArgTypes{Irrelevant, IntTy},
                                             RetType{IntTy}, EvalCallAsPure)
                                         .ArgConstraint(NotNull(ArgNo(0)))}},
        {"__variadic", Summaries{Summary(ArgTypes{VoidPtrTy, ConstCharPtrTy},
                                         RetType{IntTy}, EvalCallAsPure)
                                     .ArgConstraint(NotNull(ArgNo(0)))
                                     .ArgConstraint(NotNull(ArgNo(1)))}}};
    for (auto &E : TestFunctionSummaryMap) {
      auto InsertRes =
          FunctionSummaryMap.insert({std::string(E.getKey()), E.getValue()});
      assert(InsertRes.second &&
             "Test functions must not clash with modeled functions");
      (void)InsertRes;
    }
  }
}

void ento::registerStdCLibraryFunctionsChecker(CheckerManager &mgr) {
  mgr.registerChecker<StdLibraryFunctionsChecker>();
}

bool ento::shouldRegisterStdCLibraryFunctionsChecker(const CheckerManager &mgr) {
  return true;
}

#define REGISTER_CHECKER(name)                                                 \
  void ento::register##name(CheckerManager &mgr) {                             \
    StdLibraryFunctionsChecker *checker =                                      \
        mgr.getChecker<StdLibraryFunctionsChecker>();                          \
    checker->ChecksEnabled[StdLibraryFunctionsChecker::CK_##name] = true;      \
    checker->CheckNames[StdLibraryFunctionsChecker::CK_##name] =               \
        mgr.getCurrentCheckerName();                                           \
  }                                                                            \
                                                                               \
  bool ento::shouldRegister##name(const CheckerManager &mgr) { return true; }

REGISTER_CHECKER(StdCLibraryFunctionArgsChecker)
REGISTER_CHECKER(StdCLibraryFunctionsTesterChecker)
