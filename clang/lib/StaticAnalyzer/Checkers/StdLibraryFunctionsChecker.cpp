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
//   fgetc      getline   isdigit   isupper     toascii
//   fread      isalnum   isgraph   isxdigit
//   fwrite     isalpha   islower   read
//   getc       isascii   isprint   write
//   getchar    isblank   ispunct   toupper
//   getdelim   iscntrl   isspace   tolower
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerHelpers.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/DynamicSize.h"

using namespace clang;
using namespace clang::ento;

namespace {
class StdLibraryFunctionsChecker
    : public Checker<check::PreCall, check::PostCall, eval::Call> {

  class Summary;

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
                                  const Summary &Summary,
                                  CheckerContext &C) const = 0;
    virtual ValueConstraintPtr negate() const {
      llvm_unreachable("Not implemented");
    };

    // Check whether the constraint is malformed or not. It is malformed if the
    // specified argument has a mismatch with the given FunctionDecl (e.g. the
    // arg number is out-of-range of the function's argument list).
    bool checkValidity(const FunctionDecl *FD) const {
      const bool ValidArg = ArgN == Ret || ArgN < FD->getNumParams();
      assert(ValidArg && "Arg out of range!");
      if (!ValidArg)
        return false;
      // Subclasses may further refine the validation.
      return checkSpecificValidity(FD);
    }
    ArgNo getArgNo() const { return ArgN; }

  protected:
    ArgNo ArgN; // Argument to which we apply the constraint.

    /// Do polymorphic sanity check on the constraint.
    virtual bool checkSpecificValidity(const FunctionDecl *FD) const {
      return true;
    }
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

    const IntRangeVector &getRanges() const { return Args; }

  private:
    ProgramStateRef applyAsOutOfRange(ProgramStateRef State,
                                      const CallEvent &Call,
                                      const Summary &Summary) const;
    ProgramStateRef applyAsWithinRange(ProgramStateRef State,
                                       const CallEvent &Call,
                                       const Summary &Summary) const;

  public:
    ProgramStateRef apply(ProgramStateRef State, const CallEvent &Call,
                          const Summary &Summary,
                          CheckerContext &C) const override {
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

    bool checkSpecificValidity(const FunctionDecl *FD) const override {
      const bool ValidArg =
          getArgType(FD, ArgN)->isIntegralType(FD->getASTContext());
      assert(ValidArg &&
             "This constraint should be applied on an integral type");
      return ValidArg;
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
                          const Summary &Summary,
                          CheckerContext &C) const override;
  };

  class NotNullConstraint : public ValueConstraint {
    using ValueConstraint::ValueConstraint;
    // This variable has a role when we negate the constraint.
    bool CannotBeNull = true;

  public:
    ProgramStateRef apply(ProgramStateRef State, const CallEvent &Call,
                          const Summary &Summary,
                          CheckerContext &C) const override {
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

    bool checkSpecificValidity(const FunctionDecl *FD) const override {
      const bool ValidArg = getArgType(FD, ArgN)->isPointerType();
      assert(ValidArg &&
             "This constraint should be applied only on a pointer type");
      return ValidArg;
    }
  };

  // Represents a buffer argument with an additional size argument.
  // E.g. the first two arguments here:
  //   ctime_s(char *buffer, rsize_t bufsz, const time_t *time);
  // Another example:
  //   size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
  //   // Here, ptr is the buffer, and its minimum size is `size * nmemb`.
  class BufferSizeConstraint : public ValueConstraint {
    // The argument which holds the size of the buffer.
    ArgNo SizeArgN;
    // The argument which is a multiplier to size. This is set in case of
    // `fread` like functions where the size is computed as a multiplication of
    // two arguments.
    llvm::Optional<ArgNo> SizeMultiplierArgN;
    // The operator we use in apply. This is negated in negate().
    BinaryOperator::Opcode Op = BO_LE;

  public:
    BufferSizeConstraint(ArgNo Buffer, ArgNo BufSize)
        : ValueConstraint(Buffer), SizeArgN(BufSize) {}

    BufferSizeConstraint(ArgNo Buffer, ArgNo BufSize, ArgNo BufSizeMultiplier)
        : ValueConstraint(Buffer), SizeArgN(BufSize),
          SizeMultiplierArgN(BufSizeMultiplier) {}

    ProgramStateRef apply(ProgramStateRef State, const CallEvent &Call,
                          const Summary &Summary,
                          CheckerContext &C) const override {
      SValBuilder &SvalBuilder = C.getSValBuilder();
      // The buffer argument.
      SVal BufV = getArgSVal(Call, getArgNo());
      // The size argument.
      SVal SizeV = getArgSVal(Call, SizeArgN);
      // Multiply with another argument if given.
      if (SizeMultiplierArgN) {
        SVal SizeMulV = getArgSVal(Call, *SizeMultiplierArgN);
        SizeV = SvalBuilder.evalBinOp(State, BO_Mul, SizeV, SizeMulV,
                                      Summary.getArgType(SizeArgN));
      }
      // The dynamic size of the buffer argument, got from the analyzer engine.
      SVal BufDynSize = getDynamicSizeWithOffset(State, BufV);

      SVal Feasible = SvalBuilder.evalBinOp(State, Op, SizeV, BufDynSize,
                                            SvalBuilder.getContext().BoolTy);
      if (auto F = Feasible.getAs<DefinedOrUnknownSVal>())
        return State->assume(*F, true);

      // We can get here only if the size argument or the dynamic size is
      // undefined. But the dynamic size should never be undefined, only
      // unknown. So, here, the size of the argument is undefined, i.e. we
      // cannot apply the constraint. Actually, other checkers like
      // CallAndMessage should catch this situation earlier, because we call a
      // function with an uninitialized argument.
      llvm_unreachable("Size argument or the dynamic size is Undefined");
    }

    ValueConstraintPtr negate() const override {
      BufferSizeConstraint Tmp(*this);
      Tmp.Op = BinaryOperator::negateComparisonOp(Op);
      return std::make_shared<BufferSizeConstraint>(Tmp);
    }

    bool checkSpecificValidity(const FunctionDecl *FD) const override {
      const bool ValidArg = getArgType(FD, ArgN)->isPointerType();
      assert(ValidArg &&
             "This constraint should be applied only on a pointer type");
      return ValidArg;
    }
  };

  /// The complete list of constraints that defines a single branch.
  typedef std::vector<ValueConstraintPtr> ConstraintSet;

  using ArgTypes = std::vector<QualType>;

  // A placeholder type, we use it whenever we do not care about the concrete
  // type in a Signature.
  const QualType Irrelevant{};
  bool static isIrrelevant(QualType T) { return T.isNull(); }

  // The signature of a function we want to describe with a summary. This is a
  // concessive signature, meaning there may be irrelevant types in the
  // signature which we do not check against a function with concrete types.
  struct Signature {
    ArgTypes ArgTys;
    QualType RetTy;
    Signature(ArgTypes ArgTys, QualType RetTy) : ArgTys(ArgTys), RetTy(RetTy) {
      assertRetTypeSuitableForSignature(RetTy);
      for (size_t I = 0, E = ArgTys.size(); I != E; ++I) {
        QualType ArgTy = ArgTys[I];
        assertArgTypeSuitableForSignature(ArgTy);
      }
    }

    bool matches(const FunctionDecl *FD) const;

  private:
    static void assertArgTypeSuitableForSignature(QualType T) {
      assert((T.isNull() || !T->isVoidType()) &&
             "We should have no void types in the spec");
      assert((T.isNull() || T.isCanonical()) &&
             "We should only have canonical types in the spec");
    }
    static void assertRetTypeSuitableForSignature(QualType T) {
      assert((T.isNull() || T.isCanonical()) &&
             "We should only have canonical types in the spec");
    }
  };

  static QualType getArgType(const FunctionDecl *FD, ArgNo ArgN) {
    assert(FD && "Function must be set");
    QualType T = (ArgN == Ret)
                     ? FD->getReturnType().getCanonicalType()
                     : FD->getParamDecl(ArgN)->getType().getCanonicalType();
    return T;
  }

  using Cases = std::vector<ConstraintSet>;

  /// A summary includes information about
  ///   * function prototype (signature)
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
  ///
  /// Application of a summary:
  ///   The signature and argument constraints together contain information
  ///   about which functions are handled by the summary. The signature can use
  ///   "wildcards", i.e. Irrelevant types. Irrelevant type of a parameter in
  ///   a signature means that type is not compared to the type of the parameter
  ///   in the found FunctionDecl. Argument constraints may specify additional
  ///   rules for the given parameter's type, those rules are checked once the
  ///   signature is matched.
  class Summary {
    Optional<Signature> Sign;
    const InvalidationKind InvalidationKd;
    Cases CaseConstraints;
    ConstraintSet ArgConstraints;

    // The function to which the summary applies. This is set after lookup and
    // match to the signature.
    const FunctionDecl *FD = nullptr;

  public:
    Summary(ArgTypes ArgTys, QualType RetTy, InvalidationKind InvalidationKd)
        : Sign(Signature(ArgTys, RetTy)), InvalidationKd(InvalidationKd) {}

    Summary(InvalidationKind InvalidationKd) : InvalidationKd(InvalidationKd) {}

    Summary &setSignature(const Signature &S) {
      Sign = S;
      return *this;
    }

    Summary &Case(ConstraintSet &&CS) {
      CaseConstraints.push_back(std::move(CS));
      return *this;
    }
    Summary &ArgConstraint(ValueConstraintPtr VC) {
      ArgConstraints.push_back(VC);
      return *this;
    }

    InvalidationKind getInvalidationKd() const { return InvalidationKd; }
    const Cases &getCaseConstraints() const { return CaseConstraints; }
    const ConstraintSet &getArgConstraints() const { return ArgConstraints; }

    QualType getArgType(ArgNo ArgN) const {
      return StdLibraryFunctionsChecker::getArgType(FD, ArgN);
    }

    // Returns true if the summary should be applied to the given function.
    // And if yes then store the function declaration.
    bool matchesAndSet(const FunctionDecl *FD) {
      assert(Sign &&
             "Signature must be set before comparing to a FunctionDecl");
      bool Result = Sign->matches(FD) && validateByConstraints(FD);
      if (Result) {
        assert(!this->FD && "FD must not be set more than once");
        this->FD = FD;
      }
      return Result;
    }

  private:
    // Once we know the exact type of the function then do sanity check on all
    // the given constraints.
    bool validateByConstraints(const FunctionDecl *FD) const {
      for (const ConstraintSet &Case : CaseConstraints)
        for (const ValueConstraintPtr &Constraint : Case)
          if (!Constraint->checkValidity(FD))
            return false;
      for (const ValueConstraintPtr &Constraint : ArgConstraints)
        if (!Constraint->checkValidity(FD))
          return false;
      return true;
    }
  };

  // The map of all functions supported by the checker. It is initialized
  // lazily, and it doesn't change after initialization.
  using FunctionSummaryMapType = llvm::DenseMap<const FunctionDecl *, Summary>;
  mutable FunctionSummaryMapType FunctionSummaryMap;

  mutable std::unique_ptr<BugType> BT_InvalidArg;

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

  bool DisplayLoadedSummaries = false;
  bool ModelPOSIX = false;

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
  QualType T = Summary.getArgType(getArgNo());
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
  QualType T = Summary.getArgType(getArgNo());
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
    ProgramStateRef State, const CallEvent &Call, const Summary &Summary,
    CheckerContext &C) const {

  ProgramStateManager &Mgr = State->getStateManager();
  SValBuilder &SVB = Mgr.getSValBuilder();
  QualType CondT = SVB.getConditionType();
  QualType T = Summary.getArgType(getArgNo());
  SVal V = getArgSVal(Call, getArgNo());

  BinaryOperator::Opcode Op = getOpcode();
  ArgNo OtherArg = getOtherArgNo();
  SVal OtherV = getArgSVal(Call, OtherArg);
  QualType OtherT = Summary.getArgType(OtherArg);
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
  for (const ValueConstraintPtr &Constraint : Summary.getArgConstraints()) {
    ProgramStateRef SuccessSt = Constraint->apply(NewState, Call, Summary, C);
    ProgramStateRef FailureSt =
        Constraint->negate()->apply(NewState, Call, Summary, C);
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
  for (const ConstraintSet &Case : Summary.getCaseConstraints()) {
    ProgramStateRef NewState = State;
    for (const ValueConstraintPtr &Constraint : Case) {
      NewState = Constraint->apply(NewState, Call, Summary, C);
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
  switch (Summary.getInvalidationKd()) {
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

bool StdLibraryFunctionsChecker::Signature::matches(
    const FunctionDecl *FD) const {
  // Check number of arguments:
  if (FD->param_size() != ArgTys.size())
    return false;

  // Check return type.
  if (!isIrrelevant(RetTy))
    if (RetTy != FD->getReturnType().getCanonicalType())
      return false;

  // Check argument types.
  for (size_t I = 0, E = ArgTys.size(); I != E; ++I) {
    QualType ArgTy = ArgTys[I];
    if (isIrrelevant(ArgTy))
      continue;
    if (ArgTy != FD->getParamDecl(I)->getType().getCanonicalType())
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

  auto FSMI = FunctionSummaryMap.find(FD->getCanonicalDecl());
  if (FSMI == FunctionSummaryMap.end())
    return None;
  return FSMI->second;
}

Optional<StdLibraryFunctionsChecker::Summary>
StdLibraryFunctionsChecker::findFunctionSummary(const CallEvent &Call,
                                                CheckerContext &C) const {
  const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(Call.getDecl());
  if (!FD)
    return None;
  return findFunctionSummary(FD, C);
}

static llvm::Optional<QualType> lookupType(StringRef Name,
                                           const ASTContext &ACtx) {
  IdentifierInfo &II = ACtx.Idents.get(Name);
  auto LookupRes = ACtx.getTranslationUnitDecl()->lookup(&II);
  if (LookupRes.size() == 0)
    return None;

  // Prioritze typedef declarations.
  // This is needed in case of C struct typedefs. E.g.:
  //   typedef struct FILE FILE;
  // In this case, we have a RecordDecl 'struct FILE' with the name 'FILE' and
  // we have a TypedefDecl with the name 'FILE'.
  for (Decl *D : LookupRes)
    if (auto *TD = dyn_cast<TypedefNameDecl>(D))
      return ACtx.getTypeDeclType(TD).getCanonicalType();

  // Find the first TypeDecl.
  // There maybe cases when a function has the same name as a struct.
  // E.g. in POSIX: `struct stat` and the function `stat()`:
  //   int stat(const char *restrict path, struct stat *restrict buf);
  for (Decl *D : LookupRes)
    if (auto *TD = dyn_cast<TypeDecl>(D))
      return ACtx.getTypeDeclType(TD).getCanonicalType();
  return None;
}

void StdLibraryFunctionsChecker::initFunctionSummaries(
    CheckerContext &C) const {
  if (!FunctionSummaryMap.empty())
    return;

  SValBuilder &SVB = C.getSValBuilder();
  BasicValueFactory &BVF = SVB.getBasicValueFactory();
  const ASTContext &ACtx = BVF.getContext();

  auto getRestrictTy = [&ACtx](QualType Ty) {
    return ACtx.getLangOpts().C99 ? ACtx.getRestrictType(Ty) : Ty;
  };

  // These types are useful for writing specifications quickly,
  // New specifications should probably introduce more types.
  // Some types are hard to obtain from the AST, eg. "ssize_t".
  // In such cases it should be possible to provide multiple variants
  // of function summary for common cases (eg. ssize_t could be int or long
  // or long long, so three summary variants would be enough).
  // Of course, function variants are also useful for C++ overloads.
  const QualType VoidTy = ACtx.VoidTy;
  const QualType IntTy = ACtx.IntTy;
  const QualType UnsignedIntTy = ACtx.UnsignedIntTy;
  const QualType LongTy = ACtx.LongTy;
  const QualType LongLongTy = ACtx.LongLongTy;
  const QualType SizeTy = ACtx.getSizeType();

  const QualType VoidPtrTy = ACtx.VoidPtrTy;            // void *
  const QualType IntPtrTy = ACtx.getPointerType(IntTy); // int *
  const QualType UnsignedIntPtrTy =
      ACtx.getPointerType(UnsignedIntTy); // unsigned int *
  const QualType VoidPtrRestrictTy = getRestrictTy(VoidPtrTy);
  const QualType ConstVoidPtrTy =
      ACtx.getPointerType(ACtx.VoidTy.withConst());            // const void *
  const QualType CharPtrTy = ACtx.getPointerType(ACtx.CharTy); // char *
  const QualType CharPtrRestrictTy = getRestrictTy(CharPtrTy);
  const QualType ConstCharPtrTy =
      ACtx.getPointerType(ACtx.CharTy.withConst()); // const char *
  const QualType ConstCharPtrRestrictTy = getRestrictTy(ConstCharPtrTy);
  const QualType Wchar_tPtrTy = ACtx.getPointerType(ACtx.WCharTy); // wchar_t *
  const QualType ConstWchar_tPtrTy =
      ACtx.getPointerType(ACtx.WCharTy.withConst()); // const wchar_t *
  const QualType ConstVoidPtrRestrictTy = getRestrictTy(ConstVoidPtrTy);

  const RangeInt IntMax = BVF.getMaxValue(IntTy).getLimitedValue();
  const RangeInt UnsignedIntMax =
      BVF.getMaxValue(UnsignedIntTy).getLimitedValue();
  const RangeInt LongMax = BVF.getMaxValue(LongTy).getLimitedValue();
  const RangeInt LongLongMax = BVF.getMaxValue(LongLongTy).getLimitedValue();
  const RangeInt SizeMax = BVF.getMaxValue(SizeTy).getLimitedValue();

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

  // Auxiliary class to aid adding summaries to the summary map.
  struct AddToFunctionSummaryMap {
    const ASTContext &ACtx;
    FunctionSummaryMapType &Map;
    bool DisplayLoadedSummaries;
    AddToFunctionSummaryMap(const ASTContext &ACtx, FunctionSummaryMapType &FSM,
                            bool DisplayLoadedSummaries)
        : ACtx(ACtx), Map(FSM), DisplayLoadedSummaries(DisplayLoadedSummaries) {
    }

    // Add a summary to a FunctionDecl found by lookup. The lookup is performed
    // by the given Name, and in the global scope. The summary will be attached
    // to the found FunctionDecl only if the signatures match.
    //
    // Returns true if the summary has been added, false otherwise.
    bool operator()(StringRef Name, Summary S) {
      IdentifierInfo &II = ACtx.Idents.get(Name);
      auto LookupRes = ACtx.getTranslationUnitDecl()->lookup(&II);
      if (LookupRes.size() == 0)
        return false;
      for (Decl *D : LookupRes) {
        if (auto *FD = dyn_cast<FunctionDecl>(D)) {
          if (S.matchesAndSet(FD)) {
            auto Res = Map.insert({FD->getCanonicalDecl(), S});
            assert(Res.second && "Function already has a summary set!");
            (void)Res;
            if (DisplayLoadedSummaries) {
              llvm::errs() << "Loaded summary for: ";
              FD->print(llvm::errs());
              llvm::errs() << "\n";
            }
            return true;
          }
        }
      }
      return false;
    }
    bool operator()(StringRef Name, Signature Sign, Summary Sum) {
      return operator()(Name, Sum.setSignature(Sign));
    }
    // Add several summaries for the given name.
    void operator()(StringRef Name, const std::vector<Summary> &Summaries) {
      for (const Summary &S : Summaries)
        operator()(Name, S);
    }
  } addToFunctionSummaryMap(ACtx, FunctionSummaryMap, DisplayLoadedSummaries);

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

  // Below are helpers functions to create the summaries.
  auto ArgumentCondition = [](ArgNo ArgN, RangeKind Kind,
                              IntRangeVector Ranges) {
    return std::make_shared<RangeConstraint>(ArgN, Kind, Ranges);
  };
  auto BufferSize = [](auto... Args) {
    return std::make_shared<BufferSizeConstraint>(Args...);
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

  Optional<QualType> FileTy = lookupType("FILE", ACtx);
  Optional<QualType> FilePtrTy, FilePtrRestrictTy;
  if (FileTy) {
    // FILE *
    FilePtrTy = ACtx.getPointerType(*FileTy);
    // FILE *restrict
    FilePtrRestrictTy = getRestrictTy(*FilePtrTy);
  }

  using RetType = QualType;
  // Templates for summaries that are reused by many functions.
  auto Getc = [&]() {
    return Summary(ArgTypes{*FilePtrTy}, RetType{IntTy}, NoEvalCall)
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
    return Summary(
               ArgTypes{VoidPtrRestrictTy, SizeTy, SizeTy, *FilePtrRestrictTy},
               RetType{SizeTy}, NoEvalCall)
        .Case({
            ReturnValueCondition(LessThanOrEq, ArgNo(2)),
        })
        .ArgConstraint(NotNull(ArgNo(0)));
  };
  auto Fwrite = [&]() {
    return Summary(ArgTypes{ConstVoidPtrRestrictTy, SizeTy, SizeTy,
                            *FilePtrRestrictTy},
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

  // The isascii() family of functions.
  // The behavior is undefined if the value of the argument is not
  // representable as unsigned char or is not equal to EOF. See e.g. C99
  // 7.4.1.2 The isalpha function (p: 181-182).
  addToFunctionSummaryMap(
      "isalnum",
      Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
          // Boils down to isupper() or islower() or isdigit().
          .Case({ArgumentCondition(0U, WithinRange,
                                   {{'0', '9'}, {'A', 'Z'}, {'a', 'z'}}),
                 ReturnValueCondition(OutOfRange, SingleValue(0))})
          // The locale-specific range.
          // No post-condition. We are completely unaware of
          // locale-specific return values.
          .Case({ArgumentCondition(0U, WithinRange, {{128, UCharRangeMax}})})
          .Case(
              {ArgumentCondition(
                   0U, OutOfRange,
                   {{'0', '9'}, {'A', 'Z'}, {'a', 'z'}, {128, UCharRangeMax}}),
               ReturnValueCondition(WithinRange, SingleValue(0))})
          .ArgConstraint(ArgumentCondition(
              0U, WithinRange, {{EOFv, EOFv}, {0, UCharRangeMax}})));
  addToFunctionSummaryMap(
      "isalpha",
      Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
          .Case({ArgumentCondition(0U, WithinRange, {{'A', 'Z'}, {'a', 'z'}}),
                 ReturnValueCondition(OutOfRange, SingleValue(0))})
          // The locale-specific range.
          .Case({ArgumentCondition(0U, WithinRange, {{128, UCharRangeMax}})})
          .Case({ArgumentCondition(
                     0U, OutOfRange,
                     {{'A', 'Z'}, {'a', 'z'}, {128, UCharRangeMax}}),
                 ReturnValueCondition(WithinRange, SingleValue(0))}));
  addToFunctionSummaryMap(
      "isascii",
      Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
          .Case({ArgumentCondition(0U, WithinRange, Range(0, 127)),
                 ReturnValueCondition(OutOfRange, SingleValue(0))})
          .Case({ArgumentCondition(0U, OutOfRange, Range(0, 127)),
                 ReturnValueCondition(WithinRange, SingleValue(0))}));
  addToFunctionSummaryMap(
      "isblank",
      Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
          .Case({ArgumentCondition(0U, WithinRange, {{'\t', '\t'}, {' ', ' '}}),
                 ReturnValueCondition(OutOfRange, SingleValue(0))})
          .Case({ArgumentCondition(0U, OutOfRange, {{'\t', '\t'}, {' ', ' '}}),
                 ReturnValueCondition(WithinRange, SingleValue(0))}));
  addToFunctionSummaryMap(
      "iscntrl",
      Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
          .Case({ArgumentCondition(0U, WithinRange, {{0, 32}, {127, 127}}),
                 ReturnValueCondition(OutOfRange, SingleValue(0))})
          .Case({ArgumentCondition(0U, OutOfRange, {{0, 32}, {127, 127}}),
                 ReturnValueCondition(WithinRange, SingleValue(0))}));
  addToFunctionSummaryMap(
      "isdigit",
      Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
          .Case({ArgumentCondition(0U, WithinRange, Range('0', '9')),
                 ReturnValueCondition(OutOfRange, SingleValue(0))})
          .Case({ArgumentCondition(0U, OutOfRange, Range('0', '9')),
                 ReturnValueCondition(WithinRange, SingleValue(0))}));
  addToFunctionSummaryMap(
      "isgraph",
      Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
          .Case({ArgumentCondition(0U, WithinRange, Range(33, 126)),
                 ReturnValueCondition(OutOfRange, SingleValue(0))})
          .Case({ArgumentCondition(0U, OutOfRange, Range(33, 126)),
                 ReturnValueCondition(WithinRange, SingleValue(0))}));
  addToFunctionSummaryMap(
      "islower",
      Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
          // Is certainly lowercase.
          .Case({ArgumentCondition(0U, WithinRange, Range('a', 'z')),
                 ReturnValueCondition(OutOfRange, SingleValue(0))})
          // Is ascii but not lowercase.
          .Case({ArgumentCondition(0U, WithinRange, Range(0, 127)),
                 ArgumentCondition(0U, OutOfRange, Range('a', 'z')),
                 ReturnValueCondition(WithinRange, SingleValue(0))})
          // The locale-specific range.
          .Case({ArgumentCondition(0U, WithinRange, {{128, UCharRangeMax}})})
          // Is not an unsigned char.
          .Case({ArgumentCondition(0U, OutOfRange, Range(0, UCharRangeMax)),
                 ReturnValueCondition(WithinRange, SingleValue(0))}));
  addToFunctionSummaryMap(
      "isprint",
      Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
          .Case({ArgumentCondition(0U, WithinRange, Range(32, 126)),
                 ReturnValueCondition(OutOfRange, SingleValue(0))})
          .Case({ArgumentCondition(0U, OutOfRange, Range(32, 126)),
                 ReturnValueCondition(WithinRange, SingleValue(0))}));
  addToFunctionSummaryMap(
      "ispunct",
      Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
          .Case({ArgumentCondition(
                     0U, WithinRange,
                     {{'!', '/'}, {':', '@'}, {'[', '`'}, {'{', '~'}}),
                 ReturnValueCondition(OutOfRange, SingleValue(0))})
          .Case({ArgumentCondition(
                     0U, OutOfRange,
                     {{'!', '/'}, {':', '@'}, {'[', '`'}, {'{', '~'}}),
                 ReturnValueCondition(WithinRange, SingleValue(0))}));
  addToFunctionSummaryMap(
      "isspace",
      Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
          // Space, '\f', '\n', '\r', '\t', '\v'.
          .Case({ArgumentCondition(0U, WithinRange, {{9, 13}, {' ', ' '}}),
                 ReturnValueCondition(OutOfRange, SingleValue(0))})
          // The locale-specific range.
          .Case({ArgumentCondition(0U, WithinRange, {{128, UCharRangeMax}})})
          .Case({ArgumentCondition(0U, OutOfRange,
                                   {{9, 13}, {' ', ' '}, {128, UCharRangeMax}}),
                 ReturnValueCondition(WithinRange, SingleValue(0))}));
  addToFunctionSummaryMap(
      "isupper",
      Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
          // Is certainly uppercase.
          .Case({ArgumentCondition(0U, WithinRange, Range('A', 'Z')),
                 ReturnValueCondition(OutOfRange, SingleValue(0))})
          // The locale-specific range.
          .Case({ArgumentCondition(0U, WithinRange, {{128, UCharRangeMax}})})
          // Other.
          .Case({ArgumentCondition(0U, OutOfRange,
                                   {{'A', 'Z'}, {128, UCharRangeMax}}),
                 ReturnValueCondition(WithinRange, SingleValue(0))}));
  addToFunctionSummaryMap(
      "isxdigit",
      Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
          .Case({ArgumentCondition(0U, WithinRange,
                                   {{'0', '9'}, {'A', 'F'}, {'a', 'f'}}),
                 ReturnValueCondition(OutOfRange, SingleValue(0))})
          .Case({ArgumentCondition(0U, OutOfRange,
                                   {{'0', '9'}, {'A', 'F'}, {'a', 'f'}}),
                 ReturnValueCondition(WithinRange, SingleValue(0))}));
  addToFunctionSummaryMap(
      "toupper", Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
                     .ArgConstraint(ArgumentCondition(
                         0U, WithinRange, {{EOFv, EOFv}, {0, UCharRangeMax}})));
  addToFunctionSummaryMap(
      "tolower", Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
                     .ArgConstraint(ArgumentCondition(
                         0U, WithinRange, {{EOFv, EOFv}, {0, UCharRangeMax}})));
  addToFunctionSummaryMap(
      "toascii", Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
                     .ArgConstraint(ArgumentCondition(
                         0U, WithinRange, {{EOFv, EOFv}, {0, UCharRangeMax}})));

  // The getc() family of functions that returns either a char or an EOF.
  if (FilePtrTy) {
    addToFunctionSummaryMap("getc", Getc());
    addToFunctionSummaryMap("fgetc", Getc());
  }
  addToFunctionSummaryMap(
      "getchar", Summary(ArgTypes{}, RetType{IntTy}, NoEvalCall)
                     .Case({ReturnValueCondition(
                         WithinRange, {{EOFv, EOFv}, {0, UCharRangeMax}})}));

  // read()-like functions that never return more than buffer size.
  if (FilePtrRestrictTy) {
    addToFunctionSummaryMap("fread", Fread());
    addToFunctionSummaryMap("fwrite", Fwrite());
  }

  // We are not sure how ssize_t is defined on every platform, so we
  // provide three variants that should cover common cases.
  // FIXME these are actually defined by POSIX and not by the C standard, we
  // should handle them together with the rest of the POSIX functions.
  addToFunctionSummaryMap("read", {Read(IntTy, IntMax), Read(LongTy, LongMax),
                                   Read(LongLongTy, LongLongMax)});
  addToFunctionSummaryMap("write", {Read(IntTy, IntMax), Read(LongTy, LongMax),
                                    Read(LongLongTy, LongLongMax)});

  // getline()-like functions either fail or read at least the delimiter.
  // FIXME these are actually defined by POSIX and not by the C standard, we
  // should handle them together with the rest of the POSIX functions.
  addToFunctionSummaryMap("getline",
                          {Getline(IntTy, IntMax), Getline(LongTy, LongMax),
                           Getline(LongLongTy, LongLongMax)});
  addToFunctionSummaryMap("getdelim",
                          {Getline(IntTy, IntMax), Getline(LongTy, LongMax),
                           Getline(LongLongTy, LongLongMax)});

  if (ModelPOSIX) {

    // long a64l(const char *str64);
    addToFunctionSummaryMap(
        "a64l", Summary(ArgTypes{ConstCharPtrTy}, RetType{LongTy}, NoEvalCall)
                    .ArgConstraint(NotNull(ArgNo(0))));

    // char *l64a(long value);
    addToFunctionSummaryMap(
        "l64a", Summary(ArgTypes{LongTy}, RetType{CharPtrTy}, NoEvalCall)
                    .ArgConstraint(
                        ArgumentCondition(0, WithinRange, Range(0, LongMax))));

    // int access(const char *pathname, int amode);
    addToFunctionSummaryMap("access", Summary(ArgTypes{ConstCharPtrTy, IntTy},
                                              RetType{IntTy}, NoEvalCall)
                                          .ArgConstraint(NotNull(ArgNo(0))));

    // int faccessat(int dirfd, const char *pathname, int mode, int flags);
    addToFunctionSummaryMap(
        "faccessat", Summary(ArgTypes{IntTy, ConstCharPtrTy, IntTy, IntTy},
                             RetType{IntTy}, NoEvalCall)
                         .ArgConstraint(NotNull(ArgNo(1))));

    // int dup(int fildes);
    addToFunctionSummaryMap(
        "dup", Summary(ArgTypes{IntTy}, RetType{IntTy}, NoEvalCall)
                   .ArgConstraint(
                       ArgumentCondition(0, WithinRange, Range(0, IntMax))));

    // int dup2(int fildes1, int filedes2);
    addToFunctionSummaryMap(
        "dup2",
        Summary(ArgTypes{IntTy, IntTy}, RetType{IntTy}, NoEvalCall)
            .ArgConstraint(ArgumentCondition(0, WithinRange, Range(0, IntMax)))
            .ArgConstraint(
                ArgumentCondition(1, WithinRange, Range(0, IntMax))));

    // int fdatasync(int fildes);
    addToFunctionSummaryMap(
        "fdatasync", Summary(ArgTypes{IntTy}, RetType{IntTy}, NoEvalCall)
                         .ArgConstraint(ArgumentCondition(0, WithinRange,
                                                          Range(0, IntMax))));

    // int fnmatch(const char *pattern, const char *string, int flags);
    addToFunctionSummaryMap(
        "fnmatch", Summary(ArgTypes{ConstCharPtrTy, ConstCharPtrTy, IntTy},
                           RetType{IntTy}, EvalCallAsPure)
                       .ArgConstraint(NotNull(ArgNo(0)))
                       .ArgConstraint(NotNull(ArgNo(1))));

    // int fsync(int fildes);
    addToFunctionSummaryMap(
        "fsync", Summary(ArgTypes{IntTy}, RetType{IntTy}, NoEvalCall)
                     .ArgConstraint(
                         ArgumentCondition(0, WithinRange, Range(0, IntMax))));

    Optional<QualType> Off_tTy = lookupType("off_t", ACtx);

    if (Off_tTy)
      // int truncate(const char *path, off_t length);
      addToFunctionSummaryMap("truncate",
                              Summary(ArgTypes{ConstCharPtrTy, *Off_tTy},
                                      RetType{IntTy}, NoEvalCall)
                                  .ArgConstraint(NotNull(ArgNo(0))));

    // int symlink(const char *oldpath, const char *newpath);
    addToFunctionSummaryMap("symlink",
                            Summary(ArgTypes{ConstCharPtrTy, ConstCharPtrTy},
                                    RetType{IntTy}, NoEvalCall)
                                .ArgConstraint(NotNull(ArgNo(0)))
                                .ArgConstraint(NotNull(ArgNo(1))));

    // int symlinkat(const char *oldpath, int newdirfd, const char *newpath);
    addToFunctionSummaryMap(
        "symlinkat",
        Summary(ArgTypes{ConstCharPtrTy, IntTy, ConstCharPtrTy}, RetType{IntTy},
                NoEvalCall)
            .ArgConstraint(NotNull(ArgNo(0)))
            .ArgConstraint(ArgumentCondition(1, WithinRange, Range(0, IntMax)))
            .ArgConstraint(NotNull(ArgNo(2))));

    if (Off_tTy)
      // int lockf(int fd, int cmd, off_t len);
      addToFunctionSummaryMap(
          "lockf",
          Summary(ArgTypes{IntTy, IntTy, *Off_tTy}, RetType{IntTy}, NoEvalCall)
              .ArgConstraint(
                  ArgumentCondition(0, WithinRange, Range(0, IntMax))));

    Optional<QualType> Mode_tTy = lookupType("mode_t", ACtx);

    if (Mode_tTy)
      // int creat(const char *pathname, mode_t mode);
      addToFunctionSummaryMap("creat",
                              Summary(ArgTypes{ConstCharPtrTy, *Mode_tTy},
                                      RetType{IntTy}, NoEvalCall)
                                  .ArgConstraint(NotNull(ArgNo(0))));

    // unsigned int sleep(unsigned int seconds);
    addToFunctionSummaryMap(
        "sleep",
        Summary(ArgTypes{UnsignedIntTy}, RetType{UnsignedIntTy}, NoEvalCall)
            .ArgConstraint(
                ArgumentCondition(0, WithinRange, Range(0, UnsignedIntMax))));

    Optional<QualType> DirTy = lookupType("DIR", ACtx);
    Optional<QualType> DirPtrTy;
    if (DirTy)
      DirPtrTy = ACtx.getPointerType(*DirTy);

    if (DirPtrTy)
      // int dirfd(DIR *dirp);
      addToFunctionSummaryMap(
          "dirfd", Summary(ArgTypes{*DirPtrTy}, RetType{IntTy}, NoEvalCall)
                       .ArgConstraint(NotNull(ArgNo(0))));

    // unsigned int alarm(unsigned int seconds);
    addToFunctionSummaryMap(
        "alarm",
        Summary(ArgTypes{UnsignedIntTy}, RetType{UnsignedIntTy}, NoEvalCall)
            .ArgConstraint(
                ArgumentCondition(0, WithinRange, Range(0, UnsignedIntMax))));

    if (DirPtrTy)
      // int closedir(DIR *dir);
      addToFunctionSummaryMap(
          "closedir", Summary(ArgTypes{*DirPtrTy}, RetType{IntTy}, NoEvalCall)
                          .ArgConstraint(NotNull(ArgNo(0))));

    // char *strdup(const char *s);
    addToFunctionSummaryMap("strdup", Summary(ArgTypes{ConstCharPtrTy},
                                              RetType{CharPtrTy}, NoEvalCall)
                                          .ArgConstraint(NotNull(ArgNo(0))));

    // char *strndup(const char *s, size_t n);
    addToFunctionSummaryMap(
        "strndup", Summary(ArgTypes{ConstCharPtrTy, SizeTy}, RetType{CharPtrTy},
                           NoEvalCall)
                       .ArgConstraint(NotNull(ArgNo(0)))
                       .ArgConstraint(ArgumentCondition(1, WithinRange,
                                                        Range(0, SizeMax))));

    // wchar_t *wcsdup(const wchar_t *s);
    addToFunctionSummaryMap("wcsdup", Summary(ArgTypes{ConstWchar_tPtrTy},
                                              RetType{Wchar_tPtrTy}, NoEvalCall)
                                          .ArgConstraint(NotNull(ArgNo(0))));

    // int mkstemp(char *template);
    addToFunctionSummaryMap(
        "mkstemp", Summary(ArgTypes{CharPtrTy}, RetType{IntTy}, NoEvalCall)
                       .ArgConstraint(NotNull(ArgNo(0))));

    // char *mkdtemp(char *template);
    addToFunctionSummaryMap(
        "mkdtemp", Summary(ArgTypes{CharPtrTy}, RetType{CharPtrTy}, NoEvalCall)
                       .ArgConstraint(NotNull(ArgNo(0))));

    // char *getcwd(char *buf, size_t size);
    addToFunctionSummaryMap(
        "getcwd",
        Summary(ArgTypes{CharPtrTy, SizeTy}, RetType{CharPtrTy}, NoEvalCall)
            .ArgConstraint(
                ArgumentCondition(1, WithinRange, Range(0, SizeMax))));

    if (Mode_tTy) {
      // int mkdir(const char *pathname, mode_t mode);
      addToFunctionSummaryMap("mkdir",
                              Summary(ArgTypes{ConstCharPtrTy, *Mode_tTy},
                                      RetType{IntTy}, NoEvalCall)
                                  .ArgConstraint(NotNull(ArgNo(0))));

      // int mkdirat(int dirfd, const char *pathname, mode_t mode);
      addToFunctionSummaryMap(
          "mkdirat", Summary(ArgTypes{IntTy, ConstCharPtrTy, *Mode_tTy},
                             RetType{IntTy}, NoEvalCall)
                         .ArgConstraint(NotNull(ArgNo(1))));
    }

    Optional<QualType> Dev_tTy = lookupType("dev_t", ACtx);

    if (Mode_tTy && Dev_tTy) {
      // int mknod(const char *pathname, mode_t mode, dev_t dev);
      addToFunctionSummaryMap(
          "mknod", Summary(ArgTypes{ConstCharPtrTy, *Mode_tTy, *Dev_tTy},
                           RetType{IntTy}, NoEvalCall)
                       .ArgConstraint(NotNull(ArgNo(0))));

      // int mknodat(int dirfd, const char *pathname, mode_t mode, dev_t dev);
      addToFunctionSummaryMap("mknodat", Summary(ArgTypes{IntTy, ConstCharPtrTy,
                                                          *Mode_tTy, *Dev_tTy},
                                                 RetType{IntTy}, NoEvalCall)
                                             .ArgConstraint(NotNull(ArgNo(1))));
    }

    if (Mode_tTy) {
      // int chmod(const char *path, mode_t mode);
      addToFunctionSummaryMap("chmod",
                              Summary(ArgTypes{ConstCharPtrTy, *Mode_tTy},
                                      RetType{IntTy}, NoEvalCall)
                                  .ArgConstraint(NotNull(ArgNo(0))));

      // int fchmodat(int dirfd, const char *pathname, mode_t mode, int flags);
      addToFunctionSummaryMap(
          "fchmodat", Summary(ArgTypes{IntTy, ConstCharPtrTy, *Mode_tTy, IntTy},
                              RetType{IntTy}, NoEvalCall)
                          .ArgConstraint(ArgumentCondition(0, WithinRange,
                                                           Range(0, IntMax)))
                          .ArgConstraint(NotNull(ArgNo(1))));

      // int fchmod(int fildes, mode_t mode);
      addToFunctionSummaryMap(
          "fchmod",
          Summary(ArgTypes{IntTy, *Mode_tTy}, RetType{IntTy}, NoEvalCall)
              .ArgConstraint(
                  ArgumentCondition(0, WithinRange, Range(0, IntMax))));
    }

    Optional<QualType> Uid_tTy = lookupType("uid_t", ACtx);
    Optional<QualType> Gid_tTy = lookupType("gid_t", ACtx);

    if (Uid_tTy && Gid_tTy) {
      // int fchownat(int dirfd, const char *pathname, uid_t owner, gid_t group,
      //              int flags);
      addToFunctionSummaryMap(
          "fchownat",
          Summary(ArgTypes{IntTy, ConstCharPtrTy, *Uid_tTy, *Gid_tTy, IntTy},
                  RetType{IntTy}, NoEvalCall)
              .ArgConstraint(
                  ArgumentCondition(0, WithinRange, Range(0, IntMax)))
              .ArgConstraint(NotNull(ArgNo(1))));

      // int chown(const char *path, uid_t owner, gid_t group);
      addToFunctionSummaryMap(
          "chown", Summary(ArgTypes{ConstCharPtrTy, *Uid_tTy, *Gid_tTy},
                           RetType{IntTy}, NoEvalCall)
                       .ArgConstraint(NotNull(ArgNo(0))));

      // int lchown(const char *path, uid_t owner, gid_t group);
      addToFunctionSummaryMap(
          "lchown", Summary(ArgTypes{ConstCharPtrTy, *Uid_tTy, *Gid_tTy},
                            RetType{IntTy}, NoEvalCall)
                        .ArgConstraint(NotNull(ArgNo(0))));

      // int fchown(int fildes, uid_t owner, gid_t group);
      addToFunctionSummaryMap(
          "fchown", Summary(ArgTypes{IntTy, *Uid_tTy, *Gid_tTy}, RetType{IntTy},
                            NoEvalCall)
                        .ArgConstraint(ArgumentCondition(0, WithinRange,
                                                         Range(0, IntMax))));
    }

    // int rmdir(const char *pathname);
    addToFunctionSummaryMap(
        "rmdir", Summary(ArgTypes{ConstCharPtrTy}, RetType{IntTy}, NoEvalCall)
                     .ArgConstraint(NotNull(ArgNo(0))));

    // int chdir(const char *path);
    addToFunctionSummaryMap(
        "chdir", Summary(ArgTypes{ConstCharPtrTy}, RetType{IntTy}, NoEvalCall)
                     .ArgConstraint(NotNull(ArgNo(0))));

    // int link(const char *oldpath, const char *newpath);
    addToFunctionSummaryMap("link",
                            Summary(ArgTypes{ConstCharPtrTy, ConstCharPtrTy},
                                    RetType{IntTy}, NoEvalCall)
                                .ArgConstraint(NotNull(ArgNo(0)))
                                .ArgConstraint(NotNull(ArgNo(1))));

    // int linkat(int fd1, const char *path1, int fd2, const char *path2,
    //            int flag);
    addToFunctionSummaryMap(
        "linkat",
        Summary(ArgTypes{IntTy, ConstCharPtrTy, IntTy, ConstCharPtrTy, IntTy},
                RetType{IntTy}, NoEvalCall)
            .ArgConstraint(ArgumentCondition(0, WithinRange, Range(0, IntMax)))
            .ArgConstraint(NotNull(ArgNo(1)))
            .ArgConstraint(ArgumentCondition(2, WithinRange, Range(0, IntMax)))
            .ArgConstraint(NotNull(ArgNo(3))));

    // int unlink(const char *pathname);
    addToFunctionSummaryMap(
        "unlink", Summary(ArgTypes{ConstCharPtrTy}, RetType{IntTy}, NoEvalCall)
                      .ArgConstraint(NotNull(ArgNo(0))));

    // int unlinkat(int fd, const char *path, int flag);
    addToFunctionSummaryMap(
        "unlinkat",
        Summary(ArgTypes{IntTy, ConstCharPtrTy, IntTy}, RetType{IntTy},
                NoEvalCall)
            .ArgConstraint(ArgumentCondition(0, WithinRange, Range(0, IntMax)))
            .ArgConstraint(NotNull(ArgNo(1))));

    Optional<QualType> StructStatTy = lookupType("stat", ACtx);
    Optional<QualType> StructStatPtrTy, StructStatPtrRestrictTy;
    if (StructStatTy) {
      StructStatPtrTy = ACtx.getPointerType(*StructStatTy);
      StructStatPtrRestrictTy = getRestrictTy(*StructStatPtrTy);
    }

    if (StructStatPtrTy)
      // int fstat(int fd, struct stat *statbuf);
      addToFunctionSummaryMap(
          "fstat",
          Summary(ArgTypes{IntTy, *StructStatPtrTy}, RetType{IntTy}, NoEvalCall)
              .ArgConstraint(
                  ArgumentCondition(0, WithinRange, Range(0, IntMax)))
              .ArgConstraint(NotNull(ArgNo(1))));

    if (StructStatPtrRestrictTy) {
      // int stat(const char *restrict path, struct stat *restrict buf);
      addToFunctionSummaryMap(
          "stat",
          Summary(ArgTypes{ConstCharPtrRestrictTy, *StructStatPtrRestrictTy},
                  RetType{IntTy}, NoEvalCall)
              .ArgConstraint(NotNull(ArgNo(0)))
              .ArgConstraint(NotNull(ArgNo(1))));

      // int lstat(const char *restrict path, struct stat *restrict buf);
      addToFunctionSummaryMap(
          "lstat",
          Summary(ArgTypes{ConstCharPtrRestrictTy, *StructStatPtrRestrictTy},
                  RetType{IntTy}, NoEvalCall)
              .ArgConstraint(NotNull(ArgNo(0)))
              .ArgConstraint(NotNull(ArgNo(1))));

      // int fstatat(int fd, const char *restrict path,
      //             struct stat *restrict buf, int flag);
      addToFunctionSummaryMap(
          "fstatat", Summary(ArgTypes{IntTy, ConstCharPtrRestrictTy,
                                      *StructStatPtrRestrictTy, IntTy},
                             RetType{IntTy}, NoEvalCall)
                         .ArgConstraint(ArgumentCondition(0, WithinRange,
                                                          Range(0, IntMax)))
                         .ArgConstraint(NotNull(ArgNo(1)))
                         .ArgConstraint(NotNull(ArgNo(2))));
    }

    if (DirPtrTy) {
      // DIR *opendir(const char *name);
      addToFunctionSummaryMap("opendir", Summary(ArgTypes{ConstCharPtrTy},
                                                 RetType{*DirPtrTy}, NoEvalCall)
                                             .ArgConstraint(NotNull(ArgNo(0))));

      // DIR *fdopendir(int fd);
      addToFunctionSummaryMap(
          "fdopendir", Summary(ArgTypes{IntTy}, RetType{*DirPtrTy}, NoEvalCall)
                           .ArgConstraint(ArgumentCondition(0, WithinRange,
                                                            Range(0, IntMax))));
    }

    // int isatty(int fildes);
    addToFunctionSummaryMap(
        "isatty", Summary(ArgTypes{IntTy}, RetType{IntTy}, NoEvalCall)
                      .ArgConstraint(
                          ArgumentCondition(0, WithinRange, Range(0, IntMax))));

    if (FilePtrTy) {
      // FILE *popen(const char *command, const char *type);
      addToFunctionSummaryMap("popen",
                              Summary(ArgTypes{ConstCharPtrTy, ConstCharPtrTy},
                                      RetType{*FilePtrTy}, NoEvalCall)
                                  .ArgConstraint(NotNull(ArgNo(0)))
                                  .ArgConstraint(NotNull(ArgNo(1))));

      // int pclose(FILE *stream);
      addToFunctionSummaryMap(
          "pclose", Summary(ArgTypes{*FilePtrTy}, RetType{IntTy}, NoEvalCall)
                        .ArgConstraint(NotNull(ArgNo(0))));
    }

    // int close(int fildes);
    addToFunctionSummaryMap(
        "close", Summary(ArgTypes{IntTy}, RetType{IntTy}, NoEvalCall)
                     .ArgConstraint(
                         ArgumentCondition(0, WithinRange, Range(0, IntMax))));

    // long fpathconf(int fildes, int name);
    addToFunctionSummaryMap(
        "fpathconf",
        Summary(ArgTypes{IntTy, IntTy}, RetType{LongTy}, NoEvalCall)
            .ArgConstraint(
                ArgumentCondition(0, WithinRange, Range(0, IntMax))));

    // long pathconf(const char *path, int name);
    addToFunctionSummaryMap("pathconf", Summary(ArgTypes{ConstCharPtrTy, IntTy},
                                                RetType{LongTy}, NoEvalCall)
                                            .ArgConstraint(NotNull(ArgNo(0))));

    if (FilePtrTy)
      // FILE *fdopen(int fd, const char *mode);
      addToFunctionSummaryMap(
          "fdopen", Summary(ArgTypes{IntTy, ConstCharPtrTy},
                            RetType{*FilePtrTy}, NoEvalCall)
                        .ArgConstraint(
                            ArgumentCondition(0, WithinRange, Range(0, IntMax)))
                        .ArgConstraint(NotNull(ArgNo(1))));

    if (DirPtrTy) {
      // void rewinddir(DIR *dir);
      addToFunctionSummaryMap(
          "rewinddir", Summary(ArgTypes{*DirPtrTy}, RetType{VoidTy}, NoEvalCall)
                           .ArgConstraint(NotNull(ArgNo(0))));

      // void seekdir(DIR *dirp, long loc);
      addToFunctionSummaryMap("seekdir", Summary(ArgTypes{*DirPtrTy, LongTy},
                                                 RetType{VoidTy}, NoEvalCall)
                                             .ArgConstraint(NotNull(ArgNo(0))));
    }

    // int rand_r(unsigned int *seedp);
    addToFunctionSummaryMap("rand_r", Summary(ArgTypes{UnsignedIntPtrTy},
                                              RetType{IntTy}, NoEvalCall)
                                          .ArgConstraint(NotNull(ArgNo(0))));

    // int strcasecmp(const char *s1, const char *s2);
    addToFunctionSummaryMap("strcasecmp",
                            Summary(ArgTypes{ConstCharPtrTy, ConstCharPtrTy},
                                    RetType{IntTy}, EvalCallAsPure)
                                .ArgConstraint(NotNull(ArgNo(0)))
                                .ArgConstraint(NotNull(ArgNo(1))));

    // int strncasecmp(const char *s1, const char *s2, size_t n);
    addToFunctionSummaryMap(
        "strncasecmp", Summary(ArgTypes{ConstCharPtrTy, ConstCharPtrTy, SizeTy},
                               RetType{IntTy}, EvalCallAsPure)
                           .ArgConstraint(NotNull(ArgNo(0)))
                           .ArgConstraint(NotNull(ArgNo(1)))
                           .ArgConstraint(ArgumentCondition(
                               2, WithinRange, Range(0, SizeMax))));

    if (FilePtrTy && Off_tTy) {

      // int fileno(FILE *stream);
      addToFunctionSummaryMap(
          "fileno", Summary(ArgTypes{*FilePtrTy}, RetType{IntTy}, NoEvalCall)
                        .ArgConstraint(NotNull(ArgNo(0))));

      // int fseeko(FILE *stream, off_t offset, int whence);
      addToFunctionSummaryMap("fseeko",
                              Summary(ArgTypes{*FilePtrTy, *Off_tTy, IntTy},
                                      RetType{IntTy}, NoEvalCall)
                                  .ArgConstraint(NotNull(ArgNo(0))));

      // off_t ftello(FILE *stream);
      addToFunctionSummaryMap(
          "ftello", Summary(ArgTypes{*FilePtrTy}, RetType{*Off_tTy}, NoEvalCall)
                        .ArgConstraint(NotNull(ArgNo(0))));
    }

    if (Off_tTy) {
      Optional<RangeInt> Off_tMax = BVF.getMaxValue(*Off_tTy).getLimitedValue();

      // void *mmap(void *addr, size_t length, int prot, int flags, int fd,
      // off_t offset);
      addToFunctionSummaryMap(
          "mmap",
          Summary(ArgTypes{VoidPtrTy, SizeTy, IntTy, IntTy, IntTy, *Off_tTy},
                  RetType{VoidPtrTy}, NoEvalCall)
              .ArgConstraint(
                  ArgumentCondition(1, WithinRange, Range(1, SizeMax)))
              .ArgConstraint(
                  ArgumentCondition(4, WithinRange, Range(0, *Off_tMax))));
    }

    Optional<QualType> Off64_tTy = lookupType("off64_t", ACtx);
    Optional<RangeInt> Off64_tMax;
    if (Off64_tTy) {
      Off64_tMax = BVF.getMaxValue(*Off_tTy).getLimitedValue();
      // void *mmap64(void *addr, size_t length, int prot, int flags, int fd,
      // off64_t offset);
      addToFunctionSummaryMap(
          "mmap64",
          Summary(ArgTypes{VoidPtrTy, SizeTy, IntTy, IntTy, IntTy, *Off64_tTy},
                  RetType{VoidPtrTy}, NoEvalCall)
              .ArgConstraint(
                  ArgumentCondition(1, WithinRange, Range(1, SizeMax)))
              .ArgConstraint(
                  ArgumentCondition(4, WithinRange, Range(0, *Off64_tMax))));
    }

    // int pipe(int fildes[2]);
    addToFunctionSummaryMap(
        "pipe", Summary(ArgTypes{IntPtrTy}, RetType{IntTy}, NoEvalCall)
                    .ArgConstraint(NotNull(ArgNo(0))));

    if (Off_tTy)
      // off_t lseek(int fildes, off_t offset, int whence);
      addToFunctionSummaryMap(
          "lseek", Summary(ArgTypes{IntTy, *Off_tTy, IntTy}, RetType{*Off_tTy},
                           NoEvalCall)
                       .ArgConstraint(ArgumentCondition(0, WithinRange,
                                                        Range(0, IntMax))));

    Optional<QualType> Ssize_tTy = lookupType("ssize_t", ACtx);

    if (Ssize_tTy) {
      // ssize_t readlink(const char *restrict path, char *restrict buf,
      //                  size_t bufsize);
      addToFunctionSummaryMap(
          "readlink",
          Summary(ArgTypes{ConstCharPtrRestrictTy, CharPtrRestrictTy, SizeTy},
                  RetType{*Ssize_tTy}, NoEvalCall)
              .ArgConstraint(NotNull(ArgNo(0)))
              .ArgConstraint(NotNull(ArgNo(1)))
              .ArgConstraint(BufferSize(/*Buffer=*/ArgNo(1),
                                        /*BufSize=*/ArgNo(2)))
              .ArgConstraint(
                  ArgumentCondition(2, WithinRange, Range(0, SizeMax))));

      // ssize_t readlinkat(int fd, const char *restrict path,
      //                    char *restrict buf, size_t bufsize);
      addToFunctionSummaryMap(
          "readlinkat", Summary(ArgTypes{IntTy, ConstCharPtrRestrictTy,
                                         CharPtrRestrictTy, SizeTy},
                                RetType{*Ssize_tTy}, NoEvalCall)
                            .ArgConstraint(ArgumentCondition(0, WithinRange,
                                                             Range(0, IntMax)))
                            .ArgConstraint(NotNull(ArgNo(1)))
                            .ArgConstraint(NotNull(ArgNo(2)))
                            .ArgConstraint(BufferSize(/*Buffer=*/ArgNo(2),
                                                      /*BufSize=*/ArgNo(3)))
                            .ArgConstraint(ArgumentCondition(
                                3, WithinRange, Range(0, SizeMax))));
    }

    // int renameat(int olddirfd, const char *oldpath, int newdirfd, const char
    // *newpath);
    addToFunctionSummaryMap("renameat", Summary(ArgTypes{IntTy, ConstCharPtrTy,
                                                         IntTy, ConstCharPtrTy},
                                                RetType{IntTy}, NoEvalCall)
                                            .ArgConstraint(NotNull(ArgNo(1)))
                                            .ArgConstraint(NotNull(ArgNo(3))));

    // char *realpath(const char *restrict file_name,
    //                char *restrict resolved_name);
    addToFunctionSummaryMap(
        "realpath", Summary(ArgTypes{ConstCharPtrRestrictTy, CharPtrRestrictTy},
                            RetType{CharPtrTy}, NoEvalCall)
                        .ArgConstraint(NotNull(ArgNo(0))));

    QualType CharPtrConstPtr = ACtx.getPointerType(CharPtrTy.withConst());

    // int execv(const char *path, char *const argv[]);
    addToFunctionSummaryMap("execv",
                            Summary(ArgTypes{ConstCharPtrTy, CharPtrConstPtr},
                                    RetType{IntTy}, NoEvalCall)
                                .ArgConstraint(NotNull(ArgNo(0))));

    // int execvp(const char *file, char *const argv[]);
    addToFunctionSummaryMap("execvp",
                            Summary(ArgTypes{ConstCharPtrTy, CharPtrConstPtr},
                                    RetType{IntTy}, NoEvalCall)
                                .ArgConstraint(NotNull(ArgNo(0))));

    // int getopt(int argc, char * const argv[], const char *optstring);
    addToFunctionSummaryMap(
        "getopt",
        Summary(ArgTypes{IntTy, CharPtrConstPtr, ConstCharPtrTy},
                RetType{IntTy}, NoEvalCall)
            .ArgConstraint(ArgumentCondition(0, WithinRange, Range(0, IntMax)))
            .ArgConstraint(NotNull(ArgNo(1)))
            .ArgConstraint(NotNull(ArgNo(2))));

    Optional<QualType> StructSockaddrTy = lookupType("sockaddr", ACtx);
    Optional<QualType> StructSockaddrPtrTy, ConstStructSockaddrPtrTy,
        StructSockaddrPtrRestrictTy, ConstStructSockaddrPtrRestrictTy;
    if (StructSockaddrTy) {
      StructSockaddrPtrTy = ACtx.getPointerType(*StructSockaddrTy);
      ConstStructSockaddrPtrTy =
          ACtx.getPointerType(StructSockaddrTy->withConst());
      StructSockaddrPtrRestrictTy = getRestrictTy(*StructSockaddrPtrTy);
      ConstStructSockaddrPtrRestrictTy =
          getRestrictTy(*ConstStructSockaddrPtrTy);
    }
    Optional<QualType> Socklen_tTy = lookupType("socklen_t", ACtx);
    Optional<QualType> Socklen_tPtrTy, Socklen_tPtrRestrictTy;
    Optional<RangeInt> Socklen_tMax;
    if (Socklen_tTy) {
      Socklen_tMax = BVF.getMaxValue(*Socklen_tTy).getLimitedValue();
      Socklen_tPtrTy = ACtx.getPointerType(*Socklen_tTy);
      Socklen_tPtrRestrictTy = getRestrictTy(*Socklen_tPtrTy);
    }

    // In 'socket.h' of some libc implementations with C99, sockaddr parameter
    // is a transparent union of the underlying sockaddr_ family of pointers
    // instead of being a pointer to struct sockaddr. In these cases, the
    // standardized signature will not match, thus we try to match with another
    // signature that has the joker Irrelevant type. We also remove those
    // constraints which require pointer types for the sockaddr param.
    if (StructSockaddrPtrRestrictTy && Socklen_tPtrRestrictTy) {
      auto Accept = Summary(NoEvalCall)
                        .ArgConstraint(ArgumentCondition(0, WithinRange,
                                                         Range(0, IntMax)));
      if (!addToFunctionSummaryMap(
              "accept",
              // int accept(int socket, struct sockaddr *restrict address,
              //            socklen_t *restrict address_len);
              Signature(ArgTypes{IntTy, *StructSockaddrPtrRestrictTy,
                                 *Socklen_tPtrRestrictTy},
                        RetType{IntTy}),
              Accept))
        addToFunctionSummaryMap(
            "accept",
            Signature(ArgTypes{IntTy, Irrelevant, *Socklen_tPtrRestrictTy},
                      RetType{IntTy}),
            Accept);

      // int bind(int socket, const struct sockaddr *address, socklen_t
      //          address_len);
      if (!addToFunctionSummaryMap(
              "bind",
              Summary(ArgTypes{IntTy, *ConstStructSockaddrPtrTy, *Socklen_tTy},
                      RetType{IntTy}, NoEvalCall)
                  .ArgConstraint(
                      ArgumentCondition(0, WithinRange, Range(0, IntMax)))
                  .ArgConstraint(NotNull(ArgNo(1)))
                  .ArgConstraint(
                      BufferSize(/*Buffer=*/ArgNo(1), /*BufSize=*/ArgNo(2)))
                  .ArgConstraint(ArgumentCondition(2, WithinRange,
                                                   Range(0, *Socklen_tMax)))))
        // Do not add constraints on sockaddr.
        addToFunctionSummaryMap(
            "bind", Summary(ArgTypes{IntTy, Irrelevant, *Socklen_tTy},
                            RetType{IntTy}, NoEvalCall)
                        .ArgConstraint(
                            ArgumentCondition(0, WithinRange, Range(0, IntMax)))
                        .ArgConstraint(ArgumentCondition(
                            2, WithinRange, Range(0, *Socklen_tMax))));

      // int getpeername(int socket, struct sockaddr *restrict address,
      //                 socklen_t *restrict address_len);
      if (!addToFunctionSummaryMap(
              "getpeername",
              Summary(ArgTypes{IntTy, *StructSockaddrPtrRestrictTy,
                               *Socklen_tPtrRestrictTy},
                      RetType{IntTy}, NoEvalCall)
                  .ArgConstraint(
                      ArgumentCondition(0, WithinRange, Range(0, IntMax)))
                  .ArgConstraint(NotNull(ArgNo(1)))
                  .ArgConstraint(NotNull(ArgNo(2)))))
        addToFunctionSummaryMap(
            "getpeername",
            Summary(ArgTypes{IntTy, Irrelevant, *Socklen_tPtrRestrictTy},
                    RetType{IntTy}, NoEvalCall)
                .ArgConstraint(
                    ArgumentCondition(0, WithinRange, Range(0, IntMax))));

      // int getsockname(int socket, struct sockaddr *restrict address,
      //                 socklen_t *restrict address_len);
      if (!addToFunctionSummaryMap(
              "getsockname",
              Summary(ArgTypes{IntTy, *StructSockaddrPtrRestrictTy,
                               *Socklen_tPtrRestrictTy},
                      RetType{IntTy}, NoEvalCall)
                  .ArgConstraint(
                      ArgumentCondition(0, WithinRange, Range(0, IntMax)))
                  .ArgConstraint(NotNull(ArgNo(1)))
                  .ArgConstraint(NotNull(ArgNo(2)))))
        addToFunctionSummaryMap(
            "getsockname",
            Summary(ArgTypes{IntTy, Irrelevant, *Socklen_tPtrRestrictTy},
                    RetType{IntTy}, NoEvalCall)
                .ArgConstraint(
                    ArgumentCondition(0, WithinRange, Range(0, IntMax))));

      // int connect(int socket, const struct sockaddr *address, socklen_t
      //             address_len);
      if (!addToFunctionSummaryMap(
              "connect",
              Summary(ArgTypes{IntTy, *ConstStructSockaddrPtrTy, *Socklen_tTy},
                      RetType{IntTy}, NoEvalCall)
                  .ArgConstraint(
                      ArgumentCondition(0, WithinRange, Range(0, IntMax)))
                  .ArgConstraint(NotNull(ArgNo(1)))))
        addToFunctionSummaryMap(
            "connect", Summary(ArgTypes{IntTy, Irrelevant, *Socklen_tTy},
                               RetType{IntTy}, NoEvalCall)
                           .ArgConstraint(ArgumentCondition(0, WithinRange,
                                                            Range(0, IntMax))));

      auto Recvfrom = Summary(NoEvalCall)
                          .ArgConstraint(ArgumentCondition(0, WithinRange,
                                                           Range(0, IntMax)))
                          .ArgConstraint(BufferSize(/*Buffer=*/ArgNo(1),
                                                    /*BufSize=*/ArgNo(2)));
      if (Ssize_tTy &&
          !addToFunctionSummaryMap(
              "recvfrom",
              // ssize_t recvfrom(int socket, void *restrict buffer,
              //                  size_t length,
              //                  int flags, struct sockaddr *restrict address,
              //                  socklen_t *restrict address_len);
              Signature(ArgTypes{IntTy, VoidPtrRestrictTy, SizeTy, IntTy,
                                 *StructSockaddrPtrRestrictTy,
                                 *Socklen_tPtrRestrictTy},
                        RetType{*Ssize_tTy}),
              Recvfrom))
        addToFunctionSummaryMap(
            "recvfrom",
            Signature(ArgTypes{IntTy, VoidPtrRestrictTy, SizeTy, IntTy,
                               Irrelevant, *Socklen_tPtrRestrictTy},
                      RetType{*Ssize_tTy}),
            Recvfrom);

      auto Sendto = Summary(NoEvalCall)
                        .ArgConstraint(
                            ArgumentCondition(0, WithinRange, Range(0, IntMax)))
                        .ArgConstraint(BufferSize(/*Buffer=*/ArgNo(1),
                                                  /*BufSize=*/ArgNo(2)));
      if (Ssize_tTy &&
          !addToFunctionSummaryMap(
              "sendto",
              // ssize_t sendto(int socket, const void *message, size_t length,
              //                int flags, const struct sockaddr *dest_addr,
              //                socklen_t dest_len);
              Signature(ArgTypes{IntTy, ConstVoidPtrTy, SizeTy, IntTy,
                                 *ConstStructSockaddrPtrTy, *Socklen_tTy},
                        RetType{*Ssize_tTy}),
              Sendto))
        addToFunctionSummaryMap(
            "sendto",
            Signature(ArgTypes{IntTy, ConstVoidPtrTy, SizeTy, IntTy, Irrelevant,
                               *Socklen_tTy},
                      RetType{*Ssize_tTy}),
            Sendto);
    }

    // int listen(int sockfd, int backlog);
    addToFunctionSummaryMap(
        "listen", Summary(ArgTypes{IntTy, IntTy}, RetType{IntTy}, NoEvalCall)
                      .ArgConstraint(
                          ArgumentCondition(0, WithinRange, Range(0, IntMax))));

    if (Ssize_tTy)
      // ssize_t recv(int sockfd, void *buf, size_t len, int flags);
      addToFunctionSummaryMap(
          "recv", Summary(ArgTypes{IntTy, VoidPtrTy, SizeTy, IntTy},
                          RetType{*Ssize_tTy}, NoEvalCall)
                      .ArgConstraint(
                          ArgumentCondition(0, WithinRange, Range(0, IntMax)))
                      .ArgConstraint(BufferSize(/*Buffer=*/ArgNo(1),
                                                /*BufSize=*/ArgNo(2))));

    Optional<QualType> StructMsghdrTy = lookupType("msghdr", ACtx);
    Optional<QualType> StructMsghdrPtrTy, ConstStructMsghdrPtrTy;
    if (StructMsghdrTy) {
      StructMsghdrPtrTy = ACtx.getPointerType(*StructMsghdrTy);
      ConstStructMsghdrPtrTy = ACtx.getPointerType(StructMsghdrTy->withConst());
    }

    if (Ssize_tTy && StructMsghdrPtrTy)
      // ssize_t recvmsg(int sockfd, struct msghdr *msg, int flags);
      addToFunctionSummaryMap(
          "recvmsg", Summary(ArgTypes{IntTy, *StructMsghdrPtrTy, IntTy},
                             RetType{*Ssize_tTy}, NoEvalCall)
                         .ArgConstraint(ArgumentCondition(0, WithinRange,
                                                          Range(0, IntMax))));

    if (Ssize_tTy && ConstStructMsghdrPtrTy)
      // ssize_t sendmsg(int sockfd, const struct msghdr *msg, int flags);
      addToFunctionSummaryMap(
          "sendmsg", Summary(ArgTypes{IntTy, *ConstStructMsghdrPtrTy, IntTy},
                             RetType{*Ssize_tTy}, NoEvalCall)
                         .ArgConstraint(ArgumentCondition(0, WithinRange,
                                                          Range(0, IntMax))));

    if (Socklen_tTy)
      // int setsockopt(int socket, int level, int option_name,
      //                const void *option_value, socklen_t option_len);
      addToFunctionSummaryMap(
          "setsockopt",
          Summary(ArgTypes{IntTy, IntTy, IntTy, ConstVoidPtrTy, *Socklen_tTy},
                  RetType{IntTy}, NoEvalCall)
              .ArgConstraint(NotNull(ArgNo(3)))
              .ArgConstraint(
                  BufferSize(/*Buffer=*/ArgNo(3), /*BufSize=*/ArgNo(4)))
              .ArgConstraint(
                  ArgumentCondition(4, WithinRange, Range(0, *Socklen_tMax))));

    if (Socklen_tPtrRestrictTy)
      // int getsockopt(int socket, int level, int option_name,
      //                void *restrict option_value,
      //                socklen_t *restrict option_len);
      addToFunctionSummaryMap(
          "getsockopt", Summary(ArgTypes{IntTy, IntTy, IntTy, VoidPtrRestrictTy,
                                         *Socklen_tPtrRestrictTy},
                                RetType{IntTy}, NoEvalCall)
                            .ArgConstraint(NotNull(ArgNo(3)))
                            .ArgConstraint(NotNull(ArgNo(4))));

    if (Ssize_tTy)
      // ssize_t send(int sockfd, const void *buf, size_t len, int flags);
      addToFunctionSummaryMap(
          "send", Summary(ArgTypes{IntTy, ConstVoidPtrTy, SizeTy, IntTy},
                          RetType{*Ssize_tTy}, NoEvalCall)
                      .ArgConstraint(
                          ArgumentCondition(0, WithinRange, Range(0, IntMax)))
                      .ArgConstraint(BufferSize(/*Buffer=*/ArgNo(1),
                                                /*BufSize=*/ArgNo(2))));

    // int socketpair(int domain, int type, int protocol, int sv[2]);
    addToFunctionSummaryMap("socketpair",
                            Summary(ArgTypes{IntTy, IntTy, IntTy, IntPtrTy},
                                    RetType{IntTy}, NoEvalCall)
                                .ArgConstraint(NotNull(ArgNo(3))));

    if (ConstStructSockaddrPtrRestrictTy && Socklen_tTy)
      // int getnameinfo(const struct sockaddr *restrict sa, socklen_t salen,
      //                 char *restrict node, socklen_t nodelen,
      //                 char *restrict service,
      //                 socklen_t servicelen, int flags);
      //
      // This is defined in netdb.h. And contrary to 'socket.h', the sockaddr
      // parameter is never handled as a transparent union in netdb.h
      addToFunctionSummaryMap(
          "getnameinfo",
          Summary(ArgTypes{*ConstStructSockaddrPtrRestrictTy, *Socklen_tTy,
                           CharPtrRestrictTy, *Socklen_tTy, CharPtrRestrictTy,
                           *Socklen_tTy, IntTy},
                  RetType{IntTy}, NoEvalCall)
              .ArgConstraint(
                  BufferSize(/*Buffer=*/ArgNo(0), /*BufSize=*/ArgNo(1)))
              .ArgConstraint(
                  ArgumentCondition(1, WithinRange, Range(0, *Socklen_tMax)))
              .ArgConstraint(
                  BufferSize(/*Buffer=*/ArgNo(2), /*BufSize=*/ArgNo(3)))
              .ArgConstraint(
                  ArgumentCondition(3, WithinRange, Range(0, *Socklen_tMax)))
              .ArgConstraint(
                  BufferSize(/*Buffer=*/ArgNo(4), /*BufSize=*/ArgNo(5)))
              .ArgConstraint(
                  ArgumentCondition(5, WithinRange, Range(0, *Socklen_tMax))));
  }

  // Functions for testing.
  if (ChecksEnabled[CK_StdCLibraryFunctionsTesterChecker]) {
    addToFunctionSummaryMap(
        "__two_constrained_args",
        Summary(ArgTypes{IntTy, IntTy}, RetType{IntTy}, EvalCallAsPure)
            .ArgConstraint(ArgumentCondition(0U, WithinRange, SingleValue(1)))
            .ArgConstraint(ArgumentCondition(1U, WithinRange, SingleValue(1))));
    addToFunctionSummaryMap(
        "__arg_constrained_twice",
        Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
            .ArgConstraint(ArgumentCondition(0U, OutOfRange, SingleValue(1)))
            .ArgConstraint(ArgumentCondition(0U, OutOfRange, SingleValue(2))));
    addToFunctionSummaryMap(
        "__defaultparam",
        Summary(ArgTypes{Irrelevant, IntTy}, RetType{IntTy}, EvalCallAsPure)
            .ArgConstraint(NotNull(ArgNo(0))));
    addToFunctionSummaryMap("__variadic",
                            Summary(ArgTypes{VoidPtrTy, ConstCharPtrTy},
                                    RetType{IntTy}, EvalCallAsPure)
                                .ArgConstraint(NotNull(ArgNo(0)))
                                .ArgConstraint(NotNull(ArgNo(1))));
    addToFunctionSummaryMap(
        "__buf_size_arg_constraint",
        Summary(ArgTypes{ConstVoidPtrTy, SizeTy}, RetType{IntTy},
                EvalCallAsPure)
            .ArgConstraint(
                BufferSize(/*Buffer=*/ArgNo(0), /*BufSize=*/ArgNo(1))));
    addToFunctionSummaryMap(
        "__buf_size_arg_constraint_mul",
        Summary(ArgTypes{ConstVoidPtrTy, SizeTy, SizeTy}, RetType{IntTy},
                EvalCallAsPure)
            .ArgConstraint(BufferSize(/*Buffer=*/ArgNo(0), /*BufSize=*/ArgNo(1),
                                      /*BufSizeMultiplier=*/ArgNo(2))));
  }
}

void ento::registerStdCLibraryFunctionsChecker(CheckerManager &mgr) {
  auto *Checker = mgr.registerChecker<StdLibraryFunctionsChecker>();
  Checker->DisplayLoadedSummaries =
      mgr.getAnalyzerOptions().getCheckerBooleanOption(
          Checker, "DisplayLoadedSummaries");
  Checker->ModelPOSIX =
      mgr.getAnalyzerOptions().getCheckerBooleanOption(Checker, "ModelPOSIX");
}

bool ento::shouldRegisterStdCLibraryFunctionsChecker(
    const CheckerManager &mgr) {
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
