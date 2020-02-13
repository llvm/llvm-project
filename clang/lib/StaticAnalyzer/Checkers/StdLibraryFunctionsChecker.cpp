//=== StdLibraryFunctionsChecker.cpp - Model standard functions -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This checker improves modeling of a few simple library functions.
// It does not generate warnings.
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
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerHelpers.h"

using namespace clang;
using namespace clang::ento;

namespace {
class StdLibraryFunctionsChecker : public Checker<check::PostCall, eval::Call> {
  /// Below is a series of typedefs necessary to define function specs.
  /// We avoid nesting types here because each additional qualifier
  /// would need to be repeated in every function spec.
  struct Summary;

  /// Specify how much the analyzer engine should entrust modeling this function
  /// to us. If he doesn't, he performs additional invalidations.
  enum InvalidationKind { NoEvalCall, EvalCallAsPure };

  /// A pair of ValueRangeKind and IntRangeVector would describe a range
  /// imposed on a particular argument or return value symbol.
  ///
  /// Given a range, should the argument stay inside or outside this range?
  /// The special `ComparesToArgument' value indicates that we should
  /// impose a constraint that involves other argument or return value symbols.
  enum ValueRangeKind { OutOfRange, WithinRange, ComparesToArgument };

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
  static const ArgNo Ret = std::numeric_limits<ArgNo>::max();

  /// Incapsulates a single range on a single symbol within a branch.
  class ValueRange {
    ArgNo ArgN;          // Argument to which we apply the range.
    ValueRangeKind Kind; // Kind of range definition.
    IntRangeVector Args; // Polymorphic arguments.

  public:
    ValueRange(ArgNo ArgN, ValueRangeKind Kind, const IntRangeVector &Args)
        : ArgN(ArgN), Kind(Kind), Args(Args) {}

    ArgNo getArgNo() const { return ArgN; }
    ValueRangeKind getKind() const { return Kind; }

    BinaryOperator::Opcode getOpcode() const {
      assert(Kind == ComparesToArgument);
      assert(Args.size() == 1);
      BinaryOperator::Opcode Op =
          static_cast<BinaryOperator::Opcode>(Args[0].first);
      assert(BinaryOperator::isComparisonOp(Op) &&
             "Only comparison ops are supported for ComparesToArgument");
      return Op;
    }

    ArgNo getOtherArgNo() const {
      assert(Kind == ComparesToArgument);
      assert(Args.size() == 1);
      return static_cast<ArgNo>(Args[0].second);
    }

    const IntRangeVector &getRanges() const {
      assert(Kind != ComparesToArgument);
      return Args;
    }

    // We avoid creating a virtual apply() method because
    // it makes initializer lists harder to write.
  private:
    ProgramStateRef applyAsOutOfRange(ProgramStateRef State,
                                      const CallEvent &Call,
                                      const Summary &Summary) const;
    ProgramStateRef applyAsWithinRange(ProgramStateRef State,
                                       const CallEvent &Call,
                                       const Summary &Summary) const;
    ProgramStateRef applyAsComparesToArgument(ProgramStateRef State,
                                              const CallEvent &Call,
                                              const Summary &Summary) const;

  public:
    ProgramStateRef apply(ProgramStateRef State, const CallEvent &Call,
                          const Summary &Summary) const {
      switch (Kind) {
      case OutOfRange:
        return applyAsOutOfRange(State, Call, Summary);
      case WithinRange:
        return applyAsWithinRange(State, Call, Summary);
      case ComparesToArgument:
        return applyAsComparesToArgument(State, Call, Summary);
      }
      llvm_unreachable("Unknown ValueRange kind!");
    }
  };

  /// The complete list of ranges that defines a single branch.
  typedef std::vector<ValueRange> ValueRangeSet;

  using ArgTypes = std::vector<QualType>;
  using Ranges = std::vector<ValueRangeSet>;

  /// Includes information about function prototype (which is necessary to
  /// ensure we're modeling the right function and casting values properly),
  /// approach to invalidation, and a list of branches - essentially, a list
  /// of list of ranges - essentially, a list of lists of lists of segments.
  struct Summary {
    const ArgTypes ArgTys;
    const QualType RetTy;
    const InvalidationKind InvalidationKd;
    Ranges Cases;
    ValueRangeSet ArgConstraints;

    Summary(ArgTypes ArgTys, QualType RetTy, InvalidationKind InvalidationKd)
        : ArgTys(ArgTys), RetTy(RetTy), InvalidationKd(InvalidationKd) {}

    Summary &Case(ValueRangeSet VRS) {
      Cases.push_back(VRS);
      return *this;
    }

  private:
    static void assertTypeSuitableForSummary(QualType T) {
      assert(!T->isVoidType() &&
             "We should have had no significant void types in the spec");
      assert(T.isCanonical() &&
             "We should only have canonical types in the spec");
      // FIXME: lift this assert (but not the ones above!)
      assert(T->isIntegralOrEnumerationType() &&
             "We only support integral ranges in the spec");
    }

  public:
    QualType getArgType(ArgNo ArgN) const {
      QualType T = (ArgN == Ret) ? RetTy : ArgTys[ArgN];
      assertTypeSuitableForSummary(T);
      return T;
    }

    /// Try our best to figure out if the call expression is the call of
    /// *the* library function to which this specification applies.
    bool matchesCall(const CallExpr *CE) const;
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
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  bool evalCall(const CallEvent &Call, CheckerContext &C) const;

private:
  Optional<Summary> findFunctionSummary(const FunctionDecl *FD,
                                        const CallExpr *CE,
                                        CheckerContext &C) const;

  void initFunctionSummaries(CheckerContext &C) const;
};
} // end of anonymous namespace

ProgramStateRef StdLibraryFunctionsChecker::ValueRange::applyAsOutOfRange(
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

ProgramStateRef StdLibraryFunctionsChecker::ValueRange::applyAsWithinRange(
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

ProgramStateRef
StdLibraryFunctionsChecker::ValueRange::applyAsComparesToArgument(
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

void StdLibraryFunctionsChecker::checkPostCall(const CallEvent &Call,
                                               CheckerContext &C) const {
  const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(Call.getDecl());
  if (!FD)
    return;

  const CallExpr *CE = dyn_cast_or_null<CallExpr>(Call.getOriginExpr());
  if (!CE)
    return;

  Optional<Summary> FoundSummary = findFunctionSummary(FD, CE, C);
  if (!FoundSummary)
    return;

  // Now apply ranges.
  const Summary &Summary = *FoundSummary;
  ProgramStateRef State = C.getState();

  // Apply case/branch specifications.
  for (const auto &VRS : Summary.Cases) {
    ProgramStateRef NewState = State;
    for (const auto &VR: VRS) {
      NewState = VR.apply(NewState, Call, Summary);
      if (!NewState)
        break;
    }

    if (NewState && NewState != State)
      C.addTransition(NewState);
  }
}

bool StdLibraryFunctionsChecker::evalCall(const CallEvent &Call,
                                          CheckerContext &C) const {
  const auto *FD = dyn_cast_or_null<FunctionDecl>(Call.getDecl());
  if (!FD)
    return false;

  const auto *CE = dyn_cast_or_null<CallExpr>(Call.getOriginExpr());
  if (!CE)
    return false;

  Optional<Summary> FoundSummary = findFunctionSummary(FD, CE, C);
  if (!FoundSummary)
    return false;

  const Summary &Summary = *FoundSummary;
  switch (Summary.InvalidationKd) {
  case EvalCallAsPure: {
    ProgramStateRef State = C.getState();
    const LocationContext *LC = C.getLocationContext();
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
    const CallExpr *CE) const {
  // Check number of arguments:
  if (CE->getNumArgs() != ArgTys.size())
    return false;

  // Check return type if relevant:
  if (!RetTy.isNull() && RetTy != CE->getType().getCanonicalType())
    return false;

  // Check argument types when relevant:
  for (size_t I = 0, E = ArgTys.size(); I != E; ++I) {
    QualType FormalT = ArgTys[I];
    // Null type marks irrelevant arguments.
    if (FormalT.isNull())
      continue;

    assertTypeSuitableForSummary(FormalT);

    QualType ActualT = StdLibraryFunctionsChecker::getArgType(CE, I);
    assert(ActualT.isCanonical());
    if (ActualT != FormalT)
      return false;
  }

  return true;
}

Optional<StdLibraryFunctionsChecker::Summary>
StdLibraryFunctionsChecker::findFunctionSummary(const FunctionDecl *FD,
                                                const CallExpr *CE,
                                                CheckerContext &C) const {
  // Note: we cannot always obtain FD from CE
  // (eg. virtual call, or call by pointer).
  assert(CE);

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
    if (Spec.matchesCall(CE))
      return Spec;

  return None;
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
      Irrelevant; // A placeholder, whenever we do not care about the type.
  const QualType IntTy = ACtx.IntTy;
  const QualType LongTy = ACtx.LongTy;
  const QualType LongLongTy = ACtx.LongLongTy;
  const QualType SizeTy = ACtx.getSizeType();

  const RangeInt IntMax = BVF.getMaxValue(IntTy).getLimitedValue();
  const RangeInt LongMax = BVF.getMaxValue(LongTy).getLimitedValue();
  const RangeInt LongLongMax = BVF.getMaxValue(LongLongTy).getLimitedValue();

  const RangeInt UCharMax =
      BVF.getMaxValue(ACtx.UnsignedCharTy).getLimitedValue();

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

  // Below are helper functions to create the summaries.
  auto ArgumentCondition = [](ArgNo ArgN, ValueRangeKind Kind,
                              IntRangeVector Ranges) -> ValueRange {
    ValueRange VR{ArgN, Kind, Ranges};
    return VR;
  };
  auto ReturnValueCondition = [](ValueRangeKind Kind,
                                 IntRangeVector Ranges) -> ValueRange {
    ValueRange VR{Ret, Kind, Ranges};
    return VR;
  };
  auto Range = [](RangeInt b, RangeInt e) {
    return IntRangeVector{std::pair<RangeInt, RangeInt>{b, e}};
  };
  auto SingleValue = [](RangeInt v) {
    return IntRangeVector{std::pair<RangeInt, RangeInt>{v, v}};
  };
  auto IsLessThan = [](ArgNo ArgN) { return IntRangeVector{{BO_LE, ArgN}}; };

  using RetType = QualType;

  // Templates for summaries that are reused by many functions.
  auto Getc = [&]() {
    return Summary(ArgTypes{Irrelevant}, RetType{IntTy}, NoEvalCall)
        .Case(
            {ReturnValueCondition(WithinRange, {{EOFv, EOFv}, {0, UCharMax}})});
  };
  auto Read = [&](RetType R, RangeInt Max) {
    return Summary(ArgTypes{Irrelevant, Irrelevant, SizeTy}, RetType{R},
                   NoEvalCall)
        .Case({ReturnValueCondition(ComparesToArgument, IsLessThan(2)),
               ReturnValueCondition(WithinRange, Range(-1, Max))});
  };
  auto Fread = [&]() {
    return Summary(ArgTypes{Irrelevant, Irrelevant, SizeTy, Irrelevant},
                   RetType{SizeTy}, NoEvalCall)
        .Case({
            ReturnValueCondition(ComparesToArgument, IsLessThan(2)),
        });
  };
  auto Getline = [&](RetType R, RangeInt Max) {
    return Summary(ArgTypes{Irrelevant, Irrelevant, Irrelevant}, RetType{R},
                   NoEvalCall)
        .Case({ReturnValueCondition(WithinRange, {{-1, -1}, {1, Max}})});
  };

  FunctionSummaryMap = {
      // The isascii() family of functions.
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
                  .Case({ArgumentCondition(0U, WithinRange, {{128, UCharMax}})})
                  .Case({ArgumentCondition(0U, OutOfRange,
                                           {{'0', '9'},
                                            {'A', 'Z'},
                                            {'a', 'z'},
                                            {128, UCharMax}}),
                         ReturnValueCondition(WithinRange, SingleValue(0))})},
      },
      {
          "isalpha",
          Summaries{
              Summary(ArgTypes{IntTy}, RetType{IntTy}, EvalCallAsPure)
                  .Case({ArgumentCondition(0U, WithinRange,
                                           {{'A', 'Z'}, {'a', 'z'}}),
                         ReturnValueCondition(OutOfRange, SingleValue(0))})
                  // The locale-specific range.
                  .Case({ArgumentCondition(0U, WithinRange, {{128, UCharMax}})})
                  .Case({ArgumentCondition(
                             0U, OutOfRange,
                             {{'A', 'Z'}, {'a', 'z'}, {128, UCharMax}}),
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
                  .Case({ArgumentCondition(0U, WithinRange, {{128, UCharMax}})})
                  // Is not an unsigned char.
                  .Case({ArgumentCondition(0U, OutOfRange, Range(0, UCharMax)),
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
                  .Case({ArgumentCondition(0U, WithinRange, {{128, UCharMax}})})
                  .Case({ArgumentCondition(
                             0U, OutOfRange,
                             {{9, 13}, {' ', ' '}, {128, UCharMax}}),
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
                  .Case({ArgumentCondition(0U, WithinRange, {{128, UCharMax}})})
                  // Other.
                  .Case({ArgumentCondition(0U, OutOfRange,
                                           {{'A', 'Z'}, {128, UCharMax}}),
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
                         WithinRange, {{EOFv, EOFv}, {0, UCharMax}})})}},

      // read()-like functions that never return more than buffer size.
      // We are not sure how ssize_t is defined on every platform, so we
      // provide three variants that should cover common cases.
      {"read", Summaries{Read(IntTy, IntMax), Read(LongTy, LongMax),
                         Read(LongLongTy, LongLongMax)}},
      {"write", Summaries{Read(IntTy, IntMax), Read(LongTy, LongMax),
                          Read(LongLongTy, LongLongMax)}},
      {"fread", Summaries{Fread()}},
      {"fwrite", Summaries{Fread()}},
      // getline()-like functions either fail or read at least the delimiter.
      {"getline", Summaries{Getline(IntTy, IntMax), Getline(LongTy, LongMax),
                            Getline(LongLongTy, LongLongMax)}},
      {"getdelim", Summaries{Getline(IntTy, IntMax), Getline(LongTy, LongMax),
                             Getline(LongLongTy, LongLongMax)}},
  };
}

void ento::registerStdCLibraryFunctionsChecker(CheckerManager &mgr) {
  // If this checker grows large enough to support C++, Objective-C, or other
  // standard libraries, we could use multiple register...Checker() functions,
  // which would register various checkers with the help of the same Checker
  // class, turning on different function summaries.
  mgr.registerChecker<StdLibraryFunctionsChecker>();
}

bool ento::shouldRegisterStdCLibraryFunctionsChecker(const LangOptions &LO) {
  return true;
}
