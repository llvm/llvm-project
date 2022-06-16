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
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerHelpers.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/DynamicExtent.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"

#include <string>

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

  /// Returns the string representation of an argument index.
  /// E.g.: (1) -> '1st arg', (2) - > '2nd arg'
  static SmallString<8> getArgDesc(ArgNo);

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

    // Return those arguments that should be tracked when we report a bug. By
    // default it is the argument that is constrained, however, in some special
    // cases we need to track other arguments as well. E.g. a buffer size might
    // be encoded in another argument.
    virtual std::vector<ArgNo> getArgsToTrack() const { return {ArgN}; }

    virtual StringRef getName() const = 0;

    // Give a description that explains the constraint to the user. Used when
    // the bug is reported.
    virtual std::string describe(ProgramStateRef State,
                                 const Summary &Summary) const {
      // There are some descendant classes that are not used as argument
      // constraints, e.g. ComparisonConstraint. In that case we can safely
      // ignore the implementation of this function.
      llvm_unreachable("Not implemented");
    }

  protected:
    ArgNo ArgN; // Argument to which we apply the constraint.

    /// Do polymorphic validation check on the constraint.
    virtual bool checkSpecificValidity(const FunctionDecl *FD) const {
      return true;
    }
  };

  /// Given a range, should the argument stay inside or outside this range?
  enum RangeKind { OutOfRange, WithinRange };

  /// Encapsulates a range on a single symbol.
  class RangeConstraint : public ValueConstraint {
    RangeKind Kind;
    // A range is formed as a set of intervals (sub-ranges).
    // E.g. {['A', 'Z'], ['a', 'z']}
    //
    // The default constructed RangeConstraint has an empty range set, applying
    // such constraint does not involve any assumptions, thus the State remains
    // unchanged. This is meaningful, if the range is dependent on a looked up
    // type (e.g. [0, Socklen_tMax]). If the type is not found, then the range
    // is default initialized to be empty.
    IntRangeVector Ranges;

  public:
    StringRef getName() const override { return "Range"; }
    RangeConstraint(ArgNo ArgN, RangeKind Kind, const IntRangeVector &Ranges)
        : ValueConstraint(ArgN), Kind(Kind), Ranges(Ranges) {}

    std::string describe(ProgramStateRef State,
                         const Summary &Summary) const override;

    const IntRangeVector &getRanges() const { return Ranges; }

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
    virtual StringRef getName() const override { return "Comparison"; };
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
    std::string describe(ProgramStateRef State,
                         const Summary &Summary) const override;
    StringRef getName() const override { return "NonNull"; }
    ProgramStateRef apply(ProgramStateRef State, const CallEvent &Call,
                          const Summary &Summary,
                          CheckerContext &C) const override {
      SVal V = getArgSVal(Call, getArgNo());
      if (V.isUndef())
        return State;

      DefinedOrUnknownSVal L = V.castAs<DefinedOrUnknownSVal>();
      if (!isa<Loc>(L))
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

  // Represents a buffer argument with an additional size constraint. The
  // constraint may be a concrete value, or a symbolic value in an argument.
  // Example 1. Concrete value as the minimum buffer size.
  //   char *asctime_r(const struct tm *restrict tm, char *restrict buf);
  //   // `buf` size must be at least 26 bytes according the POSIX standard.
  // Example 2. Argument as a buffer size.
  //   ctime_s(char *buffer, rsize_t bufsz, const time_t *time);
  // Example 3. The size is computed as a multiplication of other args.
  //   size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
  //   // Here, ptr is the buffer, and its minimum size is `size * nmemb`.
  class BufferSizeConstraint : public ValueConstraint {
    // The concrete value which is the minimum size for the buffer.
    llvm::Optional<llvm::APSInt> ConcreteSize;
    // The argument which holds the size of the buffer.
    llvm::Optional<ArgNo> SizeArgN;
    // The argument which is a multiplier to size. This is set in case of
    // `fread` like functions where the size is computed as a multiplication of
    // two arguments.
    llvm::Optional<ArgNo> SizeMultiplierArgN;
    // The operator we use in apply. This is negated in negate().
    BinaryOperator::Opcode Op = BO_LE;

  public:
    StringRef getName() const override { return "BufferSize"; }
    BufferSizeConstraint(ArgNo Buffer, llvm::APSInt BufMinSize)
        : ValueConstraint(Buffer), ConcreteSize(BufMinSize) {}
    BufferSizeConstraint(ArgNo Buffer, ArgNo BufSize)
        : ValueConstraint(Buffer), SizeArgN(BufSize) {}
    BufferSizeConstraint(ArgNo Buffer, ArgNo BufSize, ArgNo BufSizeMultiplier)
        : ValueConstraint(Buffer), SizeArgN(BufSize),
          SizeMultiplierArgN(BufSizeMultiplier) {}

    std::vector<ArgNo> getArgsToTrack() const override {
      std::vector<ArgNo> Result{ArgN};
      if (SizeArgN)
        Result.push_back(*SizeArgN);
      if (SizeMultiplierArgN)
        Result.push_back(*SizeMultiplierArgN);
      return Result;
    }

    std::string describe(ProgramStateRef State,
                         const Summary &Summary) const override;

    ProgramStateRef apply(ProgramStateRef State, const CallEvent &Call,
                          const Summary &Summary,
                          CheckerContext &C) const override {
      SValBuilder &SvalBuilder = C.getSValBuilder();
      // The buffer argument.
      SVal BufV = getArgSVal(Call, getArgNo());

      // Get the size constraint.
      const SVal SizeV = [this, &State, &Call, &Summary, &SvalBuilder]() {
        if (ConcreteSize) {
          return SVal(SvalBuilder.makeIntVal(*ConcreteSize));
        }
        assert(SizeArgN && "The constraint must be either a concrete value or "
                           "encoded in an argument.");
        // The size argument.
        SVal SizeV = getArgSVal(Call, *SizeArgN);
        // Multiply with another argument if given.
        if (SizeMultiplierArgN) {
          SVal SizeMulV = getArgSVal(Call, *SizeMultiplierArgN);
          SizeV = SvalBuilder.evalBinOp(State, BO_Mul, SizeV, SizeMulV,
                                        Summary.getArgType(*SizeArgN));
        }
        return SizeV;
      }();

      // The dynamic size of the buffer argument, got from the analyzer engine.
      SVal BufDynSize = getDynamicExtentWithOffset(State, BufV);

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
  using ConstraintSet = std::vector<ValueConstraintPtr>;

  /// A single branch of a function summary.
  ///
  /// A branch is defined by a series of constraints - "assumptions" -
  /// that together form a single possible outcome of invoking the function.
  /// When static analyzer considers a branch, it tries to introduce
  /// a child node in the Exploded Graph. The child node has to include
  /// constraints that define the branch. If the constraints contradict
  /// existing constraints in the state, the node is not created and the branch
  /// is dropped; otherwise it's queued for future exploration.
  /// The branch is accompanied by a note text that may be displayed
  /// to the user when a bug is found on a path that takes this branch.
  ///
  /// For example, consider the branches in `isalpha(x)`:
  ///   Branch 1)
  ///     x is in range ['A', 'Z'] or in ['a', 'z']
  ///     then the return value is not 0. (I.e. out-of-range [0, 0])
  ///     and the note may say "Assuming the character is alphabetical"
  ///   Branch 2)
  ///     x is out-of-range ['A', 'Z'] and out-of-range ['a', 'z']
  ///     then the return value is 0
  ///     and the note may say "Assuming the character is non-alphabetical".
  class SummaryCase {
    ConstraintSet Constraints;
    StringRef Note;

  public:
    SummaryCase(ConstraintSet &&Constraints, StringRef Note)
        : Constraints(std::move(Constraints)), Note(Note) {}

    SummaryCase(const ConstraintSet &Constraints, StringRef Note)
        : Constraints(Constraints), Note(Note) {}

    const ConstraintSet &getConstraints() const { return Constraints; }
    StringRef getNote() const { return Note; }
  };

  using ArgTypes = std::vector<Optional<QualType>>;
  using RetType = Optional<QualType>;

  // A placeholder type, we use it whenever we do not care about the concrete
  // type in a Signature.
  const QualType Irrelevant{};
  bool static isIrrelevant(QualType T) { return T.isNull(); }

  // The signature of a function we want to describe with a summary. This is a
  // concessive signature, meaning there may be irrelevant types in the
  // signature which we do not check against a function with concrete types.
  // All types in the spec need to be canonical.
  class Signature {
    using ArgQualTypes = std::vector<QualType>;
    ArgQualTypes ArgTys;
    QualType RetTy;
    // True if any component type is not found by lookup.
    bool Invalid = false;

  public:
    // Construct a signature from optional types. If any of the optional types
    // are not set then the signature will be invalid.
    Signature(ArgTypes ArgTys, RetType RetTy) {
      for (Optional<QualType> Arg : ArgTys) {
        if (!Arg) {
          Invalid = true;
          return;
        } else {
          assertArgTypeSuitableForSignature(*Arg);
          this->ArgTys.push_back(*Arg);
        }
      }
      if (!RetTy) {
        Invalid = true;
        return;
      } else {
        assertRetTypeSuitableForSignature(*RetTy);
        this->RetTy = *RetTy;
      }
    }

    bool isInvalid() const { return Invalid; }
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

  using SummaryCases = std::vector<SummaryCase>;

  /// A summary includes information about
  ///   * function prototype (signature)
  ///   * approach to invalidation,
  ///   * a list of branches - so, a list of list of ranges,
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
    const InvalidationKind InvalidationKd;
    SummaryCases Cases;
    ConstraintSet ArgConstraints;

    // The function to which the summary applies. This is set after lookup and
    // match to the signature.
    const FunctionDecl *FD = nullptr;

  public:
    Summary(InvalidationKind InvalidationKd) : InvalidationKd(InvalidationKd) {}

    Summary &Case(ConstraintSet &&CS, StringRef Note = "") {
      Cases.push_back(SummaryCase(std::move(CS), Note));
      return *this;
    }
    Summary &Case(const ConstraintSet &CS, StringRef Note = "") {
      Cases.push_back(SummaryCase(CS, Note));
      return *this;
    }
    Summary &ArgConstraint(ValueConstraintPtr VC) {
      assert(VC->getArgNo() != Ret &&
             "Arg constraint should not refer to the return value");
      ArgConstraints.push_back(VC);
      return *this;
    }

    InvalidationKind getInvalidationKd() const { return InvalidationKd; }
    const SummaryCases &getCases() const { return Cases; }
    const ConstraintSet &getArgConstraints() const { return ArgConstraints; }

    QualType getArgType(ArgNo ArgN) const {
      return StdLibraryFunctionsChecker::getArgType(FD, ArgN);
    }

    // Returns true if the summary should be applied to the given function.
    // And if yes then store the function declaration.
    bool matchesAndSet(const Signature &Sign, const FunctionDecl *FD) {
      bool Result = Sign.matches(FD) && validateByConstraints(FD);
      if (Result) {
        assert(!this->FD && "FD must not be set more than once");
        this->FD = FD;
      }
      return Result;
    }

  private:
    // Once we know the exact type of the function then do validation check on
    // all the given constraints.
    bool validateByConstraints(const FunctionDecl *FD) const {
      for (const SummaryCase &Case : Cases)
        for (const ValueConstraintPtr &Constraint : Case.getConstraints())
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
  mutable bool SummariesInitialized = false;

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
  bool ChecksEnabled[CK_NumCheckKinds] = {false};
  CheckerNameRef CheckNames[CK_NumCheckKinds];

  bool DisplayLoadedSummaries = false;
  bool ModelPOSIX = false;
  bool ShouldAssumeControlledEnvironment = false;

private:
  Optional<Summary> findFunctionSummary(const FunctionDecl *FD,
                                        CheckerContext &C) const;
  Optional<Summary> findFunctionSummary(const CallEvent &Call,
                                        CheckerContext &C) const;

  void initFunctionSummaries(CheckerContext &C) const;

  void reportBug(const CallEvent &Call, ExplodedNode *N,
                 const ValueConstraint *VC, const Summary &Summary,
                 CheckerContext &C) const {
    if (!ChecksEnabled[CK_StdCLibraryFunctionArgsChecker])
      return;
    std::string Msg =
        (Twine("Function argument constraint is not satisfied, constraint: ") +
         VC->getName().data())
            .str();
    if (!BT_InvalidArg)
      BT_InvalidArg = std::make_unique<BugType>(
          CheckNames[CK_StdCLibraryFunctionArgsChecker],
          "Unsatisfied argument constraints", categories::LogicError);
    auto R = std::make_unique<PathSensitiveBugReport>(*BT_InvalidArg, Msg, N);

    for (ArgNo ArgN : VC->getArgsToTrack())
      bugreporter::trackExpressionValue(N, Call.getArgExpr(ArgN), *R);

    // Highlight the range of the argument that was violated.
    R->addRange(Call.getArgSourceRange(VC->getArgNo()));

    // Describe the argument constraint in a note.
    R->addNote(VC->describe(C.getState(), Summary), R->getLocation(),
               Call.getArgSourceRange(VC->getArgNo()));

    C.emitReport(std::move(R));
  }
};

const StdLibraryFunctionsChecker::ArgNo StdLibraryFunctionsChecker::Ret =
    std::numeric_limits<ArgNo>::max();

} // end of anonymous namespace

static BasicValueFactory &getBVF(ProgramStateRef State) {
  ProgramStateManager &Mgr = State->getStateManager();
  SValBuilder &SVB = Mgr.getSValBuilder();
  return SVB.getBasicValueFactory();
}

std::string StdLibraryFunctionsChecker::NotNullConstraint::describe(
    ProgramStateRef State, const Summary &Summary) const {
  SmallString<48> Result;
  Result += "The ";
  Result += getArgDesc(ArgN);
  Result += " should not be NULL";
  return Result.c_str();
}

std::string StdLibraryFunctionsChecker::RangeConstraint::describe(
    ProgramStateRef State, const Summary &Summary) const {

  BasicValueFactory &BVF = getBVF(State);

  QualType T = Summary.getArgType(getArgNo());
  SmallString<48> Result;
  Result += "The ";
  Result += getArgDesc(ArgN);
  Result += " should be ";

  // Range kind as a string.
  Kind == OutOfRange ? Result += "out of" : Result += "within";

  // Get the range values as a string.
  Result += " the range ";
  if (Ranges.size() > 1)
    Result += "[";
  unsigned I = Ranges.size();
  for (const std::pair<RangeInt, RangeInt> &R : Ranges) {
    Result += "[";
    const llvm::APSInt &Min = BVF.getValue(R.first, T);
    const llvm::APSInt &Max = BVF.getValue(R.second, T);
    Min.toString(Result);
    Result += ", ";
    Max.toString(Result);
    Result += "]";
    if (--I > 0)
      Result += ", ";
  }
  if (Ranges.size() > 1)
    Result += "]";

  return Result.c_str();
}

SmallString<8>
StdLibraryFunctionsChecker::getArgDesc(StdLibraryFunctionsChecker::ArgNo ArgN) {
  SmallString<8> Result;
  Result += std::to_string(ArgN + 1);
  Result += llvm::getOrdinalSuffix(ArgN + 1);
  Result += " arg";
  return Result;
}

std::string StdLibraryFunctionsChecker::BufferSizeConstraint::describe(
    ProgramStateRef State, const Summary &Summary) const {
  SmallString<96> Result;
  Result += "The size of the ";
  Result += getArgDesc(ArgN);
  Result += " should be equal to or less than the value of ";
  if (ConcreteSize) {
    ConcreteSize->toString(Result);
  } else if (SizeArgN) {
    Result += "the ";
    Result += getArgDesc(*SizeArgN);
    if (SizeMultiplierArgN) {
      Result += " times the ";
      Result += getArgDesc(*SizeMultiplierArgN);
    }
  }
  return Result.c_str();
}

ProgramStateRef StdLibraryFunctionsChecker::RangeConstraint::applyAsOutOfRange(
    ProgramStateRef State, const CallEvent &Call,
    const Summary &Summary) const {
  if (Ranges.empty())
    return State;

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
  if (Ranges.empty())
    return State;

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
        reportBug(Call, N, Constraint.get(), Summary, C);
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
  const ExplodedNode *Node = C.getPredecessor();

  // Apply case/branch specifications.
  for (const SummaryCase &Case : Summary.getCases()) {
    ProgramStateRef NewState = State;
    for (const ValueConstraintPtr &Constraint : Case.getConstraints()) {
      NewState = Constraint->apply(NewState, Call, Summary, C);
      if (!NewState)
        break;
    }

    if (NewState && NewState != State) {
      StringRef Note = Case.getNote();
      const NoteTag *Tag = C.getNoteTag(
          // Sorry couldn't help myself.
          [Node, Note]() {
            // Don't emit "Assuming..." note when we ended up
            // knowing in advance which branch is taken.
            return (Node->succ_size() > 1) ? Note.str() : "";
          },
          /*IsPrunable=*/true);
      C.addTransition(NewState, Tag);
    }
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
    const auto *CE = cast<CallExpr>(Call.getOriginExpr());
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
  assert(!isInvalid());
  // Check the number of arguments.
  if (FD->param_size() != ArgTys.size())
    return false;

  // The "restrict" keyword is illegal in C++, however, many libc
  // implementations use the "__restrict" compiler intrinsic in functions
  // prototypes. The "__restrict" keyword qualifies a type as a restricted type
  // even in C++.
  // In case of any non-C99 languages, we don't want to match based on the
  // restrict qualifier because we cannot know if the given libc implementation
  // qualifies the paramter type or not.
  auto RemoveRestrict = [&FD](QualType T) {
    if (!FD->getASTContext().getLangOpts().C99)
      T.removeLocalRestrict();
    return T;
  };

  // Check the return type.
  if (!isIrrelevant(RetTy)) {
    QualType FDRetTy = RemoveRestrict(FD->getReturnType().getCanonicalType());
    if (RetTy != FDRetTy)
      return false;
  }

  // Check the argument types.
  for (size_t I = 0, E = ArgTys.size(); I != E; ++I) {
    QualType ArgTy = ArgTys[I];
    if (isIrrelevant(ArgTy))
      continue;
    QualType FDArgTy =
        RemoveRestrict(FD->getParamDecl(I)->getType().getCanonicalType());
    if (ArgTy != FDArgTy)
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

void StdLibraryFunctionsChecker::initFunctionSummaries(
    CheckerContext &C) const {
  if (SummariesInitialized)
    return;

  SValBuilder &SVB = C.getSValBuilder();
  BasicValueFactory &BVF = SVB.getBasicValueFactory();
  const ASTContext &ACtx = BVF.getContext();

  // Helper class to lookup a type by its name.
  class LookupType {
    const ASTContext &ACtx;

  public:
    LookupType(const ASTContext &ACtx) : ACtx(ACtx) {}

    // Find the type. If not found then the optional is not set.
    llvm::Optional<QualType> operator()(StringRef Name) {
      IdentifierInfo &II = ACtx.Idents.get(Name);
      auto LookupRes = ACtx.getTranslationUnitDecl()->lookup(&II);
      if (LookupRes.empty())
        return None;

      // Prioritze typedef declarations.
      // This is needed in case of C struct typedefs. E.g.:
      //   typedef struct FILE FILE;
      // In this case, we have a RecordDecl 'struct FILE' with the name 'FILE'
      // and we have a TypedefDecl with the name 'FILE'.
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
  } lookupTy(ACtx);

  // Below are auxiliary classes to handle optional types that we get as a
  // result of the lookup.
  class GetRestrictTy {
    const ASTContext &ACtx;

  public:
    GetRestrictTy(const ASTContext &ACtx) : ACtx(ACtx) {}
    QualType operator()(QualType Ty) {
      return ACtx.getLangOpts().C99 ? ACtx.getRestrictType(Ty) : Ty;
    }
    Optional<QualType> operator()(Optional<QualType> Ty) {
      if (Ty)
        return operator()(*Ty);
      return None;
    }
  } getRestrictTy(ACtx);
  class GetPointerTy {
    const ASTContext &ACtx;

  public:
    GetPointerTy(const ASTContext &ACtx) : ACtx(ACtx) {}
    QualType operator()(QualType Ty) { return ACtx.getPointerType(Ty); }
    Optional<QualType> operator()(Optional<QualType> Ty) {
      if (Ty)
        return operator()(*Ty);
      return None;
    }
  } getPointerTy(ACtx);
  class {
  public:
    Optional<QualType> operator()(Optional<QualType> Ty) {
      return Ty ? Optional<QualType>(Ty->withConst()) : None;
    }
    QualType operator()(QualType Ty) { return Ty.withConst(); }
  } getConstTy;
  class GetMaxValue {
    BasicValueFactory &BVF;

  public:
    GetMaxValue(BasicValueFactory &BVF) : BVF(BVF) {}
    Optional<RangeInt> operator()(QualType Ty) {
      return BVF.getMaxValue(Ty).getLimitedValue();
    }
    Optional<RangeInt> operator()(Optional<QualType> Ty) {
      if (Ty) {
        return operator()(*Ty);
      }
      return None;
    }
  } getMaxValue(BVF);

  // These types are useful for writing specifications quickly,
  // New specifications should probably introduce more types.
  // Some types are hard to obtain from the AST, eg. "ssize_t".
  // In such cases it should be possible to provide multiple variants
  // of function summary for common cases (eg. ssize_t could be int or long
  // or long long, so three summary variants would be enough).
  // Of course, function variants are also useful for C++ overloads.
  const QualType VoidTy = ACtx.VoidTy;
  const QualType CharTy = ACtx.CharTy;
  const QualType WCharTy = ACtx.WCharTy;
  const QualType IntTy = ACtx.IntTy;
  const QualType UnsignedIntTy = ACtx.UnsignedIntTy;
  const QualType LongTy = ACtx.LongTy;
  const QualType SizeTy = ACtx.getSizeType();

  const QualType VoidPtrTy = getPointerTy(VoidTy); // void *
  const QualType IntPtrTy = getPointerTy(IntTy);   // int *
  const QualType UnsignedIntPtrTy =
      getPointerTy(UnsignedIntTy); // unsigned int *
  const QualType VoidPtrRestrictTy = getRestrictTy(VoidPtrTy);
  const QualType ConstVoidPtrTy =
      getPointerTy(getConstTy(VoidTy));            // const void *
  const QualType CharPtrTy = getPointerTy(CharTy); // char *
  const QualType CharPtrRestrictTy = getRestrictTy(CharPtrTy);
  const QualType ConstCharPtrTy =
      getPointerTy(getConstTy(CharTy)); // const char *
  const QualType ConstCharPtrRestrictTy = getRestrictTy(ConstCharPtrTy);
  const QualType Wchar_tPtrTy = getPointerTy(WCharTy); // wchar_t *
  const QualType ConstWchar_tPtrTy =
      getPointerTy(getConstTy(WCharTy)); // const wchar_t *
  const QualType ConstVoidPtrRestrictTy = getRestrictTy(ConstVoidPtrTy);
  const QualType SizePtrTy = getPointerTy(SizeTy);
  const QualType SizePtrRestrictTy = getRestrictTy(SizePtrTy);

  const RangeInt IntMax = BVF.getMaxValue(IntTy).getLimitedValue();
  const RangeInt UnsignedIntMax =
      BVF.getMaxValue(UnsignedIntTy).getLimitedValue();
  const RangeInt LongMax = BVF.getMaxValue(LongTy).getLimitedValue();
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
    bool operator()(StringRef Name, Signature Sign, Summary Sum) {
      if (Sign.isInvalid())
        return false;
      IdentifierInfo &II = ACtx.Idents.get(Name);
      auto LookupRes = ACtx.getTranslationUnitDecl()->lookup(&II);
      if (LookupRes.empty())
        return false;
      for (Decl *D : LookupRes) {
        if (auto *FD = dyn_cast<FunctionDecl>(D)) {
          if (Sum.matchesAndSet(Sign, FD)) {
            auto Res = Map.insert({FD->getCanonicalDecl(), Sum});
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
    // Add the same summary for different names with the Signature explicitly
    // given.
    void operator()(std::vector<StringRef> Names, Signature Sign, Summary Sum) {
      for (StringRef Name : Names)
        operator()(Name, Sign, Sum);
    }
  } addToFunctionSummaryMap(ACtx, FunctionSummaryMap, DisplayLoadedSummaries);

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
  struct {
    auto operator()(RangeInt b, RangeInt e) {
      return IntRangeVector{std::pair<RangeInt, RangeInt>{b, e}};
    }
    auto operator()(RangeInt b, Optional<RangeInt> e) {
      if (e)
        return IntRangeVector{std::pair<RangeInt, RangeInt>{b, *e}};
      return IntRangeVector{};
    }
    auto operator()(std::pair<RangeInt, RangeInt> i0,
                    std::pair<RangeInt, Optional<RangeInt>> i1) {
      if (i1.second)
        return IntRangeVector{i0, {i1.first, *(i1.second)}};
      return IntRangeVector{i0};
    }
  } Range;
  auto SingleValue = [](RangeInt v) {
    return IntRangeVector{std::pair<RangeInt, RangeInt>{v, v}};
  };
  auto LessThanOrEq = BO_LE;
  auto NotNull = [&](ArgNo ArgN) {
    return std::make_shared<NotNullConstraint>(ArgN);
  };

  Optional<QualType> FileTy = lookupTy("FILE");
  Optional<QualType> FilePtrTy = getPointerTy(FileTy);
  Optional<QualType> FilePtrRestrictTy = getRestrictTy(FilePtrTy);

  // We are finally ready to define specifications for all supported functions.
  //
  // Argument ranges should always cover all variants. If return value
  // is completely unknown, omit it from the respective range set.
  //
  // Every item in the list of range sets represents a particular
  // execution path the analyzer would need to explore once
  // the call is modeled - a new program state is constructed
  // for every range set, and each range line in the range set
  // corresponds to a specific constraint within this state.

  // The isascii() family of functions.
  // The behavior is undefined if the value of the argument is not
  // representable as unsigned char or is not equal to EOF. See e.g. C99
  // 7.4.1.2 The isalpha function (p: 181-182).
  addToFunctionSummaryMap(
      "isalnum", Signature(ArgTypes{IntTy}, RetType{IntTy}),
      Summary(EvalCallAsPure)
          // Boils down to isupper() or islower() or isdigit().
          .Case({ArgumentCondition(0U, WithinRange,
                                   {{'0', '9'}, {'A', 'Z'}, {'a', 'z'}}),
                 ReturnValueCondition(OutOfRange, SingleValue(0))},
                "Assuming the character is alphanumeric")
          // The locale-specific range.
          // No post-condition. We are completely unaware of
          // locale-specific return values.
          .Case({ArgumentCondition(0U, WithinRange, {{128, UCharRangeMax}})})
          .Case(
              {ArgumentCondition(
                   0U, OutOfRange,
                   {{'0', '9'}, {'A', 'Z'}, {'a', 'z'}, {128, UCharRangeMax}}),
               ReturnValueCondition(WithinRange, SingleValue(0))},
              "Assuming the character is non-alphanumeric")
          .ArgConstraint(ArgumentCondition(
              0U, WithinRange, {{EOFv, EOFv}, {0, UCharRangeMax}})));
  addToFunctionSummaryMap(
      "isalpha", Signature(ArgTypes{IntTy}, RetType{IntTy}),
      Summary(EvalCallAsPure)
          .Case({ArgumentCondition(0U, WithinRange, {{'A', 'Z'}, {'a', 'z'}}),
                 ReturnValueCondition(OutOfRange, SingleValue(0))},
                "Assuming the character is alphabetical")
          // The locale-specific range.
          .Case({ArgumentCondition(0U, WithinRange, {{128, UCharRangeMax}})})
          .Case({ArgumentCondition(
                     0U, OutOfRange,
                     {{'A', 'Z'}, {'a', 'z'}, {128, UCharRangeMax}}),
                 ReturnValueCondition(WithinRange, SingleValue(0))},
                "Assuming the character is non-alphabetical"));
  addToFunctionSummaryMap(
      "isascii", Signature(ArgTypes{IntTy}, RetType{IntTy}),
      Summary(EvalCallAsPure)
          .Case({ArgumentCondition(0U, WithinRange, Range(0, 127)),
                 ReturnValueCondition(OutOfRange, SingleValue(0))},
                "Assuming the character is an ASCII character")
          .Case({ArgumentCondition(0U, OutOfRange, Range(0, 127)),
                 ReturnValueCondition(WithinRange, SingleValue(0))},
                "Assuming the character is not an ASCII character"));
  addToFunctionSummaryMap(
      "isblank", Signature(ArgTypes{IntTy}, RetType{IntTy}),
      Summary(EvalCallAsPure)
          .Case({ArgumentCondition(0U, WithinRange, {{'\t', '\t'}, {' ', ' '}}),
                 ReturnValueCondition(OutOfRange, SingleValue(0))},
                "Assuming the character is a blank character")
          .Case({ArgumentCondition(0U, OutOfRange, {{'\t', '\t'}, {' ', ' '}}),
                 ReturnValueCondition(WithinRange, SingleValue(0))},
                "Assuming the character is not a blank character"));
  addToFunctionSummaryMap(
      "iscntrl", Signature(ArgTypes{IntTy}, RetType{IntTy}),
      Summary(EvalCallAsPure)
          .Case({ArgumentCondition(0U, WithinRange, {{0, 32}, {127, 127}}),
                 ReturnValueCondition(OutOfRange, SingleValue(0))},
                "Assuming the character is a control character")
          .Case({ArgumentCondition(0U, OutOfRange, {{0, 32}, {127, 127}}),
                 ReturnValueCondition(WithinRange, SingleValue(0))},
                "Assuming the character is not a control character"));
  addToFunctionSummaryMap(
      "isdigit", Signature(ArgTypes{IntTy}, RetType{IntTy}),
      Summary(EvalCallAsPure)
          .Case({ArgumentCondition(0U, WithinRange, Range('0', '9')),
                 ReturnValueCondition(OutOfRange, SingleValue(0))},
                "Assuming the character is a digit")
          .Case({ArgumentCondition(0U, OutOfRange, Range('0', '9')),
                 ReturnValueCondition(WithinRange, SingleValue(0))},
                "Assuming the character is not a digit"));
  addToFunctionSummaryMap(
      "isgraph", Signature(ArgTypes{IntTy}, RetType{IntTy}),
      Summary(EvalCallAsPure)
          .Case({ArgumentCondition(0U, WithinRange, Range(33, 126)),
                 ReturnValueCondition(OutOfRange, SingleValue(0))},
                "Assuming the character has graphical representation")
          .Case(
              {ArgumentCondition(0U, OutOfRange, Range(33, 126)),
               ReturnValueCondition(WithinRange, SingleValue(0))},
              "Assuming the character does not have graphical representation"));
  addToFunctionSummaryMap(
      "islower", Signature(ArgTypes{IntTy}, RetType{IntTy}),
      Summary(EvalCallAsPure)
          // Is certainly lowercase.
          .Case({ArgumentCondition(0U, WithinRange, Range('a', 'z')),
                 ReturnValueCondition(OutOfRange, SingleValue(0))},
                "Assuming the character is a lowercase letter")
          // Is ascii but not lowercase.
          .Case({ArgumentCondition(0U, WithinRange, Range(0, 127)),
                 ArgumentCondition(0U, OutOfRange, Range('a', 'z')),
                 ReturnValueCondition(WithinRange, SingleValue(0))},
                "Assuming the character is not a lowercase letter")
          // The locale-specific range.
          .Case({ArgumentCondition(0U, WithinRange, {{128, UCharRangeMax}})})
          // Is not an unsigned char.
          .Case({ArgumentCondition(0U, OutOfRange, Range(0, UCharRangeMax)),
                 ReturnValueCondition(WithinRange, SingleValue(0))}));
  addToFunctionSummaryMap(
      "isprint", Signature(ArgTypes{IntTy}, RetType{IntTy}),
      Summary(EvalCallAsPure)
          .Case({ArgumentCondition(0U, WithinRange, Range(32, 126)),
                 ReturnValueCondition(OutOfRange, SingleValue(0))},
                "Assuming the character is printable")
          .Case({ArgumentCondition(0U, OutOfRange, Range(32, 126)),
                 ReturnValueCondition(WithinRange, SingleValue(0))},
                "Assuming the character is non-printable"));
  addToFunctionSummaryMap(
      "ispunct", Signature(ArgTypes{IntTy}, RetType{IntTy}),
      Summary(EvalCallAsPure)
          .Case({ArgumentCondition(
                     0U, WithinRange,
                     {{'!', '/'}, {':', '@'}, {'[', '`'}, {'{', '~'}}),
                 ReturnValueCondition(OutOfRange, SingleValue(0))},
                "Assuming the character is a punctuation mark")
          .Case({ArgumentCondition(
                     0U, OutOfRange,
                     {{'!', '/'}, {':', '@'}, {'[', '`'}, {'{', '~'}}),
                 ReturnValueCondition(WithinRange, SingleValue(0))},
                "Assuming the character is not a punctuation mark"));
  addToFunctionSummaryMap(
      "isspace", Signature(ArgTypes{IntTy}, RetType{IntTy}),
      Summary(EvalCallAsPure)
          // Space, '\f', '\n', '\r', '\t', '\v'.
          .Case({ArgumentCondition(0U, WithinRange, {{9, 13}, {' ', ' '}}),
                 ReturnValueCondition(OutOfRange, SingleValue(0))},
                "Assuming the character is a whitespace character")
          // The locale-specific range.
          .Case({ArgumentCondition(0U, WithinRange, {{128, UCharRangeMax}})})
          .Case({ArgumentCondition(0U, OutOfRange,
                                   {{9, 13}, {' ', ' '}, {128, UCharRangeMax}}),
                 ReturnValueCondition(WithinRange, SingleValue(0))},
                "Assuming the character is not a whitespace character"));
  addToFunctionSummaryMap(
      "isupper", Signature(ArgTypes{IntTy}, RetType{IntTy}),
      Summary(EvalCallAsPure)
          // Is certainly uppercase.
          .Case({ArgumentCondition(0U, WithinRange, Range('A', 'Z')),
                 ReturnValueCondition(OutOfRange, SingleValue(0))},
                "Assuming the character is an uppercase letter")
          // The locale-specific range.
          .Case({ArgumentCondition(0U, WithinRange, {{128, UCharRangeMax}})})
          // Other.
          .Case({ArgumentCondition(0U, OutOfRange,
                                   {{'A', 'Z'}, {128, UCharRangeMax}}),
                 ReturnValueCondition(WithinRange, SingleValue(0))},
                "Assuming the character is not an uppercase letter"));
  addToFunctionSummaryMap(
      "isxdigit", Signature(ArgTypes{IntTy}, RetType{IntTy}),
      Summary(EvalCallAsPure)
          .Case({ArgumentCondition(0U, WithinRange,
                                   {{'0', '9'}, {'A', 'F'}, {'a', 'f'}}),
                 ReturnValueCondition(OutOfRange, SingleValue(0))},
                "Assuming the character is a hexadecimal digit")
          .Case({ArgumentCondition(0U, OutOfRange,
                                   {{'0', '9'}, {'A', 'F'}, {'a', 'f'}}),
                 ReturnValueCondition(WithinRange, SingleValue(0))},
                "Assuming the character is not a hexadecimal digit"));
  addToFunctionSummaryMap(
      "toupper", Signature(ArgTypes{IntTy}, RetType{IntTy}),
      Summary(EvalCallAsPure)
          .ArgConstraint(ArgumentCondition(
              0U, WithinRange, {{EOFv, EOFv}, {0, UCharRangeMax}})));
  addToFunctionSummaryMap(
      "tolower", Signature(ArgTypes{IntTy}, RetType{IntTy}),
      Summary(EvalCallAsPure)
          .ArgConstraint(ArgumentCondition(
              0U, WithinRange, {{EOFv, EOFv}, {0, UCharRangeMax}})));
  addToFunctionSummaryMap(
      "toascii", Signature(ArgTypes{IntTy}, RetType{IntTy}),
      Summary(EvalCallAsPure)
          .ArgConstraint(ArgumentCondition(
              0U, WithinRange, {{EOFv, EOFv}, {0, UCharRangeMax}})));

  // The getc() family of functions that returns either a char or an EOF.
  addToFunctionSummaryMap(
      {"getc", "fgetc"}, Signature(ArgTypes{FilePtrTy}, RetType{IntTy}),
      Summary(NoEvalCall)
          .Case({ReturnValueCondition(WithinRange,
                                      {{EOFv, EOFv}, {0, UCharRangeMax}})}));
  addToFunctionSummaryMap(
      "getchar", Signature(ArgTypes{}, RetType{IntTy}),
      Summary(NoEvalCall)
          .Case({ReturnValueCondition(WithinRange,
                                      {{EOFv, EOFv}, {0, UCharRangeMax}})}));

  // read()-like functions that never return more than buffer size.
  auto FreadSummary =
      Summary(NoEvalCall)
          .Case({ReturnValueCondition(LessThanOrEq, ArgNo(2)),
                 ReturnValueCondition(WithinRange, Range(0, SizeMax))})
          .ArgConstraint(NotNull(ArgNo(0)))
          .ArgConstraint(NotNull(ArgNo(3)))
          .ArgConstraint(BufferSize(/*Buffer=*/ArgNo(0), /*BufSize=*/ArgNo(1),
                                    /*BufSizeMultiplier=*/ArgNo(2)));

  // size_t fread(void *restrict ptr, size_t size, size_t nitems,
  //              FILE *restrict stream);
  addToFunctionSummaryMap(
      "fread",
      Signature(ArgTypes{VoidPtrRestrictTy, SizeTy, SizeTy, FilePtrRestrictTy},
                RetType{SizeTy}),
      FreadSummary);
  // size_t fwrite(const void *restrict ptr, size_t size, size_t nitems,
  //               FILE *restrict stream);
  addToFunctionSummaryMap("fwrite",
                          Signature(ArgTypes{ConstVoidPtrRestrictTy, SizeTy,
                                             SizeTy, FilePtrRestrictTy},
                                    RetType{SizeTy}),
                          FreadSummary);

  Optional<QualType> Ssize_tTy = lookupTy("ssize_t");
  Optional<RangeInt> Ssize_tMax = getMaxValue(Ssize_tTy);

  auto ReadSummary =
      Summary(NoEvalCall)
          .Case({ReturnValueCondition(LessThanOrEq, ArgNo(2)),
                 ReturnValueCondition(WithinRange, Range(-1, Ssize_tMax))});

  // FIXME these are actually defined by POSIX and not by the C standard, we
  // should handle them together with the rest of the POSIX functions.
  // ssize_t read(int fildes, void *buf, size_t nbyte);
  addToFunctionSummaryMap(
      "read", Signature(ArgTypes{IntTy, VoidPtrTy, SizeTy}, RetType{Ssize_tTy}),
      ReadSummary);
  // ssize_t write(int fildes, const void *buf, size_t nbyte);
  addToFunctionSummaryMap(
      "write",
      Signature(ArgTypes{IntTy, ConstVoidPtrTy, SizeTy}, RetType{Ssize_tTy}),
      ReadSummary);

  auto GetLineSummary =
      Summary(NoEvalCall)
          .Case({ReturnValueCondition(WithinRange,
                                      Range({-1, -1}, {1, Ssize_tMax}))});

  QualType CharPtrPtrRestrictTy = getRestrictTy(getPointerTy(CharPtrTy));

  // getline()-like functions either fail or read at least the delimiter.
  // FIXME these are actually defined by POSIX and not by the C standard, we
  // should handle them together with the rest of the POSIX functions.
  // ssize_t getline(char **restrict lineptr, size_t *restrict n,
  //                 FILE *restrict stream);
  addToFunctionSummaryMap(
      "getline",
      Signature(
          ArgTypes{CharPtrPtrRestrictTy, SizePtrRestrictTy, FilePtrRestrictTy},
          RetType{Ssize_tTy}),
      GetLineSummary);
  // ssize_t getdelim(char **restrict lineptr, size_t *restrict n,
  //                  int delimiter, FILE *restrict stream);
  addToFunctionSummaryMap(
      "getdelim",
      Signature(ArgTypes{CharPtrPtrRestrictTy, SizePtrRestrictTy, IntTy,
                         FilePtrRestrictTy},
                RetType{Ssize_tTy}),
      GetLineSummary);

  {
    Summary GetenvSummary =
        Summary(NoEvalCall)
            .ArgConstraint(NotNull(ArgNo(0)))
            .Case({NotNull(Ret)}, "Assuming the environment variable exists");
    // In untrusted environments the envvar might not exist.
    if (!ShouldAssumeControlledEnvironment)
      GetenvSummary.Case({NotNull(Ret)->negate()},
                         "Assuming the environment variable does not exist");

    // char *getenv(const char *name);
    addToFunctionSummaryMap(
        "getenv", Signature(ArgTypes{ConstCharPtrTy}, RetType{CharPtrTy}),
        std::move(GetenvSummary));
  }

  if (ModelPOSIX) {

    // long a64l(const char *str64);
    addToFunctionSummaryMap(
        "a64l", Signature(ArgTypes{ConstCharPtrTy}, RetType{LongTy}),
        Summary(NoEvalCall).ArgConstraint(NotNull(ArgNo(0))));

    // char *l64a(long value);
    addToFunctionSummaryMap("l64a",
                            Signature(ArgTypes{LongTy}, RetType{CharPtrTy}),
                            Summary(NoEvalCall)
                                .ArgConstraint(ArgumentCondition(
                                    0, WithinRange, Range(0, LongMax))));

    const auto ReturnsZeroOrMinusOne =
        ConstraintSet{ReturnValueCondition(WithinRange, Range(-1, 0))};
    const auto ReturnsFileDescriptor =
        ConstraintSet{ReturnValueCondition(WithinRange, Range(-1, IntMax))};

    // int access(const char *pathname, int amode);
    addToFunctionSummaryMap(
        "access", Signature(ArgTypes{ConstCharPtrTy, IntTy}, RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(NotNull(ArgNo(0))));

    // int faccessat(int dirfd, const char *pathname, int mode, int flags);
    addToFunctionSummaryMap(
        "faccessat",
        Signature(ArgTypes{IntTy, ConstCharPtrTy, IntTy, IntTy},
                  RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(NotNull(ArgNo(1))));

    // int dup(int fildes);
    addToFunctionSummaryMap("dup", Signature(ArgTypes{IntTy}, RetType{IntTy}),
                            Summary(NoEvalCall)
                                .Case(ReturnsFileDescriptor)
                                .ArgConstraint(ArgumentCondition(
                                    0, WithinRange, Range(0, IntMax))));

    // int dup2(int fildes1, int filedes2);
    addToFunctionSummaryMap(
        "dup2", Signature(ArgTypes{IntTy, IntTy}, RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsFileDescriptor)
            .ArgConstraint(ArgumentCondition(0, WithinRange, Range(0, IntMax)))
            .ArgConstraint(
                ArgumentCondition(1, WithinRange, Range(0, IntMax))));

    // int fdatasync(int fildes);
    addToFunctionSummaryMap("fdatasync",
                            Signature(ArgTypes{IntTy}, RetType{IntTy}),
                            Summary(NoEvalCall)
                                .Case(ReturnsZeroOrMinusOne)
                                .ArgConstraint(ArgumentCondition(
                                    0, WithinRange, Range(0, IntMax))));

    // int fnmatch(const char *pattern, const char *string, int flags);
    addToFunctionSummaryMap(
        "fnmatch",
        Signature(ArgTypes{ConstCharPtrTy, ConstCharPtrTy, IntTy},
                  RetType{IntTy}),
        Summary(EvalCallAsPure)
            .ArgConstraint(NotNull(ArgNo(0)))
            .ArgConstraint(NotNull(ArgNo(1))));

    // int fsync(int fildes);
    addToFunctionSummaryMap("fsync", Signature(ArgTypes{IntTy}, RetType{IntTy}),
                            Summary(NoEvalCall)
                                .Case(ReturnsZeroOrMinusOne)
                                .ArgConstraint(ArgumentCondition(
                                    0, WithinRange, Range(0, IntMax))));

    Optional<QualType> Off_tTy = lookupTy("off_t");

    // int truncate(const char *path, off_t length);
    addToFunctionSummaryMap(
        "truncate",
        Signature(ArgTypes{ConstCharPtrTy, Off_tTy}, RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(NotNull(ArgNo(0))));

    // int symlink(const char *oldpath, const char *newpath);
    addToFunctionSummaryMap(
        "symlink",
        Signature(ArgTypes{ConstCharPtrTy, ConstCharPtrTy}, RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(NotNull(ArgNo(0)))
            .ArgConstraint(NotNull(ArgNo(1))));

    // int symlinkat(const char *oldpath, int newdirfd, const char *newpath);
    addToFunctionSummaryMap(
        "symlinkat",
        Signature(ArgTypes{ConstCharPtrTy, IntTy, ConstCharPtrTy},
                  RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(NotNull(ArgNo(0)))
            .ArgConstraint(ArgumentCondition(1, WithinRange, Range(0, IntMax)))
            .ArgConstraint(NotNull(ArgNo(2))));

    // int lockf(int fd, int cmd, off_t len);
    addToFunctionSummaryMap(
        "lockf", Signature(ArgTypes{IntTy, IntTy, Off_tTy}, RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(
                ArgumentCondition(0, WithinRange, Range(0, IntMax))));

    Optional<QualType> Mode_tTy = lookupTy("mode_t");

    // int creat(const char *pathname, mode_t mode);
    addToFunctionSummaryMap(
        "creat", Signature(ArgTypes{ConstCharPtrTy, Mode_tTy}, RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsFileDescriptor)
            .ArgConstraint(NotNull(ArgNo(0))));

    // unsigned int sleep(unsigned int seconds);
    addToFunctionSummaryMap(
        "sleep", Signature(ArgTypes{UnsignedIntTy}, RetType{UnsignedIntTy}),
        Summary(NoEvalCall)
            .ArgConstraint(
                ArgumentCondition(0, WithinRange, Range(0, UnsignedIntMax))));

    Optional<QualType> DirTy = lookupTy("DIR");
    Optional<QualType> DirPtrTy = getPointerTy(DirTy);

    // int dirfd(DIR *dirp);
    addToFunctionSummaryMap("dirfd",
                            Signature(ArgTypes{DirPtrTy}, RetType{IntTy}),
                            Summary(NoEvalCall)
                                .Case(ReturnsFileDescriptor)
                                .ArgConstraint(NotNull(ArgNo(0))));

    // unsigned int alarm(unsigned int seconds);
    addToFunctionSummaryMap(
        "alarm", Signature(ArgTypes{UnsignedIntTy}, RetType{UnsignedIntTy}),
        Summary(NoEvalCall)
            .ArgConstraint(
                ArgumentCondition(0, WithinRange, Range(0, UnsignedIntMax))));

    // int closedir(DIR *dir);
    addToFunctionSummaryMap("closedir",
                            Signature(ArgTypes{DirPtrTy}, RetType{IntTy}),
                            Summary(NoEvalCall)
                                .Case(ReturnsZeroOrMinusOne)
                                .ArgConstraint(NotNull(ArgNo(0))));

    // char *strdup(const char *s);
    addToFunctionSummaryMap(
        "strdup", Signature(ArgTypes{ConstCharPtrTy}, RetType{CharPtrTy}),
        Summary(NoEvalCall).ArgConstraint(NotNull(ArgNo(0))));

    // char *strndup(const char *s, size_t n);
    addToFunctionSummaryMap(
        "strndup",
        Signature(ArgTypes{ConstCharPtrTy, SizeTy}, RetType{CharPtrTy}),
        Summary(NoEvalCall)
            .ArgConstraint(NotNull(ArgNo(0)))
            .ArgConstraint(
                ArgumentCondition(1, WithinRange, Range(0, SizeMax))));

    // wchar_t *wcsdup(const wchar_t *s);
    addToFunctionSummaryMap(
        "wcsdup", Signature(ArgTypes{ConstWchar_tPtrTy}, RetType{Wchar_tPtrTy}),
        Summary(NoEvalCall).ArgConstraint(NotNull(ArgNo(0))));

    // int mkstemp(char *template);
    addToFunctionSummaryMap("mkstemp",
                            Signature(ArgTypes{CharPtrTy}, RetType{IntTy}),
                            Summary(NoEvalCall)
                                .Case(ReturnsFileDescriptor)
                                .ArgConstraint(NotNull(ArgNo(0))));

    // char *mkdtemp(char *template);
    addToFunctionSummaryMap(
        "mkdtemp", Signature(ArgTypes{CharPtrTy}, RetType{CharPtrTy}),
        Summary(NoEvalCall).ArgConstraint(NotNull(ArgNo(0))));

    // char *getcwd(char *buf, size_t size);
    addToFunctionSummaryMap(
        "getcwd", Signature(ArgTypes{CharPtrTy, SizeTy}, RetType{CharPtrTy}),
        Summary(NoEvalCall)
            .ArgConstraint(
                ArgumentCondition(1, WithinRange, Range(0, SizeMax))));

    // int mkdir(const char *pathname, mode_t mode);
    addToFunctionSummaryMap(
        "mkdir", Signature(ArgTypes{ConstCharPtrTy, Mode_tTy}, RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(NotNull(ArgNo(0))));

    // int mkdirat(int dirfd, const char *pathname, mode_t mode);
    addToFunctionSummaryMap(
        "mkdirat",
        Signature(ArgTypes{IntTy, ConstCharPtrTy, Mode_tTy}, RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(NotNull(ArgNo(1))));

    Optional<QualType> Dev_tTy = lookupTy("dev_t");

    // int mknod(const char *pathname, mode_t mode, dev_t dev);
    addToFunctionSummaryMap(
        "mknod",
        Signature(ArgTypes{ConstCharPtrTy, Mode_tTy, Dev_tTy}, RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(NotNull(ArgNo(0))));

    // int mknodat(int dirfd, const char *pathname, mode_t mode, dev_t dev);
    addToFunctionSummaryMap(
        "mknodat",
        Signature(ArgTypes{IntTy, ConstCharPtrTy, Mode_tTy, Dev_tTy},
                  RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(NotNull(ArgNo(1))));

    // int chmod(const char *path, mode_t mode);
    addToFunctionSummaryMap(
        "chmod", Signature(ArgTypes{ConstCharPtrTy, Mode_tTy}, RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(NotNull(ArgNo(0))));

    // int fchmodat(int dirfd, const char *pathname, mode_t mode, int flags);
    addToFunctionSummaryMap(
        "fchmodat",
        Signature(ArgTypes{IntTy, ConstCharPtrTy, Mode_tTy, IntTy},
                  RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(ArgumentCondition(0, WithinRange, Range(0, IntMax)))
            .ArgConstraint(NotNull(ArgNo(1))));

    // int fchmod(int fildes, mode_t mode);
    addToFunctionSummaryMap(
        "fchmod", Signature(ArgTypes{IntTy, Mode_tTy}, RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(
                ArgumentCondition(0, WithinRange, Range(0, IntMax))));

    Optional<QualType> Uid_tTy = lookupTy("uid_t");
    Optional<QualType> Gid_tTy = lookupTy("gid_t");

    // int fchownat(int dirfd, const char *pathname, uid_t owner, gid_t group,
    //              int flags);
    addToFunctionSummaryMap(
        "fchownat",
        Signature(ArgTypes{IntTy, ConstCharPtrTy, Uid_tTy, Gid_tTy, IntTy},
                  RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(ArgumentCondition(0, WithinRange, Range(0, IntMax)))
            .ArgConstraint(NotNull(ArgNo(1))));

    // int chown(const char *path, uid_t owner, gid_t group);
    addToFunctionSummaryMap(
        "chown",
        Signature(ArgTypes{ConstCharPtrTy, Uid_tTy, Gid_tTy}, RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(NotNull(ArgNo(0))));

    // int lchown(const char *path, uid_t owner, gid_t group);
    addToFunctionSummaryMap(
        "lchown",
        Signature(ArgTypes{ConstCharPtrTy, Uid_tTy, Gid_tTy}, RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(NotNull(ArgNo(0))));

    // int fchown(int fildes, uid_t owner, gid_t group);
    addToFunctionSummaryMap(
        "fchown", Signature(ArgTypes{IntTy, Uid_tTy, Gid_tTy}, RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(
                ArgumentCondition(0, WithinRange, Range(0, IntMax))));

    // int rmdir(const char *pathname);
    addToFunctionSummaryMap("rmdir",
                            Signature(ArgTypes{ConstCharPtrTy}, RetType{IntTy}),
                            Summary(NoEvalCall)
                                .Case(ReturnsZeroOrMinusOne)
                                .ArgConstraint(NotNull(ArgNo(0))));

    // int chdir(const char *path);
    addToFunctionSummaryMap("chdir",
                            Signature(ArgTypes{ConstCharPtrTy}, RetType{IntTy}),
                            Summary(NoEvalCall)
                                .Case(ReturnsZeroOrMinusOne)
                                .ArgConstraint(NotNull(ArgNo(0))));

    // int link(const char *oldpath, const char *newpath);
    addToFunctionSummaryMap(
        "link",
        Signature(ArgTypes{ConstCharPtrTy, ConstCharPtrTy}, RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(NotNull(ArgNo(0)))
            .ArgConstraint(NotNull(ArgNo(1))));

    // int linkat(int fd1, const char *path1, int fd2, const char *path2,
    //            int flag);
    addToFunctionSummaryMap(
        "linkat",
        Signature(ArgTypes{IntTy, ConstCharPtrTy, IntTy, ConstCharPtrTy, IntTy},
                  RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(ArgumentCondition(0, WithinRange, Range(0, IntMax)))
            .ArgConstraint(NotNull(ArgNo(1)))
            .ArgConstraint(ArgumentCondition(2, WithinRange, Range(0, IntMax)))
            .ArgConstraint(NotNull(ArgNo(3))));

    // int unlink(const char *pathname);
    addToFunctionSummaryMap("unlink",
                            Signature(ArgTypes{ConstCharPtrTy}, RetType{IntTy}),
                            Summary(NoEvalCall)
                                .Case(ReturnsZeroOrMinusOne)
                                .ArgConstraint(NotNull(ArgNo(0))));

    // int unlinkat(int fd, const char *path, int flag);
    addToFunctionSummaryMap(
        "unlinkat",
        Signature(ArgTypes{IntTy, ConstCharPtrTy, IntTy}, RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(ArgumentCondition(0, WithinRange, Range(0, IntMax)))
            .ArgConstraint(NotNull(ArgNo(1))));

    Optional<QualType> StructStatTy = lookupTy("stat");
    Optional<QualType> StructStatPtrTy = getPointerTy(StructStatTy);
    Optional<QualType> StructStatPtrRestrictTy = getRestrictTy(StructStatPtrTy);

    // int fstat(int fd, struct stat *statbuf);
    addToFunctionSummaryMap(
        "fstat", Signature(ArgTypes{IntTy, StructStatPtrTy}, RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(ArgumentCondition(0, WithinRange, Range(0, IntMax)))
            .ArgConstraint(NotNull(ArgNo(1))));

    // int stat(const char *restrict path, struct stat *restrict buf);
    addToFunctionSummaryMap(
        "stat",
        Signature(ArgTypes{ConstCharPtrRestrictTy, StructStatPtrRestrictTy},
                  RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(NotNull(ArgNo(0)))
            .ArgConstraint(NotNull(ArgNo(1))));

    // int lstat(const char *restrict path, struct stat *restrict buf);
    addToFunctionSummaryMap(
        "lstat",
        Signature(ArgTypes{ConstCharPtrRestrictTy, StructStatPtrRestrictTy},
                  RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(NotNull(ArgNo(0)))
            .ArgConstraint(NotNull(ArgNo(1))));

    // int fstatat(int fd, const char *restrict path,
    //             struct stat *restrict buf, int flag);
    addToFunctionSummaryMap(
        "fstatat",
        Signature(ArgTypes{IntTy, ConstCharPtrRestrictTy,
                           StructStatPtrRestrictTy, IntTy},
                  RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(ArgumentCondition(0, WithinRange, Range(0, IntMax)))
            .ArgConstraint(NotNull(ArgNo(1)))
            .ArgConstraint(NotNull(ArgNo(2))));

    // DIR *opendir(const char *name);
    addToFunctionSummaryMap(
        "opendir", Signature(ArgTypes{ConstCharPtrTy}, RetType{DirPtrTy}),
        Summary(NoEvalCall).ArgConstraint(NotNull(ArgNo(0))));

    // DIR *fdopendir(int fd);
    addToFunctionSummaryMap("fdopendir",
                            Signature(ArgTypes{IntTy}, RetType{DirPtrTy}),
                            Summary(NoEvalCall)
                                .ArgConstraint(ArgumentCondition(
                                    0, WithinRange, Range(0, IntMax))));

    // int isatty(int fildes);
    addToFunctionSummaryMap(
        "isatty", Signature(ArgTypes{IntTy}, RetType{IntTy}),
        Summary(NoEvalCall)
            .Case({ReturnValueCondition(WithinRange, Range(0, 1))})
            .ArgConstraint(
                ArgumentCondition(0, WithinRange, Range(0, IntMax))));

    // FILE *popen(const char *command, const char *type);
    addToFunctionSummaryMap(
        "popen",
        Signature(ArgTypes{ConstCharPtrTy, ConstCharPtrTy}, RetType{FilePtrTy}),
        Summary(NoEvalCall)
            .ArgConstraint(NotNull(ArgNo(0)))
            .ArgConstraint(NotNull(ArgNo(1))));

    // int pclose(FILE *stream);
    addToFunctionSummaryMap(
        "pclose", Signature(ArgTypes{FilePtrTy}, RetType{IntTy}),
        Summary(NoEvalCall).ArgConstraint(NotNull(ArgNo(0))));

    // int close(int fildes);
    addToFunctionSummaryMap("close", Signature(ArgTypes{IntTy}, RetType{IntTy}),
                            Summary(NoEvalCall)
                                .Case(ReturnsZeroOrMinusOne)
                                .ArgConstraint(ArgumentCondition(
                                    0, WithinRange, Range(-1, IntMax))));

    // long fpathconf(int fildes, int name);
    addToFunctionSummaryMap("fpathconf",
                            Signature(ArgTypes{IntTy, IntTy}, RetType{LongTy}),
                            Summary(NoEvalCall)
                                .ArgConstraint(ArgumentCondition(
                                    0, WithinRange, Range(0, IntMax))));

    // long pathconf(const char *path, int name);
    addToFunctionSummaryMap(
        "pathconf", Signature(ArgTypes{ConstCharPtrTy, IntTy}, RetType{LongTy}),
        Summary(NoEvalCall).ArgConstraint(NotNull(ArgNo(0))));

    // FILE *fdopen(int fd, const char *mode);
    addToFunctionSummaryMap(
        "fdopen",
        Signature(ArgTypes{IntTy, ConstCharPtrTy}, RetType{FilePtrTy}),
        Summary(NoEvalCall)
            .ArgConstraint(ArgumentCondition(0, WithinRange, Range(0, IntMax)))
            .ArgConstraint(NotNull(ArgNo(1))));

    // void rewinddir(DIR *dir);
    addToFunctionSummaryMap(
        "rewinddir", Signature(ArgTypes{DirPtrTy}, RetType{VoidTy}),
        Summary(NoEvalCall).ArgConstraint(NotNull(ArgNo(0))));

    // void seekdir(DIR *dirp, long loc);
    addToFunctionSummaryMap(
        "seekdir", Signature(ArgTypes{DirPtrTy, LongTy}, RetType{VoidTy}),
        Summary(NoEvalCall).ArgConstraint(NotNull(ArgNo(0))));

    // int rand_r(unsigned int *seedp);
    addToFunctionSummaryMap(
        "rand_r", Signature(ArgTypes{UnsignedIntPtrTy}, RetType{IntTy}),
        Summary(NoEvalCall).ArgConstraint(NotNull(ArgNo(0))));

    // int fileno(FILE *stream);
    addToFunctionSummaryMap("fileno",
                            Signature(ArgTypes{FilePtrTy}, RetType{IntTy}),
                            Summary(NoEvalCall)
                                .Case(ReturnsFileDescriptor)
                                .ArgConstraint(NotNull(ArgNo(0))));

    // int fseeko(FILE *stream, off_t offset, int whence);
    addToFunctionSummaryMap(
        "fseeko",
        Signature(ArgTypes{FilePtrTy, Off_tTy, IntTy}, RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(NotNull(ArgNo(0))));

    // off_t ftello(FILE *stream);
    addToFunctionSummaryMap(
        "ftello", Signature(ArgTypes{FilePtrTy}, RetType{Off_tTy}),
        Summary(NoEvalCall).ArgConstraint(NotNull(ArgNo(0))));

    // void *mmap(void *addr, size_t length, int prot, int flags, int fd,
    // off_t offset);
    addToFunctionSummaryMap(
        "mmap",
        Signature(ArgTypes{VoidPtrTy, SizeTy, IntTy, IntTy, IntTy, Off_tTy},
                  RetType{VoidPtrTy}),
        Summary(NoEvalCall)
            .ArgConstraint(ArgumentCondition(1, WithinRange, Range(1, SizeMax)))
            .ArgConstraint(
                ArgumentCondition(4, WithinRange, Range(-1, IntMax))));

    Optional<QualType> Off64_tTy = lookupTy("off64_t");
    // void *mmap64(void *addr, size_t length, int prot, int flags, int fd,
    // off64_t offset);
    addToFunctionSummaryMap(
        "mmap64",
        Signature(ArgTypes{VoidPtrTy, SizeTy, IntTy, IntTy, IntTy, Off64_tTy},
                  RetType{VoidPtrTy}),
        Summary(NoEvalCall)
            .ArgConstraint(ArgumentCondition(1, WithinRange, Range(1, SizeMax)))
            .ArgConstraint(
                ArgumentCondition(4, WithinRange, Range(-1, IntMax))));

    // int pipe(int fildes[2]);
    addToFunctionSummaryMap("pipe",
                            Signature(ArgTypes{IntPtrTy}, RetType{IntTy}),
                            Summary(NoEvalCall)
                                .Case(ReturnsZeroOrMinusOne)
                                .ArgConstraint(NotNull(ArgNo(0))));

    // off_t lseek(int fildes, off_t offset, int whence);
    addToFunctionSummaryMap(
        "lseek", Signature(ArgTypes{IntTy, Off_tTy, IntTy}, RetType{Off_tTy}),
        Summary(NoEvalCall)
            .ArgConstraint(
                ArgumentCondition(0, WithinRange, Range(0, IntMax))));

    // ssize_t readlink(const char *restrict path, char *restrict buf,
    //                  size_t bufsize);
    addToFunctionSummaryMap(
        "readlink",
        Signature(ArgTypes{ConstCharPtrRestrictTy, CharPtrRestrictTy, SizeTy},
                  RetType{Ssize_tTy}),
        Summary(NoEvalCall)
            .Case({ReturnValueCondition(LessThanOrEq, ArgNo(2)),
                   ReturnValueCondition(WithinRange, Range(-1, Ssize_tMax))})
            .ArgConstraint(NotNull(ArgNo(0)))
            .ArgConstraint(NotNull(ArgNo(1)))
            .ArgConstraint(BufferSize(/*Buffer=*/ArgNo(1),
                                      /*BufSize=*/ArgNo(2)))
            .ArgConstraint(
                ArgumentCondition(2, WithinRange, Range(0, SizeMax))));

    // ssize_t readlinkat(int fd, const char *restrict path,
    //                    char *restrict buf, size_t bufsize);
    addToFunctionSummaryMap(
        "readlinkat",
        Signature(
            ArgTypes{IntTy, ConstCharPtrRestrictTy, CharPtrRestrictTy, SizeTy},
            RetType{Ssize_tTy}),
        Summary(NoEvalCall)
            .Case({ReturnValueCondition(LessThanOrEq, ArgNo(3)),
                   ReturnValueCondition(WithinRange, Range(-1, Ssize_tMax))})
            .ArgConstraint(ArgumentCondition(0, WithinRange, Range(0, IntMax)))
            .ArgConstraint(NotNull(ArgNo(1)))
            .ArgConstraint(NotNull(ArgNo(2)))
            .ArgConstraint(BufferSize(/*Buffer=*/ArgNo(2),
                                      /*BufSize=*/ArgNo(3)))
            .ArgConstraint(
                ArgumentCondition(3, WithinRange, Range(0, SizeMax))));

    // int renameat(int olddirfd, const char *oldpath, int newdirfd, const char
    // *newpath);
    addToFunctionSummaryMap(
        "renameat",
        Signature(ArgTypes{IntTy, ConstCharPtrTy, IntTy, ConstCharPtrTy},
                  RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(NotNull(ArgNo(1)))
            .ArgConstraint(NotNull(ArgNo(3))));

    // char *realpath(const char *restrict file_name,
    //                char *restrict resolved_name);
    addToFunctionSummaryMap(
        "realpath",
        Signature(ArgTypes{ConstCharPtrRestrictTy, CharPtrRestrictTy},
                  RetType{CharPtrTy}),
        Summary(NoEvalCall).ArgConstraint(NotNull(ArgNo(0))));

    QualType CharPtrConstPtr = getPointerTy(getConstTy(CharPtrTy));

    // int execv(const char *path, char *const argv[]);
    addToFunctionSummaryMap(
        "execv",
        Signature(ArgTypes{ConstCharPtrTy, CharPtrConstPtr}, RetType{IntTy}),
        Summary(NoEvalCall)
            .Case({ReturnValueCondition(WithinRange, SingleValue(-1))})
            .ArgConstraint(NotNull(ArgNo(0))));

    // int execvp(const char *file, char *const argv[]);
    addToFunctionSummaryMap(
        "execvp",
        Signature(ArgTypes{ConstCharPtrTy, CharPtrConstPtr}, RetType{IntTy}),
        Summary(NoEvalCall)
            .Case({ReturnValueCondition(WithinRange, SingleValue(-1))})
            .ArgConstraint(NotNull(ArgNo(0))));

    // int getopt(int argc, char * const argv[], const char *optstring);
    addToFunctionSummaryMap(
        "getopt",
        Signature(ArgTypes{IntTy, CharPtrConstPtr, ConstCharPtrTy},
                  RetType{IntTy}),
        Summary(NoEvalCall)
            .Case({ReturnValueCondition(WithinRange, Range(-1, UCharRangeMax))})
            .ArgConstraint(ArgumentCondition(0, WithinRange, Range(0, IntMax)))
            .ArgConstraint(NotNull(ArgNo(1)))
            .ArgConstraint(NotNull(ArgNo(2))));

    Optional<QualType> StructSockaddrTy = lookupTy("sockaddr");
    Optional<QualType> StructSockaddrPtrTy = getPointerTy(StructSockaddrTy);
    Optional<QualType> ConstStructSockaddrPtrTy =
        getPointerTy(getConstTy(StructSockaddrTy));
    Optional<QualType> StructSockaddrPtrRestrictTy =
        getRestrictTy(StructSockaddrPtrTy);
    Optional<QualType> ConstStructSockaddrPtrRestrictTy =
        getRestrictTy(ConstStructSockaddrPtrTy);
    Optional<QualType> Socklen_tTy = lookupTy("socklen_t");
    Optional<QualType> Socklen_tPtrTy = getPointerTy(Socklen_tTy);
    Optional<QualType> Socklen_tPtrRestrictTy = getRestrictTy(Socklen_tPtrTy);
    Optional<RangeInt> Socklen_tMax = getMaxValue(Socklen_tTy);

    // In 'socket.h' of some libc implementations with C99, sockaddr parameter
    // is a transparent union of the underlying sockaddr_ family of pointers
    // instead of being a pointer to struct sockaddr. In these cases, the
    // standardized signature will not match, thus we try to match with another
    // signature that has the joker Irrelevant type. We also remove those
    // constraints which require pointer types for the sockaddr param.
    auto Accept =
        Summary(NoEvalCall)
            .Case(ReturnsFileDescriptor)
            .ArgConstraint(ArgumentCondition(0, WithinRange, Range(0, IntMax)));
    if (!addToFunctionSummaryMap(
            "accept",
            // int accept(int socket, struct sockaddr *restrict address,
            //            socklen_t *restrict address_len);
            Signature(ArgTypes{IntTy, StructSockaddrPtrRestrictTy,
                               Socklen_tPtrRestrictTy},
                      RetType{IntTy}),
            Accept))
      addToFunctionSummaryMap(
          "accept",
          Signature(ArgTypes{IntTy, Irrelevant, Socklen_tPtrRestrictTy},
                    RetType{IntTy}),
          Accept);

    // int bind(int socket, const struct sockaddr *address, socklen_t
    //          address_len);
    if (!addToFunctionSummaryMap(
            "bind",
            Signature(ArgTypes{IntTy, ConstStructSockaddrPtrTy, Socklen_tTy},
                      RetType{IntTy}),
            Summary(NoEvalCall)
                .Case(ReturnsZeroOrMinusOne)
                .ArgConstraint(
                    ArgumentCondition(0, WithinRange, Range(0, IntMax)))
                .ArgConstraint(NotNull(ArgNo(1)))
                .ArgConstraint(
                    BufferSize(/*Buffer=*/ArgNo(1), /*BufSize=*/ArgNo(2)))
                .ArgConstraint(
                    ArgumentCondition(2, WithinRange, Range(0, Socklen_tMax)))))
      // Do not add constraints on sockaddr.
      addToFunctionSummaryMap(
          "bind",
          Signature(ArgTypes{IntTy, Irrelevant, Socklen_tTy}, RetType{IntTy}),
          Summary(NoEvalCall)
              .Case(ReturnsZeroOrMinusOne)
              .ArgConstraint(
                  ArgumentCondition(0, WithinRange, Range(0, IntMax)))
              .ArgConstraint(
                  ArgumentCondition(2, WithinRange, Range(0, Socklen_tMax))));

    // int getpeername(int socket, struct sockaddr *restrict address,
    //                 socklen_t *restrict address_len);
    if (!addToFunctionSummaryMap(
            "getpeername",
            Signature(ArgTypes{IntTy, StructSockaddrPtrRestrictTy,
                               Socklen_tPtrRestrictTy},
                      RetType{IntTy}),
            Summary(NoEvalCall)
                .Case(ReturnsZeroOrMinusOne)
                .ArgConstraint(
                    ArgumentCondition(0, WithinRange, Range(0, IntMax)))
                .ArgConstraint(NotNull(ArgNo(1)))
                .ArgConstraint(NotNull(ArgNo(2)))))
      addToFunctionSummaryMap(
          "getpeername",
          Signature(ArgTypes{IntTy, Irrelevant, Socklen_tPtrRestrictTy},
                    RetType{IntTy}),
          Summary(NoEvalCall)
              .Case(ReturnsZeroOrMinusOne)
              .ArgConstraint(
                  ArgumentCondition(0, WithinRange, Range(0, IntMax))));

    // int getsockname(int socket, struct sockaddr *restrict address,
    //                 socklen_t *restrict address_len);
    if (!addToFunctionSummaryMap(
            "getsockname",
            Signature(ArgTypes{IntTy, StructSockaddrPtrRestrictTy,
                               Socklen_tPtrRestrictTy},
                      RetType{IntTy}),
            Summary(NoEvalCall)
                .Case(ReturnsZeroOrMinusOne)
                .ArgConstraint(
                    ArgumentCondition(0, WithinRange, Range(0, IntMax)))
                .ArgConstraint(NotNull(ArgNo(1)))
                .ArgConstraint(NotNull(ArgNo(2)))))
      addToFunctionSummaryMap(
          "getsockname",
          Signature(ArgTypes{IntTy, Irrelevant, Socklen_tPtrRestrictTy},
                    RetType{IntTy}),
          Summary(NoEvalCall)
              .Case(ReturnsZeroOrMinusOne)
              .ArgConstraint(
                  ArgumentCondition(0, WithinRange, Range(0, IntMax))));

    // int connect(int socket, const struct sockaddr *address, socklen_t
    //             address_len);
    if (!addToFunctionSummaryMap(
            "connect",
            Signature(ArgTypes{IntTy, ConstStructSockaddrPtrTy, Socklen_tTy},
                      RetType{IntTy}),
            Summary(NoEvalCall)
                .Case(ReturnsZeroOrMinusOne)
                .ArgConstraint(
                    ArgumentCondition(0, WithinRange, Range(0, IntMax)))
                .ArgConstraint(NotNull(ArgNo(1)))))
      addToFunctionSummaryMap(
          "connect",
          Signature(ArgTypes{IntTy, Irrelevant, Socklen_tTy}, RetType{IntTy}),
          Summary(NoEvalCall)
              .Case(ReturnsZeroOrMinusOne)
              .ArgConstraint(
                  ArgumentCondition(0, WithinRange, Range(0, IntMax))));

    auto Recvfrom =
        Summary(NoEvalCall)
            .Case({ReturnValueCondition(LessThanOrEq, ArgNo(2)),
                   ReturnValueCondition(WithinRange, Range(-1, Ssize_tMax))})
            .ArgConstraint(ArgumentCondition(0, WithinRange, Range(0, IntMax)))
            .ArgConstraint(BufferSize(/*Buffer=*/ArgNo(1),
                                      /*BufSize=*/ArgNo(2)));
    if (!addToFunctionSummaryMap(
            "recvfrom",
            // ssize_t recvfrom(int socket, void *restrict buffer,
            //                  size_t length,
            //                  int flags, struct sockaddr *restrict address,
            //                  socklen_t *restrict address_len);
            Signature(ArgTypes{IntTy, VoidPtrRestrictTy, SizeTy, IntTy,
                               StructSockaddrPtrRestrictTy,
                               Socklen_tPtrRestrictTy},
                      RetType{Ssize_tTy}),
            Recvfrom))
      addToFunctionSummaryMap(
          "recvfrom",
          Signature(ArgTypes{IntTy, VoidPtrRestrictTy, SizeTy, IntTy,
                             Irrelevant, Socklen_tPtrRestrictTy},
                    RetType{Ssize_tTy}),
          Recvfrom);

    auto Sendto =
        Summary(NoEvalCall)
            .Case({ReturnValueCondition(LessThanOrEq, ArgNo(2)),
                   ReturnValueCondition(WithinRange, Range(-1, Ssize_tMax))})
            .ArgConstraint(ArgumentCondition(0, WithinRange, Range(0, IntMax)))
            .ArgConstraint(BufferSize(/*Buffer=*/ArgNo(1),
                                      /*BufSize=*/ArgNo(2)));
    if (!addToFunctionSummaryMap(
            "sendto",
            // ssize_t sendto(int socket, const void *message, size_t length,
            //                int flags, const struct sockaddr *dest_addr,
            //                socklen_t dest_len);
            Signature(ArgTypes{IntTy, ConstVoidPtrTy, SizeTy, IntTy,
                               ConstStructSockaddrPtrTy, Socklen_tTy},
                      RetType{Ssize_tTy}),
            Sendto))
      addToFunctionSummaryMap(
          "sendto",
          Signature(ArgTypes{IntTy, ConstVoidPtrTy, SizeTy, IntTy, Irrelevant,
                             Socklen_tTy},
                    RetType{Ssize_tTy}),
          Sendto);

    // int listen(int sockfd, int backlog);
    addToFunctionSummaryMap("listen",
                            Signature(ArgTypes{IntTy, IntTy}, RetType{IntTy}),
                            Summary(NoEvalCall)
                                .Case(ReturnsZeroOrMinusOne)
                                .ArgConstraint(ArgumentCondition(
                                    0, WithinRange, Range(0, IntMax))));

    // ssize_t recv(int sockfd, void *buf, size_t len, int flags);
    addToFunctionSummaryMap(
        "recv",
        Signature(ArgTypes{IntTy, VoidPtrTy, SizeTy, IntTy},
                  RetType{Ssize_tTy}),
        Summary(NoEvalCall)
            .Case({ReturnValueCondition(LessThanOrEq, ArgNo(2)),
                   ReturnValueCondition(WithinRange, Range(-1, Ssize_tMax))})
            .ArgConstraint(ArgumentCondition(0, WithinRange, Range(0, IntMax)))
            .ArgConstraint(BufferSize(/*Buffer=*/ArgNo(1),
                                      /*BufSize=*/ArgNo(2))));

    Optional<QualType> StructMsghdrTy = lookupTy("msghdr");
    Optional<QualType> StructMsghdrPtrTy = getPointerTy(StructMsghdrTy);
    Optional<QualType> ConstStructMsghdrPtrTy =
        getPointerTy(getConstTy(StructMsghdrTy));

    // ssize_t recvmsg(int sockfd, struct msghdr *msg, int flags);
    addToFunctionSummaryMap(
        "recvmsg",
        Signature(ArgTypes{IntTy, StructMsghdrPtrTy, IntTy},
                  RetType{Ssize_tTy}),
        Summary(NoEvalCall)
            .Case({ReturnValueCondition(WithinRange, Range(-1, Ssize_tMax))})
            .ArgConstraint(
                ArgumentCondition(0, WithinRange, Range(0, IntMax))));

    // ssize_t sendmsg(int sockfd, const struct msghdr *msg, int flags);
    addToFunctionSummaryMap(
        "sendmsg",
        Signature(ArgTypes{IntTy, ConstStructMsghdrPtrTy, IntTy},
                  RetType{Ssize_tTy}),
        Summary(NoEvalCall)
            .Case({ReturnValueCondition(WithinRange, Range(-1, Ssize_tMax))})
            .ArgConstraint(
                ArgumentCondition(0, WithinRange, Range(0, IntMax))));

    // int setsockopt(int socket, int level, int option_name,
    //                const void *option_value, socklen_t option_len);
    addToFunctionSummaryMap(
        "setsockopt",
        Signature(ArgTypes{IntTy, IntTy, IntTy, ConstVoidPtrTy, Socklen_tTy},
                  RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(NotNull(ArgNo(3)))
            .ArgConstraint(
                BufferSize(/*Buffer=*/ArgNo(3), /*BufSize=*/ArgNo(4)))
            .ArgConstraint(
                ArgumentCondition(4, WithinRange, Range(0, Socklen_tMax))));

    // int getsockopt(int socket, int level, int option_name,
    //                void *restrict option_value,
    //                socklen_t *restrict option_len);
    addToFunctionSummaryMap(
        "getsockopt",
        Signature(ArgTypes{IntTy, IntTy, IntTy, VoidPtrRestrictTy,
                           Socklen_tPtrRestrictTy},
                  RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(NotNull(ArgNo(3)))
            .ArgConstraint(NotNull(ArgNo(4))));

    // ssize_t send(int sockfd, const void *buf, size_t len, int flags);
    addToFunctionSummaryMap(
        "send",
        Signature(ArgTypes{IntTy, ConstVoidPtrTy, SizeTy, IntTy},
                  RetType{Ssize_tTy}),
        Summary(NoEvalCall)
            .Case({ReturnValueCondition(LessThanOrEq, ArgNo(2)),
                   ReturnValueCondition(WithinRange, Range(-1, Ssize_tMax))})
            .ArgConstraint(ArgumentCondition(0, WithinRange, Range(0, IntMax)))
            .ArgConstraint(BufferSize(/*Buffer=*/ArgNo(1),
                                      /*BufSize=*/ArgNo(2))));

    // int socketpair(int domain, int type, int protocol, int sv[2]);
    addToFunctionSummaryMap(
        "socketpair",
        Signature(ArgTypes{IntTy, IntTy, IntTy, IntPtrTy}, RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(NotNull(ArgNo(3))));

    // int getnameinfo(const struct sockaddr *restrict sa, socklen_t salen,
    //                 char *restrict node, socklen_t nodelen,
    //                 char *restrict service,
    //                 socklen_t servicelen, int flags);
    //
    // This is defined in netdb.h. And contrary to 'socket.h', the sockaddr
    // parameter is never handled as a transparent union in netdb.h
    addToFunctionSummaryMap(
        "getnameinfo",
        Signature(ArgTypes{ConstStructSockaddrPtrRestrictTy, Socklen_tTy,
                           CharPtrRestrictTy, Socklen_tTy, CharPtrRestrictTy,
                           Socklen_tTy, IntTy},
                  RetType{IntTy}),
        Summary(NoEvalCall)
            .ArgConstraint(
                BufferSize(/*Buffer=*/ArgNo(0), /*BufSize=*/ArgNo(1)))
            .ArgConstraint(
                ArgumentCondition(1, WithinRange, Range(0, Socklen_tMax)))
            .ArgConstraint(
                BufferSize(/*Buffer=*/ArgNo(2), /*BufSize=*/ArgNo(3)))
            .ArgConstraint(
                ArgumentCondition(3, WithinRange, Range(0, Socklen_tMax)))
            .ArgConstraint(
                BufferSize(/*Buffer=*/ArgNo(4), /*BufSize=*/ArgNo(5)))
            .ArgConstraint(
                ArgumentCondition(5, WithinRange, Range(0, Socklen_tMax))));

    Optional<QualType> StructUtimbufTy = lookupTy("utimbuf");
    Optional<QualType> StructUtimbufPtrTy = getPointerTy(StructUtimbufTy);

    // int utime(const char *filename, struct utimbuf *buf);
    addToFunctionSummaryMap(
        "utime",
        Signature(ArgTypes{ConstCharPtrTy, StructUtimbufPtrTy}, RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(NotNull(ArgNo(0))));

    Optional<QualType> StructTimespecTy = lookupTy("timespec");
    Optional<QualType> StructTimespecPtrTy = getPointerTy(StructTimespecTy);
    Optional<QualType> ConstStructTimespecPtrTy =
        getPointerTy(getConstTy(StructTimespecTy));

    // int futimens(int fd, const struct timespec times[2]);
    addToFunctionSummaryMap(
        "futimens",
        Signature(ArgTypes{IntTy, ConstStructTimespecPtrTy}, RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(
                ArgumentCondition(0, WithinRange, Range(0, IntMax))));

    // int utimensat(int dirfd, const char *pathname,
    //               const struct timespec times[2], int flags);
    addToFunctionSummaryMap("utimensat",
                            Signature(ArgTypes{IntTy, ConstCharPtrTy,
                                               ConstStructTimespecPtrTy, IntTy},
                                      RetType{IntTy}),
                            Summary(NoEvalCall)
                                .Case(ReturnsZeroOrMinusOne)
                                .ArgConstraint(NotNull(ArgNo(1))));

    Optional<QualType> StructTimevalTy = lookupTy("timeval");
    Optional<QualType> ConstStructTimevalPtrTy =
        getPointerTy(getConstTy(StructTimevalTy));

    // int utimes(const char *filename, const struct timeval times[2]);
    addToFunctionSummaryMap(
        "utimes",
        Signature(ArgTypes{ConstCharPtrTy, ConstStructTimevalPtrTy},
                  RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(NotNull(ArgNo(0))));

    // int nanosleep(const struct timespec *rqtp, struct timespec *rmtp);
    addToFunctionSummaryMap(
        "nanosleep",
        Signature(ArgTypes{ConstStructTimespecPtrTy, StructTimespecPtrTy},
                  RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(NotNull(ArgNo(0))));

    Optional<QualType> Time_tTy = lookupTy("time_t");
    Optional<QualType> ConstTime_tPtrTy = getPointerTy(getConstTy(Time_tTy));
    Optional<QualType> ConstTime_tPtrRestrictTy =
        getRestrictTy(ConstTime_tPtrTy);

    Optional<QualType> StructTmTy = lookupTy("tm");
    Optional<QualType> StructTmPtrTy = getPointerTy(StructTmTy);
    Optional<QualType> StructTmPtrRestrictTy = getRestrictTy(StructTmPtrTy);
    Optional<QualType> ConstStructTmPtrTy =
        getPointerTy(getConstTy(StructTmTy));
    Optional<QualType> ConstStructTmPtrRestrictTy =
        getRestrictTy(ConstStructTmPtrTy);

    // struct tm * localtime(const time_t *tp);
    addToFunctionSummaryMap(
        "localtime",
        Signature(ArgTypes{ConstTime_tPtrTy}, RetType{StructTmPtrTy}),
        Summary(NoEvalCall).ArgConstraint(NotNull(ArgNo(0))));

    // struct tm *localtime_r(const time_t *restrict timer,
    //                        struct tm *restrict result);
    addToFunctionSummaryMap(
        "localtime_r",
        Signature(ArgTypes{ConstTime_tPtrRestrictTy, StructTmPtrRestrictTy},
                  RetType{StructTmPtrTy}),
        Summary(NoEvalCall)
            .ArgConstraint(NotNull(ArgNo(0)))
            .ArgConstraint(NotNull(ArgNo(1))));

    // char *asctime_r(const struct tm *restrict tm, char *restrict buf);
    addToFunctionSummaryMap(
        "asctime_r",
        Signature(ArgTypes{ConstStructTmPtrRestrictTy, CharPtrRestrictTy},
                  RetType{CharPtrTy}),
        Summary(NoEvalCall)
            .ArgConstraint(NotNull(ArgNo(0)))
            .ArgConstraint(NotNull(ArgNo(1)))
            .ArgConstraint(BufferSize(/*Buffer=*/ArgNo(1),
                                      /*MinBufSize=*/BVF.getValue(26, IntTy))));

    // char *ctime_r(const time_t *timep, char *buf);
    addToFunctionSummaryMap(
        "ctime_r",
        Signature(ArgTypes{ConstTime_tPtrTy, CharPtrTy}, RetType{CharPtrTy}),
        Summary(NoEvalCall)
            .ArgConstraint(NotNull(ArgNo(0)))
            .ArgConstraint(NotNull(ArgNo(1)))
            .ArgConstraint(BufferSize(
                /*Buffer=*/ArgNo(1),
                /*MinBufSize=*/BVF.getValue(26, IntTy))));

    // struct tm *gmtime_r(const time_t *restrict timer,
    //                     struct tm *restrict result);
    addToFunctionSummaryMap(
        "gmtime_r",
        Signature(ArgTypes{ConstTime_tPtrRestrictTy, StructTmPtrRestrictTy},
                  RetType{StructTmPtrTy}),
        Summary(NoEvalCall)
            .ArgConstraint(NotNull(ArgNo(0)))
            .ArgConstraint(NotNull(ArgNo(1))));

    // struct tm * gmtime(const time_t *tp);
    addToFunctionSummaryMap(
        "gmtime", Signature(ArgTypes{ConstTime_tPtrTy}, RetType{StructTmPtrTy}),
        Summary(NoEvalCall).ArgConstraint(NotNull(ArgNo(0))));

    Optional<QualType> Clockid_tTy = lookupTy("clockid_t");

    // int clock_gettime(clockid_t clock_id, struct timespec *tp);
    addToFunctionSummaryMap(
        "clock_gettime",
        Signature(ArgTypes{Clockid_tTy, StructTimespecPtrTy}, RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(NotNull(ArgNo(1))));

    Optional<QualType> StructItimervalTy = lookupTy("itimerval");
    Optional<QualType> StructItimervalPtrTy = getPointerTy(StructItimervalTy);

    // int getitimer(int which, struct itimerval *curr_value);
    addToFunctionSummaryMap(
        "getitimer",
        Signature(ArgTypes{IntTy, StructItimervalPtrTy}, RetType{IntTy}),
        Summary(NoEvalCall)
            .Case(ReturnsZeroOrMinusOne)
            .ArgConstraint(NotNull(ArgNo(1))));

    Optional<QualType> Pthread_cond_tTy = lookupTy("pthread_cond_t");
    Optional<QualType> Pthread_cond_tPtrTy = getPointerTy(Pthread_cond_tTy);
    Optional<QualType> Pthread_tTy = lookupTy("pthread_t");
    Optional<QualType> Pthread_tPtrTy = getPointerTy(Pthread_tTy);
    Optional<QualType> Pthread_tPtrRestrictTy = getRestrictTy(Pthread_tPtrTy);
    Optional<QualType> Pthread_mutex_tTy = lookupTy("pthread_mutex_t");
    Optional<QualType> Pthread_mutex_tPtrTy = getPointerTy(Pthread_mutex_tTy);
    Optional<QualType> Pthread_mutex_tPtrRestrictTy =
        getRestrictTy(Pthread_mutex_tPtrTy);
    Optional<QualType> Pthread_attr_tTy = lookupTy("pthread_attr_t");
    Optional<QualType> Pthread_attr_tPtrTy = getPointerTy(Pthread_attr_tTy);
    Optional<QualType> ConstPthread_attr_tPtrTy =
        getPointerTy(getConstTy(Pthread_attr_tTy));
    Optional<QualType> ConstPthread_attr_tPtrRestrictTy =
        getRestrictTy(ConstPthread_attr_tPtrTy);
    Optional<QualType> Pthread_mutexattr_tTy = lookupTy("pthread_mutexattr_t");
    Optional<QualType> ConstPthread_mutexattr_tPtrTy =
        getPointerTy(getConstTy(Pthread_mutexattr_tTy));
    Optional<QualType> ConstPthread_mutexattr_tPtrRestrictTy =
        getRestrictTy(ConstPthread_mutexattr_tPtrTy);

    QualType PthreadStartRoutineTy = getPointerTy(
        ACtx.getFunctionType(/*ResultTy=*/VoidPtrTy, /*Args=*/VoidPtrTy,
                             FunctionProtoType::ExtProtoInfo()));

    // int pthread_cond_signal(pthread_cond_t *cond);
    // int pthread_cond_broadcast(pthread_cond_t *cond);
    addToFunctionSummaryMap(
        {"pthread_cond_signal", "pthread_cond_broadcast"},
        Signature(ArgTypes{Pthread_cond_tPtrTy}, RetType{IntTy}),
        Summary(NoEvalCall).ArgConstraint(NotNull(ArgNo(0))));

    // int pthread_create(pthread_t *restrict thread,
    //                    const pthread_attr_t *restrict attr,
    //                    void *(*start_routine)(void*), void *restrict arg);
    addToFunctionSummaryMap(
        "pthread_create",
        Signature(ArgTypes{Pthread_tPtrRestrictTy,
                           ConstPthread_attr_tPtrRestrictTy,
                           PthreadStartRoutineTy, VoidPtrRestrictTy},
                  RetType{IntTy}),
        Summary(NoEvalCall)
            .ArgConstraint(NotNull(ArgNo(0)))
            .ArgConstraint(NotNull(ArgNo(2))));

    // int pthread_attr_destroy(pthread_attr_t *attr);
    // int pthread_attr_init(pthread_attr_t *attr);
    addToFunctionSummaryMap(
        {"pthread_attr_destroy", "pthread_attr_init"},
        Signature(ArgTypes{Pthread_attr_tPtrTy}, RetType{IntTy}),
        Summary(NoEvalCall).ArgConstraint(NotNull(ArgNo(0))));

    // int pthread_attr_getstacksize(const pthread_attr_t *restrict attr,
    //                               size_t *restrict stacksize);
    // int pthread_attr_getguardsize(const pthread_attr_t *restrict attr,
    //                               size_t *restrict guardsize);
    addToFunctionSummaryMap(
        {"pthread_attr_getstacksize", "pthread_attr_getguardsize"},
        Signature(ArgTypes{ConstPthread_attr_tPtrRestrictTy, SizePtrRestrictTy},
                  RetType{IntTy}),
        Summary(NoEvalCall)
            .ArgConstraint(NotNull(ArgNo(0)))
            .ArgConstraint(NotNull(ArgNo(1))));

    // int pthread_attr_setstacksize(pthread_attr_t *attr, size_t stacksize);
    // int pthread_attr_setguardsize(pthread_attr_t *attr, size_t guardsize);
    addToFunctionSummaryMap(
        {"pthread_attr_setstacksize", "pthread_attr_setguardsize"},
        Signature(ArgTypes{Pthread_attr_tPtrTy, SizeTy}, RetType{IntTy}),
        Summary(NoEvalCall)
            .ArgConstraint(NotNull(ArgNo(0)))
            .ArgConstraint(
                ArgumentCondition(1, WithinRange, Range(0, SizeMax))));

    // int pthread_mutex_init(pthread_mutex_t *restrict mutex, const
    //                        pthread_mutexattr_t *restrict attr);
    addToFunctionSummaryMap(
        "pthread_mutex_init",
        Signature(ArgTypes{Pthread_mutex_tPtrRestrictTy,
                           ConstPthread_mutexattr_tPtrRestrictTy},
                  RetType{IntTy}),
        Summary(NoEvalCall).ArgConstraint(NotNull(ArgNo(0))));

    // int pthread_mutex_destroy(pthread_mutex_t *mutex);
    // int pthread_mutex_lock(pthread_mutex_t *mutex);
    // int pthread_mutex_trylock(pthread_mutex_t *mutex);
    // int pthread_mutex_unlock(pthread_mutex_t *mutex);
    addToFunctionSummaryMap(
        {"pthread_mutex_destroy", "pthread_mutex_lock", "pthread_mutex_trylock",
         "pthread_mutex_unlock"},
        Signature(ArgTypes{Pthread_mutex_tPtrTy}, RetType{IntTy}),
        Summary(NoEvalCall).ArgConstraint(NotNull(ArgNo(0))));
  }

  // Functions for testing.
  if (ChecksEnabled[CK_StdCLibraryFunctionsTesterChecker]) {
    addToFunctionSummaryMap(
        "__not_null", Signature(ArgTypes{IntPtrTy}, RetType{IntTy}),
        Summary(EvalCallAsPure).ArgConstraint(NotNull(ArgNo(0))));

    // Test range values.
    addToFunctionSummaryMap(
        "__single_val_1", Signature(ArgTypes{IntTy}, RetType{IntTy}),
        Summary(EvalCallAsPure)
            .ArgConstraint(ArgumentCondition(0U, WithinRange, SingleValue(1))));
    addToFunctionSummaryMap(
        "__range_1_2", Signature(ArgTypes{IntTy}, RetType{IntTy}),
        Summary(EvalCallAsPure)
            .ArgConstraint(ArgumentCondition(0U, WithinRange, Range(1, 2))));
    addToFunctionSummaryMap("__range_1_2__4_5",
                            Signature(ArgTypes{IntTy}, RetType{IntTy}),
                            Summary(EvalCallAsPure)
                                .ArgConstraint(ArgumentCondition(
                                    0U, WithinRange, Range({1, 2}, {4, 5}))));

    // Test range kind.
    addToFunctionSummaryMap(
        "__within", Signature(ArgTypes{IntTy}, RetType{IntTy}),
        Summary(EvalCallAsPure)
            .ArgConstraint(ArgumentCondition(0U, WithinRange, SingleValue(1))));
    addToFunctionSummaryMap(
        "__out_of", Signature(ArgTypes{IntTy}, RetType{IntTy}),
        Summary(EvalCallAsPure)
            .ArgConstraint(ArgumentCondition(0U, OutOfRange, SingleValue(1))));

    addToFunctionSummaryMap(
        "__two_constrained_args",
        Signature(ArgTypes{IntTy, IntTy}, RetType{IntTy}),
        Summary(EvalCallAsPure)
            .ArgConstraint(ArgumentCondition(0U, WithinRange, SingleValue(1)))
            .ArgConstraint(ArgumentCondition(1U, WithinRange, SingleValue(1))));
    addToFunctionSummaryMap(
        "__arg_constrained_twice", Signature(ArgTypes{IntTy}, RetType{IntTy}),
        Summary(EvalCallAsPure)
            .ArgConstraint(ArgumentCondition(0U, OutOfRange, SingleValue(1)))
            .ArgConstraint(ArgumentCondition(0U, OutOfRange, SingleValue(2))));
    addToFunctionSummaryMap(
        "__defaultparam",
        Signature(ArgTypes{Irrelevant, IntTy}, RetType{IntTy}),
        Summary(EvalCallAsPure).ArgConstraint(NotNull(ArgNo(0))));
    addToFunctionSummaryMap(
        "__variadic",
        Signature(ArgTypes{VoidPtrTy, ConstCharPtrTy}, RetType{IntTy}),
        Summary(EvalCallAsPure)
            .ArgConstraint(NotNull(ArgNo(0)))
            .ArgConstraint(NotNull(ArgNo(1))));
    addToFunctionSummaryMap(
        "__buf_size_arg_constraint",
        Signature(ArgTypes{ConstVoidPtrTy, SizeTy}, RetType{IntTy}),
        Summary(EvalCallAsPure)
            .ArgConstraint(
                BufferSize(/*Buffer=*/ArgNo(0), /*BufSize=*/ArgNo(1))));
    addToFunctionSummaryMap(
        "__buf_size_arg_constraint_mul",
        Signature(ArgTypes{ConstVoidPtrTy, SizeTy, SizeTy}, RetType{IntTy}),
        Summary(EvalCallAsPure)
            .ArgConstraint(BufferSize(/*Buffer=*/ArgNo(0), /*BufSize=*/ArgNo(1),
                                      /*BufSizeMultiplier=*/ArgNo(2))));
    addToFunctionSummaryMap(
        "__buf_size_arg_constraint_concrete",
        Signature(ArgTypes{ConstVoidPtrTy}, RetType{IntTy}),
        Summary(EvalCallAsPure)
            .ArgConstraint(BufferSize(/*Buffer=*/ArgNo(0),
                                      /*BufSize=*/BVF.getValue(10, IntTy))));
    addToFunctionSummaryMap(
        {"__test_restrict_param_0", "__test_restrict_param_1",
         "__test_restrict_param_2"},
        Signature(ArgTypes{VoidPtrRestrictTy}, RetType{VoidTy}),
        Summary(EvalCallAsPure));
  }

  SummariesInitialized = true;
}

void ento::registerStdCLibraryFunctionsChecker(CheckerManager &mgr) {
  auto *Checker = mgr.registerChecker<StdLibraryFunctionsChecker>();
  const AnalyzerOptions &Opts = mgr.getAnalyzerOptions();
  Checker->DisplayLoadedSummaries =
      Opts.getCheckerBooleanOption(Checker, "DisplayLoadedSummaries");
  Checker->ModelPOSIX = Opts.getCheckerBooleanOption(Checker, "ModelPOSIX");
  Checker->ShouldAssumeControlledEnvironment =
      Opts.ShouldAssumeControlledEnvironment;
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
