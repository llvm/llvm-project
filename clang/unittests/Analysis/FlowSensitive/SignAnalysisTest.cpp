//===- unittests/Analysis/FlowSensitive/SignAnalysisTest.cpp --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a simplistic version of Sign Analysis as a demo of a
//  forward, monotonic dataflow analysis. The implementation uses 3 boolean
//  values to represent the sign lattice (negative, zero, positive). In
//  practice, 2 booleans would be enough, however, this approach has the
//  advantage of clarity over the optimized solution.
//
//===----------------------------------------------------------------------===//

#include "TestingSupport.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/FlowSensitive/CFGMatchSwitch.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/NoopLattice.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Testing/Annotations/Annotations.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <memory>

namespace {

using namespace clang;
using namespace dataflow;
using namespace ast_matchers;
using namespace test;
using ::testing::UnorderedElementsAre;

enum class Sign : int { Negative, Zero, Positive };

Sign getSign(int64_t V) {
  return V == 0 ? Sign::Zero : (V < 0 ? Sign::Negative : Sign::Positive);
}

using LatticeTransferState = TransferState<NoopLattice>;

constexpr char kVar[] = "var";

void initNegative(Value &Val, Environment &Env) {
  Val.setProperty("neg", Env.getBoolLiteralValue(true));
  Val.setProperty("zero", Env.getBoolLiteralValue(false));
  Val.setProperty("pos", Env.getBoolLiteralValue(false));
}
void initPositive(Value &Val, Environment &Env) {
  Val.setProperty("neg", Env.getBoolLiteralValue(false));
  Val.setProperty("zero", Env.getBoolLiteralValue(false));
  Val.setProperty("pos", Env.getBoolLiteralValue(true));
}
void initZero(Value &Val, Environment &Env) {
  Val.setProperty("neg", Env.getBoolLiteralValue(false));
  Val.setProperty("zero", Env.getBoolLiteralValue(true));
  Val.setProperty("pos", Env.getBoolLiteralValue(false));
}

// The boolean properties that are associated to a Value. If a property is not
// set then these are null pointers, otherwise, the pointed BoolValues are
// owned by the Environment.
struct SignProperties {
  BoolValue *Neg, *Zero, *Pos;
};
void setSignProperties(Value &Val, const SignProperties &Ps) {
  Val.setProperty("neg", *Ps.Neg);
  Val.setProperty("zero", *Ps.Zero);
  Val.setProperty("pos", *Ps.Pos);
}
SignProperties initUnknown(Value &Val, Environment &Env) {
  SignProperties Ps{&Env.makeAtomicBoolValue(), &Env.makeAtomicBoolValue(),
                    &Env.makeAtomicBoolValue()};
  setSignProperties(Val, Ps);
  return Ps;
}
SignProperties getSignProperties(const Value &Val, const Environment &Env) {
  return {dyn_cast_or_null<BoolValue>(Val.getProperty("neg")),
          dyn_cast_or_null<BoolValue>(Val.getProperty("zero")),
          dyn_cast_or_null<BoolValue>(Val.getProperty("pos"))};
}

void transferUninitializedInt(const DeclStmt *D,
                              const MatchFinder::MatchResult &M,
                              LatticeTransferState &State) {
  const auto *Var = M.Nodes.getNodeAs<clang::VarDecl>(kVar);
  assert(Var != nullptr);
  const StorageLocation *Loc = State.Env.getStorageLocation(*Var);
  Value *Val = State.Env.getValue(*Loc);
  initUnknown(*Val, State.Env);
}

// Get the Value (1), the properties for the operand (2), and the properties
// for the unary operator (3). The return value is a tuple of (1,2,3).
//
// The returned Value (1) is a nullptr, if there is no Value associated to the
// operand of the unary operator, or if the properties are not set for that
// operand.
// Other than that, new sign properties are created for the Value of the
// unary operator and a new Value is created for the unary operator itself if
// it hadn't have any previously.
std::tuple<Value *, SignProperties, SignProperties>
getValueAndSignProperties(const UnaryOperator *UO,
                          const MatchFinder::MatchResult &M,
                          LatticeTransferState &State) {
  // The DeclRefExpr refers to this variable in the operand.
  const auto *OperandVar = M.Nodes.getNodeAs<clang::VarDecl>(kVar);
  assert(OperandVar != nullptr);
  const auto *OperandValue = State.Env.getValue(*OperandVar);
  if (!OperandValue)
    return {nullptr, {}, {}};

  // Value of the unary op.
  auto *UnaryOpValue = State.Env.getValue(*UO);
  if (!UnaryOpValue) {
    UnaryOpValue = &State.Env.makeAtomicBoolValue();
    State.Env.setValue(*UO, *UnaryOpValue);
  }

  // Properties for the operand (sub expression).
  SignProperties OperandProps = getSignProperties(*OperandValue, State.Env);
  if (OperandProps.Neg == nullptr)
    return {nullptr, {}, {}};
  // Properties for the operator expr itself.
  SignProperties UnaryOpProps = initUnknown(*UnaryOpValue, State.Env);
  return {UnaryOpValue, UnaryOpProps, OperandProps};
}

void transferBinary(const BinaryOperator *BO, const MatchFinder::MatchResult &M,
                    LatticeTransferState &State) {
  auto &A = State.Env.arena();
  const Formula *Comp;
  if (BoolValue *V = State.Env.get<BoolValue>(*BO)) {
    Comp = &V->formula();
  } else {
    Comp = &A.makeAtomRef(A.makeAtom());
    State.Env.setValue(*BO, A.makeBoolValue(*Comp));
  }

  // FIXME Use this as well:
  // auto *NegatedComp = &State.Env.makeNot(*Comp);

  auto *LHS = State.Env.getValue(*BO->getLHS());
  auto *RHS = State.Env.getValue(*BO->getRHS());

  if (!LHS || !RHS)
    return;

  SignProperties LHSProps = getSignProperties(*LHS, State.Env);
  SignProperties RHSProps = getSignProperties(*RHS, State.Env);
  if (LHSProps.Neg == nullptr || RHSProps.Neg == nullptr)
    return;

  switch (BO->getOpcode()) {
  case BO_GT:
    // pos > pos
    State.Env.assume(
        A.makeImplies(*Comp, A.makeImplies(RHSProps.Pos->formula(),
                                           LHSProps.Pos->formula())));
    // pos > zero
    State.Env.assume(
        A.makeImplies(*Comp, A.makeImplies(RHSProps.Zero->formula(),
                                           LHSProps.Pos->formula())));
    break;
  case BO_LT:
    // neg < neg
    State.Env.assume(
        A.makeImplies(*Comp, A.makeImplies(RHSProps.Neg->formula(),
                                           LHSProps.Neg->formula())));
    // neg < zero
    State.Env.assume(
        A.makeImplies(*Comp, A.makeImplies(RHSProps.Zero->formula(),
                                           LHSProps.Neg->formula())));
    break;
  case BO_GE:
    // pos >= pos
    State.Env.assume(
        A.makeImplies(*Comp, A.makeImplies(RHSProps.Pos->formula(),
                                           LHSProps.Pos->formula())));
    break;
  case BO_LE:
    // neg <= neg
    State.Env.assume(
        A.makeImplies(*Comp, A.makeImplies(RHSProps.Neg->formula(),
                                           LHSProps.Neg->formula())));
    break;
  case BO_EQ:
    State.Env.assume(
        A.makeImplies(*Comp, A.makeImplies(RHSProps.Neg->formula(),
                                           LHSProps.Neg->formula())));
    State.Env.assume(
        A.makeImplies(*Comp, A.makeImplies(RHSProps.Zero->formula(),
                                           LHSProps.Zero->formula())));
    State.Env.assume(
        A.makeImplies(*Comp, A.makeImplies(RHSProps.Pos->formula(),
                                           LHSProps.Pos->formula())));
    break;
  case BO_NE: // Noop.
    break;
  default:
    llvm_unreachable("not implemented");
  }
}

void transferUnaryMinus(const UnaryOperator *UO,
                        const MatchFinder::MatchResult &M,
                        LatticeTransferState &State) {
  auto &A = State.Env.arena();
  auto [UnaryOpValue, UnaryOpProps, OperandProps] =
      getValueAndSignProperties(UO, M, State);
  if (!UnaryOpValue)
    return;

  // a is pos ==> -a is neg
  State.Env.assume(
      A.makeImplies(OperandProps.Pos->formula(), UnaryOpProps.Neg->formula()));
  // a is neg ==> -a is pos
  State.Env.assume(
      A.makeImplies(OperandProps.Neg->formula(), UnaryOpProps.Pos->formula()));
  // a is zero ==> -a is zero
  State.Env.assume(A.makeImplies(OperandProps.Zero->formula(),
                                 UnaryOpProps.Zero->formula()));
}

void transferUnaryNot(const UnaryOperator *UO,
                      const MatchFinder::MatchResult &M,
                      LatticeTransferState &State) {
  auto &A = State.Env.arena();
  auto [UnaryOpValue, UnaryOpProps, OperandProps] =
      getValueAndSignProperties(UO, M, State);
  if (!UnaryOpValue)
    return;

  // a is neg or pos ==> !a is zero
  State.Env.assume(A.makeImplies(
      A.makeOr(OperandProps.Pos->formula(), OperandProps.Neg->formula()),
      UnaryOpProps.Zero->formula()));

  // FIXME Handle this logic universally, not just for unary not. But Where to
  // put the generic handler, transferExpr maybe?
  if (auto *UOBoolVal = dyn_cast<BoolValue>(UnaryOpValue)) {
    // !a <==> a is zero
    State.Env.assume(
        A.makeEquals(UOBoolVal->formula(), OperandProps.Zero->formula()));
    // !a <==> !a is not zero
    State.Env.assume(A.makeEquals(UOBoolVal->formula(),
                                  A.makeNot(UnaryOpProps.Zero->formula())));
  }
}

// Returns the `Value` associated with `E` (which may be either a prvalue or
// glvalue). Creates a `Value` or `StorageLocation` as needed if `E` does not
// have either of these associated with it yet.
//
// If this functionality turns out to be needed in more cases, this function
// should be moved to a more central location.
Value *getOrCreateValue(const Expr *E, Environment &Env) {
  Value *Val = nullptr;
  if (E->isGLValue()) {
    StorageLocation *Loc = Env.getStorageLocation(*E);
    if (!Loc) {
      Loc = &Env.createStorageLocation(*E);
      Env.setStorageLocation(*E, *Loc);
    }
    Val = Env.getValue(*Loc);
    if (!Val) {
      Val = Env.createValue(E->getType());
      Env.setValue(*Loc, *Val);
    }
  } else {
    Val = Env.getValue(*E);
    if (!Val) {
      Val = Env.createValue(E->getType());
      Env.setValue(*E, *Val);
    }
  }
  assert(Val != nullptr);

  return Val;
}

void transferExpr(const Expr *E, const MatchFinder::MatchResult &M,
                  LatticeTransferState &State) {
  const ASTContext &Context = *M.Context;

  Value *Val = getOrCreateValue(E, State.Env);

  // The sign symbolic values have been initialized already.
  if (Val->getProperty("neg"))
    return;

  Expr::EvalResult R;
  // An integer expression which we cannot evaluate.
  if (!(E->EvaluateAsInt(R, Context) && R.Val.isInt())) {
    initUnknown(*Val, State.Env);
    return;
  }

  const Sign S = getSign(R.Val.getInt().getExtValue());
  switch (S) {
  case Sign::Negative:
    initNegative(*Val, State.Env);
    break;
  case Sign::Zero:
    initZero(*Val, State.Env);
    break;
  case Sign::Positive:
    initPositive(*Val, State.Env);
    break;
  }
}

auto refToVar() { return declRefExpr(to(varDecl().bind(kVar))); }

auto buildTransferMatchSwitch() {
  // Note, the order of the cases is important, the most generic should be
  // added last.
  // FIXME Discover what happens if there are multiple matching ASTMatchers for
  // one Stmt? All matching case's handler should be called and in what order?
  return CFGMatchSwitchBuilder<LatticeTransferState>()
      // a op b (comparison)
      .CaseOfCFGStmt<BinaryOperator>(binaryOperator(isComparisonOperator()),
                                     transferBinary)

      // FIXME handle binop +,-,*,/

      // -a
      .CaseOfCFGStmt<UnaryOperator>(
          unaryOperator(hasOperatorName("-"),
                        hasUnaryOperand(hasDescendant(refToVar()))),
          transferUnaryMinus)

      // !a
      .CaseOfCFGStmt<UnaryOperator>(
          unaryOperator(hasOperatorName("!"),
                        hasUnaryOperand(hasDescendant(refToVar()))),
          transferUnaryNot)

      // int a;
      .CaseOfCFGStmt<DeclStmt>(declStmt(hasSingleDecl(varDecl(
                                   decl().bind(kVar), hasType(isInteger()),
                                   unless(hasInitializer(expr()))))),
                               transferUninitializedInt)

      // constexpr int
      .CaseOfCFGStmt<Expr>(expr(hasType(isInteger())), transferExpr)

      .Build();
}

class SignPropagationAnalysis
    : public DataflowAnalysis<SignPropagationAnalysis, NoopLattice> {
public:
  SignPropagationAnalysis(ASTContext &Context)
      : DataflowAnalysis<SignPropagationAnalysis, NoopLattice>(Context),
        TransferMatchSwitch(buildTransferMatchSwitch()) {}

  static NoopLattice initialElement() { return {}; }

  void transfer(const CFGElement &Elt, NoopLattice &L, Environment &Env) {
    LatticeTransferState State(L, Env);
    TransferMatchSwitch(Elt, getASTContext(), State);
  }
  bool merge(QualType Type, const Value &Val1, const Environment &Env1,
             const Value &Val2, const Environment &Env2, Value &MergedVal,
             Environment &MergedEnv) override;

private:
  CFGMatchSwitch<TransferState<NoopLattice>> TransferMatchSwitch;
};

// Copied from crubit.
BoolValue &mergeBoolValues(BoolValue &Bool1, const Environment &Env1,
                           BoolValue &Bool2, const Environment &Env2,
                           Environment &MergedEnv) {
  if (&Bool1 == &Bool2) {
    return Bool1;
  }

  auto &B1 = Bool1.formula();
  auto &B2 = Bool2.formula();

  auto &A = MergedEnv.arena();
  auto &MergedBool = MergedEnv.makeAtomicBoolValue();

  // If `Bool1` and `Bool2` is constrained to the same true / false value,
  // `MergedBool` can be constrained similarly without needing to consider the
  // path taken - this simplifies the flow condition tracked in `MergedEnv`.
  // Otherwise, information about which path was taken is used to associate
  // `MergedBool` with `Bool1` and `Bool2`.
  if (Env1.proves(B1) && Env2.proves(B2)) {
    MergedEnv.assume(MergedBool.formula());
  } else if (Env1.proves(A.makeNot(B1)) && Env2.proves(A.makeNot(B2))) {
    MergedEnv.assume(A.makeNot(MergedBool.formula()));
  }
  return MergedBool;
}

bool SignPropagationAnalysis::merge(QualType Type, const Value &Val1,
                                    const Environment &Env1, const Value &Val2,
                                    const Environment &Env2, Value &MergedVal,
                                    Environment &MergedEnv) {
  if (!Type->isIntegerType())
    return false;
  SignProperties Ps1 = getSignProperties(Val1, Env1);
  SignProperties Ps2 = getSignProperties(Val2, Env2);
  if (!Ps1.Neg || !Ps2.Neg)
    return false;
  BoolValue &MergedNeg =
      mergeBoolValues(*Ps1.Neg, Env1, *Ps2.Neg, Env2, MergedEnv);
  BoolValue &MergedZero =
      mergeBoolValues(*Ps1.Zero, Env1, *Ps2.Zero, Env2, MergedEnv);
  BoolValue &MergedPos =
      mergeBoolValues(*Ps1.Pos, Env1, *Ps2.Pos, Env2, MergedEnv);
  setSignProperties(MergedVal,
                    SignProperties{&MergedNeg, &MergedZero, &MergedPos});
  return true;
}

template <typename Matcher>
void runDataflow(llvm::StringRef Code, Matcher Match,
                 LangStandard::Kind Std = LangStandard::lang_cxx17,
                 llvm::StringRef TargetFun = "fun") {
  using ast_matchers::hasName;
  ASSERT_THAT_ERROR(
      checkDataflow<SignPropagationAnalysis>(
          AnalysisInputs<SignPropagationAnalysis>(
              Code, hasName(TargetFun),
              [](ASTContext &C, Environment &) {
                return SignPropagationAnalysis(C);
              })
              .withASTBuildArgs(
                  {"-fsyntax-only", "-fno-delayed-template-parsing",
                   "-std=" +
                       std::string(LangStandard::getLangStandardForKind(Std)
                                       .getName())}),
          /*VerifyResults=*/
          [&Match](const llvm::StringMap<DataflowAnalysisState<NoopLattice>>
                       &Results,
                   const AnalysisOutputs &AO) { Match(Results, AO.ASTCtx); }),
      llvm::Succeeded());
}

// FIXME add this to testing support.
template <typename NodeType, typename MatcherType>
const NodeType *findFirst(ASTContext &ASTCtx, const MatcherType &M) {
  auto TargetNodes = match(M.bind("v"), ASTCtx);
  assert(TargetNodes.size() == 1 && "Match must be unique");
  auto *const Result = selectFirst<NodeType>("v", TargetNodes);
  assert(Result != nullptr);
  return Result;
}

template <typename Node>
std::pair<testing::AssertionResult, Value *>
getProperty(const Environment &Env, ASTContext &ASTCtx, const Node *N,
            StringRef Property) {
  if (!N)
    return {testing::AssertionFailure() << "No node", nullptr};
  const StorageLocation *Loc = Env.getStorageLocation(*N);
  if (!isa_and_nonnull<ScalarStorageLocation>(Loc))
    return {testing::AssertionFailure() << "No location", nullptr};
  const Value *Val = Env.getValue(*Loc);
  if (!Val)
    return {testing::AssertionFailure() << "No value", nullptr};
  auto *Prop = Val->getProperty(Property);
  if (!isa_and_nonnull<BoolValue>(Prop))
    return {testing::AssertionFailure() << "No property for " << Property,
            nullptr};
  return {testing::AssertionSuccess(), Prop};
}

// Test if the given property of the given node is implied by the flow
// condition. If 'Implies' is false then check if it is not implied.
template <typename Node>
testing::AssertionResult isPropertyImplied(const Environment &Env,
                                           ASTContext &ASTCtx, const Node *N,
                                           StringRef Property, bool Implies) {
  auto [Result, Prop] = getProperty(Env, ASTCtx, N, Property);
  if (!Prop)
    return Result;
  auto *BVProp = cast<BoolValue>(Prop);
  if (Env.proves(BVProp->formula()) != Implies)
    return testing::AssertionFailure()
           << Property << " is " << (Implies ? "not" : "") << " implied"
           << ", but should " << (Implies ? "" : "not ") << "be";
  return testing::AssertionSuccess();
}

template <typename Node>
testing::AssertionResult isNegative(const Node *N, ASTContext &ASTCtx,
                                    const Environment &Env) {
  testing::AssertionResult R = isPropertyImplied(Env, ASTCtx, N, "neg", true);
  if (!R)
    return R;
  R = isPropertyImplied(Env, ASTCtx, N, "zero", false);
  if (!R)
    return R;
  return isPropertyImplied(Env, ASTCtx, N, "pos", false);
}
template <typename Node>
testing::AssertionResult isPositive(const Node *N, ASTContext &ASTCtx,
                                    const Environment &Env) {
  testing::AssertionResult R = isPropertyImplied(Env, ASTCtx, N, "pos", true);
  if (!R)
    return R;
  R = isPropertyImplied(Env, ASTCtx, N, "zero", false);
  if (!R)
    return R;
  return isPropertyImplied(Env, ASTCtx, N, "neg", false);
}
template <typename Node>
testing::AssertionResult isZero(const Node *N, ASTContext &ASTCtx,
                                const Environment &Env) {
  testing::AssertionResult R = isPropertyImplied(Env, ASTCtx, N, "zero", true);
  if (!R)
    return R;
  R = isPropertyImplied(Env, ASTCtx, N, "pos", false);
  if (!R)
    return R;
  return isPropertyImplied(Env, ASTCtx, N, "neg", false);
}
template <typename Node>
testing::AssertionResult isTop(const Node *N, ASTContext &ASTCtx,
                               const Environment &Env) {
  testing::AssertionResult R = isPropertyImplied(Env, ASTCtx, N, "zero", false);
  if (!R)
    return R;
  R = isPropertyImplied(Env, ASTCtx, N, "pos", false);
  if (!R)
    return R;
  return isPropertyImplied(Env, ASTCtx, N, "neg", false);
}

TEST(SignAnalysisTest, Init) {
  std::string Code = R"(
    int foo();
    void fun() {
      int a = -1;
      int b = 0;
      int c = 1;
      int d;
      int e = foo();
      int f = c;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        // ASTCtx.getTranslationUnitDecl()->dump();
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *A = findValueDecl(ASTCtx, "a");
        const ValueDecl *B = findValueDecl(ASTCtx, "b");
        const ValueDecl *C = findValueDecl(ASTCtx, "c");
        const ValueDecl *D = findValueDecl(ASTCtx, "d");
        const ValueDecl *E = findValueDecl(ASTCtx, "e");
        const ValueDecl *F = findValueDecl(ASTCtx, "f");

        EXPECT_TRUE(isNegative(A, ASTCtx, Env));
        EXPECT_TRUE(isZero(B, ASTCtx, Env));
        EXPECT_TRUE(isPositive(C, ASTCtx, Env));
        EXPECT_TRUE(isTop(D, ASTCtx, Env));
        EXPECT_TRUE(isTop(E, ASTCtx, Env));
        EXPECT_TRUE(isPositive(F, ASTCtx, Env));
      },
      LangStandard::lang_cxx17);
}

TEST(SignAnalysisTest, UnaryMinus) {
  std::string Code = R"(
    void fun() {
      int a = 1;
      int b = -a;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *A = findValueDecl(ASTCtx, "a");
        const ValueDecl *B = findValueDecl(ASTCtx, "b");
        EXPECT_TRUE(isPositive(A, ASTCtx, Env));
        EXPECT_TRUE(isNegative(B, ASTCtx, Env));
      },
      LangStandard::lang_cxx17);
}

TEST(SignAnalysisTest, UnaryNot) {
  std::string Code = R"(
    void fun() {
      int a = 2;
      int b = !a;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *A = findValueDecl(ASTCtx, "a");
        const ValueDecl *B = findValueDecl(ASTCtx, "b");
        EXPECT_TRUE(isPositive(A, ASTCtx, Env));
        EXPECT_TRUE(isZero(B, ASTCtx, Env));
      },
      LangStandard::lang_cxx17);
}

TEST(SignAnalysisTest, UnaryNotInIf) {
  std::string Code = R"(
    int foo();
    void fun() {
      int a = foo();
      if (!a) {
        int b1;
        int p_a = a;
        int p_not_a = !a;
        // [[p]]
      } else {
        int q_a = a;
        int q_not_a = !a;
        // [[q]]
      }
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p", "q"));
        const Environment &EnvP = getEnvironmentAtAnnotation(Results, "p");
        const Environment &EnvQ = getEnvironmentAtAnnotation(Results, "q");

        const ValueDecl *A = findValueDecl(ASTCtx, "a");
        const ValueDecl *PA = findValueDecl(ASTCtx, "p_a");
        const ValueDecl *PNA = findValueDecl(ASTCtx, "p_not_a");
        const ValueDecl *QA = findValueDecl(ASTCtx, "q_a");
        const ValueDecl *QNA = findValueDecl(ASTCtx, "q_not_a");

        // p
        EXPECT_TRUE(isZero(A, ASTCtx, EnvP));
        EXPECT_TRUE(isZero(PA, ASTCtx, EnvP));
        EXPECT_TRUE(isTop(PNA, ASTCtx, EnvP));

        // q
        EXPECT_TRUE(isTop(A, ASTCtx, EnvQ));
        EXPECT_TRUE(isTop(QA, ASTCtx, EnvQ));
        EXPECT_TRUE(isZero(QNA, ASTCtx, EnvQ));
      },
      LangStandard::lang_cxx17);
}

TEST(SignAnalysisTest, BinaryGT) {
  std::string Code = R"(
    int foo();
    void fun() {
      int a = foo();
      int b = 1;
      if (a > 1) {
        (void)0;
        // [[p]]
      }
      if (a > 0) {
        (void)0;
        // [[q]]
      }
      if (a > -1) {
        (void)0;
        // [[r]]
      }
      if (a > b) {
        (void)0;
        // [[s]]
      }
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p", "q", "r", "s"));
        const Environment &EnvP = getEnvironmentAtAnnotation(Results, "p");
        const Environment &EnvQ = getEnvironmentAtAnnotation(Results, "q");
        const Environment &EnvR = getEnvironmentAtAnnotation(Results, "r");
        const Environment &EnvS = getEnvironmentAtAnnotation(Results, "s");

        const ValueDecl *A = findValueDecl(ASTCtx, "a");

        // p
        EXPECT_TRUE(isPositive(A, ASTCtx, EnvP));
        // q
        EXPECT_TRUE(isPositive(A, ASTCtx, EnvQ));
        // r
        EXPECT_TRUE(isTop(A, ASTCtx, EnvR));
        // s
        EXPECT_TRUE(isPositive(A, ASTCtx, EnvS));
      },
      LangStandard::lang_cxx17);
}

TEST(SignAnalysisTest, BinaryLT) {
  std::string Code = R"(
    int foo();
    void fun() {
      int a = foo();
      int b = -1;
      if (a < -1) {
        (void)0;
        // [[p]]
      }
      if (a < 0) {
        (void)0;
        // [[q]]
      }
      if (a < 1) {
        (void)0;
        // [[r]]
      }
      if (a < b) {
        (void)0;
        // [[s]]
      }
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p", "q", "r", "s"));
        const Environment &EnvP = getEnvironmentAtAnnotation(Results, "p");
        const Environment &EnvQ = getEnvironmentAtAnnotation(Results, "q");
        const Environment &EnvR = getEnvironmentAtAnnotation(Results, "r");
        const Environment &EnvS = getEnvironmentAtAnnotation(Results, "s");

        const ValueDecl *A = findValueDecl(ASTCtx, "a");

        // p
        EXPECT_TRUE(isNegative(A, ASTCtx, EnvP));
        // q
        EXPECT_TRUE(isNegative(A, ASTCtx, EnvQ));
        // r
        EXPECT_TRUE(isTop(A, ASTCtx, EnvR));
        // s
        EXPECT_TRUE(isNegative(A, ASTCtx, EnvS));
      },
      LangStandard::lang_cxx17);
}

TEST(SignAnalysisTest, BinaryGE) {
  std::string Code = R"(
    int foo();
    void fun() {
      int a = foo();
      int b = 1;
      if (a >= 1) {
        (void)0;
        // [[p]]
      }
      if (a >= 0) {
        (void)0;
        // [[q]]
      }
      if (a >= -1) {
        (void)0;
        // [[r]]
      }
      if (a >= b) {
        (void)0;
        // [[s]]
      }
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p", "q", "r", "s"));
        const Environment &EnvP = getEnvironmentAtAnnotation(Results, "p");
        const Environment &EnvQ = getEnvironmentAtAnnotation(Results, "q");
        const Environment &EnvR = getEnvironmentAtAnnotation(Results, "r");
        const Environment &EnvS = getEnvironmentAtAnnotation(Results, "s");

        const ValueDecl *A = findValueDecl(ASTCtx, "a");

        // p
        EXPECT_TRUE(isPositive(A, ASTCtx, EnvP));
        // q
        EXPECT_TRUE(isTop(A, ASTCtx, EnvQ));
        // r
        EXPECT_TRUE(isTop(A, ASTCtx, EnvR));
        // s
        EXPECT_TRUE(isPositive(A, ASTCtx, EnvS));
      },
      LangStandard::lang_cxx17);
}

TEST(SignAnalysisTest, BinaryLE) {
  std::string Code = R"(
    int foo();
    void fun() {
      int a = foo();
      int b = -1;
      if (a <= -1) {
        (void)0;
        // [[p]]
      }
      if (a <= 0) {
        (void)0;
        // [[q]]
      }
      if (a <= 1) {
        (void)0;
        // [[r]]
      }
      if (a <= b) {
        (void)0;
        // [[s]]
      }
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p", "q", "r", "s"));
        const Environment &EnvP = getEnvironmentAtAnnotation(Results, "p");
        const Environment &EnvQ = getEnvironmentAtAnnotation(Results, "q");
        const Environment &EnvR = getEnvironmentAtAnnotation(Results, "r");
        const Environment &EnvS = getEnvironmentAtAnnotation(Results, "s");

        const ValueDecl *A = findValueDecl(ASTCtx, "a");

        // p
        EXPECT_TRUE(isNegative(A, ASTCtx, EnvP));
        // q
        EXPECT_TRUE(isTop(A, ASTCtx, EnvQ));
        // r
        EXPECT_TRUE(isTop(A, ASTCtx, EnvR));
        // s
        EXPECT_TRUE(isNegative(A, ASTCtx, EnvS));
      },
      LangStandard::lang_cxx17);
}

TEST(SignAnalysisTest, BinaryEQ) {
  std::string Code = R"(
    int foo();
    void fun() {
      int a = foo();
      if (a == -1) {
        (void)0;
        // [[n]]
      }
      if (a == 0) {
        (void)0;
        // [[z]]
      }
      if (a == 1) {
        (void)0;
        // [[p]]
      }
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("n", "z", "p"));
        const Environment &EnvN = getEnvironmentAtAnnotation(Results, "n");
        const Environment &EnvZ = getEnvironmentAtAnnotation(Results, "z");
        const Environment &EnvP = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *A = findValueDecl(ASTCtx, "a");

        // n
        EXPECT_TRUE(isNegative(A, ASTCtx, EnvN));
        // z
        EXPECT_TRUE(isZero(A, ASTCtx, EnvZ));
        // p
        EXPECT_TRUE(isPositive(A, ASTCtx, EnvP));
      },
      LangStandard::lang_cxx17);
}

TEST(SignAnalysisTest, ComplexLoopCondition) {
  std::string Code = R"(
    int foo();
    void fun() {
      int a, b;
      while ((a = foo()) > 0 && (b = foo()) > 0) {
        a;
        b;
        // [[p]]
      }
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *A = findValueDecl(ASTCtx, "a");
        const ValueDecl *B = findValueDecl(ASTCtx, "b");

        EXPECT_TRUE(isPositive(A, ASTCtx, Env));
        EXPECT_TRUE(isPositive(B, ASTCtx, Env));
      },
      LangStandard::lang_cxx17);
}

TEST(SignAnalysisTest, JoinToTop) {
  std::string Code = R"(
    int foo();
    void fun(bool b) {
      int a = foo();
      if (b) {
        a = -1;
        (void)0;
        // [[p]]
      } else {
        a = 1;
        (void)0;
        // [[q]]
      }
      (void)0;
      // [[r]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p", "q", "r"));
        const Environment &EnvP = getEnvironmentAtAnnotation(Results, "p");
        const Environment &EnvQ = getEnvironmentAtAnnotation(Results, "q");
        const Environment &EnvR = getEnvironmentAtAnnotation(Results, "r");

        const ValueDecl *A = findValueDecl(ASTCtx, "a");

        // p
        EXPECT_TRUE(isNegative(A, ASTCtx, EnvP));
        // q
        EXPECT_TRUE(isPositive(A, ASTCtx, EnvQ));
        // r
        EXPECT_TRUE(isTop(A, ASTCtx, EnvR));
      },
      LangStandard::lang_cxx17);
}

TEST(SignAnalysisTest, JoinToNeg) {
  std::string Code = R"(
    int foo();
    void fun() {
      int a = foo();
      if (a < 1) {
        a = -1;
        (void)0;
        // [[p]]
      } else {
        a = -1;
        (void)0;
        // [[q]]
      }
      (void)0;
      // [[r]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p", "q", "r"));
        const Environment &EnvP = getEnvironmentAtAnnotation(Results, "p");
        const Environment &EnvQ = getEnvironmentAtAnnotation(Results, "q");
        const Environment &EnvR = getEnvironmentAtAnnotation(Results, "r");

        const ValueDecl *A = findValueDecl(ASTCtx, "a");

        // p
        EXPECT_TRUE(isNegative(A, ASTCtx, EnvP));
        // q
        EXPECT_TRUE(isNegative(A, ASTCtx, EnvQ));
        // r
        EXPECT_TRUE(isNegative(A, ASTCtx, EnvR));
      },
      LangStandard::lang_cxx17);
}

TEST(SignAnalysisTest, NestedIfs) {
  std::string Code = R"(
    int foo();
    void fun() {
      int a = foo();
      if (a >= 0) {
        (void)0;
        // [[p]]
        if (a == 0) {
          (void)0;
          // [[q]]
        }
      }
      (void)0;
      // [[r]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p", "q", "r"));
        const Environment &EnvP = getEnvironmentAtAnnotation(Results, "p");
        const Environment &EnvQ = getEnvironmentAtAnnotation(Results, "q");
        const Environment &EnvR = getEnvironmentAtAnnotation(Results, "r");

        const ValueDecl *A = findValueDecl(ASTCtx, "a");

        // p
        EXPECT_TRUE(isTop(A, ASTCtx, EnvP));
        // q
        EXPECT_TRUE(isZero(A, ASTCtx, EnvQ));
        // r
        EXPECT_TRUE(isTop(A, ASTCtx, EnvR));
      },
      LangStandard::lang_cxx17);
}

} // namespace
