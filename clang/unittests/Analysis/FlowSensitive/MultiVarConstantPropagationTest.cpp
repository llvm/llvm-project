//===- unittests/Analysis/FlowSensitive/MultiVarConstantPropagation.cpp --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a simplistic version of Constant Propagation as an example
//  of a forward, monotonic dataflow analysis. The analysis tracks all
//  variables in the scope, but lacks escape analysis.
//
//===----------------------------------------------------------------------===//

#include "TestingSupport.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "clang/Analysis/FlowSensitive/MapLattice.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/ADT/StringMapEntry.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>

namespace clang {
namespace dataflow {
namespace {
using namespace ast_matchers;

// Models the value of an expression at a program point, for all paths through
// the program.
struct ValueLattice {
  // FIXME: change the internal representation to use a `std::variant`, once
  // clang admits C++17 constructs.
  enum class ValueState : bool {
    Undefined,
    Defined,
  };
  // `State` determines the meaning of the lattice when `Value` is `None`:
  //  * `Undefined` -> bottom,
  //  * `Defined` -> top.
  ValueState State;

  // When `None`, the lattice is either at top or bottom, based on `State`.
  std::optional<int64_t> Value;

  constexpr ValueLattice()
      : State(ValueState::Undefined), Value(std::nullopt) {}
  constexpr ValueLattice(int64_t V) : State(ValueState::Defined), Value(V) {}
  constexpr ValueLattice(ValueState S) : State(S), Value(std::nullopt) {}

  static constexpr ValueLattice bottom() {
    return ValueLattice(ValueState::Undefined);
  }
  static constexpr ValueLattice top() {
    return ValueLattice(ValueState::Defined);
  }

  friend bool operator==(const ValueLattice &Lhs, const ValueLattice &Rhs) {
    return Lhs.State == Rhs.State && Lhs.Value == Rhs.Value;
  }
  friend bool operator!=(const ValueLattice &Lhs, const ValueLattice &Rhs) {
    return !(Lhs == Rhs);
  }

  LatticeJoinEffect join(const ValueLattice &Other) {
    if (*this == Other || Other == bottom() || *this == top())
      return LatticeJoinEffect::Unchanged;

    if (*this == bottom()) {
      *this = Other;
      return LatticeJoinEffect::Changed;
    }

    *this = top();
    return LatticeJoinEffect::Changed;
  }
};

std::ostream &operator<<(std::ostream &OS, const ValueLattice &L) {
  if (L.Value)
    return OS << *L.Value;
  switch (L.State) {
  case ValueLattice::ValueState::Undefined:
    return OS << "None";
  case ValueLattice::ValueState::Defined:
    return OS << "Any";
  }
  llvm_unreachable("unknown ValueState!");
}

using ConstantPropagationLattice = VarMapLattice<ValueLattice>;

constexpr char kDecl[] = "decl";
constexpr char kVar[] = "var";
constexpr char kInit[] = "init";
constexpr char kJustAssignment[] = "just-assignment";
constexpr char kAssignment[] = "assignment";
constexpr char kRHS[] = "rhs";

auto refToVar() { return declRefExpr(to(varDecl().bind(kVar))); }

// N.B. This analysis is deliberately simplistic, leaving out many important
// details needed for a real analysis. Most notably, the transfer function does
// not account for the variable's address possibly escaping, which would
// invalidate the analysis. It also could be optimized to drop out-of-scope
// variables from the map.
class ConstantPropagationAnalysis
    : public DataflowAnalysis<ConstantPropagationAnalysis,
                              ConstantPropagationLattice> {
public:
  explicit ConstantPropagationAnalysis(ASTContext &Context)
      : DataflowAnalysis<ConstantPropagationAnalysis,
                         ConstantPropagationLattice>(Context) {}

  static ConstantPropagationLattice initialElement() {
    return ConstantPropagationLattice::bottom();
  }

  void transfer(const CFGElement *E, ConstantPropagationLattice &Vars,
                Environment &Env) {
    auto CS = E->getAs<CFGStmt>();
    if (!CS)
      return;
    auto S = CS->getStmt();
    auto matcher =
        stmt(anyOf(declStmt(hasSingleDecl(
                       varDecl(decl().bind(kVar), hasType(isInteger()),
                               optionally(hasInitializer(expr().bind(kInit))))
                           .bind(kDecl))),
                   binaryOperator(hasOperatorName("="), hasLHS(refToVar()),
                                  hasRHS(expr().bind(kRHS)))
                       .bind(kJustAssignment),
                   binaryOperator(isAssignmentOperator(), hasLHS(refToVar()))
                       .bind(kAssignment)));

    ASTContext &Context = getASTContext();
    auto Results = match(matcher, *S, Context);
    if (Results.empty())
      return;
    const BoundNodes &Nodes = Results[0];

    const auto *Var = Nodes.getNodeAs<clang::VarDecl>(kVar);
    assert(Var != nullptr);

    if (Nodes.getNodeAs<clang::VarDecl>(kDecl) != nullptr) {
      if (const auto *E = Nodes.getNodeAs<clang::Expr>(kInit)) {
        Expr::EvalResult R;
        Vars[Var] = (E->EvaluateAsInt(R, Context) && R.Val.isInt())
                        ? ValueLattice(R.Val.getInt().getExtValue())
                        : ValueLattice::top();
      } else {
        // An unitialized variable holds *some* value, but we don't know what it
        // is (it is implementation defined), so we set it to top.
        Vars[Var] = ValueLattice::top();
      }
    } else if (Nodes.getNodeAs<clang::Expr>(kJustAssignment)) {
      const auto *E = Nodes.getNodeAs<clang::Expr>(kRHS);
      assert(E != nullptr);

      Expr::EvalResult R;
      Vars[Var] = (E->EvaluateAsInt(R, Context) && R.Val.isInt())
                      ? ValueLattice(R.Val.getInt().getExtValue())
                      : ValueLattice::top();
    } else if (Nodes.getNodeAs<clang::Expr>(kAssignment)) {
      // Any assignment involving the expression itself resets the variable to
      // "unknown". A more advanced analysis could try to evaluate the compound
      // assignment. For example, `x += 0` need not invalidate `x`.
      Vars[Var] = ValueLattice::top();
    }
  }
};

using ::clang::dataflow::test::AnalysisInputs;
using ::clang::dataflow::test::AnalysisOutputs;
using ::clang::dataflow::test::checkDataflow;
using ::llvm::IsStringMapEntry;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

MATCHER_P(Var, name,
          (llvm::Twine(negation ? "isn't" : "is") + " a variable named `" +
           name + "`")
              .str()) {
  return arg->getName() == name;
}

MATCHER_P(HasConstantVal, v, "") { return arg.Value && *arg.Value == v; }

MATCHER(Varies, "") { return arg == arg.top(); }

MATCHER_P(HoldsCPLattice, m,
          ((negation ? "doesn't hold" : "holds") +
           llvm::StringRef(" a lattice element that ") +
           ::testing::DescribeMatcher<ConstantPropagationLattice>(m, negation))
              .str()) {
  return ExplainMatchResult(m, arg.Lattice, result_listener);
}

template <typename Matcher>
void RunDataflow(llvm::StringRef Code, Matcher Expectations) {
  ASSERT_THAT_ERROR(
      checkDataflow<ConstantPropagationAnalysis>(
          AnalysisInputs<ConstantPropagationAnalysis>(
              Code, hasName("fun"),
              [](ASTContext &C, Environment &) {
                return ConstantPropagationAnalysis(C);
              })
              .withASTBuildArgs({"-fsyntax-only", "-std=c++17"}),
          /*VerifyResults=*/
          [&Expectations](const llvm::StringMap<DataflowAnalysisState<
                              ConstantPropagationAnalysis::Lattice>> &Results,
                          const AnalysisOutputs &) {
            EXPECT_THAT(Results, Expectations);
          }),
      llvm::Succeeded());
}

TEST(MultiVarConstantPropagationTest, JustInit) {
  std::string Code = R"(
    void fun() {
      int target = 1;
      // [[p]]
    }
  )";
  RunDataflow(Code, UnorderedElementsAre(IsStringMapEntry(
                        "p", HoldsCPLattice(UnorderedElementsAre(
                                 Pair(Var("target"), HasConstantVal(1)))))));
}

TEST(MultiVarConstantPropagationTest, Assignment) {
  std::string Code = R"(
    void fun() {
      int target = 1;
      // [[p1]]
      target = 2;
      // [[p2]]
    }
  )";
  RunDataflow(
      Code,
      UnorderedElementsAre(
          IsStringMapEntry("p1", HoldsCPLattice(UnorderedElementsAre(
                                     Pair(Var("target"), HasConstantVal(1))))),
          IsStringMapEntry("p2", HoldsCPLattice(UnorderedElementsAre(Pair(
                                     Var("target"), HasConstantVal(2)))))));
}

TEST(MultiVarConstantPropagationTest, AssignmentCall) {
  std::string Code = R"(
    int g();
    void fun() {
      int target;
      target = g();
      // [[p]]
    }
  )";
  RunDataflow(Code, UnorderedElementsAre(IsStringMapEntry(
                        "p", HoldsCPLattice(UnorderedElementsAre(
                                 Pair(Var("target"), Varies()))))));
}

TEST(MultiVarConstantPropagationTest, AssignmentBinOp) {
  std::string Code = R"(
    void fun() {
      int target;
      target = 2 + 3;
      // [[p]]
    }
  )";
  RunDataflow(Code, UnorderedElementsAre(IsStringMapEntry(
                        "p", HoldsCPLattice(UnorderedElementsAre(
                                 Pair(Var("target"), HasConstantVal(5)))))));
}

TEST(MultiVarConstantPropagationTest, PlusAssignment) {
  std::string Code = R"(
    void fun() {
      int target = 1;
      // [[p1]]
      target += 2;
      // [[p2]]
    }
  )";
  RunDataflow(
      Code, UnorderedElementsAre(
                IsStringMapEntry("p1", HoldsCPLattice(UnorderedElementsAre(Pair(
                                           Var("target"), HasConstantVal(1))))),
                IsStringMapEntry("p2", HoldsCPLattice(UnorderedElementsAre(
                                           Pair(Var("target"), Varies()))))));
}

TEST(MultiVarConstantPropagationTest, SameAssignmentInBranches) {
  std::string Code = R"cc(
    void fun(bool b) {
      int target;
      // [[p1]]
      if (b) {
        target = 2;
        // [[pT]]
      } else {
        target = 2;
        // [[pF]]
      }
      (void)0;
      // [[p2]]
    }
  )cc";
  RunDataflow(
      Code,
      UnorderedElementsAre(
          IsStringMapEntry("p1", HoldsCPLattice(UnorderedElementsAre(
                                     Pair(Var("target"), Varies())))),
          IsStringMapEntry("pT", HoldsCPLattice(UnorderedElementsAre(
                                     Pair(Var("target"), HasConstantVal(2))))),
          IsStringMapEntry("pF", HoldsCPLattice(UnorderedElementsAre(
                                     Pair(Var("target"), HasConstantVal(2))))),
          IsStringMapEntry("p2", HoldsCPLattice(UnorderedElementsAre(Pair(
                                     Var("target"), HasConstantVal(2)))))));
}

// Verifies that the analysis tracks multiple variables simultaneously.
TEST(MultiVarConstantPropagationTest, TwoVariables) {
  std::string Code = R"(
    void fun() {
      int target = 1;
      // [[p1]]
      int other = 2;
      // [[p2]]
      target = 3;
      // [[p3]]
    }
  )";
  RunDataflow(
      Code,
      UnorderedElementsAre(
          IsStringMapEntry("p1", HoldsCPLattice(UnorderedElementsAre(
                                     Pair(Var("target"), HasConstantVal(1))))),
          IsStringMapEntry("p2", HoldsCPLattice(UnorderedElementsAre(
                                     Pair(Var("target"), HasConstantVal(1)),
                                     Pair(Var("other"), HasConstantVal(2))))),
          IsStringMapEntry("p3", HoldsCPLattice(UnorderedElementsAre(
                                     Pair(Var("target"), HasConstantVal(3)),
                                     Pair(Var("other"), HasConstantVal(2)))))));
}

TEST(MultiVarConstantPropagationTest, TwoVariablesInBranches) {
  std::string Code = R"cc(
    void fun(bool b) {
      int target;
      int other;
      // [[p1]]
      if (b) {
        target = 2;
        // [[pT]]
      } else {
        other = 3;
        // [[pF]]
      }
      (void)0;
      // [[p2]]
    }
  )cc";
  RunDataflow(
      Code,
      UnorderedElementsAre(
          IsStringMapEntry("p1", HoldsCPLattice(UnorderedElementsAre(
                                     Pair(Var("target"), Varies()),
                                     Pair(Var("other"), Varies())))),
          IsStringMapEntry("pT", HoldsCPLattice(UnorderedElementsAre(
                                     Pair(Var("target"), HasConstantVal(2)),
                                     Pair(Var("other"), Varies())))),
          IsStringMapEntry("pF", HoldsCPLattice(UnorderedElementsAre(
                                     Pair(Var("other"), HasConstantVal(3)),
                                     Pair(Var("target"), Varies())))),
          IsStringMapEntry("p2", HoldsCPLattice(UnorderedElementsAre(
                                     Pair(Var("target"), Varies()),
                                     Pair(Var("other"), Varies()))))));
}

TEST(MultiVarConstantPropagationTest, SameAssignmentInBranch) {
  std::string Code = R"cc(
    void fun(bool b) {
      int target = 1;
      // [[p1]]
      if (b) {
        target = 1;
      }
      (void)0;
      // [[p2]]
    }
  )cc";
  RunDataflow(
      Code,
      UnorderedElementsAre(
          IsStringMapEntry("p1", HoldsCPLattice(UnorderedElementsAre(
                                     Pair(Var("target"), HasConstantVal(1))))),
          IsStringMapEntry("p2", HoldsCPLattice(UnorderedElementsAre(Pair(
                                     Var("target"), HasConstantVal(1)))))));
}

TEST(MultiVarConstantPropagationTest, NewVarInBranch) {
  std::string Code = R"cc(
    void fun(bool b) {
      if (b) {
        int target;
        // [[p1]]
        target = 1;
        // [[p2]]
      } else {
        int target;
        // [[p3]]
        target = 1;
        // [[p4]]
      }
    }
  )cc";
  RunDataflow(
      Code,
      UnorderedElementsAre(
          IsStringMapEntry("p1", HoldsCPLattice(UnorderedElementsAre(
                                     Pair(Var("target"), Varies())))),
          IsStringMapEntry("p2", HoldsCPLattice(UnorderedElementsAre(
                                     Pair(Var("target"), HasConstantVal(1))))),
          IsStringMapEntry("p3", HoldsCPLattice(UnorderedElementsAre(
                                     Pair(Var("target"), Varies())))),
          IsStringMapEntry("p4", HoldsCPLattice(UnorderedElementsAre(Pair(
                                     Var("target"), HasConstantVal(1)))))));
}

TEST(MultiVarConstantPropagationTest, DifferentAssignmentInBranches) {
  std::string Code = R"cc(
    void fun(bool b) {
      int target;
      // [[p1]]
      if (b) {
        target = 1;
        // [[pT]]
      } else {
        target = 2;
        // [[pF]]
      }
      (void)0;
      // [[p2]]
    }
  )cc";
  RunDataflow(
      Code, UnorderedElementsAre(
                IsStringMapEntry("p1", HoldsCPLattice(UnorderedElementsAre(
                                           Pair(Var("target"), Varies())))),
                IsStringMapEntry("pT", HoldsCPLattice(UnorderedElementsAre(Pair(
                                           Var("target"), HasConstantVal(1))))),
                IsStringMapEntry("pF", HoldsCPLattice(UnorderedElementsAre(Pair(
                                           Var("target"), HasConstantVal(2))))),
                IsStringMapEntry("p2", HoldsCPLattice(UnorderedElementsAre(
                                           Pair(Var("target"), Varies()))))));
}

TEST(MultiVarConstantPropagationTest, DifferentAssignmentInBranch) {
  std::string Code = R"cc(
    void fun(bool b) {
      int target = 1;
      // [[p1]]
      if (b) {
        target = 3;
      }
      (void)0;
      // [[p2]]
    }
  )cc";
  RunDataflow(
      Code, UnorderedElementsAre(
                IsStringMapEntry("p1", HoldsCPLattice(UnorderedElementsAre(Pair(
                                           Var("target"), HasConstantVal(1))))),
                IsStringMapEntry("p2", HoldsCPLattice(UnorderedElementsAre(
                                           Pair(Var("target"), Varies()))))));
}

} // namespace
} // namespace dataflow
} // namespace clang
