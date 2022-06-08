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
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "clang/Analysis/FlowSensitive/MapLattice.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Annotations.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <memory>
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
  llvm::Optional<int64_t> Value;

  constexpr ValueLattice() : State(ValueState::Undefined), Value(llvm::None) {}
  constexpr ValueLattice(int64_t V) : State(ValueState::Defined), Value(V) {}
  constexpr ValueLattice(ValueState S) : State(S), Value(llvm::None) {}

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
  if (L.Value.hasValue())
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

  void transfer(const Stmt *S, ConstantPropagationLattice &Vars,
                Environment &Env) {
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

using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

MATCHER_P(Var, name,
          (llvm::Twine(negation ? "isn't" : "is") + " a variable named `" +
           name + "`")
              .str()) {
  return arg->getName() == name;
}

MATCHER_P(HasConstantVal, v, "") {
  return arg.Value.hasValue() && *arg.Value == v;
}

MATCHER(Varies, "") { return arg == arg.top(); }

MATCHER_P(HoldsCPLattice, m,
          ((negation ? "doesn't hold" : "holds") +
           llvm::StringRef(" a lattice element that ") +
           ::testing::DescribeMatcher<ConstantPropagationLattice>(m, negation))
              .str()) {
  return ExplainMatchResult(m, arg.Lattice, result_listener);
}

class MultiVarConstantPropagationTest : public ::testing::Test {
protected:
  template <typename Matcher>
  void RunDataflow(llvm::StringRef Code, Matcher Expectations) {
    ASSERT_THAT_ERROR(
        test::checkDataflow<ConstantPropagationAnalysis>(
            Code, "fun",
            [](ASTContext &C, Environment &) {
              return ConstantPropagationAnalysis(C);
            },
            [&Expectations](
                llvm::ArrayRef<std::pair<
                    std::string, DataflowAnalysisState<
                                     ConstantPropagationAnalysis::Lattice>>>
                    Results,
                ASTContext &) { EXPECT_THAT(Results, Expectations); },
            {"-fsyntax-only", "-std=c++17"}),
        llvm::Succeeded());
  }
};

TEST_F(MultiVarConstantPropagationTest, JustInit) {
  std::string Code = R"(
    void fun() {
      int target = 1;
      // [[p]]
    }
  )";
  RunDataflow(Code, UnorderedElementsAre(
                        Pair("p", HoldsCPLattice(UnorderedElementsAre(Pair(
                                      Var("target"), HasConstantVal(1)))))));
}

TEST_F(MultiVarConstantPropagationTest, Assignment) {
  std::string Code = R"(
    void fun() {
      int target = 1;
      // [[p1]]
      target = 2;
      // [[p2]]
    }
  )";
  RunDataflow(Code, UnorderedElementsAre(
                        Pair("p1", HoldsCPLattice(UnorderedElementsAre(Pair(
                                       Var("target"), HasConstantVal(1))))),
                        Pair("p2", HoldsCPLattice(UnorderedElementsAre(Pair(
                                       Var("target"), HasConstantVal(2)))))));
}

TEST_F(MultiVarConstantPropagationTest, AssignmentCall) {
  std::string Code = R"(
    int g();
    void fun() {
      int target;
      target = g();
      // [[p]]
    }
  )";
  RunDataflow(Code, UnorderedElementsAre(
                        Pair("p", HoldsCPLattice(UnorderedElementsAre(
                                      Pair(Var("target"), Varies()))))));
}

TEST_F(MultiVarConstantPropagationTest, AssignmentBinOp) {
  std::string Code = R"(
    void fun() {
      int target;
      target = 2 + 3;
      // [[p]]
    }
  )";
  RunDataflow(Code, UnorderedElementsAre(
                        Pair("p", HoldsCPLattice(UnorderedElementsAre(Pair(
                                      Var("target"), HasConstantVal(5)))))));
}

TEST_F(MultiVarConstantPropagationTest, PlusAssignment) {
  std::string Code = R"(
    void fun() {
      int target = 1;
      // [[p1]]
      target += 2;
      // [[p2]]
    }
  )";
  RunDataflow(Code, UnorderedElementsAre(
                        Pair("p1", HoldsCPLattice(UnorderedElementsAre(Pair(
                                       Var("target"), HasConstantVal(1))))),
                        Pair("p2", HoldsCPLattice(UnorderedElementsAre(
                                       Pair(Var("target"), Varies()))))));
}

TEST_F(MultiVarConstantPropagationTest, SameAssignmentInBranches) {
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
  RunDataflow(Code,
              UnorderedElementsAre(
                  Pair("p1", HoldsCPLattice(UnorderedElementsAre(
                                 Pair(Var("target"), Varies())))),
                  Pair("pT", HoldsCPLattice(UnorderedElementsAre(
                                 Pair(Var("target"), HasConstantVal(2))))),
                  Pair("pF", HoldsCPLattice(UnorderedElementsAre(
                                 Pair(Var("target"), HasConstantVal(2))))),
                  Pair("p2", HoldsCPLattice(UnorderedElementsAre(
                                 Pair(Var("target"), HasConstantVal(2)))))));
}

// Verifies that the analysis tracks multiple variables simultaneously.
TEST_F(MultiVarConstantPropagationTest, TwoVariables) {
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
  RunDataflow(Code,
              UnorderedElementsAre(
                  Pair("p1", HoldsCPLattice(UnorderedElementsAre(
                                 Pair(Var("target"), HasConstantVal(1))))),
                  Pair("p2", HoldsCPLattice(UnorderedElementsAre(
                                 Pair(Var("target"), HasConstantVal(1)),
                                 Pair(Var("other"), HasConstantVal(2))))),
                  Pair("p3", HoldsCPLattice(UnorderedElementsAre(
                                 Pair(Var("target"), HasConstantVal(3)),
                                 Pair(Var("other"), HasConstantVal(2)))))));
}

TEST_F(MultiVarConstantPropagationTest, TwoVariablesInBranches) {
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
  RunDataflow(Code, UnorderedElementsAre(
                        Pair("p1", HoldsCPLattice(UnorderedElementsAre(
                                       Pair(Var("target"), Varies()),
                                       Pair(Var("other"), Varies())))),
                        Pair("pT", HoldsCPLattice(UnorderedElementsAre(
                                       Pair(Var("target"), HasConstantVal(2)),
                                       Pair(Var("other"), Varies())))),
                        Pair("pF", HoldsCPLattice(UnorderedElementsAre(
                                       Pair(Var("other"), HasConstantVal(3)),
                                       Pair(Var("target"), Varies())))),
                        Pair("p2", HoldsCPLattice(UnorderedElementsAre(
                                       Pair(Var("target"), Varies()),
                                       Pair(Var("other"), Varies()))))));
}

TEST_F(MultiVarConstantPropagationTest, SameAssignmentInBranch) {
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
  RunDataflow(Code, UnorderedElementsAre(
                        Pair("p1", HoldsCPLattice(UnorderedElementsAre(Pair(
                                       Var("target"), HasConstantVal(1))))),
                        Pair("p2", HoldsCPLattice(UnorderedElementsAre(Pair(
                                       Var("target"), HasConstantVal(1)))))));
}

TEST_F(MultiVarConstantPropagationTest, NewVarInBranch) {
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
  RunDataflow(Code, UnorderedElementsAre(
                        Pair("p1", HoldsCPLattice(UnorderedElementsAre(
                                       Pair(Var("target"), Varies())))),
                        Pair("p2", HoldsCPLattice(UnorderedElementsAre(Pair(
                                       Var("target"), HasConstantVal(1))))),
                        Pair("p3", HoldsCPLattice(UnorderedElementsAre(
                                       Pair(Var("target"), Varies())))),
                        Pair("p4", HoldsCPLattice(UnorderedElementsAre(Pair(
                                       Var("target"), HasConstantVal(1)))))));
}

TEST_F(MultiVarConstantPropagationTest, DifferentAssignmentInBranches) {
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
  RunDataflow(Code, UnorderedElementsAre(
                        Pair("p1", HoldsCPLattice(UnorderedElementsAre(
                                       Pair(Var("target"), Varies())))),
                        Pair("pT", HoldsCPLattice(UnorderedElementsAre(Pair(
                                       Var("target"), HasConstantVal(1))))),
                        Pair("pF", HoldsCPLattice(UnorderedElementsAre(Pair(
                                       Var("target"), HasConstantVal(2))))),
                        Pair("p2", HoldsCPLattice(UnorderedElementsAre(
                                       Pair(Var("target"), Varies()))))));
}

TEST_F(MultiVarConstantPropagationTest, DifferentAssignmentInBranch) {
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
  RunDataflow(Code, UnorderedElementsAre(
                        Pair("p1", HoldsCPLattice(UnorderedElementsAre(Pair(
                                       Var("target"), HasConstantVal(1))))),
                        Pair("p2", HoldsCPLattice(UnorderedElementsAre(
                                       Pair(Var("target"), Varies()))))));
}

} // namespace
} // namespace dataflow
} // namespace clang
