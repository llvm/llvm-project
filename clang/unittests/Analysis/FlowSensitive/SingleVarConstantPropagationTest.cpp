//===- unittests/Analysis/FlowSensitive/SingleVarConstantPropagation.cpp --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a simplistic version of Constant Propagation as an example
// of a forward, monotonic dataflow analysis. The analysis only tracks one
// variable at a time -- the one with the most recent declaration encountered.
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

// A semi-lattice for dataflow analysis that tracks the value of a single
// integer variable. If it can be identified with a single (constant) value,
// then that value is stored.
struct ConstantPropagationLattice {
  // A null `Var` represents "top": either more than one value is possible or
  // more than one variable was encountered. Otherwise, `Data` indicates that
  // `Var` has the given `Value` at the program point with which this lattice
  // element is associated, for all paths through the program.
  struct VarValue {
    const VarDecl *Var;
    int64_t Value;

    friend bool operator==(VarValue Lhs, VarValue Rhs) {
      return Lhs.Var == Rhs.Var && Lhs.Value == Rhs.Value;
    }
  };
  // `None` is "bottom".
  llvm::Optional<VarValue> Data;

  static constexpr ConstantPropagationLattice bottom() { return {llvm::None}; }
  static constexpr ConstantPropagationLattice top() {
    return {VarValue{nullptr, 0}};
  }

  friend bool operator==(const ConstantPropagationLattice &Lhs,
                         const ConstantPropagationLattice &Rhs) {
    return Lhs.Data == Rhs.Data;
  }

  LatticeJoinEffect join(const ConstantPropagationLattice &Other) {
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

std::ostream &operator<<(std::ostream &OS,
                         const ConstantPropagationLattice &L) {
  if (L == L.bottom())
    return OS << "None";
  if (L == L.top())
    return OS << "Any";
  return OS << L.Data->Var->getName().str() << " = " << L.Data->Value;
}

} // namespace

static constexpr char kVar[] = "var";
static constexpr char kInit[] = "init";
static constexpr char kJustAssignment[] = "just-assignment";
static constexpr char kAssignment[] = "assignment";
static constexpr char kRHS[] = "rhs";

static auto refToVar() { return declRefExpr(to(varDecl().bind(kVar))); }

namespace {
// N.B. This analysis is deliberately simplistic, leaving out many important
// details needed for a real analysis in production. Most notably, the transfer
// function does not account for the variable's address possibly escaping, which
// would invalidate the analysis.
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

  void transfer(const Stmt *S, ConstantPropagationLattice &Element,
                Environment &Env) {
    auto matcher = stmt(
        anyOf(declStmt(hasSingleDecl(varDecl(hasType(isInteger()),
                                             hasInitializer(expr().bind(kInit)))
                                         .bind(kVar))),
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

    if (const auto *E = Nodes.getNodeAs<clang::Expr>(kInit)) {
      Expr::EvalResult R;
      Element =
          (E->EvaluateAsInt(R, Context) && R.Val.isInt())
              ? ConstantPropagationLattice{{{Var,
                                             R.Val.getInt().getExtValue()}}}
              : ConstantPropagationLattice::top();
    } else if (Nodes.getNodeAs<clang::Expr>(kJustAssignment)) {
      const auto *RHS = Nodes.getNodeAs<clang::Expr>(kRHS);
      assert(RHS != nullptr);

      Expr::EvalResult R;
      Element =
          (RHS->EvaluateAsInt(R, Context) && R.Val.isInt())
              ? ConstantPropagationLattice{{{Var,
                                             R.Val.getInt().getExtValue()}}}
              : ConstantPropagationLattice::top();
    } else if (Nodes.getNodeAs<clang::Expr>(kAssignment))
      // Any assignment involving the expression itself resets the variable to
      // "unknown". A more advanced analysis could try to evaluate the compound
      // assignment. For example, `x += 0` need not invalidate `x`.
      Element = ConstantPropagationLattice::top();
  }
};

using ::testing::Pair;
using ::testing::UnorderedElementsAre;

MATCHER_P(HasConstantVal, v, "") { return arg.Data && arg.Data->Value == v; }

MATCHER(IsUnknown, "") { return arg == arg.bottom(); }
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
      test::checkDataflow<ConstantPropagationAnalysis>(
          Code, "fun",
          [](ASTContext &C, Environment &) {
            return ConstantPropagationAnalysis(C);
          },
          [&Expectations](
              llvm::ArrayRef<std::pair<
                  std::string,
                  DataflowAnalysisState<ConstantPropagationAnalysis::Lattice>>>
                  Results,
              ASTContext &) { EXPECT_THAT(Results, Expectations); },
          {"-fsyntax-only", "-std=c++17"}),
      llvm::Succeeded());
}

TEST(ConstantPropagationTest, JustInit) {
  std::string Code = R"(
    void fun() {
      int target = 1;
      // [[p]]
    }
  )";
  RunDataflow(
      Code, UnorderedElementsAre(Pair("p", HoldsCPLattice(HasConstantVal(1)))));
}

// Verifies that the analysis tracks the last variable seen.
TEST(ConstantPropagationTest, TwoVariables) {
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
  RunDataflow(Code, UnorderedElementsAre(
                        Pair("p1", HoldsCPLattice(HasConstantVal(1))),
                        Pair("p2", HoldsCPLattice(HasConstantVal(2))),
                        Pair("p3", HoldsCPLattice(HasConstantVal(3)))));
}

TEST(ConstantPropagationTest, Assignment) {
  std::string Code = R"(
    void fun() {
      int target = 1;
      // [[p1]]
      target = 2;
      // [[p2]]
    }
  )";
  RunDataflow(Code, UnorderedElementsAre(
                        Pair("p1", HoldsCPLattice(HasConstantVal(1))),
                        Pair("p2", HoldsCPLattice(HasConstantVal(2)))));
}

TEST(ConstantPropagationTest, AssignmentCall) {
  std::string Code = R"(
    int g();
    void fun() {
      int target;
      target = g();
      // [[p]]
    }
  )";
  RunDataflow(Code, UnorderedElementsAre(Pair("p", HoldsCPLattice(Varies()))));
}

TEST(ConstantPropagationTest, AssignmentBinOp) {
  std::string Code = R"(
    void fun() {
      int target;
      target = 2 + 3;
      // [[p]]
    }
  )";
  RunDataflow(
      Code, UnorderedElementsAre(Pair("p", HoldsCPLattice(HasConstantVal(5)))));
}

TEST(ConstantPropagationTest, PlusAssignment) {
  std::string Code = R"(
    void fun() {
      int target = 1;
      // [[p1]]
      target += 2;
      // [[p2]]
    }
  )";
  RunDataflow(
      Code, UnorderedElementsAre(Pair("p1", HoldsCPLattice(HasConstantVal(1))),
                                 Pair("p2", HoldsCPLattice(Varies()))));
}

TEST(ConstantPropagationTest, SameAssignmentInBranches) {
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
  RunDataflow(Code, UnorderedElementsAre(
                        Pair("p1", HoldsCPLattice(IsUnknown())),
                        Pair("pT", HoldsCPLattice(HasConstantVal(2))),
                        Pair("pF", HoldsCPLattice(HasConstantVal(2))),
                        Pair("p2", HoldsCPLattice(HasConstantVal(2)))));
}

TEST(ConstantPropagationTest, SameAssignmentInBranch) {
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
                        Pair("p1", HoldsCPLattice(HasConstantVal(1))),
                        Pair("p2", HoldsCPLattice(HasConstantVal(1)))));
}

TEST(ConstantPropagationTest, NewVarInBranch) {
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
                        Pair("p1", HoldsCPLattice(IsUnknown())),
                        Pair("p2", HoldsCPLattice(HasConstantVal(1))),
                        Pair("p3", HoldsCPLattice(IsUnknown())),
                        Pair("p4", HoldsCPLattice(HasConstantVal(1)))));
}

TEST(ConstantPropagationTest, DifferentAssignmentInBranches) {
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
      Code, UnorderedElementsAre(Pair("p1", HoldsCPLattice(IsUnknown())),
                                 Pair("pT", HoldsCPLattice(HasConstantVal(1))),
                                 Pair("pF", HoldsCPLattice(HasConstantVal(2))),
                                 Pair("p2", HoldsCPLattice(Varies()))));
}

TEST(ConstantPropagationTest, DifferentAssignmentInBranch) {
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
      Code, UnorderedElementsAre(Pair("p1", HoldsCPLattice(HasConstantVal(1))),
                                 Pair("p2", HoldsCPLattice(Varies()))));
}

} // namespace
} // namespace dataflow
} // namespace clang
