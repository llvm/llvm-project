//===-- UncheckedExpectedAccessModel.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a dataflow analysis that detects unsafe accesses to the
//  values of `std::expected` and `tl::expected` objects.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/Models/UncheckedExpectedAccessModel.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"
#include "clang/Analysis/FlowSensitive/Arena.h"
#include "clang/Analysis/FlowSensitive/Formula.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "clang/Analysis/FlowSensitive/Value.h"

namespace clang {
namespace dataflow {

/// Returns true if `DC` is `std` or the top-level `tl` namespace, which holds
/// `tl::expected` (https://github.com/TartanLlama/expected).
static bool isSupportedNamespace(const DeclContext &DC) {
  if (DC.isStdNamespace())
    return true;
  const auto *NS = dyn_cast<NamespaceDecl>(&DC);
  return NS != nullptr && NS->getDeclName().isIdentifier() &&
         NS->getName() == "tl" && NS->getParent()->isTranslationUnit();
}

static bool isExpectedClass(const CXXRecordDecl &RD) {
  return RD.getDeclName().isIdentifier() && RD.getName() == "expected" &&
         isSupportedNamespace(*RD.getDeclContext());
}

/// Returns the name of the expected class `RD` to show in diagnostics.
static llvm::StringRef getExpectedTypeName(const CXXRecordDecl &RD) {
  return RD.getDeclContext()->isStdNamespace() ? "std::expected"
                                               : "tl::expected";
}

// Classes derived from the expected class are deliberately not
// supported.
static bool isExpectedType(QualType Ty) {
  const CXXRecordDecl *RD = Ty->getAsCXXRecordDecl();
  return RD != nullptr && isExpectedClass(*RD);
}

namespace {

using namespace ::clang::ast_matchers;

using LatticeTransferState = TransferState<NoopLattice>;

AST_MATCHER(CXXRecordDecl, expectedClass) { return isExpectedClass(Node); }

auto expectedMemberCall(ast_matchers::internal::Matcher<NamedDecl> Name) {
  return cxxMemberCallExpr(
      callee(cxxMethodDecl(Name, ofClass(expectedClass()))));
}

auto expectedOperatorCall(ArrayRef<StringRef> Names) {
  return cxxOperatorCallExpr(hasAnyOverloadedOperatorName(Names),
                             callee(cxxMethodDecl(ofClass(expectedClass()))));
}

StorageLocation &locForHasValue(const RecordStorageLocation &ExpectedLoc) {
  return ExpectedLoc.getSyntheticField("has_value");
}

/// Returns the symbolic value that represents the "has_value" property of the
/// expected at `ExpectedLoc`. Returns null if `ExpectedLoc` is null or is not
/// the location of a supported expected object (e.g. of a
/// derived class object).
BoolValue *getHasValue(Environment &Env, RecordStorageLocation *ExpectedLoc) {
  if (ExpectedLoc == nullptr || !isExpectedType(ExpectedLoc->getType()))
    return nullptr;
  StorageLocation &HasValueLoc = locForHasValue(*ExpectedLoc);
  auto *HasValueVal = Env.get<BoolValue>(HasValueLoc);
  if (HasValueVal == nullptr) {
    HasValueVal = &Env.makeAtomicBoolValue();
    Env.setValue(HasValueLoc, *HasValueVal);
  }
  return HasValueVal;
}

void transferHasValueCall(const CXXMemberCallExpr *E,
                          const MatchFinder::MatchResult &,
                          LatticeTransferState &State) {
  if (BoolValue *HasValueVal =
          getHasValue(State.Env, getImplicitObjectLocation(*E, State.Env)))
    State.Env.setValue(*E, *HasValueVal);
}

// FIXME: Model the constructors, assignments, `emplace` and `swap`.
auto buildTransferMatchSwitch() {
  return CFGMatchSwitchBuilder<LatticeTransferState>()
      // expected::has_value, expected::operator bool
      .CaseOfCFGStmt<CXXMemberCallExpr>(
          expectedMemberCall(hasAnyName("has_value", "operator bool")),
          transferHasValueCall)
      .Build();
}

/// `ExpectedClass` is the class of the called accessor.
llvm::SmallVector<UncheckedExpectedAccessDiagnostic>
diagnoseAccess(SourceLocation AccessLoc, UncheckedExpectedAccessKind Kind,
               const CXXRecordDecl &ExpectedClass,
               const RecordStorageLocation *ExpectedLoc,
               const Environment &Env) {
  if (ExpectedLoc != nullptr) {
    // The matchers only check the class of the called accessor, but the object
    // may be of a derived class, whose location has no "has_value" field.
    if (!isExpectedType(ExpectedLoc->getType()))
      return {};
    if (auto *HasValueVal = Env.get<BoolValue>(locForHasValue(*ExpectedLoc))) {
      // The value may only be accessed if the expected holds a value, and the
      // error only if it does not.
      const Formula &Safe = Kind == UncheckedExpectedAccessKind::Value
                                ? HasValueVal->formula()
                                : Env.arena().makeNot(HasValueVal->formula());
      if (Env.proves(Safe))
        return {};
    }
  }

  return {{AccessLoc, Kind, getExpectedTypeName(ExpectedClass)}};
}

auto buildDiagnoseMatchSwitch() {
  return CFGMatchSwitchBuilder<
             const Environment,
             llvm::SmallVector<UncheckedExpectedAccessDiagnostic>>()
      // expected::value
      .CaseOfCFGStmt<CXXMemberCallExpr>(
          expectedMemberCall(hasName("value")),
          [](const CXXMemberCallExpr *E, const MatchFinder::MatchResult &,
             const Environment &Env) {
            return diagnoseAccess(E->getExprLoc(),
                                  UncheckedExpectedAccessKind::Value,
                                  *E->getMethodDecl()->getParent(),
                                  getImplicitObjectLocation(*E, Env), Env);
          })
      // expected::operator*, expected::operator->
      .CaseOfCFGStmt<CXXOperatorCallExpr>(
          expectedOperatorCall({"*", "->"}),
          [](const CXXOperatorCallExpr *E, const MatchFinder::MatchResult &,
             const Environment &Env) {
            // `getExprLoc` of `operator->` is the start of the object
            // expression, so point at the operator token instead.
            return diagnoseAccess(
                E->getOperatorLoc(), UncheckedExpectedAccessKind::Value,
                *cast<CXXMethodDecl>(E->getCalleeDecl())->getParent(),
                Env.get<RecordStorageLocation>(*E->getArg(0)), Env);
          })
      // expected::error
      .CaseOfCFGStmt<CXXMemberCallExpr>(
          expectedMemberCall(hasName("error")),
          [](const CXXMemberCallExpr *E, const MatchFinder::MatchResult &,
             const Environment &Env) {
            return diagnoseAccess(E->getExprLoc(),
                                  UncheckedExpectedAccessKind::Error,
                                  *E->getMethodDecl()->getParent(),
                                  getImplicitObjectLocation(*E, Env), Env);
          })
      .Build();
}

} // namespace

ast_matchers::StatementMatcher
UncheckedExpectedAccessModel::callToExpectedClass() {
  return callExpr(callee(cxxMethodDecl(ofClass(expectedClass()))));
}

UncheckedExpectedAccessModel::UncheckedExpectedAccessModel(ASTContext &Ctx,
                                                           Environment &Env)
    : DataflowAnalysis<UncheckedExpectedAccessModel, NoopLattice>(Ctx),
      TransferMatchSwitch(buildTransferMatchSwitch()) {
  Env.getDataflowAnalysisContext().setSyntheticFieldCallback(
      [&Ctx](QualType Ty) -> llvm::StringMap<QualType> {
        if (!isExpectedType(Ty))
          return {};
        return {{"has_value", Ctx.BoolTy}};
      });
}

void UncheckedExpectedAccessModel::transfer(const CFGElement &Elt,
                                            NoopLattice &L, Environment &Env) {
  LatticeTransferState State(L, Env);
  TransferMatchSwitch(Elt, getASTContext(), State);
}

UncheckedExpectedAccessDiagnoser::UncheckedExpectedAccessDiagnoser()
    : DiagnoseMatchSwitch(buildDiagnoseMatchSwitch()) {}

} // namespace dataflow
} // namespace clang
