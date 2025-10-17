//===- UncheckedStatusOrAccessModel.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/Models/UncheckedStatusOrAccessModel.h"

#include <cassert>
#include <utility>

#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/TypeBase.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersInternal.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/CFGMatchSwitch.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/MatchSwitch.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/StringMap.h"

namespace clang::dataflow::statusor_model {
namespace {

using ::clang::ast_matchers::MatchFinder;
using ::clang::ast_matchers::StatementMatcher;

} // namespace

static bool namespaceEquals(const NamespaceDecl *NS,
                            clang::ArrayRef<clang::StringRef> NamespaceNames) {
  while (!NamespaceNames.empty() && NS) {
    if (NS->getName() != NamespaceNames.consume_back())
      return false;
    NS = dyn_cast_or_null<NamespaceDecl>(NS->getParent());
  }
  return NamespaceNames.empty() && !NS;
}

// TODO: move this to a proper place to share with the rest of clang
static bool isTypeNamed(QualType Type, clang::ArrayRef<clang::StringRef> NS,
                        StringRef Name) {
  if (Type.isNull())
    return false;
  if (auto *RD = Type->getAsRecordDecl())
    if (RD->getName() == Name)
      if (const auto *N = dyn_cast_or_null<NamespaceDecl>(RD->getDeclContext()))
        return namespaceEquals(N, NS);
  return false;
}

static bool isStatusOrOperatorBaseType(QualType Type) {
  return isTypeNamed(Type, {"absl", "internal_statusor"}, "OperatorBase");
}

static bool isSafeUnwrap(RecordStorageLocation *StatusOrLoc,
                         const Environment &Env) {
  if (!StatusOrLoc)
    return false;
  auto &StatusLoc = locForStatus(*StatusOrLoc);
  auto *OkVal = Env.get<BoolValue>(locForOk(StatusLoc));
  return OkVal != nullptr && Env.proves(OkVal->formula());
}

static ClassTemplateSpecializationDecl *
getStatusOrBaseClass(const QualType &Ty) {
  auto *RD = Ty->getAsCXXRecordDecl();
  if (RD == nullptr)
    return nullptr;
  if (isStatusOrType(Ty) ||
      // In case we are analyzing code under OperatorBase itself that uses
      // operator* (e.g. to implement operator->).
      isStatusOrOperatorBaseType(Ty))
    return cast<ClassTemplateSpecializationDecl>(RD);
  if (!RD->hasDefinition())
    return nullptr;
  for (const auto &Base : RD->bases())
    if (auto *QT = getStatusOrBaseClass(Base.getType()))
      return QT;
  return nullptr;
}

static QualType getStatusOrValueType(ClassTemplateSpecializationDecl *TRD) {
  return TRD->getTemplateArgs().get(0).getAsType();
}

static auto isStatusOrMemberCallWithName(llvm::StringRef member_name) {
  using namespace ::clang::ast_matchers; // NOLINT: Too many names
  return cxxMemberCallExpr(
      on(expr(unless(cxxThisExpr()))),
      callee(cxxMethodDecl(
          hasName(member_name),
          ofClass(anyOf(statusOrClass(), statusOrOperatorBaseClass())))));
}

static auto isStatusOrOperatorCallWithName(llvm::StringRef operator_name) {
  using namespace ::clang::ast_matchers; // NOLINT: Too many names
  return cxxOperatorCallExpr(
      hasOverloadedOperatorName(operator_name),
      callee(cxxMethodDecl(
          ofClass(anyOf(statusOrClass(), statusOrOperatorBaseClass())))));
}

static auto valueCall() {
  using namespace ::clang::ast_matchers; // NOLINT: Too many names
  return anyOf(isStatusOrMemberCallWithName("value"),
               isStatusOrMemberCallWithName("ValueOrDie"));
}

static auto valueOperatorCall() {
  using namespace ::clang::ast_matchers; // NOLINT: Too many names
  return expr(anyOf(isStatusOrOperatorCallWithName("*"),
                    isStatusOrOperatorCallWithName("->")));
}

static auto
buildDiagnoseMatchSwitch(const UncheckedStatusOrAccessModelOptions &Options) {
  return CFGMatchSwitchBuilder<const Environment,
                               llvm::SmallVector<SourceLocation>>()
      // StatusOr::value, StatusOr::ValueOrDie
      .CaseOfCFGStmt<CXXMemberCallExpr>(
          valueCall(),
          [](const CXXMemberCallExpr *E,
             const ast_matchers::MatchFinder::MatchResult &,
             const Environment &Env) {
            if (!isSafeUnwrap(getImplicitObjectLocation(*E, Env), Env))
              return llvm::SmallVector<SourceLocation>({E->getExprLoc()});
            return llvm::SmallVector<SourceLocation>();
          })

      // StatusOr::operator*, StatusOr::operator->
      .CaseOfCFGStmt<CXXOperatorCallExpr>(
          valueOperatorCall(),
          [](const CXXOperatorCallExpr *E,
             const ast_matchers::MatchFinder::MatchResult &,
             const Environment &Env) {
            RecordStorageLocation *StatusOrLoc =
                Env.get<RecordStorageLocation>(*E->getArg(0));
            if (!isSafeUnwrap(StatusOrLoc, Env))
              return llvm::SmallVector<SourceLocation>({E->getOperatorLoc()});
            return llvm::SmallVector<SourceLocation>();
          })
      .Build();
}

UncheckedStatusOrAccessDiagnoser::UncheckedStatusOrAccessDiagnoser(
    UncheckedStatusOrAccessModelOptions Options)
    : DiagnoseMatchSwitch(buildDiagnoseMatchSwitch(Options)) {}

llvm::SmallVector<SourceLocation> UncheckedStatusOrAccessDiagnoser::operator()(
    const CFGElement &Elt, ASTContext &Ctx,
    const TransferStateForDiagnostics<UncheckedStatusOrAccessModel::Lattice>
        &State) {
  return DiagnoseMatchSwitch(Elt, Ctx, State.Env);
}

BoolValue &initializeStatus(RecordStorageLocation &StatusLoc,
                            Environment &Env) {
  auto &OkVal = Env.makeAtomicBoolValue();
  Env.setValue(locForOk(StatusLoc), OkVal);
  return OkVal;
}

BoolValue &initializeStatusOr(RecordStorageLocation &StatusOrLoc,
                              Environment &Env) {
  return initializeStatus(locForStatus(StatusOrLoc), Env);
}

clang::ast_matchers::DeclarationMatcher statusOrClass() {
  using namespace ::clang::ast_matchers; // NOLINT: Too many names
  return classTemplateSpecializationDecl(
      hasName("absl::StatusOr"),
      hasTemplateArgument(0, refersToType(type().bind("T"))));
}

clang::ast_matchers::DeclarationMatcher statusClass() {
  using namespace ::clang::ast_matchers; // NOLINT: Too many names
  return cxxRecordDecl(hasName("absl::Status"));
}

clang::ast_matchers::DeclarationMatcher statusOrOperatorBaseClass() {
  using namespace ::clang::ast_matchers; // NOLINT: Too many names
  return classTemplateSpecializationDecl(
      hasName("absl::internal_statusor::OperatorBase"));
}

clang::ast_matchers::TypeMatcher statusOrType() {
  using namespace ::clang::ast_matchers; // NOLINT: Too many names
  return hasCanonicalType(qualType(hasDeclaration(statusOrClass())));
}

bool isRecordTypeWithName(QualType Type, llvm::StringRef TypeName) {
  return Type->isRecordType() &&
         Type->getAsCXXRecordDecl()->getQualifiedNameAsString() == TypeName;
}

bool isStatusOrType(QualType Type) {
  return isTypeNamed(Type, {"absl"}, "StatusOr");
}

bool isStatusType(QualType Type) {
  return isTypeNamed(Type, {"absl"}, "Status");
}

llvm::StringMap<QualType> getSyntheticFields(QualType Ty, QualType StatusType,
                                             const CXXRecordDecl &RD) {
  if (auto *TRD = getStatusOrBaseClass(Ty))
    return {{"status", StatusType}, {"value", getStatusOrValueType(TRD)}};
  if (isStatusType(Ty) || (RD.hasDefinition() &&
                           RD.isDerivedFrom(StatusType->getAsCXXRecordDecl())))
    return {{"ok", RD.getASTContext().BoolTy}};
  return {};
}

RecordStorageLocation &locForStatus(RecordStorageLocation &StatusOrLoc) {
  return cast<RecordStorageLocation>(StatusOrLoc.getSyntheticField("status"));
}

StorageLocation &locForOk(RecordStorageLocation &StatusLoc) {
  return StatusLoc.getSyntheticField("ok");
}

BoolValue &valForOk(RecordStorageLocation &StatusLoc, Environment &Env) {
  if (auto *Val = Env.get<BoolValue>(locForOk(StatusLoc)))
    return *Val;
  return initializeStatus(StatusLoc, Env);
}

static void transferStatusOrOkCall(const CXXMemberCallExpr *Expr,
                                   const MatchFinder::MatchResult &,
                                   LatticeTransferState &State) {
  RecordStorageLocation *StatusOrLoc =
      getImplicitObjectLocation(*Expr, State.Env);
  if (StatusOrLoc == nullptr)
    return;

  auto &OkVal = valForOk(locForStatus(*StatusOrLoc), State.Env);
  State.Env.setValue(*Expr, OkVal);
}

CFGMatchSwitch<LatticeTransferState>
buildTransferMatchSwitch(ASTContext &Ctx,
                         CFGMatchSwitchBuilder<LatticeTransferState> Builder) {
  using namespace ::clang::ast_matchers; // NOLINT: Too many names
  return std::move(Builder)
      .CaseOfCFGStmt<CXXMemberCallExpr>(isStatusOrMemberCallWithName("ok"),
                                        transferStatusOrOkCall)
      .Build();
}

QualType findStatusType(const ASTContext &Ctx) {
  for (Type *Ty : Ctx.getTypes())
    if (isStatusType(QualType(Ty, 0)))
      return QualType(Ty, 0);

  return QualType();
}

UncheckedStatusOrAccessModel::UncheckedStatusOrAccessModel(ASTContext &Ctx,
                                                           Environment &Env)
    : DataflowAnalysis<UncheckedStatusOrAccessModel,
                       UncheckedStatusOrAccessModel::Lattice>(Ctx),
      TransferMatchSwitch(buildTransferMatchSwitch(Ctx, {})) {
  QualType StatusType = findStatusType(Ctx);
  Env.getDataflowAnalysisContext().setSyntheticFieldCallback(
      [StatusType](QualType Ty) -> llvm::StringMap<QualType> {
        CXXRecordDecl *RD = Ty->getAsCXXRecordDecl();
        if (RD == nullptr)
          return {};

        if (auto Fields = getSyntheticFields(Ty, StatusType, *RD);
            !Fields.empty())
          return Fields;
        return {};
      });
}

void UncheckedStatusOrAccessModel::transfer(const CFGElement &Elt, Lattice &L,
                                            Environment &Env) {
  LatticeTransferState State(L, Env);
  TransferMatchSwitch(Elt, getASTContext(), State);
}

} // namespace clang::dataflow::statusor_model
