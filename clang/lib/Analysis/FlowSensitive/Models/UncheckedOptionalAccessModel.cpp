#include "clang/Analysis/FlowSensitive/Models/UncheckedOptionalAccessModel.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/MatchSwitch.h"
#include "clang/Analysis/FlowSensitive/SourceLocationsLattice.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <memory>
#include <utility>

namespace clang {
namespace dataflow {
namespace {

using namespace ::clang::ast_matchers;

using LatticeTransferState = TransferState<SourceLocationsLattice>;

auto optionalClass() {
  return classTemplateSpecializationDecl(
      anyOf(hasName("std::optional"), hasName("std::__optional_storage_base"),
            hasName("__optional_destruct_base"), hasName("absl::optional"),
            hasName("base::Optional")),
      hasTemplateArgument(0, refersToType(type().bind("T"))));
}

auto hasOptionalType() { return hasType(optionalClass()); }

auto isOptionalMemberCallWithName(llvm::StringRef MemberName) {
  return cxxMemberCallExpr(
      on(expr(unless(cxxThisExpr()))),
      callee(cxxMethodDecl(hasName(MemberName), ofClass(optionalClass()))));
}

auto isOptionalOperatorCallWithName(llvm::StringRef OperatorName) {
  return cxxOperatorCallExpr(hasOverloadedOperatorName(OperatorName),
                             callee(cxxMethodDecl(ofClass(optionalClass()))));
}

auto isMakeOptionalCall() {
  return callExpr(
      callee(functionDecl(hasAnyName(
          "std::make_optional", "base::make_optional", "absl::make_optional"))),
      hasOptionalType());
}

auto hasNulloptType() {
  return hasType(namedDecl(
      hasAnyName("std::nullopt_t", "absl::nullopt_t", "base::nullopt_t")));
}

auto inPlaceClass() {
  return recordDecl(
      hasAnyName("std::in_place_t", "absl::in_place_t", "base::in_place_t"));
}

auto isOptionalNulloptConstructor() {
  return cxxConstructExpr(hasOptionalType(), argumentCountIs(1),
                          hasArgument(0, hasNulloptType()));
}

auto isOptionalInPlaceConstructor() {
  return cxxConstructExpr(hasOptionalType(),
                          hasArgument(0, hasType(inPlaceClass())));
}

auto isOptionalValueOrConversionConstructor() {
  return cxxConstructExpr(
      hasOptionalType(),
      unless(hasDeclaration(
          cxxConstructorDecl(anyOf(isCopyConstructor(), isMoveConstructor())))),
      argumentCountIs(1), hasArgument(0, unless(hasNulloptType())));
}

/// Creates a symbolic value for an `optional` value using `HasValueVal` as the
/// symbolic value of its "has_value" property.
StructValue &createOptionalValue(Environment &Env, BoolValue &HasValueVal) {
  auto OptionalVal = std::make_unique<StructValue>();
  OptionalVal->setProperty("has_value", HasValueVal);
  return Env.takeOwnership(std::move(OptionalVal));
}

/// Returns the symbolic value that represents the "has_value" property of the
/// optional value `Val`. Returns null if `Val` is null.
BoolValue *getHasValue(Value *Val) {
  if (auto *OptionalVal = cast_or_null<StructValue>(Val)) {
    return cast<BoolValue>(OptionalVal->getProperty("has_value"));
  }
  return nullptr;
}

/// If `Type` is a reference type, returns the type of its pointee. Otherwise,
/// returns `Type` itself.
QualType stripReference(QualType Type) {
  return Type->isReferenceType() ? Type->getPointeeType() : Type;
}

/// Returns true if and only if `Type` is an optional type.
bool IsOptionalType(QualType Type) {
  if (!Type->isRecordType())
    return false;
  // FIXME: Optimize this by avoiding the `getQualifiedNameAsString` call.
  auto TypeName = Type->getAsCXXRecordDecl()->getQualifiedNameAsString();
  return TypeName == "std::optional" || TypeName == "absl::optional" ||
         TypeName == "base::Optional";
}

/// Returns the number of optional wrappers in `Type`.
///
/// For example, if `Type` is `optional<optional<int>>`, the result of this
/// function will be 2.
int countOptionalWrappers(const ASTContext &ASTCtx, QualType Type) {
  if (!IsOptionalType(Type))
    return 0;
  return 1 + countOptionalWrappers(
                 ASTCtx,
                 cast<ClassTemplateSpecializationDecl>(Type->getAsRecordDecl())
                     ->getTemplateArgs()
                     .get(0)
                     .getAsType()
                     .getDesugaredType(ASTCtx));
}

void initializeOptionalReference(const Expr *OptionalExpr,
                                 const MatchFinder::MatchResult &,
                                 LatticeTransferState &State) {
  if (auto *OptionalVal = cast_or_null<StructValue>(
          State.Env.getValue(*OptionalExpr, SkipPast::Reference))) {
    if (OptionalVal->getProperty("has_value") == nullptr) {
      OptionalVal->setProperty("has_value", State.Env.makeAtomicBoolValue());
    }
  }
}

void transferUnwrapCall(const Expr *UnwrapExpr, const Expr *ObjectExpr,
                        LatticeTransferState &State) {
  if (auto *OptionalVal = cast_or_null<StructValue>(
          State.Env.getValue(*ObjectExpr, SkipPast::ReferenceThenPointer))) {
    auto *HasValueVal = getHasValue(OptionalVal);
    assert(HasValueVal != nullptr);

    if (State.Env.flowConditionImplies(*HasValueVal))
      return;
  }

  // Record that this unwrap is *not* provably safe.
  State.Lattice.getSourceLocations().insert(ObjectExpr->getBeginLoc());
}

void transferMakeOptionalCall(const CallExpr *E,
                              const MatchFinder::MatchResult &,
                              LatticeTransferState &State) {
  auto &Loc = State.Env.createStorageLocation(*E);
  State.Env.setStorageLocation(*E, Loc);
  State.Env.setValue(
      Loc, createOptionalValue(State.Env, State.Env.getBoolLiteralValue(true)));
}

void transferOptionalHasValueCall(const CXXMemberCallExpr *CallExpr,
                                  const MatchFinder::MatchResult &,
                                  LatticeTransferState &State) {
  if (auto *OptionalVal = cast_or_null<StructValue>(
          State.Env.getValue(*CallExpr->getImplicitObjectArgument(),
                             SkipPast::ReferenceThenPointer))) {
    auto *HasValueVal = getHasValue(OptionalVal);
    assert(HasValueVal != nullptr);

    auto &CallExprLoc = State.Env.createStorageLocation(*CallExpr);
    State.Env.setValue(CallExprLoc, *HasValueVal);
    State.Env.setStorageLocation(*CallExpr, CallExprLoc);
  }
}

void assignOptionalValue(const Expr &E, LatticeTransferState &State,
                         BoolValue &HasValueVal) {
  if (auto *OptionalLoc =
          State.Env.getStorageLocation(E, SkipPast::ReferenceThenPointer)) {
    State.Env.setValue(*OptionalLoc,
                       createOptionalValue(State.Env, HasValueVal));
  }
}

void transferValueOrConversionConstructor(
    const CXXConstructExpr *E, const MatchFinder::MatchResult &MatchRes,
    LatticeTransferState &State) {
  assert(E->getConstructor()->getTemplateSpecializationArgs()->size() > 0);
  assert(E->getNumArgs() > 0);

  const int TemplateParamOptionalWrappersCount = countOptionalWrappers(
      *MatchRes.Context, stripReference(E->getConstructor()
                                            ->getTemplateSpecializationArgs()
                                            ->get(0)
                                            .getAsType()));
  const int ArgTypeOptionalWrappersCount = countOptionalWrappers(
      *MatchRes.Context, stripReference(E->getArg(0)->getType()));
  auto *HasValueVal =
      (TemplateParamOptionalWrappersCount == ArgTypeOptionalWrappersCount)
          // This is a constructor call for optional<T> with argument of type U
          // such that T is constructible from U.
          ? &State.Env.getBoolLiteralValue(true)
          // This is a constructor call for optional<T> with argument of type
          // optional<U> such that T is constructible from U.
          : getHasValue(State.Env.getValue(*E->getArg(0), SkipPast::Reference));
  if (HasValueVal == nullptr)
    HasValueVal = &State.Env.makeAtomicBoolValue();

  assignOptionalValue(*E, State, *HasValueVal);
}

auto buildTransferMatchSwitch() {
  return MatchSwitchBuilder<LatticeTransferState>()
      // Attach a symbolic "has_value" state to optional values that we see for
      // the first time.
      .CaseOf<Expr>(expr(anyOf(declRefExpr(), memberExpr()), hasOptionalType()),
                    initializeOptionalReference)

      // make_optional
      .CaseOf<CallExpr>(isMakeOptionalCall(), transferMakeOptionalCall)

      // constructors:
      .CaseOf<CXXConstructExpr>(
          isOptionalInPlaceConstructor(),
          [](const CXXConstructExpr *E, const MatchFinder::MatchResult &,
             LatticeTransferState &State) {
            assignOptionalValue(*E, State, State.Env.getBoolLiteralValue(true));
          })
      .CaseOf<CXXConstructExpr>(
          isOptionalNulloptConstructor(),
          [](const CXXConstructExpr *E, const MatchFinder::MatchResult &,
             LatticeTransferState &State) {
            assignOptionalValue(*E, State,
                                State.Env.getBoolLiteralValue(false));
          })
      .CaseOf<CXXConstructExpr>(isOptionalValueOrConversionConstructor(),
                                transferValueOrConversionConstructor)

      // optional::value
      .CaseOf<CXXMemberCallExpr>(
          isOptionalMemberCallWithName("value"),
          [](const CXXMemberCallExpr *E, const MatchFinder::MatchResult &,
             LatticeTransferState &State) {
            transferUnwrapCall(E, E->getImplicitObjectArgument(), State);
          })

      // optional::operator*, optional::operator->
      .CaseOf<CallExpr>(expr(anyOf(isOptionalOperatorCallWithName("*"),
                                   isOptionalOperatorCallWithName("->"))),
                        [](const CallExpr *E, const MatchFinder::MatchResult &,
                           LatticeTransferState &State) {
                          transferUnwrapCall(E, E->getArg(0), State);
                        })

      // optional::has_value
      .CaseOf<CXXMemberCallExpr>(isOptionalMemberCallWithName("has_value"),
                                 transferOptionalHasValueCall)

      // optional::operator bool
      .CaseOf<CXXMemberCallExpr>(isOptionalMemberCallWithName("operator bool"),
                                 transferOptionalHasValueCall)

      // optional::emplace
      .CaseOf<CXXMemberCallExpr>(
          isOptionalMemberCallWithName("emplace"),
          [](const CXXMemberCallExpr *E, const MatchFinder::MatchResult &,
             LatticeTransferState &State) {
            assignOptionalValue(*E->getImplicitObjectArgument(), State,
                                State.Env.getBoolLiteralValue(true));
          })

      // optional::reset
      .CaseOf<CXXMemberCallExpr>(
          isOptionalMemberCallWithName("reset"),
          [](const CXXMemberCallExpr *E, const MatchFinder::MatchResult &,
             LatticeTransferState &State) {
            assignOptionalValue(*E->getImplicitObjectArgument(), State,
                                State.Env.getBoolLiteralValue(false));
          })

      .Build();
}

} // namespace

UncheckedOptionalAccessModel::UncheckedOptionalAccessModel(ASTContext &Ctx)
    : DataflowAnalysis<UncheckedOptionalAccessModel, SourceLocationsLattice>(
          Ctx),
      TransferMatchSwitch(buildTransferMatchSwitch()) {}

void UncheckedOptionalAccessModel::transfer(const Stmt *S,
                                            SourceLocationsLattice &L,
                                            Environment &Env) {
  LatticeTransferState State(L, Env);
  TransferMatchSwitch(*S, getASTContext(), State);
}

} // namespace dataflow
} // namespace clang
