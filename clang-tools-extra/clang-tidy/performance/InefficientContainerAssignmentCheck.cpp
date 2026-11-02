//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InefficientContainerAssignmentCheck.h"
#include "../utils/OptionsUtils.h"
#include "../utils/TypeTraits.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Transformer/SourceCode.h"
#include "clang/Tooling/Transformer/SourceCodeBuilders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include <optional>
#include <string>

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

namespace {

// The order matches the %select in the diagnostic.
enum class Rewrite { Assign, AssignRange };

using DeclSet = llvm::SmallPtrSet<const ValueDecl *, 4>;

} // namespace

static bool
anyReferencedDecl(const Stmt &S,
                  llvm::function_ref<bool(const ValueDecl *)> Pred) {
  if (const auto *Ref = dyn_cast<DeclRefExpr>(&S); Ref && Pred(Ref->getDecl()))
    return true;
  if (const auto *Member = dyn_cast<MemberExpr>(&S);
      Member && Pred(Member->getMemberDecl()))
    return true;
  return llvm::any_of(S.children(), [&](const Stmt *Child) {
    return Child && anyReferencedDecl(*Child, Pred);
  });
}

static DeclSet referencedDecls(const Stmt &S) {
  DeclSet Decls;
  anyReferencedDecl(S, [&](const ValueDecl *D) {
    Decls.insert(D);
    return false;
  });
  return Decls;
}

static bool refersToAnyOf(const Stmt &S, const DeclSet &Decls) {
  return anyReferencedDecl(
      S, [&](const ValueDecl *D) { return Decls.contains(D); });
}

// An explicitly passed allocator has no counterpart in 'assign'. It is the
// parameter whose type is a template argument of the container other than the
// element type.
static bool passesAllocator(const CXXConstructorDecl &Ctor, unsigned NumArgs) {
  const auto *Spec =
      dyn_cast<ClassTemplateSpecializationDecl>(Ctor.getParent());
  if (!Spec)
    return false;
  const TemplateArgumentList &TemplateArgs = Spec->getTemplateArgs();
  for (unsigned I = 0, E = std::min(NumArgs, Ctor.getNumParams()); I < E; ++I) {
    const QualType ParamType =
        Ctor.getParamDecl(I)->getType().getNonReferenceType();
    if (!ParamType->isRecordType())
      continue;
    for (unsigned J = 1; J < TemplateArgs.size(); ++J) {
      const TemplateArgument &Arg = TemplateArgs[J];
      if (Arg.getKind() == TemplateArgument::Type &&
          ASTContext::hasSameUnqualifiedType(Arg.getAsType(), ParamType))
        return true;
    }
  }
  return false;
}

static bool isFromRangeTag(QualType T) {
  const CXXRecordDecl *RD = T.getNonReferenceType()->getAsCXXRecordDecl();
  return RD && RD->isInStdNamespace() && RD->getName() == "from_range_t";
}

static QualType getElementType(QualType Container) {
  const auto *Spec = dyn_cast_if_present<ClassTemplateSpecializationDecl>(
      Container->getAsCXXRecordDecl());
  if (!Spec || Spec->getTemplateArgs().size() == 0)
    return {};
  const TemplateArgument &Arg = Spec->getTemplateArgs()[0];
  return Arg.getKind() == TemplateArgument::Type ? Arg.getAsType() : QualType();
}

static const CXXRecordDecl *getDefinition(QualType T) {
  const CXXRecordDecl *Record = T->getAsCXXRecordDecl();
  return Record ? Record->getDefinition() : nullptr;
}

static bool isCopyAssignable(QualType T) {
  T = T.getCanonicalType();
  if (T.isConstQualified())
    return false;
  if (T->isScalarType())
    return true;
  const CXXRecordDecl *Record = getDefinition(T);
  return Record && utils::type_traits::isCopyAssignable(*Record);
}

static const CXXMethodDecl *findNullaryMethod(const CXXRecordDecl &Record,
                                              DeclarationName Name) {
  for (const NamedDecl *D : Record.lookup(Name))
    if (const auto *Method = dyn_cast<CXXMethodDecl>(D);
        Method && Method->getNumParams() == 0)
      return Method;
  for (const CXXBaseSpecifier &Base : Record.bases())
    if (const CXXRecordDecl *BaseRecord = getDefinition(Base.getType()))
      if (const CXXMethodDecl *Method = findNullaryMethod(*BaseRecord, Name))
        return Method;
  return nullptr;
}

// Returns the type of '*It', or a null type if it is unknown.
static QualType getDereferencedType(QualType It, ASTContext &Ctx) {
  It = It.getNonReferenceType();
  if (const auto *Pointer = It->getAs<PointerType>())
    return Pointer->getPointeeType();
  const CXXRecordDecl *Record = getDefinition(It);
  const CXXMethodDecl *Deref =
      Record ? findNullaryMethod(
                   *Record, Ctx.DeclarationNames.getCXXOperatorName(OO_Star))
             : nullptr;
  return Deref ? Deref->getReturnType() : QualType();
}

static QualType getRangeElementType(QualType Range, ASTContext &Ctx) {
  Range = Range.getNonReferenceType();
  if (const ArrayType *Array = Ctx.getAsArrayType(Range))
    return Array->getElementType();
  const CXXRecordDecl *Record = getDefinition(Range);
  const CXXMethodDecl *Begin =
      Record ? findNullaryMethod(*Record, &Ctx.Idents.get("begin")) : nullptr;
  return Begin ? getDereferencedType(Begin->getReturnType(), Ctx) : QualType();
}

static bool isAssignableFrom(QualType T, QualType Source,
                             const ASTContext &Ctx) {
  if (Source.isNull())
    return false;
  T = T.getCanonicalType();
  Source = Source.getNonReferenceType().getCanonicalType().getUnqualifiedType();
  if (T->isScalarType())
    return !T.isConstQualified() && Source->isScalarType();
  if (ASTContext::hasSameUnqualifiedType(T, Source))
    return isCopyAssignable(T);
  const CXXRecordDecl *Record = getDefinition(T);
  return Record &&
         llvm::any_of(Record->methods(), [&](const CXXMethodDecl *Method) {
           if (Method->getOverloadedOperator() != OO_Equal ||
               Method->getNumParams() != 1 || Method->isDeleted() ||
               Method->getAccess() != AS_public)
             return false;
           const QualType Param = Method->getParamDecl(0)
                                      ->getType()
                                      .getNonReferenceType()
                                      .getCanonicalType()
                                      .getUnqualifiedType();
           if (Param == Source)
             return true;
           const auto *ParamPointer = Param->getAs<PointerType>();
           const auto *SourcePointer = Source->getAs<PointerType>();
           return ParamPointer && SourcePointer &&
                  ASTContext::hasSameUnqualifiedType(
                      ParamPointer->getPointeeType(),
                      SourcePointer->getPointeeType()) &&
                  ParamPointer->getPointeeType().isAtLeastAsQualifiedAs(
                      SourcePointer->getPointeeType(), Ctx);
         });
}

// 'assign' needs the elements to be assignable from its arguments, which
// constructing the temporary does not.
static bool canAssign(const CXXConstructorDecl &Ctor,
                      ArrayRef<const Expr *> Args, QualType Element,
                      ASTContext &Ctx) {
  const bool IteratorPair =
      Ctor.getPrimaryTemplate() && Args.size() == 2 &&
      ASTContext::hasSameType(Ctor.getParamDecl(0)->getType(),
                              Ctor.getParamDecl(1)->getType());
  if (IteratorPair)
    return isAssignableFrom(Element,
                            getDereferencedType(Args[0]->getType(), Ctx), Ctx);
  return isCopyAssignable(Element);
}

static std::optional<Rewrite> getRewrite(const CXXConstructorDecl &Ctor,
                                         ArrayRef<const Expr *> Args,
                                         const Expr &LHS, QualType Element,
                                         ASTContext &Ctx) {
  if (Args.size() < 2)
    return std::nullopt;

  // 'assign' must not be given iterators, pointers or elements that refer into
  // the container it replaces.
  const DeclSet LHSDecls = referencedDecls(LHS);
  if (llvm::any_of(Args, [&](const Expr *Arg) {
        return !Arg->getType()->isArithmeticType() &&
               refersToAnyOf(*Arg, LHSDecls);
      }))
    return std::nullopt;

  if (isFromRangeTag(Ctor.getParamDecl(0)->getType())) {
    if (Args.size() != 2 ||
        !isAssignableFrom(Element, getRangeElementType(Args[1]->getType(), Ctx),
                          Ctx))
      return std::nullopt;
    return Rewrite::AssignRange;
  }
  if (!canAssign(Ctor, Args, Element, Ctx))
    return std::nullopt;
  return Rewrite::Assign;
}

// Returns true if E designates a local variable, or a member of one, that
// Other does not mention.
static bool isUnmentionedLocal(const Expr &E, const DeclSet &Other) {
  const Expr *Base = E.IgnoreParenImpCasts();
  for (const auto *Member = dyn_cast<MemberExpr>(Base);
       Member && !Member->isArrow(); Member = dyn_cast<MemberExpr>(Base))
    Base = Member->getBase()->IgnoreParenImpCasts();
  const auto *Ref = dyn_cast<DeclRefExpr>(Base);
  const auto *Var = Ref ? dyn_cast<VarDecl>(Ref->getDecl()) : nullptr;
  return Var && Var->hasLocalStorage() && !Var->getType()->isReferenceType() &&
         !Var->getType().isVolatileQualified() && !Other.contains(Var);
}

static bool readsOnlyUnmentionedLocals(const Stmt &S, const DeclSet &Other) {
  if (isa<CallExpr>(S))
    return false;
  if (const auto *Cast = dyn_cast<ImplicitCastExpr>(&S);
      Cast && Cast->getCastKind() == CK_LValueToRValue)
    return isUnmentionedLocal(*Cast->getSubExpr(), Other);
  return llvm::all_of(S.children(), [&](const Stmt *Child) {
    return !Child || readsOnlyUnmentionedLocals(*Child, Other);
  });
}

// Arguments bound to reference parameters are read by the callee.
static bool isUnaffectedArgument(const Expr &Arg, const DeclSet &LHSDecls,
                                 const ASTContext &Ctx) {
  if (Arg.HasSideEffects(Ctx))
    return false;
  const Expr *E = &Arg;
  if (const auto *Temporary = dyn_cast<MaterializeTemporaryExpr>(E))
    E = Temporary->getSubExpr();
  if (E->isGLValue() && !isUnmentionedLocal(*E, LHSDecls))
    return false;
  return readsOnlyUnmentionedLocals(*E, LHSDecls);
}

// An assignment evaluates its right operand first (since C++17), whereas a
// member call evaluates its object expression first. The rewrite is safe when
// neither side can affect the other.
static bool isEvaluationOrderIndependent(const Expr &LHS,
                                         ArrayRef<const Expr *> Args,
                                         ASTContext &Ctx) {
  DeclSet ArgDecls;
  for (const Expr *Arg : Args)
    ArgDecls.insert_range(referencedDecls(*Arg));
  if (!LHS.HasSideEffects(Ctx) && readsOnlyUnmentionedLocals(LHS, ArgDecls))
    return true;
  const DeclSet LHSDecls = referencedDecls(LHS);
  return llvm::all_of(Args, [&](const Expr *Arg) {
    return isUnaffectedArgument(*Arg, LHSDecls, Ctx);
  });
}

static const Stmt *getParentStmt(const Stmt &S, ASTContext &Ctx) {
  const DynTypedNodeList Parents = Ctx.getParents(S);
  return Parents.size() == 1 ? Parents[0].get<Stmt>() : nullptr;
}

// 'assign', 'clear' and 'resize' do not yield the container.
static bool isValueDiscarded(const Expr &E, ASTContext &Ctx) {
  const Stmt *Child = &E;
  const Stmt *Parent = getParentStmt(E, Ctx);
  while (isa_and_nonnull<ParenExpr, ExprWithCleanups>(Parent)) {
    Child = Parent;
    Parent = getParentStmt(*Parent, Ctx);
  }
  if (!Parent)
    return false;
  if (const auto *Block = dyn_cast<CompoundStmt>(Parent))
    return Block->body_back() != Child ||
           !isa_and_nonnull<StmtExpr>(getParentStmt(*Block, Ctx));
  if (const auto *If = dyn_cast<IfStmt>(Parent))
    return Child != If->getCond();
  if (const auto *For = dyn_cast<ForStmt>(Parent))
    return Child != For->getCond();
  if (const auto *While = dyn_cast<WhileStmt>(Parent))
    return Child == While->getBody();
  if (const auto *Do = dyn_cast<DoStmt>(Parent))
    return Child == Do->getBody();
  if (const auto *Range = dyn_cast<CXXForRangeStmt>(Parent))
    return Child == Range->getBody();
  return isa<CaseStmt, DefaultStmt, LabelStmt, AttributedStmt>(Parent);
}

static std::optional<FixItHint>
createFix(Rewrite Form, const CXXOperatorCallExpr &Op, const Expr &LHS,
          ArrayRef<const Expr *> Args, ASTContext &Ctx) {
  if (Op.getBeginLoc().isMacroID() || Op.getEndLoc().isMacroID())
    return std::nullopt;

  if (!isValueDiscarded(Op, Ctx) ||
      !isEvaluationOrderIndependent(LHS, Args, Ctx))
    return std::nullopt;

  const std::optional<std::string> Access = tooling::buildAccess(LHS, Ctx);
  if (!Access)
    return std::nullopt;
  SmallVector<StringRef, 4> ArgTexts;
  for (const Expr *Arg :
       Form == Rewrite::AssignRange ? Args.drop_front() : Args) {
    ArgTexts.push_back(tooling::getText(*Arg, Ctx));
    if (ArgTexts.back().empty())
      return std::nullopt;
  }

  const StringRef Method =
      Form == Rewrite::AssignRange ? "assign_range" : "assign";
  return FixItHint::CreateReplacement(Op.getSourceRange(),
                                      *Access + Method.str() + "(" +
                                          llvm::join(ArgTexts, ", ") + ")");
}

InefficientContainerAssignmentCheck::InefficientContainerAssignmentCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      ContainerClasses(utils::options::parseStringList(Options.get(
          "ContainerClasses", "::std::vector;::std::deque;::std::list;"
                              "::std::forward_list;::std::basic_string"))) {}

void InefficientContainerAssignmentCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "ContainerClasses",
                utils::options::serializeStringList(ContainerClasses));
}

void InefficientContainerAssignmentCheck::registerMatchers(
    MatchFinder *Finder) {
  if (ContainerClasses.empty())
    return;

  const auto Container = cxxRecordDecl(hasAnyName(ContainerClasses));
  const auto Temporary =
      cxxConstructExpr(hasDeclaration(cxxConstructorDecl(ofClass(Container))))
          .bind("ctor");

  // A rewrite might not fit every instantiation of a template.
  Finder->addMatcher(
      cxxOperatorCallExpr(
          unless(isInTemplateInstantiation()), hasOverloadedOperatorName("="),
          callee(cxxMethodDecl(ofClass(Container))),
          hasArgument(0, expr().bind("lhs")),
          hasArgument(1, ignoringImplicit(anyOf(
                             Temporary, cxxFunctionalCastExpr(has(
                                            ignoringImplicit(Temporary)))))))
          .bind("op"),
      this);
}

void InefficientContainerAssignmentCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Op = Result.Nodes.getNodeAs<CXXOperatorCallExpr>("op");
  const auto *LHS = Result.Nodes.getNodeAs<Expr>("lhs");
  const auto *Ctor = Result.Nodes.getNodeAs<CXXConstructExpr>("ctor");
  ASTContext &Ctx = *Result.Context;

  // A derived class or another specialization would go through a different
  // 'operator='.
  if (!ASTContext::hasSameUnqualifiedType(LHS->getType(), Ctor->getType()))
    return;
  const QualType Element = getElementType(Ctor->getType());
  if (Element.isNull())
    return;

  const CXXConstructorDecl *CtorDecl = Ctor->getConstructor();
  SmallVector<const Expr *, 4> Args;
  for (const Expr *Arg : Ctor->arguments())
    if (!isa<CXXDefaultArgExpr>(Arg))
      Args.push_back(Arg);

  // 'x = Container();' releases the storage, which 'clear()' would keep.
  if (Args.empty() ||
      (Args.size() >= 2 && passesAllocator(*CtorDecl, Args.size())))
    return;

  const std::optional<Rewrite> Form =
      getRewrite(*CtorDecl, Args, *LHS, Element, Ctx);
  if (!Form)
    return;

  const auto Diag = diag(Op->getOperatorLoc(),
                         "inefficient assignment from a temporary '%0'; "
                         "%select{use 'assign'|use 'assign_range'}1 "
                         "to reuse the existing storage")
                    << CtorDecl->getParent()->getQualifiedNameAsString()
                    << static_cast<int>(*Form);
  if (std::optional<FixItHint> Fix = createFix(*Form, *Op, *LHS, Args, Ctx))
    Diag << *Fix;
}

} // namespace clang::tidy::performance
