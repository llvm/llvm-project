//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantCastingCheck.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/NestedNameSpecifierBase.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/TypeBase.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/FixIt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang::ast_matchers;

namespace clang::tidy::llvm_check {

namespace {
AST_MATCHER(Expr, isMacroID) { return Node.getExprLoc().isMacroID(); }
AST_MATCHER_P(OverloadExpr, hasAnyUnresolvedName, ArrayRef<StringRef>, Names) {
  const DeclarationName DeclName = Node.getName();
  if (!DeclName.isIdentifier())
    return false;
  const IdentifierInfo *II = DeclName.getAsIdentifierInfo();
  return llvm::any_of(Names, [II](StringRef Name) { return II->isStr(Name); });
}
} // namespace

static constexpr StringRef CastFunctionNames[] = {
    "cast",     "cast_or_null",     "cast_if_present",
    "dyn_cast", "dyn_cast_or_null", "dyn_cast_if_present"};

static constexpr StringRef IsaFunctionNames[] = {"isa", "isa_and_nonnull",
                                                 "isa_and_present"};

void RedundantCastingCheck::registerMatchers(MatchFinder *Finder) {
  auto IsInLLVMNamespace = hasDeclContext(
      namespaceDecl(hasName("llvm"), hasDeclContext(translationUnitDecl())));
  auto AnyCastCalleeName =
      allOf(unless(isMacroID()), unless(cxxMemberCallExpr()),
            callee(expr(declRefExpr(to(namedDecl(hasAnyName(CastFunctionNames),
                                                 IsInLLVMNamespace)),
                                    templateArgumentLocCountIs(1))
                            .bind("callee"))));
  auto AnyCastCalleeNameInUninstantiatedTemplate = allOf(
      unless(isMacroID()), unless(cxxMemberCallExpr()),
      callee(expr(unresolvedLookupExpr(hasAnyUnresolvedName(CastFunctionNames),
                                       templateArgumentLocCountIs(1))
                      .bind("callee"))));
  Finder->addMatcher(
      callExpr(AnyCastCalleeName, argumentCountIs(1),
               optionally(
                   hasParent(callExpr(AnyCastCalleeName).bind("parent_cast"))))
          .bind("call"),
      this);

  auto AnyIsaCalleeName =
      allOf(unless(isMacroID()), unless(cxxMemberCallExpr()),
            callee(expr(declRefExpr(to(namedDecl(hasAnyName(IsaFunctionNames),
                                                 IsInLLVMNamespace)),
                                    hasAnyTemplateArgumentLoc(anything()))
                            .bind("callee"))));
  Finder->addMatcher(
      callExpr(AnyIsaCalleeName, argumentCountIs(1)).bind("call"), this);

  auto AnyIsaCalleeNameInUninstantiatedTemplate = allOf(
      unless(isMacroID()), unless(cxxMemberCallExpr()),
      callee(expr(unresolvedLookupExpr(hasAnyUnresolvedName(IsaFunctionNames),
                                       hasAnyTemplateArgumentLoc(anything()))
                      .bind("callee"))));
  Finder->addMatcher(
      callExpr(
          anyOf(AnyCastCalleeNameInUninstantiatedTemplate,
                AnyIsaCalleeNameInUninstantiatedTemplate),
          argumentCountIs(1),
          optionally(hasAncestor(
              namespaceDecl(hasName("llvm"), hasParent(translationUnitDecl()))
                  .bind("llvm_ns"))))
          .bind("call"),
      this);
}

static QualType stripPointerOrReference(QualType Ty) {
  QualType Pointee = Ty->getPointeeType();
  if (Pointee.isNull())
    return Ty;
  return Pointee;
}

static bool isLLVMNamespace(NestedNameSpecifier NNS) {
  if (NNS.getKind() != NestedNameSpecifier::Kind::Namespace)
    return false;
  auto Pair = NNS.getAsNamespaceAndPrefix();
  if (Pair.Namespace->getNamespace()->getName() != "llvm")
    return false;
  const NestedNameSpecifier::Kind Kind = Pair.Prefix.getKind();
  return Kind == NestedNameSpecifier::Kind::Null ||
         Kind == NestedNameSpecifier::Kind::Global;
}

void RedundantCastingCheck::check(const MatchFinder::MatchResult &Result) {
  const BoundNodes &Nodes = Result.Nodes;
  const auto *Call = Nodes.getNodeAs<CallExpr>("call");

  ArrayRef<TemplateArgumentLoc> TArgLocs;
  std::string FuncName;
  if (const auto *ResolvedCallee = Nodes.getNodeAs<DeclRefExpr>("callee")) {
    TArgLocs = ResolvedCallee->template_arguments();
    const auto *F = cast<FunctionDecl>(ResolvedCallee->getDecl());
    FuncName = F->getName();
  } else if (const auto *UnresolvedCallee =
                 Nodes.getNodeAs<UnresolvedLookupExpr>("callee")) {
    const bool IsExplicitlyLLVM =
        isLLVMNamespace(UnresolvedCallee->getQualifier());
    const auto *CallerNS = Nodes.getNodeAs<NamedDecl>("llvm_ns");
    if (!IsExplicitlyLLVM && !CallerNS)
      return;

    TArgLocs = UnresolvedCallee->template_arguments();
    FuncName = UnresolvedCallee->getName().getAsString();
  } else {
    llvm_unreachable("unexpected callee kind");
  }

  llvm::SmallVector<CanQualType, 4> TargetTypes;
  for (const TemplateArgumentLoc &TArgLoc : TArgLocs) {
    const TemplateArgument TArg = TArgLoc.getArgument();
    if (TArg.getKind() == TemplateArgument::Type) {
      const CanQualType TargetTy =
          TArg.getAsType()->getCanonicalTypeUnqualified();
      TargetTypes.emplace_back(TargetTy);
    } else if (TArg.getKind() == TemplateArgument::Pack) {
      for (const TemplateArgument &E : TArg.pack_elements()) {
        if (E.getKind() != TemplateArgument::Type)
          return;
        TargetTypes.emplace_back(E.getAsType()->getCanonicalTypeUnqualified());
      }
    } else {
      llvm_unreachable("unexpected template argument");
    }
  }

  const Expr *Arg = Call->getArg(0);
  const QualType ArgTy = Arg->getType();
  const QualType ArgPointeeTy = stripPointerOrReference(ArgTy);
  const CanQualType FromTy = ArgPointeeTy->getCanonicalTypeUnqualified();
  const auto *FromDecl = FromTy->getAsCXXRecordDecl();
  const bool IsIsa = StringRef(FuncName).starts_with("isa");
  if (!IsIsa && TargetTypes.size() != 1)
    return;

  for (const CanQualType TargetTy : TargetTypes) {
    const auto *RetDecl = TargetTy->getAsCXXRecordDecl();
    const bool IsDerived =
        FromDecl && RetDecl && FromDecl->isDerivedFrom(RetDecl);
    if (FromTy != TargetTy && !IsDerived)
      continue;

    if (IsIsa) {
      diag(Call->getExprLoc(), "call to '%0' always succeeds") << FuncName;
    } else {
      QualType ParentTy;
      if (const auto *ParentCast = Nodes.getNodeAs<Expr>("parent_cast")) {
        ParentTy = ParentCast->getType();
      } else {
        // IgnoreUnlessSpelledInSource prevents matching implicit casts
        const TraversalKindScope TmpTraversalKind(*Result.Context, TK_AsIs);
        for (const DynTypedNode Parent : Result.Context->getParents(*Call)) {
          if (const auto *ParentCastExpr = Parent.get<CastExpr>()) {
            ParentTy = ParentCastExpr->getType();
            break;
          }
        }
      }
      if (!ParentTy.isNull()) {
        const CXXRecordDecl *ParentDecl = ParentTy->getAsCXXRecordDecl();
        if (FromDecl && ParentDecl) {
          CXXBasePaths Paths(/*FindAmbiguities=*/true,
                             /*RecordPaths=*/false,
                             /*DetectVirtual=*/false);
          const bool IsDerivedFromParent =
              FromDecl && ParentDecl &&
              FromDecl->isDerivedFrom(ParentDecl, Paths);
          // For the following case a direct `cast<A>(d)` would be ambiguous:
          //   struct A {};
          //   struct B : A {};
          //   struct C : A {};
          //   struct D : B, C {};
          // So we should not warn for `A *a = cast<C>(d)`.
          if (IsDerivedFromParent &&
              Paths.isAmbiguous(ParentTy->getCanonicalTypeUnqualified()))
            return;
        }
      }

      diag(Call->getExprLoc(), "redundant use of '%0'")
          << FuncName
          << tooling::fixit::createReplacement(*Call, *Arg, *Result.Context);
    }
    // printing the canonical type for a template parameter prints as e.g.
    // 'type-parameter-0-0'
    const QualType DiagFromTy(ArgPointeeTy->getUnqualifiedDesugaredType(), 0);
    diag(
        Arg->getExprLoc(),
        "source expression has%select{| pointee}0 type %1%select{|, which is a "
        "subtype of %3}2",
        DiagnosticIDs::Note)
        << Arg->getSourceRange() << ArgTy->isPointerType() << DiagFromTy
        << (FromTy != TargetTy) << TargetTy;
    return;
  }
}

} // namespace clang::tidy::llvm_check
