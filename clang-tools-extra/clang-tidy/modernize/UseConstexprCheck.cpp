//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseConstexprCheck.h"
#include "../utils/ASTUtils.h"
#include "../utils/LexerUtils.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

namespace {
AST_MATCHER(FunctionDecl, locationPermitsConstexpr) {
  const bool IsInMainFile =
      Finder->getASTContext().getSourceManager().isInMainFile(
          Node.getLocation());

  if (IsInMainFile && Node.hasExternalFormalLinkage())
    return false;
  if (!IsInMainFile && !Node.isInlined())
    return false;

  return true;
}

AST_MATCHER(Expr, isCXX11ConstantExpr) {
  return !Node.isValueDependent() &&
         Node.isCXX11ConstantExpr(Finder->getASTContext());
}

AST_MATCHER(DeclaratorDecl, isInMacro) {
  const SourceRange R =
      SourceRange(Node.getTypeSpecStartLoc(), Node.getLocation());

  return Node.getLocation().isMacroID() || Node.getEndLoc().isMacroID() ||
         utils::rangeContainsMacroExpansion(
             R, &Finder->getASTContext().getSourceManager()) ||
         utils::rangeIsEntirelyWithinMacroArgument(
             R, &Finder->getASTContext().getSourceManager());
}

AST_MATCHER(Decl, hasNoRedecl) {
  // There is always the actual declaration
  return !Node.redecls().empty() &&
         std::next(Node.redecls_begin()) == Node.redecls_end();
}

AST_MATCHER(Decl, allRedeclsInSameFile) {
  const SourceManager &SM = Finder->getASTContext().getSourceManager();
  const SourceLocation L = Node.getLocation();
  for (const Decl *ReDecl : Node.redecls()) {
    if (!SM.isWrittenInSameFile(L, ReDecl->getLocation()))
      return false;
  }
  return true;
}
} // namespace

static bool
satisfiesConstructorPropertiesUntil20(const CXXConstructorDecl *Ctor,
                                      ASTContext &Ctx) {
  const CXXRecordDecl *Rec = Ctor->getParent();
  llvm::SmallPtrSet<const RecordDecl *, 8> Bases{};
  for (const CXXBaseSpecifier Base : Rec->bases())
    Bases.insert(Base.getType()->getAsRecordDecl());

  llvm::SmallPtrSet<const FieldDecl *, 8> Fields{Rec->field_begin(),
                                                 Rec->field_end()};
  llvm::SmallPtrSet<const FieldDecl *, 4> Indirects{};

  for (const CXXCtorInitializer *const Init : Ctor->inits()) {
    const Type *InitType = Init->getBaseClass();
    if (InitType && InitType->isRecordType()) {
      const auto *ConstructingInit =
          llvm::dyn_cast<CXXConstructExpr>(Init->getInit());
      if (ConstructingInit &&
          !ConstructingInit->getConstructor()->isConstexprSpecified())
        return false;
    }

    if (Init->isBaseInitializer()) {
      Bases.erase(Init->getBaseClass()->getAsRecordDecl());
      continue;
    }

    if (Init->isMemberInitializer()) {
      const FieldDecl *Field = Init->getMember();

      if (Field->isAnonymousStructOrUnion())
        Indirects.insert(Field);

      Fields.erase(Field);
      continue;
    }
  }

  for (const auto &Match :
       match(cxxRecordDecl(forEach(indirectFieldDecl().bind("indirect"))), *Rec,
             Ctx)) {
    const auto *IField = Match.getNodeAs<IndirectFieldDecl>("indirect");

    size_t NumInitializations = false;
    for (const NamedDecl *ND : IField->chain())
      NumInitializations += Indirects.erase(llvm::dyn_cast<FieldDecl>(ND));

    if (NumInitializations != 1)
      return false;

    for (const NamedDecl *ND : IField->chain())
      Fields.erase(llvm::dyn_cast<FieldDecl>(ND));
  }

  if (!Fields.empty())
    return false;

  return true;
}

static bool isLiteralType(QualType QT, const ASTContext &Ctx,
                          bool ConservativeLiteralType);

static bool isLiteralType(const Type *T, const ASTContext &Ctx,
                          const bool ConservativeLiteralType) {
  if (!T)
    return false;

  if (!T->isLiteralType(Ctx))
    return false;

  if (!ConservativeLiteralType)
    return T->isLiteralType(Ctx) && !T->isVoidType();

  if (T->isIncompleteType() || T->isIncompleteArrayType())
    return false;

  T = utils::unwrapPointee(T);
  if (!T)
    return false;

  assert(!T->isPointerOrReferenceType());

  if (T->isIncompleteType() || T->isIncompleteArrayType())
    return false;

  if (T->isLiteralType(Ctx))
    return true;

  if (const CXXRecordDecl *Rec = T->getAsCXXRecordDecl()) {
    if (llvm::any_of(Rec->ctors(), [](const CXXConstructorDecl *Ctor) {
          return !Ctor->isCopyOrMoveConstructor() &&
                 Ctor->isConstexprSpecified();
        }))
      return false;

    for (const CXXBaseSpecifier Base : Rec->bases()) {
      if (!isLiteralType(Base.getType(), Ctx, ConservativeLiteralType))
        return false;
    }
  }

  if (const Type *ArrayElementType = T->getArrayElementTypeNoTypeQual())
    return isLiteralType(ArrayElementType, Ctx, ConservativeLiteralType);

  return false;
}

static bool isLiteralType(QualType QT, const ASTContext &Ctx,
                          const bool ConservativeLiteralType) {
  return !QT.isVolatileQualified() &&
         isLiteralType(QT.getTypePtr(), Ctx, ConservativeLiteralType);
}

static bool satisfiesProperties11(
    const FunctionDecl *FDecl, ASTContext &Ctx,
    const bool ConservativeLiteralType,
    const bool AddConstexprToMethodOfClassWithoutConstexprConstructor) {
  if (FDecl->isConstexprSpecified())
    return true;

  const LangOptions LO = Ctx.getLangOpts();
  const auto *Method = llvm::dyn_cast<CXXMethodDecl>(FDecl);
  if (Method && !Method->isStatic() &&
      !Method->getParent()->hasConstexprNonCopyMoveConstructor() &&
      !AddConstexprToMethodOfClassWithoutConstexprConstructor)
    return false;

  if (Method &&
      (Method->isVirtual() ||
       !match(cxxMethodDecl(hasBody(cxxTryStmt())), *Method, Ctx).empty()))
    return false;

  if (const auto *Ctor = llvm::dyn_cast<CXXConstructorDecl>(FDecl);
      Ctor && (!satisfiesConstructorPropertiesUntil20(Ctor, Ctx) ||
               llvm::any_of(Ctor->getParent()->bases(),
                            [](const CXXBaseSpecifier &Base) {
                              return Base.isVirtual();
                            })))
    return false;

  if (const auto *Dtor = llvm::dyn_cast<CXXDestructorDecl>(FDecl);
      Dtor && !Dtor->isTrivial())
    return false;

  if (!isLiteralType(FDecl->getReturnType(), Ctx, ConservativeLiteralType))
    return false;

  for (const ParmVarDecl *Param : FDecl->parameters())
    if (!isLiteralType(Param->getType(), Ctx, ConservativeLiteralType))
      return false;

  class Visitor11 : public clang::RecursiveASTVisitor<Visitor11> {
  public:
    using Base = clang::RecursiveASTVisitor<Visitor11>;
    bool shouldVisitImplicitCode() const { return true; }

    Visitor11(ASTContext &Ctx, bool ConservativeLiteralType)
        : Ctx(Ctx), ConservativeLiteralType(ConservativeLiteralType) {}

    bool WalkUpFromNullStmt(NullStmt *) {
      Possible = false;
      return false;
    }
    bool WalkUpFromDeclStmt(DeclStmt *DS) {
      for (const Decl *D : DS->decls())
        if (!llvm::isa<StaticAssertDecl, TypedefNameDecl, UsingDecl,
                       UsingDirectiveDecl>(D)) {
          Possible = false;
          return false;
        }
      return true;
    }

    bool WalkUpFromExpr(Expr *) { return true; }
    bool WalkUpFromCompoundStmt(CompoundStmt *S) {
      for (const DynTypedNode &Node : Ctx.getParents(*S))
        if (Node.get<FunctionDecl>() != nullptr)
          return true;

      Possible = false;
      return false;
    }
    bool WalkUpFromStmt(Stmt *) {
      Possible = false;
      return false;
    }

    bool WalkUpFromReturnStmt(ReturnStmt *) {
      ++NumReturns;
      if (NumReturns != 1U) {
        Possible = false;
        return false;
      }
      return true;
    }

    bool WalkUpFromCastExpr(CastExpr *CE) {
      if (llvm::is_contained(
              {
                  CK_LValueBitCast,
                  CK_IntegralToPointer,
                  CK_PointerToIntegral,
              },
              CE->getCastKind())) {
        Possible = false;
        return false;
      }
      return true;
    }

    bool TraverseCXXDynamicCastExpr(CXXDynamicCastExpr *) {
      Possible = false;
      return false;
    }

    bool TraverseCXXReinterpretCastExpr(CXXReinterpretCastExpr *) {
      Possible = false;
      return false;
    }

    bool TraverseType(QualType QT, const bool TraverseQualifier = true) {
      if (QT.isNull())
        return true;
      if (!isLiteralType(QT, Ctx, ConservativeLiteralType)) {
        Possible = false;
        return false;
      }
      return Base::TraverseType(QT, TraverseQualifier);
    }

    bool WalkUpFromCXXConstructExpr(CXXConstructExpr *CE) {
      if (const CXXConstructorDecl *Ctor = CE->getConstructor();
          Ctor && !Ctor->isConstexprSpecified()) {
        Possible = false;
        return false;
      }

      return true;
    }
    bool WalkUpFromCallExpr(CallExpr *CE) {
      if (const auto *FDecl =
              llvm::dyn_cast_if_present<FunctionDecl>(CE->getCalleeDecl());
          FDecl && !FDecl->isConstexprSpecified()) {
        Possible = false;
        return false;
      }
      return true;
    }

    bool TraverseCXXNewExpr(CXXNewExpr *) {
      Possible = false;
      return false;
    }

    bool TraverseDeclRefExpr(DeclRefExpr *DRef) {
      if (DRef->getType().isVolatileQualified()) {
        Possible = false;
        return false;
      }
      return Base::TraverseDeclRefExpr(DRef);
    }

    ASTContext &Ctx;
    const bool ConservativeLiteralType;
    bool Possible = true;
    size_t NumReturns = 0;
  };

  Visitor11 V{Ctx, ConservativeLiteralType};
  V.TraverseDecl(const_cast<FunctionDecl *>(FDecl));
  if (!V.Possible)
    return false;

  return true;
}

namespace {
AST_MATCHER_P2(FunctionDecl, satisfiesProperties, bool, ConservativeLiteralType,
               bool, AddConstexprToMethodOfClassWithoutConstexprConstructor) {
  ASTContext &Ctx = Finder->getASTContext();
  const LangOptions LO = Ctx.getLangOpts();

  if (LO.CPlusPlus11)
    return satisfiesProperties11(
        &Node, Ctx, ConservativeLiteralType,
        AddConstexprToMethodOfClassWithoutConstexprConstructor);

  return false;
}

AST_MATCHER_P(VarDecl, satisfiesVariableProperties, bool,
              ConservativeLiteralType) {
  ASTContext &Ctx = Finder->getASTContext();

  const QualType QT = Node.getType();
  const Type *T = QT.getTypePtr();
  if (!T)
    return false;

  if (!isLiteralType(QT, Ctx, ConservativeLiteralType))
    return false;

  const bool IsDeclaredInsideConstexprFunction = std::invoke([&Node]() {
    const auto *Func = llvm::dyn_cast<FunctionDecl>(Node.getDeclContext());
    if (!Func)
      return false;
    return Func->isConstexpr();
  });

  if (Node.isStaticLocal() && IsDeclaredInsideConstexprFunction)
    return false;

  if (!Ctx.getLangOpts().CPlusPlus20)
    return true;

  const CXXRecordDecl *RDecl = T->getAsCXXRecordDecl();
  const Type *const ArrayOrPtrElement = T->getPointeeOrArrayElementType();
  if (ArrayOrPtrElement)
    RDecl = ArrayOrPtrElement->getAsCXXRecordDecl();

  if (RDecl && (!RDecl->hasDefinition() || !RDecl->hasConstexprDestructor()))
    return false;

  return true;
}
} // namespace

void UseConstexprCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      functionDecl(
          isDefinition(),
          unless(anyOf(isConstexpr(), isImplicit(), hasExternalFormalLinkage(),
                       isInMacro(), isMain(), isInStdNamespace(),
                       isExpansionInSystemHeader(), isExternC(),
                       cxxMethodDecl(ofClass(cxxRecordDecl(isLambda()))))),
          locationPermitsConstexpr(), allRedeclsInSameFile(),
          satisfiesProperties(
              ConservativeLiteralType,
              AddConstexprToMethodOfClassWithoutConstexprConstructor))
          .bind("func"),
      this);

  const auto CallToNonConstexprFunction =
      callExpr(callee(functionDecl(unless(isConstexpr()))));

  const auto VarSupportingConstexpr =
      varDecl(
          unless(anyOf(parmVarDecl(), isImplicit(), isInStdNamespace(),
                       isExpansionInSystemHeader(), isConstexpr(), isExternC(),
                       hasExternalFormalLinkage(), isInMacro())),
          hasNoRedecl(), hasType(qualType(isConstQualified())),
          satisfiesVariableProperties(ConservativeLiteralType),
          hasInitializer(
              expr(isCXX11ConstantExpr(), unless(CallToNonConstexprFunction),
                   unless(hasDescendant(CallToNonConstexprFunction)))))
          .bind("var");
  Finder->addMatcher(mapAnyOf(translationUnitDecl, namespaceDecl)
                         .with(forEach(VarSupportingConstexpr)),
                     this);
  Finder->addMatcher(declStmt(hasSingleDecl(VarSupportingConstexpr)), this);
}

static const FunctionDecl *
maybeResolveToTemplateDecl(const FunctionDecl *Func) {
  if (Func && Func->isTemplateInstantiation())
    Func = Func->getTemplateInstantiationPattern();
  return Func;
}

void UseConstexprCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("func")) {
    Func = maybeResolveToTemplateDecl(Func);
    if (Func)
      Functions.insert(Func);
    return;
  }

  if (const auto *Var = Result.Nodes.getNodeAs<VarDecl>("var")) {
    if (const VarDecl *VarTemplate = Var->getTemplateInstantiationPattern())
      Var = VarTemplate;

    Variables.insert(Var);
    return;
  }
}

void UseConstexprCheck::onEndOfTranslationUnit() {
  const std::string FunctionReplacement = ConstexprString + " ";

  for (const FunctionDecl *Func : Functions) {
    const SourceRange R =
        SourceRange(Func->getTypeSpecStartLoc(), Func->getLocation());
    auto Diag = diag(Func->getLocation(), "declare function %0 as 'constexpr'")
                << Func << R;

    for (const Decl *D : Func->redecls())
      if (const auto *FDecl = llvm::dyn_cast<FunctionDecl>(D))
        Diag << FixItHint::CreateInsertion(FDecl->getTypeSpecStartLoc(),
                                           FunctionReplacement);
  }

  const auto MaybeRemoveConst = [&, this](DiagnosticBuilder &Diag,
                                          const VarDecl *Var) {
    // Since either of the locs can be in a macro, use `makeFileCharRange` to be
    // sure that we have a consistent `CharSourceRange`, located entirely in the
    // source file.
    const CharSourceRange FileRange = Lexer::makeFileCharRange(
        CharSourceRange::getCharRange(Var->getInnerLocStart(),
                                      Var->getLocation()),
        Var->getASTContext().getSourceManager(), getLangOpts());
    if (const std::optional<Token> ConstToken =
            utils::lexer::getQualifyingToken(
                tok::TokenKind::kw_const, FileRange, Var->getASTContext(),
                Var->getASTContext().getSourceManager())) {
      Diag << FixItHint::CreateRemoval(ConstToken->getLocation());
    }
  };

  for (const auto *Var : Variables) {
    const SourceRange R =
        SourceRange(Var->getTypeSpecStartLoc(), Var->getLocation());
    auto Diag = diag(Var->getLocation(), "declare variable %0 as 'constexpr'")
                << Var << R
                << FixItHint::CreateInsertion(Var->getTypeSpecStartLoc(),
                                              FunctionReplacement);
    MaybeRemoveConst(Diag, Var);
  }

  Functions.clear();
  Variables.clear();
}

UseConstexprCheck::UseConstexprCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      ConservativeLiteralType(Options.get("ConservativeLiteralType", true)),
      AddConstexprToMethodOfClassWithoutConstexprConstructor(Options.get(
          "AddConstexprToMethodOfClassWithoutConstexprConstructor", false)),
      ConstexprString(Options.get("ConstexprString", "constexpr")),
      StaticConstexprString(
          Options.get("StaticConstexprString", "static " + ConstexprString)) {}

void UseConstexprCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "ConservativeLiteralType", ConservativeLiteralType);
  Options.store(Opts, "AddConstexprToMethodOfClassWithoutConstexprConstructor",
                AddConstexprToMethodOfClassWithoutConstexprConstructor);
  Options.store(Opts, "ConstexprString", ConstexprString);
  Options.store(Opts, "StaticConstexprString", StaticConstexprString);
}
} // namespace clang::tidy::modernize
