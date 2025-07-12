//===--- UseConstexprCheck.cpp - clang-tidy--------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseConstexprCheck.h"
#include "../utils/ASTUtils.h"
#include "../utils/LexerUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersInternal.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/TokenKinds.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include <cstddef>
#include <functional>

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

static const Type *unwrapPointee(const Type *T) {
  if (!T->isPointerOrReferenceType())
    return T;

  while (T && T->isPointerOrReferenceType()) {
    if (T->isReferenceType()) {
      const QualType QType = T->getPointeeType();
      if (!QType.isNull())
        T = QType.getTypePtr();
    } else
      T = T->getPointeeOrArrayElementType();
  }

  return T;
}

static bool isLiteralType(QualType QT, const ASTContext &Ctx,
                          const bool ConservativeLiteralType);

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

  T = unwrapPointee(T);
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

    bool WalkUpFromReturnStmt(ReturnStmt *S) {
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

    bool TraverseType(QualType QT) {
      if (QT.isNull())
        return true;
      if (!isLiteralType(QT, Ctx, ConservativeLiteralType)) {
        Possible = false;
        return false;
      }
      return Base::TraverseType(QT);
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

// The only difference between C++14 and C++17 is that `constexpr` lambdas
// can be used in C++17.
static bool satisfiesProperties1417(
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

  if (Method && Method->isVirtual())
    return false;

  if (llvm::isa<CXXConstructorDecl>(FDecl) &&
      llvm::any_of(
          Method->getParent()->bases(),
          [](const CXXBaseSpecifier &Base) { return Base.isVirtual(); }))
    return false;

  if (!isLiteralType(FDecl->getReturnType(), Ctx, ConservativeLiteralType))
    return false;

  for (const ParmVarDecl *Param : FDecl->parameters())
    if (!isLiteralType(Param->getType(), Ctx, ConservativeLiteralType))
      return false;

  class Visitor14 : public clang::RecursiveASTVisitor<Visitor14> {
  public:
    using Base = clang::RecursiveASTVisitor<Visitor14>;
    bool shouldVisitImplicitCode() const { return true; }

    Visitor14(bool CXX17, ASTContext &Ctx, bool ConservativeLiteralType,
              bool AddConstexprToMethodOfClassWithoutConstexprConstructor)
        : CXX17(CXX17), Ctx(Ctx),
          ConservativeLiteralType(ConservativeLiteralType),
          AddConstexprToMethodOfClassWithoutConstexprConstructor(
              AddConstexprToMethodOfClassWithoutConstexprConstructor) {}

    bool TraverseGotoStmt(GotoStmt *) {
      Possible = false;
      return false;
    }
    bool TraverseLabelStmt(LabelStmt *) {
      Possible = false;
      return false;
    }
    bool TraverseCXXTryStmt(CXXTryStmt *) {
      Possible = false;
      return false;
    }
    bool TraverseGCCAsmStmt(GCCAsmStmt *) {
      Possible = false;
      return false;
    }
    bool TraverseMSAsmStmt(MSAsmStmt *) {
      Possible = false;
      return false;
    }
    bool TraverseDecompositionDecl(DecompositionDecl * /*DD*/) {
      Possible = false;
      return false;
    }
    bool TraverseVarDecl(VarDecl *VD) {
      const StorageDuration StorageDur = VD->getStorageDuration();
      Possible = VD->hasInit() &&
                 isLiteralType(VD->getType(), VD->getASTContext(),
                               ConservativeLiteralType) &&
                 (StorageDur != StorageDuration::SD_Static &&
                  StorageDur != StorageDuration::SD_Thread);
      return Possible && Base::TraverseVarDecl(VD);
    }
    bool TraverseLambdaExpr(LambdaExpr *LE) {
      if (CXX17) {
        Possible = satisfiesProperties1417(
            LE->getCallOperator(), Ctx, ConservativeLiteralType,
            AddConstexprToMethodOfClassWithoutConstexprConstructor);
        return Possible;
      }
      Possible = false;
      return false;
    }
    bool TraverseCXXNewExpr(CXXNewExpr *) {
      Possible = false;
      return false;
    }

    bool TraverseDeclRefExpr(DeclRefExpr *DRef) {
      if (const auto *D = llvm::dyn_cast_if_present<VarDecl>(DRef->getDecl());
          D && !D->isLocalVarDeclOrParm() && D->hasGlobalStorage()) {
        Possible = false;
        return false;
      }

      if (DRef->getType().isVolatileQualified()) {
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

    bool TraverseType(QualType QT) {
      if (QT.isNull())
        return true;
      if (!isLiteralType(QT, Ctx, ConservativeLiteralType)) {
        Possible = false;
        return false;
      }
      return Base::TraverseType(QT);
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

    const bool CXX17;
    bool Possible = true;
    ASTContext &Ctx;
    const bool ConservativeLiteralType;
    const bool AddConstexprToMethodOfClassWithoutConstexprConstructor;
  };

  Visitor14 V{Ctx.getLangOpts().CPlusPlus17 != 0, Ctx, ConservativeLiteralType,
              AddConstexprToMethodOfClassWithoutConstexprConstructor};
  V.TraverseDecl(const_cast<FunctionDecl *>(FDecl));
  if (!V.Possible)
    return false;

  if (const auto *Ctor = llvm::dyn_cast<CXXConstructorDecl>(FDecl);
      Ctor && !satisfiesConstructorPropertiesUntil20(Ctor, Ctx))
    return false;

  if (const auto *Dtor = llvm::dyn_cast<CXXDestructorDecl>(FDecl);
      Dtor && !Dtor->isTrivial())
    return false;

  return true;
}

static bool satisfiesProperties20(
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

  if (FDecl->hasBody() && llvm::isa<CoroutineBodyStmt>(FDecl->getBody()))
    return false;

  if ((llvm::isa<CXXConstructorDecl>(FDecl) ||
       llvm::isa<CXXDestructorDecl>(FDecl)) &&
      llvm::any_of(
          Method->getParent()->bases(),
          [](const CXXBaseSpecifier &Base) { return Base.isVirtual(); }))
    return false;

  if (!isLiteralType(FDecl->getReturnType(), Ctx, ConservativeLiteralType))
    return false;

  for (const ParmVarDecl *Param : FDecl->parameters())
    if (!isLiteralType(Param->getType(), Ctx, ConservativeLiteralType))
      return false;

  class Visitor20 : public clang::RecursiveASTVisitor<Visitor20> {
  public:
    bool shouldVisitImplicitCode() const { return true; }

    Visitor20(bool ConservativeLiteralType)
        : ConservativeLiteralType(ConservativeLiteralType) {}

    bool TraverseGotoStmt(GotoStmt *) {
      Possible = false;
      return false;
    }
    bool TraverseLabelStmt(LabelStmt *) {
      Possible = false;
      return false;
    }
    bool TraverseCXXTryStmt(CXXTryStmt *) {
      Possible = false;
      return false;
    }
    bool TraverseGCCAsmStmt(GCCAsmStmt *) {
      Possible = false;
      return false;
    }
    bool TraverseMSAsmStmt(MSAsmStmt *) {
      Possible = false;
      return false;
    }
    bool TraverseDecompositionDecl(DecompositionDecl * /*DD*/) {
      Possible = false;
      return false;
    }
    bool TraverseVarDecl(VarDecl *VD) {
      const StorageDuration StorageDur = VD->getStorageDuration();
      Possible = isLiteralType(VD->getType(), VD->getASTContext(),
                               ConservativeLiteralType) &&
                 (StorageDur != StorageDuration::SD_Static &&
                  StorageDur != StorageDuration::SD_Thread);
      return Possible;
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

    bool TraverseCXXReinterpretCastExpr(CXXReinterpretCastExpr *) {
      Possible = false;
      return false;
    }

    bool TraverseDeclRefExpr(DeclRefExpr *DRef) {
      if (DRef->getType().isVolatileQualified()) {
        Possible = false;
        return false;
      }
      return true;
    }

    bool Possible = true;
    bool ConservativeLiteralType;
  };

  Visitor20 V{ConservativeLiteralType};
  V.TraverseDecl(const_cast<FunctionDecl *>(FDecl));
  if (!V.Possible)
    return false;

  if (const auto *Ctor = llvm::dyn_cast<CXXConstructorDecl>(FDecl))
    satisfiesConstructorPropertiesUntil20(Ctor, Ctx);

  if (const auto *Dtor = llvm::dyn_cast<CXXDestructorDecl>(FDecl);
      Dtor && !Dtor->isTrivial())
    return false;

  class BodyVisitor : public clang::RecursiveASTVisitor<BodyVisitor> {
  public:
    using Base = clang::RecursiveASTVisitor<BodyVisitor>;
    bool shouldVisitImplicitCode() const { return true; }

    explicit BodyVisitor(const ASTContext &Ctx, bool ConservativeLiteralType)
        : Ctx(Ctx), LO(Ctx.getLangOpts()),
          ConservativeLiteralType(ConservativeLiteralType) {}

    bool TraverseType(QualType QT) {
      if (QT.isNull())
        return true;
      if (!isLiteralType(QT, Ctx, ConservativeLiteralType)) {
        Possible = false;
        return false;
      }
      return Base::TraverseType(QT);
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

    const ASTContext &Ctx;
    const LangOptions &LO;
    const bool ConservativeLiteralType;
    bool Possible = true;
  };

  if (FDecl->hasBody() && ConservativeLiteralType) {
    BodyVisitor Visitor(Ctx, ConservativeLiteralType);
    Visitor.TraverseStmt(FDecl->getBody());
    if (!Visitor.Possible)
      return false;
  }
  return true;
}

static bool satisfiesProperties2326(
    const FunctionDecl *FDecl, ASTContext &Ctx,
    const bool AddConstexprToMethodOfClassWithoutConstexprConstructor) {
  if (FDecl->isConstexprSpecified())
    return true;

  const LangOptions LO = Ctx.getLangOpts();
  const auto *Method = llvm::dyn_cast<CXXMethodDecl>(FDecl);
  if (Method && !Method->isStatic() &&
      !Method->getParent()->hasConstexprNonCopyMoveConstructor() &&
      !AddConstexprToMethodOfClassWithoutConstexprConstructor)
    return false;

  if (FDecl->hasBody() && llvm::isa<CoroutineBodyStmt>(FDecl->getBody()))
    return false;

  if ((llvm::isa<CXXConstructorDecl>(FDecl) ||
       llvm::isa<CXXDestructorDecl>(FDecl)) &&
      llvm::any_of(
          Method->getParent()->bases(),
          [](const CXXBaseSpecifier &Base) { return Base.isVirtual(); }))
    return false;
  return true;
}

namespace {
// FIXME: fix CXX23 allowing decomposition decls, but it is only a feature since
// CXX26
AST_MATCHER_P2(FunctionDecl, satisfiesProperties, bool, ConservativeLiteralType,
               bool, AddConstexprToMethodOfClassWithoutConstexprConstructor) {
  ASTContext &Ctx = Finder->getASTContext();
  const LangOptions LO = Ctx.getLangOpts();

  if (LO.CPlusPlus26) {
    return satisfiesProperties2326(
        &Node, Ctx, AddConstexprToMethodOfClassWithoutConstexprConstructor);
  }
  if (LO.CPlusPlus23) {
    return satisfiesProperties2326(
        &Node, Ctx, AddConstexprToMethodOfClassWithoutConstexprConstructor);
  }
  if (LO.CPlusPlus20) {
    return satisfiesProperties20(
        &Node, Ctx, ConservativeLiteralType,
        AddConstexprToMethodOfClassWithoutConstexprConstructor);
  }
  if (LO.CPlusPlus17) {
    return satisfiesProperties1417(
        &Node, Ctx, ConservativeLiteralType,
        AddConstexprToMethodOfClassWithoutConstexprConstructor);
  }
  if (LO.CPlusPlus14) {
    return satisfiesProperties1417(
        &Node, Ctx, ConservativeLiteralType,
        AddConstexprToMethodOfClassWithoutConstexprConstructor);
  }
  if (LO.CPlusPlus11)
    return satisfiesProperties11(
        &Node, Ctx, ConservativeLiteralType,
        AddConstexprToMethodOfClassWithoutConstexprConstructor);

  return false;
}

AST_MATCHER_P(VarDecl, satisfiesVariableProperties, bool,
              ConservativeLiteralType) {
  ASTContext &Ctx = Finder->getASTContext();
  const LangOptions LO = Ctx.getLangOpts();

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
    return !Func->isConstexpr();
  });

  if (!Finder->getASTContext().getLangOpts().CPlusPlus23 &&
      Node.isStaticLocal() && IsDeclaredInsideConstexprFunction)
    return false;

  if (!Finder->getASTContext().getLangOpts().CPlusPlus20)
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

  Finder->addMatcher(
      varDecl(
          unless(anyOf(parmVarDecl(), isImplicit(), isInStdNamespace(),
                       isExpansionInSystemHeader(), isConstexpr(), isExternC(),
                       hasExternalFormalLinkage(), isInMacro())),
          hasNoRedecl(), hasType(qualType(isConstQualified())),
          satisfiesVariableProperties(ConservativeLiteralType),
          hasInitializer(expr(isCXX11ConstantExpr())))
          .bind("var"),
      this);
}

void UseConstexprCheck::check(const MatchFinder::MatchResult &Result) {
  constexpr const auto MaybeResolveToTemplateDecl =
      [](const FunctionDecl *Func) {
        if (Func && Func->isTemplateInstantiation())
          Func = Func->getTemplateInstantiationPattern();
        return Func;
      };

  if (const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("func")) {
    Func = MaybeResolveToTemplateDecl(Func);
    if (Func)
      Functions.insert(Func);
    return;
  }

  if (const auto *Var = Result.Nodes.getNodeAs<VarDecl>("var")) {
    if (const VarDecl *VarTemplate = Var->getTemplateInstantiationPattern())
      Var = VarTemplate;

    VariableMapping.insert({Var, MaybeResolveToTemplateDecl(
                                     llvm::dyn_cast_if_present<FunctionDecl>(
                                         Var->getDeclContext()))});
    return;
  }
}

void UseConstexprCheck::onEndOfTranslationUnit() {
  const std::string FunctionReplacement = ConstexprString + " ";
  for (const FunctionDecl *Func : Functions) {
    const SourceRange R =
        SourceRange(Func->getTypeSpecStartLoc(), Func->getLocation());
    auto Diag =
        diag(Func->getLocation(), "function %0 can be declared 'constexpr'")
        << Func << R;

    for (const Decl *D : Func->redecls())
      if (const auto *FDecl = llvm::dyn_cast<FunctionDecl>(D))
        Diag << FixItHint::CreateInsertion(FDecl->getTypeSpecStartLoc(),
                                           FunctionReplacement);
  }

  const std::string VariableReplacementWithStatic = StaticConstexprString + " ";
  const auto VariableReplacement =
      [&FunctionReplacement, this, &VariableReplacementWithStatic](
          const VarDecl *Var, const FunctionDecl *FuncCtx,
          const bool IsAddingConstexprToFuncCtx) -> const std::string & {
    if (!FuncCtx)
      return FunctionReplacement;

    if (!getLangOpts().CPlusPlus23)
      return FunctionReplacement;

    // We'll prefer the function to be constexpr over the function not being
    // constexpr just for the var to be static constexpr instead of just
    // constexpr.
    if (IsAddingConstexprToFuncCtx)
      return FunctionReplacement;

    if (Var->isStaticLocal())
      return FunctionReplacement;

    return VariableReplacementWithStatic;
  };

  for (const auto &[Var, FuncCtx] : VariableMapping) {
    const bool IsAddingConstexprToFuncCtx = Functions.contains(FuncCtx);
    if (FuncCtx && getLangOpts().CPlusPlus23 && Var->isStaticLocal() &&
        IsAddingConstexprToFuncCtx)
      continue;
    const SourceRange R =
        SourceRange(Var->getTypeSpecStartLoc(), Var->getLocation());
    auto Diag =
        diag(Var->getLocation(), "variable %0 can be declared 'constexpr'")
        << Var << R
        << FixItHint::CreateInsertion(
               Var->getTypeSpecStartLoc(),
               VariableReplacement(Var, FuncCtx, IsAddingConstexprToFuncCtx));
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
  }

  Functions.clear();
  VariableMapping.clear();
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
