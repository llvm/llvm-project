//===--- TypeUtils.cpp - Type helper functions ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TypeUtils.h"
#include "clang/AST/Expr.h"
#include "clang/AST/NSAPI.h"
#include "clang/AST/RecursiveASTVisitor.h"

using namespace clang;

namespace {

/// Returns false if a BOOL expression is found.
class BOOLUseFinder : public RecursiveASTVisitor<BOOLUseFinder> {
public:
  NSAPI API;

  BOOLUseFinder(const ASTContext &Context)
      : API(const_cast<ASTContext &>(Context)) {}

  bool VisitStmt(const Stmt *S) {
    if (const auto *E = dyn_cast<Expr>(S))
      return !API.isObjCBOOLType(E->getType());
    return true;
  }

  static bool hasUseOfObjCBOOL(const ASTContext &Ctx, const Expr *E) {
    return !BOOLUseFinder(Ctx).TraverseStmt(const_cast<Expr *>(E));
  }
};

} // end anonymous namespace

static QualType preferredBoolType(const Decl *FunctionLikeParentDecl,
                                  const Expr *E, QualType T,
                                  const PrintingPolicy &Policy,
                                  const ASTContext &Ctx) {
  // We want to target expressions that return either 'int' or 'bool'
  const auto *BTy = T->getAs<BuiltinType>();
  if (!BTy)
    return T;
  switch (BTy->getKind()) {
  case BuiltinType::Int:
  case BuiltinType::Bool:
    // In Objective-C[++] we want to try to use 'BOOL' when the 'BOOL' typedef
    // is defined.
    if (Ctx.getLangOpts().ObjC && Ctx.getBOOLDecl()) {
      if (Ctx.getLangOpts().CPlusPlus && FunctionLikeParentDecl) {
        // When extracting expression from a standalone function in
        // Objective-C++ we should use BOOL when expression uses BOOL, otherwise
        // we should use bool.
        if (isa<FunctionDecl>(FunctionLikeParentDecl)) {
          if (BOOLUseFinder::hasUseOfObjCBOOL(Ctx, E))
            return Ctx.getBOOLType();
          return T;
        }
      }
      return Ctx.getBOOLType();
    }
    // In C mode we want to use 'bool' instead of 'int' when the 'bool' macro
    // is defined.
    if (!Ctx.getLangOpts().CPlusPlus && Policy.Bool)
      return Ctx.BoolTy;
    break;
  default:
    break;
  }
  return T;
}

static bool isInStdNamespace(const Decl *D) {
  const DeclContext *DC = D->getDeclContext()->getEnclosingNamespaceContext();
  const NamespaceDecl *ND = dyn_cast<NamespaceDecl>(DC);
  if (!ND)
    return false;

  while (const DeclContext *Parent = ND->getParent()) {
    if (!isa<NamespaceDecl>(Parent))
      break;
    ND = cast<NamespaceDecl>(Parent);
  }

  return ND->isStdNamespace();
}

static QualType desugarStdTypedef(QualType T) {
  const auto *TT = T->getAs<TypedefType>();
  if (!TT)
    return QualType();
  const TypedefNameDecl *TND = TT->getDecl();
  if (!isInStdNamespace(TND))
    return QualType();
  return TT->desugar();
}

// Desugars a typedef of a typedef that are both defined in STL.
//
// This is used to find the right type for a c_str() call on a std::string
// object: we want to return const char *, not const value_type *.
static QualType desugarStdType(QualType T) {
  QualType DesugaredType = T;
  if (const auto *PT = T->getAs<PointerType>())
    DesugaredType = PT->getPointeeType();
  DesugaredType = desugarStdTypedef(DesugaredType);
  if (DesugaredType.isNull())
    return T;
  if (const auto *ET = DesugaredType->getAs<ElaboratedType>())
    DesugaredType = ET->desugar();
  DesugaredType = desugarStdTypedef(DesugaredType);
  if (DesugaredType.isNull())
    return T;
  return T.getCanonicalType();
}

// Given an operator call like std::string() + "", we would like to ensure
// that we return std::string instead of std::basic_string.
static QualType canonicalizeStdOperatorReturnType(const Expr *E, QualType T) {
  const auto *OCE = dyn_cast<CXXOperatorCallExpr>(E->IgnoreParenImpCasts());
  if (!OCE)
    return T;
  if (OCE->getNumArgs() < 2 || !isInStdNamespace(OCE->getCalleeDecl()))
    return T;
  QualType CanonicalReturn = T.getCanonicalType();
  if (const auto *RD = CanonicalReturn->getAsCXXRecordDecl()) {
    if (!isInStdNamespace(RD))
      return T;
  } else
    return T;
  for (unsigned I = 0, E = OCE->getNumArgs(); I < E; ++I) {
    const Expr *Arg = OCE->getArgs()[I];
    QualType T = Arg->getType();
    if (const auto *ET = dyn_cast<ElaboratedType>(T))
      T = ET->desugar();
    if (desugarStdTypedef(T).isNull())
      continue;
    QualType CanonicalArg = Arg->getType().getCanonicalType();
    CanonicalArg.removeLocalFastQualifiers();
    if (CanonicalArg == CanonicalReturn) {
      QualType Result = Arg->getType();
      Result.removeLocalFastQualifiers();
      return Result;
    }
  }
  return T;
}

namespace clang {
namespace tooling {

/// Tthe return type of the extracted function should match user's intent,
/// e.g. we want to use bool type whenever possible.
QualType findExpressionLexicalType(const Decl *FunctionLikeParentDecl,
                                   const Expr *E, QualType T,
                                   const PrintingPolicy &Policy,
                                   const ASTContext &Ctx) {
  // Get the correct property type.
  if (const auto *PRE = dyn_cast<ObjCPropertyRefExpr>(E)) {
    if (PRE->isMessagingGetter()) {
      if (PRE->isExplicitProperty()) {
        QualType ReceiverType = PRE->getReceiverType(Ctx);
        return PRE->getExplicitProperty()->getUsageType(ReceiverType);
      }
      if (const ObjCMethodDecl *M = PRE->getImplicitPropertyGetter()) {
        if (!PRE->isObjectReceiver())
          return M->getSendResultType(PRE->getReceiverType(Ctx));
        const Expr *Base = PRE->getBase();
        return M->getSendResultType(findExpressionLexicalType(
            FunctionLikeParentDecl, Base, Base->getType(), Policy, Ctx));
      }
    }
  }

  // Perform STL-specific type corrections.
  if (Ctx.getLangOpts().CPlusPlus) {
    T = desugarStdType(T);
    T = canonicalizeStdOperatorReturnType(E, T);
  }

  // The bool type adjustment is required only in C or Objective-C[++].
  if (Ctx.getLangOpts().CPlusPlus && !Ctx.getLangOpts().ObjC)
    return T;
  E = E->IgnoreParenImpCasts();
  if (const auto *BinOp = dyn_cast<BinaryOperator>(E)) {
    if (BinOp->isLogicalOp() || BinOp->isComparisonOp())
      return preferredBoolType(FunctionLikeParentDecl, E, T, Policy, Ctx);
  } else if (const auto *UnOp = dyn_cast<UnaryOperator>(E)) {
    if (UnOp->getOpcode() == UO_LNot)
      return preferredBoolType(FunctionLikeParentDecl, E, T, Policy, Ctx);
  }
  return T;
}

} // end namespace tooling
} // end namespace clang
