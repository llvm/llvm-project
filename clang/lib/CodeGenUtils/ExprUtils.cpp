//===--- ExprUtils.cpp - Shared expression emission queries ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGenUtils/ExprUtils.h"
#include "clang/AST/Attr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtVisitor.h"

namespace clang::CodeGenUtils {

QualType getFixedSizeElementType(const ASTContext &Ctx,
                                 const VariableArrayType *VLA) {
  QualType EltType;
  do {
    EltType = VLA->getElementType();
  } while ((VLA = Ctx.getAsVariableArrayType(EltType)));
  return EltType;
}

bool isBlockVarRef(const Expr *E) {
  // Make sure we look through parens.
  E = E->IgnoreParens();

  // Check for a direct reference to a __block variable.
  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    const VarDecl *Var = dyn_cast<VarDecl>(DRE->getDecl());
    return (Var && Var->hasAttr<BlocksAttr>());
  }

  // More complicated stuff.

  // Binary operators.
  if (const BinaryOperator *Op = dyn_cast<BinaryOperator>(E)) {
    // For an assignment or pointer-to-member operation, just care
    // about the LHS.
    if (Op->isAssignmentOp() || Op->isPtrMemOp())
      return isBlockVarRef(Op->getLHS());

    // For a comma, just care about the RHS.
    if (Op->getOpcode() == BO_Comma)
      return isBlockVarRef(Op->getRHS());

    // FIXME: pointer arithmetic?
    return false;

    // Check both sides of a conditional operator.
  } else if (const AbstractConditionalOperator *Op =
                 dyn_cast<AbstractConditionalOperator>(E)) {
    return isBlockVarRef(Op->getTrueExpr()) ||
           isBlockVarRef(Op->getFalseExpr());

    // OVEs are required to support BinaryConditionalOperators.
  } else if (const OpaqueValueExpr *Op = dyn_cast<OpaqueValueExpr>(E)) {
    if (const Expr *Src = Op->getSourceExpr())
      return isBlockVarRef(Src);

    // Casts are necessary to get things like (*(int*)&var) = foo().
    // We don't really care about the kind of cast here, except
    // we don't want to look through l2r casts, because it's okay
    // to get the *value* in a __block variable.
  } else if (const CastExpr *Cast = dyn_cast<CastExpr>(E)) {
    if (Cast->getCastKind() == CK_LValueToRValue)
      return false;
    return isBlockVarRef(Cast->getSubExpr());

    // Handle unary operators.  Again, just aggressively look through
    // it, ignoring the operation.
  } else if (const UnaryOperator *UOp = dyn_cast<UnaryOperator>(E)) {
    return isBlockVarRef(UOp->getSubExpr());

    // Look into the base of a field access.
  } else if (const MemberExpr *Mem = dyn_cast<MemberExpr>(E)) {
    return isBlockVarRef(Mem->getBase());

    // Look into the base of a subscript.
  } else if (const ArraySubscriptExpr *Sub = dyn_cast<ArraySubscriptExpr>(E)) {
    return isBlockVarRef(Sub->getBase());
  }

  return false;
}

bool isCheapEnoughToEvaluateUnconditionally(const Expr *E,
                                            const ASTContext &Ctx) {
  // Anything that is an integer or floating point constant is fine.
  return E->IgnoreParens()->isEvaluatable(Ctx);

  // Even non-volatile automatic variables can't be evaluated unconditionally.
  // Referencing a thread_local may cause non-trivial initialization work to
  // occur. If we're inside a lambda and one of the variables is from the scope
  // outside the lambda, that function may have returned already. Reading its
  // locals is a bad idea. Also, these reads may introduce races there didn't
  // exist in the source-level program.
}

bool isTrivialFiller(const Expr *E) {
  if (!E)
    return true;

  if (isa<ImplicitValueInitExpr>(E))
    return true;

  if (const auto *ILE = dyn_cast<InitListExpr>(E)) {
    if (ILE->getNumInits())
      return false;
    return isTrivialFiller(ILE->getArrayFiller());
  }

  if (const auto *Cons = dyn_cast_or_null<CXXConstructExpr>(E))
    return Cons->getConstructor()->isDefaultConstructor() &&
           Cons->getConstructor()->isTrivial();

  // FIXME: Are there other cases where we can avoid emitting an initializer?
  return false;
}

bool onlyHasInlineBuiltinDeclaration(const FunctionDecl *FD) {
  for (const FunctionDecl *PD = FD; PD; PD = PD->getPreviousDecl())
    if (!PD->isInlineBuiltinDeclaration())
      return false;
  return true;
}

/// Returns true if \p Field is reachable from \p RD either as a direct field or
/// through a chain of nested record fields (including anonymous
/// structs/unions). This mirrors the field path that the code generators walk
/// to address the count field, and is used to identify the right anchor
/// expression in Base.
static bool RecordContainsField(const RecordDecl *RD, const FieldDecl *Field) {
  for (const FieldDecl *FD : RD->fields()) {
    if (FD == Field)
      return true;
    QualType Ty = FD->getType();
    if (Ty->isRecordType())
      if (RecordContainsField(Ty->getAsRecordDecl(), Field))
        return true;
  }
  return false;
}

namespace {

/// \p StructAccessBase returns the base \p Expr of a field access. It returns
/// either a \p DeclRefExpr, representing the base pointer to the struct, i.e.:
///
///     p in p-> a.b.c
///
/// or a \p MemberExpr, if the \p MemberExpr has the \p RecordDecl we're
/// looking for:
///
///     struct s {
///       struct s *ptr;
///       int count;
///       char array[] __attribute__((counted_by(count)));
///     };
///
/// If we have an expression like \p p->ptr->array[index], we want the
/// \p MemberExpr for \p p->ptr instead of \p p.
class StructAccessBase
    : public ConstStmtVisitor<StructAccessBase, const Expr *> {
  /// The count field we're navigating to. We stop at the innermost expression
  /// whose struct type transitively contains this field, so that the code
  /// generators can navigate from that struct down to it.
  const FieldDecl *CountDecl;

  /// Returns true if E's record type (or pointee record type) transitively
  /// contains CountDecl. Handles both direct containment and nested structs,
  /// so we don't need a pre-computed RD from the caller.
  bool IsExpectedRecordDecl(const Expr *E) const {
    QualType Ty = E->getType();
    if (Ty->isPointerType())
      Ty = Ty->getPointeeType();
    const RecordDecl *RD = Ty->getAsRecordDecl();
    return RD && RecordContainsField(RD, CountDecl);
  }

public:
  StructAccessBase(const FieldDecl *CountDecl) : CountDecl(CountDecl) {}

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  // NOTE: If we build C++ support for counted_by, then we'll have to handle
  // horrors like this:
  //
  //     struct S {
  //       int x, y;
  //       int blah[] __attribute__((counted_by(x)));
  //     } s;
  //
  //     int foo(int index, int val) {
  //       int (S::*IHatePMDs)[] = &S::blah;
  //       (s.*IHatePMDs)[index] = val;
  //     }

  const Expr *Visit(const Expr *E) {
    return ConstStmtVisitor<StructAccessBase, const Expr *>::Visit(E);
  }

  const Expr *VisitStmt(const Stmt *S) { return nullptr; }

  // These are the types we expect to return (in order of most to least
  // likely):
  //
  //   1. DeclRefExpr - This is the expression for the base of the structure.
  //      It's exactly what we want to build an access to the \p counted_by
  //      field.
  //   2. MemberExpr - This is the expression that has the same \p RecordDecl
  //      as the flexble array member's lexical enclosing \p RecordDecl. This
  //      allows us to catch things like: "p->p->array"
  //   3. CompoundLiteralExpr - This is for people who create something
  //      heretical like (struct foo has a flexible array member):
  //
  //        (struct foo){ 1, 2 }.blah[idx];
  const Expr *VisitDeclRefExpr(const DeclRefExpr *E) {
    return IsExpectedRecordDecl(E) ? E : nullptr;
  }
  const Expr *VisitMemberExpr(const MemberExpr *E) {
    if (IsExpectedRecordDecl(E) && E->isArrow())
      return E;
    const Expr *Res = Visit(E->getBase());
    return !Res && IsExpectedRecordDecl(E) ? E : Res;
  }
  const Expr *VisitCompoundLiteralExpr(const CompoundLiteralExpr *E) {
    return IsExpectedRecordDecl(E) ? E : nullptr;
  }
  const Expr *VisitCallExpr(const CallExpr *E) {
    return IsExpectedRecordDecl(E) ? E : nullptr;
  }

  const Expr *VisitArraySubscriptExpr(const ArraySubscriptExpr *E) {
    if (IsExpectedRecordDecl(E))
      return E;
    return Visit(E->getBase());
  }
  const Expr *VisitCastExpr(const CastExpr *E) {
    if (E->getCastKind() == CK_LValueToRValue)
      return IsExpectedRecordDecl(E) ? E : nullptr;
    return Visit(E->getSubExpr());
  }
  const Expr *VisitParenExpr(const ParenExpr *E) {
    return Visit(E->getSubExpr());
  }
  const Expr *VisitUnaryAddrOf(const UnaryOperator *E) {
    return Visit(E->getSubExpr());
  }
  const Expr *VisitUnaryDeref(const UnaryOperator *E) {
    return Visit(E->getSubExpr());
  }
};

} // end anonymous namespace

const Expr *findCountedByFieldBase(const Expr *Base,
                                   const FieldDecl *CountDecl) {
  return StructAccessBase(CountDecl).Visit(Base);
}

} // namespace clang::CodeGenUtils
