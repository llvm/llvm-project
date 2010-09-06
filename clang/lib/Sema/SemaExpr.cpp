//===--- SemaExpr.cpp - Semantic Analysis for Expressions -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for expressions.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/AnalysisBasedWarnings.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/LiteralSupport.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Designator.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/ParsedTemplate.h"
#include "clang/Sema/Template.h"
using namespace clang;
using namespace sema;


/// \brief Determine whether the use of this declaration is valid, and
/// emit any corresponding diagnostics.
///
/// This routine diagnoses various problems with referencing
/// declarations that can occur when using a declaration. For example,
/// it might warn if a deprecated or unavailable declaration is being
/// used, or produce an error (and return true) if a C++0x deleted
/// function is being used.
///
/// If IgnoreDeprecated is set to true, this should not want about deprecated
/// decls.
///
/// \returns true if there was an error (this declaration cannot be
/// referenced), false otherwise.
///
bool Sema::DiagnoseUseOfDecl(NamedDecl *D, SourceLocation Loc) {
  // See if the decl is deprecated.
  if (D->getAttr<DeprecatedAttr>()) {
    EmitDeprecationWarning(D, Loc);
  }

  // See if the decl is unavailable
  if (D->getAttr<UnavailableAttr>()) {
    Diag(Loc, diag::err_unavailable) << D->getDeclName();
    Diag(D->getLocation(), diag::note_unavailable_here) << 0;
  }

  // See if this is a deleted function.
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    if (FD->isDeleted()) {
      Diag(Loc, diag::err_deleted_function_use);
      Diag(D->getLocation(), diag::note_unavailable_here) << true;
      return true;
    }
  }

  return false;
}

/// DiagnoseSentinelCalls - This routine checks on method dispatch calls
/// (and other functions in future), which have been declared with sentinel
/// attribute. It warns if call does not have the sentinel argument.
///
void Sema::DiagnoseSentinelCalls(NamedDecl *D, SourceLocation Loc,
                                 Expr **Args, unsigned NumArgs) {
  const SentinelAttr *attr = D->getAttr<SentinelAttr>();
  if (!attr)
    return;

  // FIXME: In C++0x, if any of the arguments are parameter pack
  // expansions, we can't check for the sentinel now.
  int sentinelPos = attr->getSentinel();
  int nullPos = attr->getNullPos();

  // FIXME. ObjCMethodDecl and FunctionDecl need be derived from the same common
  // base class. Then we won't be needing two versions of the same code.
  unsigned int i = 0;
  bool warnNotEnoughArgs = false;
  int isMethod = 0;
  if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D)) {
    // skip over named parameters.
    ObjCMethodDecl::param_iterator P, E = MD->param_end();
    for (P = MD->param_begin(); (P != E && i < NumArgs); ++P) {
      if (nullPos)
        --nullPos;
      else
        ++i;
    }
    warnNotEnoughArgs = (P != E || i >= NumArgs);
    isMethod = 1;
  } else if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    // skip over named parameters.
    ObjCMethodDecl::param_iterator P, E = FD->param_end();
    for (P = FD->param_begin(); (P != E && i < NumArgs); ++P) {
      if (nullPos)
        --nullPos;
      else
        ++i;
    }
    warnNotEnoughArgs = (P != E || i >= NumArgs);
  } else if (VarDecl *V = dyn_cast<VarDecl>(D)) {
    // block or function pointer call.
    QualType Ty = V->getType();
    if (Ty->isBlockPointerType() || Ty->isFunctionPointerType()) {
      const FunctionType *FT = Ty->isFunctionPointerType()
      ? Ty->getAs<PointerType>()->getPointeeType()->getAs<FunctionType>()
      : Ty->getAs<BlockPointerType>()->getPointeeType()->getAs<FunctionType>();
      if (const FunctionProtoType *Proto = dyn_cast<FunctionProtoType>(FT)) {
        unsigned NumArgsInProto = Proto->getNumArgs();
        unsigned k;
        for (k = 0; (k != NumArgsInProto && i < NumArgs); k++) {
          if (nullPos)
            --nullPos;
          else
            ++i;
        }
        warnNotEnoughArgs = (k != NumArgsInProto || i >= NumArgs);
      }
      if (Ty->isBlockPointerType())
        isMethod = 2;
    } else
      return;
  } else
    return;

  if (warnNotEnoughArgs) {
    Diag(Loc, diag::warn_not_enough_argument) << D->getDeclName();
    Diag(D->getLocation(), diag::note_sentinel_here) << isMethod;
    return;
  }
  int sentinel = i;
  while (sentinelPos > 0 && i < NumArgs-1) {
    --sentinelPos;
    ++i;
  }
  if (sentinelPos > 0) {
    Diag(Loc, diag::warn_not_enough_argument) << D->getDeclName();
    Diag(D->getLocation(), diag::note_sentinel_here) << isMethod;
    return;
  }
  while (i < NumArgs-1) {
    ++i;
    ++sentinel;
  }
  Expr *sentinelExpr = Args[sentinel];
  if (!sentinelExpr) return;
  if (sentinelExpr->isTypeDependent()) return;
  if (sentinelExpr->isValueDependent()) return;
  if (sentinelExpr->getType()->isAnyPointerType() &&
      sentinelExpr->IgnoreParenCasts()->isNullPointerConstant(Context,
                                            Expr::NPC_ValueDependentIsNull))
    return;

  // Unfortunately, __null has type 'int'.
  if (isa<GNUNullExpr>(sentinelExpr)) return;

  Diag(Loc, diag::warn_missing_sentinel) << isMethod;
  Diag(D->getLocation(), diag::note_sentinel_here) << isMethod;
}

SourceRange Sema::getExprRange(ExprTy *E) const {
  Expr *Ex = (Expr *)E;
  return Ex? Ex->getSourceRange() : SourceRange();
}

//===----------------------------------------------------------------------===//
//  Standard Promotions and Conversions
//===----------------------------------------------------------------------===//

/// DefaultFunctionArrayConversion (C99 6.3.2.1p3, C99 6.3.2.1p4).
void Sema::DefaultFunctionArrayConversion(Expr *&E) {
  QualType Ty = E->getType();
  assert(!Ty.isNull() && "DefaultFunctionArrayConversion - missing type");

  if (Ty->isFunctionType())
    ImpCastExprToType(E, Context.getPointerType(Ty),
                      CK_FunctionToPointerDecay);
  else if (Ty->isArrayType()) {
    // In C90 mode, arrays only promote to pointers if the array expression is
    // an lvalue.  The relevant legalese is C90 6.2.2.1p3: "an lvalue that has
    // type 'array of type' is converted to an expression that has type 'pointer
    // to type'...".  In C99 this was changed to: C99 6.3.2.1p3: "an expression
    // that has type 'array of type' ...".  The relevant change is "an lvalue"
    // (C90) to "an expression" (C99).
    //
    // C++ 4.2p1:
    // An lvalue or rvalue of type "array of N T" or "array of unknown bound of
    // T" can be converted to an rvalue of type "pointer to T".
    //
    if (getLangOptions().C99 || getLangOptions().CPlusPlus ||
        E->isLvalue(Context) == Expr::LV_Valid)
      ImpCastExprToType(E, Context.getArrayDecayedType(Ty),
                        CK_ArrayToPointerDecay);
  }
}

void Sema::DefaultFunctionArrayLvalueConversion(Expr *&E) {
  DefaultFunctionArrayConversion(E);

  QualType Ty = E->getType();
  assert(!Ty.isNull() && "DefaultFunctionArrayLvalueConversion - missing type");
  if (!Ty->isDependentType() && Ty.hasQualifiers() &&
      (!getLangOptions().CPlusPlus || !Ty->isRecordType()) &&
      E->isLvalue(Context) == Expr::LV_Valid) {
    // C++ [conv.lval]p1:
    //   [...] If T is a non-class type, the type of the rvalue is the
    //   cv-unqualified version of T. Otherwise, the type of the
    //   rvalue is T
    //
    // C99 6.3.2.1p2:
    //   If the lvalue has qualified type, the value has the unqualified
    //   version of the type of the lvalue; otherwise, the value has the
    //   type of the lvalue.
    ImpCastExprToType(E, Ty.getUnqualifiedType(), CK_NoOp);
  }
}


/// UsualUnaryConversions - Performs various conversions that are common to most
/// operators (C99 6.3). The conversions of array and function types are
/// sometimes surpressed. For example, the array->pointer conversion doesn't
/// apply if the array is an argument to the sizeof or address (&) operators.
/// In these instances, this routine should *not* be called.
Expr *Sema::UsualUnaryConversions(Expr *&Expr) {
  QualType Ty = Expr->getType();
  assert(!Ty.isNull() && "UsualUnaryConversions - missing type");

  // C99 6.3.1.1p2:
  //
  //   The following may be used in an expression wherever an int or
  //   unsigned int may be used:
  //     - an object or expression with an integer type whose integer
  //       conversion rank is less than or equal to the rank of int
  //       and unsigned int.
  //     - A bit-field of type _Bool, int, signed int, or unsigned int.
  //
  //   If an int can represent all values of the original type, the
  //   value is converted to an int; otherwise, it is converted to an
  //   unsigned int. These are called the integer promotions. All
  //   other types are unchanged by the integer promotions.
  QualType PTy = Context.isPromotableBitField(Expr);
  if (!PTy.isNull()) {
    ImpCastExprToType(Expr, PTy, CK_IntegralCast);
    return Expr;
  }
  if (Ty->isPromotableIntegerType()) {
    QualType PT = Context.getPromotedIntegerType(Ty);
    ImpCastExprToType(Expr, PT, CK_IntegralCast);
    return Expr;
  }

  DefaultFunctionArrayLvalueConversion(Expr);
  return Expr;
}

/// DefaultArgumentPromotion (C99 6.5.2.2p6). Used for function calls that
/// do not have a prototype. Arguments that have type float are promoted to
/// double. All other argument types are converted by UsualUnaryConversions().
void Sema::DefaultArgumentPromotion(Expr *&Expr) {
  QualType Ty = Expr->getType();
  assert(!Ty.isNull() && "DefaultArgumentPromotion - missing type");

  // If this is a 'float' (CVR qualified or typedef) promote to double.
  if (Ty->isSpecificBuiltinType(BuiltinType::Float))
    return ImpCastExprToType(Expr, Context.DoubleTy,
                             CK_FloatingCast);

  UsualUnaryConversions(Expr);
}

/// DefaultVariadicArgumentPromotion - Like DefaultArgumentPromotion, but
/// will warn if the resulting type is not a POD type, and rejects ObjC
/// interfaces passed by value.  This returns true if the argument type is
/// completely illegal.
bool Sema::DefaultVariadicArgumentPromotion(Expr *&Expr, VariadicCallType CT,
                                            FunctionDecl *FDecl) {
  DefaultArgumentPromotion(Expr);

  // __builtin_va_start takes the second argument as a "varargs" argument, but
  // it doesn't actually do anything with it.  It doesn't need to be non-pod
  // etc.
  if (FDecl && FDecl->getBuiltinID() == Builtin::BI__builtin_va_start)
    return false;
  
  if (Expr->getType()->isObjCObjectType() &&
      DiagRuntimeBehavior(Expr->getLocStart(),
        PDiag(diag::err_cannot_pass_objc_interface_to_vararg)
          << Expr->getType() << CT))
    return true;

  if (!Expr->getType()->isPODType() &&
      DiagRuntimeBehavior(Expr->getLocStart(),
                          PDiag(diag::warn_cannot_pass_non_pod_arg_to_vararg)
                            << Expr->getType() << CT))
    return true;

  return false;
}


/// UsualArithmeticConversions - Performs various conversions that are common to
/// binary operators (C99 6.3.1.8). If both operands aren't arithmetic, this
/// routine returns the first non-arithmetic type found. The client is
/// responsible for emitting appropriate error diagnostics.
/// FIXME: verify the conversion rules for "complex int" are consistent with
/// GCC.
QualType Sema::UsualArithmeticConversions(Expr *&lhsExpr, Expr *&rhsExpr,
                                          bool isCompAssign) {
  if (!isCompAssign)
    UsualUnaryConversions(lhsExpr);

  UsualUnaryConversions(rhsExpr);

  // For conversion purposes, we ignore any qualifiers.
  // For example, "const float" and "float" are equivalent.
  QualType lhs =
    Context.getCanonicalType(lhsExpr->getType()).getUnqualifiedType();
  QualType rhs =
    Context.getCanonicalType(rhsExpr->getType()).getUnqualifiedType();

  // If both types are identical, no conversion is needed.
  if (lhs == rhs)
    return lhs;

  // If either side is a non-arithmetic type (e.g. a pointer), we are done.
  // The caller can deal with this (e.g. pointer + int).
  if (!lhs->isArithmeticType() || !rhs->isArithmeticType())
    return lhs;

  // Perform bitfield promotions.
  QualType LHSBitfieldPromoteTy = Context.isPromotableBitField(lhsExpr);
  if (!LHSBitfieldPromoteTy.isNull())
    lhs = LHSBitfieldPromoteTy;
  QualType RHSBitfieldPromoteTy = Context.isPromotableBitField(rhsExpr);
  if (!RHSBitfieldPromoteTy.isNull())
    rhs = RHSBitfieldPromoteTy;

  QualType destType = Context.UsualArithmeticConversionsType(lhs, rhs);
  if (!isCompAssign)
    ImpCastExprToType(lhsExpr, destType, CK_Unknown);
  ImpCastExprToType(rhsExpr, destType, CK_Unknown);
  return destType;
}

//===----------------------------------------------------------------------===//
//  Semantic Analysis for various Expression Types
//===----------------------------------------------------------------------===//


/// ActOnStringLiteral - The specified tokens were lexed as pasted string
/// fragments (e.g. "foo" "bar" L"baz").  The result string has to handle string
/// concatenation ([C99 5.1.1.2, translation phase #6]), so it may come from
/// multiple tokens.  However, the common case is that StringToks points to one
/// string.
///
ExprResult
Sema::ActOnStringLiteral(const Token *StringToks, unsigned NumStringToks) {
  assert(NumStringToks && "Must have at least one string!");

  StringLiteralParser Literal(StringToks, NumStringToks, PP);
  if (Literal.hadError)
    return ExprError();

  llvm::SmallVector<SourceLocation, 4> StringTokLocs;
  for (unsigned i = 0; i != NumStringToks; ++i)
    StringTokLocs.push_back(StringToks[i].getLocation());

  QualType StrTy = Context.CharTy;
  if (Literal.AnyWide) StrTy = Context.getWCharType();
  if (Literal.Pascal) StrTy = Context.UnsignedCharTy;

  // A C++ string literal has a const-qualified element type (C++ 2.13.4p1).
  if (getLangOptions().CPlusPlus || getLangOptions().ConstStrings)
    StrTy.addConst();

  // Get an array type for the string, according to C99 6.4.5.  This includes
  // the nul terminator character as well as the string length for pascal
  // strings.
  StrTy = Context.getConstantArrayType(StrTy,
                                 llvm::APInt(32, Literal.GetNumStringChars()+1),
                                       ArrayType::Normal, 0);

  // Pass &StringTokLocs[0], StringTokLocs.size() to factory!
  return Owned(StringLiteral::Create(Context, Literal.GetString(),
                                     Literal.GetStringLength(),
                                     Literal.AnyWide, StrTy,
                                     &StringTokLocs[0],
                                     StringTokLocs.size()));
}

/// ShouldSnapshotBlockValueReference - Return true if a reference inside of
/// CurBlock to VD should cause it to be snapshotted (as we do for auto
/// variables defined outside the block) or false if this is not needed (e.g.
/// for values inside the block or for globals).
///
/// This also keeps the 'hasBlockDeclRefExprs' in the BlockScopeInfo records
/// up-to-date.
///
static bool ShouldSnapshotBlockValueReference(Sema &S, BlockScopeInfo *CurBlock,
                                              ValueDecl *VD) {
  // If the value is defined inside the block, we couldn't snapshot it even if
  // we wanted to.
  if (CurBlock->TheDecl == VD->getDeclContext())
    return false;

  // If this is an enum constant or function, it is constant, don't snapshot.
  if (isa<EnumConstantDecl>(VD) || isa<FunctionDecl>(VD))
    return false;

  // If this is a reference to an extern, static, or global variable, no need to
  // snapshot it.
  // FIXME: What about 'const' variables in C++?
  if (const VarDecl *Var = dyn_cast<VarDecl>(VD))
    if (!Var->hasLocalStorage())
      return false;

  // Blocks that have these can't be constant.
  CurBlock->hasBlockDeclRefExprs = true;

  // If we have nested blocks, the decl may be declared in an outer block (in
  // which case that outer block doesn't get "hasBlockDeclRefExprs") or it may
  // be defined outside all of the current blocks (in which case the blocks do
  // all get the bit).  Walk the nesting chain.
  for (unsigned I = S.FunctionScopes.size() - 1; I; --I) {
    BlockScopeInfo *NextBlock = dyn_cast<BlockScopeInfo>(S.FunctionScopes[I]);

    if (!NextBlock)
      continue;

    // If we found the defining block for the variable, don't mark the block as
    // having a reference outside it.
    if (NextBlock->TheDecl == VD->getDeclContext())
      break;

    // Otherwise, the DeclRef from the inner block causes the outer one to need
    // a snapshot as well.
    NextBlock->hasBlockDeclRefExprs = true;
  }

  return true;
}


ExprResult
Sema::BuildDeclRefExpr(ValueDecl *D, QualType Ty, SourceLocation Loc,
                       const CXXScopeSpec *SS) {
  DeclarationNameInfo NameInfo(D->getDeclName(), Loc);
  return BuildDeclRefExpr(D, Ty, NameInfo, SS);
}

/// BuildDeclRefExpr - Build a DeclRefExpr.
ExprResult
Sema::BuildDeclRefExpr(ValueDecl *D, QualType Ty,
                       const DeclarationNameInfo &NameInfo,
                       const CXXScopeSpec *SS) {
  if (Context.getCanonicalType(Ty) == Context.UndeducedAutoTy) {
    Diag(NameInfo.getLoc(),
         diag::err_auto_variable_cannot_appear_in_own_initializer)
      << D->getDeclName();
    return ExprError();
  }

  if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
    if (isa<NonTypeTemplateParmDecl>(VD)) {
      // Non-type template parameters can be referenced anywhere they are
      // visible.
      Ty = Ty.getNonLValueExprType(Context);
    } else if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(CurContext)) {
      if (const FunctionDecl *FD = MD->getParent()->isLocalClass()) {
        if (VD->hasLocalStorage() && VD->getDeclContext() != CurContext) {
          Diag(NameInfo.getLoc(),
               diag::err_reference_to_local_var_in_enclosing_function)
            << D->getIdentifier() << FD->getDeclName();
          Diag(D->getLocation(), diag::note_local_variable_declared_here)
            << D->getIdentifier();
          return ExprError();
        }
      }
    }
  }

  MarkDeclarationReferenced(NameInfo.getLoc(), D);

  return Owned(DeclRefExpr::Create(Context,
                              SS? (NestedNameSpecifier *)SS->getScopeRep() : 0,
                                   SS? SS->getRange() : SourceRange(),
                                   D, NameInfo, Ty));
}

/// \brief Given a field that represents a member of an anonymous
/// struct/union, build the path from that field's context to the
/// actual member.
///
/// Construct the sequence of field member references we'll have to
/// perform to get to the field in the anonymous union/struct. The
/// list of members is built from the field outward, so traverse it
/// backwards to go from an object in the current context to the field
/// we found.
///
/// \returns The variable from which the field access should begin,
/// for an anonymous struct/union that is not a member of another
/// class. Otherwise, returns NULL.
VarDecl *Sema::BuildAnonymousStructUnionMemberPath(FieldDecl *Field,
                                   llvm::SmallVectorImpl<FieldDecl *> &Path) {
  assert(Field->getDeclContext()->isRecord() &&
         cast<RecordDecl>(Field->getDeclContext())->isAnonymousStructOrUnion()
         && "Field must be stored inside an anonymous struct or union");

  Path.push_back(Field);
  VarDecl *BaseObject = 0;
  DeclContext *Ctx = Field->getDeclContext();
  do {
    RecordDecl *Record = cast<RecordDecl>(Ctx);
    ValueDecl *AnonObject = Record->getAnonymousStructOrUnionObject();
    if (FieldDecl *AnonField = dyn_cast<FieldDecl>(AnonObject))
      Path.push_back(AnonField);
    else {
      BaseObject = cast<VarDecl>(AnonObject);
      break;
    }
    Ctx = Ctx->getParent();
  } while (Ctx->isRecord() &&
           cast<RecordDecl>(Ctx)->isAnonymousStructOrUnion());

  return BaseObject;
}

ExprResult
Sema::BuildAnonymousStructUnionMemberReference(SourceLocation Loc,
                                               FieldDecl *Field,
                                               Expr *BaseObjectExpr,
                                               SourceLocation OpLoc) {
  llvm::SmallVector<FieldDecl *, 4> AnonFields;
  VarDecl *BaseObject = BuildAnonymousStructUnionMemberPath(Field,
                                                            AnonFields);

  // Build the expression that refers to the base object, from
  // which we will build a sequence of member references to each
  // of the anonymous union objects and, eventually, the field we
  // found via name lookup.
  bool BaseObjectIsPointer = false;
  Qualifiers BaseQuals;
  if (BaseObject) {
    // BaseObject is an anonymous struct/union variable (and is,
    // therefore, not part of another non-anonymous record).
    MarkDeclarationReferenced(Loc, BaseObject);
    BaseObjectExpr = new (Context) DeclRefExpr(BaseObject,BaseObject->getType(),
                                               SourceLocation());
    BaseQuals
      = Context.getCanonicalType(BaseObject->getType()).getQualifiers();
  } else if (BaseObjectExpr) {
    // The caller provided the base object expression. Determine
    // whether its a pointer and whether it adds any qualifiers to the
    // anonymous struct/union fields we're looking into.
    QualType ObjectType = BaseObjectExpr->getType();
    if (const PointerType *ObjectPtr = ObjectType->getAs<PointerType>()) {
      BaseObjectIsPointer = true;
      ObjectType = ObjectPtr->getPointeeType();
    }
    BaseQuals
      = Context.getCanonicalType(ObjectType).getQualifiers();
  } else {
    // We've found a member of an anonymous struct/union that is
    // inside a non-anonymous struct/union, so in a well-formed
    // program our base object expression is "this".
    DeclContext *DC = getFunctionLevelDeclContext();
    if (CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(DC)) {
      if (!MD->isStatic()) {
        QualType AnonFieldType
          = Context.getTagDeclType(
                     cast<RecordDecl>(AnonFields.back()->getDeclContext()));
        QualType ThisType = Context.getTagDeclType(MD->getParent());
        if ((Context.getCanonicalType(AnonFieldType)
               == Context.getCanonicalType(ThisType)) ||
            IsDerivedFrom(ThisType, AnonFieldType)) {
          // Our base object expression is "this".
          BaseObjectExpr = new (Context) CXXThisExpr(Loc,
                                                     MD->getThisType(Context),
                                                     /*isImplicit=*/true);
          BaseObjectIsPointer = true;
        }
      } else {
        return ExprError(Diag(Loc,diag::err_invalid_member_use_in_static_method)
          << Field->getDeclName());
      }
      BaseQuals = Qualifiers::fromCVRMask(MD->getTypeQualifiers());
    }

    if (!BaseObjectExpr)
      return ExprError(Diag(Loc, diag::err_invalid_non_static_member_use)
        << Field->getDeclName());
  }

  // Build the implicit member references to the field of the
  // anonymous struct/union.
  Expr *Result = BaseObjectExpr;
  Qualifiers ResultQuals = BaseQuals;
  for (llvm::SmallVector<FieldDecl *, 4>::reverse_iterator
         FI = AnonFields.rbegin(), FIEnd = AnonFields.rend();
       FI != FIEnd; ++FI) {
    QualType MemberType = (*FI)->getType();
    Qualifiers MemberTypeQuals =
      Context.getCanonicalType(MemberType).getQualifiers();

    // CVR attributes from the base are picked up by members,
    // except that 'mutable' members don't pick up 'const'.
    if ((*FI)->isMutable())
      ResultQuals.removeConst();

    // GC attributes are never picked up by members.
    ResultQuals.removeObjCGCAttr();

    // TR 18037 does not allow fields to be declared with address spaces.
    assert(!MemberTypeQuals.hasAddressSpace());

    Qualifiers NewQuals = ResultQuals + MemberTypeQuals;
    if (NewQuals != MemberTypeQuals)
      MemberType = Context.getQualifiedType(MemberType, NewQuals);

    MarkDeclarationReferenced(Loc, *FI);
    PerformObjectMemberConversion(Result, /*FIXME:Qualifier=*/0, *FI, *FI);
    // FIXME: Might this end up being a qualified name?
    Result = new (Context) MemberExpr(Result, BaseObjectIsPointer, *FI,
                                      OpLoc, MemberType);
    BaseObjectIsPointer = false;
    ResultQuals = NewQuals;
  }

  return Owned(Result);
}

/// Decomposes the given name into a DeclarationNameInfo, its location, and
/// possibly a list of template arguments.
///
/// If this produces template arguments, it is permitted to call
/// DecomposeTemplateName.
///
/// This actually loses a lot of source location information for
/// non-standard name kinds; we should consider preserving that in
/// some way.
static void DecomposeUnqualifiedId(Sema &SemaRef,
                                   const UnqualifiedId &Id,
                                   TemplateArgumentListInfo &Buffer,
                                   DeclarationNameInfo &NameInfo,
                             const TemplateArgumentListInfo *&TemplateArgs) {
  if (Id.getKind() == UnqualifiedId::IK_TemplateId) {
    Buffer.setLAngleLoc(Id.TemplateId->LAngleLoc);
    Buffer.setRAngleLoc(Id.TemplateId->RAngleLoc);

    ASTTemplateArgsPtr TemplateArgsPtr(SemaRef,
                                       Id.TemplateId->getTemplateArgs(),
                                       Id.TemplateId->NumArgs);
    SemaRef.translateTemplateArguments(TemplateArgsPtr, Buffer);
    TemplateArgsPtr.release();

    TemplateName TName = Id.TemplateId->Template.get();
    SourceLocation TNameLoc = Id.TemplateId->TemplateNameLoc;
    NameInfo = SemaRef.Context.getNameForTemplate(TName, TNameLoc);
    TemplateArgs = &Buffer;
  } else {
    NameInfo = SemaRef.GetNameFromUnqualifiedId(Id);
    TemplateArgs = 0;
  }
}

/// Determines whether the given record is "fully-formed" at the given
/// location, i.e. whether a qualified lookup into it is assured of
/// getting consistent results already.
static bool IsFullyFormedScope(Sema &SemaRef, CXXRecordDecl *Record) {
  if (!Record->hasDefinition())
    return false;

  for (CXXRecordDecl::base_class_iterator I = Record->bases_begin(),
         E = Record->bases_end(); I != E; ++I) {
    CanQualType BaseT = SemaRef.Context.getCanonicalType((*I).getType());
    CanQual<RecordType> BaseRT = BaseT->getAs<RecordType>();
    if (!BaseRT) return false;

    CXXRecordDecl *BaseRecord = cast<CXXRecordDecl>(BaseRT->getDecl());
    if (!BaseRecord->hasDefinition() ||
        !IsFullyFormedScope(SemaRef, BaseRecord))
      return false;
  }

  return true;
}

/// Determines if the given class is provably not derived from all of
/// the prospective base classes.
static bool IsProvablyNotDerivedFrom(Sema &SemaRef,
                                     CXXRecordDecl *Record,
                            const llvm::SmallPtrSet<CXXRecordDecl*, 4> &Bases) {
  if (Bases.count(Record->getCanonicalDecl()))
    return false;

  RecordDecl *RD = Record->getDefinition();
  if (!RD) return false;
  Record = cast<CXXRecordDecl>(RD);

  for (CXXRecordDecl::base_class_iterator I = Record->bases_begin(),
         E = Record->bases_end(); I != E; ++I) {
    CanQualType BaseT = SemaRef.Context.getCanonicalType((*I).getType());
    CanQual<RecordType> BaseRT = BaseT->getAs<RecordType>();
    if (!BaseRT) return false;

    CXXRecordDecl *BaseRecord = cast<CXXRecordDecl>(BaseRT->getDecl());
    if (!IsProvablyNotDerivedFrom(SemaRef, BaseRecord, Bases))
      return false;
  }

  return true;
}

enum IMAKind {
  /// The reference is definitely not an instance member access.
  IMA_Static,

  /// The reference may be an implicit instance member access.
  IMA_Mixed,

  /// The reference may be to an instance member, but it is invalid if
  /// so, because the context is not an instance method.
  IMA_Mixed_StaticContext,

  /// The reference may be to an instance member, but it is invalid if
  /// so, because the context is from an unrelated class.
  IMA_Mixed_Unrelated,

  /// The reference is definitely an implicit instance member access.
  IMA_Instance,

  /// The reference may be to an unresolved using declaration.
  IMA_Unresolved,

  /// The reference may be to an unresolved using declaration and the
  /// context is not an instance method.
  IMA_Unresolved_StaticContext,

  /// The reference is to a member of an anonymous structure in a
  /// non-class context.
  IMA_AnonymousMember,

  /// All possible referrents are instance members and the current
  /// context is not an instance method.
  IMA_Error_StaticContext,

  /// All possible referrents are instance members of an unrelated
  /// class.
  IMA_Error_Unrelated
};

/// The given lookup names class member(s) and is not being used for
/// an address-of-member expression.  Classify the type of access
/// according to whether it's possible that this reference names an
/// instance member.  This is best-effort; it is okay to
/// conservatively answer "yes", in which case some errors will simply
/// not be caught until template-instantiation.
static IMAKind ClassifyImplicitMemberAccess(Sema &SemaRef,
                                            const LookupResult &R) {
  assert(!R.empty() && (*R.begin())->isCXXClassMember());

  DeclContext *DC = SemaRef.getFunctionLevelDeclContext();
  bool isStaticContext =
    (!isa<CXXMethodDecl>(DC) ||
     cast<CXXMethodDecl>(DC)->isStatic());

  if (R.isUnresolvableResult())
    return isStaticContext ? IMA_Unresolved_StaticContext : IMA_Unresolved;

  // Collect all the declaring classes of instance members we find.
  bool hasNonInstance = false;
  llvm::SmallPtrSet<CXXRecordDecl*, 4> Classes;
  for (LookupResult::iterator I = R.begin(), E = R.end(); I != E; ++I) {
    NamedDecl *D = *I;
    if (D->isCXXInstanceMember()) {
      CXXRecordDecl *R = cast<CXXRecordDecl>(D->getDeclContext());

      // If this is a member of an anonymous record, move out to the
      // innermost non-anonymous struct or union.  If there isn't one,
      // that's a special case.
      while (R->isAnonymousStructOrUnion()) {
        R = dyn_cast<CXXRecordDecl>(R->getParent());
        if (!R) return IMA_AnonymousMember;
      }
      Classes.insert(R->getCanonicalDecl());
    }
    else
      hasNonInstance = true;
  }

  // If we didn't find any instance members, it can't be an implicit
  // member reference.
  if (Classes.empty())
    return IMA_Static;

  // If the current context is not an instance method, it can't be
  // an implicit member reference.
  if (isStaticContext)
    return (hasNonInstance ? IMA_Mixed_StaticContext : IMA_Error_StaticContext);

  // If we can prove that the current context is unrelated to all the
  // declaring classes, it can't be an implicit member reference (in
  // which case it's an error if any of those members are selected).
  if (IsProvablyNotDerivedFrom(SemaRef,
                               cast<CXXMethodDecl>(DC)->getParent(),
                               Classes))
    return (hasNonInstance ? IMA_Mixed_Unrelated : IMA_Error_Unrelated);

  return (hasNonInstance ? IMA_Mixed : IMA_Instance);
}

/// Diagnose a reference to a field with no object available.
static void DiagnoseInstanceReference(Sema &SemaRef,
                                      const CXXScopeSpec &SS,
                                      const LookupResult &R) {
  SourceLocation Loc = R.getNameLoc();
  SourceRange Range(Loc);
  if (SS.isSet()) Range.setBegin(SS.getRange().getBegin());

  if (R.getAsSingle<FieldDecl>()) {
    if (CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(SemaRef.CurContext)) {
      if (MD->isStatic()) {
        // "invalid use of member 'x' in static member function"
        SemaRef.Diag(Loc, diag::err_invalid_member_use_in_static_method)
          << Range << R.getLookupName();
        return;
      }
    }

    SemaRef.Diag(Loc, diag::err_invalid_non_static_member_use)
      << R.getLookupName() << Range;
    return;
  }

  SemaRef.Diag(Loc, diag::err_member_call_without_object) << Range;
}

/// Diagnose an empty lookup.
///
/// \return false if new lookup candidates were found
bool Sema::DiagnoseEmptyLookup(Scope *S, CXXScopeSpec &SS, LookupResult &R,
                               CorrectTypoContext CTC) {
  DeclarationName Name = R.getLookupName();

  unsigned diagnostic = diag::err_undeclared_var_use;
  unsigned diagnostic_suggest = diag::err_undeclared_var_use_suggest;
  if (Name.getNameKind() == DeclarationName::CXXOperatorName ||
      Name.getNameKind() == DeclarationName::CXXLiteralOperatorName ||
      Name.getNameKind() == DeclarationName::CXXConversionFunctionName) {
    diagnostic = diag::err_undeclared_use;
    diagnostic_suggest = diag::err_undeclared_use_suggest;
  }

  // If the original lookup was an unqualified lookup, fake an
  // unqualified lookup.  This is useful when (for example) the
  // original lookup would not have found something because it was a
  // dependent name.
  for (DeclContext *DC = SS.isEmpty() ? CurContext : 0;
       DC; DC = DC->getParent()) {
    if (isa<CXXRecordDecl>(DC)) {
      LookupQualifiedName(R, DC);

      if (!R.empty()) {
        // Don't give errors about ambiguities in this lookup.
        R.suppressDiagnostics();

        CXXMethodDecl *CurMethod = dyn_cast<CXXMethodDecl>(CurContext);
        bool isInstance = CurMethod &&
                          CurMethod->isInstance() &&
                          DC == CurMethod->getParent();

        // Give a code modification hint to insert 'this->'.
        // TODO: fixit for inserting 'Base<T>::' in the other cases.
        // Actually quite difficult!
        if (isInstance) {
          UnresolvedLookupExpr *ULE = cast<UnresolvedLookupExpr>(
              CallsUndergoingInstantiation.back()->getCallee());
          CXXMethodDecl *DepMethod = cast_or_null<CXXMethodDecl>(
              CurMethod->getInstantiatedFromMemberFunction());
          if (DepMethod) {
            Diag(R.getNameLoc(), diagnostic) << Name
              << FixItHint::CreateInsertion(R.getNameLoc(), "this->");
            QualType DepThisType = DepMethod->getThisType(Context);
            CXXThisExpr *DepThis = new (Context) CXXThisExpr(
                                       R.getNameLoc(), DepThisType, false);
            TemplateArgumentListInfo TList;
            if (ULE->hasExplicitTemplateArgs())
              ULE->copyTemplateArgumentsInto(TList);
            CXXDependentScopeMemberExpr *DepExpr =
                CXXDependentScopeMemberExpr::Create(
                    Context, DepThis, DepThisType, true, SourceLocation(),
                    ULE->getQualifier(), ULE->getQualifierRange(), NULL,
                    R.getLookupNameInfo(), &TList);
            CallsUndergoingInstantiation.back()->setCallee(DepExpr);
          } else {
            // FIXME: we should be able to handle this case too. It is correct
            // to add this-> here. This is a workaround for PR7947.
            Diag(R.getNameLoc(), diagnostic) << Name;
          }
        } else {
          Diag(R.getNameLoc(), diagnostic) << Name;
        }

        // Do we really want to note all of these?
        for (LookupResult::iterator I = R.begin(), E = R.end(); I != E; ++I)
          Diag((*I)->getLocation(), diag::note_dependent_var_use);

        // Tell the callee to try to recover.
        return false;
      }

      R.clear();
    }
  }

  // We didn't find anything, so try to correct for a typo.
  DeclarationName Corrected;
  if (S && (Corrected = CorrectTypo(R, S, &SS, 0, false, CTC))) {
    if (!R.empty()) {
      if (isa<ValueDecl>(*R.begin()) || isa<FunctionTemplateDecl>(*R.begin())) {
        if (SS.isEmpty())
          Diag(R.getNameLoc(), diagnostic_suggest) << Name << R.getLookupName()
            << FixItHint::CreateReplacement(R.getNameLoc(),
                                            R.getLookupName().getAsString());
        else
          Diag(R.getNameLoc(), diag::err_no_member_suggest)
            << Name << computeDeclContext(SS, false) << R.getLookupName()
            << SS.getRange()
            << FixItHint::CreateReplacement(R.getNameLoc(),
                                            R.getLookupName().getAsString());
        if (NamedDecl *ND = R.getAsSingle<NamedDecl>())
          Diag(ND->getLocation(), diag::note_previous_decl)
            << ND->getDeclName();

        // Tell the callee to try to recover.
        return false;
      }

      if (isa<TypeDecl>(*R.begin()) || isa<ObjCInterfaceDecl>(*R.begin())) {
        // FIXME: If we ended up with a typo for a type name or
        // Objective-C class name, we're in trouble because the parser
        // is in the wrong place to recover. Suggest the typo
        // correction, but don't make it a fix-it since we're not going
        // to recover well anyway.
        if (SS.isEmpty())
          Diag(R.getNameLoc(), diagnostic_suggest) << Name << R.getLookupName();
        else
          Diag(R.getNameLoc(), diag::err_no_member_suggest)
            << Name << computeDeclContext(SS, false) << R.getLookupName()
            << SS.getRange();

        // Don't try to recover; it won't work.
        return true;
      }
    } else {
      // FIXME: We found a keyword. Suggest it, but don't provide a fix-it
      // because we aren't able to recover.
      if (SS.isEmpty())
        Diag(R.getNameLoc(), diagnostic_suggest) << Name << Corrected;
      else
        Diag(R.getNameLoc(), diag::err_no_member_suggest)
        << Name << computeDeclContext(SS, false) << Corrected
        << SS.getRange();
      return true;
    }
    R.clear();
  }

  // Emit a special diagnostic for failed member lookups.
  // FIXME: computing the declaration context might fail here (?)
  if (!SS.isEmpty()) {
    Diag(R.getNameLoc(), diag::err_no_member)
      << Name << computeDeclContext(SS, false)
      << SS.getRange();
    return true;
  }

  // Give up, we can't recover.
  Diag(R.getNameLoc(), diagnostic) << Name;
  return true;
}

static ObjCPropertyDecl *OkToSynthesizeProvisionalIvar(Sema &SemaRef,
                                                       IdentifierInfo *II,
                                                       SourceLocation NameLoc) {
  ObjCMethodDecl *CurMeth = SemaRef.getCurMethodDecl();
  ObjCInterfaceDecl *IDecl = CurMeth->getClassInterface();
  if (!IDecl)
    return 0;
  ObjCImplementationDecl *ClassImpDecl = IDecl->getImplementation();
  if (!ClassImpDecl)
    return 0;
  ObjCPropertyDecl *property = SemaRef.LookupPropertyDecl(IDecl, II);
  if (!property)
    return 0;
  if (ObjCPropertyImplDecl *PIDecl = ClassImpDecl->FindPropertyImplDecl(II))
    if (PIDecl->getPropertyImplementation() == ObjCPropertyImplDecl::Dynamic)
      return 0;
  return property;
}

static ObjCIvarDecl *SynthesizeProvisionalIvar(Sema &SemaRef,
                                               LookupResult &Lookup,
                                               IdentifierInfo *II,
                                               SourceLocation NameLoc) {
  ObjCMethodDecl *CurMeth = SemaRef.getCurMethodDecl();
  bool LookForIvars;
  if (Lookup.empty())
    LookForIvars = true;
  else if (CurMeth->isClassMethod())
    LookForIvars = false;
  else
    LookForIvars = (Lookup.isSingleResult() &&
                    Lookup.getFoundDecl()->isDefinedOutsideFunctionOrMethod());
  if (!LookForIvars)
    return 0;
  
  ObjCInterfaceDecl *IDecl = CurMeth->getClassInterface();
  if (!IDecl)
    return 0;
  ObjCImplementationDecl *ClassImpDecl = IDecl->getImplementation();
  if (!ClassImpDecl)
    return 0;
  bool DynamicImplSeen = false;
  ObjCPropertyDecl *property = SemaRef.LookupPropertyDecl(IDecl, II);
  if (!property)
    return 0;
  if (ObjCPropertyImplDecl *PIDecl = ClassImpDecl->FindPropertyImplDecl(II))
    DynamicImplSeen = 
      (PIDecl->getPropertyImplementation() == ObjCPropertyImplDecl::Dynamic);
  if (!DynamicImplSeen) {
    QualType PropType = SemaRef.Context.getCanonicalType(property->getType());
    ObjCIvarDecl *Ivar = ObjCIvarDecl::Create(SemaRef.Context, ClassImpDecl, 
                                              NameLoc,
                                              II, PropType, /*Dinfo=*/0,
                                              ObjCIvarDecl::Protected,
                                              (Expr *)0, true);
    ClassImpDecl->addDecl(Ivar);
    IDecl->makeDeclVisibleInContext(Ivar, false);
    property->setPropertyIvarDecl(Ivar);
    return Ivar;
  }
  return 0;
}

ExprResult Sema::ActOnIdExpression(Scope *S,
                                   CXXScopeSpec &SS,
                                   UnqualifiedId &Id,
                                   bool HasTrailingLParen,
                                   bool isAddressOfOperand) {
  assert(!(isAddressOfOperand && HasTrailingLParen) &&
         "cannot be direct & operand and have a trailing lparen");

  if (SS.isInvalid())
    return ExprError();

  TemplateArgumentListInfo TemplateArgsBuffer;

  // Decompose the UnqualifiedId into the following data.
  DeclarationNameInfo NameInfo;
  const TemplateArgumentListInfo *TemplateArgs;
  DecomposeUnqualifiedId(*this, Id, TemplateArgsBuffer, NameInfo, TemplateArgs);

  DeclarationName Name = NameInfo.getName();
  IdentifierInfo *II = Name.getAsIdentifierInfo();
  SourceLocation NameLoc = NameInfo.getLoc();

  // C++ [temp.dep.expr]p3:
  //   An id-expression is type-dependent if it contains:
  //     -- an identifier that was declared with a dependent type,
  //        (note: handled after lookup)
  //     -- a template-id that is dependent,
  //        (note: handled in BuildTemplateIdExpr)
  //     -- a conversion-function-id that specifies a dependent type,
  //     -- a nested-name-specifier that contains a class-name that
  //        names a dependent type.
  // Determine whether this is a member of an unknown specialization;
  // we need to handle these differently.
  bool DependentID = false;
  if (Name.getNameKind() == DeclarationName::CXXConversionFunctionName &&
      Name.getCXXNameType()->isDependentType()) {
    DependentID = true;
  } else if (SS.isSet()) {
    DeclContext *DC = computeDeclContext(SS, false);
    if (DC) {
      if (RequireCompleteDeclContext(SS, DC))
        return ExprError();
      // FIXME: We should be checking whether DC is the current instantiation.
      if (CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(DC))
        DependentID = !IsFullyFormedScope(*this, RD);
    } else {
      DependentID = true;
    }
  }

  if (DependentID) {
    return ActOnDependentIdExpression(SS, NameInfo, isAddressOfOperand,
                                      TemplateArgs);
  }
  bool IvarLookupFollowUp = false;
  // Perform the required lookup.
  LookupResult R(*this, NameInfo, LookupOrdinaryName);
  if (TemplateArgs) {
    // Lookup the template name again to correctly establish the context in
    // which it was found. This is really unfortunate as we already did the
    // lookup to determine that it was a template name in the first place. If
    // this becomes a performance hit, we can work harder to preserve those
    // results until we get here but it's likely not worth it.
    bool MemberOfUnknownSpecialization;
    LookupTemplateName(R, S, SS, QualType(), /*EnteringContext=*/false,
                       MemberOfUnknownSpecialization);
  } else {
    IvarLookupFollowUp = (!SS.isSet() && II && getCurMethodDecl());
    LookupParsedName(R, S, &SS, !IvarLookupFollowUp);

    // If this reference is in an Objective-C method, then we need to do
    // some special Objective-C lookup, too.
    if (IvarLookupFollowUp) {
      ExprResult E(LookupInObjCMethod(R, S, II, true));
      if (E.isInvalid())
        return ExprError();

      Expr *Ex = E.takeAs<Expr>();
      if (Ex) return Owned(Ex);
      // Synthesize ivars lazily
      if (getLangOptions().ObjCNonFragileABI2) {
        if (SynthesizeProvisionalIvar(*this, R, II, NameLoc))
          return ActOnIdExpression(S, SS, Id, HasTrailingLParen,
                                   isAddressOfOperand);
      }
      // for further use, this must be set to false if in class method.
      IvarLookupFollowUp = getCurMethodDecl()->isInstanceMethod();
    }
  }

  if (R.isAmbiguous())
    return ExprError();

  // Determine whether this name might be a candidate for
  // argument-dependent lookup.
  bool ADL = UseArgumentDependentLookup(SS, R, HasTrailingLParen);

  if (R.empty() && !ADL) {
    // Otherwise, this could be an implicitly declared function reference (legal
    // in C90, extension in C99, forbidden in C++).
    if (HasTrailingLParen && II && !getLangOptions().CPlusPlus) {
      NamedDecl *D = ImplicitlyDefineFunction(NameLoc, *II, S);
      if (D) R.addDecl(D);
    }

    // If this name wasn't predeclared and if this is not a function
    // call, diagnose the problem.
    if (R.empty()) {
      if (DiagnoseEmptyLookup(S, SS, R, CTC_Unknown))
        return ExprError();

      assert(!R.empty() &&
             "DiagnoseEmptyLookup returned false but added no results");

      // If we found an Objective-C instance variable, let
      // LookupInObjCMethod build the appropriate expression to
      // reference the ivar.
      if (ObjCIvarDecl *Ivar = R.getAsSingle<ObjCIvarDecl>()) {
        R.clear();
        ExprResult E(LookupInObjCMethod(R, S, Ivar->getIdentifier()));
        assert(E.isInvalid() || E.get());
        return move(E);
      }
    }
  }

  // This is guaranteed from this point on.
  assert(!R.empty() || ADL);

  if (VarDecl *Var = R.getAsSingle<VarDecl>()) {
    if (getLangOptions().ObjCNonFragileABI && IvarLookupFollowUp &&
        !getLangOptions().ObjCNonFragileABI2 &&
        Var->isFileVarDecl()) {
      ObjCPropertyDecl *Property = 
        OkToSynthesizeProvisionalIvar(*this, II, NameLoc);
      if (Property) {
        Diag(NameLoc, diag::warn_ivar_variable_conflict) << Var->getDeclName();
        Diag(Property->getLocation(), diag::note_property_declare);
        Diag(Var->getLocation(), diag::note_global_declared_at);
      }
    }
  } else if (FunctionDecl *Func = R.getAsSingle<FunctionDecl>()) {
    if (!getLangOptions().CPlusPlus && !Func->hasPrototype()) {
      // C99 DR 316 says that, if a function type comes from a
      // function definition (without a prototype), that type is only
      // used for checking compatibility. Therefore, when referencing
      // the function, we pretend that we don't have the full function
      // type.
      if (DiagnoseUseOfDecl(Func, NameLoc))
        return ExprError();

      QualType T = Func->getType();
      QualType NoProtoType = T;
      if (const FunctionProtoType *Proto = T->getAs<FunctionProtoType>())
        NoProtoType = Context.getFunctionNoProtoType(Proto->getResultType(),
                                                     Proto->getExtInfo());
      return BuildDeclRefExpr(Func, NoProtoType, NameLoc, &SS);
    }
  }

  // Check whether this might be a C++ implicit instance member access.
  // C++ [class.mfct.non-static]p3:
  //   When an id-expression that is not part of a class member access
  //   syntax and not used to form a pointer to member is used in the
  //   body of a non-static member function of class X, if name lookup
  //   resolves the name in the id-expression to a non-static non-type
  //   member of some class C, the id-expression is transformed into a
  //   class member access expression using (*this) as the
  //   postfix-expression to the left of the . operator.
  //
  // But we don't actually need to do this for '&' operands if R
  // resolved to a function or overloaded function set, because the
  // expression is ill-formed if it actually works out to be a
  // non-static member function:
  //
  // C++ [expr.ref]p4:
  //   Otherwise, if E1.E2 refers to a non-static member function. . .
  //   [t]he expression can be used only as the left-hand operand of a
  //   member function call.
  //
  // There are other safeguards against such uses, but it's important
  // to get this right here so that we don't end up making a
  // spuriously dependent expression if we're inside a dependent
  // instance method.
  if (!R.empty() && (*R.begin())->isCXXClassMember()) {
    bool MightBeImplicitMember;
    if (!isAddressOfOperand)
      MightBeImplicitMember = true;
    else if (!SS.isEmpty())
      MightBeImplicitMember = false;
    else if (R.isOverloadedResult())
      MightBeImplicitMember = false;
    else if (R.isUnresolvableResult())
      MightBeImplicitMember = true;
    else
      MightBeImplicitMember = isa<FieldDecl>(R.getFoundDecl());

    if (MightBeImplicitMember)
      return BuildPossibleImplicitMemberExpr(SS, R, TemplateArgs);
  }

  if (TemplateArgs)
    return BuildTemplateIdExpr(SS, R, ADL, *TemplateArgs);

  return BuildDeclarationNameExpr(SS, R, ADL);
}

/// Builds an expression which might be an implicit member expression.
ExprResult
Sema::BuildPossibleImplicitMemberExpr(const CXXScopeSpec &SS,
                                      LookupResult &R,
                                const TemplateArgumentListInfo *TemplateArgs) {
  switch (ClassifyImplicitMemberAccess(*this, R)) {
  case IMA_Instance:
    return BuildImplicitMemberExpr(SS, R, TemplateArgs, true);

  case IMA_AnonymousMember:
    assert(R.isSingleResult());
    return BuildAnonymousStructUnionMemberReference(R.getNameLoc(),
                                                    R.getAsSingle<FieldDecl>());

  case IMA_Mixed:
  case IMA_Mixed_Unrelated:
  case IMA_Unresolved:
    return BuildImplicitMemberExpr(SS, R, TemplateArgs, false);

  case IMA_Static:
  case IMA_Mixed_StaticContext:
  case IMA_Unresolved_StaticContext:
    if (TemplateArgs)
      return BuildTemplateIdExpr(SS, R, false, *TemplateArgs);
    return BuildDeclarationNameExpr(SS, R, false);

  case IMA_Error_StaticContext:
  case IMA_Error_Unrelated:
    DiagnoseInstanceReference(*this, SS, R);
    return ExprError();
  }

  llvm_unreachable("unexpected instance member access kind");
  return ExprError();
}

/// BuildQualifiedDeclarationNameExpr - Build a C++ qualified
/// declaration name, generally during template instantiation.
/// There's a large number of things which don't need to be done along
/// this path.
ExprResult
Sema::BuildQualifiedDeclarationNameExpr(CXXScopeSpec &SS,
                                        const DeclarationNameInfo &NameInfo) {
  DeclContext *DC;
  if (!(DC = computeDeclContext(SS, false)) || DC->isDependentContext())
    return BuildDependentDeclRefExpr(SS, NameInfo, 0);

  if (RequireCompleteDeclContext(SS, DC))
    return ExprError();

  LookupResult R(*this, NameInfo, LookupOrdinaryName);
  LookupQualifiedName(R, DC);

  if (R.isAmbiguous())
    return ExprError();

  if (R.empty()) {
    Diag(NameInfo.getLoc(), diag::err_no_member)
      << NameInfo.getName() << DC << SS.getRange();
    return ExprError();
  }

  return BuildDeclarationNameExpr(SS, R, /*ADL*/ false);
}

/// LookupInObjCMethod - The parser has read a name in, and Sema has
/// detected that we're currently inside an ObjC method.  Perform some
/// additional lookup.
///
/// Ideally, most of this would be done by lookup, but there's
/// actually quite a lot of extra work involved.
///
/// Returns a null sentinel to indicate trivial success.
ExprResult
Sema::LookupInObjCMethod(LookupResult &Lookup, Scope *S,
                         IdentifierInfo *II, bool AllowBuiltinCreation) {
  SourceLocation Loc = Lookup.getNameLoc();
  ObjCMethodDecl *CurMethod = getCurMethodDecl();

  // There are two cases to handle here.  1) scoped lookup could have failed,
  // in which case we should look for an ivar.  2) scoped lookup could have
  // found a decl, but that decl is outside the current instance method (i.e.
  // a global variable).  In these two cases, we do a lookup for an ivar with
  // this name, if the lookup sucedes, we replace it our current decl.

  // If we're in a class method, we don't normally want to look for
  // ivars.  But if we don't find anything else, and there's an
  // ivar, that's an error.
  bool IsClassMethod = CurMethod->isClassMethod();

  bool LookForIvars;
  if (Lookup.empty())
    LookForIvars = true;
  else if (IsClassMethod)
    LookForIvars = false;
  else
    LookForIvars = (Lookup.isSingleResult() &&
                    Lookup.getFoundDecl()->isDefinedOutsideFunctionOrMethod());
  ObjCInterfaceDecl *IFace = 0;
  if (LookForIvars) {
    IFace = CurMethod->getClassInterface();
    ObjCInterfaceDecl *ClassDeclared;
    if (ObjCIvarDecl *IV = IFace->lookupInstanceVariable(II, ClassDeclared)) {
      // Diagnose using an ivar in a class method.
      if (IsClassMethod)
        return ExprError(Diag(Loc, diag::error_ivar_use_in_class_method)
                         << IV->getDeclName());

      // If we're referencing an invalid decl, just return this as a silent
      // error node.  The error diagnostic was already emitted on the decl.
      if (IV->isInvalidDecl())
        return ExprError();

      // Check if referencing a field with __attribute__((deprecated)).
      if (DiagnoseUseOfDecl(IV, Loc))
        return ExprError();

      // Diagnose the use of an ivar outside of the declaring class.
      if (IV->getAccessControl() == ObjCIvarDecl::Private &&
          ClassDeclared != IFace)
        Diag(Loc, diag::error_private_ivar_access) << IV->getDeclName();

      // FIXME: This should use a new expr for a direct reference, don't
      // turn this into Self->ivar, just return a BareIVarExpr or something.
      IdentifierInfo &II = Context.Idents.get("self");
      UnqualifiedId SelfName;
      SelfName.setIdentifier(&II, SourceLocation());
      CXXScopeSpec SelfScopeSpec;
      ExprResult SelfExpr = ActOnIdExpression(S, SelfScopeSpec,
                                                    SelfName, false, false);
      MarkDeclarationReferenced(Loc, IV);
      return Owned(new (Context)
                   ObjCIvarRefExpr(IV, IV->getType(), Loc,
                                   SelfExpr.takeAs<Expr>(), true, true));
    }
  } else if (CurMethod->isInstanceMethod()) {
    // We should warn if a local variable hides an ivar.
    ObjCInterfaceDecl *IFace = CurMethod->getClassInterface();
    ObjCInterfaceDecl *ClassDeclared;
    if (ObjCIvarDecl *IV = IFace->lookupInstanceVariable(II, ClassDeclared)) {
      if (IV->getAccessControl() != ObjCIvarDecl::Private ||
          IFace == ClassDeclared)
        Diag(Loc, diag::warn_ivar_use_hidden) << IV->getDeclName();
    }
  }

  if (Lookup.empty() && II && AllowBuiltinCreation) {
    // FIXME. Consolidate this with similar code in LookupName.
    if (unsigned BuiltinID = II->getBuiltinID()) {
      if (!(getLangOptions().CPlusPlus &&
            Context.BuiltinInfo.isPredefinedLibFunction(BuiltinID))) {
        NamedDecl *D = LazilyCreateBuiltin((IdentifierInfo *)II, BuiltinID,
                                           S, Lookup.isForRedeclaration(),
                                           Lookup.getNameLoc());
        if (D) Lookup.addDecl(D);
      }
    }
  }
  // Sentinel value saying that we didn't do anything special.
  return Owned((Expr*) 0);
}

/// \brief Cast a base object to a member's actual type.
///
/// Logically this happens in three phases:
///
/// * First we cast from the base type to the naming class.
///   The naming class is the class into which we were looking
///   when we found the member;  it's the qualifier type if a
///   qualifier was provided, and otherwise it's the base type.
///
/// * Next we cast from the naming class to the declaring class.
///   If the member we found was brought into a class's scope by
///   a using declaration, this is that class;  otherwise it's
///   the class declaring the member.
///
/// * Finally we cast from the declaring class to the "true"
///   declaring class of the member.  This conversion does not
///   obey access control.
bool
Sema::PerformObjectMemberConversion(Expr *&From,
                                    NestedNameSpecifier *Qualifier,
                                    NamedDecl *FoundDecl,
                                    NamedDecl *Member) {
  CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(Member->getDeclContext());
  if (!RD)
    return false;

  QualType DestRecordType;
  QualType DestType;
  QualType FromRecordType;
  QualType FromType = From->getType();
  bool PointerConversions = false;
  if (isa<FieldDecl>(Member)) {
    DestRecordType = Context.getCanonicalType(Context.getTypeDeclType(RD));

    if (FromType->getAs<PointerType>()) {
      DestType = Context.getPointerType(DestRecordType);
      FromRecordType = FromType->getPointeeType();
      PointerConversions = true;
    } else {
      DestType = DestRecordType;
      FromRecordType = FromType;
    }
  } else if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(Member)) {
    if (Method->isStatic())
      return false;

    DestType = Method->getThisType(Context);
    DestRecordType = DestType->getPointeeType();

    if (FromType->getAs<PointerType>()) {
      FromRecordType = FromType->getPointeeType();
      PointerConversions = true;
    } else {
      FromRecordType = FromType;
      DestType = DestRecordType;
    }
  } else {
    // No conversion necessary.
    return false;
  }

  if (DestType->isDependentType() || FromType->isDependentType())
    return false;

  // If the unqualified types are the same, no conversion is necessary.
  if (Context.hasSameUnqualifiedType(FromRecordType, DestRecordType))
    return false;

  SourceRange FromRange = From->getSourceRange();
  SourceLocation FromLoc = FromRange.getBegin();

  ExprValueKind VK = CastCategory(From);

  // C++ [class.member.lookup]p8:
  //   [...] Ambiguities can often be resolved by qualifying a name with its
  //   class name.
  //
  // If the member was a qualified name and the qualified referred to a
  // specific base subobject type, we'll cast to that intermediate type
  // first and then to the object in which the member is declared. That allows
  // one to resolve ambiguities in, e.g., a diamond-shaped hierarchy such as:
  //
  //   class Base { public: int x; };
  //   class Derived1 : public Base { };
  //   class Derived2 : public Base { };
  //   class VeryDerived : public Derived1, public Derived2 { void f(); };
  //
  //   void VeryDerived::f() {
  //     x = 17; // error: ambiguous base subobjects
  //     Derived1::x = 17; // okay, pick the Base subobject of Derived1
  //   }
  if (Qualifier) {
    QualType QType = QualType(Qualifier->getAsType(), 0);
    assert(!QType.isNull() && "lookup done with dependent qualifier?");
    assert(QType->isRecordType() && "lookup done with non-record type");

    QualType QRecordType = QualType(QType->getAs<RecordType>(), 0);

    // In C++98, the qualifier type doesn't actually have to be a base
    // type of the object type, in which case we just ignore it.
    // Otherwise build the appropriate casts.
    if (IsDerivedFrom(FromRecordType, QRecordType)) {
      CXXCastPath BasePath;
      if (CheckDerivedToBaseConversion(FromRecordType, QRecordType,
                                       FromLoc, FromRange, &BasePath))
        return true;

      if (PointerConversions)
        QType = Context.getPointerType(QType);
      ImpCastExprToType(From, QType, CK_UncheckedDerivedToBase,
                        VK, &BasePath);

      FromType = QType;
      FromRecordType = QRecordType;

      // If the qualifier type was the same as the destination type,
      // we're done.
      if (Context.hasSameUnqualifiedType(FromRecordType, DestRecordType))
        return false;
    }
  }

  bool IgnoreAccess = false;

  // If we actually found the member through a using declaration, cast
  // down to the using declaration's type.
  //
  // Pointer equality is fine here because only one declaration of a
  // class ever has member declarations.
  if (FoundDecl->getDeclContext() != Member->getDeclContext()) {
    assert(isa<UsingShadowDecl>(FoundDecl));
    QualType URecordType = Context.getTypeDeclType(
                           cast<CXXRecordDecl>(FoundDecl->getDeclContext()));

    // We only need to do this if the naming-class to declaring-class
    // conversion is non-trivial.
    if (!Context.hasSameUnqualifiedType(FromRecordType, URecordType)) {
      assert(IsDerivedFrom(FromRecordType, URecordType));
      CXXCastPath BasePath;
      if (CheckDerivedToBaseConversion(FromRecordType, URecordType,
                                       FromLoc, FromRange, &BasePath))
        return true;

      QualType UType = URecordType;
      if (PointerConversions)
        UType = Context.getPointerType(UType);
      ImpCastExprToType(From, UType, CK_UncheckedDerivedToBase,
                        VK, &BasePath);
      FromType = UType;
      FromRecordType = URecordType;
    }

    // We don't do access control for the conversion from the
    // declaring class to the true declaring class.
    IgnoreAccess = true;
  }

  CXXCastPath BasePath;
  if (CheckDerivedToBaseConversion(FromRecordType, DestRecordType,
                                   FromLoc, FromRange, &BasePath,
                                   IgnoreAccess))
    return true;

  ImpCastExprToType(From, DestType, CK_UncheckedDerivedToBase,
                    VK, &BasePath);
  return false;
}

/// \brief Build a MemberExpr AST node.
static MemberExpr *BuildMemberExpr(ASTContext &C, Expr *Base, bool isArrow,
                                   const CXXScopeSpec &SS, ValueDecl *Member,
                                   DeclAccessPair FoundDecl,
                                   const DeclarationNameInfo &MemberNameInfo,
                                   QualType Ty,
                          const TemplateArgumentListInfo *TemplateArgs = 0) {
  NestedNameSpecifier *Qualifier = 0;
  SourceRange QualifierRange;
  if (SS.isSet()) {
    Qualifier = (NestedNameSpecifier *) SS.getScopeRep();
    QualifierRange = SS.getRange();
  }

  return MemberExpr::Create(C, Base, isArrow, Qualifier, QualifierRange,
                            Member, FoundDecl, MemberNameInfo,
                            TemplateArgs, Ty);
}

/// Builds an implicit member access expression.  The current context
/// is known to be an instance method, and the given unqualified lookup
/// set is known to contain only instance members, at least one of which
/// is from an appropriate type.
ExprResult
Sema::BuildImplicitMemberExpr(const CXXScopeSpec &SS,
                              LookupResult &R,
                              const TemplateArgumentListInfo *TemplateArgs,
                              bool IsKnownInstance) {
  assert(!R.empty() && !R.isAmbiguous());

  SourceLocation Loc = R.getNameLoc();

  // We may have found a field within an anonymous union or struct
  // (C++ [class.union]).
  // FIXME: This needs to happen post-isImplicitMemberReference?
  // FIXME: template-ids inside anonymous structs?
  if (FieldDecl *FD = R.getAsSingle<FieldDecl>())
    if (cast<RecordDecl>(FD->getDeclContext())->isAnonymousStructOrUnion())
      return BuildAnonymousStructUnionMemberReference(Loc, FD);

  // If this is known to be an instance access, go ahead and build a
  // 'this' expression now.
  DeclContext *DC = getFunctionLevelDeclContext();
  QualType ThisType = cast<CXXMethodDecl>(DC)->getThisType(Context);
  Expr *This = 0; // null signifies implicit access
  if (IsKnownInstance) {
    SourceLocation Loc = R.getNameLoc();
    if (SS.getRange().isValid())
      Loc = SS.getRange().getBegin();
    This = new (Context) CXXThisExpr(Loc, ThisType, /*isImplicit=*/true);
  }

  return BuildMemberReferenceExpr(This, ThisType,
                                  /*OpLoc*/ SourceLocation(),
                                  /*IsArrow*/ true,
                                  SS,
                                  /*FirstQualifierInScope*/ 0,
                                  R, TemplateArgs);
}

bool Sema::UseArgumentDependentLookup(const CXXScopeSpec &SS,
                                      const LookupResult &R,
                                      bool HasTrailingLParen) {
  // Only when used directly as the postfix-expression of a call.
  if (!HasTrailingLParen)
    return false;

  // Never if a scope specifier was provided.
  if (SS.isSet())
    return false;

  // Only in C++ or ObjC++.
  if (!getLangOptions().CPlusPlus)
    return false;

  // Turn off ADL when we find certain kinds of declarations during
  // normal lookup:
  for (LookupResult::iterator I = R.begin(), E = R.end(); I != E; ++I) {
    NamedDecl *D = *I;

    // C++0x [basic.lookup.argdep]p3:
    //     -- a declaration of a class member
    // Since using decls preserve this property, we check this on the
    // original decl.
    if (D->isCXXClassMember())
      return false;

    // C++0x [basic.lookup.argdep]p3:
    //     -- a block-scope function declaration that is not a
    //        using-declaration
    // NOTE: we also trigger this for function templates (in fact, we
    // don't check the decl type at all, since all other decl types
    // turn off ADL anyway).
    if (isa<UsingShadowDecl>(D))
      D = cast<UsingShadowDecl>(D)->getTargetDecl();
    else if (D->getDeclContext()->isFunctionOrMethod())
      return false;

    // C++0x [basic.lookup.argdep]p3:
    //     -- a declaration that is neither a function or a function
    //        template
    // And also for builtin functions.
    if (isa<FunctionDecl>(D)) {
      FunctionDecl *FDecl = cast<FunctionDecl>(D);

      // But also builtin functions.
      if (FDecl->getBuiltinID() && FDecl->isImplicit())
        return false;
    } else if (!isa<FunctionTemplateDecl>(D))
      return false;
  }

  return true;
}


/// Diagnoses obvious problems with the use of the given declaration
/// as an expression.  This is only actually called for lookups that
/// were not overloaded, and it doesn't promise that the declaration
/// will in fact be used.
static bool CheckDeclInExpr(Sema &S, SourceLocation Loc, NamedDecl *D) {
  if (isa<TypedefDecl>(D)) {
    S.Diag(Loc, diag::err_unexpected_typedef) << D->getDeclName();
    return true;
  }

  if (isa<ObjCInterfaceDecl>(D)) {
    S.Diag(Loc, diag::err_unexpected_interface) << D->getDeclName();
    return true;
  }

  if (isa<NamespaceDecl>(D)) {
    S.Diag(Loc, diag::err_unexpected_namespace) << D->getDeclName();
    return true;
  }

  return false;
}

ExprResult
Sema::BuildDeclarationNameExpr(const CXXScopeSpec &SS,
                               LookupResult &R,
                               bool NeedsADL) {
  // If this is a single, fully-resolved result and we don't need ADL,
  // just build an ordinary singleton decl ref.
  if (!NeedsADL && R.isSingleResult() && !R.getAsSingle<FunctionTemplateDecl>())
    return BuildDeclarationNameExpr(SS, R.getLookupNameInfo(),
                                    R.getFoundDecl());

  // We only need to check the declaration if there's exactly one
  // result, because in the overloaded case the results can only be
  // functions and function templates.
  if (R.isSingleResult() &&
      CheckDeclInExpr(*this, R.getNameLoc(), R.getFoundDecl()))
    return ExprError();

  // Otherwise, just build an unresolved lookup expression.  Suppress
  // any lookup-related diagnostics; we'll hash these out later, when
  // we've picked a target.
  R.suppressDiagnostics();

  bool Dependent
    = UnresolvedLookupExpr::ComputeDependence(R.begin(), R.end(), 0);
  UnresolvedLookupExpr *ULE
    = UnresolvedLookupExpr::Create(Context, Dependent, R.getNamingClass(),
                                   (NestedNameSpecifier*) SS.getScopeRep(),
                                   SS.getRange(), R.getLookupNameInfo(),
                                   NeedsADL, R.isOverloadedResult(),
                                   R.begin(), R.end());

  return Owned(ULE);
}


/// \brief Complete semantic analysis for a reference to the given declaration.
ExprResult
Sema::BuildDeclarationNameExpr(const CXXScopeSpec &SS,
                               const DeclarationNameInfo &NameInfo,
                               NamedDecl *D) {
  assert(D && "Cannot refer to a NULL declaration");
  assert(!isa<FunctionTemplateDecl>(D) &&
         "Cannot refer unambiguously to a function template");

  SourceLocation Loc = NameInfo.getLoc();
  if (CheckDeclInExpr(*this, Loc, D))
    return ExprError();

  if (TemplateDecl *Template = dyn_cast<TemplateDecl>(D)) {
    // Specifically diagnose references to class templates that are missing
    // a template argument list.
    Diag(Loc, diag::err_template_decl_ref)
      << Template << SS.getRange();
    Diag(Template->getLocation(), diag::note_template_decl_here);
    return ExprError();
  }

  // Make sure that we're referring to a value.
  ValueDecl *VD = dyn_cast<ValueDecl>(D);
  if (!VD) {
    Diag(Loc, diag::err_ref_non_value)
      << D << SS.getRange();
    Diag(D->getLocation(), diag::note_declared_at);
    return ExprError();
  }

  // Check whether this declaration can be used. Note that we suppress
  // this check when we're going to perform argument-dependent lookup
  // on this function name, because this might not be the function
  // that overload resolution actually selects.
  if (DiagnoseUseOfDecl(VD, Loc))
    return ExprError();

  // Only create DeclRefExpr's for valid Decl's.
  if (VD->isInvalidDecl())
    return ExprError();

  // If the identifier reference is inside a block, and it refers to a value
  // that is outside the block, create a BlockDeclRefExpr instead of a
  // DeclRefExpr.  This ensures the value is treated as a copy-in snapshot when
  // the block is formed.
  //
  // We do not do this for things like enum constants, global variables, etc,
  // as they do not get snapshotted.
  //
  if (getCurBlock() &&
      ShouldSnapshotBlockValueReference(*this, getCurBlock(), VD)) {
    if (VD->getType().getTypePtr()->isVariablyModifiedType()) {
      Diag(Loc, diag::err_ref_vm_type);
      Diag(D->getLocation(), diag::note_declared_at);
      return ExprError();
    }

    if (VD->getType()->isArrayType()) {
      Diag(Loc, diag::err_ref_array_type);
      Diag(D->getLocation(), diag::note_declared_at);
      return ExprError();
    }

    MarkDeclarationReferenced(Loc, VD);
    QualType ExprTy = VD->getType().getNonReferenceType();
    // The BlocksAttr indicates the variable is bound by-reference.
    if (VD->getAttr<BlocksAttr>())
      return Owned(new (Context) BlockDeclRefExpr(VD, ExprTy, Loc, true));
    // This is to record that a 'const' was actually synthesize and added.
    bool constAdded = !ExprTy.isConstQualified();
    // Variable will be bound by-copy, make it const within the closure.

    ExprTy.addConst();
    QualType T = VD->getType();
    BlockDeclRefExpr *BDRE = new (Context) BlockDeclRefExpr(VD, 
                                                            ExprTy, Loc, false,
                                                            constAdded);
    if (getLangOptions().CPlusPlus) {
      if (!T->isDependentType() && !T->isReferenceType()) {
        Expr *E = new (Context) 
                    DeclRefExpr(const_cast<ValueDecl*>(BDRE->getDecl()), T,
                                          SourceLocation());
      
        ExprResult Res = PerformCopyInitialization(
                          InitializedEntity::InitializeBlock(VD->getLocation(), 
                                                         T, false),
                                                         SourceLocation(),
                                                         Owned(E));
        if (!Res.isInvalid()) {
          Res = MaybeCreateCXXExprWithTemporaries(Res.get());
          Expr *Init = Res.takeAs<Expr>();
          BDRE->setCopyConstructorExpr(Init);
        }
      }
    }
    return Owned(BDRE);
  }
  // If this reference is not in a block or if the referenced variable is
  // within the block, create a normal DeclRefExpr.

  return BuildDeclRefExpr(VD, VD->getType().getNonReferenceType(),
                          NameInfo, &SS);
}

ExprResult Sema::ActOnPredefinedExpr(SourceLocation Loc,
                                                 tok::TokenKind Kind) {
  PredefinedExpr::IdentType IT;

  switch (Kind) {
  default: assert(0 && "Unknown simple primary expr!");
  case tok::kw___func__: IT = PredefinedExpr::Func; break; // [C99 6.4.2.2]
  case tok::kw___FUNCTION__: IT = PredefinedExpr::Function; break;
  case tok::kw___PRETTY_FUNCTION__: IT = PredefinedExpr::PrettyFunction; break;
  }

  // Pre-defined identifiers are of type char[x], where x is the length of the
  // string.

  Decl *currentDecl = getCurFunctionOrMethodDecl();
  if (!currentDecl && getCurBlock())
    currentDecl = getCurBlock()->TheDecl;
  if (!currentDecl) {
    Diag(Loc, diag::ext_predef_outside_function);
    currentDecl = Context.getTranslationUnitDecl();
  }

  QualType ResTy;
  if (cast<DeclContext>(currentDecl)->isDependentContext()) {
    ResTy = Context.DependentTy;
  } else {
    unsigned Length = PredefinedExpr::ComputeName(IT, currentDecl).length();

    llvm::APInt LengthI(32, Length + 1);
    ResTy = Context.CharTy.withConst();
    ResTy = Context.getConstantArrayType(ResTy, LengthI, ArrayType::Normal, 0);
  }
  return Owned(new (Context) PredefinedExpr(Loc, ResTy, IT));
}

ExprResult Sema::ActOnCharacterConstant(const Token &Tok) {
  llvm::SmallString<16> CharBuffer;
  bool Invalid = false;
  llvm::StringRef ThisTok = PP.getSpelling(Tok, CharBuffer, &Invalid);
  if (Invalid)
    return ExprError();

  CharLiteralParser Literal(ThisTok.begin(), ThisTok.end(), Tok.getLocation(),
                            PP);
  if (Literal.hadError())
    return ExprError();

  QualType Ty;
  if (!getLangOptions().CPlusPlus)
    Ty = Context.IntTy;   // 'x' and L'x' -> int in C.
  else if (Literal.isWide())
    Ty = Context.WCharTy; // L'x' -> wchar_t in C++.
  else if (Literal.isMultiChar())
    Ty = Context.IntTy;   // 'wxyz' -> int in C++.
  else
    Ty = Context.CharTy;  // 'x' -> char in C++

  return Owned(new (Context) CharacterLiteral(Literal.getValue(),
                                              Literal.isWide(),
                                              Ty, Tok.getLocation()));
}

ExprResult Sema::ActOnNumericConstant(const Token &Tok) {
  // Fast path for a single digit (which is quite common).  A single digit
  // cannot have a trigraph, escaped newline, radix prefix, or type suffix.
  if (Tok.getLength() == 1) {
    const char Val = PP.getSpellingOfSingleCharacterNumericConstant(Tok);
    unsigned IntSize = Context.Target.getIntWidth();
    return Owned(IntegerLiteral::Create(Context, llvm::APInt(IntSize, Val-'0'),
                    Context.IntTy, Tok.getLocation()));
  }

  llvm::SmallString<512> IntegerBuffer;
  // Add padding so that NumericLiteralParser can overread by one character.
  IntegerBuffer.resize(Tok.getLength()+1);
  const char *ThisTokBegin = &IntegerBuffer[0];

  // Get the spelling of the token, which eliminates trigraphs, etc.
  bool Invalid = false;
  unsigned ActualLength = PP.getSpelling(Tok, ThisTokBegin, &Invalid);
  if (Invalid)
    return ExprError();

  NumericLiteralParser Literal(ThisTokBegin, ThisTokBegin+ActualLength,
                               Tok.getLocation(), PP);
  if (Literal.hadError)
    return ExprError();

  Expr *Res;

  if (Literal.isFloatingLiteral()) {
    QualType Ty;
    if (Literal.isFloat)
      Ty = Context.FloatTy;
    else if (!Literal.isLong)
      Ty = Context.DoubleTy;
    else
      Ty = Context.LongDoubleTy;

    const llvm::fltSemantics &Format = Context.getFloatTypeSemantics(Ty);

    using llvm::APFloat;
    APFloat Val(Format);

    APFloat::opStatus result = Literal.GetFloatValue(Val);

    // Overflow is always an error, but underflow is only an error if
    // we underflowed to zero (APFloat reports denormals as underflow).
    if ((result & APFloat::opOverflow) ||
        ((result & APFloat::opUnderflow) && Val.isZero())) {
      unsigned diagnostic;
      llvm::SmallString<20> buffer;
      if (result & APFloat::opOverflow) {
        diagnostic = diag::warn_float_overflow;
        APFloat::getLargest(Format).toString(buffer);
      } else {
        diagnostic = diag::warn_float_underflow;
        APFloat::getSmallest(Format).toString(buffer);
      }

      Diag(Tok.getLocation(), diagnostic)
        << Ty
        << llvm::StringRef(buffer.data(), buffer.size());
    }

    bool isExact = (result == APFloat::opOK);
    Res = FloatingLiteral::Create(Context, Val, isExact, Ty, Tok.getLocation());

  } else if (!Literal.isIntegerLiteral()) {
    return ExprError();
  } else {
    QualType Ty;

    // long long is a C99 feature.
    if (!getLangOptions().C99 && !getLangOptions().CPlusPlus0x &&
        Literal.isLongLong)
      Diag(Tok.getLocation(), diag::ext_longlong);

    // Get the value in the widest-possible width.
    llvm::APInt ResultVal(Context.Target.getIntMaxTWidth(), 0);

    if (Literal.GetIntegerValue(ResultVal)) {
      // If this value didn't fit into uintmax_t, warn and force to ull.
      Diag(Tok.getLocation(), diag::warn_integer_too_large);
      Ty = Context.UnsignedLongLongTy;
      assert(Context.getTypeSize(Ty) == ResultVal.getBitWidth() &&
             "long long is not intmax_t?");
    } else {
      // If this value fits into a ULL, try to figure out what else it fits into
      // according to the rules of C99 6.4.4.1p5.

      // Octal, Hexadecimal, and integers with a U suffix are allowed to
      // be an unsigned int.
      bool AllowUnsigned = Literal.isUnsigned || Literal.getRadix() != 10;

      // Check from smallest to largest, picking the smallest type we can.
      unsigned Width = 0;
      if (!Literal.isLong && !Literal.isLongLong) {
        // Are int/unsigned possibilities?
        unsigned IntSize = Context.Target.getIntWidth();

        // Does it fit in a unsigned int?
        if (ResultVal.isIntN(IntSize)) {
          // Does it fit in a signed int?
          if (!Literal.isUnsigned && ResultVal[IntSize-1] == 0)
            Ty = Context.IntTy;
          else if (AllowUnsigned)
            Ty = Context.UnsignedIntTy;
          Width = IntSize;
        }
      }

      // Are long/unsigned long possibilities?
      if (Ty.isNull() && !Literal.isLongLong) {
        unsigned LongSize = Context.Target.getLongWidth();

        // Does it fit in a unsigned long?
        if (ResultVal.isIntN(LongSize)) {
          // Does it fit in a signed long?
          if (!Literal.isUnsigned && ResultVal[LongSize-1] == 0)
            Ty = Context.LongTy;
          else if (AllowUnsigned)
            Ty = Context.UnsignedLongTy;
          Width = LongSize;
        }
      }

      // Finally, check long long if needed.
      if (Ty.isNull()) {
        unsigned LongLongSize = Context.Target.getLongLongWidth();

        // Does it fit in a unsigned long long?
        if (ResultVal.isIntN(LongLongSize)) {
          // Does it fit in a signed long long?
          if (!Literal.isUnsigned && ResultVal[LongLongSize-1] == 0)
            Ty = Context.LongLongTy;
          else if (AllowUnsigned)
            Ty = Context.UnsignedLongLongTy;
          Width = LongLongSize;
        }
      }

      // If we still couldn't decide a type, we probably have something that
      // does not fit in a signed long long, but has no U suffix.
      if (Ty.isNull()) {
        Diag(Tok.getLocation(), diag::warn_integer_too_large_for_signed);
        Ty = Context.UnsignedLongLongTy;
        Width = Context.Target.getLongLongWidth();
      }

      if (ResultVal.getBitWidth() != Width)
        ResultVal.trunc(Width);
    }
    Res = IntegerLiteral::Create(Context, ResultVal, Ty, Tok.getLocation());
  }

  // If this is an imaginary literal, create the ImaginaryLiteral wrapper.
  if (Literal.isImaginary)
    Res = new (Context) ImaginaryLiteral(Res,
                                        Context.getComplexType(Res->getType()));

  return Owned(Res);
}

ExprResult Sema::ActOnParenExpr(SourceLocation L,
                                              SourceLocation R, Expr *E) {
  assert((E != 0) && "ActOnParenExpr() missing expr");
  return Owned(new (Context) ParenExpr(L, R, E));
}

/// The UsualUnaryConversions() function is *not* called by this routine.
/// See C99 6.3.2.1p[2-4] for more details.
bool Sema::CheckSizeOfAlignOfOperand(QualType exprType,
                                     SourceLocation OpLoc,
                                     const SourceRange &ExprRange,
                                     bool isSizeof) {
  if (exprType->isDependentType())
    return false;

  // C++ [expr.sizeof]p2: "When applied to a reference or a reference type,
  //   the result is the size of the referenced type."
  // C++ [expr.alignof]p3: "When alignof is applied to a reference type, the
  //   result shall be the alignment of the referenced type."
  if (const ReferenceType *Ref = exprType->getAs<ReferenceType>())
    exprType = Ref->getPointeeType();

  // C99 6.5.3.4p1:
  if (exprType->isFunctionType()) {
    // alignof(function) is allowed as an extension.
    if (isSizeof)
      Diag(OpLoc, diag::ext_sizeof_function_type) << ExprRange;
    return false;
  }

  // Allow sizeof(void)/alignof(void) as an extension.
  if (exprType->isVoidType()) {
    Diag(OpLoc, diag::ext_sizeof_void_type)
      << (isSizeof ? "sizeof" : "__alignof") << ExprRange;
    return false;
  }

  if (RequireCompleteType(OpLoc, exprType,
                          PDiag(diag::err_sizeof_alignof_incomplete_type)
                          << int(!isSizeof) << ExprRange))
    return true;

  // Reject sizeof(interface) and sizeof(interface<proto>) in 64-bit mode.
  if (LangOpts.ObjCNonFragileABI && exprType->isObjCObjectType()) {
    Diag(OpLoc, diag::err_sizeof_nonfragile_interface)
      << exprType << isSizeof << ExprRange;
    return true;
  }

  if (Context.hasSameUnqualifiedType(exprType, Context.OverloadTy)) {
    Diag(OpLoc, diag::err_sizeof_alignof_overloaded_function_type)
      << !isSizeof << ExprRange;
    return true;
  }
  
  return false;
}

bool Sema::CheckAlignOfExpr(Expr *E, SourceLocation OpLoc,
                            const SourceRange &ExprRange) {
  E = E->IgnoreParens();

  // alignof decl is always ok.
  if (isa<DeclRefExpr>(E))
    return false;

  // Cannot know anything else if the expression is dependent.
  if (E->isTypeDependent())
    return false;

  if (E->getBitField()) {
    Diag(OpLoc, diag::err_sizeof_alignof_bitfield) << 1 << ExprRange;
    return true;
  }

  // Alignment of a field access is always okay, so long as it isn't a
  // bit-field.
  if (MemberExpr *ME = dyn_cast<MemberExpr>(E))
    if (isa<FieldDecl>(ME->getMemberDecl()))
      return false;

  return CheckSizeOfAlignOfOperand(E->getType(), OpLoc, ExprRange, false);
}

/// \brief Build a sizeof or alignof expression given a type operand.
ExprResult
Sema::CreateSizeOfAlignOfExpr(TypeSourceInfo *TInfo,
                              SourceLocation OpLoc,
                              bool isSizeOf, SourceRange R) {
  if (!TInfo)
    return ExprError();

  QualType T = TInfo->getType();

  if (!T->isDependentType() &&
      CheckSizeOfAlignOfOperand(T, OpLoc, R, isSizeOf))
    return ExprError();

  // C99 6.5.3.4p4: the type (an unsigned integer type) is size_t.
  return Owned(new (Context) SizeOfAlignOfExpr(isSizeOf, TInfo,
                                               Context.getSizeType(), OpLoc,
                                               R.getEnd()));
}

/// \brief Build a sizeof or alignof expression given an expression
/// operand.
ExprResult
Sema::CreateSizeOfAlignOfExpr(Expr *E, SourceLocation OpLoc,
                              bool isSizeOf, SourceRange R) {
  // Verify that the operand is valid.
  bool isInvalid = false;
  if (E->isTypeDependent()) {
    // Delay type-checking for type-dependent expressions.
  } else if (!isSizeOf) {
    isInvalid = CheckAlignOfExpr(E, OpLoc, R);
  } else if (E->getBitField()) {  // C99 6.5.3.4p1.
    Diag(OpLoc, diag::err_sizeof_alignof_bitfield) << 0;
    isInvalid = true;
  } else {
    isInvalid = CheckSizeOfAlignOfOperand(E->getType(), OpLoc, R, true);
  }

  if (isInvalid)
    return ExprError();

  // C99 6.5.3.4p4: the type (an unsigned integer type) is size_t.
  return Owned(new (Context) SizeOfAlignOfExpr(isSizeOf, E,
                                               Context.getSizeType(), OpLoc,
                                               R.getEnd()));
}

/// ActOnSizeOfAlignOfExpr - Handle @c sizeof(type) and @c sizeof @c expr and
/// the same for @c alignof and @c __alignof
/// Note that the ArgRange is invalid if isType is false.
ExprResult
Sema::ActOnSizeOfAlignOfExpr(SourceLocation OpLoc, bool isSizeof, bool isType,
                             void *TyOrEx, const SourceRange &ArgRange) {
  // If error parsing type, ignore.
  if (TyOrEx == 0) return ExprError();

  if (isType) {
    TypeSourceInfo *TInfo;
    (void) GetTypeFromParser(ParsedType::getFromOpaquePtr(TyOrEx), &TInfo);
    return CreateSizeOfAlignOfExpr(TInfo, OpLoc, isSizeof, ArgRange);
  }

  Expr *ArgEx = (Expr *)TyOrEx;
  ExprResult Result
    = CreateSizeOfAlignOfExpr(ArgEx, OpLoc, isSizeof, ArgEx->getSourceRange());

  return move(Result);
}

QualType Sema::CheckRealImagOperand(Expr *&V, SourceLocation Loc, bool isReal) {
  if (V->isTypeDependent())
    return Context.DependentTy;

  // These operators return the element type of a complex type.
  if (const ComplexType *CT = V->getType()->getAs<ComplexType>())
    return CT->getElementType();

  // Otherwise they pass through real integer and floating point types here.
  if (V->getType()->isArithmeticType())
    return V->getType();

  // Reject anything else.
  Diag(Loc, diag::err_realimag_invalid_type) << V->getType()
    << (isReal ? "__real" : "__imag");
  return QualType();
}



ExprResult
Sema::ActOnPostfixUnaryOp(Scope *S, SourceLocation OpLoc,
                          tok::TokenKind Kind, Expr *Input) {
  UnaryOperatorKind Opc;
  switch (Kind) {
  default: assert(0 && "Unknown unary op!");
  case tok::plusplus:   Opc = UO_PostInc; break;
  case tok::minusminus: Opc = UO_PostDec; break;
  }

  return BuildUnaryOp(S, OpLoc, Opc, Input);
}

ExprResult
Sema::ActOnArraySubscriptExpr(Scope *S, Expr *Base, SourceLocation LLoc,
                              Expr *Idx, SourceLocation RLoc) {
  // Since this might be a postfix expression, get rid of ParenListExprs.
  ExprResult Result = MaybeConvertParenListExprToParenExpr(S, Base);
  if (Result.isInvalid()) return ExprError();
  Base = Result.take();

  Expr *LHSExp = Base, *RHSExp = Idx;

  if (getLangOptions().CPlusPlus &&
      (LHSExp->isTypeDependent() || RHSExp->isTypeDependent())) {
    return Owned(new (Context) ArraySubscriptExpr(LHSExp, RHSExp,
                                                  Context.DependentTy, RLoc));
  }

  if (getLangOptions().CPlusPlus &&
      (LHSExp->getType()->isRecordType() ||
       LHSExp->getType()->isEnumeralType() ||
       RHSExp->getType()->isRecordType() ||
       RHSExp->getType()->isEnumeralType())) {
    return CreateOverloadedArraySubscriptExpr(LLoc, RLoc, Base, Idx);
  }

  return CreateBuiltinArraySubscriptExpr(Base, LLoc, Idx, RLoc);
}


ExprResult
Sema::CreateBuiltinArraySubscriptExpr(Expr *Base, SourceLocation LLoc,
                                     Expr *Idx, SourceLocation RLoc) {
  Expr *LHSExp = Base;
  Expr *RHSExp = Idx;

  // Perform default conversions.
  if (!LHSExp->getType()->getAs<VectorType>())
      DefaultFunctionArrayLvalueConversion(LHSExp);
  DefaultFunctionArrayLvalueConversion(RHSExp);

  QualType LHSTy = LHSExp->getType(), RHSTy = RHSExp->getType();

  // C99 6.5.2.1p2: the expression e1[e2] is by definition precisely equivalent
  // to the expression *((e1)+(e2)). This means the array "Base" may actually be
  // in the subscript position. As a result, we need to derive the array base
  // and index from the expression types.
  Expr *BaseExpr, *IndexExpr;
  QualType ResultType;
  if (LHSTy->isDependentType() || RHSTy->isDependentType()) {
    BaseExpr = LHSExp;
    IndexExpr = RHSExp;
    ResultType = Context.DependentTy;
  } else if (const PointerType *PTy = LHSTy->getAs<PointerType>()) {
    BaseExpr = LHSExp;
    IndexExpr = RHSExp;
    ResultType = PTy->getPointeeType();
  } else if (const PointerType *PTy = RHSTy->getAs<PointerType>()) {
     // Handle the uncommon case of "123[Ptr]".
    BaseExpr = RHSExp;
    IndexExpr = LHSExp;
    ResultType = PTy->getPointeeType();
  } else if (const ObjCObjectPointerType *PTy =
               LHSTy->getAs<ObjCObjectPointerType>()) {
    BaseExpr = LHSExp;
    IndexExpr = RHSExp;
    ResultType = PTy->getPointeeType();
  } else if (const ObjCObjectPointerType *PTy =
               RHSTy->getAs<ObjCObjectPointerType>()) {
     // Handle the uncommon case of "123[Ptr]".
    BaseExpr = RHSExp;
    IndexExpr = LHSExp;
    ResultType = PTy->getPointeeType();
  } else if (const VectorType *VTy = LHSTy->getAs<VectorType>()) {
    BaseExpr = LHSExp;    // vectors: V[123]
    IndexExpr = RHSExp;

    // FIXME: need to deal with const...
    ResultType = VTy->getElementType();
  } else if (LHSTy->isArrayType()) {
    // If we see an array that wasn't promoted by
    // DefaultFunctionArrayLvalueConversion, it must be an array that
    // wasn't promoted because of the C90 rule that doesn't
    // allow promoting non-lvalue arrays.  Warn, then
    // force the promotion here.
    Diag(LHSExp->getLocStart(), diag::ext_subscript_non_lvalue) <<
        LHSExp->getSourceRange();
    ImpCastExprToType(LHSExp, Context.getArrayDecayedType(LHSTy),
                      CK_ArrayToPointerDecay);
    LHSTy = LHSExp->getType();

    BaseExpr = LHSExp;
    IndexExpr = RHSExp;
    ResultType = LHSTy->getAs<PointerType>()->getPointeeType();
  } else if (RHSTy->isArrayType()) {
    // Same as previous, except for 123[f().a] case
    Diag(RHSExp->getLocStart(), diag::ext_subscript_non_lvalue) <<
        RHSExp->getSourceRange();
    ImpCastExprToType(RHSExp, Context.getArrayDecayedType(RHSTy),
                      CK_ArrayToPointerDecay);
    RHSTy = RHSExp->getType();

    BaseExpr = RHSExp;
    IndexExpr = LHSExp;
    ResultType = RHSTy->getAs<PointerType>()->getPointeeType();
  } else {
    return ExprError(Diag(LLoc, diag::err_typecheck_subscript_value)
       << LHSExp->getSourceRange() << RHSExp->getSourceRange());
  }
  // C99 6.5.2.1p1
  if (!IndexExpr->getType()->isIntegerType() && !IndexExpr->isTypeDependent())
    return ExprError(Diag(LLoc, diag::err_typecheck_subscript_not_integer)
                     << IndexExpr->getSourceRange());

  if ((IndexExpr->getType()->isSpecificBuiltinType(BuiltinType::Char_S) ||
       IndexExpr->getType()->isSpecificBuiltinType(BuiltinType::Char_U))
         && !IndexExpr->isTypeDependent())
    Diag(LLoc, diag::warn_subscript_is_char) << IndexExpr->getSourceRange();

  // C99 6.5.2.1p1: "shall have type "pointer to *object* type". Similarly,
  // C++ [expr.sub]p1: The type "T" shall be a completely-defined object
  // type. Note that Functions are not objects, and that (in C99 parlance)
  // incomplete types are not object types.
  if (ResultType->isFunctionType()) {
    Diag(BaseExpr->getLocStart(), diag::err_subscript_function_type)
      << ResultType << BaseExpr->getSourceRange();
    return ExprError();
  }

  if (!ResultType->isDependentType() &&
      RequireCompleteType(LLoc, ResultType,
                          PDiag(diag::err_subscript_incomplete_type)
                            << BaseExpr->getSourceRange()))
    return ExprError();

  // Diagnose bad cases where we step over interface counts.
  if (ResultType->isObjCObjectType() && LangOpts.ObjCNonFragileABI) {
    Diag(LLoc, diag::err_subscript_nonfragile_interface)
      << ResultType << BaseExpr->getSourceRange();
    return ExprError();
  }

  return Owned(new (Context) ArraySubscriptExpr(LHSExp, RHSExp,
                                                ResultType, RLoc));
}

QualType Sema::
CheckExtVectorComponent(QualType baseType, SourceLocation OpLoc,
                        const IdentifierInfo *CompName,
                        SourceLocation CompLoc) {
  // FIXME: Share logic with ExtVectorElementExpr::containsDuplicateElements,
  // see FIXME there.
  //
  // FIXME: This logic can be greatly simplified by splitting it along
  // halving/not halving and reworking the component checking.
  const ExtVectorType *vecType = baseType->getAs<ExtVectorType>();

  // The vector accessor can't exceed the number of elements.
  const char *compStr = CompName->getNameStart();

  // This flag determines whether or not the component is one of the four
  // special names that indicate a subset of exactly half the elements are
  // to be selected.
  bool HalvingSwizzle = false;

  // This flag determines whether or not CompName has an 's' char prefix,
  // indicating that it is a string of hex values to be used as vector indices.
  bool HexSwizzle = *compStr == 's' || *compStr == 'S';

  // Check that we've found one of the special components, or that the component
  // names must come from the same set.
  if (!strcmp(compStr, "hi") || !strcmp(compStr, "lo") ||
      !strcmp(compStr, "even") || !strcmp(compStr, "odd")) {
    HalvingSwizzle = true;
  } else if (vecType->getPointAccessorIdx(*compStr) != -1) {
    do
      compStr++;
    while (*compStr && vecType->getPointAccessorIdx(*compStr) != -1);
  } else if (HexSwizzle || vecType->getNumericAccessorIdx(*compStr) != -1) {
    do
      compStr++;
    while (*compStr && vecType->getNumericAccessorIdx(*compStr) != -1);
  }

  if (!HalvingSwizzle && *compStr) {
    // We didn't get to the end of the string. This means the component names
    // didn't come from the same set *or* we encountered an illegal name.
    Diag(OpLoc, diag::err_ext_vector_component_name_illegal)
      << llvm::StringRef(compStr, 1) << SourceRange(CompLoc);
    return QualType();
  }

  // Ensure no component accessor exceeds the width of the vector type it
  // operates on.
  if (!HalvingSwizzle) {
    compStr = CompName->getNameStart();

    if (HexSwizzle)
      compStr++;

    while (*compStr) {
      if (!vecType->isAccessorWithinNumElements(*compStr++)) {
        Diag(OpLoc, diag::err_ext_vector_component_exceeds_length)
          << baseType << SourceRange(CompLoc);
        return QualType();
      }
    }
  }

  // The component accessor looks fine - now we need to compute the actual type.
  // The vector type is implied by the component accessor. For example,
  // vec4.b is a float, vec4.xy is a vec2, vec4.rgb is a vec3, etc.
  // vec4.s0 is a float, vec4.s23 is a vec3, etc.
  // vec4.hi, vec4.lo, vec4.e, and vec4.o all return vec2.
  unsigned CompSize = HalvingSwizzle ? (vecType->getNumElements() + 1) / 2
                                     : CompName->getLength();
  if (HexSwizzle)
    CompSize--;

  if (CompSize == 1)
    return vecType->getElementType();

  QualType VT = Context.getExtVectorType(vecType->getElementType(), CompSize);
  // Now look up the TypeDefDecl from the vector type. Without this,
  // diagostics look bad. We want extended vector types to appear built-in.
  for (unsigned i = 0, E = ExtVectorDecls.size(); i != E; ++i) {
    if (ExtVectorDecls[i]->getUnderlyingType() == VT)
      return Context.getTypedefType(ExtVectorDecls[i]);
  }
  return VT; // should never get here (a typedef type should always be found).
}

static Decl *FindGetterNameDeclFromProtocolList(const ObjCProtocolDecl*PDecl,
                                                IdentifierInfo *Member,
                                                const Selector &Sel,
                                                ASTContext &Context) {

  if (ObjCPropertyDecl *PD = PDecl->FindPropertyDeclaration(Member))
    return PD;
  if (ObjCMethodDecl *OMD = PDecl->getInstanceMethod(Sel))
    return OMD;

  for (ObjCProtocolDecl::protocol_iterator I = PDecl->protocol_begin(),
       E = PDecl->protocol_end(); I != E; ++I) {
    if (Decl *D = FindGetterNameDeclFromProtocolList(*I, Member, Sel,
                                                     Context))
      return D;
  }
  return 0;
}

static Decl *FindGetterNameDecl(const ObjCObjectPointerType *QIdTy,
                                IdentifierInfo *Member,
                                const Selector &Sel,
                                ASTContext &Context) {
  // Check protocols on qualified interfaces.
  Decl *GDecl = 0;
  for (ObjCObjectPointerType::qual_iterator I = QIdTy->qual_begin(),
       E = QIdTy->qual_end(); I != E; ++I) {
    if (ObjCPropertyDecl *PD = (*I)->FindPropertyDeclaration(Member)) {
      GDecl = PD;
      break;
    }
    // Also must look for a getter name which uses property syntax.
    if (ObjCMethodDecl *OMD = (*I)->getInstanceMethod(Sel)) {
      GDecl = OMD;
      break;
    }
  }
  if (!GDecl) {
    for (ObjCObjectPointerType::qual_iterator I = QIdTy->qual_begin(),
         E = QIdTy->qual_end(); I != E; ++I) {
      // Search in the protocol-qualifier list of current protocol.
      GDecl = FindGetterNameDeclFromProtocolList(*I, Member, Sel, Context);
      if (GDecl)
        return GDecl;
    }
  }
  return GDecl;
}

ExprResult
Sema::ActOnDependentMemberExpr(Expr *BaseExpr, QualType BaseType,
                               bool IsArrow, SourceLocation OpLoc,
                               const CXXScopeSpec &SS,
                               NamedDecl *FirstQualifierInScope,
                               const DeclarationNameInfo &NameInfo,
                               const TemplateArgumentListInfo *TemplateArgs) {
  // Even in dependent contexts, try to diagnose base expressions with
  // obviously wrong types, e.g.:
  //
  // T* t;
  // t.f;
  //
  // In Obj-C++, however, the above expression is valid, since it could be
  // accessing the 'f' property if T is an Obj-C interface. The extra check
  // allows this, while still reporting an error if T is a struct pointer.
  if (!IsArrow) {
    const PointerType *PT = BaseType->getAs<PointerType>();
    if (PT && (!getLangOptions().ObjC1 ||
               PT->getPointeeType()->isRecordType())) {
      assert(BaseExpr && "cannot happen with implicit member accesses");
      Diag(NameInfo.getLoc(), diag::err_typecheck_member_reference_struct_union)
        << BaseType << BaseExpr->getSourceRange();
      return ExprError();
    }
  }

  assert(BaseType->isDependentType() ||
         NameInfo.getName().isDependentName() ||
         isDependentScopeSpecifier(SS));

  // Get the type being accessed in BaseType.  If this is an arrow, the BaseExpr
  // must have pointer type, and the accessed type is the pointee.
  return Owned(CXXDependentScopeMemberExpr::Create(Context, BaseExpr, BaseType,
                                                   IsArrow, OpLoc,
                                                   SS.getScopeRep(),
                                                   SS.getRange(),
                                                   FirstQualifierInScope,
                                                   NameInfo, TemplateArgs));
}

/// We know that the given qualified member reference points only to
/// declarations which do not belong to the static type of the base
/// expression.  Diagnose the problem.
static void DiagnoseQualifiedMemberReference(Sema &SemaRef,
                                             Expr *BaseExpr,
                                             QualType BaseType,
                                             const CXXScopeSpec &SS,
                                             const LookupResult &R) {
  // If this is an implicit member access, use a different set of
  // diagnostics.
  if (!BaseExpr)
    return DiagnoseInstanceReference(SemaRef, SS, R);

  SemaRef.Diag(R.getNameLoc(), diag::err_qualified_member_of_unrelated)
    << SS.getRange() << R.getRepresentativeDecl() << BaseType;
}

// Check whether the declarations we found through a nested-name
// specifier in a member expression are actually members of the base
// type.  The restriction here is:
//
//   C++ [expr.ref]p2:
//     ... In these cases, the id-expression shall name a
//     member of the class or of one of its base classes.
//
// So it's perfectly legitimate for the nested-name specifier to name
// an unrelated class, and for us to find an overload set including
// decls from classes which are not superclasses, as long as the decl
// we actually pick through overload resolution is from a superclass.
bool Sema::CheckQualifiedMemberReference(Expr *BaseExpr,
                                         QualType BaseType,
                                         const CXXScopeSpec &SS,
                                         const LookupResult &R) {
  const RecordType *BaseRT = BaseType->getAs<RecordType>();
  if (!BaseRT) {
    // We can't check this yet because the base type is still
    // dependent.
    assert(BaseType->isDependentType());
    return false;
  }
  CXXRecordDecl *BaseRecord = cast<CXXRecordDecl>(BaseRT->getDecl());

  for (LookupResult::iterator I = R.begin(), E = R.end(); I != E; ++I) {
    // If this is an implicit member reference and we find a
    // non-instance member, it's not an error.
    if (!BaseExpr && !(*I)->isCXXInstanceMember())
      return false;

    // Note that we use the DC of the decl, not the underlying decl.
    DeclContext *DC = (*I)->getDeclContext();
    while (DC->isTransparentContext())
      DC = DC->getParent();

    if (!DC->isRecord())
      continue;
    
    llvm::SmallPtrSet<CXXRecordDecl*,4> MemberRecord;
    MemberRecord.insert(cast<CXXRecordDecl>(DC)->getCanonicalDecl());

    if (!IsProvablyNotDerivedFrom(*this, BaseRecord, MemberRecord))
      return false;
  }

  DiagnoseQualifiedMemberReference(*this, BaseExpr, BaseType, SS, R);
  return true;
}

static bool
LookupMemberExprInRecord(Sema &SemaRef, LookupResult &R,
                         SourceRange BaseRange, const RecordType *RTy,
                         SourceLocation OpLoc, CXXScopeSpec &SS,
                         bool HasTemplateArgs) {
  RecordDecl *RDecl = RTy->getDecl();
  if (SemaRef.RequireCompleteType(OpLoc, QualType(RTy, 0),
                              SemaRef.PDiag(diag::err_typecheck_incomplete_tag)
                                    << BaseRange))
    return true;

  if (HasTemplateArgs) {
    // LookupTemplateName doesn't expect these both to exist simultaneously.
    QualType ObjectType = SS.isSet() ? QualType() : QualType(RTy, 0);

    bool MOUS;
    SemaRef.LookupTemplateName(R, 0, SS, ObjectType, false, MOUS);
    return false;
  }

  DeclContext *DC = RDecl;
  if (SS.isSet()) {
    // If the member name was a qualified-id, look into the
    // nested-name-specifier.
    DC = SemaRef.computeDeclContext(SS, false);

    if (SemaRef.RequireCompleteDeclContext(SS, DC)) {
      SemaRef.Diag(SS.getRange().getEnd(), diag::err_typecheck_incomplete_tag)
        << SS.getRange() << DC;
      return true;
    }

    assert(DC && "Cannot handle non-computable dependent contexts in lookup");

    if (!isa<TypeDecl>(DC)) {
      SemaRef.Diag(R.getNameLoc(), diag::err_qualified_member_nonclass)
        << DC << SS.getRange();
      return true;
    }
  }

  // The record definition is complete, now look up the member.
  SemaRef.LookupQualifiedName(R, DC);

  if (!R.empty())
    return false;

  // We didn't find anything with the given name, so try to correct
  // for typos.
  DeclarationName Name = R.getLookupName();
  if (SemaRef.CorrectTypo(R, 0, &SS, DC, false, Sema::CTC_MemberLookup) &&
      !R.empty() &&
      (isa<ValueDecl>(*R.begin()) || isa<FunctionTemplateDecl>(*R.begin()))) {
    SemaRef.Diag(R.getNameLoc(), diag::err_no_member_suggest)
      << Name << DC << R.getLookupName() << SS.getRange()
      << FixItHint::CreateReplacement(R.getNameLoc(),
                                      R.getLookupName().getAsString());
    if (NamedDecl *ND = R.getAsSingle<NamedDecl>())
      SemaRef.Diag(ND->getLocation(), diag::note_previous_decl)
        << ND->getDeclName();
    return false;
  } else {
    R.clear();
    R.setLookupName(Name);
  }

  return false;
}

ExprResult
Sema::BuildMemberReferenceExpr(Expr *Base, QualType BaseType,
                               SourceLocation OpLoc, bool IsArrow,
                               CXXScopeSpec &SS,
                               NamedDecl *FirstQualifierInScope,
                               const DeclarationNameInfo &NameInfo,
                               const TemplateArgumentListInfo *TemplateArgs) {
  if (BaseType->isDependentType() ||
      (SS.isSet() && isDependentScopeSpecifier(SS)))
    return ActOnDependentMemberExpr(Base, BaseType,
                                    IsArrow, OpLoc,
                                    SS, FirstQualifierInScope,
                                    NameInfo, TemplateArgs);

  LookupResult R(*this, NameInfo, LookupMemberName);

  // Implicit member accesses.
  if (!Base) {
    QualType RecordTy = BaseType;
    if (IsArrow) RecordTy = RecordTy->getAs<PointerType>()->getPointeeType();
    if (LookupMemberExprInRecord(*this, R, SourceRange(),
                                 RecordTy->getAs<RecordType>(),
                                 OpLoc, SS, TemplateArgs != 0))
      return ExprError();

  // Explicit member accesses.
  } else {
    ExprResult Result =
      LookupMemberExpr(R, Base, IsArrow, OpLoc,
                       SS, /*ObjCImpDecl*/ 0, TemplateArgs != 0);

    if (Result.isInvalid()) {
      Owned(Base);
      return ExprError();
    }

    if (Result.get())
      return move(Result);

    // LookupMemberExpr can modify Base, and thus change BaseType
    BaseType = Base->getType();
  }

  return BuildMemberReferenceExpr(Base, BaseType,
                                  OpLoc, IsArrow, SS, FirstQualifierInScope,
                                  R, TemplateArgs);
}

ExprResult
Sema::BuildMemberReferenceExpr(Expr *BaseExpr, QualType BaseExprType,
                               SourceLocation OpLoc, bool IsArrow,
                               const CXXScopeSpec &SS,
                               NamedDecl *FirstQualifierInScope,
                               LookupResult &R,
                         const TemplateArgumentListInfo *TemplateArgs,
                               bool SuppressQualifierCheck) {
  QualType BaseType = BaseExprType;
  if (IsArrow) {
    assert(BaseType->isPointerType());
    BaseType = BaseType->getAs<PointerType>()->getPointeeType();
  }
  R.setBaseObjectType(BaseType);

  NestedNameSpecifier *Qualifier = SS.getScopeRep();
  const DeclarationNameInfo &MemberNameInfo = R.getLookupNameInfo();
  DeclarationName MemberName = MemberNameInfo.getName();
  SourceLocation MemberLoc = MemberNameInfo.getLoc();

  if (R.isAmbiguous())
    return ExprError();

  if (R.empty()) {
    // Rederive where we looked up.
    DeclContext *DC = (SS.isSet()
                       ? computeDeclContext(SS, false)
                       : BaseType->getAs<RecordType>()->getDecl());

    Diag(R.getNameLoc(), diag::err_no_member)
      << MemberName << DC
      << (BaseExpr ? BaseExpr->getSourceRange() : SourceRange());
    return ExprError();
  }

  // Diagnose lookups that find only declarations from a non-base
  // type.  This is possible for either qualified lookups (which may
  // have been qualified with an unrelated type) or implicit member
  // expressions (which were found with unqualified lookup and thus
  // may have come from an enclosing scope).  Note that it's okay for
  // lookup to find declarations from a non-base type as long as those
  // aren't the ones picked by overload resolution.
  if ((SS.isSet() || !BaseExpr ||
       (isa<CXXThisExpr>(BaseExpr) &&
        cast<CXXThisExpr>(BaseExpr)->isImplicit())) &&
      !SuppressQualifierCheck &&
      CheckQualifiedMemberReference(BaseExpr, BaseType, SS, R))
    return ExprError();

  // Construct an unresolved result if we in fact got an unresolved
  // result.
  if (R.isOverloadedResult() || R.isUnresolvableResult()) {
    bool Dependent =
      BaseExprType->isDependentType() ||
      R.isUnresolvableResult() ||
      OverloadExpr::ComputeDependence(R.begin(), R.end(), TemplateArgs);

    // Suppress any lookup-related diagnostics; we'll do these when we
    // pick a member.
    R.suppressDiagnostics();

    UnresolvedMemberExpr *MemExpr
      = UnresolvedMemberExpr::Create(Context, Dependent,
                                     R.isUnresolvableResult(),
                                     BaseExpr, BaseExprType,
                                     IsArrow, OpLoc,
                                     Qualifier, SS.getRange(),
                                     MemberNameInfo,
                                     TemplateArgs, R.begin(), R.end());

    return Owned(MemExpr);
  }

  assert(R.isSingleResult());
  DeclAccessPair FoundDecl = R.begin().getPair();
  NamedDecl *MemberDecl = R.getFoundDecl();

  // FIXME: diagnose the presence of template arguments now.

  // If the decl being referenced had an error, return an error for this
  // sub-expr without emitting another error, in order to avoid cascading
  // error cases.
  if (MemberDecl->isInvalidDecl())
    return ExprError();

  // Handle the implicit-member-access case.
  if (!BaseExpr) {
    // If this is not an instance member, convert to a non-member access.
    if (!MemberDecl->isCXXInstanceMember())
      return BuildDeclarationNameExpr(SS, R.getLookupNameInfo(), MemberDecl);

    SourceLocation Loc = R.getNameLoc();
    if (SS.getRange().isValid())
      Loc = SS.getRange().getBegin();
    BaseExpr = new (Context) CXXThisExpr(Loc, BaseExprType,/*isImplicit=*/true);
  }

  bool ShouldCheckUse = true;
  if (CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(MemberDecl)) {
    // Don't diagnose the use of a virtual member function unless it's
    // explicitly qualified.
    if (MD->isVirtual() && !SS.isSet())
      ShouldCheckUse = false;
  }

  // Check the use of this member.
  if (ShouldCheckUse && DiagnoseUseOfDecl(MemberDecl, MemberLoc)) {
    Owned(BaseExpr);
    return ExprError();
  }

  if (FieldDecl *FD = dyn_cast<FieldDecl>(MemberDecl)) {
    // We may have found a field within an anonymous union or struct
    // (C++ [class.union]).
    if (cast<RecordDecl>(FD->getDeclContext())->isAnonymousStructOrUnion() &&
        !BaseType->getAs<RecordType>()->getDecl()->isAnonymousStructOrUnion())
      return BuildAnonymousStructUnionMemberReference(MemberLoc, FD,
                                                      BaseExpr, OpLoc);

    // Figure out the type of the member; see C99 6.5.2.3p3, C++ [expr.ref]
    QualType MemberType = FD->getType();
    if (const ReferenceType *Ref = MemberType->getAs<ReferenceType>())
      MemberType = Ref->getPointeeType();
    else {
      Qualifiers BaseQuals = BaseType.getQualifiers();
      BaseQuals.removeObjCGCAttr();
      if (FD->isMutable()) BaseQuals.removeConst();

      Qualifiers MemberQuals
        = Context.getCanonicalType(MemberType).getQualifiers();

      Qualifiers Combined = BaseQuals + MemberQuals;
      if (Combined != MemberQuals)
        MemberType = Context.getQualifiedType(MemberType, Combined);
    }

    MarkDeclarationReferenced(MemberLoc, FD);
    if (PerformObjectMemberConversion(BaseExpr, Qualifier, FoundDecl, FD))
      return ExprError();
    return Owned(BuildMemberExpr(Context, BaseExpr, IsArrow, SS,
                                 FD, FoundDecl, MemberNameInfo,
                                 MemberType));
  }

  if (VarDecl *Var = dyn_cast<VarDecl>(MemberDecl)) {
    MarkDeclarationReferenced(MemberLoc, Var);
    return Owned(BuildMemberExpr(Context, BaseExpr, IsArrow, SS,
                                 Var, FoundDecl, MemberNameInfo,
                                 Var->getType().getNonReferenceType()));
  }

  if (FunctionDecl *MemberFn = dyn_cast<FunctionDecl>(MemberDecl)) {
    MarkDeclarationReferenced(MemberLoc, MemberDecl);
    return Owned(BuildMemberExpr(Context, BaseExpr, IsArrow, SS,
                                 MemberFn, FoundDecl, MemberNameInfo,
                                 MemberFn->getType()));
  }

  if (EnumConstantDecl *Enum = dyn_cast<EnumConstantDecl>(MemberDecl)) {
    MarkDeclarationReferenced(MemberLoc, MemberDecl);
    return Owned(BuildMemberExpr(Context, BaseExpr, IsArrow, SS,
                                 Enum, FoundDecl, MemberNameInfo,
                                 Enum->getType()));
  }

  Owned(BaseExpr);

  // We found something that we didn't expect. Complain.
  if (isa<TypeDecl>(MemberDecl))
    Diag(MemberLoc, diag::err_typecheck_member_reference_type)
      << MemberName << BaseType << int(IsArrow);
  else
    Diag(MemberLoc, diag::err_typecheck_member_reference_unknown)
      << MemberName << BaseType << int(IsArrow);

  Diag(MemberDecl->getLocation(), diag::note_member_declared_here)
    << MemberName;
  R.suppressDiagnostics();
  return ExprError();
}

/// Look up the given member of the given non-type-dependent
/// expression.  This can return in one of two ways:
///  * If it returns a sentinel null-but-valid result, the caller will
///    assume that lookup was performed and the results written into
///    the provided structure.  It will take over from there.
///  * Otherwise, the returned expression will be produced in place of
///    an ordinary member expression.
///
/// The ObjCImpDecl bit is a gross hack that will need to be properly
/// fixed for ObjC++.
ExprResult
Sema::LookupMemberExpr(LookupResult &R, Expr *&BaseExpr,
                       bool &IsArrow, SourceLocation OpLoc,
                       CXXScopeSpec &SS,
                       Decl *ObjCImpDecl, bool HasTemplateArgs) {
  assert(BaseExpr && "no base expression");

  // Perform default conversions.
  DefaultFunctionArrayConversion(BaseExpr);

  QualType BaseType = BaseExpr->getType();
  assert(!BaseType->isDependentType());

  DeclarationName MemberName = R.getLookupName();
  SourceLocation MemberLoc = R.getNameLoc();

  // If the user is trying to apply -> or . to a function pointer
  // type, it's probably because they forgot parentheses to call that
  // function. Suggest the addition of those parentheses, build the
  // call, and continue on.
  if (const PointerType *Ptr = BaseType->getAs<PointerType>()) {
    if (const FunctionProtoType *Fun
          = Ptr->getPointeeType()->getAs<FunctionProtoType>()) {
      QualType ResultTy = Fun->getResultType();
      if (Fun->getNumArgs() == 0 &&
          ((!IsArrow && ResultTy->isRecordType()) ||
           (IsArrow && ResultTy->isPointerType() &&
            ResultTy->getAs<PointerType>()->getPointeeType()
                                                          ->isRecordType()))) {
        SourceLocation Loc = PP.getLocForEndOfToken(BaseExpr->getLocEnd());
        Diag(Loc, diag::err_member_reference_needs_call)
          << QualType(Fun, 0)
          << FixItHint::CreateInsertion(Loc, "()");

        ExprResult NewBase
          = ActOnCallExpr(0, BaseExpr, Loc,
                          MultiExprArg(*this, 0, 0), 0, Loc);
        BaseExpr = 0;
        if (NewBase.isInvalid())
          return ExprError();

        BaseExpr = NewBase.takeAs<Expr>();
        DefaultFunctionArrayConversion(BaseExpr);
        BaseType = BaseExpr->getType();
      }
    }
  }

  // If this is an Objective-C pseudo-builtin and a definition is provided then
  // use that.
  if (BaseType->isObjCIdType()) {
    if (IsArrow) {
      // Handle the following exceptional case PObj->isa.
      if (const ObjCObjectPointerType *OPT =
          BaseType->getAs<ObjCObjectPointerType>()) {
        if (OPT->getObjectType()->isObjCId() &&
            MemberName.getAsIdentifierInfo()->isStr("isa"))
          return Owned(new (Context) ObjCIsaExpr(BaseExpr, true, MemberLoc,
                                                 Context.getObjCClassType()));
      }
    }
    // We have an 'id' type. Rather than fall through, we check if this
    // is a reference to 'isa'.
    if (BaseType != Context.ObjCIdRedefinitionType) {
      BaseType = Context.ObjCIdRedefinitionType;
      ImpCastExprToType(BaseExpr, BaseType, CK_BitCast);
    }
  }

  // If this is an Objective-C pseudo-builtin and a definition is provided then
  // use that.
  if (Context.isObjCSelType(BaseType)) {
    // We have an 'SEL' type. Rather than fall through, we check if this
    // is a reference to 'sel_id'.
    if (BaseType != Context.ObjCSelRedefinitionType) {
      BaseType = Context.ObjCSelRedefinitionType;
      ImpCastExprToType(BaseExpr, BaseType, CK_BitCast);
    }
  }

  assert(!BaseType.isNull() && "no type for member expression");

  // Handle properties on ObjC 'Class' types.
  if (!IsArrow && BaseType->isObjCClassType()) {
    // Also must look for a getter name which uses property syntax.
    IdentifierInfo *Member = MemberName.getAsIdentifierInfo();
    Selector Sel = PP.getSelectorTable().getNullarySelector(Member);
    if (ObjCMethodDecl *MD = getCurMethodDecl()) {
      ObjCInterfaceDecl *IFace = MD->getClassInterface();
      ObjCMethodDecl *Getter;
      // FIXME: need to also look locally in the implementation.
      if ((Getter = IFace->lookupClassMethod(Sel))) {
        // Check the use of this method.
        if (DiagnoseUseOfDecl(Getter, MemberLoc))
          return ExprError();
      }
      // If we found a getter then this may be a valid dot-reference, we
      // will look for the matching setter, in case it is needed.
      Selector SetterSel =
      SelectorTable::constructSetterName(PP.getIdentifierTable(),
                                         PP.getSelectorTable(), Member);
      ObjCMethodDecl *Setter = IFace->lookupClassMethod(SetterSel);
      if (!Setter) {
        // If this reference is in an @implementation, also check for 'private'
        // methods.
        Setter = IFace->lookupPrivateInstanceMethod(SetterSel);
      }
      // Look through local category implementations associated with the class.
      if (!Setter)
        Setter = IFace->getCategoryClassMethod(SetterSel);

      if (Setter && DiagnoseUseOfDecl(Setter, MemberLoc))
        return ExprError();

      if (Getter || Setter) {
        QualType PType;

        if (Getter)
          PType = Getter->getSendResultType();
        else
          // Get the expression type from Setter's incoming parameter.
          PType = (*(Setter->param_end() -1))->getType();
        // FIXME: we must check that the setter has property type.
        return Owned(new (Context) ObjCImplicitSetterGetterRefExpr(Getter,
                                                  PType,
                                                  Setter, MemberLoc, BaseExpr));
      }
      return ExprError(Diag(MemberLoc, diag::err_property_not_found)
                       << MemberName << BaseType);
    }
  }

  if (BaseType->isObjCClassType() &&
      BaseType != Context.ObjCClassRedefinitionType) {
    BaseType = Context.ObjCClassRedefinitionType;
    ImpCastExprToType(BaseExpr, BaseType, CK_BitCast);
  }

  if (IsArrow) {
    if (const PointerType *PT = BaseType->getAs<PointerType>())
      BaseType = PT->getPointeeType();
    else if (BaseType->isObjCObjectPointerType())
      ;
    else if (BaseType->isRecordType()) {
      // Recover from arrow accesses to records, e.g.:
      //   struct MyRecord foo;
      //   foo->bar
      // This is actually well-formed in C++ if MyRecord has an
      // overloaded operator->, but that should have been dealt with
      // by now.
      Diag(OpLoc, diag::err_typecheck_member_reference_suggestion)
        << BaseType << int(IsArrow) << BaseExpr->getSourceRange()
        << FixItHint::CreateReplacement(OpLoc, ".");
      IsArrow = false;
    } else {
      Diag(MemberLoc, diag::err_typecheck_member_reference_arrow)
        << BaseType << BaseExpr->getSourceRange();
      return ExprError();
    }
  } else {
    // Recover from dot accesses to pointers, e.g.:
    //   type *foo;
    //   foo.bar
    // This is actually well-formed in two cases:
    //   - 'type' is an Objective C type
    //   - 'bar' is a pseudo-destructor name which happens to refer to
    //     the appropriate pointer type
    if (MemberName.getNameKind() != DeclarationName::CXXDestructorName) {
      const PointerType *PT = BaseType->getAs<PointerType>();
      if (PT && PT->getPointeeType()->isRecordType()) {
        Diag(OpLoc, diag::err_typecheck_member_reference_suggestion)
          << BaseType << int(IsArrow) << BaseExpr->getSourceRange()
          << FixItHint::CreateReplacement(OpLoc, "->");
        BaseType = PT->getPointeeType();
        IsArrow = true;
      }
    }
  }

  // Handle field access to simple records.
  if (const RecordType *RTy = BaseType->getAs<RecordType>()) {
    if (LookupMemberExprInRecord(*this, R, BaseExpr->getSourceRange(),
                                 RTy, OpLoc, SS, HasTemplateArgs))
      return ExprError();
    return Owned((Expr*) 0);
  }

  // Handle access to Objective-C instance variables, such as "Obj->ivar" and
  // (*Obj).ivar.
  if ((IsArrow && BaseType->isObjCObjectPointerType()) ||
      (!IsArrow && BaseType->isObjCObjectType())) {
    const ObjCObjectPointerType *OPT = BaseType->getAs<ObjCObjectPointerType>();
    ObjCInterfaceDecl *IDecl =
      OPT ? OPT->getInterfaceDecl()
          : BaseType->getAs<ObjCObjectType>()->getInterface();
    if (IDecl) {
      IdentifierInfo *Member = MemberName.getAsIdentifierInfo();

      ObjCInterfaceDecl *ClassDeclared;
      ObjCIvarDecl *IV = IDecl->lookupInstanceVariable(Member, ClassDeclared);

      if (!IV) {
        // Attempt to correct for typos in ivar names.
        LookupResult Res(*this, R.getLookupName(), R.getNameLoc(),
                         LookupMemberName);
        if (CorrectTypo(Res, 0, 0, IDecl, false, CTC_MemberLookup) &&
            (IV = Res.getAsSingle<ObjCIvarDecl>())) {
          Diag(R.getNameLoc(),
               diag::err_typecheck_member_reference_ivar_suggest)
            << IDecl->getDeclName() << MemberName << IV->getDeclName()
            << FixItHint::CreateReplacement(R.getNameLoc(),
                                            IV->getNameAsString());
          Diag(IV->getLocation(), diag::note_previous_decl)
            << IV->getDeclName();
        } else {
          Res.clear();
          Res.setLookupName(Member);
        }
      }

      if (IV) {
        // If the decl being referenced had an error, return an error for this
        // sub-expr without emitting another error, in order to avoid cascading
        // error cases.
        if (IV->isInvalidDecl())
          return ExprError();

        // Check whether we can reference this field.
        if (DiagnoseUseOfDecl(IV, MemberLoc))
          return ExprError();
        if (IV->getAccessControl() != ObjCIvarDecl::Public &&
            IV->getAccessControl() != ObjCIvarDecl::Package) {
          ObjCInterfaceDecl *ClassOfMethodDecl = 0;
          if (ObjCMethodDecl *MD = getCurMethodDecl())
            ClassOfMethodDecl =  MD->getClassInterface();
          else if (ObjCImpDecl && getCurFunctionDecl()) {
            // Case of a c-function declared inside an objc implementation.
            // FIXME: For a c-style function nested inside an objc implementation
            // class, there is no implementation context available, so we pass
            // down the context as argument to this routine. Ideally, this context
            // need be passed down in the AST node and somehow calculated from the
            // AST for a function decl.
            if (ObjCImplementationDecl *IMPD =
                dyn_cast<ObjCImplementationDecl>(ObjCImpDecl))
              ClassOfMethodDecl = IMPD->getClassInterface();
            else if (ObjCCategoryImplDecl* CatImplClass =
                        dyn_cast<ObjCCategoryImplDecl>(ObjCImpDecl))
              ClassOfMethodDecl = CatImplClass->getClassInterface();
          }

          if (IV->getAccessControl() == ObjCIvarDecl::Private) {
            if (ClassDeclared != IDecl ||
                ClassOfMethodDecl != ClassDeclared)
              Diag(MemberLoc, diag::error_private_ivar_access)
                << IV->getDeclName();
          } else if (!IDecl->isSuperClassOf(ClassOfMethodDecl))
            // @protected
            Diag(MemberLoc, diag::error_protected_ivar_access)
              << IV->getDeclName();
        }

        return Owned(new (Context) ObjCIvarRefExpr(IV, IV->getType(),
                                                   MemberLoc, BaseExpr,
                                                   IsArrow));
      }
      return ExprError(Diag(MemberLoc, diag::err_typecheck_member_reference_ivar)
                         << IDecl->getDeclName() << MemberName
                         << BaseExpr->getSourceRange());
    }
  }
  // Handle properties on 'id' and qualified "id".
  if (!IsArrow && (BaseType->isObjCIdType() ||
                   BaseType->isObjCQualifiedIdType())) {
    const ObjCObjectPointerType *QIdTy = BaseType->getAs<ObjCObjectPointerType>();
    IdentifierInfo *Member = MemberName.getAsIdentifierInfo();

    // Check protocols on qualified interfaces.
    Selector Sel = PP.getSelectorTable().getNullarySelector(Member);
    if (Decl *PMDecl = FindGetterNameDecl(QIdTy, Member, Sel, Context)) {
      if (ObjCPropertyDecl *PD = dyn_cast<ObjCPropertyDecl>(PMDecl)) {
        // Check the use of this declaration
        if (DiagnoseUseOfDecl(PD, MemberLoc))
          return ExprError();

        return Owned(new (Context) ObjCPropertyRefExpr(PD, PD->getType(),
                                                       MemberLoc, BaseExpr));
      }
      if (ObjCMethodDecl *OMD = dyn_cast<ObjCMethodDecl>(PMDecl)) {
        // Check the use of this method.
        if (DiagnoseUseOfDecl(OMD, MemberLoc))
          return ExprError();

        return Owned(ObjCMessageExpr::Create(Context,
                                             OMD->getSendResultType(),
                                             OpLoc, BaseExpr, Sel,
                                             OMD, NULL, 0, MemberLoc));
      }
    }

    return ExprError(Diag(MemberLoc, diag::err_property_not_found)
                       << MemberName << BaseType);
  }

  // Handle Objective-C property access, which is "Obj.property" where Obj is a
  // pointer to a (potentially qualified) interface type.
  if (!IsArrow)
    if (const ObjCObjectPointerType *OPT =
          BaseType->getAsObjCInterfacePointerType())
      return HandleExprPropertyRefExpr(OPT, BaseExpr, MemberName, MemberLoc);

  // Handle the following exceptional case (*Obj).isa.
  if (!IsArrow &&
      BaseType->isObjCObjectType() &&
      BaseType->getAs<ObjCObjectType>()->isObjCId() &&
      MemberName.getAsIdentifierInfo()->isStr("isa"))
    return Owned(new (Context) ObjCIsaExpr(BaseExpr, false, MemberLoc,
                                           Context.getObjCClassType()));

  // Handle 'field access' to vectors, such as 'V.xx'.
  if (BaseType->isExtVectorType()) {
    IdentifierInfo *Member = MemberName.getAsIdentifierInfo();
    QualType ret = CheckExtVectorComponent(BaseType, OpLoc, Member, MemberLoc);
    if (ret.isNull())
      return ExprError();
    return Owned(new (Context) ExtVectorElementExpr(ret, BaseExpr, *Member,
                                                    MemberLoc));
  }

  Diag(MemberLoc, diag::err_typecheck_member_reference_struct_union)
    << BaseType << BaseExpr->getSourceRange();

  return ExprError();
}

/// The main callback when the parser finds something like
///   expression . [nested-name-specifier] identifier
///   expression -> [nested-name-specifier] identifier
/// where 'identifier' encompasses a fairly broad spectrum of
/// possibilities, including destructor and operator references.
///
/// \param OpKind either tok::arrow or tok::period
/// \param HasTrailingLParen whether the next token is '(', which
///   is used to diagnose mis-uses of special members that can
///   only be called
/// \param ObjCImpDecl the current ObjC @implementation decl;
///   this is an ugly hack around the fact that ObjC @implementations
///   aren't properly put in the context chain
ExprResult Sema::ActOnMemberAccessExpr(Scope *S, Expr *Base,
                                                   SourceLocation OpLoc,
                                                   tok::TokenKind OpKind,
                                                   CXXScopeSpec &SS,
                                                   UnqualifiedId &Id,
                                                   Decl *ObjCImpDecl,
                                                   bool HasTrailingLParen) {
  if (SS.isSet() && SS.isInvalid())
    return ExprError();

  TemplateArgumentListInfo TemplateArgsBuffer;

  // Decompose the name into its component parts.
  DeclarationNameInfo NameInfo;
  const TemplateArgumentListInfo *TemplateArgs;
  DecomposeUnqualifiedId(*this, Id, TemplateArgsBuffer,
                         NameInfo, TemplateArgs);

  DeclarationName Name = NameInfo.getName();
  bool IsArrow = (OpKind == tok::arrow);

  NamedDecl *FirstQualifierInScope
    = (!SS.isSet() ? 0 : FindFirstQualifierInScope(S,
                       static_cast<NestedNameSpecifier*>(SS.getScopeRep())));

  // This is a postfix expression, so get rid of ParenListExprs.
  ExprResult Result = MaybeConvertParenListExprToParenExpr(S, Base);
  if (Result.isInvalid()) return ExprError();
  Base = Result.take();

  if (Base->getType()->isDependentType() || Name.isDependentName() ||
      isDependentScopeSpecifier(SS)) {
    Result = ActOnDependentMemberExpr(Base, Base->getType(),
                                      IsArrow, OpLoc,
                                      SS, FirstQualifierInScope,
                                      NameInfo, TemplateArgs);
  } else {
    LookupResult R(*this, NameInfo, LookupMemberName);
    Result = LookupMemberExpr(R, Base, IsArrow, OpLoc,
                              SS, ObjCImpDecl, TemplateArgs != 0);

    if (Result.isInvalid()) {
      Owned(Base);
      return ExprError();
    }

    if (Result.get()) {
      // The only way a reference to a destructor can be used is to
      // immediately call it, which falls into this case.  If the
      // next token is not a '(', produce a diagnostic and build the
      // call now.
      if (!HasTrailingLParen &&
          Id.getKind() == UnqualifiedId::IK_DestructorName)
        return DiagnoseDtorReference(NameInfo.getLoc(), Result.get());

      return move(Result);
    }

    Result = BuildMemberReferenceExpr(Base, Base->getType(),
                                      OpLoc, IsArrow, SS, FirstQualifierInScope,
                                      R, TemplateArgs);
  }

  return move(Result);
}

ExprResult Sema::BuildCXXDefaultArgExpr(SourceLocation CallLoc,
                                                    FunctionDecl *FD,
                                                    ParmVarDecl *Param) {
  if (Param->hasUnparsedDefaultArg()) {
    Diag (CallLoc,
          diag::err_use_of_default_argument_to_function_declared_later) <<
      FD << cast<CXXRecordDecl>(FD->getDeclContext())->getDeclName();
    Diag(UnparsedDefaultArgLocs[Param],
          diag::note_default_argument_declared_here);
  } else {
    if (Param->hasUninstantiatedDefaultArg()) {
      Expr *UninstExpr = Param->getUninstantiatedDefaultArg();

      // Instantiate the expression.
      MultiLevelTemplateArgumentList ArgList
        = getTemplateInstantiationArgs(FD, 0, /*RelativeToPrimary=*/true);

      std::pair<const TemplateArgument *, unsigned> Innermost 
        = ArgList.getInnermost();
      InstantiatingTemplate Inst(*this, CallLoc, Param, Innermost.first,
                                 Innermost.second);

      ExprResult Result = SubstExpr(UninstExpr, ArgList);
      if (Result.isInvalid())
        return ExprError();

      // Check the expression as an initializer for the parameter.
      InitializedEntity Entity
        = InitializedEntity::InitializeParameter(Param);
      InitializationKind Kind
        = InitializationKind::CreateCopy(Param->getLocation(),
               /*FIXME:EqualLoc*/UninstExpr->getSourceRange().getBegin());
      Expr *ResultE = Result.takeAs<Expr>();

      InitializationSequence InitSeq(*this, Entity, Kind, &ResultE, 1);
      Result = InitSeq.Perform(*this, Entity, Kind,
                               MultiExprArg(*this, &ResultE, 1));
      if (Result.isInvalid())
        return ExprError();

      // Build the default argument expression.
      return Owned(CXXDefaultArgExpr::Create(Context, CallLoc, Param,
                                             Result.takeAs<Expr>()));
    }

    // If the default expression creates temporaries, we need to
    // push them to the current stack of expression temporaries so they'll
    // be properly destroyed.
    // FIXME: We should really be rebuilding the default argument with new
    // bound temporaries; see the comment in PR5810.
    for (unsigned i = 0, e = Param->getNumDefaultArgTemporaries(); i != e; ++i)
      ExprTemporaries.push_back(Param->getDefaultArgTemporary(i));
  }

  // We already type-checked the argument, so we know it works.
  return Owned(CXXDefaultArgExpr::Create(Context, CallLoc, Param));
}

/// ConvertArgumentsForCall - Converts the arguments specified in
/// Args/NumArgs to the parameter types of the function FDecl with
/// function prototype Proto. Call is the call expression itself, and
/// Fn is the function expression. For a C++ member function, this
/// routine does not attempt to convert the object argument. Returns
/// true if the call is ill-formed.
bool
Sema::ConvertArgumentsForCall(CallExpr *Call, Expr *Fn,
                              FunctionDecl *FDecl,
                              const FunctionProtoType *Proto,
                              Expr **Args, unsigned NumArgs,
                              SourceLocation RParenLoc) {
  // C99 6.5.2.2p7 - the arguments are implicitly converted, as if by
  // assignment, to the types of the corresponding parameter, ...
  unsigned NumArgsInProto = Proto->getNumArgs();
  bool Invalid = false;

  // If too few arguments are available (and we don't have default
  // arguments for the remaining parameters), don't make the call.
  if (NumArgs < NumArgsInProto) {
    if (!FDecl || NumArgs < FDecl->getMinRequiredArguments())
      return Diag(RParenLoc, diag::err_typecheck_call_too_few_args)
        << Fn->getType()->isBlockPointerType()
        << NumArgsInProto << NumArgs << Fn->getSourceRange();
    Call->setNumArgs(Context, NumArgsInProto);
  }

  // If too many are passed and not variadic, error on the extras and drop
  // them.
  if (NumArgs > NumArgsInProto) {
    if (!Proto->isVariadic()) {
      Diag(Args[NumArgsInProto]->getLocStart(),
           diag::err_typecheck_call_too_many_args)
        << Fn->getType()->isBlockPointerType()
        << NumArgsInProto << NumArgs << Fn->getSourceRange()
        << SourceRange(Args[NumArgsInProto]->getLocStart(),
                       Args[NumArgs-1]->getLocEnd());
      // This deletes the extra arguments.
      Call->setNumArgs(Context, NumArgsInProto);
      return true;
    }
  }
  llvm::SmallVector<Expr *, 8> AllArgs;
  VariadicCallType CallType =
    Proto->isVariadic() ? VariadicFunction : VariadicDoesNotApply;
  if (Fn->getType()->isBlockPointerType())
    CallType = VariadicBlock; // Block
  else if (isa<MemberExpr>(Fn))
    CallType = VariadicMethod;
  Invalid = GatherArgumentsForCall(Call->getSourceRange().getBegin(), FDecl,
                                   Proto, 0, Args, NumArgs, AllArgs, CallType);
  if (Invalid)
    return true;
  unsigned TotalNumArgs = AllArgs.size();
  for (unsigned i = 0; i < TotalNumArgs; ++i)
    Call->setArg(i, AllArgs[i]);

  return false;
}

bool Sema::GatherArgumentsForCall(SourceLocation CallLoc,
                                  FunctionDecl *FDecl,
                                  const FunctionProtoType *Proto,
                                  unsigned FirstProtoArg,
                                  Expr **Args, unsigned NumArgs,
                                  llvm::SmallVector<Expr *, 8> &AllArgs,
                                  VariadicCallType CallType) {
  unsigned NumArgsInProto = Proto->getNumArgs();
  unsigned NumArgsToCheck = NumArgs;
  bool Invalid = false;
  if (NumArgs != NumArgsInProto)
    // Use default arguments for missing arguments
    NumArgsToCheck = NumArgsInProto;
  unsigned ArgIx = 0;
  // Continue to check argument types (even if we have too few/many args).
  for (unsigned i = FirstProtoArg; i != NumArgsToCheck; i++) {
    QualType ProtoArgType = Proto->getArgType(i);

    Expr *Arg;
    if (ArgIx < NumArgs) {
      Arg = Args[ArgIx++];

      if (RequireCompleteType(Arg->getSourceRange().getBegin(),
                              ProtoArgType,
                              PDiag(diag::err_call_incomplete_argument)
                              << Arg->getSourceRange()))
        return true;

      // Pass the argument
      ParmVarDecl *Param = 0;
      if (FDecl && i < FDecl->getNumParams())
        Param = FDecl->getParamDecl(i);


      InitializedEntity Entity =
        Param? InitializedEntity::InitializeParameter(Param)
             : InitializedEntity::InitializeParameter(ProtoArgType);
      ExprResult ArgE = PerformCopyInitialization(Entity,
                                                        SourceLocation(),
                                                        Owned(Arg));
      if (ArgE.isInvalid())
        return true;

      Arg = ArgE.takeAs<Expr>();
    } else {
      ParmVarDecl *Param = FDecl->getParamDecl(i);

      ExprResult ArgExpr =
        BuildCXXDefaultArgExpr(CallLoc, FDecl, Param);
      if (ArgExpr.isInvalid())
        return true;

      Arg = ArgExpr.takeAs<Expr>();
    }
    AllArgs.push_back(Arg);
  }

  // If this is a variadic call, handle args passed through "...".
  if (CallType != VariadicDoesNotApply) {
    // Promote the arguments (C99 6.5.2.2p7).
    for (unsigned i = ArgIx; i != NumArgs; ++i) {
      Expr *Arg = Args[i];
      Invalid |= DefaultVariadicArgumentPromotion(Arg, CallType, FDecl);
      AllArgs.push_back(Arg);
    }
  }
  return Invalid;
}

/// ActOnCallExpr - Handle a call to Fn with the specified array of arguments.
/// This provides the location of the left/right parens and a list of comma
/// locations.
ExprResult
Sema::ActOnCallExpr(Scope *S, Expr *Fn, SourceLocation LParenLoc,
                    MultiExprArg args,
                    SourceLocation *CommaLocs, SourceLocation RParenLoc) {
  unsigned NumArgs = args.size();

  // Since this might be a postfix expression, get rid of ParenListExprs.
  ExprResult Result = MaybeConvertParenListExprToParenExpr(S, Fn);
  if (Result.isInvalid()) return ExprError();
  Fn = Result.take();

  Expr **Args = args.release();

  if (getLangOptions().CPlusPlus) {
    // If this is a pseudo-destructor expression, build the call immediately.
    if (isa<CXXPseudoDestructorExpr>(Fn)) {
      if (NumArgs > 0) {
        // Pseudo-destructor calls should not have any arguments.
        Diag(Fn->getLocStart(), diag::err_pseudo_dtor_call_with_args)
          << FixItHint::CreateRemoval(
                                    SourceRange(Args[0]->getLocStart(),
                                                Args[NumArgs-1]->getLocEnd()));

        NumArgs = 0;
      }

      return Owned(new (Context) CallExpr(Context, Fn, 0, 0, Context.VoidTy,
                                          RParenLoc));
    }

    // Determine whether this is a dependent call inside a C++ template,
    // in which case we won't do any semantic analysis now.
    // FIXME: Will need to cache the results of name lookup (including ADL) in
    // Fn.
    bool Dependent = false;
    if (Fn->isTypeDependent())
      Dependent = true;
    else if (Expr::hasAnyTypeDependentArguments(Args, NumArgs))
      Dependent = true;

    if (Dependent)
      return Owned(new (Context) CallExpr(Context, Fn, Args, NumArgs,
                                          Context.DependentTy, RParenLoc));

    // Determine whether this is a call to an object (C++ [over.call.object]).
    if (Fn->getType()->isRecordType())
      return Owned(BuildCallToObjectOfClassType(S, Fn, LParenLoc, Args, NumArgs,
                                                CommaLocs, RParenLoc));

    Expr *NakedFn = Fn->IgnoreParens();

    // Determine whether this is a call to an unresolved member function.
    if (UnresolvedMemberExpr *MemE = dyn_cast<UnresolvedMemberExpr>(NakedFn)) {
      // If lookup was unresolved but not dependent (i.e. didn't find
      // an unresolved using declaration), it has to be an overloaded
      // function set, which means it must contain either multiple
      // declarations (all methods or method templates) or a single
      // method template.
      assert((MemE->getNumDecls() > 1) ||
             isa<FunctionTemplateDecl>(
                                 (*MemE->decls_begin())->getUnderlyingDecl()));
      (void)MemE;

      return BuildCallToMemberFunction(S, Fn, LParenLoc, Args, NumArgs,
                                       CommaLocs, RParenLoc);
    }

    // Determine whether this is a call to a member function.
    if (MemberExpr *MemExpr = dyn_cast<MemberExpr>(NakedFn)) {
      NamedDecl *MemDecl = MemExpr->getMemberDecl();
      if (isa<CXXMethodDecl>(MemDecl))
        return BuildCallToMemberFunction(S, Fn, LParenLoc, Args, NumArgs,
                                         CommaLocs, RParenLoc);
    }

    // Determine whether this is a call to a pointer-to-member function.
    if (BinaryOperator *BO = dyn_cast<BinaryOperator>(NakedFn)) {
      if (BO->getOpcode() == BO_PtrMemD ||
          BO->getOpcode() == BO_PtrMemI) {
        if (const FunctionProtoType *FPT
                                = BO->getType()->getAs<FunctionProtoType>()) {
          QualType ResultTy = FPT->getCallResultType(Context);

          CXXMemberCallExpr *TheCall
            = new (Context) CXXMemberCallExpr(Context, BO, Args,
                                              NumArgs, ResultTy,
                                              RParenLoc);

          if (CheckCallReturnType(FPT->getResultType(),
                                  BO->getRHS()->getSourceRange().getBegin(),
                                  TheCall, 0))
            return ExprError();

          if (ConvertArgumentsForCall(TheCall, BO, 0, FPT, Args, NumArgs,
                                      RParenLoc))
            return ExprError();

          return MaybeBindToTemporary(TheCall);
        }
        return ExprError(Diag(Fn->getLocStart(),
                              diag::err_typecheck_call_not_function)
                              << Fn->getType() << Fn->getSourceRange());
      }
    }
  }

  // If we're directly calling a function, get the appropriate declaration.
  // Also, in C++, keep track of whether we should perform argument-dependent
  // lookup and whether there were any explicitly-specified template arguments.

  Expr *NakedFn = Fn->IgnoreParens();
  if (isa<UnresolvedLookupExpr>(NakedFn)) {
    UnresolvedLookupExpr *ULE = cast<UnresolvedLookupExpr>(NakedFn);
    return BuildOverloadedCallExpr(S, Fn, ULE, LParenLoc, Args, NumArgs,
                                   CommaLocs, RParenLoc);
  }

  NamedDecl *NDecl = 0;
  if (isa<DeclRefExpr>(NakedFn))
    NDecl = cast<DeclRefExpr>(NakedFn)->getDecl();

  return BuildResolvedCallExpr(Fn, NDecl, LParenLoc, Args, NumArgs, RParenLoc);
}

/// BuildResolvedCallExpr - Build a call to a resolved expression,
/// i.e. an expression not of \p OverloadTy.  The expression should
/// unary-convert to an expression of function-pointer or
/// block-pointer type.
///
/// \param NDecl the declaration being called, if available
ExprResult
Sema::BuildResolvedCallExpr(Expr *Fn, NamedDecl *NDecl,
                            SourceLocation LParenLoc,
                            Expr **Args, unsigned NumArgs,
                            SourceLocation RParenLoc) {
  FunctionDecl *FDecl = dyn_cast_or_null<FunctionDecl>(NDecl);

  // Promote the function operand.
  UsualUnaryConversions(Fn);

  // Make the call expr early, before semantic checks.  This guarantees cleanup
  // of arguments and function on error.
  CallExpr *TheCall = new (Context) CallExpr(Context, Fn,
                                             Args, NumArgs,
                                             Context.BoolTy,
                                             RParenLoc);

  const FunctionType *FuncT;
  if (!Fn->getType()->isBlockPointerType()) {
    // C99 6.5.2.2p1 - "The expression that denotes the called function shall
    // have type pointer to function".
    const PointerType *PT = Fn->getType()->getAs<PointerType>();
    if (PT == 0)
      return ExprError(Diag(LParenLoc, diag::err_typecheck_call_not_function)
        << Fn->getType() << Fn->getSourceRange());
    FuncT = PT->getPointeeType()->getAs<FunctionType>();
  } else { // This is a block call.
    FuncT = Fn->getType()->getAs<BlockPointerType>()->getPointeeType()->
                getAs<FunctionType>();
  }
  if (FuncT == 0)
    return ExprError(Diag(LParenLoc, diag::err_typecheck_call_not_function)
      << Fn->getType() << Fn->getSourceRange());

  // Check for a valid return type
  if (CheckCallReturnType(FuncT->getResultType(),
                          Fn->getSourceRange().getBegin(), TheCall,
                          FDecl))
    return ExprError();

  // We know the result type of the call, set it.
  TheCall->setType(FuncT->getCallResultType(Context));

  if (const FunctionProtoType *Proto = dyn_cast<FunctionProtoType>(FuncT)) {
    if (ConvertArgumentsForCall(TheCall, Fn, FDecl, Proto, Args, NumArgs,
                                RParenLoc))
      return ExprError();
  } else {
    assert(isa<FunctionNoProtoType>(FuncT) && "Unknown FunctionType!");

    if (FDecl) {
      // Check if we have too few/too many template arguments, based
      // on our knowledge of the function definition.
      const FunctionDecl *Def = 0;
      if (FDecl->hasBody(Def) && NumArgs != Def->param_size()) {
        const FunctionProtoType *Proto =
            Def->getType()->getAs<FunctionProtoType>();
        if (!Proto || !(Proto->isVariadic() && NumArgs >= Def->param_size())) {
          Diag(RParenLoc, diag::warn_call_wrong_number_of_arguments)
            << (NumArgs > Def->param_size()) << FDecl << Fn->getSourceRange();
        }
      }
    }

    // Promote the arguments (C99 6.5.2.2p6).
    for (unsigned i = 0; i != NumArgs; i++) {
      Expr *Arg = Args[i];
      DefaultArgumentPromotion(Arg);
      if (RequireCompleteType(Arg->getSourceRange().getBegin(),
                              Arg->getType(),
                              PDiag(diag::err_call_incomplete_argument)
                                << Arg->getSourceRange()))
        return ExprError();
      TheCall->setArg(i, Arg);
    }
  }

  if (CXXMethodDecl *Method = dyn_cast_or_null<CXXMethodDecl>(FDecl))
    if (!Method->isStatic())
      return ExprError(Diag(LParenLoc, diag::err_member_call_without_object)
        << Fn->getSourceRange());

  // Check for sentinels
  if (NDecl)
    DiagnoseSentinelCalls(NDecl, LParenLoc, Args, NumArgs);

  // Do special checking on direct calls to functions.
  if (FDecl) {
    if (CheckFunctionCall(FDecl, TheCall))
      return ExprError();

    if (unsigned BuiltinID = FDecl->getBuiltinID())
      return CheckBuiltinFunctionCall(BuiltinID, TheCall);
  } else if (NDecl) {
    if (CheckBlockCall(NDecl, TheCall))
      return ExprError();
  }

  return MaybeBindToTemporary(TheCall);
}

ExprResult
Sema::ActOnCompoundLiteral(SourceLocation LParenLoc, ParsedType Ty,
                           SourceLocation RParenLoc, Expr *InitExpr) {
  assert((Ty != 0) && "ActOnCompoundLiteral(): missing type");
  // FIXME: put back this assert when initializers are worked out.
  //assert((InitExpr != 0) && "ActOnCompoundLiteral(): missing expression");

  TypeSourceInfo *TInfo;
  QualType literalType = GetTypeFromParser(Ty, &TInfo);
  if (!TInfo)
    TInfo = Context.getTrivialTypeSourceInfo(literalType);

  return BuildCompoundLiteralExpr(LParenLoc, TInfo, RParenLoc, InitExpr);
}

ExprResult
Sema::BuildCompoundLiteralExpr(SourceLocation LParenLoc, TypeSourceInfo *TInfo,
                               SourceLocation RParenLoc, Expr *literalExpr) {
  QualType literalType = TInfo->getType();

  if (literalType->isArrayType()) {
    if (literalType->isVariableArrayType())
      return ExprError(Diag(LParenLoc, diag::err_variable_object_no_init)
        << SourceRange(LParenLoc, literalExpr->getSourceRange().getEnd()));
  } else if (!literalType->isDependentType() &&
             RequireCompleteType(LParenLoc, literalType,
                      PDiag(diag::err_typecheck_decl_incomplete_type)
                        << SourceRange(LParenLoc,
                                       literalExpr->getSourceRange().getEnd())))
    return ExprError();

  InitializedEntity Entity
    = InitializedEntity::InitializeTemporary(literalType);
  InitializationKind Kind
    = InitializationKind::CreateCast(SourceRange(LParenLoc, RParenLoc),
                                     /*IsCStyleCast=*/true);
  InitializationSequence InitSeq(*this, Entity, Kind, &literalExpr, 1);
  ExprResult Result = InitSeq.Perform(*this, Entity, Kind,
                                       MultiExprArg(*this, &literalExpr, 1),
                                            &literalType);
  if (Result.isInvalid())
    return ExprError();
  literalExpr = Result.get();

  bool isFileScope = getCurFunctionOrMethodDecl() == 0;
  if (isFileScope) { // 6.5.2.5p3
    if (CheckForConstantInitializer(literalExpr, literalType))
      return ExprError();
  }

  return Owned(new (Context) CompoundLiteralExpr(LParenLoc, TInfo, literalType,
                                                 literalExpr, isFileScope));
}

ExprResult
Sema::ActOnInitList(SourceLocation LBraceLoc, MultiExprArg initlist,
                    SourceLocation RBraceLoc) {
  unsigned NumInit = initlist.size();
  Expr **InitList = initlist.release();

  // Semantic analysis for initializers is done by ActOnDeclarator() and
  // CheckInitializer() - it requires knowledge of the object being intialized.

  InitListExpr *E = new (Context) InitListExpr(Context, LBraceLoc, InitList,
                                               NumInit, RBraceLoc);
  E->setType(Context.VoidTy); // FIXME: just a place holder for now.
  return Owned(E);
}

static CastKind getScalarCastKind(ASTContext &Context,
                                            QualType SrcTy, QualType DestTy) {
  if (Context.hasSameUnqualifiedType(SrcTy, DestTy))
    return CK_NoOp;

  if (SrcTy->hasPointerRepresentation()) {
    if (DestTy->hasPointerRepresentation())
      return DestTy->isObjCObjectPointerType() ?
                CK_AnyPointerToObjCPointerCast :
                CK_BitCast;
    if (DestTy->isIntegerType())
      return CK_PointerToIntegral;
  }

  if (SrcTy->isIntegerType()) {
    if (DestTy->isIntegerType())
      return CK_IntegralCast;
    if (DestTy->hasPointerRepresentation())
      return CK_IntegralToPointer;
    if (DestTy->isRealFloatingType())
      return CK_IntegralToFloating;
  }

  if (SrcTy->isRealFloatingType()) {
    if (DestTy->isRealFloatingType())
      return CK_FloatingCast;
    if (DestTy->isIntegerType())
      return CK_FloatingToIntegral;
  }

  // FIXME: Assert here.
  // assert(false && "Unhandled cast combination!");
  return CK_Unknown;
}

/// CheckCastTypes - Check type constraints for casting between types.
bool Sema::CheckCastTypes(SourceRange TyR, QualType castType, Expr *&castExpr,
                          CastKind& Kind,
                          CXXCastPath &BasePath,
                          bool FunctionalStyle) {
  if (getLangOptions().CPlusPlus)
    return CXXCheckCStyleCast(TyR, castType, castExpr, Kind, BasePath,
                              FunctionalStyle);

  DefaultFunctionArrayLvalueConversion(castExpr);

  // C99 6.5.4p2: the cast type needs to be void or scalar and the expression
  // type needs to be scalar.
  if (castType->isVoidType()) {
    // Cast to void allows any expr type.
    Kind = CK_ToVoid;
    return false;
  }

  if (RequireCompleteType(TyR.getBegin(), castType,
                          diag::err_typecheck_cast_to_incomplete))
    return true;

  if (!castType->isScalarType() && !castType->isVectorType()) {
    if (Context.hasSameUnqualifiedType(castType, castExpr->getType()) &&
        (castType->isStructureType() || castType->isUnionType())) {
      // GCC struct/union extension: allow cast to self.
      // FIXME: Check that the cast destination type is complete.
      Diag(TyR.getBegin(), diag::ext_typecheck_cast_nonscalar)
        << castType << castExpr->getSourceRange();
      Kind = CK_NoOp;
      return false;
    }

    if (castType->isUnionType()) {
      // GCC cast to union extension
      RecordDecl *RD = castType->getAs<RecordType>()->getDecl();
      RecordDecl::field_iterator Field, FieldEnd;
      for (Field = RD->field_begin(), FieldEnd = RD->field_end();
           Field != FieldEnd; ++Field) {
        if (Context.hasSameUnqualifiedType(Field->getType(),
                                           castExpr->getType())) {
          Diag(TyR.getBegin(), diag::ext_typecheck_cast_to_union)
            << castExpr->getSourceRange();
          break;
        }
      }
      if (Field == FieldEnd)
        return Diag(TyR.getBegin(), diag::err_typecheck_cast_to_union_no_type)
          << castExpr->getType() << castExpr->getSourceRange();
      Kind = CK_ToUnion;
      return false;
    }

    // Reject any other conversions to non-scalar types.
    return Diag(TyR.getBegin(), diag::err_typecheck_cond_expect_scalar)
      << castType << castExpr->getSourceRange();
  }

  if (!castExpr->getType()->isScalarType() &&
      !castExpr->getType()->isVectorType()) {
    return Diag(castExpr->getLocStart(),
                diag::err_typecheck_expect_scalar_operand)
      << castExpr->getType() << castExpr->getSourceRange();
  }

  if (castType->isExtVectorType())
    return CheckExtVectorCast(TyR, castType, castExpr, Kind);

  if (castType->isVectorType())
    return CheckVectorCast(TyR, castType, castExpr->getType(), Kind);
  if (castExpr->getType()->isVectorType())
    return CheckVectorCast(TyR, castExpr->getType(), castType, Kind);

  if (isa<ObjCSelectorExpr>(castExpr))
    return Diag(castExpr->getLocStart(), diag::err_cast_selector_expr);

  if (!castType->isArithmeticType()) {
    QualType castExprType = castExpr->getType();
    if (!castExprType->isIntegralType(Context) && 
        castExprType->isArithmeticType())
      return Diag(castExpr->getLocStart(),
                  diag::err_cast_pointer_from_non_pointer_int)
        << castExprType << castExpr->getSourceRange();
  } else if (!castExpr->getType()->isArithmeticType()) {
    if (!castType->isIntegralType(Context) && castType->isArithmeticType())
      return Diag(castExpr->getLocStart(),
                  diag::err_cast_pointer_to_non_pointer_int)
        << castType << castExpr->getSourceRange();
  }

  Kind = getScalarCastKind(Context, castExpr->getType(), castType);

  if (Kind == CK_Unknown || Kind == CK_BitCast)
    CheckCastAlign(castExpr, castType, TyR);

  return false;
}

bool Sema::CheckVectorCast(SourceRange R, QualType VectorTy, QualType Ty,
                           CastKind &Kind) {
  assert(VectorTy->isVectorType() && "Not a vector type!");

  if (Ty->isVectorType() || Ty->isIntegerType()) {
    if (Context.getTypeSize(VectorTy) != Context.getTypeSize(Ty))
      return Diag(R.getBegin(),
                  Ty->isVectorType() ?
                  diag::err_invalid_conversion_between_vectors :
                  diag::err_invalid_conversion_between_vector_and_integer)
        << VectorTy << Ty << R;
  } else
    return Diag(R.getBegin(),
                diag::err_invalid_conversion_between_vector_and_scalar)
      << VectorTy << Ty << R;

  Kind = CK_BitCast;
  return false;
}

bool Sema::CheckExtVectorCast(SourceRange R, QualType DestTy, Expr *&CastExpr,
                              CastKind &Kind) {
  assert(DestTy->isExtVectorType() && "Not an extended vector type!");

  QualType SrcTy = CastExpr->getType();

  // If SrcTy is a VectorType, the total size must match to explicitly cast to
  // an ExtVectorType.
  if (SrcTy->isVectorType()) {
    if (Context.getTypeSize(DestTy) != Context.getTypeSize(SrcTy))
      return Diag(R.getBegin(),diag::err_invalid_conversion_between_ext_vectors)
        << DestTy << SrcTy << R;
    Kind = CK_BitCast;
    return false;
  }

  // All non-pointer scalars can be cast to ExtVector type.  The appropriate
  // conversion will take place first from scalar to elt type, and then
  // splat from elt type to vector.
  if (SrcTy->isPointerType())
    return Diag(R.getBegin(),
                diag::err_invalid_conversion_between_vector_and_scalar)
      << DestTy << SrcTy << R;

  QualType DestElemTy = DestTy->getAs<ExtVectorType>()->getElementType();
  ImpCastExprToType(CastExpr, DestElemTy,
                    getScalarCastKind(Context, SrcTy, DestElemTy));

  Kind = CK_VectorSplat;
  return false;
}

ExprResult
Sema::ActOnCastExpr(Scope *S, SourceLocation LParenLoc, ParsedType Ty,
                    SourceLocation RParenLoc, Expr *castExpr) {
  assert((Ty != 0) && (castExpr != 0) &&
         "ActOnCastExpr(): missing type or expr");

  TypeSourceInfo *castTInfo;
  QualType castType = GetTypeFromParser(Ty, &castTInfo);
  if (!castTInfo)
    castTInfo = Context.getTrivialTypeSourceInfo(castType);

  // If the Expr being casted is a ParenListExpr, handle it specially.
  if (isa<ParenListExpr>(castExpr))
    return ActOnCastOfParenListExpr(S, LParenLoc, RParenLoc, castExpr,
                                    castTInfo);

  return BuildCStyleCastExpr(LParenLoc, castTInfo, RParenLoc, castExpr);
}

ExprResult
Sema::BuildCStyleCastExpr(SourceLocation LParenLoc, TypeSourceInfo *Ty,
                          SourceLocation RParenLoc, Expr *castExpr) {
  CastKind Kind = CK_Unknown;
  CXXCastPath BasePath;
  if (CheckCastTypes(SourceRange(LParenLoc, RParenLoc), Ty->getType(), castExpr,
                     Kind, BasePath))
    return ExprError();

  return Owned(CStyleCastExpr::Create(Context,
                                    Ty->getType().getNonLValueExprType(Context),
                                      Kind, castExpr, &BasePath, Ty,
                                      LParenLoc, RParenLoc));
}

/// This is not an AltiVec-style cast, so turn the ParenListExpr into a sequence
/// of comma binary operators.
ExprResult
Sema::MaybeConvertParenListExprToParenExpr(Scope *S, Expr *expr) {
  ParenListExpr *E = dyn_cast<ParenListExpr>(expr);
  if (!E)
    return Owned(expr);

  ExprResult Result(E->getExpr(0));

  for (unsigned i = 1, e = E->getNumExprs(); i != e && !Result.isInvalid(); ++i)
    Result = ActOnBinOp(S, E->getExprLoc(), tok::comma, Result.get(),
                        E->getExpr(i));

  if (Result.isInvalid()) return ExprError();

  return ActOnParenExpr(E->getLParenLoc(), E->getRParenLoc(), Result.get());
}

ExprResult
Sema::ActOnCastOfParenListExpr(Scope *S, SourceLocation LParenLoc,
                               SourceLocation RParenLoc, Expr *Op,
                               TypeSourceInfo *TInfo) {
  ParenListExpr *PE = cast<ParenListExpr>(Op);
  QualType Ty = TInfo->getType();
  bool isAltiVecLiteral = false;

  // Check for an altivec literal,
  // i.e. all the elements are integer constants.
  if (getLangOptions().AltiVec && Ty->isVectorType()) {
    if (PE->getNumExprs() == 0) {
      Diag(PE->getExprLoc(), diag::err_altivec_empty_initializer);
      return ExprError();
    }
    if (PE->getNumExprs() == 1) {
      if (!PE->getExpr(0)->getType()->isVectorType())
        isAltiVecLiteral = true;
    }
    else
      isAltiVecLiteral = true;
  }

  // If this is an altivec initializer, '(' type ')' '(' init, ..., init ')'
  // then handle it as such.
  if (isAltiVecLiteral) {
    llvm::SmallVector<Expr *, 8> initExprs;
    for (unsigned i = 0, e = PE->getNumExprs(); i != e; ++i)
      initExprs.push_back(PE->getExpr(i));

    // FIXME: This means that pretty-printing the final AST will produce curly
    // braces instead of the original commas.
    InitListExpr *E = new (Context) InitListExpr(Context, LParenLoc,
                                                 &initExprs[0],
                                                 initExprs.size(), RParenLoc);
    E->setType(Ty);
    return BuildCompoundLiteralExpr(LParenLoc, TInfo, RParenLoc, E);
  } else {
    // This is not an AltiVec-style cast, so turn the ParenListExpr into a
    // sequence of BinOp comma operators.
    ExprResult Result = MaybeConvertParenListExprToParenExpr(S, Op);
    if (Result.isInvalid()) return ExprError();
    return BuildCStyleCastExpr(LParenLoc, TInfo, RParenLoc, Result.take());
  }
}

ExprResult Sema::ActOnParenOrParenListExpr(SourceLocation L,
                                                  SourceLocation R,
                                                  MultiExprArg Val,
                                                  ParsedType TypeOfCast) {
  unsigned nexprs = Val.size();
  Expr **exprs = reinterpret_cast<Expr**>(Val.release());
  assert((exprs != 0) && "ActOnParenOrParenListExpr() missing expr list");
  Expr *expr;
  if (nexprs == 1 && TypeOfCast && !TypeIsVectorType(TypeOfCast))
    expr = new (Context) ParenExpr(L, R, exprs[0]);
  else
    expr = new (Context) ParenListExpr(Context, L, exprs, nexprs, R);
  return Owned(expr);
}

/// Note that lhs is not null here, even if this is the gnu "x ?: y" extension.
/// In that case, lhs = cond.
/// C99 6.5.15
QualType Sema::CheckConditionalOperands(Expr *&Cond, Expr *&LHS, Expr *&RHS,
                                        SourceLocation QuestionLoc) {
  // C++ is sufficiently different to merit its own checker.
  if (getLangOptions().CPlusPlus)
    return CXXCheckConditionalOperands(Cond, LHS, RHS, QuestionLoc);

  UsualUnaryConversions(Cond);
  UsualUnaryConversions(LHS);
  UsualUnaryConversions(RHS);
  QualType CondTy = Cond->getType();
  QualType LHSTy = LHS->getType();
  QualType RHSTy = RHS->getType();

  // first, check the condition.
  if (!CondTy->isScalarType()) { // C99 6.5.15p2
    Diag(Cond->getLocStart(), diag::err_typecheck_cond_expect_scalar)
      << CondTy;
    return QualType();
  }

  // Now check the two expressions.
  if (LHSTy->isVectorType() || RHSTy->isVectorType())
    return CheckVectorOperands(QuestionLoc, LHS, RHS);

  // If both operands have arithmetic type, do the usual arithmetic conversions
  // to find a common type: C99 6.5.15p3,5.
  if (LHSTy->isArithmeticType() && RHSTy->isArithmeticType()) {
    UsualArithmeticConversions(LHS, RHS);
    return LHS->getType();
  }

  // If both operands are the same structure or union type, the result is that
  // type.
  if (const RecordType *LHSRT = LHSTy->getAs<RecordType>()) {    // C99 6.5.15p3
    if (const RecordType *RHSRT = RHSTy->getAs<RecordType>())
      if (LHSRT->getDecl() == RHSRT->getDecl())
        // "If both the operands have structure or union type, the result has
        // that type."  This implies that CV qualifiers are dropped.
        return LHSTy.getUnqualifiedType();
    // FIXME: Type of conditional expression must be complete in C mode.
  }

  // C99 6.5.15p5: "If both operands have void type, the result has void type."
  // The following || allows only one side to be void (a GCC-ism).
  if (LHSTy->isVoidType() || RHSTy->isVoidType()) {
    if (!LHSTy->isVoidType())
      Diag(RHS->getLocStart(), diag::ext_typecheck_cond_one_void)
        << RHS->getSourceRange();
    if (!RHSTy->isVoidType())
      Diag(LHS->getLocStart(), diag::ext_typecheck_cond_one_void)
        << LHS->getSourceRange();
    ImpCastExprToType(LHS, Context.VoidTy, CK_ToVoid);
    ImpCastExprToType(RHS, Context.VoidTy, CK_ToVoid);
    return Context.VoidTy;
  }
  // C99 6.5.15p6 - "if one operand is a null pointer constant, the result has
  // the type of the other operand."
  if ((LHSTy->isAnyPointerType() || LHSTy->isBlockPointerType()) &&
      RHS->isNullPointerConstant(Context, Expr::NPC_ValueDependentIsNull)) {
    // promote the null to a pointer.
    ImpCastExprToType(RHS, LHSTy, CK_Unknown);
    return LHSTy;
  }
  if ((RHSTy->isAnyPointerType() || RHSTy->isBlockPointerType()) &&
      LHS->isNullPointerConstant(Context, Expr::NPC_ValueDependentIsNull)) {
    ImpCastExprToType(LHS, RHSTy, CK_Unknown);
    return RHSTy;
  }

  // All objective-c pointer type analysis is done here.
  QualType compositeType = FindCompositeObjCPointerType(LHS, RHS,
                                                        QuestionLoc);
  if (!compositeType.isNull())
    return compositeType;


  // Handle block pointer types.
  if (LHSTy->isBlockPointerType() || RHSTy->isBlockPointerType()) {
    if (!LHSTy->isBlockPointerType() || !RHSTy->isBlockPointerType()) {
      if (LHSTy->isVoidPointerType() || RHSTy->isVoidPointerType()) {
        QualType destType = Context.getPointerType(Context.VoidTy);
        ImpCastExprToType(LHS, destType, CK_BitCast);
        ImpCastExprToType(RHS, destType, CK_BitCast);
        return destType;
      }
      Diag(QuestionLoc, diag::err_typecheck_cond_incompatible_operands)
      << LHSTy << RHSTy << LHS->getSourceRange() << RHS->getSourceRange();
      return QualType();
    }
    // We have 2 block pointer types.
    if (Context.getCanonicalType(LHSTy) == Context.getCanonicalType(RHSTy)) {
      // Two identical block pointer types are always compatible.
      return LHSTy;
    }
    // The block pointer types aren't identical, continue checking.
    QualType lhptee = LHSTy->getAs<BlockPointerType>()->getPointeeType();
    QualType rhptee = RHSTy->getAs<BlockPointerType>()->getPointeeType();

    if (!Context.typesAreCompatible(lhptee.getUnqualifiedType(),
                                    rhptee.getUnqualifiedType())) {
      Diag(QuestionLoc, diag::warn_typecheck_cond_incompatible_pointers)
      << LHSTy << RHSTy << LHS->getSourceRange() << RHS->getSourceRange();
      // In this situation, we assume void* type. No especially good
      // reason, but this is what gcc does, and we do have to pick
      // to get a consistent AST.
      QualType incompatTy = Context.getPointerType(Context.VoidTy);
      ImpCastExprToType(LHS, incompatTy, CK_BitCast);
      ImpCastExprToType(RHS, incompatTy, CK_BitCast);
      return incompatTy;
    }
    // The block pointer types are compatible.
    ImpCastExprToType(LHS, LHSTy, CK_BitCast);
    ImpCastExprToType(RHS, LHSTy, CK_BitCast);
    return LHSTy;
  }

  // Check constraints for C object pointers types (C99 6.5.15p3,6).
  if (LHSTy->isPointerType() && RHSTy->isPointerType()) {
    // get the "pointed to" types
    QualType lhptee = LHSTy->getAs<PointerType>()->getPointeeType();
    QualType rhptee = RHSTy->getAs<PointerType>()->getPointeeType();

    // ignore qualifiers on void (C99 6.5.15p3, clause 6)
    if (lhptee->isVoidType() && rhptee->isIncompleteOrObjectType()) {
      // Figure out necessary qualifiers (C99 6.5.15p6)
      QualType destPointee
        = Context.getQualifiedType(lhptee, rhptee.getQualifiers());
      QualType destType = Context.getPointerType(destPointee);
      // Add qualifiers if necessary.
      ImpCastExprToType(LHS, destType, CK_NoOp);
      // Promote to void*.
      ImpCastExprToType(RHS, destType, CK_BitCast);
      return destType;
    }
    if (rhptee->isVoidType() && lhptee->isIncompleteOrObjectType()) {
      QualType destPointee
        = Context.getQualifiedType(rhptee, lhptee.getQualifiers());
      QualType destType = Context.getPointerType(destPointee);
      // Add qualifiers if necessary.
      ImpCastExprToType(RHS, destType, CK_NoOp);
      // Promote to void*.
      ImpCastExprToType(LHS, destType, CK_BitCast);
      return destType;
    }

    if (Context.getCanonicalType(LHSTy) == Context.getCanonicalType(RHSTy)) {
      // Two identical pointer types are always compatible.
      return LHSTy;
    }
    if (!Context.typesAreCompatible(lhptee.getUnqualifiedType(),
                                    rhptee.getUnqualifiedType())) {
      Diag(QuestionLoc, diag::warn_typecheck_cond_incompatible_pointers)
        << LHSTy << RHSTy << LHS->getSourceRange() << RHS->getSourceRange();
      // In this situation, we assume void* type. No especially good
      // reason, but this is what gcc does, and we do have to pick
      // to get a consistent AST.
      QualType incompatTy = Context.getPointerType(Context.VoidTy);
      ImpCastExprToType(LHS, incompatTy, CK_BitCast);
      ImpCastExprToType(RHS, incompatTy, CK_BitCast);
      return incompatTy;
    }
    // The pointer types are compatible.
    // C99 6.5.15p6: If both operands are pointers to compatible types *or* to
    // differently qualified versions of compatible types, the result type is
    // a pointer to an appropriately qualified version of the *composite*
    // type.
    // FIXME: Need to calculate the composite type.
    // FIXME: Need to add qualifiers
    ImpCastExprToType(LHS, LHSTy, CK_BitCast);
    ImpCastExprToType(RHS, LHSTy, CK_BitCast);
    return LHSTy;
  }

  // GCC compatibility: soften pointer/integer mismatch.
  if (RHSTy->isPointerType() && LHSTy->isIntegerType()) {
    Diag(QuestionLoc, diag::warn_typecheck_cond_pointer_integer_mismatch)
      << LHSTy << RHSTy << LHS->getSourceRange() << RHS->getSourceRange();
    ImpCastExprToType(LHS, RHSTy, CK_IntegralToPointer);
    return RHSTy;
  }
  if (LHSTy->isPointerType() && RHSTy->isIntegerType()) {
    Diag(QuestionLoc, diag::warn_typecheck_cond_pointer_integer_mismatch)
      << LHSTy << RHSTy << LHS->getSourceRange() << RHS->getSourceRange();
    ImpCastExprToType(RHS, LHSTy, CK_IntegralToPointer);
    return LHSTy;
  }

  // Otherwise, the operands are not compatible.
  Diag(QuestionLoc, diag::err_typecheck_cond_incompatible_operands)
    << LHSTy << RHSTy << LHS->getSourceRange() << RHS->getSourceRange();
  return QualType();
}

/// FindCompositeObjCPointerType - Helper method to find composite type of
/// two objective-c pointer types of the two input expressions.
QualType Sema::FindCompositeObjCPointerType(Expr *&LHS, Expr *&RHS,
                                        SourceLocation QuestionLoc) {
  QualType LHSTy = LHS->getType();
  QualType RHSTy = RHS->getType();

  // Handle things like Class and struct objc_class*.  Here we case the result
  // to the pseudo-builtin, because that will be implicitly cast back to the
  // redefinition type if an attempt is made to access its fields.
  if (LHSTy->isObjCClassType() &&
      (RHSTy.getDesugaredType() == Context.ObjCClassRedefinitionType)) {
    ImpCastExprToType(RHS, LHSTy, CK_BitCast);
    return LHSTy;
  }
  if (RHSTy->isObjCClassType() &&
      (LHSTy.getDesugaredType() == Context.ObjCClassRedefinitionType)) {
    ImpCastExprToType(LHS, RHSTy, CK_BitCast);
    return RHSTy;
  }
  // And the same for struct objc_object* / id
  if (LHSTy->isObjCIdType() &&
      (RHSTy.getDesugaredType() == Context.ObjCIdRedefinitionType)) {
    ImpCastExprToType(RHS, LHSTy, CK_BitCast);
    return LHSTy;
  }
  if (RHSTy->isObjCIdType() &&
      (LHSTy.getDesugaredType() == Context.ObjCIdRedefinitionType)) {
    ImpCastExprToType(LHS, RHSTy, CK_BitCast);
    return RHSTy;
  }
  // And the same for struct objc_selector* / SEL
  if (Context.isObjCSelType(LHSTy) &&
      (RHSTy.getDesugaredType() == Context.ObjCSelRedefinitionType)) {
    ImpCastExprToType(RHS, LHSTy, CK_BitCast);
    return LHSTy;
  }
  if (Context.isObjCSelType(RHSTy) &&
      (LHSTy.getDesugaredType() == Context.ObjCSelRedefinitionType)) {
    ImpCastExprToType(LHS, RHSTy, CK_BitCast);
    return RHSTy;
  }
  // Check constraints for Objective-C object pointers types.
  if (LHSTy->isObjCObjectPointerType() && RHSTy->isObjCObjectPointerType()) {

    if (Context.getCanonicalType(LHSTy) == Context.getCanonicalType(RHSTy)) {
      // Two identical object pointer types are always compatible.
      return LHSTy;
    }
    const ObjCObjectPointerType *LHSOPT = LHSTy->getAs<ObjCObjectPointerType>();
    const ObjCObjectPointerType *RHSOPT = RHSTy->getAs<ObjCObjectPointerType>();
    QualType compositeType = LHSTy;

    // If both operands are interfaces and either operand can be
    // assigned to the other, use that type as the composite
    // type. This allows
    //   xxx ? (A*) a : (B*) b
    // where B is a subclass of A.
    //
    // Additionally, as for assignment, if either type is 'id'
    // allow silent coercion. Finally, if the types are
    // incompatible then make sure to use 'id' as the composite
    // type so the result is acceptable for sending messages to.

    // FIXME: Consider unifying with 'areComparableObjCPointerTypes'.
    // It could return the composite type.
    if (Context.canAssignObjCInterfaces(LHSOPT, RHSOPT)) {
      compositeType = RHSOPT->isObjCBuiltinType() ? RHSTy : LHSTy;
    } else if (Context.canAssignObjCInterfaces(RHSOPT, LHSOPT)) {
      compositeType = LHSOPT->isObjCBuiltinType() ? LHSTy : RHSTy;
    } else if ((LHSTy->isObjCQualifiedIdType() ||
                RHSTy->isObjCQualifiedIdType()) &&
               Context.ObjCQualifiedIdTypesAreCompatible(LHSTy, RHSTy, true)) {
      // Need to handle "id<xx>" explicitly.
      // GCC allows qualified id and any Objective-C type to devolve to
      // id. Currently localizing to here until clear this should be
      // part of ObjCQualifiedIdTypesAreCompatible.
      compositeType = Context.getObjCIdType();
    } else if (LHSTy->isObjCIdType() || RHSTy->isObjCIdType()) {
      compositeType = Context.getObjCIdType();
    } else if (!(compositeType =
                 Context.areCommonBaseCompatible(LHSOPT, RHSOPT)).isNull())
      ;
    else {
      Diag(QuestionLoc, diag::ext_typecheck_cond_incompatible_operands)
      << LHSTy << RHSTy
      << LHS->getSourceRange() << RHS->getSourceRange();
      QualType incompatTy = Context.getObjCIdType();
      ImpCastExprToType(LHS, incompatTy, CK_BitCast);
      ImpCastExprToType(RHS, incompatTy, CK_BitCast);
      return incompatTy;
    }
    // The object pointer types are compatible.
    ImpCastExprToType(LHS, compositeType, CK_BitCast);
    ImpCastExprToType(RHS, compositeType, CK_BitCast);
    return compositeType;
  }
  // Check Objective-C object pointer types and 'void *'
  if (LHSTy->isVoidPointerType() && RHSTy->isObjCObjectPointerType()) {
    QualType lhptee = LHSTy->getAs<PointerType>()->getPointeeType();
    QualType rhptee = RHSTy->getAs<ObjCObjectPointerType>()->getPointeeType();
    QualType destPointee
    = Context.getQualifiedType(lhptee, rhptee.getQualifiers());
    QualType destType = Context.getPointerType(destPointee);
    // Add qualifiers if necessary.
    ImpCastExprToType(LHS, destType, CK_NoOp);
    // Promote to void*.
    ImpCastExprToType(RHS, destType, CK_BitCast);
    return destType;
  }
  if (LHSTy->isObjCObjectPointerType() && RHSTy->isVoidPointerType()) {
    QualType lhptee = LHSTy->getAs<ObjCObjectPointerType>()->getPointeeType();
    QualType rhptee = RHSTy->getAs<PointerType>()->getPointeeType();
    QualType destPointee
    = Context.getQualifiedType(rhptee, lhptee.getQualifiers());
    QualType destType = Context.getPointerType(destPointee);
    // Add qualifiers if necessary.
    ImpCastExprToType(RHS, destType, CK_NoOp);
    // Promote to void*.
    ImpCastExprToType(LHS, destType, CK_BitCast);
    return destType;
  }
  return QualType();
}

/// ActOnConditionalOp - Parse a ?: operation.  Note that 'LHS' may be null
/// in the case of a the GNU conditional expr extension.
ExprResult Sema::ActOnConditionalOp(SourceLocation QuestionLoc,
                                                  SourceLocation ColonLoc,
                                                  Expr *CondExpr, Expr *LHSExpr,
                                                  Expr *RHSExpr) {
  // If this is the gnu "x ?: y" extension, analyze the types as though the LHS
  // was the condition.
  bool isLHSNull = LHSExpr == 0;
  Expr *SAVEExpr = 0;
  if (isLHSNull) {
    LHSExpr = SAVEExpr = CondExpr;
  }

  QualType result = CheckConditionalOperands(CondExpr, LHSExpr,
                                             RHSExpr, QuestionLoc);
  if (result.isNull())
    return ExprError();

  return Owned(new (Context) ConditionalOperator(CondExpr, QuestionLoc,
                                                 LHSExpr, ColonLoc, 
                                                 RHSExpr, SAVEExpr,
                                                 result));
}

// CheckPointerTypesForAssignment - This is a very tricky routine (despite
// being closely modeled after the C99 spec:-). The odd characteristic of this
// routine is it effectively iqnores the qualifiers on the top level pointee.
// This circumvents the usual type rules specified in 6.2.7p1 & 6.7.5.[1-3].
// FIXME: add a couple examples in this comment.
Sema::AssignConvertType
Sema::CheckPointerTypesForAssignment(QualType lhsType, QualType rhsType) {
  QualType lhptee, rhptee;

  if ((lhsType->isObjCClassType() &&
       (rhsType.getDesugaredType() == Context.ObjCClassRedefinitionType)) ||
     (rhsType->isObjCClassType() &&
       (lhsType.getDesugaredType() == Context.ObjCClassRedefinitionType))) {
      return Compatible;
  }

  // get the "pointed to" type (ignoring qualifiers at the top level)
  lhptee = lhsType->getAs<PointerType>()->getPointeeType();
  rhptee = rhsType->getAs<PointerType>()->getPointeeType();

  // make sure we operate on the canonical type
  lhptee = Context.getCanonicalType(lhptee);
  rhptee = Context.getCanonicalType(rhptee);

  AssignConvertType ConvTy = Compatible;

  // C99 6.5.16.1p1: This following citation is common to constraints
  // 3 & 4 (below). ...and the type *pointed to* by the left has all the
  // qualifiers of the type *pointed to* by the right;
  // FIXME: Handle ExtQualType
  if (!lhptee.isAtLeastAsQualifiedAs(rhptee))
    ConvTy = CompatiblePointerDiscardsQualifiers;

  // C99 6.5.16.1p1 (constraint 4): If one operand is a pointer to an object or
  // incomplete type and the other is a pointer to a qualified or unqualified
  // version of void...
  if (lhptee->isVoidType()) {
    if (rhptee->isIncompleteOrObjectType())
      return ConvTy;

    // As an extension, we allow cast to/from void* to function pointer.
    assert(rhptee->isFunctionType());
    return FunctionVoidPointer;
  }

  if (rhptee->isVoidType()) {
    if (lhptee->isIncompleteOrObjectType())
      return ConvTy;

    // As an extension, we allow cast to/from void* to function pointer.
    assert(lhptee->isFunctionType());
    return FunctionVoidPointer;
  }
  // C99 6.5.16.1p1 (constraint 3): both operands are pointers to qualified or
  // unqualified versions of compatible types, ...
  lhptee = lhptee.getUnqualifiedType();
  rhptee = rhptee.getUnqualifiedType();
  if (!Context.typesAreCompatible(lhptee, rhptee)) {
    // Check if the pointee types are compatible ignoring the sign.
    // We explicitly check for char so that we catch "char" vs
    // "unsigned char" on systems where "char" is unsigned.
    if (lhptee->isCharType())
      lhptee = Context.UnsignedCharTy;
    else if (lhptee->hasSignedIntegerRepresentation())
      lhptee = Context.getCorrespondingUnsignedType(lhptee);

    if (rhptee->isCharType())
      rhptee = Context.UnsignedCharTy;
    else if (rhptee->hasSignedIntegerRepresentation())
      rhptee = Context.getCorrespondingUnsignedType(rhptee);

    if (lhptee == rhptee) {
      // Types are compatible ignoring the sign. Qualifier incompatibility
      // takes priority over sign incompatibility because the sign
      // warning can be disabled.
      if (ConvTy != Compatible)
        return ConvTy;
      return IncompatiblePointerSign;
    }

    // If we are a multi-level pointer, it's possible that our issue is simply
    // one of qualification - e.g. char ** -> const char ** is not allowed. If
    // the eventual target type is the same and the pointers have the same
    // level of indirection, this must be the issue.
    if (lhptee->isPointerType() && rhptee->isPointerType()) {
      do {
        lhptee = lhptee->getAs<PointerType>()->getPointeeType();
        rhptee = rhptee->getAs<PointerType>()->getPointeeType();

        lhptee = Context.getCanonicalType(lhptee);
        rhptee = Context.getCanonicalType(rhptee);
      } while (lhptee->isPointerType() && rhptee->isPointerType());

      if (Context.hasSameUnqualifiedType(lhptee, rhptee))
        return IncompatibleNestedPointerQualifiers;
    }

    // General pointer incompatibility takes priority over qualifiers.
    return IncompatiblePointer;
  }
  return ConvTy;
}

/// CheckBlockPointerTypesForAssignment - This routine determines whether two
/// block pointer types are compatible or whether a block and normal pointer
/// are compatible. It is more restrict than comparing two function pointer
// types.
Sema::AssignConvertType
Sema::CheckBlockPointerTypesForAssignment(QualType lhsType,
                                          QualType rhsType) {
  QualType lhptee, rhptee;

  // get the "pointed to" type (ignoring qualifiers at the top level)
  lhptee = lhsType->getAs<BlockPointerType>()->getPointeeType();
  rhptee = rhsType->getAs<BlockPointerType>()->getPointeeType();

  // make sure we operate on the canonical type
  lhptee = Context.getCanonicalType(lhptee);
  rhptee = Context.getCanonicalType(rhptee);

  AssignConvertType ConvTy = Compatible;

  // For blocks we enforce that qualifiers are identical.
  if (lhptee.getLocalCVRQualifiers() != rhptee.getLocalCVRQualifiers())
    ConvTy = CompatiblePointerDiscardsQualifiers;

  if (!getLangOptions().CPlusPlus) {
    if (!Context.typesAreBlockPointerCompatible(lhsType, rhsType))
      return IncompatibleBlockPointer;
  }
  else if (!Context.typesAreCompatible(lhptee, rhptee))
    return IncompatibleBlockPointer;
  return ConvTy;
}

/// CheckObjCPointerTypesForAssignment - Compares two objective-c pointer types
/// for assignment compatibility.
Sema::AssignConvertType
Sema::CheckObjCPointerTypesForAssignment(QualType lhsType, QualType rhsType) {
  if (lhsType->isObjCBuiltinType()) {
    // Class is not compatible with ObjC object pointers.
    if (lhsType->isObjCClassType() && !rhsType->isObjCBuiltinType() &&
        !rhsType->isObjCQualifiedClassType())
      return IncompatiblePointer;
    return Compatible;
  }
  if (rhsType->isObjCBuiltinType()) {
    // Class is not compatible with ObjC object pointers.
    if (rhsType->isObjCClassType() && !lhsType->isObjCBuiltinType() &&
        !lhsType->isObjCQualifiedClassType())
      return IncompatiblePointer;
    return Compatible;
  }
  QualType lhptee =
  lhsType->getAs<ObjCObjectPointerType>()->getPointeeType();
  QualType rhptee =
  rhsType->getAs<ObjCObjectPointerType>()->getPointeeType();
  // make sure we operate on the canonical type
  lhptee = Context.getCanonicalType(lhptee);
  rhptee = Context.getCanonicalType(rhptee);
  if (!lhptee.isAtLeastAsQualifiedAs(rhptee))
    return CompatiblePointerDiscardsQualifiers;

  if (Context.typesAreCompatible(lhsType, rhsType))
    return Compatible;
  if (lhsType->isObjCQualifiedIdType() || rhsType->isObjCQualifiedIdType())
    return IncompatibleObjCQualifiedId;
  return IncompatiblePointer;
}

/// CheckAssignmentConstraints (C99 6.5.16) - This routine currently
/// has code to accommodate several GCC extensions when type checking
/// pointers. Here are some objectionable examples that GCC considers warnings:
///
///  int a, *pint;
///  short *pshort;
///  struct foo *pfoo;
///
///  pint = pshort; // warning: assignment from incompatible pointer type
///  a = pint; // warning: assignment makes integer from pointer without a cast
///  pint = a; // warning: assignment makes pointer from integer without a cast
///  pint = pfoo; // warning: assignment from incompatible pointer type
///
/// As a result, the code for dealing with pointers is more complex than the
/// C99 spec dictates.
///
Sema::AssignConvertType
Sema::CheckAssignmentConstraints(QualType lhsType, QualType rhsType) {
  // Get canonical types.  We're not formatting these types, just comparing
  // them.
  lhsType = Context.getCanonicalType(lhsType).getUnqualifiedType();
  rhsType = Context.getCanonicalType(rhsType).getUnqualifiedType();

  if (lhsType == rhsType)
    return Compatible; // Common case: fast path an exact match.

  if ((lhsType->isObjCClassType() &&
       (rhsType.getDesugaredType() == Context.ObjCClassRedefinitionType)) ||
     (rhsType->isObjCClassType() &&
       (lhsType.getDesugaredType() == Context.ObjCClassRedefinitionType))) {
      return Compatible;
  }

  // If the left-hand side is a reference type, then we are in a
  // (rare!) case where we've allowed the use of references in C,
  // e.g., as a parameter type in a built-in function. In this case,
  // just make sure that the type referenced is compatible with the
  // right-hand side type. The caller is responsible for adjusting
  // lhsType so that the resulting expression does not have reference
  // type.
  if (const ReferenceType *lhsTypeRef = lhsType->getAs<ReferenceType>()) {
    if (Context.typesAreCompatible(lhsTypeRef->getPointeeType(), rhsType))
      return Compatible;
    return Incompatible;
  }
  // Allow scalar to ExtVector assignments, and assignments of an ExtVector type
  // to the same ExtVector type.
  if (lhsType->isExtVectorType()) {
    if (rhsType->isExtVectorType())
      return lhsType == rhsType ? Compatible : Incompatible;
    if (rhsType->isArithmeticType())
      return Compatible;
  }

  if (lhsType->isVectorType() || rhsType->isVectorType()) {
    if (lhsType->isVectorType() && rhsType->isVectorType()) {
      // If we are allowing lax vector conversions, and LHS and RHS are both
      // vectors, the total size only needs to be the same. This is a bitcast;
      // no bits are changed but the result type is different.
      if (getLangOptions().LaxVectorConversions &&
         (Context.getTypeSize(lhsType) == Context.getTypeSize(rhsType)))
        return IncompatibleVectors;

      // Allow assignments of an AltiVec vector type to an equivalent GCC
      // vector type and vice versa
      if (Context.areCompatibleVectorTypes(lhsType, rhsType))
        return Compatible;
    }
    return Incompatible;
  }

  if (lhsType->isArithmeticType() && rhsType->isArithmeticType() &&
      !(getLangOptions().CPlusPlus && lhsType->isEnumeralType()))
    return Compatible;

  if (isa<PointerType>(lhsType)) {
    if (rhsType->isIntegerType())
      return IntToPointer;

    if (isa<PointerType>(rhsType))
      return CheckPointerTypesForAssignment(lhsType, rhsType);

    // In general, C pointers are not compatible with ObjC object pointers.
    if (isa<ObjCObjectPointerType>(rhsType)) {
      if (lhsType->isVoidPointerType()) // an exception to the rule.
        return Compatible;
      return IncompatiblePointer;
    }
    if (rhsType->getAs<BlockPointerType>()) {
      if (lhsType->getAs<PointerType>()->getPointeeType()->isVoidType())
        return Compatible;

      // Treat block pointers as objects.
      if (getLangOptions().ObjC1 && lhsType->isObjCIdType())
        return Compatible;
    }
    return Incompatible;
  }

  if (isa<BlockPointerType>(lhsType)) {
    if (rhsType->isIntegerType())
      return IntToBlockPointer;

    // Treat block pointers as objects.
    if (getLangOptions().ObjC1 && rhsType->isObjCIdType())
      return Compatible;

    if (rhsType->isBlockPointerType())
      return CheckBlockPointerTypesForAssignment(lhsType, rhsType);

    if (const PointerType *RHSPT = rhsType->getAs<PointerType>()) {
      if (RHSPT->getPointeeType()->isVoidType())
        return Compatible;
    }
    return Incompatible;
  }

  if (isa<ObjCObjectPointerType>(lhsType)) {
    if (rhsType->isIntegerType())
      return IntToPointer;

    // In general, C pointers are not compatible with ObjC object pointers.
    if (isa<PointerType>(rhsType)) {
      if (rhsType->isVoidPointerType()) // an exception to the rule.
        return Compatible;
      return IncompatiblePointer;
    }
    if (rhsType->isObjCObjectPointerType()) {
      return CheckObjCPointerTypesForAssignment(lhsType, rhsType);
    }
    if (const PointerType *RHSPT = rhsType->getAs<PointerType>()) {
      if (RHSPT->getPointeeType()->isVoidType())
        return Compatible;
    }
    // Treat block pointers as objects.
    if (rhsType->isBlockPointerType())
      return Compatible;
    return Incompatible;
  }
  if (isa<PointerType>(rhsType)) {
    // C99 6.5.16.1p1: the left operand is _Bool and the right is a pointer.
    if (lhsType == Context.BoolTy)
      return Compatible;

    if (lhsType->isIntegerType())
      return PointerToInt;

    if (isa<PointerType>(lhsType))
      return CheckPointerTypesForAssignment(lhsType, rhsType);

    if (isa<BlockPointerType>(lhsType) &&
        rhsType->getAs<PointerType>()->getPointeeType()->isVoidType())
      return Compatible;
    return Incompatible;
  }
  if (isa<ObjCObjectPointerType>(rhsType)) {
    // C99 6.5.16.1p1: the left operand is _Bool and the right is a pointer.
    if (lhsType == Context.BoolTy)
      return Compatible;

    if (lhsType->isIntegerType())
      return PointerToInt;

    // In general, C pointers are not compatible with ObjC object pointers.
    if (isa<PointerType>(lhsType)) {
      if (lhsType->isVoidPointerType()) // an exception to the rule.
        return Compatible;
      return IncompatiblePointer;
    }
    if (isa<BlockPointerType>(lhsType) &&
        rhsType->getAs<PointerType>()->getPointeeType()->isVoidType())
      return Compatible;
    return Incompatible;
  }

  if (isa<TagType>(lhsType) && isa<TagType>(rhsType)) {
    if (Context.typesAreCompatible(lhsType, rhsType))
      return Compatible;
  }
  return Incompatible;
}

/// \brief Constructs a transparent union from an expression that is
/// used to initialize the transparent union.
static void ConstructTransparentUnion(ASTContext &C, Expr *&E,
                                      QualType UnionType, FieldDecl *Field) {
  // Build an initializer list that designates the appropriate member
  // of the transparent union.
  InitListExpr *Initializer = new (C) InitListExpr(C, SourceLocation(),
                                                   &E, 1,
                                                   SourceLocation());
  Initializer->setType(UnionType);
  Initializer->setInitializedFieldInUnion(Field);

  // Build a compound literal constructing a value of the transparent
  // union type from this initializer list.
  TypeSourceInfo *unionTInfo = C.getTrivialTypeSourceInfo(UnionType);
  E = new (C) CompoundLiteralExpr(SourceLocation(), unionTInfo, UnionType,
                                  Initializer, false);
}

Sema::AssignConvertType
Sema::CheckTransparentUnionArgumentConstraints(QualType ArgType, Expr *&rExpr) {
  QualType FromType = rExpr->getType();

  // If the ArgType is a Union type, we want to handle a potential
  // transparent_union GCC extension.
  const RecordType *UT = ArgType->getAsUnionType();
  if (!UT || !UT->getDecl()->hasAttr<TransparentUnionAttr>())
    return Incompatible;

  // The field to initialize within the transparent union.
  RecordDecl *UD = UT->getDecl();
  FieldDecl *InitField = 0;
  // It's compatible if the expression matches any of the fields.
  for (RecordDecl::field_iterator it = UD->field_begin(),
         itend = UD->field_end();
       it != itend; ++it) {
    if (it->getType()->isPointerType()) {
      // If the transparent union contains a pointer type, we allow:
      // 1) void pointer
      // 2) null pointer constant
      if (FromType->isPointerType())
        if (FromType->getAs<PointerType>()->getPointeeType()->isVoidType()) {
          ImpCastExprToType(rExpr, it->getType(), CK_BitCast);
          InitField = *it;
          break;
        }

      if (rExpr->isNullPointerConstant(Context,
                                       Expr::NPC_ValueDependentIsNull)) {
        ImpCastExprToType(rExpr, it->getType(), CK_IntegralToPointer);
        InitField = *it;
        break;
      }
    }

    if (CheckAssignmentConstraints(it->getType(), rExpr->getType())
          == Compatible) {
      InitField = *it;
      break;
    }
  }

  if (!InitField)
    return Incompatible;

  ConstructTransparentUnion(Context, rExpr, ArgType, InitField);
  return Compatible;
}

Sema::AssignConvertType
Sema::CheckSingleAssignmentConstraints(QualType lhsType, Expr *&rExpr) {
  if (getLangOptions().CPlusPlus) {
    if (!lhsType->isRecordType()) {
      // C++ 5.17p3: If the left operand is not of class type, the
      // expression is implicitly converted (C++ 4) to the
      // cv-unqualified type of the left operand.
      if (PerformImplicitConversion(rExpr, lhsType.getUnqualifiedType(),
                                    AA_Assigning))
        return Incompatible;
      return Compatible;
    }

    // FIXME: Currently, we fall through and treat C++ classes like C
    // structures.
  }

  // C99 6.5.16.1p1: the left operand is a pointer and the right is
  // a null pointer constant.
  if ((lhsType->isPointerType() ||
       lhsType->isObjCObjectPointerType() ||
       lhsType->isBlockPointerType())
      && rExpr->isNullPointerConstant(Context,
                                      Expr::NPC_ValueDependentIsNull)) {
    ImpCastExprToType(rExpr, lhsType, CK_Unknown);
    return Compatible;
  }

  // This check seems unnatural, however it is necessary to ensure the proper
  // conversion of functions/arrays. If the conversion were done for all
  // DeclExpr's (created by ActOnIdExpression), it would mess up the unary
  // expressions that suppress this implicit conversion (&, sizeof).
  //
  // Suppress this for references: C++ 8.5.3p5.
  if (!lhsType->isReferenceType())
    DefaultFunctionArrayLvalueConversion(rExpr);

  Sema::AssignConvertType result =
    CheckAssignmentConstraints(lhsType, rExpr->getType());

  // C99 6.5.16.1p2: The value of the right operand is converted to the
  // type of the assignment expression.
  // CheckAssignmentConstraints allows the left-hand side to be a reference,
  // so that we can use references in built-in functions even in C.
  // The getNonReferenceType() call makes sure that the resulting expression
  // does not have reference type.
  if (result != Incompatible && rExpr->getType() != lhsType)
    ImpCastExprToType(rExpr, lhsType.getNonLValueExprType(Context),
                      CK_Unknown);
  return result;
}

QualType Sema::InvalidOperands(SourceLocation Loc, Expr *&lex, Expr *&rex) {
  Diag(Loc, diag::err_typecheck_invalid_operands)
    << lex->getType() << rex->getType()
    << lex->getSourceRange() << rex->getSourceRange();
  return QualType();
}

QualType Sema::CheckVectorOperands(SourceLocation Loc, Expr *&lex, Expr *&rex) {
  // For conversion purposes, we ignore any qualifiers.
  // For example, "const float" and "float" are equivalent.
  QualType lhsType =
    Context.getCanonicalType(lex->getType()).getUnqualifiedType();
  QualType rhsType =
    Context.getCanonicalType(rex->getType()).getUnqualifiedType();

  // If the vector types are identical, return.
  if (lhsType == rhsType)
    return lhsType;

  // Handle the case of a vector & extvector type of the same size and element
  // type.  It would be nice if we only had one vector type someday.
  if (getLangOptions().LaxVectorConversions) {
    if (const VectorType *LV = lhsType->getAs<VectorType>()) {
      if (const VectorType *RV = rhsType->getAs<VectorType>()) {
        if (LV->getElementType() == RV->getElementType() &&
            LV->getNumElements() == RV->getNumElements()) {
          if (lhsType->isExtVectorType()) {
            ImpCastExprToType(rex, lhsType, CK_BitCast);
            return lhsType;
          } 

          ImpCastExprToType(lex, rhsType, CK_BitCast);
          return rhsType;
        } else if (Context.getTypeSize(lhsType) ==Context.getTypeSize(rhsType)){
          // If we are allowing lax vector conversions, and LHS and RHS are both
          // vectors, the total size only needs to be the same. This is a
          // bitcast; no bits are changed but the result type is different.
          ImpCastExprToType(rex, lhsType, CK_BitCast);
          return lhsType;
        }
      }
    }
  }

  // Handle the case of equivalent AltiVec and GCC vector types
  if (lhsType->isVectorType() && rhsType->isVectorType() &&
      Context.areCompatibleVectorTypes(lhsType, rhsType)) {
    ImpCastExprToType(lex, rhsType, CK_BitCast);
    return rhsType;
  }

  // Canonicalize the ExtVector to the LHS, remember if we swapped so we can
  // swap back (so that we don't reverse the inputs to a subtract, for instance.
  bool swapped = false;
  if (rhsType->isExtVectorType()) {
    swapped = true;
    std::swap(rex, lex);
    std::swap(rhsType, lhsType);
  }

  // Handle the case of an ext vector and scalar.
  if (const ExtVectorType *LV = lhsType->getAs<ExtVectorType>()) {
    QualType EltTy = LV->getElementType();
    if (EltTy->isIntegralType(Context) && rhsType->isIntegralType(Context)) {
      if (Context.getIntegerTypeOrder(EltTy, rhsType) >= 0) {
        ImpCastExprToType(rex, lhsType, CK_IntegralCast);
        if (swapped) std::swap(rex, lex);
        return lhsType;
      }
    }
    if (EltTy->isRealFloatingType() && rhsType->isScalarType() &&
        rhsType->isRealFloatingType()) {
      if (Context.getFloatingTypeOrder(EltTy, rhsType) >= 0) {
        ImpCastExprToType(rex, lhsType, CK_FloatingCast);
        if (swapped) std::swap(rex, lex);
        return lhsType;
      }
    }
  }

  // Vectors of different size or scalar and non-ext-vector are errors.
  Diag(Loc, diag::err_typecheck_vector_not_convertable)
    << lex->getType() << rex->getType()
    << lex->getSourceRange() << rex->getSourceRange();
  return QualType();
}

QualType Sema::CheckMultiplyDivideOperands(
  Expr *&lex, Expr *&rex, SourceLocation Loc, bool isCompAssign, bool isDiv) {
  if (lex->getType()->isVectorType() || rex->getType()->isVectorType())
    return CheckVectorOperands(Loc, lex, rex);

  QualType compType = UsualArithmeticConversions(lex, rex, isCompAssign);

  if (!lex->getType()->isArithmeticType() ||
      !rex->getType()->isArithmeticType())
    return InvalidOperands(Loc, lex, rex);

  // Check for division by zero.
  if (isDiv &&
      rex->isNullPointerConstant(Context, Expr::NPC_ValueDependentIsNotNull))
    DiagRuntimeBehavior(Loc, PDiag(diag::warn_division_by_zero)
                                     << rex->getSourceRange());

  return compType;
}

QualType Sema::CheckRemainderOperands(
  Expr *&lex, Expr *&rex, SourceLocation Loc, bool isCompAssign) {
  if (lex->getType()->isVectorType() || rex->getType()->isVectorType()) {
    if (lex->getType()->hasIntegerRepresentation() && 
        rex->getType()->hasIntegerRepresentation())
      return CheckVectorOperands(Loc, lex, rex);
    return InvalidOperands(Loc, lex, rex);
  }

  QualType compType = UsualArithmeticConversions(lex, rex, isCompAssign);

  if (!lex->getType()->isIntegerType() || !rex->getType()->isIntegerType())
    return InvalidOperands(Loc, lex, rex);

  // Check for remainder by zero.
  if (rex->isNullPointerConstant(Context, Expr::NPC_ValueDependentIsNotNull))
    DiagRuntimeBehavior(Loc, PDiag(diag::warn_remainder_by_zero)
                                 << rex->getSourceRange());

  return compType;
}

QualType Sema::CheckAdditionOperands( // C99 6.5.6
  Expr *&lex, Expr *&rex, SourceLocation Loc, QualType* CompLHSTy) {
  if (lex->getType()->isVectorType() || rex->getType()->isVectorType()) {
    QualType compType = CheckVectorOperands(Loc, lex, rex);
    if (CompLHSTy) *CompLHSTy = compType;
    return compType;
  }

  QualType compType = UsualArithmeticConversions(lex, rex, CompLHSTy);

  // handle the common case first (both operands are arithmetic).
  if (lex->getType()->isArithmeticType() &&
      rex->getType()->isArithmeticType()) {
    if (CompLHSTy) *CompLHSTy = compType;
    return compType;
  }

  // Put any potential pointer into PExp
  Expr* PExp = lex, *IExp = rex;
  if (IExp->getType()->isAnyPointerType())
    std::swap(PExp, IExp);

  if (PExp->getType()->isAnyPointerType()) {

    if (IExp->getType()->isIntegerType()) {
      QualType PointeeTy = PExp->getType()->getPointeeType();

      // Check for arithmetic on pointers to incomplete types.
      if (PointeeTy->isVoidType()) {
        if (getLangOptions().CPlusPlus) {
          Diag(Loc, diag::err_typecheck_pointer_arith_void_type)
            << lex->getSourceRange() << rex->getSourceRange();
          return QualType();
        }

        // GNU extension: arithmetic on pointer to void
        Diag(Loc, diag::ext_gnu_void_ptr)
          << lex->getSourceRange() << rex->getSourceRange();
      } else if (PointeeTy->isFunctionType()) {
        if (getLangOptions().CPlusPlus) {
          Diag(Loc, diag::err_typecheck_pointer_arith_function_type)
            << lex->getType() << lex->getSourceRange();
          return QualType();
        }

        // GNU extension: arithmetic on pointer to function
        Diag(Loc, diag::ext_gnu_ptr_func_arith)
          << lex->getType() << lex->getSourceRange();
      } else {
        // Check if we require a complete type.
        if (((PExp->getType()->isPointerType() &&
              !PExp->getType()->isDependentType()) ||
              PExp->getType()->isObjCObjectPointerType()) &&
             RequireCompleteType(Loc, PointeeTy,
                           PDiag(diag::err_typecheck_arithmetic_incomplete_type)
                             << PExp->getSourceRange()
                             << PExp->getType()))
          return QualType();
      }
      // Diagnose bad cases where we step over interface counts.
      if (PointeeTy->isObjCObjectType() && LangOpts.ObjCNonFragileABI) {
        Diag(Loc, diag::err_arithmetic_nonfragile_interface)
          << PointeeTy << PExp->getSourceRange();
        return QualType();
      }

      if (CompLHSTy) {
        QualType LHSTy = Context.isPromotableBitField(lex);
        if (LHSTy.isNull()) {
          LHSTy = lex->getType();
          if (LHSTy->isPromotableIntegerType())
            LHSTy = Context.getPromotedIntegerType(LHSTy);
        }
        *CompLHSTy = LHSTy;
      }
      return PExp->getType();
    }
  }

  return InvalidOperands(Loc, lex, rex);
}

// C99 6.5.6
QualType Sema::CheckSubtractionOperands(Expr *&lex, Expr *&rex,
                                        SourceLocation Loc, QualType* CompLHSTy) {
  if (lex->getType()->isVectorType() || rex->getType()->isVectorType()) {
    QualType compType = CheckVectorOperands(Loc, lex, rex);
    if (CompLHSTy) *CompLHSTy = compType;
    return compType;
  }

  QualType compType = UsualArithmeticConversions(lex, rex, CompLHSTy);

  // Enforce type constraints: C99 6.5.6p3.

  // Handle the common case first (both operands are arithmetic).
  if (lex->getType()->isArithmeticType()
      && rex->getType()->isArithmeticType()) {
    if (CompLHSTy) *CompLHSTy = compType;
    return compType;
  }

  // Either ptr - int   or   ptr - ptr.
  if (lex->getType()->isAnyPointerType()) {
    QualType lpointee = lex->getType()->getPointeeType();

    // The LHS must be an completely-defined object type.

    bool ComplainAboutVoid = false;
    Expr *ComplainAboutFunc = 0;
    if (lpointee->isVoidType()) {
      if (getLangOptions().CPlusPlus) {
        Diag(Loc, diag::err_typecheck_pointer_arith_void_type)
          << lex->getSourceRange() << rex->getSourceRange();
        return QualType();
      }

      // GNU C extension: arithmetic on pointer to void
      ComplainAboutVoid = true;
    } else if (lpointee->isFunctionType()) {
      if (getLangOptions().CPlusPlus) {
        Diag(Loc, diag::err_typecheck_pointer_arith_function_type)
          << lex->getType() << lex->getSourceRange();
        return QualType();
      }

      // GNU C extension: arithmetic on pointer to function
      ComplainAboutFunc = lex;
    } else if (!lpointee->isDependentType() &&
               RequireCompleteType(Loc, lpointee,
                                   PDiag(diag::err_typecheck_sub_ptr_object)
                                     << lex->getSourceRange()
                                     << lex->getType()))
      return QualType();

    // Diagnose bad cases where we step over interface counts.
    if (lpointee->isObjCObjectType() && LangOpts.ObjCNonFragileABI) {
      Diag(Loc, diag::err_arithmetic_nonfragile_interface)
        << lpointee << lex->getSourceRange();
      return QualType();
    }

    // The result type of a pointer-int computation is the pointer type.
    if (rex->getType()->isIntegerType()) {
      if (ComplainAboutVoid)
        Diag(Loc, diag::ext_gnu_void_ptr)
          << lex->getSourceRange() << rex->getSourceRange();
      if (ComplainAboutFunc)
        Diag(Loc, diag::ext_gnu_ptr_func_arith)
          << ComplainAboutFunc->getType()
          << ComplainAboutFunc->getSourceRange();

      if (CompLHSTy) *CompLHSTy = lex->getType();
      return lex->getType();
    }

    // Handle pointer-pointer subtractions.
    if (const PointerType *RHSPTy = rex->getType()->getAs<PointerType>()) {
      QualType rpointee = RHSPTy->getPointeeType();

      // RHS must be a completely-type object type.
      // Handle the GNU void* extension.
      if (rpointee->isVoidType()) {
        if (getLangOptions().CPlusPlus) {
          Diag(Loc, diag::err_typecheck_pointer_arith_void_type)
            << lex->getSourceRange() << rex->getSourceRange();
          return QualType();
        }

        ComplainAboutVoid = true;
      } else if (rpointee->isFunctionType()) {
        if (getLangOptions().CPlusPlus) {
          Diag(Loc, diag::err_typecheck_pointer_arith_function_type)
            << rex->getType() << rex->getSourceRange();
          return QualType();
        }

        // GNU extension: arithmetic on pointer to function
        if (!ComplainAboutFunc)
          ComplainAboutFunc = rex;
      } else if (!rpointee->isDependentType() &&
                 RequireCompleteType(Loc, rpointee,
                                     PDiag(diag::err_typecheck_sub_ptr_object)
                                       << rex->getSourceRange()
                                       << rex->getType()))
        return QualType();

      if (getLangOptions().CPlusPlus) {
        // Pointee types must be the same: C++ [expr.add]
        if (!Context.hasSameUnqualifiedType(lpointee, rpointee)) {
          Diag(Loc, diag::err_typecheck_sub_ptr_compatible)
            << lex->getType() << rex->getType()
            << lex->getSourceRange() << rex->getSourceRange();
          return QualType();
        }
      } else {
        // Pointee types must be compatible C99 6.5.6p3
        if (!Context.typesAreCompatible(
                Context.getCanonicalType(lpointee).getUnqualifiedType(),
                Context.getCanonicalType(rpointee).getUnqualifiedType())) {
          Diag(Loc, diag::err_typecheck_sub_ptr_compatible)
            << lex->getType() << rex->getType()
            << lex->getSourceRange() << rex->getSourceRange();
          return QualType();
        }
      }

      if (ComplainAboutVoid)
        Diag(Loc, diag::ext_gnu_void_ptr)
          << lex->getSourceRange() << rex->getSourceRange();
      if (ComplainAboutFunc)
        Diag(Loc, diag::ext_gnu_ptr_func_arith)
          << ComplainAboutFunc->getType()
          << ComplainAboutFunc->getSourceRange();

      if (CompLHSTy) *CompLHSTy = lex->getType();
      return Context.getPointerDiffType();
    }
  }

  return InvalidOperands(Loc, lex, rex);
}

// C99 6.5.7
QualType Sema::CheckShiftOperands(Expr *&lex, Expr *&rex, SourceLocation Loc,
                                  bool isCompAssign) {
  // C99 6.5.7p2: Each of the operands shall have integer type.
  if (!lex->getType()->hasIntegerRepresentation() || 
      !rex->getType()->hasIntegerRepresentation())
    return InvalidOperands(Loc, lex, rex);

  // Vector shifts promote their scalar inputs to vector type.
  if (lex->getType()->isVectorType() || rex->getType()->isVectorType())
    return CheckVectorOperands(Loc, lex, rex);

  // Shifts don't perform usual arithmetic conversions, they just do integer
  // promotions on each operand. C99 6.5.7p3
  QualType LHSTy = Context.isPromotableBitField(lex);
  if (LHSTy.isNull()) {
    LHSTy = lex->getType();
    if (LHSTy->isPromotableIntegerType())
      LHSTy = Context.getPromotedIntegerType(LHSTy);
  }
  if (!isCompAssign)
    ImpCastExprToType(lex, LHSTy, CK_IntegralCast);

  UsualUnaryConversions(rex);

  // Sanity-check shift operands
  llvm::APSInt Right;
  // Check right/shifter operand
  if (!rex->isValueDependent() &&
      rex->isIntegerConstantExpr(Right, Context)) {
    if (Right.isNegative())
      Diag(Loc, diag::warn_shift_negative) << rex->getSourceRange();
    else {
      llvm::APInt LeftBits(Right.getBitWidth(),
                          Context.getTypeSize(lex->getType()));
      if (Right.uge(LeftBits))
        Diag(Loc, diag::warn_shift_gt_typewidth) << rex->getSourceRange();
    }
  }

  // "The type of the result is that of the promoted left operand."
  return LHSTy;
}

static bool IsWithinTemplateSpecialization(Decl *D) {
  if (DeclContext *DC = D->getDeclContext()) {
    if (isa<ClassTemplateSpecializationDecl>(DC))
      return true;
    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(DC))
      return FD->isFunctionTemplateSpecialization();
  }
  return false;
}

// C99 6.5.8, C++ [expr.rel]
QualType Sema::CheckCompareOperands(Expr *&lex, Expr *&rex, SourceLocation Loc,
                                    unsigned OpaqueOpc, bool isRelational) {
  BinaryOperatorKind Opc = (BinaryOperatorKind) OpaqueOpc;

  // Handle vector comparisons separately.
  if (lex->getType()->isVectorType() || rex->getType()->isVectorType())
    return CheckVectorCompareOperands(lex, rex, Loc, isRelational);

  QualType lType = lex->getType();
  QualType rType = rex->getType();

  if (!lType->hasFloatingRepresentation() &&
      !(lType->isBlockPointerType() && isRelational)) {
    // For non-floating point types, check for self-comparisons of the form
    // x == x, x != x, x < x, etc.  These always evaluate to a constant, and
    // often indicate logic errors in the program.
    //
    // NOTE: Don't warn about comparison expressions resulting from macro
    // expansion. Also don't warn about comparisons which are only self
    // comparisons within a template specialization. The warnings should catch
    // obvious cases in the definition of the template anyways. The idea is to
    // warn when the typed comparison operator will always evaluate to the same
    // result.
    Expr *LHSStripped = lex->IgnoreParens();
    Expr *RHSStripped = rex->IgnoreParens();
    if (DeclRefExpr* DRL = dyn_cast<DeclRefExpr>(LHSStripped)) {
      if (DeclRefExpr* DRR = dyn_cast<DeclRefExpr>(RHSStripped)) {
        if (DRL->getDecl() == DRR->getDecl() && !Loc.isMacroID() &&
            !IsWithinTemplateSpecialization(DRL->getDecl())) {
          DiagRuntimeBehavior(Loc, PDiag(diag::warn_comparison_always)
                              << 0 // self-
                              << (Opc == BO_EQ
                                  || Opc == BO_LE
                                  || Opc == BO_GE));
        } else if (lType->isArrayType() && rType->isArrayType() &&
                   !DRL->getDecl()->getType()->isReferenceType() &&
                   !DRR->getDecl()->getType()->isReferenceType()) {
            // what is it always going to eval to?
            char always_evals_to;
            switch(Opc) {
            case BO_EQ: // e.g. array1 == array2
              always_evals_to = 0; // false
              break;
            case BO_NE: // e.g. array1 != array2
              always_evals_to = 1; // true
              break;
            default:
              // best we can say is 'a constant'
              always_evals_to = 2; // e.g. array1 <= array2
              break;
            }
            DiagRuntimeBehavior(Loc, PDiag(diag::warn_comparison_always)
                                << 1 // array
                                << always_evals_to);
        }
      }
    }

    if (isa<CastExpr>(LHSStripped))
      LHSStripped = LHSStripped->IgnoreParenCasts();
    if (isa<CastExpr>(RHSStripped))
      RHSStripped = RHSStripped->IgnoreParenCasts();

    // Warn about comparisons against a string constant (unless the other
    // operand is null), the user probably wants strcmp.
    Expr *literalString = 0;
    Expr *literalStringStripped = 0;
    if ((isa<StringLiteral>(LHSStripped) || isa<ObjCEncodeExpr>(LHSStripped)) &&
        !RHSStripped->isNullPointerConstant(Context,
                                            Expr::NPC_ValueDependentIsNull)) {
      literalString = lex;
      literalStringStripped = LHSStripped;
    } else if ((isa<StringLiteral>(RHSStripped) ||
                isa<ObjCEncodeExpr>(RHSStripped)) &&
               !LHSStripped->isNullPointerConstant(Context,
                                            Expr::NPC_ValueDependentIsNull)) {
      literalString = rex;
      literalStringStripped = RHSStripped;
    }

    if (literalString) {
      std::string resultComparison;
      switch (Opc) {
      case BO_LT: resultComparison = ") < 0"; break;
      case BO_GT: resultComparison = ") > 0"; break;
      case BO_LE: resultComparison = ") <= 0"; break;
      case BO_GE: resultComparison = ") >= 0"; break;
      case BO_EQ: resultComparison = ") == 0"; break;
      case BO_NE: resultComparison = ") != 0"; break;
      default: assert(false && "Invalid comparison operator");
      }

      DiagRuntimeBehavior(Loc,
        PDiag(diag::warn_stringcompare)
          << isa<ObjCEncodeExpr>(literalStringStripped)
          << literalString->getSourceRange());
    }
  }

  // C99 6.5.8p3 / C99 6.5.9p4
  if (lex->getType()->isArithmeticType() && rex->getType()->isArithmeticType())
    UsualArithmeticConversions(lex, rex);
  else {
    UsualUnaryConversions(lex);
    UsualUnaryConversions(rex);
  }

  lType = lex->getType();
  rType = rex->getType();

  // The result of comparisons is 'bool' in C++, 'int' in C.
  QualType ResultTy = getLangOptions().CPlusPlus ? Context.BoolTy:Context.IntTy;

  if (isRelational) {
    if (lType->isRealType() && rType->isRealType())
      return ResultTy;
  } else {
    // Check for comparisons of floating point operands using != and ==.
    if (lType->hasFloatingRepresentation())
      CheckFloatComparison(Loc,lex,rex);

    if (lType->isArithmeticType() && rType->isArithmeticType())
      return ResultTy;
  }

  bool LHSIsNull = lex->isNullPointerConstant(Context,
                                              Expr::NPC_ValueDependentIsNull);
  bool RHSIsNull = rex->isNullPointerConstant(Context,
                                              Expr::NPC_ValueDependentIsNull);

  // All of the following pointer-related warnings are GCC extensions, except
  // when handling null pointer constants. 
  if (lType->isPointerType() && rType->isPointerType()) { // C99 6.5.8p2
    QualType LCanPointeeTy =
      Context.getCanonicalType(lType->getAs<PointerType>()->getPointeeType());
    QualType RCanPointeeTy =
      Context.getCanonicalType(rType->getAs<PointerType>()->getPointeeType());

    if (getLangOptions().CPlusPlus) {
      if (LCanPointeeTy == RCanPointeeTy)
        return ResultTy;
      if (!isRelational &&
          (LCanPointeeTy->isVoidType() || RCanPointeeTy->isVoidType())) {
        // Valid unless comparison between non-null pointer and function pointer
        // This is a gcc extension compatibility comparison.
        // In a SFINAE context, we treat this as a hard error to maintain
        // conformance with the C++ standard.
        if ((LCanPointeeTy->isFunctionType() || RCanPointeeTy->isFunctionType())
            && !LHSIsNull && !RHSIsNull) {
          Diag(Loc, 
               isSFINAEContext()? 
                   diag::err_typecheck_comparison_of_fptr_to_void
                 : diag::ext_typecheck_comparison_of_fptr_to_void)
            << lType << rType << lex->getSourceRange() << rex->getSourceRange();
          
          if (isSFINAEContext())
            return QualType();
          
          ImpCastExprToType(rex, lType, CK_BitCast);
          return ResultTy;
        }
      }
      // C++ [expr.rel]p2:
      //   [...] Pointer conversions (4.10) and qualification
      //   conversions (4.4) are performed on pointer operands (or on
      //   a pointer operand and a null pointer constant) to bring
      //   them to their composite pointer type. [...]
      //
      // C++ [expr.eq]p1 uses the same notion for (in)equality
      // comparisons of pointers.
      bool NonStandardCompositeType = false;
      QualType T = FindCompositePointerType(Loc, lex, rex,
                              isSFINAEContext()? 0 : &NonStandardCompositeType);
      if (T.isNull()) {
        Diag(Loc, diag::err_typecheck_comparison_of_distinct_pointers)
          << lType << rType << lex->getSourceRange() << rex->getSourceRange();
        return QualType();
      } else if (NonStandardCompositeType) {
        Diag(Loc,
             diag::ext_typecheck_comparison_of_distinct_pointers_nonstandard)
          << lType << rType << T
          << lex->getSourceRange() << rex->getSourceRange();
      }

      ImpCastExprToType(lex, T, CK_BitCast);
      ImpCastExprToType(rex, T, CK_BitCast);
      return ResultTy;
    }
    // C99 6.5.9p2 and C99 6.5.8p2
    if (Context.typesAreCompatible(LCanPointeeTy.getUnqualifiedType(),
                                   RCanPointeeTy.getUnqualifiedType())) {
      // Valid unless a relational comparison of function pointers
      if (isRelational && LCanPointeeTy->isFunctionType()) {
        Diag(Loc, diag::ext_typecheck_ordered_comparison_of_function_pointers)
          << lType << rType << lex->getSourceRange() << rex->getSourceRange();
      }
    } else if (!isRelational &&
               (LCanPointeeTy->isVoidType() || RCanPointeeTy->isVoidType())) {
      // Valid unless comparison between non-null pointer and function pointer
      if ((LCanPointeeTy->isFunctionType() || RCanPointeeTy->isFunctionType())
          && !LHSIsNull && !RHSIsNull) {
        Diag(Loc, diag::ext_typecheck_comparison_of_fptr_to_void)
          << lType << rType << lex->getSourceRange() << rex->getSourceRange();
      }
    } else {
      // Invalid
      Diag(Loc, diag::ext_typecheck_comparison_of_distinct_pointers)
        << lType << rType << lex->getSourceRange() << rex->getSourceRange();
    }
    if (LCanPointeeTy != RCanPointeeTy)
      ImpCastExprToType(rex, lType, CK_BitCast);
    return ResultTy;
  }

  if (getLangOptions().CPlusPlus) {
    // Comparison of pointers with null pointer constants and equality
    // comparisons of member pointers to null pointer constants.
    if (RHSIsNull &&
        (lType->isPointerType() ||
         (!isRelational && lType->isMemberPointerType()))) {
      ImpCastExprToType(rex, lType, 
                        lType->isMemberPointerType()
                          ? CK_NullToMemberPointer
                          : CK_IntegralToPointer);
      return ResultTy;
    }
    if (LHSIsNull &&
        (rType->isPointerType() ||
         (!isRelational && rType->isMemberPointerType()))) {
      ImpCastExprToType(lex, rType, 
                        rType->isMemberPointerType()
                          ? CK_NullToMemberPointer
                          : CK_IntegralToPointer);
      return ResultTy;
    }

    // Comparison of member pointers.
    if (!isRelational &&
        lType->isMemberPointerType() && rType->isMemberPointerType()) {
      // C++ [expr.eq]p2:
      //   In addition, pointers to members can be compared, or a pointer to
      //   member and a null pointer constant. Pointer to member conversions
      //   (4.11) and qualification conversions (4.4) are performed to bring
      //   them to a common type. If one operand is a null pointer constant,
      //   the common type is the type of the other operand. Otherwise, the
      //   common type is a pointer to member type similar (4.4) to the type
      //   of one of the operands, with a cv-qualification signature (4.4)
      //   that is the union of the cv-qualification signatures of the operand
      //   types.
      bool NonStandardCompositeType = false;
      QualType T = FindCompositePointerType(Loc, lex, rex,
                              isSFINAEContext()? 0 : &NonStandardCompositeType);
      if (T.isNull()) {
        Diag(Loc, diag::err_typecheck_comparison_of_distinct_pointers)
          << lType << rType << lex->getSourceRange() << rex->getSourceRange();
        return QualType();
      } else if (NonStandardCompositeType) {
        Diag(Loc,
             diag::ext_typecheck_comparison_of_distinct_pointers_nonstandard)
          << lType << rType << T
          << lex->getSourceRange() << rex->getSourceRange();
      }

      ImpCastExprToType(lex, T, CK_BitCast);
      ImpCastExprToType(rex, T, CK_BitCast);
      return ResultTy;
    }

    // Comparison of nullptr_t with itself.
    if (lType->isNullPtrType() && rType->isNullPtrType())
      return ResultTy;
  }

  // Handle block pointer types.
  if (!isRelational && lType->isBlockPointerType() && rType->isBlockPointerType()) {
    QualType lpointee = lType->getAs<BlockPointerType>()->getPointeeType();
    QualType rpointee = rType->getAs<BlockPointerType>()->getPointeeType();

    if (!LHSIsNull && !RHSIsNull &&
        !Context.typesAreCompatible(lpointee, rpointee)) {
      Diag(Loc, diag::err_typecheck_comparison_of_distinct_blocks)
        << lType << rType << lex->getSourceRange() << rex->getSourceRange();
    }
    ImpCastExprToType(rex, lType, CK_BitCast);
    return ResultTy;
  }
  // Allow block pointers to be compared with null pointer constants.
  if (!isRelational
      && ((lType->isBlockPointerType() && rType->isPointerType())
          || (lType->isPointerType() && rType->isBlockPointerType()))) {
    if (!LHSIsNull && !RHSIsNull) {
      if (!((rType->isPointerType() && rType->getAs<PointerType>()
             ->getPointeeType()->isVoidType())
            || (lType->isPointerType() && lType->getAs<PointerType>()
                ->getPointeeType()->isVoidType())))
        Diag(Loc, diag::err_typecheck_comparison_of_distinct_blocks)
          << lType << rType << lex->getSourceRange() << rex->getSourceRange();
    }
    ImpCastExprToType(rex, lType, CK_BitCast);
    return ResultTy;
  }

  if ((lType->isObjCObjectPointerType() || rType->isObjCObjectPointerType())) {
    if (lType->isPointerType() || rType->isPointerType()) {
      const PointerType *LPT = lType->getAs<PointerType>();
      const PointerType *RPT = rType->getAs<PointerType>();
      bool LPtrToVoid = LPT ?
        Context.getCanonicalType(LPT->getPointeeType())->isVoidType() : false;
      bool RPtrToVoid = RPT ?
        Context.getCanonicalType(RPT->getPointeeType())->isVoidType() : false;

      if (!LPtrToVoid && !RPtrToVoid &&
          !Context.typesAreCompatible(lType, rType)) {
        Diag(Loc, diag::ext_typecheck_comparison_of_distinct_pointers)
          << lType << rType << lex->getSourceRange() << rex->getSourceRange();
      }
      ImpCastExprToType(rex, lType, CK_BitCast);
      return ResultTy;
    }
    if (lType->isObjCObjectPointerType() && rType->isObjCObjectPointerType()) {
      if (!Context.areComparableObjCPointerTypes(lType, rType))
        Diag(Loc, diag::ext_typecheck_comparison_of_distinct_pointers)
          << lType << rType << lex->getSourceRange() << rex->getSourceRange();
      ImpCastExprToType(rex, lType, CK_BitCast);
      return ResultTy;
    }
  }
  if ((lType->isAnyPointerType() && rType->isIntegerType()) ||
      (lType->isIntegerType() && rType->isAnyPointerType())) {
    unsigned DiagID = 0;
    bool isError = false;
    if ((LHSIsNull && lType->isIntegerType()) ||
        (RHSIsNull && rType->isIntegerType())) {
      if (isRelational && !getLangOptions().CPlusPlus)
        DiagID = diag::ext_typecheck_ordered_comparison_of_pointer_and_zero;
    } else if (isRelational && !getLangOptions().CPlusPlus)
      DiagID = diag::ext_typecheck_ordered_comparison_of_pointer_integer;
    else if (getLangOptions().CPlusPlus) {
      DiagID = diag::err_typecheck_comparison_of_pointer_integer;
      isError = true;
    } else
      DiagID = diag::ext_typecheck_comparison_of_pointer_integer;

    if (DiagID) {
      Diag(Loc, DiagID)
        << lType << rType << lex->getSourceRange() << rex->getSourceRange();
      if (isError)
        return QualType();
    }
    
    if (lType->isIntegerType())
      ImpCastExprToType(lex, rType, CK_IntegralToPointer);
    else
      ImpCastExprToType(rex, lType, CK_IntegralToPointer);
    return ResultTy;
  }
  
  // Handle block pointers.
  if (!isRelational && RHSIsNull
      && lType->isBlockPointerType() && rType->isIntegerType()) {
    ImpCastExprToType(rex, lType, CK_IntegralToPointer);
    return ResultTy;
  }
  if (!isRelational && LHSIsNull
      && lType->isIntegerType() && rType->isBlockPointerType()) {
    ImpCastExprToType(lex, rType, CK_IntegralToPointer);
    return ResultTy;
  }
  return InvalidOperands(Loc, lex, rex);
}

/// CheckVectorCompareOperands - vector comparisons are a clang extension that
/// operates on extended vector types.  Instead of producing an IntTy result,
/// like a scalar comparison, a vector comparison produces a vector of integer
/// types.
QualType Sema::CheckVectorCompareOperands(Expr *&lex, Expr *&rex,
                                          SourceLocation Loc,
                                          bool isRelational) {
  // Check to make sure we're operating on vectors of the same type and width,
  // Allowing one side to be a scalar of element type.
  QualType vType = CheckVectorOperands(Loc, lex, rex);
  if (vType.isNull())
    return vType;

  QualType lType = lex->getType();
  QualType rType = rex->getType();

  // For non-floating point types, check for self-comparisons of the form
  // x == x, x != x, x < x, etc.  These always evaluate to a constant, and
  // often indicate logic errors in the program.
  if (!lType->hasFloatingRepresentation()) {
    if (DeclRefExpr* DRL = dyn_cast<DeclRefExpr>(lex->IgnoreParens()))
      if (DeclRefExpr* DRR = dyn_cast<DeclRefExpr>(rex->IgnoreParens()))
        if (DRL->getDecl() == DRR->getDecl())
          DiagRuntimeBehavior(Loc,
                              PDiag(diag::warn_comparison_always)
                                << 0 // self-
                                << 2 // "a constant"
                              );
  }

  // Check for comparisons of floating point operands using != and ==.
  if (!isRelational && lType->hasFloatingRepresentation()) {
    assert (rType->hasFloatingRepresentation());
    CheckFloatComparison(Loc,lex,rex);
  }

  // Return the type for the comparison, which is the same as vector type for
  // integer vectors, or an integer type of identical size and number of
  // elements for floating point vectors.
  if (lType->hasIntegerRepresentation())
    return lType;

  const VectorType *VTy = lType->getAs<VectorType>();
  unsigned TypeSize = Context.getTypeSize(VTy->getElementType());
  if (TypeSize == Context.getTypeSize(Context.IntTy))
    return Context.getExtVectorType(Context.IntTy, VTy->getNumElements());
  if (TypeSize == Context.getTypeSize(Context.LongTy))
    return Context.getExtVectorType(Context.LongTy, VTy->getNumElements());

  assert(TypeSize == Context.getTypeSize(Context.LongLongTy) &&
         "Unhandled vector element size in vector compare");
  return Context.getExtVectorType(Context.LongLongTy, VTy->getNumElements());
}

inline QualType Sema::CheckBitwiseOperands(
  Expr *&lex, Expr *&rex, SourceLocation Loc, bool isCompAssign) {
  if (lex->getType()->isVectorType() || rex->getType()->isVectorType()) {
    if (lex->getType()->hasIntegerRepresentation() &&
        rex->getType()->hasIntegerRepresentation())
      return CheckVectorOperands(Loc, lex, rex);
    
    return InvalidOperands(Loc, lex, rex);
  }

  QualType compType = UsualArithmeticConversions(lex, rex, isCompAssign);

  if (lex->getType()->isIntegerType() && rex->getType()->isIntegerType())
    return compType;
  return InvalidOperands(Loc, lex, rex);
}

inline QualType Sema::CheckLogicalOperands( // C99 6.5.[13,14]
  Expr *&lex, Expr *&rex, SourceLocation Loc, unsigned Opc) {
  
  // Diagnose cases where the user write a logical and/or but probably meant a
  // bitwise one.  We do this when the LHS is a non-bool integer and the RHS
  // is a constant.
  if (lex->getType()->isIntegerType() && !lex->getType()->isBooleanType() &&
      rex->getType()->isIntegerType() && !rex->isValueDependent() &&
      // Don't warn in macros.
      !Loc.isMacroID()) {
    // If the RHS can be constant folded, and if it constant folds to something
    // that isn't 0 or 1 (which indicate a potential logical operation that
    // happened to fold to true/false) then warn.
    Expr::EvalResult Result;
    if (rex->Evaluate(Result, Context) && !Result.HasSideEffects &&
        Result.Val.getInt() != 0 && Result.Val.getInt() != 1) {
      Diag(Loc, diag::warn_logical_instead_of_bitwise)
       << rex->getSourceRange()
        << (Opc == BO_LAnd ? "&&" : "||")
        << (Opc == BO_LAnd ? "&" : "|");
    }
  }
  
  if (!Context.getLangOptions().CPlusPlus) {
    UsualUnaryConversions(lex);
    UsualUnaryConversions(rex);

    if (!lex->getType()->isScalarType() || !rex->getType()->isScalarType())
      return InvalidOperands(Loc, lex, rex);

    return Context.IntTy;
  }

  // The following is safe because we only use this method for
  // non-overloadable operands.

  // C++ [expr.log.and]p1
  // C++ [expr.log.or]p1
  // The operands are both contextually converted to type bool.
  if (PerformContextuallyConvertToBool(lex) ||
      PerformContextuallyConvertToBool(rex))
    return InvalidOperands(Loc, lex, rex);

  // C++ [expr.log.and]p2
  // C++ [expr.log.or]p2
  // The result is a bool.
  return Context.BoolTy;
}

/// IsReadonlyProperty - Verify that otherwise a valid l-value expression
/// is a read-only property; return true if so. A readonly property expression
/// depends on various declarations and thus must be treated specially.
///
static bool IsReadonlyProperty(Expr *E, Sema &S) {
  if (E->getStmtClass() == Expr::ObjCPropertyRefExprClass) {
    const ObjCPropertyRefExpr* PropExpr = cast<ObjCPropertyRefExpr>(E);
    if (ObjCPropertyDecl *PDecl = PropExpr->getProperty()) {
      QualType BaseType = PropExpr->getBase()->getType();
      if (const ObjCObjectPointerType *OPT =
            BaseType->getAsObjCInterfacePointerType())
        if (ObjCInterfaceDecl *IFace = OPT->getInterfaceDecl())
          if (S.isPropertyReadonly(PDecl, IFace))
            return true;
    }
  }
  return false;
}

/// CheckForModifiableLvalue - Verify that E is a modifiable lvalue.  If not,
/// emit an error and return true.  If so, return false.
static bool CheckForModifiableLvalue(Expr *E, SourceLocation Loc, Sema &S) {
  SourceLocation OrigLoc = Loc;
  Expr::isModifiableLvalueResult IsLV = E->isModifiableLvalue(S.Context,
                                                              &Loc);
  if (IsLV == Expr::MLV_Valid && IsReadonlyProperty(E, S))
    IsLV = Expr::MLV_ReadonlyProperty;
  if (IsLV == Expr::MLV_Valid)
    return false;

  unsigned Diag = 0;
  bool NeedType = false;
  switch (IsLV) { // C99 6.5.16p2
  case Expr::MLV_ConstQualified: Diag = diag::err_typecheck_assign_const; break;
  case Expr::MLV_ArrayType:
    Diag = diag::err_typecheck_array_not_modifiable_lvalue;
    NeedType = true;
    break;
  case Expr::MLV_NotObjectType:
    Diag = diag::err_typecheck_non_object_not_modifiable_lvalue;
    NeedType = true;
    break;
  case Expr::MLV_LValueCast:
    Diag = diag::err_typecheck_lvalue_casts_not_supported;
    break;
  case Expr::MLV_Valid:
    llvm_unreachable("did not take early return for MLV_Valid");
  case Expr::MLV_InvalidExpression:
  case Expr::MLV_MemberFunction:
  case Expr::MLV_ClassTemporary:
    Diag = diag::err_typecheck_expression_not_modifiable_lvalue;
    break;
  case Expr::MLV_IncompleteType:
  case Expr::MLV_IncompleteVoidType:
    return S.RequireCompleteType(Loc, E->getType(),
              S.PDiag(diag::err_typecheck_incomplete_type_not_modifiable_lvalue)
                  << E->getSourceRange());
  case Expr::MLV_DuplicateVectorComponents:
    Diag = diag::err_typecheck_duplicate_vector_components_not_mlvalue;
    break;
  case Expr::MLV_NotBlockQualified:
    Diag = diag::err_block_decl_ref_not_modifiable_lvalue;
    break;
  case Expr::MLV_ReadonlyProperty:
    Diag = diag::error_readonly_property_assignment;
    break;
  case Expr::MLV_NoSetterProperty:
    Diag = diag::error_nosetter_property_assignment;
    break;
  case Expr::MLV_SubObjCPropertySetting:
    Diag = diag::error_no_subobject_property_setting;
    break;
  }

  SourceRange Assign;
  if (Loc != OrigLoc)
    Assign = SourceRange(OrigLoc, OrigLoc);
  if (NeedType)
    S.Diag(Loc, Diag) << E->getType() << E->getSourceRange() << Assign;
  else
    S.Diag(Loc, Diag) << E->getSourceRange() << Assign;
  return true;
}



// C99 6.5.16.1
QualType Sema::CheckAssignmentOperands(Expr *LHS, Expr *&RHS,
                                       SourceLocation Loc,
                                       QualType CompoundType) {
  // Verify that LHS is a modifiable lvalue, and emit error if not.
  if (CheckForModifiableLvalue(LHS, Loc, *this))
    return QualType();

  QualType LHSType = LHS->getType();
  QualType RHSType = CompoundType.isNull() ? RHS->getType() : CompoundType;
  AssignConvertType ConvTy;
  if (CompoundType.isNull()) {
    QualType LHSTy(LHSType);
    // Simple assignment "x = y".
    if (const ObjCImplicitSetterGetterRefExpr *OISGE = 
        dyn_cast<ObjCImplicitSetterGetterRefExpr>(LHS)) {
      // If using property-dot syntax notation for assignment, and there is a
      // setter, RHS expression is being passed to the setter argument. So,
      // type conversion (and comparison) is RHS to setter's argument type.
      if (const ObjCMethodDecl *SetterMD = OISGE->getSetterMethod()) {
        ObjCMethodDecl::param_iterator P = SetterMD->param_begin();
        LHSTy = (*P)->getType();
      }
    }
    
    ConvTy = CheckSingleAssignmentConstraints(LHSTy, RHS);
    // Special case of NSObject attributes on c-style pointer types.
    if (ConvTy == IncompatiblePointer &&
        ((Context.isObjCNSObjectType(LHSType) &&
          RHSType->isObjCObjectPointerType()) ||
         (Context.isObjCNSObjectType(RHSType) &&
          LHSType->isObjCObjectPointerType())))
      ConvTy = Compatible;

    // If the RHS is a unary plus or minus, check to see if they = and + are
    // right next to each other.  If so, the user may have typo'd "x =+ 4"
    // instead of "x += 4".
    Expr *RHSCheck = RHS;
    if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(RHSCheck))
      RHSCheck = ICE->getSubExpr();
    if (UnaryOperator *UO = dyn_cast<UnaryOperator>(RHSCheck)) {
      if ((UO->getOpcode() == UO_Plus ||
           UO->getOpcode() == UO_Minus) &&
          Loc.isFileID() && UO->getOperatorLoc().isFileID() &&
          // Only if the two operators are exactly adjacent.
          Loc.getFileLocWithOffset(1) == UO->getOperatorLoc() &&
          // And there is a space or other character before the subexpr of the
          // unary +/-.  We don't want to warn on "x=-1".
          Loc.getFileLocWithOffset(2) != UO->getSubExpr()->getLocStart() &&
          UO->getSubExpr()->getLocStart().isFileID()) {
        Diag(Loc, diag::warn_not_compound_assign)
          << (UO->getOpcode() == UO_Plus ? "+" : "-")
          << SourceRange(UO->getOperatorLoc(), UO->getOperatorLoc());
      }
    }
  } else {
    // Compound assignment "x += y"
    ConvTy = CheckAssignmentConstraints(LHSType, RHSType);
  }

  if (DiagnoseAssignmentResult(ConvTy, Loc, LHSType, RHSType,
                               RHS, AA_Assigning))
    return QualType();

  
  // Check to see if the destination operand is a dereferenced null pointer.  If
  // so, and if not volatile-qualified, this is undefined behavior that the
  // optimizer will delete, so warn about it.  People sometimes try to use this
  // to get a deterministic trap and are surprised by clang's behavior.  This
  // only handles the pattern "*null = whatever", which is a very syntactic
  // check.
  if (UnaryOperator *UO = dyn_cast<UnaryOperator>(LHS->IgnoreParenCasts()))
    if (UO->getOpcode() == UO_Deref &&
        UO->getSubExpr()->IgnoreParenCasts()->
          isNullPointerConstant(Context, Expr::NPC_ValueDependentIsNotNull) &&
        !UO->getType().isVolatileQualified()) {
    Diag(UO->getOperatorLoc(), diag::warn_indirection_through_null)
        << UO->getSubExpr()->getSourceRange();
    Diag(UO->getOperatorLoc(), diag::note_indirection_through_null);
  }
  
  // C99 6.5.16p3: The type of an assignment expression is the type of the
  // left operand unless the left operand has qualified type, in which case
  // it is the unqualified version of the type of the left operand.
  // C99 6.5.16.1p2: In simple assignment, the value of the right operand
  // is converted to the type of the assignment expression (above).
  // C++ 5.17p1: the type of the assignment expression is that of its left
  // operand.
  return LHSType.getUnqualifiedType();
}

// C99 6.5.17
QualType Sema::CheckCommaOperands(Expr *LHS, Expr *&RHS, SourceLocation Loc) {
  DiagnoseUnusedExprResult(LHS);

  // Comma performs lvalue conversion (C99 6.3.2.1), but not unary conversions.
  // C++ does not perform this conversion (C++ [expr.comma]p1).
  if (!getLangOptions().CPlusPlus)
    DefaultFunctionArrayLvalueConversion(RHS);

  // FIXME: Check that RHS type is complete in C mode (it's legal for it to be
  // incomplete in C++).

  return RHS->getType();
}

/// CheckIncrementDecrementOperand - unlike most "Check" methods, this routine
/// doesn't need to call UsualUnaryConversions or UsualArithmeticConversions.
QualType Sema::CheckIncrementDecrementOperand(Expr *Op, SourceLocation OpLoc,
                                              bool isInc, bool isPrefix) {
  if (Op->isTypeDependent())
    return Context.DependentTy;

  QualType ResType = Op->getType();
  assert(!ResType.isNull() && "no type for increment/decrement expression");

  if (getLangOptions().CPlusPlus && ResType->isBooleanType()) {
    // Decrement of bool is not allowed.
    if (!isInc) {
      Diag(OpLoc, diag::err_decrement_bool) << Op->getSourceRange();
      return QualType();
    }
    // Increment of bool sets it to true, but is deprecated.
    Diag(OpLoc, diag::warn_increment_bool) << Op->getSourceRange();
  } else if (ResType->isRealType()) {
    // OK!
  } else if (ResType->isAnyPointerType()) {
    QualType PointeeTy = ResType->getPointeeType();

    // C99 6.5.2.4p2, 6.5.6p2
    if (PointeeTy->isVoidType()) {
      if (getLangOptions().CPlusPlus) {
        Diag(OpLoc, diag::err_typecheck_pointer_arith_void_type)
          << Op->getSourceRange();
        return QualType();
      }

      // Pointer to void is a GNU extension in C.
      Diag(OpLoc, diag::ext_gnu_void_ptr) << Op->getSourceRange();
    } else if (PointeeTy->isFunctionType()) {
      if (getLangOptions().CPlusPlus) {
        Diag(OpLoc, diag::err_typecheck_pointer_arith_function_type)
          << Op->getType() << Op->getSourceRange();
        return QualType();
      }

      Diag(OpLoc, diag::ext_gnu_ptr_func_arith)
        << ResType << Op->getSourceRange();
    } else if (RequireCompleteType(OpLoc, PointeeTy,
                           PDiag(diag::err_typecheck_arithmetic_incomplete_type)
                             << Op->getSourceRange()
                             << ResType))
      return QualType();
    // Diagnose bad cases where we step over interface counts.
    else if (PointeeTy->isObjCObjectType() && LangOpts.ObjCNonFragileABI) {
      Diag(OpLoc, diag::err_arithmetic_nonfragile_interface)
        << PointeeTy << Op->getSourceRange();
      return QualType();
    }
  } else if (ResType->isAnyComplexType()) {
    // C99 does not support ++/-- on complex types, we allow as an extension.
    Diag(OpLoc, diag::ext_integer_increment_complex)
      << ResType << Op->getSourceRange();
  } else {
    Diag(OpLoc, diag::err_typecheck_illegal_increment_decrement)
      << ResType << int(isInc) << Op->getSourceRange();
    return QualType();
  }
  // At this point, we know we have a real, complex or pointer type.
  // Now make sure the operand is a modifiable lvalue.
  if (CheckForModifiableLvalue(Op, OpLoc, *this))
    return QualType();
  // In C++, a prefix increment is the same type as the operand. Otherwise
  // (in C or with postfix), the increment is the unqualified type of the
  // operand.
  return isPrefix && getLangOptions().CPlusPlus
    ? ResType : ResType.getUnqualifiedType();
}

/// getPrimaryDecl - Helper function for CheckAddressOfOperand().
/// This routine allows us to typecheck complex/recursive expressions
/// where the declaration is needed for type checking. We only need to
/// handle cases when the expression references a function designator
/// or is an lvalue. Here are some examples:
///  - &(x) => x
///  - &*****f => f for f a function designator.
///  - &s.xx => s
///  - &s.zz[1].yy -> s, if zz is an array
///  - *(x + 1) -> x, if x is an array
///  - &"123"[2] -> 0
///  - & __real__ x -> x
static NamedDecl *getPrimaryDecl(Expr *E) {
  switch (E->getStmtClass()) {
  case Stmt::DeclRefExprClass:
    return cast<DeclRefExpr>(E)->getDecl();
  case Stmt::MemberExprClass:
    // If this is an arrow operator, the address is an offset from
    // the base's value, so the object the base refers to is
    // irrelevant.
    if (cast<MemberExpr>(E)->isArrow())
      return 0;
    // Otherwise, the expression refers to a part of the base
    return getPrimaryDecl(cast<MemberExpr>(E)->getBase());
  case Stmt::ArraySubscriptExprClass: {
    // FIXME: This code shouldn't be necessary!  We should catch the implicit
    // promotion of register arrays earlier.
    Expr* Base = cast<ArraySubscriptExpr>(E)->getBase();
    if (ImplicitCastExpr* ICE = dyn_cast<ImplicitCastExpr>(Base)) {
      if (ICE->getSubExpr()->getType()->isArrayType())
        return getPrimaryDecl(ICE->getSubExpr());
    }
    return 0;
  }
  case Stmt::UnaryOperatorClass: {
    UnaryOperator *UO = cast<UnaryOperator>(E);

    switch(UO->getOpcode()) {
    case UO_Real:
    case UO_Imag:
    case UO_Extension:
      return getPrimaryDecl(UO->getSubExpr());
    default:
      return 0;
    }
  }
  case Stmt::ParenExprClass:
    return getPrimaryDecl(cast<ParenExpr>(E)->getSubExpr());
  case Stmt::ImplicitCastExprClass:
    // If the result of an implicit cast is an l-value, we care about
    // the sub-expression; otherwise, the result here doesn't matter.
    return getPrimaryDecl(cast<ImplicitCastExpr>(E)->getSubExpr());
  default:
    return 0;
  }
}

/// CheckAddressOfOperand - The operand of & must be either a function
/// designator or an lvalue designating an object. If it is an lvalue, the
/// object cannot be declared with storage class register or be a bit field.
/// Note: The usual conversions are *not* applied to the operand of the &
/// operator (C99 6.3.2.1p[2-4]), and its result is never an lvalue.
/// In C++, the operand might be an overloaded function name, in which case
/// we allow the '&' but retain the overloaded-function type.
QualType Sema::CheckAddressOfOperand(Expr *OrigOp, SourceLocation OpLoc) {
  if (OrigOp->isTypeDependent())
    return Context.DependentTy;
  if (OrigOp->getType() == Context.OverloadTy)
    return Context.OverloadTy;

  // Make sure to ignore parentheses in subsequent checks
  Expr *op = OrigOp->IgnoreParens();

  if (getLangOptions().C99) {
    // Implement C99-only parts of addressof rules.
    if (UnaryOperator* uOp = dyn_cast<UnaryOperator>(op)) {
      if (uOp->getOpcode() == UO_Deref)
        // Per C99 6.5.3.2, the address of a deref always returns a valid result
        // (assuming the deref expression is valid).
        return uOp->getSubExpr()->getType();
    }
    // Technically, there should be a check for array subscript
    // expressions here, but the result of one is always an lvalue anyway.
  }
  NamedDecl *dcl = getPrimaryDecl(op);
  Expr::isLvalueResult lval = op->isLvalue(Context);

  if (lval == Expr::LV_ClassTemporary) {
    Diag(OpLoc, isSFINAEContext()? diag::err_typecheck_addrof_class_temporary
                                 : diag::ext_typecheck_addrof_class_temporary)
      << op->getType() << op->getSourceRange();
    if (isSFINAEContext())
      return QualType();
  } else if (isa<ObjCSelectorExpr>(op)) {
    return Context.getPointerType(op->getType());
  } else if (lval == Expr::LV_MemberFunction) {
    // If it's an instance method, make a member pointer.
    // The expression must have exactly the form &A::foo.

    // If the underlying expression isn't a decl ref, give up.
    if (!isa<DeclRefExpr>(op)) {
      Diag(OpLoc, diag::err_invalid_form_pointer_member_function)
        << OrigOp->getSourceRange();
      return QualType();
    }
    DeclRefExpr *DRE = cast<DeclRefExpr>(op);
    CXXMethodDecl *MD = cast<CXXMethodDecl>(DRE->getDecl());

    // The id-expression was parenthesized.
    if (OrigOp != DRE) {
      Diag(OpLoc, diag::err_parens_pointer_member_function)
        << OrigOp->getSourceRange();

    // The method was named without a qualifier.
    } else if (!DRE->getQualifier()) {
      Diag(OpLoc, diag::err_unqualified_pointer_member_function)
        << op->getSourceRange();
    }

    return Context.getMemberPointerType(op->getType(),
              Context.getTypeDeclType(MD->getParent()).getTypePtr());
  } else if (lval != Expr::LV_Valid && lval != Expr::LV_IncompleteVoidType) {
    // C99 6.5.3.2p1
    // The operand must be either an l-value or a function designator
    if (!op->getType()->isFunctionType()) {
      // FIXME: emit more specific diag...
      Diag(OpLoc, diag::err_typecheck_invalid_lvalue_addrof)
        << op->getSourceRange();
      return QualType();
    }
  } else if (op->getBitField()) { // C99 6.5.3.2p1
    // The operand cannot be a bit-field
    Diag(OpLoc, diag::err_typecheck_address_of)
      << "bit-field" << op->getSourceRange();
        return QualType();
  } else if (op->refersToVectorElement()) {
    // The operand cannot be an element of a vector
    Diag(OpLoc, diag::err_typecheck_address_of)
      << "vector element" << op->getSourceRange();
    return QualType();
  } else if (isa<ObjCPropertyRefExpr>(op)) {
    // cannot take address of a property expression.
    Diag(OpLoc, diag::err_typecheck_address_of)
      << "property expression" << op->getSourceRange();
    return QualType();
  } else if (ConditionalOperator *CO = dyn_cast<ConditionalOperator>(op)) {
    // FIXME: Can LHS ever be null here?
    if (!CheckAddressOfOperand(CO->getTrueExpr(), OpLoc).isNull())
      return CheckAddressOfOperand(CO->getFalseExpr(), OpLoc);
  } else if (dcl) { // C99 6.5.3.2p1
    // We have an lvalue with a decl. Make sure the decl is not declared
    // with the register storage-class specifier.
    if (const VarDecl *vd = dyn_cast<VarDecl>(dcl)) {
      // in C++ it is not error to take address of a register
      // variable (c++03 7.1.1P3)
      if (vd->getStorageClass() == SC_Register &&
          !getLangOptions().CPlusPlus) {
        Diag(OpLoc, diag::err_typecheck_address_of)
          << "register variable" << op->getSourceRange();
        return QualType();
      }
    } else if (isa<FunctionTemplateDecl>(dcl)) {
      return Context.OverloadTy;
    } else if (FieldDecl *FD = dyn_cast<FieldDecl>(dcl)) {
      // Okay: we can take the address of a field.
      // Could be a pointer to member, though, if there is an explicit
      // scope qualifier for the class.
      if (isa<DeclRefExpr>(op) && cast<DeclRefExpr>(op)->getQualifier()) {
        DeclContext *Ctx = dcl->getDeclContext();
        if (Ctx && Ctx->isRecord()) {
          if (FD->getType()->isReferenceType()) {
            Diag(OpLoc,
                 diag::err_cannot_form_pointer_to_member_of_reference_type)
              << FD->getDeclName() << FD->getType();
            return QualType();
          }

          return Context.getMemberPointerType(op->getType(),
                Context.getTypeDeclType(cast<RecordDecl>(Ctx)).getTypePtr());
        }
      }
    } else if (!isa<FunctionDecl>(dcl))
      assert(0 && "Unknown/unexpected decl type");
  }

  if (lval == Expr::LV_IncompleteVoidType) {
    // Taking the address of a void variable is technically illegal, but we
    // allow it in cases which are otherwise valid.
    // Example: "extern void x; void* y = &x;".
    Diag(OpLoc, diag::ext_typecheck_addrof_void) << op->getSourceRange();
  }

  // If the operand has type "type", the result has type "pointer to type".
  if (op->getType()->isObjCObjectType())
    return Context.getObjCObjectPointerType(op->getType());
  return Context.getPointerType(op->getType());
}

/// CheckIndirectionOperand - Type check unary indirection (prefix '*').
QualType Sema::CheckIndirectionOperand(Expr *Op, SourceLocation OpLoc) {
  if (Op->isTypeDependent())
    return Context.DependentTy;

  UsualUnaryConversions(Op);
  QualType OpTy = Op->getType();
  QualType Result;
  
  // Note that per both C89 and C99, indirection is always legal, even if OpTy
  // is an incomplete type or void.  It would be possible to warn about
  // dereferencing a void pointer, but it's completely well-defined, and such a
  // warning is unlikely to catch any mistakes.
  if (const PointerType *PT = OpTy->getAs<PointerType>())
    Result = PT->getPointeeType();
  else if (const ObjCObjectPointerType *OPT =
             OpTy->getAs<ObjCObjectPointerType>())
    Result = OPT->getPointeeType();

  if (Result.isNull()) {
    Diag(OpLoc, diag::err_typecheck_indirection_requires_pointer)
      << OpTy << Op->getSourceRange();
    return QualType();
  }
  
  return Result;
}

static inline BinaryOperatorKind ConvertTokenKindToBinaryOpcode(
  tok::TokenKind Kind) {
  BinaryOperatorKind Opc;
  switch (Kind) {
  default: assert(0 && "Unknown binop!");
  case tok::periodstar:           Opc = BO_PtrMemD; break;
  case tok::arrowstar:            Opc = BO_PtrMemI; break;
  case tok::star:                 Opc = BO_Mul; break;
  case tok::slash:                Opc = BO_Div; break;
  case tok::percent:              Opc = BO_Rem; break;
  case tok::plus:                 Opc = BO_Add; break;
  case tok::minus:                Opc = BO_Sub; break;
  case tok::lessless:             Opc = BO_Shl; break;
  case tok::greatergreater:       Opc = BO_Shr; break;
  case tok::lessequal:            Opc = BO_LE; break;
  case tok::less:                 Opc = BO_LT; break;
  case tok::greaterequal:         Opc = BO_GE; break;
  case tok::greater:              Opc = BO_GT; break;
  case tok::exclaimequal:         Opc = BO_NE; break;
  case tok::equalequal:           Opc = BO_EQ; break;
  case tok::amp:                  Opc = BO_And; break;
  case tok::caret:                Opc = BO_Xor; break;
  case tok::pipe:                 Opc = BO_Or; break;
  case tok::ampamp:               Opc = BO_LAnd; break;
  case tok::pipepipe:             Opc = BO_LOr; break;
  case tok::equal:                Opc = BO_Assign; break;
  case tok::starequal:            Opc = BO_MulAssign; break;
  case tok::slashequal:           Opc = BO_DivAssign; break;
  case tok::percentequal:         Opc = BO_RemAssign; break;
  case tok::plusequal:            Opc = BO_AddAssign; break;
  case tok::minusequal:           Opc = BO_SubAssign; break;
  case tok::lesslessequal:        Opc = BO_ShlAssign; break;
  case tok::greatergreaterequal:  Opc = BO_ShrAssign; break;
  case tok::ampequal:             Opc = BO_AndAssign; break;
  case tok::caretequal:           Opc = BO_XorAssign; break;
  case tok::pipeequal:            Opc = BO_OrAssign; break;
  case tok::comma:                Opc = BO_Comma; break;
  }
  return Opc;
}

static inline UnaryOperatorKind ConvertTokenKindToUnaryOpcode(
  tok::TokenKind Kind) {
  UnaryOperatorKind Opc;
  switch (Kind) {
  default: assert(0 && "Unknown unary op!");
  case tok::plusplus:     Opc = UO_PreInc; break;
  case tok::minusminus:   Opc = UO_PreDec; break;
  case tok::amp:          Opc = UO_AddrOf; break;
  case tok::star:         Opc = UO_Deref; break;
  case tok::plus:         Opc = UO_Plus; break;
  case tok::minus:        Opc = UO_Minus; break;
  case tok::tilde:        Opc = UO_Not; break;
  case tok::exclaim:      Opc = UO_LNot; break;
  case tok::kw___real:    Opc = UO_Real; break;
  case tok::kw___imag:    Opc = UO_Imag; break;
  case tok::kw___extension__: Opc = UO_Extension; break;
  }
  return Opc;
}

/// CreateBuiltinBinOp - Creates a new built-in binary operation with
/// operator @p Opc at location @c TokLoc. This routine only supports
/// built-in operations; ActOnBinOp handles overloaded operators.
ExprResult Sema::CreateBuiltinBinOp(SourceLocation OpLoc,
                                    unsigned Op,
                                    Expr *lhs, Expr *rhs) {
  QualType ResultTy;     // Result type of the binary operator.
  BinaryOperatorKind Opc = (BinaryOperatorKind) Op;
  // The following two variables are used for compound assignment operators
  QualType CompLHSTy;    // Type of LHS after promotions for computation
  QualType CompResultTy; // Type of computation result

  switch (Opc) {
  case BO_Assign:
    ResultTy = CheckAssignmentOperands(lhs, rhs, OpLoc, QualType());
    break;
  case BO_PtrMemD:
  case BO_PtrMemI:
    ResultTy = CheckPointerToMemberOperands(lhs, rhs, OpLoc,
                                            Opc == BO_PtrMemI);
    break;
  case BO_Mul:
  case BO_Div:
    ResultTy = CheckMultiplyDivideOperands(lhs, rhs, OpLoc, false,
                                           Opc == BO_Div);
    break;
  case BO_Rem:
    ResultTy = CheckRemainderOperands(lhs, rhs, OpLoc);
    break;
  case BO_Add:
    ResultTy = CheckAdditionOperands(lhs, rhs, OpLoc);
    break;
  case BO_Sub:
    ResultTy = CheckSubtractionOperands(lhs, rhs, OpLoc);
    break;
  case BO_Shl:
  case BO_Shr:
    ResultTy = CheckShiftOperands(lhs, rhs, OpLoc);
    break;
  case BO_LE:
  case BO_LT:
  case BO_GE:
  case BO_GT:
    ResultTy = CheckCompareOperands(lhs, rhs, OpLoc, Opc, true);
    break;
  case BO_EQ:
  case BO_NE:
    ResultTy = CheckCompareOperands(lhs, rhs, OpLoc, Opc, false);
    break;
  case BO_And:
  case BO_Xor:
  case BO_Or:
    ResultTy = CheckBitwiseOperands(lhs, rhs, OpLoc);
    break;
  case BO_LAnd:
  case BO_LOr:
    ResultTy = CheckLogicalOperands(lhs, rhs, OpLoc, Opc);
    break;
  case BO_MulAssign:
  case BO_DivAssign:
    CompResultTy = CheckMultiplyDivideOperands(lhs, rhs, OpLoc, true,
                                              Opc == BO_DivAssign);
    CompLHSTy = CompResultTy;
    if (!CompResultTy.isNull())
      ResultTy = CheckAssignmentOperands(lhs, rhs, OpLoc, CompResultTy);
    break;
  case BO_RemAssign:
    CompResultTy = CheckRemainderOperands(lhs, rhs, OpLoc, true);
    CompLHSTy = CompResultTy;
    if (!CompResultTy.isNull())
      ResultTy = CheckAssignmentOperands(lhs, rhs, OpLoc, CompResultTy);
    break;
  case BO_AddAssign:
    CompResultTy = CheckAdditionOperands(lhs, rhs, OpLoc, &CompLHSTy);
    if (!CompResultTy.isNull())
      ResultTy = CheckAssignmentOperands(lhs, rhs, OpLoc, CompResultTy);
    break;
  case BO_SubAssign:
    CompResultTy = CheckSubtractionOperands(lhs, rhs, OpLoc, &CompLHSTy);
    if (!CompResultTy.isNull())
      ResultTy = CheckAssignmentOperands(lhs, rhs, OpLoc, CompResultTy);
    break;
  case BO_ShlAssign:
  case BO_ShrAssign:
    CompResultTy = CheckShiftOperands(lhs, rhs, OpLoc, true);
    CompLHSTy = CompResultTy;
    if (!CompResultTy.isNull())
      ResultTy = CheckAssignmentOperands(lhs, rhs, OpLoc, CompResultTy);
    break;
  case BO_AndAssign:
  case BO_XorAssign:
  case BO_OrAssign:
    CompResultTy = CheckBitwiseOperands(lhs, rhs, OpLoc, true);
    CompLHSTy = CompResultTy;
    if (!CompResultTy.isNull())
      ResultTy = CheckAssignmentOperands(lhs, rhs, OpLoc, CompResultTy);
    break;
  case BO_Comma:
    ResultTy = CheckCommaOperands(lhs, rhs, OpLoc);
    break;
  }
  if (ResultTy.isNull())
    return ExprError();
  if (ResultTy->isObjCObjectType() && LangOpts.ObjCNonFragileABI) {
    if (Opc >= BO_Assign && Opc <= BO_OrAssign) 
          Diag(OpLoc, diag::err_assignment_requires_nonfragile_object)
                << ResultTy;
  }
  if (CompResultTy.isNull())
    return Owned(new (Context) BinaryOperator(lhs, rhs, Opc, ResultTy, OpLoc));
  else
    return Owned(new (Context) CompoundAssignOperator(lhs, rhs, Opc, ResultTy,
                                                      CompLHSTy, CompResultTy,
                                                      OpLoc));
}

/// SuggestParentheses - Emit a diagnostic together with a fixit hint that wraps
/// ParenRange in parentheses.
static void SuggestParentheses(Sema &Self, SourceLocation Loc,
                               const PartialDiagnostic &PD,
                               const PartialDiagnostic &FirstNote,
                               SourceRange FirstParenRange,
                               const PartialDiagnostic &SecondNote,
                               SourceRange SecondParenRange) {
  Self.Diag(Loc, PD);

  if (!FirstNote.getDiagID())
    return;

  SourceLocation EndLoc = Self.PP.getLocForEndOfToken(FirstParenRange.getEnd());
  if (!FirstParenRange.getEnd().isFileID() || EndLoc.isInvalid()) {
    // We can't display the parentheses, so just return.
    return;
  }

  Self.Diag(Loc, FirstNote)
    << FixItHint::CreateInsertion(FirstParenRange.getBegin(), "(")
    << FixItHint::CreateInsertion(EndLoc, ")");

  if (!SecondNote.getDiagID())
    return;

  EndLoc = Self.PP.getLocForEndOfToken(SecondParenRange.getEnd());
  if (!SecondParenRange.getEnd().isFileID() || EndLoc.isInvalid()) {
    // We can't display the parentheses, so just dig the
    // warning/error and return.
    Self.Diag(Loc, SecondNote);
    return;
  }

  Self.Diag(Loc, SecondNote)
    << FixItHint::CreateInsertion(SecondParenRange.getBegin(), "(")
    << FixItHint::CreateInsertion(EndLoc, ")");
}

/// DiagnoseBitwisePrecedence - Emit a warning when bitwise and comparison
/// operators are mixed in a way that suggests that the programmer forgot that
/// comparison operators have higher precedence. The most typical example of
/// such code is "flags & 0x0020 != 0", which is equivalent to "flags & 1".
static void DiagnoseBitwisePrecedence(Sema &Self, BinaryOperatorKind Opc,
                                      SourceLocation OpLoc,Expr *lhs,Expr *rhs){
  typedef BinaryOperator BinOp;
  BinOp::Opcode lhsopc = static_cast<BinOp::Opcode>(-1),
                rhsopc = static_cast<BinOp::Opcode>(-1);
  if (BinOp *BO = dyn_cast<BinOp>(lhs))
    lhsopc = BO->getOpcode();
  if (BinOp *BO = dyn_cast<BinOp>(rhs))
    rhsopc = BO->getOpcode();

  // Subs are not binary operators.
  if (lhsopc == -1 && rhsopc == -1)
    return;

  // Bitwise operations are sometimes used as eager logical ops.
  // Don't diagnose this.
  if ((BinOp::isComparisonOp(lhsopc) || BinOp::isBitwiseOp(lhsopc)) &&
      (BinOp::isComparisonOp(rhsopc) || BinOp::isBitwiseOp(rhsopc)))
    return;

  if (BinOp::isComparisonOp(lhsopc))
    SuggestParentheses(Self, OpLoc,
      Self.PDiag(diag::warn_precedence_bitwise_rel)
          << SourceRange(lhs->getLocStart(), OpLoc)
          << BinOp::getOpcodeStr(Opc) << BinOp::getOpcodeStr(lhsopc),
      Self.PDiag(diag::note_precedence_bitwise_first)
          << BinOp::getOpcodeStr(Opc),
      SourceRange(cast<BinOp>(lhs)->getRHS()->getLocStart(), rhs->getLocEnd()),
      Self.PDiag(diag::note_precedence_bitwise_silence)
          << BinOp::getOpcodeStr(lhsopc),
                       lhs->getSourceRange());
  else if (BinOp::isComparisonOp(rhsopc))
    SuggestParentheses(Self, OpLoc,
      Self.PDiag(diag::warn_precedence_bitwise_rel)
          << SourceRange(OpLoc, rhs->getLocEnd())
          << BinOp::getOpcodeStr(Opc) << BinOp::getOpcodeStr(rhsopc),
      Self.PDiag(diag::note_precedence_bitwise_first)
        << BinOp::getOpcodeStr(Opc),
      SourceRange(lhs->getLocEnd(), cast<BinOp>(rhs)->getLHS()->getLocStart()),
      Self.PDiag(diag::note_precedence_bitwise_silence)
        << BinOp::getOpcodeStr(rhsopc),
                       rhs->getSourceRange());
}

/// DiagnoseBinOpPrecedence - Emit warnings for expressions with tricky
/// precedence. This currently diagnoses only "arg1 'bitwise' arg2 'eq' arg3".
/// But it could also warn about arg1 && arg2 || arg3, as GCC 4.3+ does.
static void DiagnoseBinOpPrecedence(Sema &Self, BinaryOperatorKind Opc,
                                    SourceLocation OpLoc, Expr *lhs, Expr *rhs){
  if (BinaryOperator::isBitwiseOp(Opc))
    DiagnoseBitwisePrecedence(Self, Opc, OpLoc, lhs, rhs);
}

// Binary Operators.  'Tok' is the token for the operator.
ExprResult Sema::ActOnBinOp(Scope *S, SourceLocation TokLoc,
                            tok::TokenKind Kind,
                            Expr *lhs, Expr *rhs) {
  BinaryOperatorKind Opc = ConvertTokenKindToBinaryOpcode(Kind);
  assert((lhs != 0) && "ActOnBinOp(): missing left expression");
  assert((rhs != 0) && "ActOnBinOp(): missing right expression");

  // Emit warnings for tricky precedence issues, e.g. "bitfield & 0x4 == 0"
  DiagnoseBinOpPrecedence(*this, Opc, TokLoc, lhs, rhs);

  return BuildBinOp(S, TokLoc, Opc, lhs, rhs);
}

ExprResult Sema::BuildBinOp(Scope *S, SourceLocation OpLoc,
                            BinaryOperatorKind Opc,
                            Expr *lhs, Expr *rhs) {
  if (getLangOptions().CPlusPlus &&
      (lhs->getType()->isOverloadableType() ||
       rhs->getType()->isOverloadableType())) {
    // Find all of the overloaded operators visible from this
    // point. We perform both an operator-name lookup from the local
    // scope and an argument-dependent lookup based on the types of
    // the arguments.
    UnresolvedSet<16> Functions;
    OverloadedOperatorKind OverOp = BinaryOperator::getOverloadedOperator(Opc);
    if (S && OverOp != OO_None)
      LookupOverloadedOperatorName(OverOp, S, lhs->getType(), rhs->getType(),
                                   Functions);

    // Build the (potentially-overloaded, potentially-dependent)
    // binary operation.
    return CreateOverloadedBinOp(OpLoc, Opc, Functions, lhs, rhs);
  }

  // Build a built-in binary operation.
  return CreateBuiltinBinOp(OpLoc, Opc, lhs, rhs);
}

ExprResult Sema::CreateBuiltinUnaryOp(SourceLocation OpLoc,
                                                    unsigned OpcIn,
                                                    Expr *Input) {
  UnaryOperatorKind Opc = static_cast<UnaryOperatorKind>(OpcIn);

  QualType resultType;
  switch (Opc) {
  case UO_PreInc:
  case UO_PreDec:
  case UO_PostInc:
  case UO_PostDec:
    resultType = CheckIncrementDecrementOperand(Input, OpLoc,
                                                Opc == UO_PreInc ||
                                                Opc == UO_PostInc,
                                                Opc == UO_PreInc ||
                                                Opc == UO_PreDec);
    break;
  case UO_AddrOf:
    resultType = CheckAddressOfOperand(Input, OpLoc);
    break;
  case UO_Deref:
    DefaultFunctionArrayLvalueConversion(Input);
    resultType = CheckIndirectionOperand(Input, OpLoc);
    break;
  case UO_Plus:
  case UO_Minus:
    UsualUnaryConversions(Input);
    resultType = Input->getType();
    if (resultType->isDependentType())
      break;
    if (resultType->isArithmeticType() || // C99 6.5.3.3p1
        resultType->isVectorType()) 
      break;
    else if (getLangOptions().CPlusPlus && // C++ [expr.unary.op]p6-7
             resultType->isEnumeralType())
      break;
    else if (getLangOptions().CPlusPlus && // C++ [expr.unary.op]p6
             Opc == UO_Plus &&
             resultType->isPointerType())
      break;

    return ExprError(Diag(OpLoc, diag::err_typecheck_unary_expr)
      << resultType << Input->getSourceRange());
  case UO_Not: // bitwise complement
    UsualUnaryConversions(Input);
    resultType = Input->getType();
    if (resultType->isDependentType())
      break;
    // C99 6.5.3.3p1. We allow complex int and float as a GCC extension.
    if (resultType->isComplexType() || resultType->isComplexIntegerType())
      // C99 does not support '~' for complex conjugation.
      Diag(OpLoc, diag::ext_integer_complement_complex)
        << resultType << Input->getSourceRange();
    else if (!resultType->hasIntegerRepresentation())
      return ExprError(Diag(OpLoc, diag::err_typecheck_unary_expr)
        << resultType << Input->getSourceRange());
    break;
  case UO_LNot: // logical negation
    // Unlike +/-/~, integer promotions aren't done here (C99 6.5.3.3p5).
    DefaultFunctionArrayLvalueConversion(Input);
    resultType = Input->getType();
    if (resultType->isDependentType())
      break;
    if (!resultType->isScalarType()) // C99 6.5.3.3p1
      return ExprError(Diag(OpLoc, diag::err_typecheck_unary_expr)
        << resultType << Input->getSourceRange());
    // LNot always has type int. C99 6.5.3.3p5.
    // In C++, it's bool. C++ 5.3.1p8
    resultType = getLangOptions().CPlusPlus ? Context.BoolTy : Context.IntTy;
    break;
  case UO_Real:
  case UO_Imag:
    resultType = CheckRealImagOperand(Input, OpLoc, Opc == UO_Real);
    break;
  case UO_Extension:
    resultType = Input->getType();
    break;
  }
  if (resultType.isNull())
    return ExprError();

  return Owned(new (Context) UnaryOperator(Input, Opc, resultType, OpLoc));
}

ExprResult Sema::BuildUnaryOp(Scope *S, SourceLocation OpLoc,
                              UnaryOperatorKind Opc,
                              Expr *Input) {
  if (getLangOptions().CPlusPlus && Input->getType()->isOverloadableType() &&
      UnaryOperator::getOverloadedOperator(Opc) != OO_None) {
    // Find all of the overloaded operators visible from this
    // point. We perform both an operator-name lookup from the local
    // scope and an argument-dependent lookup based on the types of
    // the arguments.
    UnresolvedSet<16> Functions;
    OverloadedOperatorKind OverOp = UnaryOperator::getOverloadedOperator(Opc);
    if (S && OverOp != OO_None)
      LookupOverloadedOperatorName(OverOp, S, Input->getType(), QualType(),
                                   Functions);

    return CreateOverloadedUnaryOp(OpLoc, Opc, Functions, Input);
  }

  return CreateBuiltinUnaryOp(OpLoc, Opc, Input);
}

// Unary Operators.  'Tok' is the token for the operator.
ExprResult Sema::ActOnUnaryOp(Scope *S, SourceLocation OpLoc,
                                            tok::TokenKind Op, Expr *Input) {
  return BuildUnaryOp(S, OpLoc, ConvertTokenKindToUnaryOpcode(Op), Input);
}

/// ActOnAddrLabel - Parse the GNU address of label extension: "&&foo".
ExprResult Sema::ActOnAddrLabel(SourceLocation OpLoc,
                                            SourceLocation LabLoc,
                                            IdentifierInfo *LabelII) {
  // Look up the record for this label identifier.
  LabelStmt *&LabelDecl = getCurFunction()->LabelMap[LabelII];

  // If we haven't seen this label yet, create a forward reference. It
  // will be validated and/or cleaned up in ActOnFinishFunctionBody.
  if (LabelDecl == 0)
    LabelDecl = new (Context) LabelStmt(LabLoc, LabelII, 0);

  // Create the AST node.  The address of a label always has type 'void*'.
  return Owned(new (Context) AddrLabelExpr(OpLoc, LabLoc, LabelDecl,
                                       Context.getPointerType(Context.VoidTy)));
}

ExprResult
Sema::ActOnStmtExpr(SourceLocation LPLoc, Stmt *SubStmt,
                    SourceLocation RPLoc) { // "({..})"
  assert(SubStmt && isa<CompoundStmt>(SubStmt) && "Invalid action invocation!");
  CompoundStmt *Compound = cast<CompoundStmt>(SubStmt);

  bool isFileScope
    = (getCurFunctionOrMethodDecl() == 0) && (getCurBlock() == 0);
  if (isFileScope)
    return ExprError(Diag(LPLoc, diag::err_stmtexpr_file_scope));

  // FIXME: there are a variety of strange constraints to enforce here, for
  // example, it is not possible to goto into a stmt expression apparently.
  // More semantic analysis is needed.

  // If there are sub stmts in the compound stmt, take the type of the last one
  // as the type of the stmtexpr.
  QualType Ty = Context.VoidTy;

  if (!Compound->body_empty()) {
    Stmt *LastStmt = Compound->body_back();
    // If LastStmt is a label, skip down through into the body.
    while (LabelStmt *Label = dyn_cast<LabelStmt>(LastStmt))
      LastStmt = Label->getSubStmt();

    if (Expr *LastExpr = dyn_cast<Expr>(LastStmt))
      Ty = LastExpr->getType();
  }

  // FIXME: Check that expression type is complete/non-abstract; statement
  // expressions are not lvalues.

  return Owned(new (Context) StmtExpr(Compound, Ty, LPLoc, RPLoc));
}

ExprResult Sema::BuildBuiltinOffsetOf(SourceLocation BuiltinLoc,
                                                  TypeSourceInfo *TInfo,
                                                  OffsetOfComponent *CompPtr,
                                                  unsigned NumComponents,
                                                  SourceLocation RParenLoc) {
  QualType ArgTy = TInfo->getType();
  bool Dependent = ArgTy->isDependentType();
  SourceRange TypeRange = TInfo->getTypeLoc().getLocalSourceRange();
  
  // We must have at least one component that refers to the type, and the first
  // one is known to be a field designator.  Verify that the ArgTy represents
  // a struct/union/class.
  if (!Dependent && !ArgTy->isRecordType())
    return ExprError(Diag(BuiltinLoc, diag::err_offsetof_record_type) 
                       << ArgTy << TypeRange);
  
  // Type must be complete per C99 7.17p3 because a declaring a variable
  // with an incomplete type would be ill-formed.
  if (!Dependent 
      && RequireCompleteType(BuiltinLoc, ArgTy,
                             PDiag(diag::err_offsetof_incomplete_type)
                               << TypeRange))
    return ExprError();
  
  // offsetof with non-identifier designators (e.g. "offsetof(x, a.b[c])") are a
  // GCC extension, diagnose them.
  // FIXME: This diagnostic isn't actually visible because the location is in
  // a system header!
  if (NumComponents != 1)
    Diag(BuiltinLoc, diag::ext_offsetof_extended_field_designator)
      << SourceRange(CompPtr[1].LocStart, CompPtr[NumComponents-1].LocEnd);
  
  bool DidWarnAboutNonPOD = false;
  QualType CurrentType = ArgTy;
  typedef OffsetOfExpr::OffsetOfNode OffsetOfNode;
  llvm::SmallVector<OffsetOfNode, 4> Comps;
  llvm::SmallVector<Expr*, 4> Exprs;
  for (unsigned i = 0; i != NumComponents; ++i) {
    const OffsetOfComponent &OC = CompPtr[i];
    if (OC.isBrackets) {
      // Offset of an array sub-field.  TODO: Should we allow vector elements?
      if (!CurrentType->isDependentType()) {
        const ArrayType *AT = Context.getAsArrayType(CurrentType);
        if(!AT)
          return ExprError(Diag(OC.LocEnd, diag::err_offsetof_array_type)
                           << CurrentType);
        CurrentType = AT->getElementType();
      } else
        CurrentType = Context.DependentTy;
      
      // The expression must be an integral expression.
      // FIXME: An integral constant expression?
      Expr *Idx = static_cast<Expr*>(OC.U.E);
      if (!Idx->isTypeDependent() && !Idx->isValueDependent() &&
          !Idx->getType()->isIntegerType())
        return ExprError(Diag(Idx->getLocStart(),
                              diag::err_typecheck_subscript_not_integer)
                         << Idx->getSourceRange());
      
      // Record this array index.
      Comps.push_back(OffsetOfNode(OC.LocStart, Exprs.size(), OC.LocEnd));
      Exprs.push_back(Idx);
      continue;
    }
    
    // Offset of a field.
    if (CurrentType->isDependentType()) {
      // We have the offset of a field, but we can't look into the dependent
      // type. Just record the identifier of the field.
      Comps.push_back(OffsetOfNode(OC.LocStart, OC.U.IdentInfo, OC.LocEnd));
      CurrentType = Context.DependentTy;
      continue;
    }
    
    // We need to have a complete type to look into.
    if (RequireCompleteType(OC.LocStart, CurrentType,
                            diag::err_offsetof_incomplete_type))
      return ExprError();
    
    // Look for the designated field.
    const RecordType *RC = CurrentType->getAs<RecordType>();
    if (!RC) 
      return ExprError(Diag(OC.LocEnd, diag::err_offsetof_record_type)
                       << CurrentType);
    RecordDecl *RD = RC->getDecl();
    
    // C++ [lib.support.types]p5:
    //   The macro offsetof accepts a restricted set of type arguments in this
    //   International Standard. type shall be a POD structure or a POD union
    //   (clause 9).
    if (CXXRecordDecl *CRD = dyn_cast<CXXRecordDecl>(RD)) {
      if (!CRD->isPOD() && !DidWarnAboutNonPOD &&
          DiagRuntimeBehavior(BuiltinLoc,
                              PDiag(diag::warn_offsetof_non_pod_type)
                              << SourceRange(CompPtr[0].LocStart, OC.LocEnd)
                              << CurrentType))
        DidWarnAboutNonPOD = true;
    }
    
    // Look for the field.
    LookupResult R(*this, OC.U.IdentInfo, OC.LocStart, LookupMemberName);
    LookupQualifiedName(R, RD);
    FieldDecl *MemberDecl = R.getAsSingle<FieldDecl>();
    if (!MemberDecl)
      return ExprError(Diag(BuiltinLoc, diag::err_no_member)
                       << OC.U.IdentInfo << RD << SourceRange(OC.LocStart, 
                                                              OC.LocEnd));
    
    // C99 7.17p3:
    //   (If the specified member is a bit-field, the behavior is undefined.)
    //
    // We diagnose this as an error.
    if (MemberDecl->getBitWidth()) {
      Diag(OC.LocEnd, diag::err_offsetof_bitfield)
        << MemberDecl->getDeclName()
        << SourceRange(BuiltinLoc, RParenLoc);
      Diag(MemberDecl->getLocation(), diag::note_bitfield_decl);
      return ExprError();
    }

    RecordDecl *Parent = MemberDecl->getParent();
    bool AnonStructUnion = Parent->isAnonymousStructOrUnion();
    if (AnonStructUnion) {
      do {
        Parent = cast<RecordDecl>(Parent->getParent());
      } while (Parent->isAnonymousStructOrUnion());
    }

    // If the member was found in a base class, introduce OffsetOfNodes for
    // the base class indirections.
    CXXBasePaths Paths(/*FindAmbiguities=*/true, /*RecordPaths=*/true,
                       /*DetectVirtual=*/false);
    if (IsDerivedFrom(CurrentType, Context.getTypeDeclType(Parent), Paths)) {
      CXXBasePath &Path = Paths.front();
      for (CXXBasePath::iterator B = Path.begin(), BEnd = Path.end();
           B != BEnd; ++B)
        Comps.push_back(OffsetOfNode(B->Base));
    }

    if (AnonStructUnion) {
      llvm::SmallVector<FieldDecl*, 4> Path;
      BuildAnonymousStructUnionMemberPath(MemberDecl, Path);
      unsigned n = Path.size();
      for (int j = n - 1; j > -1; --j)
        Comps.push_back(OffsetOfNode(OC.LocStart, Path[j], OC.LocEnd));
    } else {
      Comps.push_back(OffsetOfNode(OC.LocStart, MemberDecl, OC.LocEnd));
    }
    CurrentType = MemberDecl->getType().getNonReferenceType(); 
  }
  
  return Owned(OffsetOfExpr::Create(Context, Context.getSizeType(), BuiltinLoc, 
                                    TInfo, Comps.data(), Comps.size(),
                                    Exprs.data(), Exprs.size(), RParenLoc));  
}

ExprResult Sema::ActOnBuiltinOffsetOf(Scope *S,
                                                  SourceLocation BuiltinLoc,
                                                  SourceLocation TypeLoc,
                                                  ParsedType argty,
                                                  OffsetOfComponent *CompPtr,
                                                  unsigned NumComponents,
                                                  SourceLocation RPLoc) {

  TypeSourceInfo *ArgTInfo;
  QualType ArgTy = GetTypeFromParser(argty, &ArgTInfo);
  if (ArgTy.isNull())
    return ExprError();

  if (!ArgTInfo)
    ArgTInfo = Context.getTrivialTypeSourceInfo(ArgTy, TypeLoc);

  return BuildBuiltinOffsetOf(BuiltinLoc, ArgTInfo, CompPtr, NumComponents, 
                              RPLoc);
}


ExprResult Sema::ActOnTypesCompatibleExpr(SourceLocation BuiltinLoc,
                                                      ParsedType arg1,ParsedType arg2,
                                                      SourceLocation RPLoc) {
  TypeSourceInfo *argTInfo1;
  QualType argT1 = GetTypeFromParser(arg1, &argTInfo1);
  TypeSourceInfo *argTInfo2;
  QualType argT2 = GetTypeFromParser(arg2, &argTInfo2);

  assert((!argT1.isNull() && !argT2.isNull()) && "Missing type argument(s)");

  return BuildTypesCompatibleExpr(BuiltinLoc, argTInfo1, argTInfo2, RPLoc);
}

ExprResult
Sema::BuildTypesCompatibleExpr(SourceLocation BuiltinLoc,
                               TypeSourceInfo *argTInfo1,
                               TypeSourceInfo *argTInfo2,
                               SourceLocation RPLoc) {
  if (getLangOptions().CPlusPlus) {
    Diag(BuiltinLoc, diag::err_types_compatible_p_in_cplusplus)
      << SourceRange(BuiltinLoc, RPLoc);
    return ExprError();
  }

  return Owned(new (Context) TypesCompatibleExpr(Context.IntTy, BuiltinLoc,
                                                 argTInfo1, argTInfo2, RPLoc));
}


ExprResult Sema::ActOnChooseExpr(SourceLocation BuiltinLoc,
                                             Expr *CondExpr,
                                             Expr *LHSExpr, Expr *RHSExpr,
                                             SourceLocation RPLoc) {
  assert((CondExpr && LHSExpr && RHSExpr) && "Missing type argument(s)");

  QualType resType;
  bool ValueDependent = false;
  if (CondExpr->isTypeDependent() || CondExpr->isValueDependent()) {
    resType = Context.DependentTy;
    ValueDependent = true;
  } else {
    // The conditional expression is required to be a constant expression.
    llvm::APSInt condEval(32);
    SourceLocation ExpLoc;
    if (!CondExpr->isIntegerConstantExpr(condEval, Context, &ExpLoc))
      return ExprError(Diag(ExpLoc,
                       diag::err_typecheck_choose_expr_requires_constant)
        << CondExpr->getSourceRange());

    // If the condition is > zero, then the AST type is the same as the LSHExpr.
    resType = condEval.getZExtValue() ? LHSExpr->getType() : RHSExpr->getType();
    ValueDependent = condEval.getZExtValue() ? LHSExpr->isValueDependent()
                                             : RHSExpr->isValueDependent();
  }

  return Owned(new (Context) ChooseExpr(BuiltinLoc, CondExpr, LHSExpr, RHSExpr,
                                        resType, RPLoc,
                                        resType->isDependentType(),
                                        ValueDependent));
}

//===----------------------------------------------------------------------===//
// Clang Extensions.
//===----------------------------------------------------------------------===//

/// ActOnBlockStart - This callback is invoked when a block literal is started.
void Sema::ActOnBlockStart(SourceLocation CaretLoc, Scope *BlockScope) {
  BlockDecl *Block = BlockDecl::Create(Context, CurContext, CaretLoc);
  PushBlockScope(BlockScope, Block);
  CurContext->addDecl(Block);
  if (BlockScope)
    PushDeclContext(BlockScope, Block);
  else
    CurContext = Block;
}

void Sema::ActOnBlockArguments(Declarator &ParamInfo, Scope *CurScope) {
  assert(ParamInfo.getIdentifier()==0 && "block-id should have no identifier!");
  BlockScopeInfo *CurBlock = getCurBlock();

  TypeSourceInfo *Sig = GetTypeForDeclarator(ParamInfo, CurScope);
  CurBlock->TheDecl->setSignatureAsWritten(Sig);
  QualType T = Sig->getType();

  bool isVariadic;
  QualType RetTy;
  if (const FunctionType *Fn = T->getAs<FunctionType>()) {
    CurBlock->FunctionType = T;
    RetTy = Fn->getResultType();
    isVariadic =
      !isa<FunctionProtoType>(Fn) || cast<FunctionProtoType>(Fn)->isVariadic();
  } else {
    RetTy = T;
    isVariadic = false;
  }

  CurBlock->TheDecl->setIsVariadic(isVariadic);

  // Don't allow returning an array by value.
  if (RetTy->isArrayType()) {
    Diag(ParamInfo.getSourceRange().getBegin(), diag::err_block_returns_array);
    return;
  }

  // Don't allow returning a objc interface by value.
  if (RetTy->isObjCObjectType()) {
    Diag(ParamInfo.getSourceRange().getBegin(),
         diag::err_object_cannot_be_passed_returned_by_value) << 0 << RetTy;
    return;
  }

  // Context.DependentTy is used as a placeholder for a missing block
  // return type.  TODO:  what should we do with declarators like:
  //   ^ * { ... }
  // If the answer is "apply template argument deduction"....
  if (RetTy != Context.DependentTy)
    CurBlock->ReturnType = RetTy;

  // Push block parameters from the declarator if we had them.
  llvm::SmallVector<ParmVarDecl*, 8> Params;
  if (isa<FunctionProtoType>(T)) {
    FunctionProtoTypeLoc TL = cast<FunctionProtoTypeLoc>(Sig->getTypeLoc());
    for (unsigned I = 0, E = TL.getNumArgs(); I != E; ++I) {
      ParmVarDecl *Param = TL.getArg(I);
      if (Param->getIdentifier() == 0 &&
          !Param->isImplicit() &&
          !Param->isInvalidDecl() &&
          !getLangOptions().CPlusPlus)
        Diag(Param->getLocation(), diag::err_parameter_name_omitted);
      Params.push_back(Param);
    }

  // Fake up parameter variables if we have a typedef, like
  //   ^ fntype { ... }
  } else if (const FunctionProtoType *Fn = T->getAs<FunctionProtoType>()) {
    for (FunctionProtoType::arg_type_iterator
           I = Fn->arg_type_begin(), E = Fn->arg_type_end(); I != E; ++I) {
      ParmVarDecl *Param =
        BuildParmVarDeclForTypedef(CurBlock->TheDecl,
                                   ParamInfo.getSourceRange().getBegin(),
                                   *I);
      Params.push_back(Param);
    }
  }

  // Set the parameters on the block decl.
  if (!Params.empty())
    CurBlock->TheDecl->setParams(Params.data(), Params.size());

  // Finally we can process decl attributes.
  ProcessDeclAttributes(CurScope, CurBlock->TheDecl, ParamInfo);

  if (!isVariadic && CurBlock->TheDecl->getAttr<SentinelAttr>()) {
    Diag(ParamInfo.getAttributes()->getLoc(),
         diag::warn_attribute_sentinel_not_variadic) << 1;
    // FIXME: remove the attribute.
  }

  // Put the parameter variables in scope.  We can bail out immediately
  // if we don't have any.
  if (Params.empty())
    return;

  bool ShouldCheckShadow =
    Diags.getDiagnosticLevel(diag::warn_decl_shadow) != Diagnostic::Ignored;

  for (BlockDecl::param_iterator AI = CurBlock->TheDecl->param_begin(),
         E = CurBlock->TheDecl->param_end(); AI != E; ++AI) {
    (*AI)->setOwningFunction(CurBlock->TheDecl);

    // If this has an identifier, add it to the scope stack.
    if ((*AI)->getIdentifier()) {
      if (ShouldCheckShadow)
        CheckShadow(CurBlock->TheScope, *AI);

      PushOnScopeChains(*AI, CurBlock->TheScope);
    }
  }
}

/// ActOnBlockError - If there is an error parsing a block, this callback
/// is invoked to pop the information about the block from the action impl.
void Sema::ActOnBlockError(SourceLocation CaretLoc, Scope *CurScope) {
  // Pop off CurBlock, handle nested blocks.
  PopDeclContext();
  PopFunctionOrBlockScope();
  // FIXME: Delete the ParmVarDecl objects as well???
}

/// ActOnBlockStmtExpr - This is called when the body of a block statement
/// literal was successfully completed.  ^(int x){...}
ExprResult Sema::ActOnBlockStmtExpr(SourceLocation CaretLoc,
                                                Stmt *Body, Scope *CurScope) {
  // If blocks are disabled, emit an error.
  if (!LangOpts.Blocks)
    Diag(CaretLoc, diag::err_blocks_disable);

  BlockScopeInfo *BSI = cast<BlockScopeInfo>(FunctionScopes.back());
  
  PopDeclContext();

  QualType RetTy = Context.VoidTy;
  if (!BSI->ReturnType.isNull())
    RetTy = BSI->ReturnType;

  bool NoReturn = BSI->TheDecl->getAttr<NoReturnAttr>();
  QualType BlockTy;

  // If the user wrote a function type in some form, try to use that.
  if (!BSI->FunctionType.isNull()) {
    const FunctionType *FTy = BSI->FunctionType->getAs<FunctionType>();

    FunctionType::ExtInfo Ext = FTy->getExtInfo();
    if (NoReturn && !Ext.getNoReturn()) Ext = Ext.withNoReturn(true);
    
    // Turn protoless block types into nullary block types.
    if (isa<FunctionNoProtoType>(FTy)) {
      BlockTy = Context.getFunctionType(RetTy, 0, 0, false, 0,
                                        false, false, 0, 0, Ext);

    // Otherwise, if we don't need to change anything about the function type,
    // preserve its sugar structure.
    } else if (FTy->getResultType() == RetTy &&
               (!NoReturn || FTy->getNoReturnAttr())) {
      BlockTy = BSI->FunctionType;

    // Otherwise, make the minimal modifications to the function type.
    } else {
      const FunctionProtoType *FPT = cast<FunctionProtoType>(FTy);
      BlockTy = Context.getFunctionType(RetTy,
                                        FPT->arg_type_begin(),
                                        FPT->getNumArgs(),
                                        FPT->isVariadic(),
                                        /*quals*/ 0,
                                        FPT->hasExceptionSpec(),
                                        FPT->hasAnyExceptionSpec(),
                                        FPT->getNumExceptions(),
                                        FPT->exception_begin(),
                                        Ext);
    }

  // If we don't have a function type, just build one from nothing.
  } else {
    BlockTy = Context.getFunctionType(RetTy, 0, 0, false, 0,
                                      false, false, 0, 0,
                             FunctionType::ExtInfo(NoReturn, 0, CC_Default));
  }

  // FIXME: Check that return/parameter types are complete/non-abstract
  DiagnoseUnusedParameters(BSI->TheDecl->param_begin(),
                           BSI->TheDecl->param_end());
  BlockTy = Context.getBlockPointerType(BlockTy);

  // If needed, diagnose invalid gotos and switches in the block.
  if (getCurFunction()->NeedsScopeChecking() && !hasAnyErrorsInThisFunction())
    DiagnoseInvalidJumps(cast<CompoundStmt>(Body));

  BSI->TheDecl->setBody(cast<CompoundStmt>(Body));

  bool Good = true;
  // Check goto/label use.
  for (llvm::DenseMap<IdentifierInfo*, LabelStmt*>::iterator
         I = BSI->LabelMap.begin(), E = BSI->LabelMap.end(); I != E; ++I) {
    LabelStmt *L = I->second;

    // Verify that we have no forward references left.  If so, there was a goto
    // or address of a label taken, but no definition of it.
    if (L->getSubStmt() != 0)
      continue;

    // Emit error.
    Diag(L->getIdentLoc(), diag::err_undeclared_label_use) << L->getName();
    Good = false;
  }
  if (!Good) {
    PopFunctionOrBlockScope();
    return ExprError();
  }

  BlockExpr *Result = new (Context) BlockExpr(BSI->TheDecl, BlockTy,
                                              BSI->hasBlockDeclRefExprs);

  // Issue any analysis-based warnings.
  const sema::AnalysisBasedWarnings::Policy &WP =
    AnalysisWarnings.getDefaultPolicy();
  AnalysisWarnings.IssueWarnings(WP, Result);

  PopFunctionOrBlockScope();
  return Owned(Result);
}

ExprResult Sema::ActOnVAArg(SourceLocation BuiltinLoc,
                                        Expr *expr, ParsedType type,
                                        SourceLocation RPLoc) {
  TypeSourceInfo *TInfo;
  QualType T = GetTypeFromParser(type, &TInfo);
  return BuildVAArgExpr(BuiltinLoc, expr, TInfo, RPLoc);
}

ExprResult Sema::BuildVAArgExpr(SourceLocation BuiltinLoc,
                                            Expr *E, TypeSourceInfo *TInfo,
                                            SourceLocation RPLoc) {
  Expr *OrigExpr = E;

  InitBuiltinVaListType();

  // Get the va_list type
  QualType VaListType = Context.getBuiltinVaListType();
  if (VaListType->isArrayType()) {
    // Deal with implicit array decay; for example, on x86-64,
    // va_list is an array, but it's supposed to decay to
    // a pointer for va_arg.
    VaListType = Context.getArrayDecayedType(VaListType);
    // Make sure the input expression also decays appropriately.
    UsualUnaryConversions(E);
  } else {
    // Otherwise, the va_list argument must be an l-value because
    // it is modified by va_arg.
    if (!E->isTypeDependent() &&
        CheckForModifiableLvalue(E, BuiltinLoc, *this))
      return ExprError();
  }

  if (!E->isTypeDependent() &&
      !Context.hasSameType(VaListType, E->getType())) {
    return ExprError(Diag(E->getLocStart(),
                         diag::err_first_argument_to_va_arg_not_of_type_va_list)
      << OrigExpr->getType() << E->getSourceRange());
  }

  // FIXME: Check that type is complete/non-abstract
  // FIXME: Warn if a non-POD type is passed in.

  QualType T = TInfo->getType().getNonLValueExprType(Context);
  return Owned(new (Context) VAArgExpr(BuiltinLoc, E, TInfo, RPLoc, T));
}

ExprResult Sema::ActOnGNUNullExpr(SourceLocation TokenLoc) {
  // The type of __null will be int or long, depending on the size of
  // pointers on the target.
  QualType Ty;
  if (Context.Target.getPointerWidth(0) == Context.Target.getIntWidth())
    Ty = Context.IntTy;
  else
    Ty = Context.LongTy;

  return Owned(new (Context) GNUNullExpr(Ty, TokenLoc));
}

static void MakeObjCStringLiteralFixItHint(Sema& SemaRef, QualType DstType,
                                           Expr *SrcExpr, FixItHint &Hint) {
  if (!SemaRef.getLangOptions().ObjC1)
    return;

  const ObjCObjectPointerType *PT = DstType->getAs<ObjCObjectPointerType>();
  if (!PT)
    return;

  // Check if the destination is of type 'id'.
  if (!PT->isObjCIdType()) {
    // Check if the destination is the 'NSString' interface.
    const ObjCInterfaceDecl *ID = PT->getInterfaceDecl();
    if (!ID || !ID->getIdentifier()->isStr("NSString"))
      return;
  }

  // Strip off any parens and casts.
  StringLiteral *SL = dyn_cast<StringLiteral>(SrcExpr->IgnoreParenCasts());
  if (!SL || SL->isWide())
    return;

  Hint = FixItHint::CreateInsertion(SL->getLocStart(), "@");
}

bool Sema::DiagnoseAssignmentResult(AssignConvertType ConvTy,
                                    SourceLocation Loc,
                                    QualType DstType, QualType SrcType,
                                    Expr *SrcExpr, AssignmentAction Action,
                                    bool *Complained) {
  if (Complained)
    *Complained = false;

  // Decode the result (notice that AST's are still created for extensions).
  bool isInvalid = false;
  unsigned DiagKind;
  FixItHint Hint;

  switch (ConvTy) {
  default: assert(0 && "Unknown conversion type");
  case Compatible: return false;
  case PointerToInt:
    DiagKind = diag::ext_typecheck_convert_pointer_int;
    break;
  case IntToPointer:
    DiagKind = diag::ext_typecheck_convert_int_pointer;
    break;
  case IncompatiblePointer:
    MakeObjCStringLiteralFixItHint(*this, DstType, SrcExpr, Hint);
    DiagKind = diag::ext_typecheck_convert_incompatible_pointer;
    break;
  case IncompatiblePointerSign:
    DiagKind = diag::ext_typecheck_convert_incompatible_pointer_sign;
    break;
  case FunctionVoidPointer:
    DiagKind = diag::ext_typecheck_convert_pointer_void_func;
    break;
  case CompatiblePointerDiscardsQualifiers:
    // If the qualifiers lost were because we were applying the
    // (deprecated) C++ conversion from a string literal to a char*
    // (or wchar_t*), then there was no error (C++ 4.2p2).  FIXME:
    // Ideally, this check would be performed in
    // CheckPointerTypesForAssignment. However, that would require a
    // bit of refactoring (so that the second argument is an
    // expression, rather than a type), which should be done as part
    // of a larger effort to fix CheckPointerTypesForAssignment for
    // C++ semantics.
    if (getLangOptions().CPlusPlus &&
        IsStringLiteralToNonConstPointerConversion(SrcExpr, DstType))
      return false;
    DiagKind = diag::ext_typecheck_convert_discards_qualifiers;
    break;
  case IncompatibleNestedPointerQualifiers:
    DiagKind = diag::ext_nested_pointer_qualifier_mismatch;
    break;
  case IntToBlockPointer:
    DiagKind = diag::err_int_to_block_pointer;
    break;
  case IncompatibleBlockPointer:
    DiagKind = diag::err_typecheck_convert_incompatible_block_pointer;
    break;
  case IncompatibleObjCQualifiedId:
    // FIXME: Diagnose the problem in ObjCQualifiedIdTypesAreCompatible, since
    // it can give a more specific diagnostic.
    DiagKind = diag::warn_incompatible_qualified_id;
    break;
  case IncompatibleVectors:
    DiagKind = diag::warn_incompatible_vectors;
    break;
  case Incompatible:
    DiagKind = diag::err_typecheck_convert_incompatible;
    isInvalid = true;
    break;
  }

  QualType FirstType, SecondType;
  switch (Action) {
  case AA_Assigning:
  case AA_Initializing:
    // The destination type comes first.
    FirstType = DstType;
    SecondType = SrcType;
    break;

  case AA_Returning:
  case AA_Passing:
  case AA_Converting:
  case AA_Sending:
  case AA_Casting:
    // The source type comes first.
    FirstType = SrcType;
    SecondType = DstType;
    break;
  }

  Diag(Loc, DiagKind) << FirstType << SecondType << Action
    << SrcExpr->getSourceRange() << Hint;
  if (Complained)
    *Complained = true;
  return isInvalid;
}

bool Sema::VerifyIntegerConstantExpression(const Expr *E, llvm::APSInt *Result){
  llvm::APSInt ICEResult;
  if (E->isIntegerConstantExpr(ICEResult, Context)) {
    if (Result)
      *Result = ICEResult;
    return false;
  }

  Expr::EvalResult EvalResult;

  if (!E->Evaluate(EvalResult, Context) || !EvalResult.Val.isInt() ||
      EvalResult.HasSideEffects) {
    Diag(E->getExprLoc(), diag::err_expr_not_ice) << E->getSourceRange();

    if (EvalResult.Diag) {
      // We only show the note if it's not the usual "invalid subexpression"
      // or if it's actually in a subexpression.
      if (EvalResult.Diag != diag::note_invalid_subexpr_in_ice ||
          E->IgnoreParens() != EvalResult.DiagExpr->IgnoreParens())
        Diag(EvalResult.DiagLoc, EvalResult.Diag);
    }

    return true;
  }

  Diag(E->getExprLoc(), diag::ext_expr_not_ice) <<
    E->getSourceRange();

  if (EvalResult.Diag &&
      Diags.getDiagnosticLevel(diag::ext_expr_not_ice) != Diagnostic::Ignored)
    Diag(EvalResult.DiagLoc, EvalResult.Diag);

  if (Result)
    *Result = EvalResult.Val.getInt();
  return false;
}

void
Sema::PushExpressionEvaluationContext(ExpressionEvaluationContext NewContext) {
  ExprEvalContexts.push_back(
        ExpressionEvaluationContextRecord(NewContext, ExprTemporaries.size()));
}

void
Sema::PopExpressionEvaluationContext() {
  // Pop the current expression evaluation context off the stack.
  ExpressionEvaluationContextRecord Rec = ExprEvalContexts.back();
  ExprEvalContexts.pop_back();

  if (Rec.Context == PotentiallyPotentiallyEvaluated) {
    if (Rec.PotentiallyReferenced) {
      // Mark any remaining declarations in the current position of the stack
      // as "referenced". If they were not meant to be referenced, semantic
      // analysis would have eliminated them (e.g., in ActOnCXXTypeId).
      for (PotentiallyReferencedDecls::iterator
             I = Rec.PotentiallyReferenced->begin(),
             IEnd = Rec.PotentiallyReferenced->end();
           I != IEnd; ++I)
        MarkDeclarationReferenced(I->first, I->second);
    }

    if (Rec.PotentiallyDiagnosed) {
      // Emit any pending diagnostics.
      for (PotentiallyEmittedDiagnostics::iterator
                I = Rec.PotentiallyDiagnosed->begin(),
             IEnd = Rec.PotentiallyDiagnosed->end();
           I != IEnd; ++I)
        Diag(I->first, I->second);
    }
  }

  // When are coming out of an unevaluated context, clear out any
  // temporaries that we may have created as part of the evaluation of
  // the expression in that context: they aren't relevant because they
  // will never be constructed.
  if (Rec.Context == Unevaluated &&
      ExprTemporaries.size() > Rec.NumTemporaries)
    ExprTemporaries.erase(ExprTemporaries.begin() + Rec.NumTemporaries,
                          ExprTemporaries.end());

  // Destroy the popped expression evaluation record.
  Rec.Destroy();
}

/// \brief Note that the given declaration was referenced in the source code.
///
/// This routine should be invoke whenever a given declaration is referenced
/// in the source code, and where that reference occurred. If this declaration
/// reference means that the the declaration is used (C++ [basic.def.odr]p2,
/// C99 6.9p3), then the declaration will be marked as used.
///
/// \param Loc the location where the declaration was referenced.
///
/// \param D the declaration that has been referenced by the source code.
void Sema::MarkDeclarationReferenced(SourceLocation Loc, Decl *D) {
  assert(D && "No declaration?");

  if (D->isUsed(false))
    return;

  // Mark a parameter or variable declaration "used", regardless of whether we're in a
  // template or not. The reason for this is that unevaluated expressions
  // (e.g. (void)sizeof()) constitute a use for warning purposes (-Wunused-variables and
  // -Wunused-parameters)
  if (isa<ParmVarDecl>(D) ||
      (isa<VarDecl>(D) && D->getDeclContext()->isFunctionOrMethod())) {
    D->setUsed(true);
    return;
  }

  if (!isa<VarDecl>(D) && !isa<FunctionDecl>(D))
    return;

  // Do not mark anything as "used" within a dependent context; wait for
  // an instantiation.
  if (CurContext->isDependentContext())
    return;

  switch (ExprEvalContexts.back().Context) {
    case Unevaluated:
      // We are in an expression that is not potentially evaluated; do nothing.
      return;

    case PotentiallyEvaluated:
      // We are in a potentially-evaluated expression, so this declaration is
      // "used"; handle this below.
      break;

    case PotentiallyPotentiallyEvaluated:
      // We are in an expression that may be potentially evaluated; queue this
      // declaration reference until we know whether the expression is
      // potentially evaluated.
      ExprEvalContexts.back().addReferencedDecl(Loc, D);
      return;
  }

  // Note that this declaration has been used.
  if (CXXConstructorDecl *Constructor = dyn_cast<CXXConstructorDecl>(D)) {
    unsigned TypeQuals;
    if (Constructor->isImplicit() && Constructor->isDefaultConstructor()) {
      if (Constructor->getParent()->hasTrivialConstructor())
        return;
      if (!Constructor->isUsed(false))
        DefineImplicitDefaultConstructor(Loc, Constructor);
    } else if (Constructor->isImplicit() &&
               Constructor->isCopyConstructor(TypeQuals)) {
      if (!Constructor->isUsed(false))
        DefineImplicitCopyConstructor(Loc, Constructor, TypeQuals);
    }

    MarkVTableUsed(Loc, Constructor->getParent());
  } else if (CXXDestructorDecl *Destructor = dyn_cast<CXXDestructorDecl>(D)) {
    if (Destructor->isImplicit() && !Destructor->isUsed(false))
      DefineImplicitDestructor(Loc, Destructor);
    if (Destructor->isVirtual())
      MarkVTableUsed(Loc, Destructor->getParent());
  } else if (CXXMethodDecl *MethodDecl = dyn_cast<CXXMethodDecl>(D)) {
    if (MethodDecl->isImplicit() && MethodDecl->isOverloadedOperator() &&
        MethodDecl->getOverloadedOperator() == OO_Equal) {
      if (!MethodDecl->isUsed(false))
        DefineImplicitCopyAssignment(Loc, MethodDecl);
    } else if (MethodDecl->isVirtual())
      MarkVTableUsed(Loc, MethodDecl->getParent());
  }
  if (FunctionDecl *Function = dyn_cast<FunctionDecl>(D)) {
    // Implicit instantiation of function templates and member functions of
    // class templates.
    if (Function->isImplicitlyInstantiable()) {
      bool AlreadyInstantiated = false;
      if (FunctionTemplateSpecializationInfo *SpecInfo
                                = Function->getTemplateSpecializationInfo()) {
        if (SpecInfo->getPointOfInstantiation().isInvalid())
          SpecInfo->setPointOfInstantiation(Loc);
        else if (SpecInfo->getTemplateSpecializationKind()
                   == TSK_ImplicitInstantiation)
          AlreadyInstantiated = true;
      } else if (MemberSpecializationInfo *MSInfo
                                  = Function->getMemberSpecializationInfo()) {
        if (MSInfo->getPointOfInstantiation().isInvalid())
          MSInfo->setPointOfInstantiation(Loc);
        else if (MSInfo->getTemplateSpecializationKind()
                   == TSK_ImplicitInstantiation)
          AlreadyInstantiated = true;
      }

      if (!AlreadyInstantiated) {
        if (isa<CXXRecordDecl>(Function->getDeclContext()) &&
            cast<CXXRecordDecl>(Function->getDeclContext())->isLocalClass())
          PendingLocalImplicitInstantiations.push_back(std::make_pair(Function,
                                                                      Loc));
        else
          PendingInstantiations.push_back(std::make_pair(Function, Loc));
      }
    } else // Walk redefinitions, as some of them may be instantiable.
      for (FunctionDecl::redecl_iterator i(Function->redecls_begin()),
           e(Function->redecls_end()); i != e; ++i) {
        if (!i->isUsed(false) && i->isImplicitlyInstantiable())
          MarkDeclarationReferenced(Loc, *i);
      }

    // FIXME: keep track of references to static functions

    // Recursive functions should be marked when used from another function.
    if (CurContext != Function)
      Function->setUsed(true);

    return;
  }

  if (VarDecl *Var = dyn_cast<VarDecl>(D)) {
    // Implicit instantiation of static data members of class templates.
    if (Var->isStaticDataMember() &&
        Var->getInstantiatedFromStaticDataMember()) {
      MemberSpecializationInfo *MSInfo = Var->getMemberSpecializationInfo();
      assert(MSInfo && "Missing member specialization information?");
      if (MSInfo->getPointOfInstantiation().isInvalid() &&
          MSInfo->getTemplateSpecializationKind()== TSK_ImplicitInstantiation) {
        MSInfo->setPointOfInstantiation(Loc);
        PendingInstantiations.push_back(std::make_pair(Var, Loc));
      }
    }

    // FIXME: keep track of references to static data?

    D->setUsed(true);
    return;
  }
}

namespace {
  // Mark all of the declarations referenced
  // FIXME: Not fully implemented yet! We need to have a better understanding
  // of when we're entering
  class MarkReferencedDecls : public RecursiveASTVisitor<MarkReferencedDecls> {
    Sema &S;
    SourceLocation Loc;

  public:
    typedef RecursiveASTVisitor<MarkReferencedDecls> Inherited;

    MarkReferencedDecls(Sema &S, SourceLocation Loc) : S(S), Loc(Loc) { }

    bool TraverseTemplateArgument(const TemplateArgument &Arg);
    bool TraverseRecordType(RecordType *T);
  };
}

bool MarkReferencedDecls::TraverseTemplateArgument(
  const TemplateArgument &Arg) {
  if (Arg.getKind() == TemplateArgument::Declaration) {
    S.MarkDeclarationReferenced(Loc, Arg.getAsDecl());
  }

  return Inherited::TraverseTemplateArgument(Arg);
}

bool MarkReferencedDecls::TraverseRecordType(RecordType *T) {
  if (ClassTemplateSpecializationDecl *Spec
                  = dyn_cast<ClassTemplateSpecializationDecl>(T->getDecl())) {
    const TemplateArgumentList &Args = Spec->getTemplateArgs();
    return TraverseTemplateArguments(Args.getFlatArgumentList(),
                                     Args.flat_size());
  }

  return true;
}

void Sema::MarkDeclarationsReferencedInType(SourceLocation Loc, QualType T) {
  MarkReferencedDecls Marker(*this, Loc);
  Marker.TraverseType(Context.getCanonicalType(T));
}

/// \brief Emit a diagnostic that describes an effect on the run-time behavior
/// of the program being compiled.
///
/// This routine emits the given diagnostic when the code currently being
/// type-checked is "potentially evaluated", meaning that there is a
/// possibility that the code will actually be executable. Code in sizeof()
/// expressions, code used only during overload resolution, etc., are not
/// potentially evaluated. This routine will suppress such diagnostics or,
/// in the absolutely nutty case of potentially potentially evaluated
/// expressions (C++ typeid), queue the diagnostic to potentially emit it
/// later.
///
/// This routine should be used for all diagnostics that describe the run-time
/// behavior of a program, such as passing a non-POD value through an ellipsis.
/// Failure to do so will likely result in spurious diagnostics or failures
/// during overload resolution or within sizeof/alignof/typeof/typeid.
bool Sema::DiagRuntimeBehavior(SourceLocation Loc,
                               const PartialDiagnostic &PD) {
  switch (ExprEvalContexts.back().Context ) {
  case Unevaluated:
    // The argument will never be evaluated, so don't complain.
    break;

  case PotentiallyEvaluated:
    Diag(Loc, PD);
    return true;

  case PotentiallyPotentiallyEvaluated:
    ExprEvalContexts.back().addDiagnostic(Loc, PD);
    break;
  }

  return false;
}

bool Sema::CheckCallReturnType(QualType ReturnType, SourceLocation Loc,
                               CallExpr *CE, FunctionDecl *FD) {
  if (ReturnType->isVoidType() || !ReturnType->isIncompleteType())
    return false;

  PartialDiagnostic Note =
    FD ? PDiag(diag::note_function_with_incomplete_return_type_declared_here)
    << FD->getDeclName() : PDiag();
  SourceLocation NoteLoc = FD ? FD->getLocation() : SourceLocation();

  if (RequireCompleteType(Loc, ReturnType,
                          FD ?
                          PDiag(diag::err_call_function_incomplete_return)
                            << CE->getSourceRange() << FD->getDeclName() :
                          PDiag(diag::err_call_incomplete_return)
                            << CE->getSourceRange(),
                          std::make_pair(NoteLoc, Note)))
    return true;

  return false;
}

// Diagnose the common s/=/==/ typo.  Note that adding parentheses
// will prevent this condition from triggering, which is what we want.
void Sema::DiagnoseAssignmentAsCondition(Expr *E) {
  SourceLocation Loc;

  unsigned diagnostic = diag::warn_condition_is_assignment;

  if (isa<BinaryOperator>(E)) {
    BinaryOperator *Op = cast<BinaryOperator>(E);
    if (Op->getOpcode() != BO_Assign)
      return;

    // Greylist some idioms by putting them into a warning subcategory.
    if (ObjCMessageExpr *ME
          = dyn_cast<ObjCMessageExpr>(Op->getRHS()->IgnoreParenCasts())) {
      Selector Sel = ME->getSelector();

      // self = [<foo> init...]
      if (isSelfExpr(Op->getLHS())
          && Sel.getIdentifierInfoForSlot(0)->getName().startswith("init"))
        diagnostic = diag::warn_condition_is_idiomatic_assignment;

      // <foo> = [<bar> nextObject]
      else if (Sel.isUnarySelector() &&
               Sel.getIdentifierInfoForSlot(0)->getName() == "nextObject")
        diagnostic = diag::warn_condition_is_idiomatic_assignment;
    }

    Loc = Op->getOperatorLoc();
  } else if (isa<CXXOperatorCallExpr>(E)) {
    CXXOperatorCallExpr *Op = cast<CXXOperatorCallExpr>(E);
    if (Op->getOperator() != OO_Equal)
      return;

    Loc = Op->getOperatorLoc();
  } else {
    // Not an assignment.
    return;
  }

  SourceLocation Open = E->getSourceRange().getBegin();
  SourceLocation Close = PP.getLocForEndOfToken(E->getSourceRange().getEnd());

  Diag(Loc, diagnostic) << E->getSourceRange();
  Diag(Loc, diag::note_condition_assign_to_comparison)
    << FixItHint::CreateReplacement(Loc, "==");
  Diag(Loc, diag::note_condition_assign_silence)
    << FixItHint::CreateInsertion(Open, "(")
    << FixItHint::CreateInsertion(Close, ")");
}

bool Sema::CheckBooleanCondition(Expr *&E, SourceLocation Loc) {
  DiagnoseAssignmentAsCondition(E);

  if (!E->isTypeDependent()) {
    DefaultFunctionArrayLvalueConversion(E);

    QualType T = E->getType();

    if (getLangOptions().CPlusPlus) {
      if (CheckCXXBooleanCondition(E)) // C++ 6.4p4
        return true;
    } else if (!T->isScalarType()) { // C99 6.8.4.1p1
      Diag(Loc, diag::err_typecheck_statement_requires_scalar)
        << T << E->getSourceRange();
      return true;
    }
  }

  return false;
}

ExprResult Sema::ActOnBooleanCondition(Scope *S, SourceLocation Loc,
                                       Expr *Sub) {
  if (!Sub)
    return ExprError();
  
  if (CheckBooleanCondition(Sub, Loc))
    return ExprError();
  
  return Owned(Sub);
}
