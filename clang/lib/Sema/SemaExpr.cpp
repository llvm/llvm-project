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

#include "Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/LiteralSupport.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Parse/Designator.h"
#include "clang/Parse/Scope.h"
using namespace clang;


/// \brief Determine whether the use of this declaration is valid, and
/// emit any corresponding diagnostics.
///
/// This routine diagnoses various problems with referencing
/// declarations that can occur when using a declaration. For example,
/// it might warn if a deprecated or unavailable declaration is being
/// used, or produce an error (and return true) if a C++0x deleted
/// function is being used.
///
/// \returns true if there was an error (this declaration cannot be
/// referenced), false otherwise.
bool Sema::DiagnoseUseOfDecl(NamedDecl *D, SourceLocation Loc) {
  // See if the decl is deprecated.
  if (D->getAttr<DeprecatedAttr>()) {
    // Implementing deprecated stuff requires referencing deprecated
    // stuff. Don't warn if we are implementing a deprecated
    // construct.
    bool isSilenced = false;
    
    if (NamedDecl *ND = getCurFunctionOrMethodDecl()) {
      // If this reference happens *in* a deprecated function or method, don't
      // warn.
      isSilenced = ND->getAttr<DeprecatedAttr>();
      
      // If this is an Objective-C method implementation, check to see if the
      // method was deprecated on the declaration, not the definition.
      if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(ND)) {
        // The semantic decl context of a ObjCMethodDecl is the
        // ObjCImplementationDecl.
        if (ObjCImplementationDecl *Impl
              = dyn_cast<ObjCImplementationDecl>(MD->getParent())) {
          
          MD = Impl->getClassInterface()->getMethod(MD->getSelector(),
                                                    MD->isInstanceMethod());
          isSilenced |= MD && MD->getAttr<DeprecatedAttr>();
        }
      }
    }
    
    if (!isSilenced)
      Diag(Loc, diag::warn_deprecated) << D->getDeclName();
  }

  // See if this is a deleted function.
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    if (FD->isDeleted()) {
      Diag(Loc, diag::err_deleted_function_use);
      Diag(D->getLocation(), diag::note_unavailable_here) << true;
      return true;
    }
  }

  // See if the decl is unavailable
  if (D->getAttr<UnavailableAttr>()) {
    Diag(Loc, diag::warn_unavailable) << D->getDeclName();
    Diag(D->getLocation(), diag::note_unavailable_here) << 0;
  }

  return false;
}

/// DiagnoseSentinelCalls - This routine checks on method dispatch calls
/// (and other functions in future), which have been declared with sentinel 
/// attribute. It warns if call does not have the sentinel argument.
///
void Sema::DiagnoseSentinelCalls(NamedDecl *D, SourceLocation Loc,
                                 Expr **Args, unsigned NumArgs)
{
  const SentinelAttr *attr = D->getAttr<SentinelAttr>();
  if (!attr) 
    return;
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
      ? Ty->getAs<PointerType>()->getPointeeType()->getAsFunctionType()
      : Ty->getAs<BlockPointerType>()->getPointeeType()->getAsFunctionType();
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
  if (sentinelExpr && (!sentinelExpr->getType()->isPointerType() ||
                       !sentinelExpr->isNullPointerConstant(Context))) {
    Diag(Loc, diag::warn_missing_sentinel) << isMethod;
    Diag(D->getLocation(), diag::note_sentinel_here) << isMethod;
  }
  return;
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
    ImpCastExprToType(E, Context.getPointerType(Ty));
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
                        CastExpr::CK_ArrayToPointerDecay);
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
    ImpCastExprToType(Expr, PTy);
    return Expr;
  }
  if (Ty->isPromotableIntegerType()) {
    QualType PT = Context.getPromotedIntegerType(Ty);
    ImpCastExprToType(Expr, PT);
    return Expr;
  }

  DefaultFunctionArrayConversion(Expr);
  return Expr;
}

/// DefaultArgumentPromotion (C99 6.5.2.2p6). Used for function calls that
/// do not have a prototype. Arguments that have type float are promoted to 
/// double. All other argument types are converted by UsualUnaryConversions().
void Sema::DefaultArgumentPromotion(Expr *&Expr) {
  QualType Ty = Expr->getType();
  assert(!Ty.isNull() && "DefaultArgumentPromotion - missing type");
  
  // If this is a 'float' (CVR qualified or typedef) promote to double.
  if (const BuiltinType *BT = Ty->getAsBuiltinType())
    if (BT->getKind() == BuiltinType::Float)
      return ImpCastExprToType(Expr, Context.DoubleTy);
  
  UsualUnaryConversions(Expr);
}

/// DefaultVariadicArgumentPromotion - Like DefaultArgumentPromotion, but
/// will warn if the resulting type is not a POD type, and rejects ObjC
/// interfaces passed by value.  This returns true if the argument type is
/// completely illegal.
bool Sema::DefaultVariadicArgumentPromotion(Expr *&Expr, VariadicCallType CT) {
  DefaultArgumentPromotion(Expr);
  
  if (Expr->getType()->isObjCInterfaceType()) {
    Diag(Expr->getLocStart(),
         diag::err_cannot_pass_objc_interface_to_vararg)
      << Expr->getType() << CT;
    return true;
  }
  
  if (!Expr->getType()->isPODType())
    Diag(Expr->getLocStart(), diag::warn_cannot_pass_non_pod_arg_to_vararg)
      << Expr->getType() << CT;

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
    ImpCastExprToType(lhsExpr, destType);
  ImpCastExprToType(rhsExpr, destType);
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
Action::OwningExprResult
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
  if (getLangOptions().CPlusPlus)
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
/// This also keeps the 'hasBlockDeclRefExprs' in the BlockSemaInfo records
/// up-to-date.
///
static bool ShouldSnapshotBlockValueReference(BlockSemaInfo *CurBlock,
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
  for (BlockSemaInfo *NextBlock = CurBlock->PrevBlockInfo; NextBlock;
       NextBlock = NextBlock->PrevBlockInfo) {
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
    


/// ActOnIdentifierExpr - The parser read an identifier in expression context,
/// validate it per-C99 6.5.1.  HasTrailingLParen indicates whether this
/// identifier is used in a function call context.
/// SS is only used for a C++ qualified-id (foo::bar) to indicate the
/// class or namespace that the identifier must be a member of.
Sema::OwningExprResult Sema::ActOnIdentifierExpr(Scope *S, SourceLocation Loc,
                                                 IdentifierInfo &II,
                                                 bool HasTrailingLParen,
                                                 const CXXScopeSpec *SS,
                                                 bool isAddressOfOperand) {
  return ActOnDeclarationNameExpr(S, Loc, &II, HasTrailingLParen, SS,
                                  isAddressOfOperand);
}

/// BuildDeclRefExpr - Build either a DeclRefExpr or a
/// QualifiedDeclRefExpr based on whether or not SS is a
/// nested-name-specifier.
Sema::OwningExprResult
Sema::BuildDeclRefExpr(NamedDecl *D, QualType Ty, SourceLocation Loc,
                       bool TypeDependent, bool ValueDependent,
                       const CXXScopeSpec *SS) {
  if (Context.getCanonicalType(Ty) == Context.UndeducedAutoTy) {
    Diag(Loc,
         diag::err_auto_variable_cannot_appear_in_own_initializer) 
      << D->getDeclName();
    return ExprError();
  }
  
  if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
    if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(CurContext)) {
      if (const FunctionDecl *FD = MD->getParent()->isLocalClass()) {
        if (VD->hasLocalStorage() && VD->getDeclContext() != CurContext) {
          Diag(Loc, diag::err_reference_to_local_var_in_enclosing_function) 
            << D->getIdentifier() << FD->getDeclName();
          Diag(D->getLocation(), diag::note_local_variable_declared_here) 
            << D->getIdentifier();
          return ExprError();
        }
      }
    }
  }
  
  MarkDeclarationReferenced(Loc, D);
  
  Expr *E;
  if (SS && !SS->isEmpty()) {
    E = new (Context) QualifiedDeclRefExpr(D, Ty, Loc, TypeDependent, 
                                          ValueDependent, SS->getRange(),
                  static_cast<NestedNameSpecifier *>(SS->getScopeRep()));
  } else
    E = new (Context) DeclRefExpr(D, Ty, Loc, TypeDependent, ValueDependent);
  
  return Owned(E);
}

/// getObjectForAnonymousRecordDecl - Retrieve the (unnamed) field or
/// variable corresponding to the anonymous union or struct whose type
/// is Record.
static Decl *getObjectForAnonymousRecordDecl(ASTContext &Context,
                                             RecordDecl *Record) {
  assert(Record->isAnonymousStructOrUnion() && 
         "Record must be an anonymous struct or union!");
  
  // FIXME: Once Decls are directly linked together, this will be an O(1)
  // operation rather than a slow walk through DeclContext's vector (which
  // itself will be eliminated). DeclGroups might make this even better.
  DeclContext *Ctx = Record->getDeclContext();
  for (DeclContext::decl_iterator D = Ctx->decls_begin(), 
                               DEnd = Ctx->decls_end();
       D != DEnd; ++D) {
    if (*D == Record) {
      // The object for the anonymous struct/union directly
      // follows its type in the list of declarations.
      ++D;
      assert(D != DEnd && "Missing object for anonymous record");
      assert(!cast<NamedDecl>(*D)->getDeclName() && "Decl should be unnamed");
      return *D;
    }
  }

  assert(false && "Missing object for anonymous record");
  return 0;
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
    Decl *AnonObject = getObjectForAnonymousRecordDecl(Context, Record);
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

Sema::OwningExprResult
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
  unsigned ExtraQuals = 0;
  if (BaseObject) {
    // BaseObject is an anonymous struct/union variable (and is,
    // therefore, not part of another non-anonymous record).
    if (BaseObjectExpr) BaseObjectExpr->Destroy(Context);
    MarkDeclarationReferenced(Loc, BaseObject);
    BaseObjectExpr = new (Context) DeclRefExpr(BaseObject,BaseObject->getType(),
                                               SourceLocation());
    ExtraQuals 
      = Context.getCanonicalType(BaseObject->getType()).getCVRQualifiers();
  } else if (BaseObjectExpr) {
    // The caller provided the base object expression. Determine
    // whether its a pointer and whether it adds any qualifiers to the
    // anonymous struct/union fields we're looking into.
    QualType ObjectType = BaseObjectExpr->getType();
    if (const PointerType *ObjectPtr = ObjectType->getAs<PointerType>()) {
      BaseObjectIsPointer = true;
      ObjectType = ObjectPtr->getPointeeType();
    }
    ExtraQuals = Context.getCanonicalType(ObjectType).getCVRQualifiers();
  } else {
    // We've found a member of an anonymous struct/union that is
    // inside a non-anonymous struct/union, so in a well-formed
    // program our base object expression is "this".
    if (CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(CurContext)) {
      if (!MD->isStatic()) {
        QualType AnonFieldType 
          = Context.getTagDeclType(
                     cast<RecordDecl>(AnonFields.back()->getDeclContext()));
        QualType ThisType = Context.getTagDeclType(MD->getParent());
        if ((Context.getCanonicalType(AnonFieldType) 
               == Context.getCanonicalType(ThisType)) ||
            IsDerivedFrom(ThisType, AnonFieldType)) {
          // Our base object expression is "this".
          BaseObjectExpr = new (Context) CXXThisExpr(SourceLocation(),
                                                     MD->getThisType(Context));
          BaseObjectIsPointer = true;
        }
      } else {
        return ExprError(Diag(Loc,diag::err_invalid_member_use_in_static_method)
          << Field->getDeclName());
      }
      ExtraQuals = MD->getTypeQualifiers();
    }

    if (!BaseObjectExpr) 
      return ExprError(Diag(Loc, diag::err_invalid_non_static_member_use)
        << Field->getDeclName());
  }

  // Build the implicit member references to the field of the
  // anonymous struct/union.
  Expr *Result = BaseObjectExpr;
  unsigned BaseAddrSpace = BaseObjectExpr->getType().getAddressSpace();
  for (llvm::SmallVector<FieldDecl *, 4>::reverse_iterator
         FI = AnonFields.rbegin(), FIEnd = AnonFields.rend();
       FI != FIEnd; ++FI) {
    QualType MemberType = (*FI)->getType();
    if (!(*FI)->isMutable()) {
      unsigned combinedQualifiers 
        = MemberType.getCVRQualifiers() | ExtraQuals;
      MemberType = MemberType.getQualifiedType(combinedQualifiers);
    }
    if (BaseAddrSpace != MemberType.getAddressSpace())
      MemberType = Context.getAddrSpaceQualType(MemberType, BaseAddrSpace);
    MarkDeclarationReferenced(Loc, *FI);
    Result = new (Context) MemberExpr(Result, BaseObjectIsPointer, *FI,
                                      OpLoc, MemberType);
    BaseObjectIsPointer = false;
    ExtraQuals = Context.getCanonicalType(MemberType).getCVRQualifiers();
  }

  return Owned(Result);
}

/// ActOnDeclarationNameExpr - The parser has read some kind of name
/// (e.g., a C++ id-expression (C++ [expr.prim]p1)). This routine
/// performs lookup on that name and returns an expression that refers
/// to that name. This routine isn't directly called from the parser,
/// because the parser doesn't know about DeclarationName. Rather,
/// this routine is called by ActOnIdentifierExpr,
/// ActOnOperatorFunctionIdExpr, and ActOnConversionFunctionExpr,
/// which form the DeclarationName from the corresponding syntactic
/// forms.
///
/// HasTrailingLParen indicates whether this identifier is used in a
/// function call context.  LookupCtx is only used for a C++
/// qualified-id (foo::bar) to indicate the class or namespace that
/// the identifier must be a member of.
///
/// isAddressOfOperand means that this expression is the direct operand
/// of an address-of operator. This matters because this is the only
/// situation where a qualified name referencing a non-static member may
/// appear outside a member function of this class.
Sema::OwningExprResult
Sema::ActOnDeclarationNameExpr(Scope *S, SourceLocation Loc,
                               DeclarationName Name, bool HasTrailingLParen,
                               const CXXScopeSpec *SS, 
                               bool isAddressOfOperand) {
  // Could be enum-constant, value decl, instance variable, etc.
  if (SS && SS->isInvalid())
    return ExprError();

  // C++ [temp.dep.expr]p3:
  //   An id-expression is type-dependent if it contains:
  //     -- a nested-name-specifier that contains a class-name that
  //        names a dependent type.
  // FIXME: Member of the current instantiation.
  if (SS && isDependentScopeSpecifier(*SS)) {
    return Owned(new (Context) UnresolvedDeclRefExpr(Name, Context.DependentTy,
                                                     Loc, SS->getRange(), 
                static_cast<NestedNameSpecifier *>(SS->getScopeRep()),
                                                     isAddressOfOperand));
  }

  LookupResult Lookup = LookupParsedName(S, SS, Name, LookupOrdinaryName,
                                         false, true, Loc);

  if (Lookup.isAmbiguous()) {
    DiagnoseAmbiguousLookup(Lookup, Name, Loc,
                            SS && SS->isSet() ? SS->getRange()
                                              : SourceRange());
    return ExprError();
  }
  
  NamedDecl *D = Lookup.getAsDecl();

  // If this reference is in an Objective-C method, then ivar lookup happens as
  // well.
  IdentifierInfo *II = Name.getAsIdentifierInfo();
  if (II && getCurMethodDecl()) {
    // There are two cases to handle here.  1) scoped lookup could have failed,
    // in which case we should look for an ivar.  2) scoped lookup could have
    // found a decl, but that decl is outside the current instance method (i.e. 
    // a global variable).  In these two cases, we do a lookup for an ivar with 
    // this name, if the lookup sucedes, we replace it our current decl.
    if (D == 0 || D->isDefinedOutsideFunctionOrMethod()) {
      ObjCInterfaceDecl *IFace = getCurMethodDecl()->getClassInterface();
      ObjCInterfaceDecl *ClassDeclared;
      if (ObjCIvarDecl *IV = IFace->lookupInstanceVariable(II, ClassDeclared)) {
        // Check if referencing a field with __attribute__((deprecated)).
        if (DiagnoseUseOfDecl(IV, Loc))
          return ExprError();
        
        // If we're referencing an invalid decl, just return this as a silent
        // error node.  The error diagnostic was already emitted on the decl.
        if (IV->isInvalidDecl())
          return ExprError();
        
        bool IsClsMethod = getCurMethodDecl()->isClassMethod();
        // If a class method attemps to use a free standing ivar, this is
        // an error.
        if (IsClsMethod && D && !D->isDefinedOutsideFunctionOrMethod())
           return ExprError(Diag(Loc, diag::error_ivar_use_in_class_method)
                           << IV->getDeclName());
        // If a class method uses a global variable, even if an ivar with
        // same name exists, use the global.
        if (!IsClsMethod) {
          if (IV->getAccessControl() == ObjCIvarDecl::Private &&
              ClassDeclared != IFace)
           Diag(Loc, diag::error_private_ivar_access) << IV->getDeclName();
          // FIXME: This should use a new expr for a direct reference, don't
          // turn this into Self->ivar, just return a BareIVarExpr or something.
          IdentifierInfo &II = Context.Idents.get("self");
          OwningExprResult SelfExpr = ActOnIdentifierExpr(S, SourceLocation(),
                                                          II, false);
          MarkDeclarationReferenced(Loc, IV);
          return Owned(new (Context) 
                       ObjCIvarRefExpr(IV, IV->getType(), Loc, 
                                       SelfExpr.takeAs<Expr>(), true, true));
        }
      }
    } else if (getCurMethodDecl()->isInstanceMethod()) {
      // We should warn if a local variable hides an ivar.
      ObjCInterfaceDecl *IFace = getCurMethodDecl()->getClassInterface();
      ObjCInterfaceDecl *ClassDeclared;
      if (ObjCIvarDecl *IV = IFace->lookupInstanceVariable(II, ClassDeclared)) {
        if (IV->getAccessControl() != ObjCIvarDecl::Private ||
            IFace == ClassDeclared)
          Diag(Loc, diag::warn_ivar_use_hidden) << IV->getDeclName();
      }
    }
    // Needed to implement property "super.method" notation.
    if (D == 0 && II->isStr("super")) {
      QualType T;
      
      if (getCurMethodDecl()->isInstanceMethod())
        T = Context.getObjCObjectPointerType(Context.getObjCInterfaceType(
                                      getCurMethodDecl()->getClassInterface()));
      else
        T = Context.getObjCClassType();
      return Owned(new (Context) ObjCSuperExpr(Loc, T));
    }
  }

  // Determine whether this name might be a candidate for
  // argument-dependent lookup.
  bool ADL = getLangOptions().CPlusPlus && (!SS || !SS->isSet()) && 
             HasTrailingLParen;

  if (ADL && D == 0) {
    // We've seen something of the form
    //
    //   identifier(
    //
    // and we did not find any entity by the name
    // "identifier". However, this identifier is still subject to
    // argument-dependent lookup, so keep track of the name.
    return Owned(new (Context) UnresolvedFunctionNameExpr(Name,
                                                          Context.OverloadTy,
                                                          Loc));
  }

  if (D == 0) {
    // Otherwise, this could be an implicitly declared function reference (legal
    // in C90, extension in C99).
    if (HasTrailingLParen && II &&
        !getLangOptions().CPlusPlus) // Not in C++.
      D = ImplicitlyDefineFunction(Loc, *II, S);
    else {
      // If this name wasn't predeclared and if this is not a function call,
      // diagnose the problem.
      if (SS && !SS->isEmpty())
        return ExprError(Diag(Loc, diag::err_typecheck_no_member)
          << Name << SS->getRange());
      else if (Name.getNameKind() == DeclarationName::CXXOperatorName ||
               Name.getNameKind() == DeclarationName::CXXConversionFunctionName)
        return ExprError(Diag(Loc, diag::err_undeclared_use)
          << Name.getAsString());
      else
        return ExprError(Diag(Loc, diag::err_undeclared_var_use) << Name);
    }
  }
  
  if (VarDecl *Var = dyn_cast<VarDecl>(D)) {
    // Warn about constructs like:
    //   if (void *X = foo()) { ... } else { X }.
    // In the else block, the pointer is always false.
    
    // FIXME: In a template instantiation, we don't have scope
    // information to check this property.
    if (Var->isDeclaredInCondition() && Var->getType()->isScalarType()) {
      Scope *CheckS = S;
      while (CheckS) {
        if (CheckS->isWithinElse() && 
            CheckS->getControlParent()->isDeclScope(DeclPtrTy::make(Var))) {
          if (Var->getType()->isBooleanType())
            ExprError(Diag(Loc, diag::warn_value_always_false)
                      << Var->getDeclName());
          else
            ExprError(Diag(Loc, diag::warn_value_always_zero)
                      << Var->getDeclName());
          break;
        }
        
        // Move up one more control parent to check again.
        CheckS = CheckS->getControlParent();
        if (CheckS)
          CheckS = CheckS->getParent();
      }
    }
  } else if (FunctionDecl *Func = dyn_cast<FunctionDecl>(D)) {
    if (!getLangOptions().CPlusPlus && !Func->hasPrototype()) {
      // C99 DR 316 says that, if a function type comes from a
      // function definition (without a prototype), that type is only
      // used for checking compatibility. Therefore, when referencing
      // the function, we pretend that we don't have the full function
      // type.
      if (DiagnoseUseOfDecl(Func, Loc))
        return ExprError();

      QualType T = Func->getType();
      QualType NoProtoType = T;
      if (const FunctionProtoType *Proto = T->getAsFunctionProtoType())
        NoProtoType = Context.getFunctionNoProtoType(Proto->getResultType());
      return BuildDeclRefExpr(Func, NoProtoType, Loc, false, false, SS);
    }
  }
  
  return BuildDeclarationNameExpr(Loc, D, HasTrailingLParen, SS, isAddressOfOperand);
}
/// \brief Cast member's object to its own class if necessary.
bool
Sema::PerformObjectMemberConversion(Expr *&From, NamedDecl *Member) {
  if (FieldDecl *FD = dyn_cast<FieldDecl>(Member))
    if (CXXRecordDecl *RD = 
        dyn_cast<CXXRecordDecl>(FD->getDeclContext())) {
      QualType DestType = 
        Context.getCanonicalType(Context.getTypeDeclType(RD));
      if (DestType->isDependentType() || From->getType()->isDependentType())
        return false;
      QualType FromRecordType = From->getType();
      QualType DestRecordType = DestType;
      if (FromRecordType->getAs<PointerType>()) {
        DestType = Context.getPointerType(DestType);
        FromRecordType = FromRecordType->getPointeeType();
      }
      if (!Context.hasSameUnqualifiedType(FromRecordType, DestRecordType) &&
          CheckDerivedToBaseConversion(FromRecordType,
                                       DestRecordType,
                                       From->getSourceRange().getBegin(),
                                       From->getSourceRange()))
        return true;
      ImpCastExprToType(From, DestType, CastExpr::CK_DerivedToBase,
                        /*isLvalue=*/true);
    }
  return false;
}

/// \brief Complete semantic analysis for a reference to the given declaration.
Sema::OwningExprResult
Sema::BuildDeclarationNameExpr(SourceLocation Loc, NamedDecl *D,
                               bool HasTrailingLParen,
                               const CXXScopeSpec *SS, 
                               bool isAddressOfOperand) {
  assert(D && "Cannot refer to a NULL declaration");
  DeclarationName Name = D->getDeclName();
  
  // If this is an expression of the form &Class::member, don't build an
  // implicit member ref, because we want a pointer to the member in general,
  // not any specific instance's member.
  if (isAddressOfOperand && SS && !SS->isEmpty() && !HasTrailingLParen) {
    DeclContext *DC = computeDeclContext(*SS);
    if (D && isa<CXXRecordDecl>(DC)) {
      QualType DType;
      if (FieldDecl *FD = dyn_cast<FieldDecl>(D)) {
        DType = FD->getType().getNonReferenceType();
      } else if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(D)) {
        DType = Method->getType();
      } else if (isa<OverloadedFunctionDecl>(D)) {
        DType = Context.OverloadTy;
      }
      // Could be an inner type. That's diagnosed below, so ignore it here.
      if (!DType.isNull()) {
        // The pointer is type- and value-dependent if it points into something
        // dependent.
        bool Dependent = DC->isDependentContext();
        return BuildDeclRefExpr(D, DType, Loc, Dependent, Dependent, SS);
      }
    }
  }

  // We may have found a field within an anonymous union or struct
  // (C++ [class.union]).
  if (FieldDecl *FD = dyn_cast<FieldDecl>(D))
    if (cast<RecordDecl>(FD->getDeclContext())->isAnonymousStructOrUnion())
      return BuildAnonymousStructUnionMemberReference(Loc, FD);

  if (CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(CurContext)) {
    if (!MD->isStatic()) {
      // C++ [class.mfct.nonstatic]p2: 
      //   [...] if name lookup (3.4.1) resolves the name in the
      //   id-expression to a nonstatic nontype member of class X or of
      //   a base class of X, the id-expression is transformed into a
      //   class member access expression (5.2.5) using (*this) (9.3.2)
      //   as the postfix-expression to the left of the '.' operator.
      DeclContext *Ctx = 0;
      QualType MemberType;
      if (FieldDecl *FD = dyn_cast<FieldDecl>(D)) {
        Ctx = FD->getDeclContext();
        MemberType = FD->getType();

        if (const ReferenceType *RefType = MemberType->getAs<ReferenceType>())
          MemberType = RefType->getPointeeType();
        else if (!FD->isMutable()) {
          unsigned combinedQualifiers 
            = MemberType.getCVRQualifiers() | MD->getTypeQualifiers();
          MemberType = MemberType.getQualifiedType(combinedQualifiers);
        }
      } else if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(D)) {
        if (!Method->isStatic()) {
          Ctx = Method->getParent();
          MemberType = Method->getType();
        }
      } else if (FunctionTemplateDecl *FunTmpl 
                   = dyn_cast<FunctionTemplateDecl>(D)) {
        if (CXXMethodDecl *Method 
              = dyn_cast<CXXMethodDecl>(FunTmpl->getTemplatedDecl())) {
          if (!Method->isStatic()) {          
            Ctx = Method->getParent();
            MemberType = Context.OverloadTy;
          }
        }
      } else if (OverloadedFunctionDecl *Ovl 
                   = dyn_cast<OverloadedFunctionDecl>(D)) {
        // FIXME: We need an abstraction for iterating over one or more function
        // templates or functions. This code is far too repetitive!
        for (OverloadedFunctionDecl::function_iterator 
               Func = Ovl->function_begin(),
               FuncEnd = Ovl->function_end();
             Func != FuncEnd; ++Func) {
          CXXMethodDecl *DMethod = 0;
          if (FunctionTemplateDecl *FunTmpl 
                = dyn_cast<FunctionTemplateDecl>(*Func))
            DMethod = dyn_cast<CXXMethodDecl>(FunTmpl->getTemplatedDecl());
          else
            DMethod = dyn_cast<CXXMethodDecl>(*Func);

          if (DMethod && !DMethod->isStatic()) {
            Ctx = DMethod->getDeclContext();
            MemberType = Context.OverloadTy;
            break;
          }
        }
      }

      if (Ctx && Ctx->isRecord()) {
        QualType CtxType = Context.getTagDeclType(cast<CXXRecordDecl>(Ctx));
        QualType ThisType = Context.getTagDeclType(MD->getParent());
        if ((Context.getCanonicalType(CtxType) 
               == Context.getCanonicalType(ThisType)) ||
            IsDerivedFrom(ThisType, CtxType)) {
          // Build the implicit member access expression.
          Expr *This = new (Context) CXXThisExpr(SourceLocation(),
                                                 MD->getThisType(Context));
          MarkDeclarationReferenced(Loc, D);
          if (PerformObjectMemberConversion(This, D))
            return ExprError();
          if (DiagnoseUseOfDecl(D, Loc))
            return ExprError();
          return Owned(new (Context) MemberExpr(This, true, D,
                                                Loc, MemberType));
        }
      }
    }
  }

  if (FieldDecl *FD = dyn_cast<FieldDecl>(D)) {
    if (CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(CurContext)) {
      if (MD->isStatic())
        // "invalid use of member 'x' in static member function"
        return ExprError(Diag(Loc,diag::err_invalid_member_use_in_static_method)
          << FD->getDeclName());
    }

    // Any other ways we could have found the field in a well-formed
    // program would have been turned into implicit member expressions
    // above.
    return ExprError(Diag(Loc, diag::err_invalid_non_static_member_use)
      << FD->getDeclName());
  }

  if (isa<TypedefDecl>(D))
    return ExprError(Diag(Loc, diag::err_unexpected_typedef) << Name);
  if (isa<ObjCInterfaceDecl>(D))
    return ExprError(Diag(Loc, diag::err_unexpected_interface) << Name);
  if (isa<NamespaceDecl>(D))
    return ExprError(Diag(Loc, diag::err_unexpected_namespace) << Name);

  // Make the DeclRefExpr or BlockDeclRefExpr for the decl.
  if (OverloadedFunctionDecl *Ovl = dyn_cast<OverloadedFunctionDecl>(D))
    return BuildDeclRefExpr(Ovl, Context.OverloadTy, Loc,
                           false, false, SS);
  else if (TemplateDecl *Template = dyn_cast<TemplateDecl>(D))
    return BuildDeclRefExpr(Template, Context.OverloadTy, Loc,
                            false, false, SS);
  ValueDecl *VD = cast<ValueDecl>(D);

  // Check whether this declaration can be used. Note that we suppress
  // this check when we're going to perform argument-dependent lookup
  // on this function name, because this might not be the function
  // that overload resolution actually selects.
  bool ADL = getLangOptions().CPlusPlus && (!SS || !SS->isSet()) && 
             HasTrailingLParen;
  if (!(ADL && isa<FunctionDecl>(VD)) && DiagnoseUseOfDecl(VD, Loc))
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
  if (CurBlock && ShouldSnapshotBlockValueReference(CurBlock, VD)) {
    MarkDeclarationReferenced(Loc, VD);
    QualType ExprTy = VD->getType().getNonReferenceType();
    // The BlocksAttr indicates the variable is bound by-reference.
    if (VD->getAttr<BlocksAttr>())
      return Owned(new (Context) BlockDeclRefExpr(VD, ExprTy, Loc, true));
    // This is to record that a 'const' was actually synthesize and added.
    bool constAdded = !ExprTy.isConstQualified();
    // Variable will be bound by-copy, make it const within the closure.
    
    ExprTy.addConst();
    return Owned(new (Context) BlockDeclRefExpr(VD, ExprTy, Loc, false, 
                                                constAdded));
  }
  // If this reference is not in a block or if the referenced variable is
  // within the block, create a normal DeclRefExpr.

  bool TypeDependent = false;
  bool ValueDependent = false;
  if (getLangOptions().CPlusPlus) {
    // C++ [temp.dep.expr]p3:
    //   An id-expression is type-dependent if it contains:   
    //     - an identifier that was declared with a dependent type,
    if (VD->getType()->isDependentType())
      TypeDependent = true;
    //     - FIXME: a template-id that is dependent,
    //     - a conversion-function-id that specifies a dependent type,
    else if (Name.getNameKind() == DeclarationName::CXXConversionFunctionName &&
             Name.getCXXNameType()->isDependentType())
      TypeDependent = true;
    //     - a nested-name-specifier that contains a class-name that
    //       names a dependent type.
    else if (SS && !SS->isEmpty()) {
      for (DeclContext *DC = computeDeclContext(*SS);
           DC; DC = DC->getParent()) {
        // FIXME: could stop early at namespace scope.
        if (DC->isRecord()) {
          CXXRecordDecl *Record = cast<CXXRecordDecl>(DC);
          if (Context.getTypeDeclType(Record)->isDependentType()) {
            TypeDependent = true;
            break;
          }
        }
      }
    }

    // C++ [temp.dep.constexpr]p2:
    //
    //   An identifier is value-dependent if it is:
    //     - a name declared with a dependent type,
    if (TypeDependent)
      ValueDependent = true;
    //     - the name of a non-type template parameter,
    else if (isa<NonTypeTemplateParmDecl>(VD))
      ValueDependent = true;
    //    - a constant with integral or enumeration type and is
    //      initialized with an expression that is value-dependent
    else if (const VarDecl *Dcl = dyn_cast<VarDecl>(VD)) {
      if (Dcl->getType().getCVRQualifiers() == QualType::Const &&
          Dcl->getInit()) {
        ValueDependent = Dcl->getInit()->isValueDependent();
      }
    }
  }

  return BuildDeclRefExpr(VD, VD->getType().getNonReferenceType(), Loc,
                          TypeDependent, ValueDependent, SS);
}

Sema::OwningExprResult Sema::ActOnPredefinedExpr(SourceLocation Loc,
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
  unsigned Length;
  if (FunctionDecl *FD = getCurFunctionDecl())
    Length = FD->getIdentifier()->getLength();
  else if (ObjCMethodDecl *MD = getCurMethodDecl())
    Length = MD->getSynthesizedMethodSize();
  else {
    Diag(Loc, diag::ext_predef_outside_function);
    // __PRETTY_FUNCTION__ -> "top level", the others produce an empty string.
    Length = IT == PredefinedExpr::PrettyFunction ? strlen("top level") : 0;
  }


  llvm::APInt LengthI(32, Length + 1);
  QualType ResTy = Context.CharTy.getQualifiedType(QualType::Const);
  ResTy = Context.getConstantArrayType(ResTy, LengthI, ArrayType::Normal, 0);
  return Owned(new (Context) PredefinedExpr(Loc, ResTy, IT));
}

Sema::OwningExprResult Sema::ActOnCharacterConstant(const Token &Tok) {
  llvm::SmallString<16> CharBuffer;
  CharBuffer.resize(Tok.getLength());
  const char *ThisTokBegin = &CharBuffer[0];
  unsigned ActualLength = PP.getSpelling(Tok, ThisTokBegin);

  CharLiteralParser Literal(ThisTokBegin, ThisTokBegin+ActualLength,
                            Tok.getLocation(), PP);
  if (Literal.hadError())
    return ExprError();

  QualType type = getLangOptions().CPlusPlus ? Context.CharTy : Context.IntTy;

  return Owned(new (Context) CharacterLiteral(Literal.getValue(),
                                              Literal.isWide(),
                                              type, Tok.getLocation()));
}

Action::OwningExprResult Sema::ActOnNumericConstant(const Token &Tok) {
  // Fast path for a single digit (which is quite common).  A single digit
  // cannot have a trigraph, escaped newline, radix prefix, or type suffix.
  if (Tok.getLength() == 1) {
    const char Val = PP.getSpellingOfSingleCharacterNumericConstant(Tok);
    unsigned IntSize = Context.Target.getIntWidth();
    return Owned(new (Context) IntegerLiteral(llvm::APInt(IntSize, Val-'0'),
                    Context.IntTy, Tok.getLocation()));
  }

  llvm::SmallString<512> IntegerBuffer;
  // Add padding so that NumericLiteralParser can overread by one character.
  IntegerBuffer.resize(Tok.getLength()+1);
  const char *ThisTokBegin = &IntegerBuffer[0];

  // Get the spelling of the token, which eliminates trigraphs, etc.
  unsigned ActualLength = PP.getSpelling(Tok, ThisTokBegin);

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

    // isExact will be set by GetFloatValue().
    bool isExact = false;
    llvm::APFloat Val = Literal.GetFloatValue(Format, &isExact);
    Res = new (Context) FloatingLiteral(Val, isExact, Ty, Tok.getLocation());

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
    Res = new (Context) IntegerLiteral(ResultVal, Ty, Tok.getLocation());
  }

  // If this is an imaginary literal, create the ImaginaryLiteral wrapper.
  if (Literal.isImaginary)
    Res = new (Context) ImaginaryLiteral(Res, 
                                        Context.getComplexType(Res->getType()));

  return Owned(Res);
}

Action::OwningExprResult Sema::ActOnParenExpr(SourceLocation L,
                                              SourceLocation R, ExprArg Val) {
  Expr *E = Val.takeAs<Expr>();
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

  // C99 6.5.3.4p1:
  if (isa<FunctionType>(exprType)) {
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
                          isSizeof ? diag::err_sizeof_incomplete_type : 
                          diag::err_alignof_incomplete_type,
                          ExprRange))
    return true;
  
  // Reject sizeof(interface) and sizeof(interface<proto>) in 64-bit mode.
  if (LangOpts.ObjCNonFragileABI && exprType->isObjCInterfaceType()) {
    Diag(OpLoc, diag::err_sizeof_nonfragile_interface)
      << exprType << isSizeof << ExprRange;
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
Action::OwningExprResult 
Sema::CreateSizeOfAlignOfExpr(QualType T, SourceLocation OpLoc, 
                              bool isSizeOf, SourceRange R) {
  if (T.isNull())
    return ExprError();

  if (!T->isDependentType() &&
      CheckSizeOfAlignOfOperand(T, OpLoc, R, isSizeOf))
    return ExprError();

  // C99 6.5.3.4p4: the type (an unsigned integer type) is size_t.
  return Owned(new (Context) SizeOfAlignOfExpr(isSizeOf, T,
                                               Context.getSizeType(), OpLoc,
                                               R.getEnd()));
}

/// \brief Build a sizeof or alignof expression given an expression
/// operand.
Action::OwningExprResult 
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
Action::OwningExprResult
Sema::ActOnSizeOfAlignOfExpr(SourceLocation OpLoc, bool isSizeof, bool isType,
                             void *TyOrEx, const SourceRange &ArgRange) {
  // If error parsing type, ignore.
  if (TyOrEx == 0) return ExprError();

  if (isType) {
    // FIXME: Preserve type source info.
    QualType ArgTy = GetTypeFromParser(TyOrEx);
    return CreateSizeOfAlignOfExpr(ArgTy, OpLoc, isSizeof, ArgRange);
  } 

  // Get the end location.
  Expr *ArgEx = (Expr *)TyOrEx;
  Action::OwningExprResult Result
    = CreateSizeOfAlignOfExpr(ArgEx, OpLoc, isSizeof, ArgEx->getSourceRange());

  if (Result.isInvalid())
    DeleteExpr(ArgEx);

  return move(Result);
}

QualType Sema::CheckRealImagOperand(Expr *&V, SourceLocation Loc, bool isReal) {
  if (V->isTypeDependent())
    return Context.DependentTy;
  
  // These operators return the element type of a complex type.
  if (const ComplexType *CT = V->getType()->getAsComplexType())
    return CT->getElementType();
  
  // Otherwise they pass through real integer and floating point types here.
  if (V->getType()->isArithmeticType())
    return V->getType();
  
  // Reject anything else.
  Diag(Loc, diag::err_realimag_invalid_type) << V->getType()
    << (isReal ? "__real" : "__imag");
  return QualType();
}



Action::OwningExprResult
Sema::ActOnPostfixUnaryOp(Scope *S, SourceLocation OpLoc,
                          tok::TokenKind Kind, ExprArg Input) {
  // Since this might be a postfix expression, get rid of ParenListExprs.
  Input = MaybeConvertParenListExprToParenExpr(S, move(Input));
  Expr *Arg = (Expr *)Input.get();

  UnaryOperator::Opcode Opc;
  switch (Kind) {
  default: assert(0 && "Unknown unary op!");
  case tok::plusplus:   Opc = UnaryOperator::PostInc; break;
  case tok::minusminus: Opc = UnaryOperator::PostDec; break;
  }

  if (getLangOptions().CPlusPlus &&
      (Arg->getType()->isRecordType() || Arg->getType()->isEnumeralType())) {
    // Which overloaded operator?
    OverloadedOperatorKind OverOp =
      (Opc == UnaryOperator::PostInc)? OO_PlusPlus : OO_MinusMinus;

    // C++ [over.inc]p1:
    //
    //     [...] If the function is a member function with one
    //     parameter (which shall be of type int) or a non-member
    //     function with two parameters (the second of which shall be
    //     of type int), it defines the postfix increment operator ++
    //     for objects of that type. When the postfix increment is
    //     called as a result of using the ++ operator, the int
    //     argument will have value zero.
    Expr *Args[2] = { 
      Arg, 
      new (Context) IntegerLiteral(llvm::APInt(Context.Target.getIntWidth(), 0, 
                          /*isSigned=*/true), Context.IntTy, SourceLocation())
    };

    // Build the candidate set for overloading
    OverloadCandidateSet CandidateSet;
    AddOperatorCandidates(OverOp, S, OpLoc, Args, 2, CandidateSet);

    // Perform overload resolution.
    OverloadCandidateSet::iterator Best;
    switch (BestViableFunction(CandidateSet, OpLoc, Best)) {
    case OR_Success: {
      // We found a built-in operator or an overloaded operator.
      FunctionDecl *FnDecl = Best->Function;

      if (FnDecl) {
        // We matched an overloaded operator. Build a call to that
        // operator.

        // Convert the arguments.
        if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(FnDecl)) {
          if (PerformObjectArgumentInitialization(Arg, Method))
            return ExprError();
        } else {
          // Convert the arguments.
          if (PerformCopyInitialization(Arg,
                                        FnDecl->getParamDecl(0)->getType(),
                                        "passing"))
            return ExprError();
        }

        // Determine the result type
        QualType ResultTy
          = FnDecl->getType()->getAsFunctionType()->getResultType();
        ResultTy = ResultTy.getNonReferenceType();

        // Build the actual expression node.
        Expr *FnExpr = new (Context) DeclRefExpr(FnDecl, FnDecl->getType(),
                                                 SourceLocation());
        UsualUnaryConversions(FnExpr);

        Input.release();
        Args[0] = Arg;
        return Owned(new (Context) CXXOperatorCallExpr(Context, OverOp, FnExpr,
                                                       Args, 2, ResultTy, 
                                                       OpLoc));
      } else {
        // We matched a built-in operator. Convert the arguments, then
        // break out so that we will build the appropriate built-in
        // operator node.
        if (PerformCopyInitialization(Arg, Best->BuiltinTypes.ParamTypes[0],
                                      "passing"))
          return ExprError();

        break;
      }
    }

    case OR_No_Viable_Function:
      // No viable function; fall through to handling this as a
      // built-in operator, which will produce an error message for us.
      break;

    case OR_Ambiguous:
      Diag(OpLoc,  diag::err_ovl_ambiguous_oper)
          << UnaryOperator::getOpcodeStr(Opc)
          << Arg->getSourceRange();
      PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/true);
      return ExprError();

    case OR_Deleted:
      Diag(OpLoc, diag::err_ovl_deleted_oper)
        << Best->Function->isDeleted()
        << UnaryOperator::getOpcodeStr(Opc)
        << Arg->getSourceRange();
      PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/true);
      return ExprError();
    }

    // Either we found no viable overloaded operator or we matched a
    // built-in operator. In either case, fall through to trying to
    // build a built-in operation.
  }

  Input.release();
  Input = Arg;
  return CreateBuiltinUnaryOp(OpLoc, Opc, move(Input));
}

Action::OwningExprResult
Sema::ActOnArraySubscriptExpr(Scope *S, ExprArg Base, SourceLocation LLoc,
                              ExprArg Idx, SourceLocation RLoc) {
  // Since this might be a postfix expression, get rid of ParenListExprs.
  Base = MaybeConvertParenListExprToParenExpr(S, move(Base));

  Expr *LHSExp = static_cast<Expr*>(Base.get()),
       *RHSExp = static_cast<Expr*>(Idx.get());
  
  if (getLangOptions().CPlusPlus &&
      (LHSExp->isTypeDependent() || RHSExp->isTypeDependent())) {
    Base.release();
    Idx.release();
    return Owned(new (Context) ArraySubscriptExpr(LHSExp, RHSExp,
                                                  Context.DependentTy, RLoc));
  }

  if (getLangOptions().CPlusPlus && 
      (LHSExp->getType()->isRecordType() ||
       LHSExp->getType()->isEnumeralType() ||
       RHSExp->getType()->isRecordType() ||
       RHSExp->getType()->isEnumeralType())) {
    // Add the appropriate overloaded operators (C++ [over.match.oper]) 
    // to the candidate set.
    OverloadCandidateSet CandidateSet;
    Expr *Args[2] = { LHSExp, RHSExp };
    AddOperatorCandidates(OO_Subscript, S, LLoc, Args, 2, CandidateSet,
                          SourceRange(LLoc, RLoc));

    // Perform overload resolution.
    OverloadCandidateSet::iterator Best;
    switch (BestViableFunction(CandidateSet, LLoc, Best)) {
    case OR_Success: {
      // We found a built-in operator or an overloaded operator.
      FunctionDecl *FnDecl = Best->Function;

      if (FnDecl) {
        // We matched an overloaded operator. Build a call to that
        // operator.

        // Convert the arguments.
        if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(FnDecl)) {
          if (PerformObjectArgumentInitialization(LHSExp, Method) ||
              PerformCopyInitialization(RHSExp, 
                                        FnDecl->getParamDecl(0)->getType(),
                                        "passing"))
            return ExprError();
        } else {
          // Convert the arguments.
          if (PerformCopyInitialization(LHSExp,
                                        FnDecl->getParamDecl(0)->getType(),
                                        "passing") ||
              PerformCopyInitialization(RHSExp,
                                        FnDecl->getParamDecl(1)->getType(),
                                        "passing"))
            return ExprError();
        }

        // Determine the result type
        QualType ResultTy
          = FnDecl->getType()->getAsFunctionType()->getResultType();
        ResultTy = ResultTy.getNonReferenceType();

        // Build the actual expression node.
        Expr *FnExpr = new (Context) DeclRefExpr(FnDecl, FnDecl->getType(),
                                                 SourceLocation());
        UsualUnaryConversions(FnExpr);

        Base.release();
        Idx.release();
        Args[0] = LHSExp;
        Args[1] = RHSExp;
        return Owned(new (Context) CXXOperatorCallExpr(Context, OO_Subscript,
                                                       FnExpr, Args, 2, 
                                                       ResultTy, LLoc));
      } else {
        // We matched a built-in operator. Convert the arguments, then
        // break out so that we will build the appropriate built-in
        // operator node.
        if (PerformCopyInitialization(LHSExp, Best->BuiltinTypes.ParamTypes[0],
                                      "passing") ||
            PerformCopyInitialization(RHSExp, Best->BuiltinTypes.ParamTypes[1],
                                      "passing"))
          return ExprError();

        break;
      }
    }

    case OR_No_Viable_Function:
      // No viable function; fall through to handling this as a
      // built-in operator, which will produce an error message for us.
      break;

    case OR_Ambiguous:
      Diag(LLoc,  diag::err_ovl_ambiguous_oper)
          << "[]"
          << LHSExp->getSourceRange() << RHSExp->getSourceRange();
      PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/true);
      return ExprError();

    case OR_Deleted:
      Diag(LLoc, diag::err_ovl_deleted_oper)
        << Best->Function->isDeleted()
        << "[]"
        << LHSExp->getSourceRange() << RHSExp->getSourceRange();
      PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/true);
      return ExprError();
    }

    // Either we found no viable overloaded operator or we matched a
    // built-in operator. In either case, fall through to trying to
    // build a built-in operation.
  }

  // Perform default conversions.
  DefaultFunctionArrayConversion(LHSExp);
  DefaultFunctionArrayConversion(RHSExp);

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
               LHSTy->getAsObjCObjectPointerType()) {
    BaseExpr = LHSExp;
    IndexExpr = RHSExp;
    ResultType = PTy->getPointeeType();
  } else if (const ObjCObjectPointerType *PTy = 
               RHSTy->getAsObjCObjectPointerType()) {
     // Handle the uncommon case of "123[Ptr]".
    BaseExpr = RHSExp;
    IndexExpr = LHSExp;
    ResultType = PTy->getPointeeType();
  } else if (const VectorType *VTy = LHSTy->getAsVectorType()) {
    BaseExpr = LHSExp;    // vectors: V[123]
    IndexExpr = RHSExp;

    // FIXME: need to deal with const...
    ResultType = VTy->getElementType();
  } else if (LHSTy->isArrayType()) {
    // If we see an array that wasn't promoted by
    // DefaultFunctionArrayConversion, it must be an array that
    // wasn't promoted because of the C90 rule that doesn't
    // allow promoting non-lvalue arrays.  Warn, then
    // force the promotion here.
    Diag(LHSExp->getLocStart(), diag::ext_subscript_non_lvalue) <<
        LHSExp->getSourceRange();
    ImpCastExprToType(LHSExp, Context.getArrayDecayedType(LHSTy));
    LHSTy = LHSExp->getType();

    BaseExpr = LHSExp;
    IndexExpr = RHSExp;
    ResultType = LHSTy->getAs<PointerType>()->getPointeeType();
  } else if (RHSTy->isArrayType()) {
    // Same as previous, except for 123[f().a] case
    Diag(RHSExp->getLocStart(), diag::ext_subscript_non_lvalue) <<
        RHSExp->getSourceRange();
    ImpCastExprToType(RHSExp, Context.getArrayDecayedType(RHSTy));
    RHSTy = RHSExp->getType();

    BaseExpr = RHSExp;
    IndexExpr = LHSExp;
    ResultType = RHSTy->getAs<PointerType>()->getPointeeType();
  } else {
    return ExprError(Diag(LLoc, diag::err_typecheck_subscript_value)
       << LHSExp->getSourceRange() << RHSExp->getSourceRange());
  }
  // C99 6.5.2.1p1
  if (!(IndexExpr->getType()->isIntegerType() &&
        IndexExpr->getType()->isScalarType()) && !IndexExpr->isTypeDependent())
    return ExprError(Diag(LLoc, diag::err_typecheck_subscript_not_integer)
                     << IndexExpr->getSourceRange());

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
      RequireCompleteType(LLoc, ResultType, diag::err_subscript_incomplete_type,
                          BaseExpr->getSourceRange()))
    return ExprError();
  
  // Diagnose bad cases where we step over interface counts.
  if (ResultType->isObjCInterfaceType() && LangOpts.ObjCNonFragileABI) {
    Diag(LLoc, diag::err_subscript_nonfragile_interface)
      << ResultType << BaseExpr->getSourceRange();
    return ExprError();
  }
  
  Base.release();
  Idx.release();
  return Owned(new (Context) ArraySubscriptExpr(LHSExp, RHSExp,
                                                ResultType, RLoc));
}

QualType Sema::
CheckExtVectorComponent(QualType baseType, SourceLocation OpLoc,
                        IdentifierInfo &CompName, SourceLocation CompLoc) {
  const ExtVectorType *vecType = baseType->getAsExtVectorType();

  // The vector accessor can't exceed the number of elements.
  const char *compStr = CompName.getName();

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
      << std::string(compStr,compStr+1) << SourceRange(CompLoc);
    return QualType();
  }

  // Ensure no component accessor exceeds the width of the vector type it
  // operates on.
  if (!HalvingSwizzle) {
    compStr = CompName.getName();

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

  // If this is a halving swizzle, verify that the base type has an even
  // number of elements.
  if (HalvingSwizzle && (vecType->getNumElements() & 1U)) {
    Diag(OpLoc, diag::err_ext_vector_component_requires_even)
      << baseType << SourceRange(CompLoc);
    return QualType();
  }

  // The component accessor looks fine - now we need to compute the actual type.
  // The vector type is implied by the component accessor. For example,
  // vec4.b is a float, vec4.xy is a vec2, vec4.rgb is a vec3, etc.
  // vec4.s0 is a float, vec4.s23 is a vec3, etc.
  // vec4.hi, vec4.lo, vec4.e, and vec4.o all return vec2.
  unsigned CompSize = HalvingSwizzle ? vecType->getNumElements() / 2
                                     : CompName.getLength();
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
                                                IdentifierInfo &Member,
                                                const Selector &Sel,
                                                ASTContext &Context) {
  
  if (ObjCPropertyDecl *PD = PDecl->FindPropertyDeclaration(&Member))
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
                                IdentifierInfo &Member,
                                const Selector &Sel,
                                ASTContext &Context) {
  // Check protocols on qualified interfaces.
  Decl *GDecl = 0;
  for (ObjCObjectPointerType::qual_iterator I = QIdTy->qual_begin(),
       E = QIdTy->qual_end(); I != E; ++I) {
    if (ObjCPropertyDecl *PD = (*I)->FindPropertyDeclaration(&Member)) {
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

/// FindMethodInNestedImplementations - Look up a method in current and
/// all base class implementations.
///
ObjCMethodDecl *Sema::FindMethodInNestedImplementations(
                                              const ObjCInterfaceDecl *IFace,
                                              const Selector &Sel) {
  ObjCMethodDecl *Method = 0;
  if (ObjCImplementationDecl *ImpDecl = IFace->getImplementation())
    Method = ImpDecl->getInstanceMethod(Sel);
  
  if (!Method && IFace->getSuperClass())
    return FindMethodInNestedImplementations(IFace->getSuperClass(), Sel);
  return Method;
}

Action::OwningExprResult
Sema::ActOnMemberReferenceExpr(Scope *S, ExprArg Base, SourceLocation OpLoc,
                               tok::TokenKind OpKind, SourceLocation MemberLoc,
                               IdentifierInfo &Member,
                               DeclPtrTy ObjCImpDecl, const CXXScopeSpec *SS) {
  // FIXME: handle the CXXScopeSpec for proper lookup of qualified-ids
  if (SS && SS->isInvalid())
    return ExprError();

  // Since this might be a postfix expression, get rid of ParenListExprs.
  Base = MaybeConvertParenListExprToParenExpr(S, move(Base));

  Expr *BaseExpr = Base.takeAs<Expr>();
  assert(BaseExpr && "no record expression");
  
  // Perform default conversions.
  DefaultFunctionArrayConversion(BaseExpr);

  QualType BaseType = BaseExpr->getType();
  // If this is an Objective-C pseudo-builtin and a definition is provided then
  // use that.
  if (BaseType->isObjCIdType()) {
    // We have an 'id' type. Rather than fall through, we check if this
    // is a reference to 'isa'.
    if (BaseType != Context.ObjCIdRedefinitionType) {
      BaseType = Context.ObjCIdRedefinitionType;
      ImpCastExprToType(BaseExpr, BaseType);
    }
  } else if (BaseType->isObjCClassType() &&
      BaseType != Context.ObjCClassRedefinitionType) {
    BaseType = Context.ObjCClassRedefinitionType;
    ImpCastExprToType(BaseExpr, BaseType);
  }
  assert(!BaseType.isNull() && "no type for member expression");

  // Get the type being accessed in BaseType.  If this is an arrow, the BaseExpr
  // must have pointer type, and the accessed type is the pointee.
  if (OpKind == tok::arrow) {
    if (BaseType->isDependentType())
      return Owned(new (Context) CXXUnresolvedMemberExpr(Context,
                                                         BaseExpr, true, 
                                                         OpLoc, 
                                                     DeclarationName(&Member),
                                                         MemberLoc));
    else if (const PointerType *PT = BaseType->getAs<PointerType>())
      BaseType = PT->getPointeeType();
    else if (BaseType->isObjCObjectPointerType())
      ;
    else
      return ExprError(Diag(MemberLoc,
                            diag::err_typecheck_member_reference_arrow)
        << BaseType << BaseExpr->getSourceRange());
  } else {
    if (BaseType->isDependentType()) {
      // Require that the base type isn't a pointer type 
      // (so we'll report an error for)
      // T* t;
      // t.f;
      // 
      // In Obj-C++, however, the above expression is valid, since it could be
      // accessing the 'f' property if T is an Obj-C interface. The extra check
      // allows this, while still reporting an error if T is a struct pointer.
      const PointerType *PT = BaseType->getAs<PointerType>();

      if (!PT || (getLangOptions().ObjC1 && 
                  !PT->getPointeeType()->isRecordType()))
        return Owned(new (Context) CXXUnresolvedMemberExpr(Context,
                                                           BaseExpr, false, 
                                                           OpLoc, 
                                                     DeclarationName(&Member),
                                                           MemberLoc));
    }
  }

  // Handle field access to simple records.  This also handles access to fields
  // of the ObjC 'id' struct.
  if (const RecordType *RTy = BaseType->getAs<RecordType>()) {
    RecordDecl *RDecl = RTy->getDecl();
    if (RequireCompleteType(OpLoc, BaseType,
                               diag::err_typecheck_incomplete_tag,
                               BaseExpr->getSourceRange()))
      return ExprError();

    DeclContext *DC = RDecl;
    if (SS && SS->isSet()) {
      // If the member name was a qualified-id, look into the
      // nested-name-specifier.
      DC = computeDeclContext(*SS, false);
      
      // FIXME: If DC is not computable, we should build a 
      // CXXUnresolvedMemberExpr.
      assert(DC && "Cannot handle non-computable dependent contexts in lookup");
    }

    // The record definition is complete, now make sure the member is valid.
    LookupResult Result
      = LookupQualifiedName(DC, DeclarationName(&Member),  
                            LookupMemberName, false);

    if (SS && SS->isSet()) {
      QualType BaseTypeCanon 
        = Context.getCanonicalType(BaseType).getUnqualifiedType();
      QualType MemberTypeCanon 
        = Context.getCanonicalType(
            Context.getTypeDeclType(
                     dyn_cast<TypeDecl>(Result.getAsDecl()->getDeclContext())));

      if (BaseTypeCanon != MemberTypeCanon &&
          !IsDerivedFrom(BaseTypeCanon, MemberTypeCanon))
        return ExprError(Diag(SS->getBeginLoc(),
                              diag::err_not_direct_base_or_virtual)
                         << MemberTypeCanon << BaseTypeCanon);
    }

    if (!Result)
      return ExprError(Diag(MemberLoc, diag::err_typecheck_no_member)
               << &Member << BaseExpr->getSourceRange());
    if (Result.isAmbiguous()) {
      DiagnoseAmbiguousLookup(Result, DeclarationName(&Member),
                              MemberLoc, BaseExpr->getSourceRange());
      return ExprError();
    }
    
    NamedDecl *MemberDecl = Result;

    // If the decl being referenced had an error, return an error for this
    // sub-expr without emitting another error, in order to avoid cascading
    // error cases.
    if (MemberDecl->isInvalidDecl())
      return ExprError();

    // Check the use of this field
    if (DiagnoseUseOfDecl(MemberDecl, MemberLoc))
      return ExprError();

    if (FieldDecl *FD = dyn_cast<FieldDecl>(MemberDecl)) {
      // We may have found a field within an anonymous union or struct
      // (C++ [class.union]).
      if (cast<RecordDecl>(FD->getDeclContext())->isAnonymousStructOrUnion())
        return BuildAnonymousStructUnionMemberReference(MemberLoc, FD,
                                                        BaseExpr, OpLoc);

      // Figure out the type of the member; see C99 6.5.2.3p3, C++ [expr.ref]
      QualType MemberType = FD->getType();
      if (const ReferenceType *Ref = MemberType->getAs<ReferenceType>())
        MemberType = Ref->getPointeeType();
      else {
        unsigned BaseAddrSpace = BaseType.getAddressSpace();
        unsigned combinedQualifiers =
          MemberType.getCVRQualifiers() | BaseType.getCVRQualifiers();
        if (FD->isMutable())
          combinedQualifiers &= ~QualType::Const;
        MemberType = MemberType.getQualifiedType(combinedQualifiers);
        if (BaseAddrSpace != MemberType.getAddressSpace())
           MemberType = Context.getAddrSpaceQualType(MemberType, BaseAddrSpace);
      }

      MarkDeclarationReferenced(MemberLoc, FD);
      if (PerformObjectMemberConversion(BaseExpr, FD))
        return ExprError();
      return Owned(new (Context) MemberExpr(BaseExpr, OpKind == tok::arrow, FD,
                                            MemberLoc, MemberType));
    }
    
    if (VarDecl *Var = dyn_cast<VarDecl>(MemberDecl)) {
      MarkDeclarationReferenced(MemberLoc, MemberDecl);
      return Owned(new (Context) MemberExpr(BaseExpr, OpKind == tok::arrow,
                                            Var, MemberLoc,
                                         Var->getType().getNonReferenceType()));
    }
    if (FunctionDecl *MemberFn = dyn_cast<FunctionDecl>(MemberDecl)) {
      MarkDeclarationReferenced(MemberLoc, MemberDecl);
      return Owned(new (Context) MemberExpr(BaseExpr, OpKind == tok::arrow,
                                            MemberFn, MemberLoc,
                                            MemberFn->getType()));
    }
    if (FunctionTemplateDecl *FunTmpl 
          = dyn_cast<FunctionTemplateDecl>(MemberDecl)) {
      MarkDeclarationReferenced(MemberLoc, MemberDecl);
      return Owned(new (Context) MemberExpr(BaseExpr, OpKind == tok::arrow,
                                            FunTmpl, MemberLoc,
                                            Context.OverloadTy));
    }
    if (OverloadedFunctionDecl *Ovl
          = dyn_cast<OverloadedFunctionDecl>(MemberDecl))
      return Owned(new (Context) MemberExpr(BaseExpr, OpKind == tok::arrow, Ovl,
                                            MemberLoc, Context.OverloadTy));
    if (EnumConstantDecl *Enum = dyn_cast<EnumConstantDecl>(MemberDecl)) {
      MarkDeclarationReferenced(MemberLoc, MemberDecl);
      return Owned(new (Context) MemberExpr(BaseExpr, OpKind == tok::arrow,
                                            Enum, MemberLoc, Enum->getType()));
    }
    if (isa<TypeDecl>(MemberDecl))
      return ExprError(Diag(MemberLoc,diag::err_typecheck_member_reference_type)
        << DeclarationName(&Member) << int(OpKind == tok::arrow));

    // We found a declaration kind that we didn't expect. This is a
    // generic error message that tells the user that she can't refer
    // to this member with '.' or '->'.
    return ExprError(Diag(MemberLoc,
                          diag::err_typecheck_member_reference_unknown)
      << DeclarationName(&Member) << int(OpKind == tok::arrow));
  }

  // Handle properties on ObjC 'Class' types.
  if (OpKind == tok::period && BaseType->isObjCClassType()) {
    // Also must look for a getter name which uses property syntax.
    Selector Sel = PP.getSelectorTable().getNullarySelector(&Member);
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
                                           PP.getSelectorTable(), &Member);
      ObjCMethodDecl *Setter = IFace->lookupClassMethod(SetterSel);
      if (!Setter) {
        // If this reference is in an @implementation, also check for 'private'
        // methods.
        Setter = FindMethodInNestedImplementations(IFace, SetterSel);
      }
      // Look through local category implementations associated with the class.
      if (!Setter)
        Setter = IFace->getCategoryClassMethod(SetterSel);

      if (Setter && DiagnoseUseOfDecl(Setter, MemberLoc))
        return ExprError();

      if (Getter || Setter) {
        QualType PType;

        if (Getter)
          PType = Getter->getResultType();
        else
          // Get the expression type from Setter's incoming parameter.
          PType = (*(Setter->param_end() -1))->getType();
        // FIXME: we must check that the setter has property type.
        return Owned(new (Context) ObjCImplicitSetterGetterRefExpr(Getter, PType,
                                        Setter, MemberLoc, BaseExpr));
      }
      return ExprError(Diag(MemberLoc, diag::err_property_not_found)
        << &Member << BaseType);
    }
  }
  // Handle access to Objective-C instance variables, such as "Obj->ivar" and
  // (*Obj).ivar.
  if ((OpKind == tok::arrow && BaseType->isObjCObjectPointerType()) ||
      (OpKind == tok::period && BaseType->isObjCInterfaceType())) {
    const ObjCObjectPointerType *OPT = BaseType->getAsObjCObjectPointerType();
    const ObjCInterfaceType *IFaceT = 
      OPT ? OPT->getInterfaceType() : BaseType->getAsObjCInterfaceType();
    if (IFaceT) {
      ObjCInterfaceDecl *IDecl = IFaceT->getDecl();
      ObjCInterfaceDecl *ClassDeclared;
      ObjCIvarDecl *IV = IDecl->lookupInstanceVariable(&Member, ClassDeclared);
      
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
            Decl *ImplDecl = ObjCImpDecl.getAs<Decl>();
            if (ObjCImplementationDecl *IMPD = 
                dyn_cast<ObjCImplementationDecl>(ImplDecl))
              ClassOfMethodDecl = IMPD->getClassInterface();
            else if (ObjCCategoryImplDecl* CatImplClass =
                        dyn_cast<ObjCCategoryImplDecl>(ImplDecl))
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
                                                   OpKind == tok::arrow));
      }
      return ExprError(Diag(MemberLoc, diag::err_typecheck_member_reference_ivar)
                         << IDecl->getDeclName() << &Member
                         << BaseExpr->getSourceRange());
    }
  }
  // Handle properties on 'id' and qualified "id".
  if (OpKind == tok::period && (BaseType->isObjCIdType() || 
                                BaseType->isObjCQualifiedIdType())) {
    const ObjCObjectPointerType *QIdTy = BaseType->getAsObjCObjectPointerType();
    
    // Check protocols on qualified interfaces.
    Selector Sel = PP.getSelectorTable().getNullarySelector(&Member);
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
        
        return Owned(new (Context) ObjCMessageExpr(BaseExpr, Sel,
                                                   OMD->getResultType(), 
                                                   OMD, OpLoc, MemberLoc, 
                                                   NULL, 0));
      }
    }

    return ExprError(Diag(MemberLoc, diag::err_property_not_found)
                       << &Member << BaseType);
  }
  // Handle Objective-C property access, which is "Obj.property" where Obj is a
  // pointer to a (potentially qualified) interface type.
  const ObjCObjectPointerType *OPT;
  if (OpKind == tok::period && 
      (OPT = BaseType->getAsObjCInterfacePointerType())) {
    const ObjCInterfaceType *IFaceT = OPT->getInterfaceType();
    ObjCInterfaceDecl *IFace = IFaceT->getDecl();
    
    // Search for a declared property first.
    if (ObjCPropertyDecl *PD = IFace->FindPropertyDeclaration(&Member)) {
      // Check whether we can reference this property.
      if (DiagnoseUseOfDecl(PD, MemberLoc))
        return ExprError();
      QualType ResTy = PD->getType();
      Selector Sel = PP.getSelectorTable().getNullarySelector(&Member);
      ObjCMethodDecl *Getter = IFace->lookupInstanceMethod(Sel);
      if (DiagnosePropertyAccessorMismatch(PD, Getter, MemberLoc))
        ResTy = Getter->getResultType();
      return Owned(new (Context) ObjCPropertyRefExpr(PD, ResTy,
                                                     MemberLoc, BaseExpr));
    }
    // Check protocols on qualified interfaces.
    for (ObjCObjectPointerType::qual_iterator I = OPT->qual_begin(),
         E = OPT->qual_end(); I != E; ++I)
      if (ObjCPropertyDecl *PD = (*I)->FindPropertyDeclaration(&Member)) {
        // Check whether we can reference this property.
        if (DiagnoseUseOfDecl(PD, MemberLoc))
          return ExprError();

        return Owned(new (Context) ObjCPropertyRefExpr(PD, PD->getType(),
                                                       MemberLoc, BaseExpr));
      }
    for (ObjCObjectPointerType::qual_iterator I = OPT->qual_begin(),
         E = OPT->qual_end(); I != E; ++I)
      if (ObjCPropertyDecl *PD = (*I)->FindPropertyDeclaration(&Member)) {
        // Check whether we can reference this property.
        if (DiagnoseUseOfDecl(PD, MemberLoc))
          return ExprError();

        return Owned(new (Context) ObjCPropertyRefExpr(PD, PD->getType(),
                                                       MemberLoc, BaseExpr));
      }
    // If that failed, look for an "implicit" property by seeing if the nullary
    // selector is implemented.

    // FIXME: The logic for looking up nullary and unary selectors should be
    // shared with the code in ActOnInstanceMessage.

    Selector Sel = PP.getSelectorTable().getNullarySelector(&Member);
    ObjCMethodDecl *Getter = IFace->lookupInstanceMethod(Sel);

    // If this reference is in an @implementation, check for 'private' methods.
    if (!Getter)
      Getter = FindMethodInNestedImplementations(IFace, Sel);

    // Look through local category implementations associated with the class.
    if (!Getter)
      Getter = IFace->getCategoryInstanceMethod(Sel);
    if (Getter) {
      // Check if we can reference this property.
      if (DiagnoseUseOfDecl(Getter, MemberLoc))
        return ExprError();
    }
    // If we found a getter then this may be a valid dot-reference, we
    // will look for the matching setter, in case it is needed.
    Selector SetterSel = 
      SelectorTable::constructSetterName(PP.getIdentifierTable(), 
                                         PP.getSelectorTable(), &Member);
    ObjCMethodDecl *Setter = IFace->lookupInstanceMethod(SetterSel);
    if (!Setter) {
      // If this reference is in an @implementation, also check for 'private'
      // methods.
      Setter = FindMethodInNestedImplementations(IFace, SetterSel);
    }
    // Look through local category implementations associated with the class.
    if (!Setter)
      Setter = IFace->getCategoryInstanceMethod(SetterSel);

    if (Setter && DiagnoseUseOfDecl(Setter, MemberLoc))
      return ExprError();

    if (Getter || Setter) {
      QualType PType;

      if (Getter)
        PType = Getter->getResultType();
      else
        // Get the expression type from Setter's incoming parameter.
        PType = (*(Setter->param_end() -1))->getType();
      // FIXME: we must check that the setter has property type.
      return Owned(new (Context) ObjCImplicitSetterGetterRefExpr(Getter, PType,
                                      Setter, MemberLoc, BaseExpr));
    }
    return ExprError(Diag(MemberLoc, diag::err_property_not_found)
      << &Member << BaseType);
  }
  
  // Handle the following exceptional case (*Obj).isa.
  if (OpKind == tok::period && 
      BaseType->isSpecificBuiltinType(BuiltinType::ObjCId) &&
      Member.isStr("isa"))
    return Owned(new (Context) ObjCIsaExpr(BaseExpr, false, MemberLoc,
                                           Context.getObjCIdType()));

  // Handle 'field access' to vectors, such as 'V.xx'.
  if (BaseType->isExtVectorType()) {
    QualType ret = CheckExtVectorComponent(BaseType, OpLoc, Member, MemberLoc);
    if (ret.isNull())
      return ExprError();
    return Owned(new (Context) ExtVectorElementExpr(ret, BaseExpr, Member,
                                                    MemberLoc));
  }

  Diag(MemberLoc, diag::err_typecheck_member_reference_struct_union)
    << BaseType << BaseExpr->getSourceRange();

  // If the user is trying to apply -> or . to a function or function
  // pointer, it's probably because they forgot parentheses to call
  // the function. Suggest the addition of those parentheses.
  if (BaseType == Context.OverloadTy || 
      BaseType->isFunctionType() ||
      (BaseType->isPointerType() && 
       BaseType->getAs<PointerType>()->isFunctionType())) {
    SourceLocation Loc = PP.getLocForEndOfToken(BaseExpr->getLocEnd());
    Diag(Loc, diag::note_member_reference_needs_call)
      << CodeModificationHint::CreateInsertion(Loc, "()");
  }

  return ExprError();
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
  unsigned NumArgsToCheck = NumArgs;
  bool Invalid = false;

  // If too few arguments are available (and we don't have default
  // arguments for the remaining parameters), don't make the call.
  if (NumArgs < NumArgsInProto) {
    if (!FDecl || NumArgs < FDecl->getMinRequiredArguments())
      return Diag(RParenLoc, diag::err_typecheck_call_too_few_args)
        << Fn->getType()->isBlockPointerType() << Fn->getSourceRange();
    // Use default arguments for missing arguments
    NumArgsToCheck = NumArgsInProto;
    Call->setNumArgs(Context, NumArgsInProto);
  }

  // If too many are passed and not variadic, error on the extras and drop
  // them.
  if (NumArgs > NumArgsInProto) {
    if (!Proto->isVariadic()) {
      Diag(Args[NumArgsInProto]->getLocStart(),
           diag::err_typecheck_call_too_many_args)
        << Fn->getType()->isBlockPointerType() << Fn->getSourceRange()
        << SourceRange(Args[NumArgsInProto]->getLocStart(),
                       Args[NumArgs-1]->getLocEnd());
      // This deletes the extra arguments.
      Call->setNumArgs(Context, NumArgsInProto);
      Invalid = true;
    }
    NumArgsToCheck = NumArgsInProto;
  }

  // Continue to check argument types (even if we have too few/many args).
  for (unsigned i = 0; i != NumArgsToCheck; i++) {
    QualType ProtoArgType = Proto->getArgType(i);

    Expr *Arg;
    if (i < NumArgs) {
      Arg = Args[i];

      if (RequireCompleteType(Arg->getSourceRange().getBegin(),
                              ProtoArgType,
                              diag::err_call_incomplete_argument,
                              Arg->getSourceRange()))
        return true;

      // Pass the argument.
      if (PerformCopyInitialization(Arg, ProtoArgType, "passing"))
        return true;
    } else {
      if (FDecl->getParamDecl(i)->hasUnparsedDefaultArg()) {
        Diag (Call->getSourceRange().getBegin(),
              diag::err_use_of_default_argument_to_function_declared_later) <<
        FDecl << cast<CXXRecordDecl>(FDecl->getDeclContext())->getDeclName();
        Diag(UnparsedDefaultArgLocs[FDecl->getParamDecl(i)], 
              diag::note_default_argument_declared_here);
      } else {
        Expr *DefaultExpr = FDecl->getParamDecl(i)->getDefaultArg();
        
        // If the default expression creates temporaries, we need to
        // push them to the current stack of expression temporaries so they'll
        // be properly destroyed.
        if (CXXExprWithTemporaries *E 
              = dyn_cast_or_null<CXXExprWithTemporaries>(DefaultExpr)) {
          assert(!E->shouldDestroyTemporaries() && 
                 "Can't destroy temporaries in a default argument expr!");
          for (unsigned I = 0, N = E->getNumTemporaries(); I != N; ++I)
            ExprTemporaries.push_back(E->getTemporary(I));
        }
      }
  
      // We already type-checked the argument, so we know it works.
      Arg = CXXDefaultArgExpr::Create(Context, FDecl->getParamDecl(i));
    }
    
    QualType ArgType = Arg->getType();

    Call->setArg(i, Arg);
  }

  // If this is a variadic call, handle args passed through "...".
  if (Proto->isVariadic()) {
    VariadicCallType CallType = VariadicFunction;
    if (Fn->getType()->isBlockPointerType())
      CallType = VariadicBlock; // Block
    else if (isa<MemberExpr>(Fn))
      CallType = VariadicMethod;

    // Promote the arguments (C99 6.5.2.2p7).
    for (unsigned i = NumArgsInProto; i != NumArgs; i++) {
      Expr *Arg = Args[i];
      Invalid |= DefaultVariadicArgumentPromotion(Arg, CallType);
      Call->setArg(i, Arg);
    }
  }

  return Invalid;
}

/// ActOnCallExpr - Handle a call to Fn with the specified array of arguments.
/// This provides the location of the left/right parens and a list of comma
/// locations.
Action::OwningExprResult
Sema::ActOnCallExpr(Scope *S, ExprArg fn, SourceLocation LParenLoc,
                    MultiExprArg args,
                    SourceLocation *CommaLocs, SourceLocation RParenLoc) {
  unsigned NumArgs = args.size();

  // Since this might be a postfix expression, get rid of ParenListExprs.
  fn = MaybeConvertParenListExprToParenExpr(S, move(fn));
  
  Expr *Fn = fn.takeAs<Expr>();
  Expr **Args = reinterpret_cast<Expr**>(args.release());
  assert(Fn && "no function call expression");
  FunctionDecl *FDecl = NULL;
  NamedDecl *NDecl = NULL;
  DeclarationName UnqualifiedName;
  
  if (getLangOptions().CPlusPlus) {
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

    // Determine whether this is a call to a member function.
    if (MemberExpr *MemExpr = dyn_cast<MemberExpr>(Fn->IgnoreParens())) {
      NamedDecl *MemDecl = MemExpr->getMemberDecl();
      if (isa<OverloadedFunctionDecl>(MemDecl) ||
          isa<CXXMethodDecl>(MemDecl) ||
          (isa<FunctionTemplateDecl>(MemDecl) &&
           isa<CXXMethodDecl>(
                cast<FunctionTemplateDecl>(MemDecl)->getTemplatedDecl())))
        return Owned(BuildCallToMemberFunction(S, Fn, LParenLoc, Args, NumArgs,
                                               CommaLocs, RParenLoc));
    }
  }

  // If we're directly calling a function, get the appropriate declaration.
  // Also, in C++, keep track of whether we should perform argument-dependent 
  // lookup and whether there were any explicitly-specified template arguments.
  Expr *FnExpr = Fn;
  bool ADL = true;
  bool HasExplicitTemplateArgs = 0;
  const TemplateArgument *ExplicitTemplateArgs = 0;
  unsigned NumExplicitTemplateArgs = 0;
  while (true) {
    if (ImplicitCastExpr *IcExpr = dyn_cast<ImplicitCastExpr>(FnExpr))
      FnExpr = IcExpr->getSubExpr();
    else if (ParenExpr *PExpr = dyn_cast<ParenExpr>(FnExpr)) {
      // Parentheses around a function disable ADL
      // (C++0x [basic.lookup.argdep]p1).
      ADL = false;
      FnExpr = PExpr->getSubExpr();
    } else if (isa<UnaryOperator>(FnExpr) &&
               cast<UnaryOperator>(FnExpr)->getOpcode()
                 == UnaryOperator::AddrOf) {
      FnExpr = cast<UnaryOperator>(FnExpr)->getSubExpr();
    } else if (DeclRefExpr *DRExpr = dyn_cast<DeclRefExpr>(FnExpr)) {
      // Qualified names disable ADL (C++0x [basic.lookup.argdep]p1).
      ADL &= !isa<QualifiedDeclRefExpr>(DRExpr);
      NDecl = dyn_cast<NamedDecl>(DRExpr->getDecl());
      break;
    } else if (UnresolvedFunctionNameExpr *DepName
                 = dyn_cast<UnresolvedFunctionNameExpr>(FnExpr)) {
      UnqualifiedName = DepName->getName();
      break;
    } else if (TemplateIdRefExpr *TemplateIdRef 
                 = dyn_cast<TemplateIdRefExpr>(FnExpr)) {
      NDecl = TemplateIdRef->getTemplateName().getAsTemplateDecl();
      if (!NDecl)
        NDecl = TemplateIdRef->getTemplateName().getAsOverloadedFunctionDecl();
      HasExplicitTemplateArgs = true;
      ExplicitTemplateArgs = TemplateIdRef->getTemplateArgs();
      NumExplicitTemplateArgs = TemplateIdRef->getNumTemplateArgs();
      
      // C++ [temp.arg.explicit]p6:
      //   [Note: For simple function names, argument dependent lookup (3.4.2)
      //   applies even when the function name is not visible within the 
      //   scope of the call. This is because the call still has the syntactic
      //   form of a function call (3.4.1). But when a function template with
      //   explicit template arguments is used, the call does not have the
      //   correct syntactic form unless there is a function template with 
      //   that name visible at the point of the call. If no such name is 
      //   visible, the call is not syntactically well-formed and 
      //   argument-dependent lookup does not apply. If some such name is 
      //   visible, argument dependent lookup applies and additional function
      //   templates may be found in other namespaces.
      //
      // The summary of this paragraph is that, if we get to this point and the
      // template-id was not a qualified name, then argument-dependent lookup
      // is still possible.
      if (TemplateIdRef->getQualifier())
        ADL = false;
      break;
    } else {
      // Any kind of name that does not refer to a declaration (or
      // set of declarations) disables ADL (C++0x [basic.lookup.argdep]p3).
      ADL = false;
      break;
    }
  }

  OverloadedFunctionDecl *Ovl = 0;
  FunctionTemplateDecl *FunctionTemplate = 0;
  if (NDecl) {
    FDecl = dyn_cast<FunctionDecl>(NDecl);
    if ((FunctionTemplate = dyn_cast<FunctionTemplateDecl>(NDecl)))
      FDecl = FunctionTemplate->getTemplatedDecl();
    else
      FDecl = dyn_cast<FunctionDecl>(NDecl);
    Ovl = dyn_cast<OverloadedFunctionDecl>(NDecl);
  }

  if (Ovl || FunctionTemplate || 
      (getLangOptions().CPlusPlus && (FDecl || UnqualifiedName))) {
    // We don't perform ADL for implicit declarations of builtins.
    if (FDecl && FDecl->getBuiltinID(Context) && FDecl->isImplicit())
      ADL = false;

    // We don't perform ADL in C.
    if (!getLangOptions().CPlusPlus)
      ADL = false;

    if (Ovl || FunctionTemplate || ADL) {
      FDecl = ResolveOverloadedCallFn(Fn, NDecl, UnqualifiedName, 
                                      HasExplicitTemplateArgs,
                                      ExplicitTemplateArgs,
                                      NumExplicitTemplateArgs,
                                      LParenLoc, Args, NumArgs, CommaLocs, 
                                      RParenLoc, ADL);
      if (!FDecl)
        return ExprError();

      // Update Fn to refer to the actual function selected.
      Expr *NewFn = 0;
      if (QualifiedDeclRefExpr *QDRExpr
            = dyn_cast<QualifiedDeclRefExpr>(FnExpr))
        NewFn = new (Context) QualifiedDeclRefExpr(FDecl, FDecl->getType(),
                                                   QDRExpr->getLocation(),
                                                   false, false,
                                                 QDRExpr->getQualifierRange(),
                                                   QDRExpr->getQualifier());
      else
        NewFn = new (Context) DeclRefExpr(FDecl, FDecl->getType(),
                                          Fn->getSourceRange().getBegin());
      Fn->Destroy(Context);
      Fn = NewFn;
    }
  }

  // Promote the function operand.
  UsualUnaryConversions(Fn);

  // Make the call expr early, before semantic checks.  This guarantees cleanup
  // of arguments and function on error.
  ExprOwningPtr<CallExpr> TheCall(this, new (Context) CallExpr(Context, Fn,
                                                               Args, NumArgs,
                                                               Context.BoolTy,
                                                               RParenLoc));

  const FunctionType *FuncT;
  if (!Fn->getType()->isBlockPointerType()) {
    // C99 6.5.2.2p1 - "The expression that denotes the called function shall
    // have type pointer to function".
    const PointerType *PT = Fn->getType()->getAs<PointerType>();
    if (PT == 0)
      return ExprError(Diag(LParenLoc, diag::err_typecheck_call_not_function)
        << Fn->getType() << Fn->getSourceRange());
    FuncT = PT->getPointeeType()->getAsFunctionType();
  } else { // This is a block call.
    FuncT = Fn->getType()->getAs<BlockPointerType>()->getPointeeType()->
                getAsFunctionType();
  }
  if (FuncT == 0)
    return ExprError(Diag(LParenLoc, diag::err_typecheck_call_not_function)
      << Fn->getType() << Fn->getSourceRange());

  // Check for a valid return type
  if (!FuncT->getResultType()->isVoidType() &&
      RequireCompleteType(Fn->getSourceRange().getBegin(),
                          FuncT->getResultType(),
                          diag::err_call_incomplete_return,
                          TheCall->getSourceRange()))
    return ExprError();

  // We know the result type of the call, set it.
  TheCall->setType(FuncT->getResultType().getNonReferenceType());

  if (const FunctionProtoType *Proto = dyn_cast<FunctionProtoType>(FuncT)) {
    if (ConvertArgumentsForCall(&*TheCall, Fn, FDecl, Proto, Args, NumArgs,
                                RParenLoc))
      return ExprError();
  } else {
    assert(isa<FunctionNoProtoType>(FuncT) && "Unknown FunctionType!");

    if (FDecl) {
      // Check if we have too few/too many template arguments, based
      // on our knowledge of the function definition.
      const FunctionDecl *Def = 0;
      if (FDecl->getBody(Def) && NumArgs != Def->param_size()) {
        const FunctionProtoType *Proto =
            Def->getType()->getAsFunctionProtoType();
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
                              diag::err_call_incomplete_argument,
                              Arg->getSourceRange()))
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
    if (CheckFunctionCall(FDecl, TheCall.get()))
      return ExprError();
    
    if (unsigned BuiltinID = FDecl->getBuiltinID(Context)) 
      return CheckBuiltinFunctionCall(BuiltinID, TheCall.take());
  } else if (NDecl) {
    if (CheckBlockCall(NDecl, TheCall.get()))
      return ExprError();
  }

  return MaybeBindToTemporary(TheCall.take());
}

Action::OwningExprResult
Sema::ActOnCompoundLiteral(SourceLocation LParenLoc, TypeTy *Ty,
                           SourceLocation RParenLoc, ExprArg InitExpr) {
  assert((Ty != 0) && "ActOnCompoundLiteral(): missing type");
  //FIXME: Preserve type source info.
  QualType literalType = GetTypeFromParser(Ty);
  // FIXME: put back this assert when initializers are worked out.
  //assert((InitExpr != 0) && "ActOnCompoundLiteral(): missing expression");
  Expr *literalExpr = static_cast<Expr*>(InitExpr.get());

  if (literalType->isArrayType()) {
    if (literalType->isVariableArrayType())
      return ExprError(Diag(LParenLoc, diag::err_variable_object_no_init)
        << SourceRange(LParenLoc, literalExpr->getSourceRange().getEnd()));
  } else if (!literalType->isDependentType() &&
             RequireCompleteType(LParenLoc, literalType,
                                 diag::err_typecheck_decl_incomplete_type,
                SourceRange(LParenLoc, literalExpr->getSourceRange().getEnd())))
    return ExprError();

  if (CheckInitializerTypes(literalExpr, literalType, LParenLoc,
                            DeclarationName(), /*FIXME:DirectInit=*/false))
    return ExprError();

  bool isFileScope = getCurFunctionOrMethodDecl() == 0;
  if (isFileScope) { // 6.5.2.5p3
    if (CheckForConstantInitializer(literalExpr, literalType))
      return ExprError();
  }
  InitExpr.release();
  return Owned(new (Context) CompoundLiteralExpr(LParenLoc, literalType,
                                                 literalExpr, isFileScope));
}

Action::OwningExprResult
Sema::ActOnInitList(SourceLocation LBraceLoc, MultiExprArg initlist,
                    SourceLocation RBraceLoc) {
  unsigned NumInit = initlist.size();
  Expr **InitList = reinterpret_cast<Expr**>(initlist.release());

  // Semantic analysis for initializers is done by ActOnDeclarator() and
  // CheckInitializer() - it requires knowledge of the object being intialized.

  InitListExpr *E = new (Context) InitListExpr(LBraceLoc, InitList, NumInit,
                                               RBraceLoc);
  E->setType(Context.VoidTy); // FIXME: just a place holder for now.
  return Owned(E);
}

/// CheckCastTypes - Check type constraints for casting between types.
bool Sema::CheckCastTypes(SourceRange TyR, QualType castType, Expr *&castExpr,
                          CastExpr::CastKind& Kind, bool FunctionalStyle) {
  if (getLangOptions().CPlusPlus)
    return CXXCheckCStyleCast(TyR, castType, castExpr, Kind, FunctionalStyle);

  DefaultFunctionArrayConversion(castExpr);

  // C99 6.5.4p2: the cast type needs to be void or scalar and the expression
  // type needs to be scalar.
  if (castType->isVoidType()) {
    // Cast to void allows any expr type.
  } else if (!castType->isScalarType() && !castType->isVectorType()) {
    if (Context.getCanonicalType(castType).getUnqualifiedType() ==
        Context.getCanonicalType(castExpr->getType().getUnqualifiedType()) &&
        (castType->isStructureType() || castType->isUnionType())) {
      // GCC struct/union extension: allow cast to self.
      // FIXME: Check that the cast destination type is complete.
      Diag(TyR.getBegin(), diag::ext_typecheck_cast_nonscalar)
        << castType << castExpr->getSourceRange();
      Kind = CastExpr::CK_NoOp;
    } else if (castType->isUnionType()) {
      // GCC cast to union extension
      RecordDecl *RD = castType->getAs<RecordType>()->getDecl();
      RecordDecl::field_iterator Field, FieldEnd;
      for (Field = RD->field_begin(), FieldEnd = RD->field_end();
           Field != FieldEnd; ++Field) {
        if (Context.getCanonicalType(Field->getType()).getUnqualifiedType() ==
            Context.getCanonicalType(castExpr->getType()).getUnqualifiedType()) {
          Diag(TyR.getBegin(), diag::ext_typecheck_cast_to_union)
            << castExpr->getSourceRange();
          break;
        }
      }
      if (Field == FieldEnd)
        return Diag(TyR.getBegin(), diag::err_typecheck_cast_to_union_no_type)
          << castExpr->getType() << castExpr->getSourceRange();
      Kind = CastExpr::CK_ToUnion;
    } else {
      // Reject any other conversions to non-scalar types.
      return Diag(TyR.getBegin(), diag::err_typecheck_cond_expect_scalar)
        << castType << castExpr->getSourceRange();
    }
  } else if (!castExpr->getType()->isScalarType() &&
             !castExpr->getType()->isVectorType()) {
    return Diag(castExpr->getLocStart(),
                diag::err_typecheck_expect_scalar_operand)
      << castExpr->getType() << castExpr->getSourceRange();
  } else if (castType->isExtVectorType()) {
    if (CheckExtVectorCast(TyR, castType, castExpr->getType()))
      return true;
  } else if (castType->isVectorType()) {
    if (CheckVectorCast(TyR, castType, castExpr->getType()))
      return true;
  } else if (castExpr->getType()->isVectorType()) {
    if (CheckVectorCast(TyR, castExpr->getType(), castType))
      return true;
  } else if (getLangOptions().ObjC1 && isa<ObjCSuperExpr>(castExpr)) {
    return Diag(castExpr->getLocStart(), diag::err_illegal_super_cast) << TyR;
  } else if (!castType->isArithmeticType()) {
    QualType castExprType = castExpr->getType();
    if (!castExprType->isIntegralType() && castExprType->isArithmeticType())
      return Diag(castExpr->getLocStart(),
                  diag::err_cast_pointer_from_non_pointer_int)
        << castExprType << castExpr->getSourceRange();
  } else if (!castExpr->getType()->isArithmeticType()) {
    if (!castType->isIntegralType() && castType->isArithmeticType())
      return Diag(castExpr->getLocStart(),
                  diag::err_cast_pointer_to_non_pointer_int)
        << castType << castExpr->getSourceRange();
  }
  if (isa<ObjCSelectorExpr>(castExpr))
    return Diag(castExpr->getLocStart(), diag::err_cast_selector_expr);
  return false;
}

bool Sema::CheckVectorCast(SourceRange R, QualType VectorTy, QualType Ty) {
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

  return false;
}

bool Sema::CheckExtVectorCast(SourceRange R, QualType DestTy, QualType SrcTy) {
  assert(DestTy->isExtVectorType() && "Not an extended vector type!");
  
  // If SrcTy is a VectorType, the total size must match to explicitly cast to
  // an ExtVectorType.
  if (SrcTy->isVectorType()) {
    if (Context.getTypeSize(DestTy) != Context.getTypeSize(SrcTy))
      return Diag(R.getBegin(),diag::err_invalid_conversion_between_ext_vectors)
        << DestTy << SrcTy << R;
    return false;
  }

  // All non-pointer scalars can be cast to ExtVector type.  The appropriate
  // conversion will take place first from scalar to elt type, and then
  // splat from elt type to vector.
  if (SrcTy->isPointerType())
    return Diag(R.getBegin(),
                diag::err_invalid_conversion_between_vector_and_scalar)
      << DestTy << SrcTy << R;
  return false;
}

Action::OwningExprResult
Sema::ActOnCastExpr(Scope *S, SourceLocation LParenLoc, TypeTy *Ty,
                    SourceLocation RParenLoc, ExprArg Op) {
  CastExpr::CastKind Kind = CastExpr::CK_Unknown;
  
  assert((Ty != 0) && (Op.get() != 0) &&
         "ActOnCastExpr(): missing type or expr");

  Expr *castExpr = (Expr *)Op.get();
  //FIXME: Preserve type source info.
  QualType castType = GetTypeFromParser(Ty);
  
  // If the Expr being casted is a ParenListExpr, handle it specially.
  if (isa<ParenListExpr>(castExpr))
    return ActOnCastOfParenListExpr(S, LParenLoc, RParenLoc, move(Op),castType);

  if (CheckCastTypes(SourceRange(LParenLoc, RParenLoc), castType, castExpr, 
                     Kind))
    return ExprError();
  
  Op.release();
  return Owned(new (Context) CStyleCastExpr(castType.getNonReferenceType(),
                                            Kind, castExpr, castType, 
                                            LParenLoc, RParenLoc));
}

/// This is not an AltiVec-style cast, so turn the ParenListExpr into a sequence
/// of comma binary operators.
Action::OwningExprResult
Sema::MaybeConvertParenListExprToParenExpr(Scope *S, ExprArg EA) {
  Expr *expr = EA.takeAs<Expr>();
  ParenListExpr *E = dyn_cast<ParenListExpr>(expr);
  if (!E)
    return Owned(expr);
  
  OwningExprResult Result(*this, E->getExpr(0));
  
  for (unsigned i = 1, e = E->getNumExprs(); i != e && !Result.isInvalid(); ++i)
    Result = ActOnBinOp(S, E->getExprLoc(), tok::comma, move(Result),
                        Owned(E->getExpr(i)));
  
  return ActOnParenExpr(E->getLParenLoc(), E->getRParenLoc(), move(Result));
}

Action::OwningExprResult
Sema::ActOnCastOfParenListExpr(Scope *S, SourceLocation LParenLoc,
                               SourceLocation RParenLoc, ExprArg Op,
                               QualType Ty) {
  ParenListExpr *PE = (ParenListExpr *)Op.get();
  
  // If this is an altivec initializer, '(' type ')' '(' init, ..., init ')' 
  // then handle it as such.
  if (getLangOptions().AltiVec && Ty->isVectorType()) {
    if (PE->getNumExprs() == 0) {
      Diag(PE->getExprLoc(), diag::err_altivec_empty_initializer);
      return ExprError();
    }

    llvm::SmallVector<Expr *, 8> initExprs;
    for (unsigned i = 0, e = PE->getNumExprs(); i != e; ++i)
      initExprs.push_back(PE->getExpr(i));

    // FIXME: This means that pretty-printing the final AST will produce curly
    // braces instead of the original commas.
    Op.release();
    InitListExpr *E = new (Context) InitListExpr(LParenLoc, &initExprs[0], 
                                                 initExprs.size(), RParenLoc);
    E->setType(Ty);
    return ActOnCompoundLiteral(LParenLoc, Ty.getAsOpaquePtr(), RParenLoc, 
                                Owned(E));
  } else {
    // This is not an AltiVec-style cast, so turn the ParenListExpr into a 
    // sequence of BinOp comma operators.
    Op = MaybeConvertParenListExprToParenExpr(S, move(Op));
    return ActOnCastExpr(S, LParenLoc, Ty.getAsOpaquePtr(), RParenLoc,move(Op));
  }
}

Action::OwningExprResult Sema::ActOnParenListExpr(SourceLocation L,
                                                  SourceLocation R,
                                                  MultiExprArg Val) {
  unsigned nexprs = Val.size();
  Expr **exprs = reinterpret_cast<Expr**>(Val.release());
  assert((exprs != 0) && "ActOnParenListExpr() missing expr list");
  Expr *expr = new (Context) ParenListExpr(Context, L, exprs, nexprs, R);
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
    ImpCastExprToType(LHS, Context.VoidTy);
    ImpCastExprToType(RHS, Context.VoidTy);
    return Context.VoidTy;
  }
  // C99 6.5.15p6 - "if one operand is a null pointer constant, the result has
  // the type of the other operand."
  if ((LHSTy->isAnyPointerType() || LHSTy->isBlockPointerType()) &&
      RHS->isNullPointerConstant(Context)) {
    ImpCastExprToType(RHS, LHSTy); // promote the null to a pointer.
    return LHSTy;
  }
  if ((RHSTy->isAnyPointerType() || RHSTy->isBlockPointerType()) &&
      LHS->isNullPointerConstant(Context)) {
    ImpCastExprToType(LHS, RHSTy); // promote the null to a pointer.
    return RHSTy;
  }
  // Handle things like Class and struct objc_class*.  Here we case the result
  // to the pseudo-builtin, because that will be implicitly cast back to the
  // redefinition type if an attempt is made to access its fields.
  if (LHSTy->isObjCClassType() &&
      (RHSTy.getDesugaredType() == Context.ObjCClassRedefinitionType)) {
    ImpCastExprToType(RHS, LHSTy);
    return LHSTy;
  }
  if (RHSTy->isObjCClassType() &&
      (LHSTy.getDesugaredType() == Context.ObjCClassRedefinitionType)) {
    ImpCastExprToType(LHS, RHSTy);
    return RHSTy;
  }
  // And the same for struct objc_object* / id
  if (LHSTy->isObjCIdType() &&
      (RHSTy.getDesugaredType() == Context.ObjCIdRedefinitionType)) {
    ImpCastExprToType(RHS, LHSTy);
    return LHSTy;
  }
  if (RHSTy->isObjCIdType() &&
      (LHSTy.getDesugaredType() == Context.ObjCIdRedefinitionType)) {
    ImpCastExprToType(LHS, RHSTy);
    return RHSTy;
  }
  // Handle block pointer types.
  if (LHSTy->isBlockPointerType() || RHSTy->isBlockPointerType()) {
    if (!LHSTy->isBlockPointerType() || !RHSTy->isBlockPointerType()) {
      if (LHSTy->isVoidPointerType() || RHSTy->isVoidPointerType()) {
        QualType destType = Context.getPointerType(Context.VoidTy);
        ImpCastExprToType(LHS, destType); 
        ImpCastExprToType(RHS, destType);
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
      ImpCastExprToType(LHS, incompatTy);
      ImpCastExprToType(RHS, incompatTy);
      return incompatTy;
    }
    // The block pointer types are compatible.
    ImpCastExprToType(LHS, LHSTy);
    ImpCastExprToType(RHS, LHSTy);
    return LHSTy;
  }
  // Check constraints for Objective-C object pointers types.
  if (LHSTy->isObjCObjectPointerType() && RHSTy->isObjCObjectPointerType()) {
    
    if (Context.getCanonicalType(LHSTy) == Context.getCanonicalType(RHSTy)) {
      // Two identical object pointer types are always compatible.
      return LHSTy;
    }
    const ObjCObjectPointerType *LHSOPT = LHSTy->getAsObjCObjectPointerType();
    const ObjCObjectPointerType *RHSOPT = RHSTy->getAsObjCObjectPointerType();
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
    } else {
      Diag(QuestionLoc, diag::ext_typecheck_cond_incompatible_operands)
        << LHSTy << RHSTy
        << LHS->getSourceRange() << RHS->getSourceRange();
      QualType incompatTy = Context.getObjCIdType();
      ImpCastExprToType(LHS, incompatTy);
      ImpCastExprToType(RHS, incompatTy);
      return incompatTy;
    }
    // The object pointer types are compatible.
    ImpCastExprToType(LHS, compositeType);
    ImpCastExprToType(RHS, compositeType);
    return compositeType;
  }
  // Check Objective-C object pointer types and 'void *'
  if (LHSTy->isVoidPointerType() && RHSTy->isObjCObjectPointerType()) {
    QualType lhptee = LHSTy->getAs<PointerType>()->getPointeeType();
    QualType rhptee = RHSTy->getAsObjCObjectPointerType()->getPointeeType();
    QualType destPointee = lhptee.getQualifiedType(rhptee.getCVRQualifiers());
    QualType destType = Context.getPointerType(destPointee);
    ImpCastExprToType(LHS, destType); // add qualifiers if necessary
    ImpCastExprToType(RHS, destType); // promote to void*
    return destType;
  }
  if (LHSTy->isObjCObjectPointerType() && RHSTy->isVoidPointerType()) {
    QualType lhptee = LHSTy->getAsObjCObjectPointerType()->getPointeeType();
    QualType rhptee = RHSTy->getAs<PointerType>()->getPointeeType();
    QualType destPointee = rhptee.getQualifiedType(lhptee.getCVRQualifiers());
    QualType destType = Context.getPointerType(destPointee);
    ImpCastExprToType(RHS, destType); // add qualifiers if necessary
    ImpCastExprToType(LHS, destType); // promote to void*
    return destType;
  }
  // Check constraints for C object pointers types (C99 6.5.15p3,6).
  if (LHSTy->isPointerType() && RHSTy->isPointerType()) {
    // get the "pointed to" types
    QualType lhptee = LHSTy->getAs<PointerType>()->getPointeeType();
    QualType rhptee = RHSTy->getAs<PointerType>()->getPointeeType();

    // ignore qualifiers on void (C99 6.5.15p3, clause 6)
    if (lhptee->isVoidType() && rhptee->isIncompleteOrObjectType()) {
      // Figure out necessary qualifiers (C99 6.5.15p6)
      QualType destPointee=lhptee.getQualifiedType(rhptee.getCVRQualifiers());
      QualType destType = Context.getPointerType(destPointee);
      ImpCastExprToType(LHS, destType); // add qualifiers if necessary
      ImpCastExprToType(RHS, destType); // promote to void*
      return destType;
    }
    if (rhptee->isVoidType() && lhptee->isIncompleteOrObjectType()) {
      QualType destPointee=rhptee.getQualifiedType(lhptee.getCVRQualifiers());
      QualType destType = Context.getPointerType(destPointee);
      ImpCastExprToType(LHS, destType); // add qualifiers if necessary
      ImpCastExprToType(RHS, destType); // promote to void*
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
      ImpCastExprToType(LHS, incompatTy);
      ImpCastExprToType(RHS, incompatTy);
      return incompatTy;
    }
    // The pointer types are compatible.
    // C99 6.5.15p6: If both operands are pointers to compatible types *or* to
    // differently qualified versions of compatible types, the result type is
    // a pointer to an appropriately qualified version of the *composite*
    // type.
    // FIXME: Need to calculate the composite type.
    // FIXME: Need to add qualifiers
    ImpCastExprToType(LHS, LHSTy);
    ImpCastExprToType(RHS, LHSTy);
    return LHSTy;
  }
  
  // GCC compatibility: soften pointer/integer mismatch.
  if (RHSTy->isPointerType() && LHSTy->isIntegerType()) {
    Diag(QuestionLoc, diag::warn_typecheck_cond_pointer_integer_mismatch)
      << LHSTy << RHSTy << LHS->getSourceRange() << RHS->getSourceRange();
    ImpCastExprToType(LHS, RHSTy); // promote the integer to a pointer.
    return RHSTy;
  }
  if (LHSTy->isPointerType() && RHSTy->isIntegerType()) {
    Diag(QuestionLoc, diag::warn_typecheck_cond_pointer_integer_mismatch)
      << LHSTy << RHSTy << LHS->getSourceRange() << RHS->getSourceRange();
    ImpCastExprToType(RHS, LHSTy); // promote the integer to a pointer.
    return LHSTy;
  }

  // Otherwise, the operands are not compatible.
  Diag(QuestionLoc, diag::err_typecheck_cond_incompatible_operands)
    << LHSTy << RHSTy << LHS->getSourceRange() << RHS->getSourceRange();
  return QualType();
}

/// ActOnConditionalOp - Parse a ?: operation.  Note that 'LHS' may be null
/// in the case of a the GNU conditional expr extension.
Action::OwningExprResult Sema::ActOnConditionalOp(SourceLocation QuestionLoc,
                                                  SourceLocation ColonLoc,
                                                  ExprArg Cond, ExprArg LHS,
                                                  ExprArg RHS) {
  Expr *CondExpr = (Expr *) Cond.get();
  Expr *LHSExpr = (Expr *) LHS.get(), *RHSExpr = (Expr *) RHS.get();

  // If this is the gnu "x ?: y" extension, analyze the types as though the LHS
  // was the condition.
  bool isLHSNull = LHSExpr == 0;
  if (isLHSNull)
    LHSExpr = CondExpr;

  QualType result = CheckConditionalOperands(CondExpr, LHSExpr,
                                             RHSExpr, QuestionLoc);
  if (result.isNull())
    return ExprError();

  Cond.release();
  LHS.release();
  RHS.release();
  return Owned(new (Context) ConditionalOperator(CondExpr,
                                                 isLHSNull ? 0 : LHSExpr,
                                                 RHSExpr, result));
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
    if (lhptee->isCharType()) {
      lhptee = Context.UnsignedCharTy;
    } else if (lhptee->isSignedIntegerType()) {
      lhptee = Context.getCorrespondingUnsignedType(lhptee);
    }
    if (rhptee->isCharType()) {
      rhptee = Context.UnsignedCharTy;
    } else if (rhptee->isSignedIntegerType()) {
      rhptee = Context.getCorrespondingUnsignedType(rhptee);
    }
    if (lhptee == rhptee) {
      // Types are compatible ignoring the sign. Qualifier incompatibility
      // takes priority over sign incompatibility because the sign
      // warning can be disabled.
      if (ConvTy != Compatible)
        return ConvTy;
      return IncompatiblePointerSign;
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
  if (lhptee.getCVRQualifiers() != rhptee.getCVRQualifiers())
    ConvTy = CompatiblePointerDiscardsQualifiers;

  if (!Context.typesAreCompatible(lhptee, rhptee))
    return IncompatibleBlockPointer;
  return ConvTy;
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
    if (!rhsType->isVectorType() && rhsType->isArithmeticType())
      return Compatible;
  }
  
  if (lhsType->isVectorType() || rhsType->isVectorType()) {
    // If we are allowing lax vector conversions, and LHS and RHS are both
    // vectors, the total size only needs to be the same. This is a bitcast;
    // no bits are changed but the result type is different.
    if (getLangOptions().LaxVectorConversions &&
        lhsType->isVectorType() && rhsType->isVectorType()) {
      if (Context.getTypeSize(lhsType) == Context.getTypeSize(rhsType))
        return IncompatibleVectors;
    }
    return Incompatible;
  }

  if (lhsType->isArithmeticType() && rhsType->isArithmeticType())
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
      if (lhsType->isObjCBuiltinType() || rhsType->isObjCBuiltinType())
        return Compatible;
      if (Context.typesAreCompatible(lhsType, rhsType))
        return Compatible;
      if (lhsType->isObjCQualifiedIdType() || rhsType->isObjCQualifiedIdType())
        return IncompatibleObjCQualifiedId;
      return IncompatiblePointer;
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
  InitListExpr *Initializer = new (C) InitListExpr(SourceLocation(),
                                                   &E, 1,
                                                   SourceLocation());
  Initializer->setType(UnionType);
  Initializer->setInitializedFieldInUnion(Field);

  // Build a compound literal constructing a value of the transparent
  // union type from this initializer list.
  E = new (C) CompoundLiteralExpr(SourceLocation(), UnionType, Initializer,
                                  false);
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
          ImpCastExprToType(rExpr, it->getType());
          InitField = *it;
          break;
        }
      
      if (rExpr->isNullPointerConstant(Context)) {
        ImpCastExprToType(rExpr, it->getType());
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
                                    "assigning"))
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
      && rExpr->isNullPointerConstant(Context)) {
    ImpCastExprToType(rExpr, lhsType);
    return Compatible;
  }

  // This check seems unnatural, however it is necessary to ensure the proper
  // conversion of functions/arrays. If the conversion were done for all
  // DeclExpr's (created by ActOnIdentifierExpr), it would mess up the unary
  // expressions that surpress this implicit conversion (&, sizeof).
  //
  // Suppress this for references: C++ 8.5.3p5.
  if (!lhsType->isReferenceType())
    DefaultFunctionArrayConversion(rExpr);

  Sema::AssignConvertType result =
    CheckAssignmentConstraints(lhsType, rExpr->getType());

  // C99 6.5.16.1p2: The value of the right operand is converted to the
  // type of the assignment expression.
  // CheckAssignmentConstraints allows the left-hand side to be a reference,
  // so that we can use references in built-in functions even in C.
  // The getNonReferenceType() call makes sure that the resulting expression
  // does not have reference type.
  if (result != Incompatible && rExpr->getType() != lhsType)
    ImpCastExprToType(rExpr, lhsType.getNonReferenceType());
  return result;
}

QualType Sema::InvalidOperands(SourceLocation Loc, Expr *&lex, Expr *&rex) {
  Diag(Loc, diag::err_typecheck_invalid_operands)
    << lex->getType() << rex->getType()
    << lex->getSourceRange() << rex->getSourceRange();
  return QualType();
}

inline QualType Sema::CheckVectorOperands(SourceLocation Loc, Expr *&lex,
                                                              Expr *&rex) {
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
    // FIXME: Should we warn here?
    if (const VectorType *LV = lhsType->getAsVectorType()) {
      if (const VectorType *RV = rhsType->getAsVectorType())
        if (LV->getElementType() == RV->getElementType() &&
            LV->getNumElements() == RV->getNumElements()) {
          return lhsType->isExtVectorType() ? lhsType : rhsType;
        }
    }
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
  if (const ExtVectorType *LV = lhsType->getAsExtVectorType()) {
    QualType EltTy = LV->getElementType();
    if (EltTy->isIntegralType() && rhsType->isIntegralType()) {
      if (Context.getIntegerTypeOrder(EltTy, rhsType) >= 0) {
        ImpCastExprToType(rex, lhsType);
        if (swapped) std::swap(rex, lex);
        return lhsType;
      }
    }
    if (EltTy->isRealFloatingType() && rhsType->isScalarType() &&
        rhsType->isRealFloatingType()) {
      if (Context.getFloatingTypeOrder(EltTy, rhsType) >= 0) {
        ImpCastExprToType(rex, lhsType);
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

inline QualType Sema::CheckMultiplyDivideOperands(
  Expr *&lex, Expr *&rex, SourceLocation Loc, bool isCompAssign)
{
  if (lex->getType()->isVectorType() || rex->getType()->isVectorType())
    return CheckVectorOperands(Loc, lex, rex);

  QualType compType = UsualArithmeticConversions(lex, rex, isCompAssign);

  if (lex->getType()->isArithmeticType() && rex->getType()->isArithmeticType())
    return compType;
  return InvalidOperands(Loc, lex, rex);
}

inline QualType Sema::CheckRemainderOperands(
  Expr *&lex, Expr *&rex, SourceLocation Loc, bool isCompAssign)
{
  if (lex->getType()->isVectorType() || rex->getType()->isVectorType()) {
    if (lex->getType()->isIntegerType() && rex->getType()->isIntegerType())
      return CheckVectorOperands(Loc, lex, rex);
    return InvalidOperands(Loc, lex, rex);
  }

  QualType compType = UsualArithmeticConversions(lex, rex, isCompAssign);

  if (lex->getType()->isIntegerType() && rex->getType()->isIntegerType())
    return compType;
  return InvalidOperands(Loc, lex, rex);
}

inline QualType Sema::CheckAdditionOperands( // C99 6.5.6
  Expr *&lex, Expr *&rex, SourceLocation Loc, QualType* CompLHSTy)
{
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
                                 diag::err_typecheck_arithmetic_incomplete_type,
                                 PExp->getSourceRange(), SourceRange(),
                                 PExp->getType()))
          return QualType();
      }
      // Diagnose bad cases where we step over interface counts.
      if (PointeeTy->isObjCInterfaceType() && LangOpts.ObjCNonFragileABI) {
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
                                   diag::err_typecheck_sub_ptr_object,
                                   lex->getSourceRange(),
                                   SourceRange(),
                                   lex->getType()))
      return QualType();

    // Diagnose bad cases where we step over interface counts.
    if (lpointee->isObjCInterfaceType() && LangOpts.ObjCNonFragileABI) {
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
                                     diag::err_typecheck_sub_ptr_object,
                                     rex->getSourceRange(),
                                     SourceRange(),
                                     rex->getType()))
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
  if (!lex->getType()->isIntegerType() || !rex->getType()->isIntegerType())
    return InvalidOperands(Loc, lex, rex);

  // Shifts don't perform usual arithmetic conversions, they just do integer
  // promotions on each operand. C99 6.5.7p3
  QualType LHSTy = Context.isPromotableBitField(lex);
  if (LHSTy.isNull()) {
    LHSTy = lex->getType();
    if (LHSTy->isPromotableIntegerType())
      LHSTy = Context.getPromotedIntegerType(LHSTy);
  }
  if (!isCompAssign)
    ImpCastExprToType(lex, LHSTy);

  UsualUnaryConversions(rex);

  // Sanity-check shift operands
  llvm::APSInt Right;
  // Check right/shifter operand
  if (rex->isIntegerConstantExpr(Right, Context)) {
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

// C99 6.5.8, C++ [expr.rel]
QualType Sema::CheckCompareOperands(Expr *&lex, Expr *&rex, SourceLocation Loc,
                                    unsigned OpaqueOpc, bool isRelational) {
  BinaryOperator::Opcode Opc = (BinaryOperator::Opcode)OpaqueOpc;

  if (lex->getType()->isVectorType() || rex->getType()->isVectorType())
    return CheckVectorCompareOperands(lex, rex, Loc, isRelational);

  // C99 6.5.8p3 / C99 6.5.9p4
  if (lex->getType()->isArithmeticType() && rex->getType()->isArithmeticType())
    UsualArithmeticConversions(lex, rex);
  else {
    UsualUnaryConversions(lex);
    UsualUnaryConversions(rex);
  }
  QualType lType = lex->getType();
  QualType rType = rex->getType();

  if (!lType->isFloatingType()
      && !(lType->isBlockPointerType() && isRelational)) {
    // For non-floating point types, check for self-comparisons of the form
    // x == x, x != x, x < x, etc.  These always evaluate to a constant, and
    // often indicate logic errors in the program.
    // NOTE: Don't warn about comparisons of enum constants. These can arise 
    //  from macro expansions, and are usually quite deliberate.
    Expr *LHSStripped = lex->IgnoreParens();
    Expr *RHSStripped = rex->IgnoreParens();
    if (DeclRefExpr* DRL = dyn_cast<DeclRefExpr>(LHSStripped))
      if (DeclRefExpr* DRR = dyn_cast<DeclRefExpr>(RHSStripped))
        if (DRL->getDecl() == DRR->getDecl() &&
            !isa<EnumConstantDecl>(DRL->getDecl()))
          Diag(Loc, diag::warn_selfcomparison);
    
    if (isa<CastExpr>(LHSStripped))
      LHSStripped = LHSStripped->IgnoreParenCasts();
    if (isa<CastExpr>(RHSStripped))
      RHSStripped = RHSStripped->IgnoreParenCasts();
    
    // Warn about comparisons against a string constant (unless the other
    // operand is null), the user probably wants strcmp.
    Expr *literalString = 0;
    Expr *literalStringStripped = 0;
    if ((isa<StringLiteral>(LHSStripped) || isa<ObjCEncodeExpr>(LHSStripped)) &&
        !RHSStripped->isNullPointerConstant(Context)) {
      literalString = lex;
      literalStringStripped = LHSStripped;
    } else if ((isa<StringLiteral>(RHSStripped) ||
                isa<ObjCEncodeExpr>(RHSStripped)) &&
               !LHSStripped->isNullPointerConstant(Context)) {
      literalString = rex;
      literalStringStripped = RHSStripped;
    }

    if (literalString) {
      std::string resultComparison;
      switch (Opc) {
      case BinaryOperator::LT: resultComparison = ") < 0"; break;
      case BinaryOperator::GT: resultComparison = ") > 0"; break;
      case BinaryOperator::LE: resultComparison = ") <= 0"; break;
      case BinaryOperator::GE: resultComparison = ") >= 0"; break;
      case BinaryOperator::EQ: resultComparison = ") == 0"; break;
      case BinaryOperator::NE: resultComparison = ") != 0"; break;
      default: assert(false && "Invalid comparison operator");
      }
      Diag(Loc, diag::warn_stringcompare)
        << isa<ObjCEncodeExpr>(literalStringStripped)
        << literalString->getSourceRange()
        << CodeModificationHint::CreateReplacement(SourceRange(Loc), ", ")
        << CodeModificationHint::CreateInsertion(lex->getLocStart(),
                                                 "strcmp(")
        << CodeModificationHint::CreateInsertion(
                                       PP.getLocForEndOfToken(rex->getLocEnd()),
                                       resultComparison);
    }
  }

  // The result of comparisons is 'bool' in C++, 'int' in C.
  QualType ResultTy = getLangOptions().CPlusPlus? Context.BoolTy :Context.IntTy;

  if (isRelational) {
    if (lType->isRealType() && rType->isRealType())
      return ResultTy;
  } else {
    // Check for comparisons of floating point operands using != and ==.
    if (lType->isFloatingType()) {
      assert(rType->isFloatingType());
      CheckFloatComparison(Loc,lex,rex);
    }

    if (lType->isArithmeticType() && rType->isArithmeticType())
      return ResultTy;
  }

  bool LHSIsNull = lex->isNullPointerConstant(Context);
  bool RHSIsNull = rex->isNullPointerConstant(Context);

  // All of the following pointer related warnings are GCC extensions, except
  // when handling null pointer constants. One day, we can consider making them
  // errors (when -pedantic-errors is enabled).
  if (lType->isPointerType() && rType->isPointerType()) { // C99 6.5.8p2
    QualType LCanPointeeTy =
      Context.getCanonicalType(lType->getAs<PointerType>()->getPointeeType());
    QualType RCanPointeeTy =
      Context.getCanonicalType(rType->getAs<PointerType>()->getPointeeType());

    if (isRelational) {
      if (lType->isFunctionPointerType() || rType->isFunctionPointerType()) {
        Diag(Loc, diag::ext_typecheck_ordered_comparison_of_function_pointers)
          << lType << rType << lex->getSourceRange() << rex->getSourceRange();
      }
      if (LCanPointeeTy->isVoidType() != RCanPointeeTy->isVoidType()) {
        Diag(Loc, diag::ext_typecheck_comparison_of_distinct_pointers)
          << lType << rType << lex->getSourceRange() << rex->getSourceRange();
      }
    } else {
      if (lType->isFunctionPointerType() != rType->isFunctionPointerType()) {
        if (!LHSIsNull && !RHSIsNull)
          Diag(Loc, diag::ext_typecheck_comparison_of_distinct_pointers)
            << lType << rType << lex->getSourceRange() << rex->getSourceRange();
      }
    }

    // Simple check: if the pointee types are identical, we're done.
    if (LCanPointeeTy == RCanPointeeTy)
      return ResultTy;

    if (getLangOptions().CPlusPlus) {
      // C++ [expr.rel]p2:
      //   [...] Pointer conversions (4.10) and qualification
      //   conversions (4.4) are performed on pointer operands (or on
      //   a pointer operand and a null pointer constant) to bring
      //   them to their composite pointer type. [...]
      //
      // C++ [expr.eq]p2 uses the same notion for (in)equality
      // comparisons of pointers.
      QualType T = FindCompositePointerType(lex, rex);
      if (T.isNull()) {
        Diag(Loc, diag::err_typecheck_comparison_of_distinct_pointers)
          << lType << rType << lex->getSourceRange() << rex->getSourceRange();
        return QualType();
      }

      ImpCastExprToType(lex, T);
      ImpCastExprToType(rex, T);
      return ResultTy;
    }

    if (!LHSIsNull && !RHSIsNull &&                       // C99 6.5.9p2
        !LCanPointeeTy->isVoidType() && !RCanPointeeTy->isVoidType() &&
        !Context.typesAreCompatible(LCanPointeeTy.getUnqualifiedType(),
                                    RCanPointeeTy.getUnqualifiedType())) {
      Diag(Loc, diag::ext_typecheck_comparison_of_distinct_pointers)
        << lType << rType << lex->getSourceRange() << rex->getSourceRange();
    }
    ImpCastExprToType(rex, lType); // promote the pointer to pointer
    return ResultTy;
  }
  // C++ allows comparison of pointers with null pointer constants.
  if (getLangOptions().CPlusPlus) {
    if (lType->isPointerType() && RHSIsNull) {
      ImpCastExprToType(rex, lType);
      return ResultTy;
    }
    if (rType->isPointerType() && LHSIsNull) {
      ImpCastExprToType(lex, rType);
      return ResultTy;
    }
    // And comparison of nullptr_t with itself.
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
    ImpCastExprToType(rex, lType); // promote the pointer to pointer
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
    ImpCastExprToType(rex, lType); // promote the pointer to pointer
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
      ImpCastExprToType(rex, lType);
      return ResultTy;
    }
    if (lType->isObjCObjectPointerType() && rType->isObjCObjectPointerType()) {
      if (!Context.areComparableObjCPointerTypes(lType, rType))
        Diag(Loc, diag::ext_typecheck_comparison_of_distinct_pointers)
          << lType << rType << lex->getSourceRange() << rex->getSourceRange();
      ImpCastExprToType(rex, lType);
      return ResultTy;
    }
  }
  if (lType->isAnyPointerType() && rType->isIntegerType()) {
    if (!RHSIsNull) {
      unsigned DiagID = isRelational
           ? diag::ext_typecheck_ordered_comparison_of_pointer_integer
           : diag::ext_typecheck_comparison_of_pointer_integer;
      Diag(Loc, DiagID)
        << lType << rType << lex->getSourceRange() << rex->getSourceRange();
    }
    ImpCastExprToType(rex, lType); // promote the integer to pointer
    return ResultTy;
  }
  if (lType->isIntegerType() && rType->isAnyPointerType()) {
    if (!LHSIsNull) {
      unsigned DiagID = isRelational
        ? diag::ext_typecheck_ordered_comparison_of_pointer_integer
        : diag::ext_typecheck_comparison_of_pointer_integer;
      
      Diag(Loc, DiagID)
        << lType << rType << lex->getSourceRange() << rex->getSourceRange();
    }
    ImpCastExprToType(lex, rType); // promote the integer to pointer
    return ResultTy;
  }
  // Handle block pointers.
  if (!isRelational && RHSIsNull
      && lType->isBlockPointerType() && rType->isIntegerType()) {
    ImpCastExprToType(rex, lType); // promote the integer to pointer
    return ResultTy;
  }
  if (!isRelational && LHSIsNull
      && lType->isIntegerType() && rType->isBlockPointerType()) {
    ImpCastExprToType(lex, rType); // promote the integer to pointer
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
  if (!lType->isFloatingType()) {
    if (DeclRefExpr* DRL = dyn_cast<DeclRefExpr>(lex->IgnoreParens()))
      if (DeclRefExpr* DRR = dyn_cast<DeclRefExpr>(rex->IgnoreParens()))
        if (DRL->getDecl() == DRR->getDecl())
          Diag(Loc, diag::warn_selfcomparison);
  }

  // Check for comparisons of floating point operands using != and ==.
  if (!isRelational && lType->isFloatingType()) {
    assert (rType->isFloatingType());
    CheckFloatComparison(Loc,lex,rex);
  }

  // Return the type for the comparison, which is the same as vector type for
  // integer vectors, or an integer type of identical size and number of
  // elements for floating point vectors.
  if (lType->isIntegerType())
    return lType;

  const VectorType *VTy = lType->getAsVectorType();
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
  Expr *&lex, Expr *&rex, SourceLocation Loc, bool isCompAssign)
{
  if (lex->getType()->isVectorType() || rex->getType()->isVectorType())
    return CheckVectorOperands(Loc, lex, rex);

  QualType compType = UsualArithmeticConversions(lex, rex, isCompAssign);

  if (lex->getType()->isIntegerType() && rex->getType()->isIntegerType())
    return compType;
  return InvalidOperands(Loc, lex, rex);
}

inline QualType Sema::CheckLogicalOperands( // C99 6.5.[13,14]
  Expr *&lex, Expr *&rex, SourceLocation Loc)
{
  UsualUnaryConversions(lex);
  UsualUnaryConversions(rex);

  if (lex->getType()->isScalarType() && rex->getType()->isScalarType())
    return Context.IntTy;
  return InvalidOperands(Loc, lex, rex);
}

/// IsReadonlyProperty - Verify that otherwise a valid l-value expression
/// is a read-only property; return true if so. A readonly property expression
/// depends on various declarations and thus must be treated specially.
///
static bool IsReadonlyProperty(Expr *E, Sema &S)
{
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
  default: assert(0 && "Unknown result from isModifiableLvalue!");
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
  case Expr::MLV_InvalidExpression:
    Diag = diag::err_typecheck_expression_not_modifiable_lvalue;
    break;
  case Expr::MLV_IncompleteType:
  case Expr::MLV_IncompleteVoidType:
    return S.RequireCompleteType(Loc, E->getType(),
                      diag::err_typecheck_incomplete_type_not_modifiable_lvalue,
                                    E->getSourceRange());
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
    // Simple assignment "x = y".
    ConvTy = CheckSingleAssignmentConstraints(LHSType, RHS);
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
      if ((UO->getOpcode() == UnaryOperator::Plus ||
           UO->getOpcode() == UnaryOperator::Minus) &&
          Loc.isFileID() && UO->getOperatorLoc().isFileID() &&
          // Only if the two operators are exactly adjacent.
          Loc.getFileLocWithOffset(1) == UO->getOperatorLoc() &&
          // And there is a space or other character before the subexpr of the
          // unary +/-.  We don't want to warn on "x=-1".
          Loc.getFileLocWithOffset(2) != UO->getSubExpr()->getLocStart() &&
          UO->getSubExpr()->getLocStart().isFileID()) {
        Diag(Loc, diag::warn_not_compound_assign)
          << (UO->getOpcode() == UnaryOperator::Plus ? "+" : "-")
          << SourceRange(UO->getOperatorLoc(), UO->getOperatorLoc());
      }
    }
  } else {
    // Compound assignment "x += y"
    ConvTy = CheckAssignmentConstraints(LHSType, RHSType);
  }

  if (DiagnoseAssignmentResult(ConvTy, Loc, LHSType, RHSType,
                               RHS, "assigning"))
    return QualType();

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
  // Comma performs lvalue conversion (C99 6.3.2.1), but not unary conversions.
  DefaultFunctionArrayConversion(RHS);

  // FIXME: Check that RHS type is complete in C mode (it's legal for it to be
  // incomplete in C++).

  return RHS->getType();
}

/// CheckIncrementDecrementOperand - unlike most "Check" methods, this routine
/// doesn't need to call UsualUnaryConversions or UsualArithmeticConversions.
QualType Sema::CheckIncrementDecrementOperand(Expr *Op, SourceLocation OpLoc,
                                              bool isInc) {
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
                               diag::err_typecheck_arithmetic_incomplete_type,
                                   Op->getSourceRange(), SourceRange(),
                                   ResType))
      return QualType();
    // Diagnose bad cases where we step over interface counts.
    else if (PointeeTy->isObjCInterfaceType() && LangOpts.ObjCNonFragileABI) {
      Diag(OpLoc, diag::err_arithmetic_nonfragile_interface)
        << PointeeTy << Op->getSourceRange();
      return QualType();
    }
  } else if (ResType->isComplexType()) {
    // C99 does not support ++/-- on complex types, we allow as an extension.
    Diag(OpLoc, diag::ext_integer_increment_complex)
      << ResType << Op->getSourceRange();
  } else {
    Diag(OpLoc, diag::err_typecheck_illegal_increment_decrement)
      << ResType << Op->getSourceRange();
    return QualType();
  }
  // At this point, we know we have a real, complex or pointer type.
  // Now make sure the operand is a modifiable lvalue.
  if (CheckForModifiableLvalue(Op, OpLoc, *this))
    return QualType();
  return ResType;
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
  case Stmt::QualifiedDeclRefExprClass:
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
    case UnaryOperator::Real:
    case UnaryOperator::Imag:
    case UnaryOperator::Extension:
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
QualType Sema::CheckAddressOfOperand(Expr *op, SourceLocation OpLoc) {
  // Make sure to ignore parentheses in subsequent checks
  op = op->IgnoreParens();

  if (op->isTypeDependent())
    return Context.DependentTy;

  if (getLangOptions().C99) {
    // Implement C99-only parts of addressof rules.
    if (UnaryOperator* uOp = dyn_cast<UnaryOperator>(op)) {
      if (uOp->getOpcode() == UnaryOperator::Deref)
        // Per C99 6.5.3.2, the address of a deref always returns a valid result
        // (assuming the deref expression is valid).
        return uOp->getSubExpr()->getType();
    }
    // Technically, there should be a check for array subscript
    // expressions here, but the result of one is always an lvalue anyway.
  }
  NamedDecl *dcl = getPrimaryDecl(op);
  Expr::isLvalueResult lval = op->isLvalue(Context);

  if (lval != Expr::LV_Valid && lval != Expr::LV_IncompleteVoidType) {
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
  } else if (isa<ExtVectorElementExpr>(op) || (isa<ArraySubscriptExpr>(op) &&
           cast<ArraySubscriptExpr>(op)->getBase()->getType()->isVectorType())){
    // The operand cannot be an element of a vector
    Diag(OpLoc, diag::err_typecheck_address_of)
      << "vector element" << op->getSourceRange();
    return QualType();
  } else if (isa<ObjCPropertyRefExpr>(op)) {
    // cannot take address of a property expression.
    Diag(OpLoc, diag::err_typecheck_address_of)
      << "property expression" << op->getSourceRange();
    return QualType();
  } else if (dcl) { // C99 6.5.3.2p1
    // We have an lvalue with a decl. Make sure the decl is not declared
    // with the register storage-class specifier.
    if (const VarDecl *vd = dyn_cast<VarDecl>(dcl)) {
      if (vd->getStorageClass() == VarDecl::Register) {
        Diag(OpLoc, diag::err_typecheck_address_of)
          << "register variable" << op->getSourceRange();
        return QualType();
      }
    } else if (isa<OverloadedFunctionDecl>(dcl) ||
               isa<FunctionTemplateDecl>(dcl)) {
      return Context.OverloadTy;
    } else if (FieldDecl *FD = dyn_cast<FieldDecl>(dcl)) {
      // Okay: we can take the address of a field.
      // Could be a pointer to member, though, if there is an explicit
      // scope qualifier for the class.
      if (isa<QualifiedDeclRefExpr>(op)) {
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
    } else if (CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(dcl)) {
      // Okay: we can take the address of a function.
      // As above.
      if (isa<QualifiedDeclRefExpr>(op) && MD->isInstance())
        return Context.getMemberPointerType(op->getType(),
              Context.getTypeDeclType(MD->getParent()).getTypePtr());
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
  return Context.getPointerType(op->getType());
}

QualType Sema::CheckIndirectionOperand(Expr *Op, SourceLocation OpLoc) {
  if (Op->isTypeDependent())
    return Context.DependentTy;

  UsualUnaryConversions(Op);
  QualType Ty = Op->getType();

  // Note that per both C89 and C99, this is always legal, even if ptype is an
  // incomplete type or void.  It would be possible to warn about dereferencing
  // a void pointer, but it's completely well-defined, and such a warning is
  // unlikely to catch any mistakes.
  if (const PointerType *PT = Ty->getAs<PointerType>())
    return PT->getPointeeType();

  if (const ObjCObjectPointerType *OPT = Ty->getAsObjCObjectPointerType())
    return OPT->getPointeeType();

  Diag(OpLoc, diag::err_typecheck_indirection_requires_pointer)
    << Ty << Op->getSourceRange();
  return QualType();
}

static inline BinaryOperator::Opcode ConvertTokenKindToBinaryOpcode(
  tok::TokenKind Kind) {
  BinaryOperator::Opcode Opc;
  switch (Kind) {
  default: assert(0 && "Unknown binop!");
  case tok::periodstar:           Opc = BinaryOperator::PtrMemD; break;
  case tok::arrowstar:            Opc = BinaryOperator::PtrMemI; break;
  case tok::star:                 Opc = BinaryOperator::Mul; break;
  case tok::slash:                Opc = BinaryOperator::Div; break;
  case tok::percent:              Opc = BinaryOperator::Rem; break;
  case tok::plus:                 Opc = BinaryOperator::Add; break;
  case tok::minus:                Opc = BinaryOperator::Sub; break;
  case tok::lessless:             Opc = BinaryOperator::Shl; break;
  case tok::greatergreater:       Opc = BinaryOperator::Shr; break;
  case tok::lessequal:            Opc = BinaryOperator::LE; break;
  case tok::less:                 Opc = BinaryOperator::LT; break;
  case tok::greaterequal:         Opc = BinaryOperator::GE; break;
  case tok::greater:              Opc = BinaryOperator::GT; break;
  case tok::exclaimequal:         Opc = BinaryOperator::NE; break;
  case tok::equalequal:           Opc = BinaryOperator::EQ; break;
  case tok::amp:                  Opc = BinaryOperator::And; break;
  case tok::caret:                Opc = BinaryOperator::Xor; break;
  case tok::pipe:                 Opc = BinaryOperator::Or; break;
  case tok::ampamp:               Opc = BinaryOperator::LAnd; break;
  case tok::pipepipe:             Opc = BinaryOperator::LOr; break;
  case tok::equal:                Opc = BinaryOperator::Assign; break;
  case tok::starequal:            Opc = BinaryOperator::MulAssign; break;
  case tok::slashequal:           Opc = BinaryOperator::DivAssign; break;
  case tok::percentequal:         Opc = BinaryOperator::RemAssign; break;
  case tok::plusequal:            Opc = BinaryOperator::AddAssign; break;
  case tok::minusequal:           Opc = BinaryOperator::SubAssign; break;
  case tok::lesslessequal:        Opc = BinaryOperator::ShlAssign; break;
  case tok::greatergreaterequal:  Opc = BinaryOperator::ShrAssign; break;
  case tok::ampequal:             Opc = BinaryOperator::AndAssign; break;
  case tok::caretequal:           Opc = BinaryOperator::XorAssign; break;
  case tok::pipeequal:            Opc = BinaryOperator::OrAssign; break;
  case tok::comma:                Opc = BinaryOperator::Comma; break;
  }
  return Opc;
}

static inline UnaryOperator::Opcode ConvertTokenKindToUnaryOpcode(
  tok::TokenKind Kind) {
  UnaryOperator::Opcode Opc;
  switch (Kind) {
  default: assert(0 && "Unknown unary op!");
  case tok::plusplus:     Opc = UnaryOperator::PreInc; break;
  case tok::minusminus:   Opc = UnaryOperator::PreDec; break;
  case tok::amp:          Opc = UnaryOperator::AddrOf; break;
  case tok::star:         Opc = UnaryOperator::Deref; break;
  case tok::plus:         Opc = UnaryOperator::Plus; break;
  case tok::minus:        Opc = UnaryOperator::Minus; break;
  case tok::tilde:        Opc = UnaryOperator::Not; break;
  case tok::exclaim:      Opc = UnaryOperator::LNot; break;
  case tok::kw___real:    Opc = UnaryOperator::Real; break;
  case tok::kw___imag:    Opc = UnaryOperator::Imag; break;
  case tok::kw___extension__: Opc = UnaryOperator::Extension; break;
  }
  return Opc;
}

/// CreateBuiltinBinOp - Creates a new built-in binary operation with
/// operator @p Opc at location @c TokLoc. This routine only supports
/// built-in operations; ActOnBinOp handles overloaded operators.
Action::OwningExprResult Sema::CreateBuiltinBinOp(SourceLocation OpLoc,
                                                  unsigned Op,
                                                  Expr *lhs, Expr *rhs) {
  QualType ResultTy;     // Result type of the binary operator.
  BinaryOperator::Opcode Opc = (BinaryOperator::Opcode)Op;
  // The following two variables are used for compound assignment operators
  QualType CompLHSTy;    // Type of LHS after promotions for computation
  QualType CompResultTy; // Type of computation result

  switch (Opc) {
  case BinaryOperator::Assign:
    ResultTy = CheckAssignmentOperands(lhs, rhs, OpLoc, QualType());
    break;
  case BinaryOperator::PtrMemD:
  case BinaryOperator::PtrMemI:
    ResultTy = CheckPointerToMemberOperands(lhs, rhs, OpLoc,
                                            Opc == BinaryOperator::PtrMemI);
    break;
  case BinaryOperator::Mul:
  case BinaryOperator::Div:
    ResultTy = CheckMultiplyDivideOperands(lhs, rhs, OpLoc);
    break;
  case BinaryOperator::Rem:
    ResultTy = CheckRemainderOperands(lhs, rhs, OpLoc);
    break;
  case BinaryOperator::Add:
    ResultTy = CheckAdditionOperands(lhs, rhs, OpLoc);
    break;
  case BinaryOperator::Sub:
    ResultTy = CheckSubtractionOperands(lhs, rhs, OpLoc);
    break;
  case BinaryOperator::Shl:
  case BinaryOperator::Shr:
    ResultTy = CheckShiftOperands(lhs, rhs, OpLoc);
    break;
  case BinaryOperator::LE:
  case BinaryOperator::LT:
  case BinaryOperator::GE:
  case BinaryOperator::GT:
    ResultTy = CheckCompareOperands(lhs, rhs, OpLoc, Opc, true);
    break;
  case BinaryOperator::EQ:
  case BinaryOperator::NE:
    ResultTy = CheckCompareOperands(lhs, rhs, OpLoc, Opc, false);
    break;
  case BinaryOperator::And:
  case BinaryOperator::Xor:
  case BinaryOperator::Or:
    ResultTy = CheckBitwiseOperands(lhs, rhs, OpLoc);
    break;
  case BinaryOperator::LAnd:
  case BinaryOperator::LOr:
    ResultTy = CheckLogicalOperands(lhs, rhs, OpLoc);
    break;
  case BinaryOperator::MulAssign:
  case BinaryOperator::DivAssign:
    CompResultTy = CheckMultiplyDivideOperands(lhs, rhs, OpLoc, true);
    CompLHSTy = CompResultTy;
    if (!CompResultTy.isNull())
      ResultTy = CheckAssignmentOperands(lhs, rhs, OpLoc, CompResultTy);
    break;
  case BinaryOperator::RemAssign:
    CompResultTy = CheckRemainderOperands(lhs, rhs, OpLoc, true);
    CompLHSTy = CompResultTy;
    if (!CompResultTy.isNull())
      ResultTy = CheckAssignmentOperands(lhs, rhs, OpLoc, CompResultTy);
    break;
  case BinaryOperator::AddAssign:
    CompResultTy = CheckAdditionOperands(lhs, rhs, OpLoc, &CompLHSTy);
    if (!CompResultTy.isNull())
      ResultTy = CheckAssignmentOperands(lhs, rhs, OpLoc, CompResultTy);
    break;
  case BinaryOperator::SubAssign:
    CompResultTy = CheckSubtractionOperands(lhs, rhs, OpLoc, &CompLHSTy);
    if (!CompResultTy.isNull())
      ResultTy = CheckAssignmentOperands(lhs, rhs, OpLoc, CompResultTy);
    break;
  case BinaryOperator::ShlAssign:
  case BinaryOperator::ShrAssign:
    CompResultTy = CheckShiftOperands(lhs, rhs, OpLoc, true);
    CompLHSTy = CompResultTy;
    if (!CompResultTy.isNull())
      ResultTy = CheckAssignmentOperands(lhs, rhs, OpLoc, CompResultTy);
    break;
  case BinaryOperator::AndAssign:
  case BinaryOperator::XorAssign:
  case BinaryOperator::OrAssign:
    CompResultTy = CheckBitwiseOperands(lhs, rhs, OpLoc, true);
    CompLHSTy = CompResultTy;
    if (!CompResultTy.isNull())
      ResultTy = CheckAssignmentOperands(lhs, rhs, OpLoc, CompResultTy);
    break;
  case BinaryOperator::Comma:
    ResultTy = CheckCommaOperands(lhs, rhs, OpLoc);
    break;
  }
  if (ResultTy.isNull())
    return ExprError();
  if (CompResultTy.isNull())
    return Owned(new (Context) BinaryOperator(lhs, rhs, Opc, ResultTy, OpLoc));
  else
    return Owned(new (Context) CompoundAssignOperator(lhs, rhs, Opc, ResultTy,
                                                      CompLHSTy, CompResultTy,
                                                      OpLoc));
}

// Binary Operators.  'Tok' is the token for the operator.
Action::OwningExprResult Sema::ActOnBinOp(Scope *S, SourceLocation TokLoc,
                                          tok::TokenKind Kind,
                                          ExprArg LHS, ExprArg RHS) {
  BinaryOperator::Opcode Opc = ConvertTokenKindToBinaryOpcode(Kind);
  Expr *lhs = LHS.takeAs<Expr>(), *rhs = RHS.takeAs<Expr>();

  assert((lhs != 0) && "ActOnBinOp(): missing left expression");
  assert((rhs != 0) && "ActOnBinOp(): missing right expression");

  if (getLangOptions().CPlusPlus &&
      (lhs->getType()->isOverloadableType() || 
       rhs->getType()->isOverloadableType())) {
    // Find all of the overloaded operators visible from this
    // point. We perform both an operator-name lookup from the local
    // scope and an argument-dependent lookup based on the types of
    // the arguments.
    FunctionSet Functions;
    OverloadedOperatorKind OverOp = BinaryOperator::getOverloadedOperator(Opc);
    if (OverOp != OO_None) {
      LookupOverloadedOperatorName(OverOp, S, lhs->getType(), rhs->getType(),
                                   Functions);
      Expr *Args[2] = { lhs, rhs };
      DeclarationName OpName 
        = Context.DeclarationNames.getCXXOperatorName(OverOp);
      ArgumentDependentLookup(OpName, Args, 2, Functions);
    }

    // Build the (potentially-overloaded, potentially-dependent)
    // binary operation.
    return CreateOverloadedBinOp(TokLoc, Opc, Functions, lhs, rhs);
  }

  // Build a built-in binary operation.
  return CreateBuiltinBinOp(TokLoc, Opc, lhs, rhs);
}

Action::OwningExprResult Sema::CreateBuiltinUnaryOp(SourceLocation OpLoc,
                                                    unsigned OpcIn, 
                                                    ExprArg InputArg) {
  UnaryOperator::Opcode Opc = static_cast<UnaryOperator::Opcode>(OpcIn);

  // FIXME: Input is modified below, but InputArg is not updated appropriately.
  Expr *Input = (Expr *)InputArg.get();
  QualType resultType;
  switch (Opc) {
  case UnaryOperator::OffsetOf:
    assert(false && "Invalid unary operator");
    break;

  case UnaryOperator::PreInc:
  case UnaryOperator::PreDec:
  case UnaryOperator::PostInc:
  case UnaryOperator::PostDec:
    resultType = CheckIncrementDecrementOperand(Input, OpLoc,
                                                Opc == UnaryOperator::PreInc ||
                                                Opc == UnaryOperator::PostInc);
    break;
  case UnaryOperator::AddrOf:
    resultType = CheckAddressOfOperand(Input, OpLoc);
    break;
  case UnaryOperator::Deref:
    DefaultFunctionArrayConversion(Input);
    resultType = CheckIndirectionOperand(Input, OpLoc);
    break;
  case UnaryOperator::Plus:
  case UnaryOperator::Minus:
    UsualUnaryConversions(Input);
    resultType = Input->getType();
    if (resultType->isDependentType())
      break;
    if (resultType->isArithmeticType()) // C99 6.5.3.3p1
      break;
    else if (getLangOptions().CPlusPlus && // C++ [expr.unary.op]p6-7
             resultType->isEnumeralType())
      break;
    else if (getLangOptions().CPlusPlus && // C++ [expr.unary.op]p6
             Opc == UnaryOperator::Plus &&
             resultType->isPointerType())
      break;

    return ExprError(Diag(OpLoc, diag::err_typecheck_unary_expr)
      << resultType << Input->getSourceRange());
  case UnaryOperator::Not: // bitwise complement
    UsualUnaryConversions(Input);
    resultType = Input->getType();
    if (resultType->isDependentType())
      break;
    // C99 6.5.3.3p1. We allow complex int and float as a GCC extension.
    if (resultType->isComplexType() || resultType->isComplexIntegerType())
      // C99 does not support '~' for complex conjugation.
      Diag(OpLoc, diag::ext_integer_complement_complex)
        << resultType << Input->getSourceRange();
    else if (!resultType->isIntegerType())
      return ExprError(Diag(OpLoc, diag::err_typecheck_unary_expr)
        << resultType << Input->getSourceRange());
    break;
  case UnaryOperator::LNot: // logical negation
    // Unlike +/-/~, integer promotions aren't done here (C99 6.5.3.3p5).
    DefaultFunctionArrayConversion(Input);
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
  case UnaryOperator::Real:
  case UnaryOperator::Imag:
    resultType = CheckRealImagOperand(Input, OpLoc, Opc == UnaryOperator::Real);
    break;
  case UnaryOperator::Extension:
    resultType = Input->getType();
    break;
  }
  if (resultType.isNull())
    return ExprError();

  InputArg.release();
  return Owned(new (Context) UnaryOperator(Input, Opc, resultType, OpLoc));
}

// Unary Operators.  'Tok' is the token for the operator.
Action::OwningExprResult Sema::ActOnUnaryOp(Scope *S, SourceLocation OpLoc,
                                            tok::TokenKind Op, ExprArg input) {
  Expr *Input = (Expr*)input.get();
  UnaryOperator::Opcode Opc = ConvertTokenKindToUnaryOpcode(Op);

  if (getLangOptions().CPlusPlus && Input->getType()->isOverloadableType()) {
    // Find all of the overloaded operators visible from this
    // point. We perform both an operator-name lookup from the local
    // scope and an argument-dependent lookup based on the types of
    // the arguments.
    FunctionSet Functions;
    OverloadedOperatorKind OverOp = UnaryOperator::getOverloadedOperator(Opc);
    if (OverOp != OO_None) {
      LookupOverloadedOperatorName(OverOp, S, Input->getType(), QualType(),
                                   Functions);
      DeclarationName OpName 
        = Context.DeclarationNames.getCXXOperatorName(OverOp);
      ArgumentDependentLookup(OpName, &Input, 1, Functions);
    }

    return CreateOverloadedUnaryOp(OpLoc, Opc, Functions, move(input));
  }

  return CreateBuiltinUnaryOp(OpLoc, Opc, move(input));
}

/// ActOnAddrLabel - Parse the GNU address of label extension: "&&foo".
Sema::OwningExprResult Sema::ActOnAddrLabel(SourceLocation OpLoc,
                                            SourceLocation LabLoc,
                                            IdentifierInfo *LabelII) {
  // Look up the record for this label identifier.
  LabelStmt *&LabelDecl = getLabelMap()[LabelII];

  // If we haven't seen this label yet, create a forward reference. It
  // will be validated and/or cleaned up in ActOnFinishFunctionBody.
  if (LabelDecl == 0)
    LabelDecl = new (Context) LabelStmt(LabLoc, LabelII, 0);

  // Create the AST node.  The address of a label always has type 'void*'.
  return Owned(new (Context) AddrLabelExpr(OpLoc, LabLoc, LabelDecl,
                                       Context.getPointerType(Context.VoidTy)));
}

Sema::OwningExprResult
Sema::ActOnStmtExpr(SourceLocation LPLoc, StmtArg substmt,
                    SourceLocation RPLoc) { // "({..})"
  Stmt *SubStmt = static_cast<Stmt*>(substmt.get());
  assert(SubStmt && isa<CompoundStmt>(SubStmt) && "Invalid action invocation!");
  CompoundStmt *Compound = cast<CompoundStmt>(SubStmt);

  bool isFileScope = getCurFunctionOrMethodDecl() == 0;
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

  substmt.release();
  return Owned(new (Context) StmtExpr(Compound, Ty, LPLoc, RPLoc));
}

Sema::OwningExprResult Sema::ActOnBuiltinOffsetOf(Scope *S,
                                                  SourceLocation BuiltinLoc,
                                                  SourceLocation TypeLoc,
                                                  TypeTy *argty,
                                                  OffsetOfComponent *CompPtr,
                                                  unsigned NumComponents,
                                                  SourceLocation RPLoc) {
  // FIXME: This function leaks all expressions in the offset components on
  // error.
  // FIXME: Preserve type source info.
  QualType ArgTy = GetTypeFromParser(argty);
  assert(!ArgTy.isNull() && "Missing type argument!");

  bool Dependent = ArgTy->isDependentType();

  // We must have at least one component that refers to the type, and the first
  // one is known to be a field designator.  Verify that the ArgTy represents
  // a struct/union/class.
  if (!Dependent && !ArgTy->isRecordType())
    return ExprError(Diag(TypeLoc, diag::err_offsetof_record_type) << ArgTy);

  // FIXME: Type must be complete per C99 7.17p3 because a declaring a variable
  // with an incomplete type would be illegal.

  // Otherwise, create a null pointer as the base, and iteratively process
  // the offsetof designators.
  QualType ArgTyPtr = Context.getPointerType(ArgTy);
  Expr* Res = new (Context) ImplicitValueInitExpr(ArgTyPtr);
  Res = new (Context) UnaryOperator(Res, UnaryOperator::Deref,
                                    ArgTy, SourceLocation());

  // offsetof with non-identifier designators (e.g. "offsetof(x, a.b[c])") are a
  // GCC extension, diagnose them.
  // FIXME: This diagnostic isn't actually visible because the location is in
  // a system header!
  if (NumComponents != 1)
    Diag(BuiltinLoc, diag::ext_offsetof_extended_field_designator)
      << SourceRange(CompPtr[1].LocStart, CompPtr[NumComponents-1].LocEnd);

  if (!Dependent) {
    bool DidWarnAboutNonPOD = false;
    
    // FIXME: Dependent case loses a lot of information here. And probably
    // leaks like a sieve.
    for (unsigned i = 0; i != NumComponents; ++i) {
      const OffsetOfComponent &OC = CompPtr[i];
      if (OC.isBrackets) {
        // Offset of an array sub-field.  TODO: Should we allow vector elements?
        const ArrayType *AT = Context.getAsArrayType(Res->getType());
        if (!AT) {
          Res->Destroy(Context);
          return ExprError(Diag(OC.LocEnd, diag::err_offsetof_array_type)
            << Res->getType());
        }

        // FIXME: C++: Verify that operator[] isn't overloaded.

        // Promote the array so it looks more like a normal array subscript
        // expression.
        DefaultFunctionArrayConversion(Res);

        // C99 6.5.2.1p1
        Expr *Idx = static_cast<Expr*>(OC.U.E);
        // FIXME: Leaks Res
        if (!Idx->isTypeDependent() && !Idx->getType()->isIntegerType())
          return ExprError(Diag(Idx->getLocStart(),
                                diag::err_typecheck_subscript_not_integer)
            << Idx->getSourceRange());

        Res = new (Context) ArraySubscriptExpr(Res, Idx, AT->getElementType(),
                                               OC.LocEnd);
        continue;
      }

      const RecordType *RC = Res->getType()->getAs<RecordType>();
      if (!RC) {
        Res->Destroy(Context);
        return ExprError(Diag(OC.LocEnd, diag::err_offsetof_record_type)
          << Res->getType());
      }

      // Get the decl corresponding to this.
      RecordDecl *RD = RC->getDecl();
      if (CXXRecordDecl *CRD = dyn_cast<CXXRecordDecl>(RD)) {
        if (!CRD->isPOD() && !DidWarnAboutNonPOD) {
          ExprError(Diag(BuiltinLoc, diag::warn_offsetof_non_pod_type)
            << SourceRange(CompPtr[0].LocStart, OC.LocEnd)
            << Res->getType());
          DidWarnAboutNonPOD = true;
        }
      }
      
      FieldDecl *MemberDecl
        = dyn_cast_or_null<FieldDecl>(LookupQualifiedName(RD, OC.U.IdentInfo,
                                                          LookupMemberName)
                                        .getAsDecl());
      // FIXME: Leaks Res
      if (!MemberDecl)
        return ExprError(Diag(BuiltinLoc, diag::err_typecheck_no_member)
         << OC.U.IdentInfo << SourceRange(OC.LocStart, OC.LocEnd));

      // FIXME: C++: Verify that MemberDecl isn't a static field.
      // FIXME: Verify that MemberDecl isn't a bitfield.
      if (cast<RecordDecl>(MemberDecl->getDeclContext())->isAnonymousStructOrUnion()) {
        Res = BuildAnonymousStructUnionMemberReference(
            SourceLocation(), MemberDecl, Res, SourceLocation()).takeAs<Expr>();
      } else {
        // MemberDecl->getType() doesn't get the right qualifiers, but it
        // doesn't matter here.
        Res = new (Context) MemberExpr(Res, false, MemberDecl, OC.LocEnd,
                MemberDecl->getType().getNonReferenceType());
      }
    }
  }

  return Owned(new (Context) UnaryOperator(Res, UnaryOperator::OffsetOf,
                                           Context.getSizeType(), BuiltinLoc));
}


Sema::OwningExprResult Sema::ActOnTypesCompatibleExpr(SourceLocation BuiltinLoc,
                                                      TypeTy *arg1,TypeTy *arg2,
                                                      SourceLocation RPLoc) {
  // FIXME: Preserve type source info.
  QualType argT1 = GetTypeFromParser(arg1);
  QualType argT2 = GetTypeFromParser(arg2);

  assert((!argT1.isNull() && !argT2.isNull()) && "Missing type argument(s)");

  if (getLangOptions().CPlusPlus) {
    Diag(BuiltinLoc, diag::err_types_compatible_p_in_cplusplus)
      << SourceRange(BuiltinLoc, RPLoc);
    return ExprError();
  }

  return Owned(new (Context) TypesCompatibleExpr(Context.IntTy, BuiltinLoc,
                                                 argT1, argT2, RPLoc));
}

Sema::OwningExprResult Sema::ActOnChooseExpr(SourceLocation BuiltinLoc,
                                             ExprArg cond,
                                             ExprArg expr1, ExprArg expr2,
                                             SourceLocation RPLoc) {
  Expr *CondExpr = static_cast<Expr*>(cond.get());
  Expr *LHSExpr = static_cast<Expr*>(expr1.get());
  Expr *RHSExpr = static_cast<Expr*>(expr2.get());

  assert((CondExpr && LHSExpr && RHSExpr) && "Missing type argument(s)");

  QualType resType;
  if (CondExpr->isTypeDependent() || CondExpr->isValueDependent()) {
    resType = Context.DependentTy;
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
  }

  cond.release(); expr1.release(); expr2.release();
  return Owned(new (Context) ChooseExpr(BuiltinLoc, CondExpr, LHSExpr, RHSExpr,
                                        resType, RPLoc));
}

//===----------------------------------------------------------------------===//
// Clang Extensions.
//===----------------------------------------------------------------------===//

/// ActOnBlockStart - This callback is invoked when a block literal is started.
void Sema::ActOnBlockStart(SourceLocation CaretLoc, Scope *BlockScope) {
  // Analyze block parameters.
  BlockSemaInfo *BSI = new BlockSemaInfo();

  // Add BSI to CurBlock.
  BSI->PrevBlockInfo = CurBlock;
  CurBlock = BSI;

  BSI->ReturnType = QualType();
  BSI->TheScope = BlockScope;
  BSI->hasBlockDeclRefExprs = false;
  BSI->hasPrototype = false;
  BSI->SavedFunctionNeedsScopeChecking = CurFunctionNeedsScopeChecking;
  CurFunctionNeedsScopeChecking = false;

  BSI->TheDecl = BlockDecl::Create(Context, CurContext, CaretLoc);
  PushDeclContext(BlockScope, BSI->TheDecl);
}

void Sema::ActOnBlockArguments(Declarator &ParamInfo, Scope *CurScope) {
  assert(ParamInfo.getIdentifier()==0 && "block-id should have no identifier!");

  if (ParamInfo.getNumTypeObjects() == 0
      || ParamInfo.getTypeObject(0).Kind != DeclaratorChunk::Function) {
    ProcessDeclAttributes(CurScope, CurBlock->TheDecl, ParamInfo);
    QualType T = GetTypeForDeclarator(ParamInfo, CurScope);

    if (T->isArrayType()) {
      Diag(ParamInfo.getSourceRange().getBegin(),
           diag::err_block_returns_array);
      return;
    }

    // The parameter list is optional, if there was none, assume ().
    if (!T->isFunctionType())
      T = Context.getFunctionType(T, NULL, 0, 0, 0);

    CurBlock->hasPrototype = true;
    CurBlock->isVariadic = false;
    // Check for a valid sentinel attribute on this block.
    if (CurBlock->TheDecl->getAttr<SentinelAttr>()) {
      Diag(ParamInfo.getAttributes()->getLoc(), 
           diag::warn_attribute_sentinel_not_variadic) << 1;
      // FIXME: remove the attribute.
    }
    QualType RetTy = T.getTypePtr()->getAsFunctionType()->getResultType();
    
    // Do not allow returning a objc interface by-value.
    if (RetTy->isObjCInterfaceType()) {
      Diag(ParamInfo.getSourceRange().getBegin(),
           diag::err_object_cannot_be_passed_returned_by_value) << 0 << RetTy;
      return;
    }
    return;
  }

  // Analyze arguments to block.
  assert(ParamInfo.getTypeObject(0).Kind == DeclaratorChunk::Function &&
         "Not a function declarator!");
  DeclaratorChunk::FunctionTypeInfo &FTI = ParamInfo.getTypeObject(0).Fun;

  CurBlock->hasPrototype = FTI.hasPrototype;
  CurBlock->isVariadic = true;

  // Check for C99 6.7.5.3p10 - foo(void) is a non-varargs function that takes
  // no arguments, not a function that takes a single void argument.
  if (FTI.hasPrototype &&
      FTI.NumArgs == 1 && !FTI.isVariadic && FTI.ArgInfo[0].Ident == 0 &&
     (!FTI.ArgInfo[0].Param.getAs<ParmVarDecl>()->getType().getCVRQualifiers()&&
        FTI.ArgInfo[0].Param.getAs<ParmVarDecl>()->getType()->isVoidType())) {
    // empty arg list, don't push any params.
    CurBlock->isVariadic = false;
  } else if (FTI.hasPrototype) {
    for (unsigned i = 0, e = FTI.NumArgs; i != e; ++i)
      CurBlock->Params.push_back(FTI.ArgInfo[i].Param.getAs<ParmVarDecl>());
    CurBlock->isVariadic = FTI.isVariadic;
  }
  CurBlock->TheDecl->setParams(Context, CurBlock->Params.data(),
                               CurBlock->Params.size());
  CurBlock->TheDecl->setIsVariadic(CurBlock->isVariadic);
  ProcessDeclAttributes(CurScope, CurBlock->TheDecl, ParamInfo);
  for (BlockDecl::param_iterator AI = CurBlock->TheDecl->param_begin(),
       E = CurBlock->TheDecl->param_end(); AI != E; ++AI)
    // If this has an identifier, add it to the scope stack.
    if ((*AI)->getIdentifier())
      PushOnScopeChains(*AI, CurBlock->TheScope);

  // Check for a valid sentinel attribute on this block.
  if (!CurBlock->isVariadic && 
      CurBlock->TheDecl->getAttr<SentinelAttr>()) {
    Diag(ParamInfo.getAttributes()->getLoc(), 
         diag::warn_attribute_sentinel_not_variadic) << 1;
    // FIXME: remove the attribute.
  }
  
  // Analyze the return type.
  QualType T = GetTypeForDeclarator(ParamInfo, CurScope);
  QualType RetTy = T->getAsFunctionType()->getResultType();
  
  // Do not allow returning a objc interface by-value.
  if (RetTy->isObjCInterfaceType()) {
    Diag(ParamInfo.getSourceRange().getBegin(),
         diag::err_object_cannot_be_passed_returned_by_value) << 0 << RetTy;
  } else if (!RetTy->isDependentType())
    CurBlock->ReturnType = RetTy;
}

/// ActOnBlockError - If there is an error parsing a block, this callback
/// is invoked to pop the information about the block from the action impl.
void Sema::ActOnBlockError(SourceLocation CaretLoc, Scope *CurScope) {
  // Ensure that CurBlock is deleted.
  llvm::OwningPtr<BlockSemaInfo> CC(CurBlock);

  CurFunctionNeedsScopeChecking = CurBlock->SavedFunctionNeedsScopeChecking;

  // Pop off CurBlock, handle nested blocks.
  PopDeclContext();
  CurBlock = CurBlock->PrevBlockInfo;
  // FIXME: Delete the ParmVarDecl objects as well???
}

/// ActOnBlockStmtExpr - This is called when the body of a block statement
/// literal was successfully completed.  ^(int x){...}
Sema::OwningExprResult Sema::ActOnBlockStmtExpr(SourceLocation CaretLoc,
                                                StmtArg body, Scope *CurScope) {
  // If blocks are disabled, emit an error.
  if (!LangOpts.Blocks)
    Diag(CaretLoc, diag::err_blocks_disable);
  
  // Ensure that CurBlock is deleted.
  llvm::OwningPtr<BlockSemaInfo> BSI(CurBlock);

  PopDeclContext();

  // Pop off CurBlock, handle nested blocks.
  CurBlock = CurBlock->PrevBlockInfo;

  QualType RetTy = Context.VoidTy;
  if (!BSI->ReturnType.isNull())
    RetTy = BSI->ReturnType;

  llvm::SmallVector<QualType, 8> ArgTypes;
  for (unsigned i = 0, e = BSI->Params.size(); i != e; ++i)
    ArgTypes.push_back(BSI->Params[i]->getType());

  bool NoReturn = BSI->TheDecl->getAttr<NoReturnAttr>();
  QualType BlockTy;
  if (!BSI->hasPrototype)
    BlockTy = Context.getFunctionType(RetTy, 0, 0, false, 0, false, false, 0, 0,
                                      NoReturn);
  else
    BlockTy = Context.getFunctionType(RetTy, ArgTypes.data(), ArgTypes.size(),
                                      BSI->isVariadic, 0, false, false, 0, 0,
                                      NoReturn);

  // FIXME: Check that return/parameter types are complete/non-abstract
  DiagnoseUnusedParameters(BSI->Params.begin(), BSI->Params.end());
  BlockTy = Context.getBlockPointerType(BlockTy);

  // If needed, diagnose invalid gotos and switches in the block.
  if (CurFunctionNeedsScopeChecking)
    DiagnoseInvalidJumps(static_cast<CompoundStmt*>(body.get()));
  CurFunctionNeedsScopeChecking = BSI->SavedFunctionNeedsScopeChecking;
  
  BSI->TheDecl->setBody(body.takeAs<CompoundStmt>());
  CheckFallThroughForBlock(BlockTy, BSI->TheDecl->getBody());
  return Owned(new (Context) BlockExpr(BSI->TheDecl, BlockTy,
                                       BSI->hasBlockDeclRefExprs));
}

Sema::OwningExprResult Sema::ActOnVAArg(SourceLocation BuiltinLoc,
                                        ExprArg expr, TypeTy *type,
                                        SourceLocation RPLoc) {
  QualType T = GetTypeFromParser(type);
  Expr *E = static_cast<Expr*>(expr.get());
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

  expr.release();
  return Owned(new (Context) VAArgExpr(BuiltinLoc, E, T.getNonReferenceType(),
                                       RPLoc));
}

Sema::OwningExprResult Sema::ActOnGNUNullExpr(SourceLocation TokenLoc) {
  // The type of __null will be int or long, depending on the size of
  // pointers on the target.
  QualType Ty;
  if (Context.Target.getPointerWidth(0) == Context.Target.getIntWidth())
    Ty = Context.IntTy;
  else
    Ty = Context.LongTy;

  return Owned(new (Context) GNUNullExpr(Ty, TokenLoc));
}

bool Sema::DiagnoseAssignmentResult(AssignConvertType ConvTy,
                                    SourceLocation Loc,
                                    QualType DstType, QualType SrcType,
                                    Expr *SrcExpr, const char *Flavor) {
  // Decode the result (notice that AST's are still created for extensions).
  bool isInvalid = false;
  unsigned DiagKind;
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

  Diag(Loc, DiagKind) << DstType << SrcType << Flavor
    << SrcExpr->getSourceRange();
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

Sema::ExpressionEvaluationContext 
Sema::PushExpressionEvaluationContext(ExpressionEvaluationContext NewContext) { 
  // Introduce a new set of potentially referenced declarations to the stack.
  if (NewContext == PotentiallyPotentiallyEvaluated)
    PotentiallyReferencedDeclStack.push_back(PotentiallyReferencedDecls());
  
  std::swap(ExprEvalContext, NewContext);
  return NewContext;
}

void 
Sema::PopExpressionEvaluationContext(ExpressionEvaluationContext OldContext,
                                     ExpressionEvaluationContext NewContext) {
  ExprEvalContext = NewContext;

  if (OldContext == PotentiallyPotentiallyEvaluated) {
    // Mark any remaining declarations in the current position of the stack
    // as "referenced". If they were not meant to be referenced, semantic
    // analysis would have eliminated them (e.g., in ActOnCXXTypeId).
    PotentiallyReferencedDecls RemainingDecls;
    RemainingDecls.swap(PotentiallyReferencedDeclStack.back());
    PotentiallyReferencedDeclStack.pop_back();
    
    for (PotentiallyReferencedDecls::iterator I = RemainingDecls.begin(),
                                           IEnd = RemainingDecls.end();
         I != IEnd; ++I)
      MarkDeclarationReferenced(I->first, I->second);
  }
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
  
  if (D->isUsed())
    return;
  
  // Mark a parameter declaration "used", regardless of whether we're in a
  // template or not.
  if (isa<ParmVarDecl>(D))
    D->setUsed(true);
  
  // Do not mark anything as "used" within a dependent context; wait for
  // an instantiation.
  if (CurContext->isDependentContext())
    return;
  
  switch (ExprEvalContext) {
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
      PotentiallyReferencedDeclStack.back().push_back(std::make_pair(Loc, D));
      return;
  }
      
  // Note that this declaration has been used.
  if (CXXConstructorDecl *Constructor = dyn_cast<CXXConstructorDecl>(D)) {
    unsigned TypeQuals;
    if (Constructor->isImplicit() && Constructor->isDefaultConstructor()) {
        if (!Constructor->isUsed())
          DefineImplicitDefaultConstructor(Loc, Constructor);
    } else if (Constructor->isImplicit() && 
               Constructor->isCopyConstructor(Context, TypeQuals)) {
      if (!Constructor->isUsed())
        DefineImplicitCopyConstructor(Loc, Constructor, TypeQuals);
    }
  } else if (CXXDestructorDecl *Destructor = dyn_cast<CXXDestructorDecl>(D)) {
    if (Destructor->isImplicit() && !Destructor->isUsed())
      DefineImplicitDestructor(Loc, Destructor);
    
  } else if (CXXMethodDecl *MethodDecl = dyn_cast<CXXMethodDecl>(D)) {
    if (MethodDecl->isImplicit() && MethodDecl->isOverloadedOperator() &&
        MethodDecl->getOverloadedOperator() == OO_Equal) {
      if (!MethodDecl->isUsed())
        DefineImplicitOverloadedAssign(Loc, MethodDecl);
    }
  }
  if (FunctionDecl *Function = dyn_cast<FunctionDecl>(D)) {
    // Implicit instantiation of function templates and member functions of 
    // class templates.
    if (!Function->getBody()) {
      // FIXME: distinguish between implicit instantiations of function
      // templates and explicit specializations (the latter don't get
      // instantiated, naturally).
      if (Function->getInstantiatedFromMemberFunction() ||
          Function->getPrimaryTemplate())
        PendingImplicitInstantiations.push_back(std::make_pair(Function, Loc));
    }
    
    
    // FIXME: keep track of references to static functions
    Function->setUsed(true);
    return;
  }
  
  if (VarDecl *Var = dyn_cast<VarDecl>(D)) {
    // Implicit instantiation of static data members of class templates.
    // FIXME: distinguish between implicit instantiations (which we need to
    // actually instantiate) and explicit specializations.
    if (Var->isStaticDataMember() && 
        Var->getInstantiatedFromStaticDataMember())
      PendingImplicitInstantiations.push_back(std::make_pair(Var, Loc));
    
    // FIXME: keep track of references to static data?

    D->setUsed(true);
    return;
}
}

