//===--- SemaDeclAttr.cpp - Declaration Attribute Handling ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements decl-related attribute processing.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaInternal.h"
#include "TargetAttributesSema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Mangle.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/DelayedDiagnostic.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Scope.h"
#include "llvm/ADT/StringExtras.h"
using namespace clang;
using namespace sema;

/// These constants match the enumerated choices of
/// warn_attribute_wrong_decl_type and err_attribute_wrong_decl_type.
enum AttributeDeclKind {
  ExpectedFunction,
  ExpectedUnion,
  ExpectedVariableOrFunction,
  ExpectedFunctionOrMethod,
  ExpectedParameter,
  ExpectedFunctionMethodOrBlock,
  ExpectedFunctionMethodOrClass,
  ExpectedFunctionMethodOrParameter,
  ExpectedClass,
  ExpectedVariable,
  ExpectedMethod,
  ExpectedVariableFunctionOrLabel,
  ExpectedFieldOrGlobalVar,
  ExpectedStruct,
  ExpectedVariableFunctionOrTag,
  ExpectedTLSVar,
  ExpectedVariableOrField,
  ExpectedVariableFieldOrTag,
  ExpectedTypeOrNamespace
};

//===----------------------------------------------------------------------===//
//  Helper functions
//===----------------------------------------------------------------------===//

static const FunctionType *getFunctionType(const Decl *D,
                                           bool blocksToo = true) {
  QualType Ty;
  if (const ValueDecl *decl = dyn_cast<ValueDecl>(D))
    Ty = decl->getType();
  else if (const FieldDecl *decl = dyn_cast<FieldDecl>(D))
    Ty = decl->getType();
  else if (const TypedefNameDecl* decl = dyn_cast<TypedefNameDecl>(D))
    Ty = decl->getUnderlyingType();
  else
    return 0;

  if (Ty->isFunctionPointerType())
    Ty = Ty->getAs<PointerType>()->getPointeeType();
  else if (blocksToo && Ty->isBlockPointerType())
    Ty = Ty->getAs<BlockPointerType>()->getPointeeType();

  return Ty->getAs<FunctionType>();
}

// FIXME: We should provide an abstraction around a method or function
// to provide the following bits of information.

/// isFunction - Return true if the given decl has function
/// type (function or function-typed variable).
static bool isFunction(const Decl *D) {
  return getFunctionType(D, false) != NULL;
}

/// isFunctionOrMethod - Return true if the given decl has function
/// type (function or function-typed variable) or an Objective-C
/// method.
static bool isFunctionOrMethod(const Decl *D) {
  return isFunction(D) || isa<ObjCMethodDecl>(D);
}

/// isFunctionOrMethodOrBlock - Return true if the given decl has function
/// type (function or function-typed variable) or an Objective-C
/// method or a block.
static bool isFunctionOrMethodOrBlock(const Decl *D) {
  if (isFunctionOrMethod(D))
    return true;
  // check for block is more involved.
  if (const VarDecl *V = dyn_cast<VarDecl>(D)) {
    QualType Ty = V->getType();
    return Ty->isBlockPointerType();
  }
  return isa<BlockDecl>(D);
}

/// Return true if the given decl has a declarator that should have
/// been processed by Sema::GetTypeForDeclarator.
static bool hasDeclarator(const Decl *D) {
  // In some sense, TypedefDecl really *ought* to be a DeclaratorDecl.
  return isa<DeclaratorDecl>(D) || isa<BlockDecl>(D) || isa<TypedefNameDecl>(D) ||
         isa<ObjCPropertyDecl>(D);
}

/// hasFunctionProto - Return true if the given decl has a argument
/// information. This decl should have already passed
/// isFunctionOrMethod or isFunctionOrMethodOrBlock.
static bool hasFunctionProto(const Decl *D) {
  if (const FunctionType *FnTy = getFunctionType(D))
    return isa<FunctionProtoType>(FnTy);
  else {
    assert(isa<ObjCMethodDecl>(D) || isa<BlockDecl>(D));
    return true;
  }
}

/// getFunctionOrMethodNumArgs - Return number of function or method
/// arguments. It is an error to call this on a K&R function (use
/// hasFunctionProto first).
static unsigned getFunctionOrMethodNumArgs(const Decl *D) {
  if (const FunctionType *FnTy = getFunctionType(D))
    return cast<FunctionProtoType>(FnTy)->getNumArgs();
  if (const BlockDecl *BD = dyn_cast<BlockDecl>(D))
    return BD->getNumParams();
  return cast<ObjCMethodDecl>(D)->param_size();
}

static QualType getFunctionOrMethodArgType(const Decl *D, unsigned Idx) {
  if (const FunctionType *FnTy = getFunctionType(D))
    return cast<FunctionProtoType>(FnTy)->getArgType(Idx);
  if (const BlockDecl *BD = dyn_cast<BlockDecl>(D))
    return BD->getParamDecl(Idx)->getType();

  return cast<ObjCMethodDecl>(D)->param_begin()[Idx]->getType();
}

static QualType getFunctionOrMethodResultType(const Decl *D) {
  if (const FunctionType *FnTy = getFunctionType(D))
    return cast<FunctionProtoType>(FnTy)->getResultType();
  return cast<ObjCMethodDecl>(D)->getResultType();
}

static bool isFunctionOrMethodVariadic(const Decl *D) {
  if (const FunctionType *FnTy = getFunctionType(D)) {
    const FunctionProtoType *proto = cast<FunctionProtoType>(FnTy);
    return proto->isVariadic();
  } else if (const BlockDecl *BD = dyn_cast<BlockDecl>(D))
    return BD->isVariadic();
  else {
    return cast<ObjCMethodDecl>(D)->isVariadic();
  }
}

static bool isInstanceMethod(const Decl *D) {
  if (const CXXMethodDecl *MethodDecl = dyn_cast<CXXMethodDecl>(D))
    return MethodDecl->isInstance();
  return false;
}

static inline bool isNSStringType(QualType T, ASTContext &Ctx) {
  const ObjCObjectPointerType *PT = T->getAs<ObjCObjectPointerType>();
  if (!PT)
    return false;

  ObjCInterfaceDecl *Cls = PT->getObjectType()->getInterface();
  if (!Cls)
    return false;

  IdentifierInfo* ClsName = Cls->getIdentifier();

  // FIXME: Should we walk the chain of classes?
  return ClsName == &Ctx.Idents.get("NSString") ||
         ClsName == &Ctx.Idents.get("NSMutableString");
}

static inline bool isCFStringType(QualType T, ASTContext &Ctx) {
  const PointerType *PT = T->getAs<PointerType>();
  if (!PT)
    return false;

  const RecordType *RT = PT->getPointeeType()->getAs<RecordType>();
  if (!RT)
    return false;

  const RecordDecl *RD = RT->getDecl();
  if (RD->getTagKind() != TTK_Struct)
    return false;

  return RD->getIdentifier() == &Ctx.Idents.get("__CFString");
}

/// \brief Check if the attribute has exactly as many args as Num. May
/// output an error.
static bool checkAttributeNumArgs(Sema &S, const AttributeList &Attr,
                                  unsigned int Num) {
  if (Attr.getNumArgs() != Num) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << Num;
    return false;
  }

  return true;
}


/// \brief Check if the attribute has at least as many args as Num. May
/// output an error.
static bool checkAttributeAtLeastNumArgs(Sema &S, const AttributeList &Attr,
                                  unsigned int Num) {
  if (Attr.getNumArgs() < Num) {
    S.Diag(Attr.getLoc(), diag::err_attribute_too_few_arguments) << Num;
    return false;
  }

  return true;
}

/// \brief Check if IdxExpr is a valid argument index for a function or
/// instance method D.  May output an error.
///
/// \returns true if IdxExpr is a valid index.
static bool checkFunctionOrMethodArgumentIndex(Sema &S, const Decl *D,
                                               StringRef AttrName,
                                               SourceLocation AttrLoc,
                                               unsigned AttrArgNum,
                                               const Expr *IdxExpr,
                                               uint64_t &Idx)
{
  assert(isFunctionOrMethod(D) && hasFunctionProto(D));

  // In C++ the implicit 'this' function parameter also counts.
  // Parameters are counted from one.
  const bool HasImplicitThisParam = isInstanceMethod(D);
  const unsigned NumArgs = getFunctionOrMethodNumArgs(D) + HasImplicitThisParam;
  const unsigned FirstIdx = 1;

  llvm::APSInt IdxInt;
  if (IdxExpr->isTypeDependent() || IdxExpr->isValueDependent() ||
      !IdxExpr->isIntegerConstantExpr(IdxInt, S.Context)) {
    S.Diag(AttrLoc, diag::err_attribute_argument_n_not_int)
      << AttrName << AttrArgNum << IdxExpr->getSourceRange();
    return false;
  }

  Idx = IdxInt.getLimitedValue();
  if (Idx < FirstIdx || (!isFunctionOrMethodVariadic(D) && Idx > NumArgs)) {
    S.Diag(AttrLoc, diag::err_attribute_argument_out_of_bounds)
      << AttrName << AttrArgNum << IdxExpr->getSourceRange();
    return false;
  }
  Idx--; // Convert to zero-based.
  if (HasImplicitThisParam) {
    if (Idx == 0) {
      S.Diag(AttrLoc,
             diag::err_attribute_invalid_implicit_this_argument)
        << AttrName << IdxExpr->getSourceRange();
      return false;
    }
    --Idx;
  }

  return true;
}

///
/// \brief Check if passed in Decl is a field or potentially shared global var
/// \return true if the Decl is a field or potentially shared global variable
///
static bool mayBeSharedVariable(const Decl *D) {
  if (isa<FieldDecl>(D))
    return true;
  if (const VarDecl *vd = dyn_cast<VarDecl>(D))
    return vd->hasGlobalStorage() && !vd->getTLSKind();

  return false;
}

/// \brief Check if the passed-in expression is of type int or bool.
static bool isIntOrBool(Expr *Exp) {
  QualType QT = Exp->getType();
  return QT->isBooleanType() || QT->isIntegerType();
}


// Check to see if the type is a smart pointer of some kind.  We assume
// it's a smart pointer if it defines both operator-> and operator*.
static bool threadSafetyCheckIsSmartPointer(Sema &S, const RecordType* RT) {
  DeclContextLookupConstResult Res1 = RT->getDecl()->lookup(
    S.Context.DeclarationNames.getCXXOperatorName(OO_Star));
  if (Res1.empty())
    return false;

  DeclContextLookupConstResult Res2 = RT->getDecl()->lookup(
    S.Context.DeclarationNames.getCXXOperatorName(OO_Arrow));
  if (Res2.empty())
    return false;

  return true;
}

/// \brief Check if passed in Decl is a pointer type.
/// Note that this function may produce an error message.
/// \return true if the Decl is a pointer type; false otherwise
static bool threadSafetyCheckIsPointer(Sema &S, const Decl *D,
                                       const AttributeList &Attr) {
  if (const ValueDecl *vd = dyn_cast<ValueDecl>(D)) {
    QualType QT = vd->getType();
    if (QT->isAnyPointerType())
      return true;

    if (const RecordType *RT = QT->getAs<RecordType>()) {
      // If it's an incomplete type, it could be a smart pointer; skip it.
      // (We don't want to force template instantiation if we can avoid it,
      // since that would alter the order in which templates are instantiated.)
      if (RT->isIncompleteType())
        return true;

      if (threadSafetyCheckIsSmartPointer(S, RT))
        return true;
    }

    S.Diag(Attr.getLoc(), diag::warn_thread_attribute_decl_not_pointer)
      << Attr.getName()->getName() << QT;
  } else {
    S.Diag(Attr.getLoc(), diag::err_attribute_can_be_applied_only_to_value_decl)
      << Attr.getName();
  }
  return false;
}

/// \brief Checks that the passed in QualType either is of RecordType or points
/// to RecordType. Returns the relevant RecordType, null if it does not exit.
static const RecordType *getRecordType(QualType QT) {
  if (const RecordType *RT = QT->getAs<RecordType>())
    return RT;

  // Now check if we point to record type.
  if (const PointerType *PT = QT->getAs<PointerType>())
    return PT->getPointeeType()->getAs<RecordType>();

  return 0;
}


static bool checkBaseClassIsLockableCallback(const CXXBaseSpecifier *Specifier,
                                             CXXBasePath &Path, void *Unused) {
  const RecordType *RT = Specifier->getType()->getAs<RecordType>();
  if (RT->getDecl()->getAttr<LockableAttr>())
    return true;
  return false;
}


/// \brief Thread Safety Analysis: Checks that the passed in RecordType
/// resolves to a lockable object.
static void checkForLockableRecord(Sema &S, Decl *D, const AttributeList &Attr,
                                   QualType Ty) {
  const RecordType *RT = getRecordType(Ty);

  // Warn if could not get record type for this argument.
  if (!RT) {
    S.Diag(Attr.getLoc(), diag::warn_thread_attribute_argument_not_class)
      << Attr.getName() << Ty.getAsString();
    return;
  }

  // Don't check for lockable if the class hasn't been defined yet.
  if (RT->isIncompleteType())
    return;

  // Allow smart pointers to be used as lockable objects.
  // FIXME -- Check the type that the smart pointer points to.
  if (threadSafetyCheckIsSmartPointer(S, RT))
    return;

  // Check if the type is lockable.
  RecordDecl *RD = RT->getDecl();
  if (RD->getAttr<LockableAttr>())
    return;

  // Else check if any base classes are lockable.
  if (CXXRecordDecl *CRD = dyn_cast<CXXRecordDecl>(RD)) {
    CXXBasePaths BPaths(false, false);
    if (CRD->lookupInBases(checkBaseClassIsLockableCallback, 0, BPaths))
      return;
  }

  S.Diag(Attr.getLoc(), diag::warn_thread_attribute_argument_not_lockable)
    << Attr.getName() << Ty.getAsString();
}

/// \brief Thread Safety Analysis: Checks that all attribute arguments, starting
/// from Sidx, resolve to a lockable object.
/// \param Sidx The attribute argument index to start checking with.
/// \param ParamIdxOk Whether an argument can be indexing into a function
/// parameter list.
static void checkAttrArgsAreLockableObjs(Sema &S, Decl *D,
                                         const AttributeList &Attr,
                                         SmallVectorImpl<Expr*> &Args,
                                         int Sidx = 0,
                                         bool ParamIdxOk = false) {
  for(unsigned Idx = Sidx; Idx < Attr.getNumArgs(); ++Idx) {
    Expr *ArgExp = Attr.getArg(Idx);

    if (ArgExp->isTypeDependent()) {
      // FIXME -- need to check this again on template instantiation
      Args.push_back(ArgExp);
      continue;
    }

    if (StringLiteral *StrLit = dyn_cast<StringLiteral>(ArgExp)) {
      if (StrLit->getLength() == 0 ||
          StrLit->getString() == StringRef("*")) {
        // Pass empty strings to the analyzer without warnings.
        // Treat "*" as the universal lock.
        Args.push_back(ArgExp);
        continue;
      }

      // We allow constant strings to be used as a placeholder for expressions
      // that are not valid C++ syntax, but warn that they are ignored.
      S.Diag(Attr.getLoc(), diag::warn_thread_attribute_ignored) <<
        Attr.getName();
      Args.push_back(ArgExp);
      continue;
    }

    QualType ArgTy = ArgExp->getType();

    // A pointer to member expression of the form  &MyClass::mu is treated
    // specially -- we need to look at the type of the member.
    if (UnaryOperator *UOp = dyn_cast<UnaryOperator>(ArgExp))
      if (UOp->getOpcode() == UO_AddrOf)
        if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(UOp->getSubExpr()))
          if (DRE->getDecl()->isCXXInstanceMember())
            ArgTy = DRE->getDecl()->getType();

    // First see if we can just cast to record type, or point to record type.
    const RecordType *RT = getRecordType(ArgTy);

    // Now check if we index into a record type function param.
    if(!RT && ParamIdxOk) {
      FunctionDecl *FD = dyn_cast<FunctionDecl>(D);
      IntegerLiteral *IL = dyn_cast<IntegerLiteral>(ArgExp);
      if(FD && IL) {
        unsigned int NumParams = FD->getNumParams();
        llvm::APInt ArgValue = IL->getValue();
        uint64_t ParamIdxFromOne = ArgValue.getZExtValue();
        uint64_t ParamIdxFromZero = ParamIdxFromOne - 1;
        if(!ArgValue.isStrictlyPositive() || ParamIdxFromOne > NumParams) {
          S.Diag(Attr.getLoc(), diag::err_attribute_argument_out_of_range)
            << Attr.getName() << Idx + 1 << NumParams;
          continue;
        }
        ArgTy = FD->getParamDecl(ParamIdxFromZero)->getType();
      }
    }

    checkForLockableRecord(S, D, Attr, ArgTy);

    Args.push_back(ArgExp);
  }
}

//===----------------------------------------------------------------------===//
// Attribute Implementations
//===----------------------------------------------------------------------===//

// FIXME: All this manual attribute parsing code is gross. At the
// least add some helper functions to check most argument patterns (#
// and types of args).

enum ThreadAttributeDeclKind {
  ThreadExpectedFieldOrGlobalVar,
  ThreadExpectedFunctionOrMethod,
  ThreadExpectedClassOrStruct
};

static bool checkGuardedVarAttrCommon(Sema &S, Decl *D,
                                      const AttributeList &Attr) {
  assert(!Attr.isInvalid());

  if (!checkAttributeNumArgs(S, Attr, 0))
    return false;

  // D must be either a member field or global (potentially shared) variable.
  if (!mayBeSharedVariable(D)) {
    S.Diag(Attr.getLoc(), diag::warn_thread_attribute_wrong_decl_type)
      << Attr.getName() << ThreadExpectedFieldOrGlobalVar;
    return false;
  }

  return true;
}

static void handleGuardedVarAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (!checkGuardedVarAttrCommon(S, D, Attr))
    return;

  D->addAttr(::new (S.Context)
             GuardedVarAttr(Attr.getRange(), S.Context,
                            Attr.getAttributeSpellingListIndex()));
}

static void handlePtGuardedVarAttr(Sema &S, Decl *D,
                                   const AttributeList &Attr) {
  if (!checkGuardedVarAttrCommon(S, D, Attr))
    return;

  if (!threadSafetyCheckIsPointer(S, D, Attr))
    return;

  D->addAttr(::new (S.Context)
             PtGuardedVarAttr(Attr.getRange(), S.Context,
                              Attr.getAttributeSpellingListIndex()));
}

static bool checkGuardedByAttrCommon(Sema &S, Decl *D,
                                     const AttributeList &Attr,
                                     Expr* &Arg) {
  assert(!Attr.isInvalid());

  if (!checkAttributeNumArgs(S, Attr, 1))
    return false;

  // D must be either a member field or global (potentially shared) variable.
  if (!mayBeSharedVariable(D)) {
    S.Diag(Attr.getLoc(), diag::warn_thread_attribute_wrong_decl_type)
      << Attr.getName() << ThreadExpectedFieldOrGlobalVar;
    return false;
  }

  SmallVector<Expr*, 1> Args;
  // check that all arguments are lockable objects
  checkAttrArgsAreLockableObjs(S, D, Attr, Args);
  unsigned Size = Args.size();
  if (Size != 1)
    return false;

  Arg = Args[0];

  return true;
}

static void handleGuardedByAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  Expr *Arg = 0;
  if (!checkGuardedByAttrCommon(S, D, Attr, Arg))
    return;

  D->addAttr(::new (S.Context) GuardedByAttr(Attr.getRange(), S.Context, Arg));
}

static void handlePtGuardedByAttr(Sema &S, Decl *D,
                                  const AttributeList &Attr) {
  Expr *Arg = 0;
  if (!checkGuardedByAttrCommon(S, D, Attr, Arg))
    return;

  if (!threadSafetyCheckIsPointer(S, D, Attr))
    return;

  D->addAttr(::new (S.Context) PtGuardedByAttr(Attr.getRange(),
                                               S.Context, Arg));
}

static bool checkLockableAttrCommon(Sema &S, Decl *D,
                                    const AttributeList &Attr) {
  assert(!Attr.isInvalid());

  if (!checkAttributeNumArgs(S, Attr, 0))
    return false;

  // FIXME: Lockable structs for C code.
  if (!isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_thread_attribute_wrong_decl_type)
      << Attr.getName() << ThreadExpectedClassOrStruct;
    return false;
  }

  return true;
}

static void handleLockableAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (!checkLockableAttrCommon(S, D, Attr))
    return;

  D->addAttr(::new (S.Context) LockableAttr(Attr.getRange(), S.Context));
}

static void handleScopedLockableAttr(Sema &S, Decl *D,
                             const AttributeList &Attr) {
  if (!checkLockableAttrCommon(S, D, Attr))
    return;

  D->addAttr(::new (S.Context)
             ScopedLockableAttr(Attr.getRange(), S.Context,
                                Attr.getAttributeSpellingListIndex()));
}

static void handleNoThreadSafetyAnalysis(Sema &S, Decl *D,
                                         const AttributeList &Attr) {
  assert(!Attr.isInvalid());

  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  if (!isa<FunctionDecl>(D) && !isa<FunctionTemplateDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_thread_attribute_wrong_decl_type)
      << Attr.getName() << ThreadExpectedFunctionOrMethod;
    return;
  }

  D->addAttr(::new (S.Context) NoThreadSafetyAnalysisAttr(Attr.getRange(),
                                                          S.Context));
}

static void handleNoSanitizeAddressAttr(Sema &S, Decl *D,
                                      const AttributeList &Attr) {
  assert(!Attr.isInvalid());

  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  if (!isa<FunctionDecl>(D) && !isa<FunctionTemplateDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrMethod;
    return;
  }

  D->addAttr(::new (S.Context)
             NoSanitizeAddressAttr(Attr.getRange(), S.Context,
                                   Attr.getAttributeSpellingListIndex()));
}

static void handleNoSanitizeMemory(Sema &S, Decl *D,
                                   const AttributeList &Attr) {
  assert(!Attr.isInvalid());

  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  if (!isa<FunctionDecl>(D) && !isa<FunctionTemplateDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrMethod;
    return;
  }

  D->addAttr(::new (S.Context) NoSanitizeMemoryAttr(Attr.getRange(),
                                                         S.Context));
}

static void handleNoSanitizeThread(Sema &S, Decl *D,
                                   const AttributeList &Attr) {
  assert(!Attr.isInvalid());

  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  if (!isa<FunctionDecl>(D) && !isa<FunctionTemplateDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrMethod;
    return;
  }

  D->addAttr(::new (S.Context) NoSanitizeThreadAttr(Attr.getRange(),
                                                    S.Context));
}

static bool checkAcquireOrderAttrCommon(Sema &S, Decl *D,
                                        const AttributeList &Attr,
                                        SmallVector<Expr*, 1> &Args) {
  assert(!Attr.isInvalid());

  if (!checkAttributeAtLeastNumArgs(S, Attr, 1))
    return false;

  // D must be either a member field or global (potentially shared) variable.
  ValueDecl *VD = dyn_cast<ValueDecl>(D);
  if (!VD || !mayBeSharedVariable(D)) {
    S.Diag(Attr.getLoc(), diag::warn_thread_attribute_wrong_decl_type)
      << Attr.getName() << ThreadExpectedFieldOrGlobalVar;
    return false;
  }

  // Check that this attribute only applies to lockable types.
  QualType QT = VD->getType();
  if (!QT->isDependentType()) {
    const RecordType *RT = getRecordType(QT);
    if (!RT || !RT->getDecl()->getAttr<LockableAttr>()) {
      S.Diag(Attr.getLoc(), diag::warn_thread_attribute_decl_not_lockable)
        << Attr.getName();
      return false;
    }
  }

  // Check that all arguments are lockable objects.
  checkAttrArgsAreLockableObjs(S, D, Attr, Args);
  if (Args.empty())
    return false;

  return true;
}

static void handleAcquiredAfterAttr(Sema &S, Decl *D,
                                    const AttributeList &Attr) {
  SmallVector<Expr*, 1> Args;
  if (!checkAcquireOrderAttrCommon(S, D, Attr, Args))
    return;

  Expr **StartArg = &Args[0];
  D->addAttr(::new (S.Context)
             AcquiredAfterAttr(Attr.getRange(), S.Context,
                               StartArg, Args.size(),
                               Attr.getAttributeSpellingListIndex()));
}

static void handleAcquiredBeforeAttr(Sema &S, Decl *D,
                                     const AttributeList &Attr) {
  SmallVector<Expr*, 1> Args;
  if (!checkAcquireOrderAttrCommon(S, D, Attr, Args))
    return;

  Expr **StartArg = &Args[0];
  D->addAttr(::new (S.Context)
             AcquiredBeforeAttr(Attr.getRange(), S.Context,
                                StartArg, Args.size(),
                                Attr.getAttributeSpellingListIndex()));
}

static bool checkLockFunAttrCommon(Sema &S, Decl *D,
                                   const AttributeList &Attr,
                                   SmallVector<Expr*, 1> &Args) {
  assert(!Attr.isInvalid());

  // zero or more arguments ok

  // check that the attribute is applied to a function
  if (!isa<FunctionDecl>(D) && !isa<FunctionTemplateDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_thread_attribute_wrong_decl_type)
      << Attr.getName() << ThreadExpectedFunctionOrMethod;
    return false;
  }

  // check that all arguments are lockable objects
  checkAttrArgsAreLockableObjs(S, D, Attr, Args, 0, /*ParamIdxOk=*/true);

  return true;
}

static void handleSharedLockFunctionAttr(Sema &S, Decl *D,
                                         const AttributeList &Attr) {
  SmallVector<Expr*, 1> Args;
  if (!checkLockFunAttrCommon(S, D, Attr, Args))
    return;

  unsigned Size = Args.size();
  Expr **StartArg = Size == 0 ? 0 : &Args[0];
  D->addAttr(::new (S.Context)
             SharedLockFunctionAttr(Attr.getRange(), S.Context, StartArg, Size,
                                    Attr.getAttributeSpellingListIndex()));
}

static void handleExclusiveLockFunctionAttr(Sema &S, Decl *D,
                                            const AttributeList &Attr) {
  SmallVector<Expr*, 1> Args;
  if (!checkLockFunAttrCommon(S, D, Attr, Args))
    return;

  unsigned Size = Args.size();
  Expr **StartArg = Size == 0 ? 0 : &Args[0];
  D->addAttr(::new (S.Context)
             ExclusiveLockFunctionAttr(Attr.getRange(), S.Context,
                                       StartArg, Size,
                                       Attr.getAttributeSpellingListIndex()));
}

static bool checkTryLockFunAttrCommon(Sema &S, Decl *D,
                                      const AttributeList &Attr,
                                      SmallVector<Expr*, 2> &Args) {
  assert(!Attr.isInvalid());

  if (!checkAttributeAtLeastNumArgs(S, Attr, 1))
    return false;

  if (!isa<FunctionDecl>(D) && !isa<FunctionTemplateDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_thread_attribute_wrong_decl_type)
      << Attr.getName() << ThreadExpectedFunctionOrMethod;
    return false;
  }

  if (!isIntOrBool(Attr.getArg(0))) {
    S.Diag(Attr.getLoc(), diag::err_attribute_first_argument_not_int_or_bool)
      << Attr.getName();
    return false;
  }

  // check that all arguments are lockable objects
  checkAttrArgsAreLockableObjs(S, D, Attr, Args, 1);

  return true;
}

static void handleSharedTrylockFunctionAttr(Sema &S, Decl *D,
                                            const AttributeList &Attr) {
  SmallVector<Expr*, 2> Args;
  if (!checkTryLockFunAttrCommon(S, D, Attr, Args))
    return;

  unsigned Size = Args.size();
  Expr **StartArg = Size == 0 ? 0 : &Args[0];
  D->addAttr(::new (S.Context)
             SharedTrylockFunctionAttr(Attr.getRange(), S.Context,
                                       Attr.getArg(0), StartArg, Size,
                                       Attr.getAttributeSpellingListIndex()));
}

static void handleExclusiveTrylockFunctionAttr(Sema &S, Decl *D,
                                               const AttributeList &Attr) {
  SmallVector<Expr*, 2> Args;
  if (!checkTryLockFunAttrCommon(S, D, Attr, Args))
    return;

  unsigned Size = Args.size();
  Expr **StartArg = Size == 0 ? 0 : &Args[0];
  D->addAttr(::new (S.Context)
             ExclusiveTrylockFunctionAttr(Attr.getRange(), S.Context,
                                          Attr.getArg(0), StartArg, Size,
                                          Attr.getAttributeSpellingListIndex()));
}

static bool checkLocksRequiredCommon(Sema &S, Decl *D,
                                     const AttributeList &Attr,
                                     SmallVector<Expr*, 1> &Args) {
  assert(!Attr.isInvalid());

  if (!checkAttributeAtLeastNumArgs(S, Attr, 1))
    return false;

  if (!isa<FunctionDecl>(D) && !isa<FunctionTemplateDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_thread_attribute_wrong_decl_type)
      << Attr.getName() << ThreadExpectedFunctionOrMethod;
    return false;
  }

  // check that all arguments are lockable objects
  checkAttrArgsAreLockableObjs(S, D, Attr, Args);
  if (Args.empty())
    return false;

  return true;
}

static void handleExclusiveLocksRequiredAttr(Sema &S, Decl *D,
                                             const AttributeList &Attr) {
  SmallVector<Expr*, 1> Args;
  if (!checkLocksRequiredCommon(S, D, Attr, Args))
    return;

  Expr **StartArg = &Args[0];
  D->addAttr(::new (S.Context)
             ExclusiveLocksRequiredAttr(Attr.getRange(), S.Context,
                                        StartArg, Args.size(),
                                        Attr.getAttributeSpellingListIndex()));
}

static void handleSharedLocksRequiredAttr(Sema &S, Decl *D,
                                          const AttributeList &Attr) {
  SmallVector<Expr*, 1> Args;
  if (!checkLocksRequiredCommon(S, D, Attr, Args))
    return;

  Expr **StartArg = &Args[0];
  D->addAttr(::new (S.Context)
             SharedLocksRequiredAttr(Attr.getRange(), S.Context,
                                     StartArg, Args.size(),
                                     Attr.getAttributeSpellingListIndex()));
}

static void handleUnlockFunAttr(Sema &S, Decl *D,
                                const AttributeList &Attr) {
  assert(!Attr.isInvalid());

  // zero or more arguments ok

  if (!isa<FunctionDecl>(D) && !isa<FunctionTemplateDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_thread_attribute_wrong_decl_type)
      << Attr.getName() << ThreadExpectedFunctionOrMethod;
    return;
  }

  // check that all arguments are lockable objects
  SmallVector<Expr*, 1> Args;
  checkAttrArgsAreLockableObjs(S, D, Attr, Args, 0, /*ParamIdxOk=*/true);
  unsigned Size = Args.size();
  Expr **StartArg = Size == 0 ? 0 : &Args[0];

  D->addAttr(::new (S.Context)
             UnlockFunctionAttr(Attr.getRange(), S.Context, StartArg, Size,
                                Attr.getAttributeSpellingListIndex()));
}

static void handleLockReturnedAttr(Sema &S, Decl *D,
                                   const AttributeList &Attr) {
  assert(!Attr.isInvalid());

  if (!checkAttributeNumArgs(S, Attr, 1))
    return;

  if (!isa<FunctionDecl>(D) && !isa<FunctionTemplateDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_thread_attribute_wrong_decl_type)
      << Attr.getName() << ThreadExpectedFunctionOrMethod;
    return;
  }

  // check that the argument is lockable object
  SmallVector<Expr*, 1> Args;
  checkAttrArgsAreLockableObjs(S, D, Attr, Args);
  unsigned Size = Args.size();
  if (Size == 0)
    return;

  D->addAttr(::new (S.Context)
             LockReturnedAttr(Attr.getRange(), S.Context, Args[0],
                              Attr.getAttributeSpellingListIndex()));
}

static void handleLocksExcludedAttr(Sema &S, Decl *D,
                                    const AttributeList &Attr) {
  assert(!Attr.isInvalid());

  if (!checkAttributeAtLeastNumArgs(S, Attr, 1))
    return;

  if (!isa<FunctionDecl>(D) && !isa<FunctionTemplateDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_thread_attribute_wrong_decl_type)
      << Attr.getName() << ThreadExpectedFunctionOrMethod;
    return;
  }

  // check that all arguments are lockable objects
  SmallVector<Expr*, 1> Args;
  checkAttrArgsAreLockableObjs(S, D, Attr, Args);
  unsigned Size = Args.size();
  if (Size == 0)
    return;
  Expr **StartArg = &Args[0];

  D->addAttr(::new (S.Context)
             LocksExcludedAttr(Attr.getRange(), S.Context, StartArg, Size,
                               Attr.getAttributeSpellingListIndex()));
}


static void handleExtVectorTypeAttr(Sema &S, Scope *scope, Decl *D,
                                    const AttributeList &Attr) {
  TypedefNameDecl *TD = dyn_cast<TypedefNameDecl>(D);
  if (TD == 0) {
    // __attribute__((ext_vector_type(N))) can only be applied to typedefs
    // and type-ids.
    S.Diag(Attr.getLoc(), diag::err_typecheck_ext_vector_not_typedef);
    return;
  }

  // Remember this typedef decl, we will need it later for diagnostics.
  S.ExtVectorDecls.push_back(TD);
}

static void handlePackedAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  if (TagDecl *TD = dyn_cast<TagDecl>(D))
    TD->addAttr(::new (S.Context) PackedAttr(Attr.getRange(), S.Context));
  else if (FieldDecl *FD = dyn_cast<FieldDecl>(D)) {
    // If the alignment is less than or equal to 8 bits, the packed attribute
    // has no effect.
    if (!FD->getType()->isDependentType() &&
        !FD->getType()->isIncompleteType() &&
        S.Context.getTypeAlign(FD->getType()) <= 8)
      S.Diag(Attr.getLoc(), diag::warn_attribute_ignored_for_field_of_type)
        << Attr.getName() << FD->getType();
    else
      FD->addAttr(::new (S.Context)
                  PackedAttr(Attr.getRange(), S.Context,
                             Attr.getAttributeSpellingListIndex()));
  } else
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << Attr.getName();
}

static void handleMsStructAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (RecordDecl *RD = dyn_cast<RecordDecl>(D))
    RD->addAttr(::new (S.Context)
                MsStructAttr(Attr.getRange(), S.Context,
                             Attr.getAttributeSpellingListIndex()));
  else
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << Attr.getName();
}

static void handleIBAction(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  // The IBAction attributes only apply to instance methods.
  if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D))
    if (MD->isInstanceMethod()) {
      D->addAttr(::new (S.Context)
                 IBActionAttr(Attr.getRange(), S.Context,
                              Attr.getAttributeSpellingListIndex()));
      return;
    }

  S.Diag(Attr.getLoc(), diag::warn_attribute_ibaction) << Attr.getName();
}

static bool checkIBOutletCommon(Sema &S, Decl *D, const AttributeList &Attr) {
  // The IBOutlet/IBOutletCollection attributes only apply to instance
  // variables or properties of Objective-C classes.  The outlet must also
  // have an object reference type.
  if (const ObjCIvarDecl *VD = dyn_cast<ObjCIvarDecl>(D)) {
    if (!VD->getType()->getAs<ObjCObjectPointerType>()) {
      S.Diag(Attr.getLoc(), diag::warn_iboutlet_object_type)
        << Attr.getName() << VD->getType() << 0;
      return false;
    }
  }
  else if (const ObjCPropertyDecl *PD = dyn_cast<ObjCPropertyDecl>(D)) {
    if (!PD->getType()->getAs<ObjCObjectPointerType>()) {
      S.Diag(Attr.getLoc(), diag::warn_iboutlet_object_type)
        << Attr.getName() << PD->getType() << 1;
      return false;
    }
  }
  else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_iboutlet) << Attr.getName();
    return false;
  }

  return true;
}

static void handleIBOutlet(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;
  
  if (!checkIBOutletCommon(S, D, Attr))
    return;

  D->addAttr(::new (S.Context)
             IBOutletAttr(Attr.getRange(), S.Context,
                          Attr.getAttributeSpellingListIndex()));
}

static void handleIBOutletCollection(Sema &S, Decl *D,
                                     const AttributeList &Attr) {

  // The iboutletcollection attribute can have zero or one arguments.
  if (Attr.getParameterName() && Attr.getNumArgs() > 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

  if (!checkIBOutletCommon(S, D, Attr))
    return;

  IdentifierInfo *II = Attr.getParameterName();
  if (!II)
    II = &S.Context.Idents.get("NSObject");
  
  ParsedType TypeRep = S.getTypeName(*II, Attr.getLoc(), 
                        S.getScopeForContext(D->getDeclContext()->getParent()));
  if (!TypeRep) {
    S.Diag(Attr.getLoc(), diag::err_iboutletcollection_type) << II;
    return;
  }
  QualType QT = TypeRep.get();
  // Diagnose use of non-object type in iboutletcollection attribute.
  // FIXME. Gnu attribute extension ignores use of builtin types in
  // attributes. So, __attribute__((iboutletcollection(char))) will be
  // treated as __attribute__((iboutletcollection())).
  if (!QT->isObjCIdType() && !QT->isObjCObjectType()) {
    S.Diag(Attr.getLoc(), diag::err_iboutletcollection_type) << II;
    return;
  }
  D->addAttr(::new (S.Context)
             IBOutletCollectionAttr(Attr.getRange(),S.Context,
                                    QT, Attr.getParameterLoc(),
                                    Attr.getAttributeSpellingListIndex()));
}

static void possibleTransparentUnionPointerType(QualType &T) {
  if (const RecordType *UT = T->getAsUnionType())
    if (UT && UT->getDecl()->hasAttr<TransparentUnionAttr>()) {
      RecordDecl *UD = UT->getDecl();
      for (RecordDecl::field_iterator it = UD->field_begin(),
           itend = UD->field_end(); it != itend; ++it) {
        QualType QT = it->getType();
        if (QT->isAnyPointerType() || QT->isBlockPointerType()) {
          T = QT;
          return;
        }
      }
    }
}

static void handleAllocSizeAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (!isFunctionOrMethod(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
    << "alloc_size" << ExpectedFunctionOrMethod;
    return;
  }

  if (!checkAttributeAtLeastNumArgs(S, Attr, 1))
    return;

  // In C++ the implicit 'this' function parameter also counts, and they are
  // counted from one.
  bool HasImplicitThisParam = isInstanceMethod(D);
  unsigned NumArgs;
  if (hasFunctionProto(D))
    NumArgs = getFunctionOrMethodNumArgs(D) + HasImplicitThisParam;
  else
    NumArgs = 0;

  SmallVector<unsigned, 8> SizeArgs;

  for (AttributeList::arg_iterator I = Attr.arg_begin(),
       E = Attr.arg_end(); I!=E; ++I) {
    // The argument must be an integer constant expression.
    Expr *Ex = *I;
    llvm::APSInt ArgNum;
    if (Ex->isTypeDependent() || Ex->isValueDependent() ||
        !Ex->isIntegerConstantExpr(ArgNum, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_not_int)
      << "alloc_size" << Ex->getSourceRange();
      return;
    }

    uint64_t x = ArgNum.getZExtValue();

    if (x < 1 || x > NumArgs) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_out_of_bounds)
      << "alloc_size" << I.getArgNum() << Ex->getSourceRange();
      return;
    }

    --x;
    if (HasImplicitThisParam) {
      if (x == 0) {
        S.Diag(Attr.getLoc(),
               diag::err_attribute_invalid_implicit_this_argument)
        << "alloc_size" << Ex->getSourceRange();
        return;
      }
      --x;
    }

    // check if the function argument is of an integer type
    QualType T = getFunctionOrMethodArgType(D, x).getNonReferenceType();
    if (!T->isIntegerType()) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_not_int)
      << "alloc_size" << Ex->getSourceRange();
      return;
    }

    SizeArgs.push_back(x);
  }

  // check if the function returns a pointer
  if (!getFunctionType(D)->getResultType()->isAnyPointerType()) {
    S.Diag(Attr.getLoc(), diag::warn_ns_attribute_wrong_return_type)
    << "alloc_size" << 0 /*function*/<< 1 /*pointer*/ << D->getSourceRange();
  }

  D->addAttr(::new (S.Context)
             AllocSizeAttr(Attr.getRange(), S.Context,
                           SizeArgs.data(), SizeArgs.size(),
                           Attr.getAttributeSpellingListIndex()));
}

static void handleNonNullAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // GCC ignores the nonnull attribute on K&R style function prototypes, so we
  // ignore it as well
  if (!isFunctionOrMethod(D) || !hasFunctionProto(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return;
  }

  // In C++ the implicit 'this' function parameter also counts, and they are
  // counted from one.
  bool HasImplicitThisParam = isInstanceMethod(D);
  unsigned NumArgs = getFunctionOrMethodNumArgs(D) + HasImplicitThisParam;

  // The nonnull attribute only applies to pointers.
  SmallVector<unsigned, 10> NonNullArgs;

  for (AttributeList::arg_iterator I = Attr.arg_begin(),
                                   E = Attr.arg_end(); I != E; ++I) {
    // The argument must be an integer constant expression.
    Expr *Ex = *I;
    llvm::APSInt ArgNum(32);
    if (Ex->isTypeDependent() || Ex->isValueDependent() ||
        !Ex->isIntegerConstantExpr(ArgNum, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_not_int)
        << "nonnull" << Ex->getSourceRange();
      return;
    }

    unsigned x = (unsigned) ArgNum.getZExtValue();

    if (x < 1 || x > NumArgs) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_out_of_bounds)
       << "nonnull" << I.getArgNum() << Ex->getSourceRange();
      return;
    }

    --x;
    if (HasImplicitThisParam) {
      if (x == 0) {
        S.Diag(Attr.getLoc(),
               diag::err_attribute_invalid_implicit_this_argument)
          << "nonnull" << Ex->getSourceRange();
        return;
      }
      --x;
    }

    // Is the function argument a pointer type?
    QualType T = getFunctionOrMethodArgType(D, x).getNonReferenceType();
    possibleTransparentUnionPointerType(T);
    
    if (!T->isAnyPointerType() && !T->isBlockPointerType()) {
      // FIXME: Should also highlight argument in decl.
      S.Diag(Attr.getLoc(), diag::warn_nonnull_pointers_only)
        << "nonnull" << Ex->getSourceRange();
      continue;
    }

    NonNullArgs.push_back(x);
  }

  // If no arguments were specified to __attribute__((nonnull)) then all pointer
  // arguments have a nonnull attribute.
  if (NonNullArgs.empty()) {
    for (unsigned i = 0, e = getFunctionOrMethodNumArgs(D); i != e; ++i) {
      QualType T = getFunctionOrMethodArgType(D, i).getNonReferenceType();
      possibleTransparentUnionPointerType(T);
      if (T->isAnyPointerType() || T->isBlockPointerType())
        NonNullArgs.push_back(i);
    }

    // No pointer arguments?
    if (NonNullArgs.empty()) {
      // Warn the trivial case only if attribute is not coming from a
      // macro instantiation.
      if (Attr.getLoc().isFileID())
        S.Diag(Attr.getLoc(), diag::warn_attribute_nonnull_no_pointers);
      return;
    }
  }

  unsigned *start = &NonNullArgs[0];
  unsigned size = NonNullArgs.size();
  llvm::array_pod_sort(start, start + size);
  D->addAttr(::new (S.Context)
             NonNullAttr(Attr.getRange(), S.Context, start, size,
                         Attr.getAttributeSpellingListIndex()));
}

static void handleOwnershipAttr(Sema &S, Decl *D, const AttributeList &AL) {
  // This attribute must be applied to a function declaration.
  // The first argument to the attribute must be a string,
  // the name of the resource, for example "malloc".
  // The following arguments must be argument indexes, the arguments must be
  // of integer type for Returns, otherwise of pointer type.
  // The difference between Holds and Takes is that a pointer may still be used
  // after being held.  free() should be __attribute((ownership_takes)), whereas
  // a list append function may well be __attribute((ownership_holds)).

  if (!AL.getParameterName()) {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_n_not_string)
        << AL.getName()->getName() << 1;
    return;
  }
  // Figure out our Kind, and check arguments while we're at it.
  OwnershipAttr::OwnershipKind K;
  switch (AL.getKind()) {
  case AttributeList::AT_ownership_takes:
    K = OwnershipAttr::Takes;
    if (AL.getNumArgs() < 1) {
      S.Diag(AL.getLoc(), diag::err_attribute_wrong_number_arguments) << 2;
      return;
    }
    break;
  case AttributeList::AT_ownership_holds:
    K = OwnershipAttr::Holds;
    if (AL.getNumArgs() < 1) {
      S.Diag(AL.getLoc(), diag::err_attribute_wrong_number_arguments) << 2;
      return;
    }
    break;
  case AttributeList::AT_ownership_returns:
    K = OwnershipAttr::Returns;
    if (AL.getNumArgs() > 1) {
      S.Diag(AL.getLoc(), diag::err_attribute_wrong_number_arguments)
          << AL.getNumArgs() + 1;
      return;
    }
    break;
  default:
    // This should never happen given how we are called.
    llvm_unreachable("Unknown ownership attribute");
  }

  if (!isFunction(D) || !hasFunctionProto(D)) {
    S.Diag(AL.getLoc(), diag::warn_attribute_wrong_decl_type)
      << AL.getName() << ExpectedFunction;
    return;
  }

  // In C++ the implicit 'this' function parameter also counts, and they are
  // counted from one.
  bool HasImplicitThisParam = isInstanceMethod(D);
  unsigned NumArgs  = getFunctionOrMethodNumArgs(D) + HasImplicitThisParam;

  StringRef Module = AL.getParameterName()->getName();

  // Normalize the argument, __foo__ becomes foo.
  if (Module.startswith("__") && Module.endswith("__"))
    Module = Module.substr(2, Module.size() - 4);

  SmallVector<unsigned, 10> OwnershipArgs;

  for (AttributeList::arg_iterator I = AL.arg_begin(), E = AL.arg_end(); I != E;
       ++I) {

    Expr *IdxExpr = *I;
    llvm::APSInt ArgNum(32);
    if (IdxExpr->isTypeDependent() || IdxExpr->isValueDependent()
        || !IdxExpr->isIntegerConstantExpr(ArgNum, S.Context)) {
      S.Diag(AL.getLoc(), diag::err_attribute_argument_not_int)
          << AL.getName()->getName() << IdxExpr->getSourceRange();
      continue;
    }

    unsigned x = (unsigned) ArgNum.getZExtValue();

    if (x > NumArgs || x < 1) {
      S.Diag(AL.getLoc(), diag::err_attribute_argument_out_of_bounds)
          << AL.getName()->getName() << x << IdxExpr->getSourceRange();
      continue;
    }
    --x;
    if (HasImplicitThisParam) {
      if (x == 0) {
        S.Diag(AL.getLoc(), diag::err_attribute_invalid_implicit_this_argument)
          << "ownership" << IdxExpr->getSourceRange();
        return;
      }
      --x;
    }

    switch (K) {
    case OwnershipAttr::Takes:
    case OwnershipAttr::Holds: {
      // Is the function argument a pointer type?
      QualType T = getFunctionOrMethodArgType(D, x);
      if (!T->isAnyPointerType() && !T->isBlockPointerType()) {
        // FIXME: Should also highlight argument in decl.
        S.Diag(AL.getLoc(), diag::err_ownership_type)
            << ((K==OwnershipAttr::Takes)?"ownership_takes":"ownership_holds")
            << "pointer"
            << IdxExpr->getSourceRange();
        continue;
      }
      break;
    }
    case OwnershipAttr::Returns: {
      if (AL.getNumArgs() > 1) {
          // Is the function argument an integer type?
          Expr *IdxExpr = AL.getArg(0);
          llvm::APSInt ArgNum(32);
          if (IdxExpr->isTypeDependent() || IdxExpr->isValueDependent()
              || !IdxExpr->isIntegerConstantExpr(ArgNum, S.Context)) {
            S.Diag(AL.getLoc(), diag::err_ownership_type)
                << "ownership_returns" << "integer"
                << IdxExpr->getSourceRange();
            return;
          }
      }
      break;
    }
    } // switch

    // Check we don't have a conflict with another ownership attribute.
    for (specific_attr_iterator<OwnershipAttr>
          i = D->specific_attr_begin<OwnershipAttr>(),
          e = D->specific_attr_end<OwnershipAttr>();
        i != e; ++i) {
      if ((*i)->getOwnKind() != K) {
        for (const unsigned *I = (*i)->args_begin(), *E = (*i)->args_end();
             I!=E; ++I) {
          if (x == *I) {
            S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible)
                << AL.getName()->getName() << "ownership_*";
          }
        }
      }
    }
    OwnershipArgs.push_back(x);
  }

  unsigned* start = OwnershipArgs.data();
  unsigned size = OwnershipArgs.size();
  llvm::array_pod_sort(start, start + size);

  if (K != OwnershipAttr::Returns && OwnershipArgs.empty()) {
    S.Diag(AL.getLoc(), diag::err_attribute_wrong_number_arguments) << 2;
    return;
  }

  D->addAttr(::new (S.Context)
             OwnershipAttr(AL.getLoc(), S.Context, K, Module, start, size,
                           AL.getAttributeSpellingListIndex()));
}

static void handleWeakRefAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // Check the attribute arguments.
  if (Attr.getNumArgs() > 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

  if (!isa<VarDecl>(D) && !isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedVariableOrFunction;
    return;
  }

  NamedDecl *nd = cast<NamedDecl>(D);

  // gcc rejects
  // class c {
  //   static int a __attribute__((weakref ("v2")));
  //   static int b() __attribute__((weakref ("f3")));
  // };
  // and ignores the attributes of
  // void f(void) {
  //   static int a __attribute__((weakref ("v2")));
  // }
  // we reject them
  const DeclContext *Ctx = D->getDeclContext()->getRedeclContext();
  if (!Ctx->isFileContext()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_weakref_not_global_context) <<
        nd->getNameAsString();
    return;
  }

  // The GCC manual says
  //
  // At present, a declaration to which `weakref' is attached can only
  // be `static'.
  //
  // It also says
  //
  // Without a TARGET,
  // given as an argument to `weakref' or to `alias', `weakref' is
  // equivalent to `weak'.
  //
  // gcc 4.4.1 will accept
  // int a7 __attribute__((weakref));
  // as
  // int a7 __attribute__((weak));
  // This looks like a bug in gcc. We reject that for now. We should revisit
  // it if this behaviour is actually used.

  // GCC rejects
  // static ((alias ("y"), weakref)).
  // Should we? How to check that weakref is before or after alias?

  if (Attr.getNumArgs() == 1) {
    Expr *Arg = Attr.getArg(0);
    Arg = Arg->IgnoreParenCasts();
    StringLiteral *Str = dyn_cast<StringLiteral>(Arg);

    if (!Str || !Str->isAscii()) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string)
          << "weakref" << 1;
      return;
    }
    // GCC will accept anything as the argument of weakref. Should we
    // check for an existing decl?
    D->addAttr(::new (S.Context) AliasAttr(Attr.getRange(), S.Context,
                                           Str->getString()));
  }

  D->addAttr(::new (S.Context)
             WeakRefAttr(Attr.getRange(), S.Context,
                         Attr.getAttributeSpellingListIndex()));
}

static void handleAliasAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

  Expr *Arg = Attr.getArg(0);
  Arg = Arg->IgnoreParenCasts();
  StringLiteral *Str = dyn_cast<StringLiteral>(Arg);

  if (!Str || !Str->isAscii()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string)
      << "alias" << 1;
    return;
  }

  if (S.Context.getTargetInfo().getTriple().isOSDarwin()) {
    S.Diag(Attr.getLoc(), diag::err_alias_not_supported_on_darwin);
    return;
  }

  // FIXME: check if target symbol exists in current file

  D->addAttr(::new (S.Context) AliasAttr(Attr.getRange(), S.Context,
                                         Str->getString(),
                                         Attr.getAttributeSpellingListIndex()));
}

static void handleMinSizeAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // Check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  if (!isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrMethod;
    return;
  }

  D->addAttr(::new (S.Context)
             MinSizeAttr(Attr.getRange(), S.Context,
                         Attr.getAttributeSpellingListIndex()));
}

static void handleColdAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // Check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return;
  }

  if (D->hasAttr<HotAttr>()) {
    S.Diag(Attr.getLoc(), diag::err_attributes_are_not_compatible)
      << Attr.getName() << "hot";
    return;
  }

  D->addAttr(::new (S.Context) ColdAttr(Attr.getRange(), S.Context,
                                        Attr.getAttributeSpellingListIndex()));
}

static void handleHotAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // Check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return;
  }

  if (D->hasAttr<ColdAttr>()) {
    S.Diag(Attr.getLoc(), diag::err_attributes_are_not_compatible)
      << Attr.getName() << "cold";
    return;
  }

  D->addAttr(::new (S.Context) HotAttr(Attr.getRange(), S.Context,
                                       Attr.getAttributeSpellingListIndex()));
}

static void handleNakedAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // Check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return;
  }

  D->addAttr(::new (S.Context)
             NakedAttr(Attr.getRange(), S.Context,
                       Attr.getAttributeSpellingListIndex()));
}

static void handleAlwaysInlineAttr(Sema &S, Decl *D,
                                   const AttributeList &Attr) {
  // Check the attribute arguments.
  if (Attr.hasParameterOrArguments()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return;
  }

  D->addAttr(::new (S.Context)
             AlwaysInlineAttr(Attr.getRange(), S.Context,
                              Attr.getAttributeSpellingListIndex()));
}

static void handleTLSModelAttr(Sema &S, Decl *D,
                               const AttributeList &Attr) {
  // Check the attribute arguments.
  if (Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

  Expr *Arg = Attr.getArg(0);
  Arg = Arg->IgnoreParenCasts();
  StringLiteral *Str = dyn_cast<StringLiteral>(Arg);

  // Check that it is a string.
  if (!Str) {
    S.Diag(Attr.getLoc(), diag::err_attribute_not_string) << "tls_model";
    return;
  }

  if (!isa<VarDecl>(D) || !cast<VarDecl>(D)->getTLSKind()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedTLSVar;
    return;
  }

  // Check that the value.
  StringRef Model = Str->getString();
  if (Model != "global-dynamic" && Model != "local-dynamic"
      && Model != "initial-exec" && Model != "local-exec") {
    S.Diag(Attr.getLoc(), diag::err_attr_tlsmodel_arg);
    return;
  }

  D->addAttr(::new (S.Context)
             TLSModelAttr(Attr.getRange(), S.Context, Model,
                          Attr.getAttributeSpellingListIndex()));
}

static void handleMallocAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // Check the attribute arguments.
  if (Attr.hasParameterOrArguments()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    QualType RetTy = FD->getResultType();
    if (RetTy->isAnyPointerType() || RetTy->isBlockPointerType()) {
      D->addAttr(::new (S.Context)
                 MallocAttr(Attr.getRange(), S.Context,
                            Attr.getAttributeSpellingListIndex()));
      return;
    }
  }

  S.Diag(Attr.getLoc(), diag::warn_attribute_malloc_pointer_only);
}

static void handleMayAliasAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  D->addAttr(::new (S.Context)
             MayAliasAttr(Attr.getRange(), S.Context,
                          Attr.getAttributeSpellingListIndex()));
}

static void handleNoCommonAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  assert(!Attr.isInvalid());
  if (isa<VarDecl>(D))
    D->addAttr(::new (S.Context)
               NoCommonAttr(Attr.getRange(), S.Context,
                            Attr.getAttributeSpellingListIndex()));
  else
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedVariable;
}

static void handleCommonAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  assert(!Attr.isInvalid());
  if (isa<VarDecl>(D))
    D->addAttr(::new (S.Context)
               CommonAttr(Attr.getRange(), S.Context,
                          Attr.getAttributeSpellingListIndex()));
  else
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedVariable;
}

static void handleNoReturnAttr(Sema &S, Decl *D, const AttributeList &attr) {
  if (hasDeclarator(D)) return;

  if (S.CheckNoReturnAttr(attr)) return;

  if (!isa<ObjCMethodDecl>(D)) {
    S.Diag(attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << attr.getName() << ExpectedFunctionOrMethod;
    return;
  }

  D->addAttr(::new (S.Context)
             NoReturnAttr(attr.getRange(), S.Context,
                          attr.getAttributeSpellingListIndex()));
}

bool Sema::CheckNoReturnAttr(const AttributeList &attr) {
  if (attr.hasParameterOrArguments()) {
    Diag(attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    attr.setInvalid();
    return true;
  }

  return false;
}

static void handleAnalyzerNoReturnAttr(Sema &S, Decl *D,
                                       const AttributeList &Attr) {
  
  // The checking path for 'noreturn' and 'analyzer_noreturn' are different
  // because 'analyzer_noreturn' does not impact the type.
  
  if(!checkAttributeNumArgs(S, Attr, 0))
      return;
  
  if (!isFunctionOrMethod(D) && !isa<BlockDecl>(D)) {
    ValueDecl *VD = dyn_cast<ValueDecl>(D);
    if (VD == 0 || (!VD->getType()->isBlockPointerType()
                    && !VD->getType()->isFunctionPointerType())) {
      S.Diag(Attr.getLoc(),
             Attr.isCXX11Attribute() ? diag::err_attribute_wrong_decl_type
             : diag::warn_attribute_wrong_decl_type)
        << Attr.getName() << ExpectedFunctionMethodOrBlock;
      return;
    }
  }
  
  D->addAttr(::new (S.Context)
             AnalyzerNoReturnAttr(Attr.getRange(), S.Context,
                                  Attr.getAttributeSpellingListIndex()));
}

static void handleCXX11NoReturnAttr(Sema &S, Decl *D,
                                    const AttributeList &Attr) {
  // C++11 [dcl.attr.noreturn]p1:
  //   The attribute may be applied to the declarator-id in a function
  //   declaration.
  FunctionDecl *FD = dyn_cast<FunctionDecl>(D);
  if (!FD) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrMethod;
    return;
  }

  D->addAttr(::new (S.Context)
             CXX11NoReturnAttr(Attr.getRange(), S.Context,
                               Attr.getAttributeSpellingListIndex()));
}

// PS3 PPU-specific.
static void handleVecReturnAttr(Sema &S, Decl *D, const AttributeList &Attr) {
/*
  Returning a Vector Class in Registers
  
  According to the PPU ABI specifications, a class with a single member of 
  vector type is returned in memory when used as the return value of a function.
  This results in inefficient code when implementing vector classes. To return
  the value in a single vector register, add the vecreturn attribute to the
  class definition. This attribute is also applicable to struct types.
  
  Example:
  
  struct Vector
  {
    __vector float xyzw;
  } __attribute__((vecreturn));
  
  Vector Add(Vector lhs, Vector rhs)
  {
    Vector result;
    result.xyzw = vec_add(lhs.xyzw, rhs.xyzw);
    return result; // This will be returned in a register
  }
*/
  if (!isa<RecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedClass;
    return;
  }

  if (D->getAttr<VecReturnAttr>()) {
    S.Diag(Attr.getLoc(), diag::err_repeat_attribute) << "vecreturn";
    return;
  }

  RecordDecl *record = cast<RecordDecl>(D);
  int count = 0;

  if (!isa<CXXRecordDecl>(record)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_vecreturn_only_vector_member);
    return;
  }

  if (!cast<CXXRecordDecl>(record)->isPOD()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_vecreturn_only_pod_record);
    return;
  }

  for (RecordDecl::field_iterator iter = record->field_begin();
       iter != record->field_end(); iter++) {
    if ((count == 1) || !iter->getType()->isVectorType()) {
      S.Diag(Attr.getLoc(), diag::err_attribute_vecreturn_only_vector_member);
      return;
    }
    count++;
  }

  D->addAttr(::new (S.Context)
             VecReturnAttr(Attr.getRange(), S.Context,
                           Attr.getAttributeSpellingListIndex()));
}

static void handleDependencyAttr(Sema &S, Scope *Scope, Decl *D,
                                 const AttributeList &Attr) {
  if (isa<ParmVarDecl>(D)) {
    // [[carries_dependency]] can only be applied to a parameter if it is a
    // parameter of a function declaration or lambda.
    if (!(Scope->getFlags() & clang::Scope::FunctionDeclarationScope)) {
      S.Diag(Attr.getLoc(),
             diag::err_carries_dependency_param_not_function_decl);
      return;
    }
  } else if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionMethodOrParameter;
    return;
  }

  D->addAttr(::new (S.Context) CarriesDependencyAttr(
                                   Attr.getRange(), S.Context,
                                   Attr.getAttributeSpellingListIndex()));
}

static void handleUnusedAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.hasParameterOrArguments()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  if (!isa<VarDecl>(D) && !isa<ObjCIvarDecl>(D) && !isFunctionOrMethod(D) &&
      !isa<TypeDecl>(D) && !isa<LabelDecl>(D) && !isa<FieldDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedVariableFunctionOrLabel;
    return;
  }

  D->addAttr(::new (S.Context)
             UnusedAttr(Attr.getRange(), S.Context,
                        Attr.getAttributeSpellingListIndex()));
}

static void handleReturnsTwiceAttr(Sema &S, Decl *D,
                                   const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.hasParameterOrArguments()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return;
  }

  D->addAttr(::new (S.Context)
             ReturnsTwiceAttr(Attr.getRange(), S.Context,
                              Attr.getAttributeSpellingListIndex()));
}

static void handleUsedAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.hasParameterOrArguments()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
    if (VD->hasLocalStorage() || VD->hasExternalStorage()) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "used";
      return;
    }
  } else if (!isFunctionOrMethod(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedVariableOrFunction;
    return;
  }

  D->addAttr(::new (S.Context)
             UsedAttr(Attr.getRange(), S.Context,
                      Attr.getAttributeSpellingListIndex()));
}

static void handleConstructorAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.getNumArgs() > 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_too_many_arguments) << 1;
    return;
  }

  int priority = 65535; // FIXME: Do not hardcode such constants.
  if (Attr.getNumArgs() > 0) {
    Expr *E = Attr.getArg(0);
    llvm::APSInt Idx(32);
    if (E->isTypeDependent() || E->isValueDependent() ||
        !E->isIntegerConstantExpr(Idx, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_int)
        << "constructor" << 1 << E->getSourceRange();
      return;
    }
    priority = Idx.getZExtValue();
  }

  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return;
  }

  D->addAttr(::new (S.Context)
             ConstructorAttr(Attr.getRange(), S.Context, priority,
                             Attr.getAttributeSpellingListIndex()));
}

static void handleDestructorAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.getNumArgs() > 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_too_many_arguments) << 1;
    return;
  }

  int priority = 65535; // FIXME: Do not hardcode such constants.
  if (Attr.getNumArgs() > 0) {
    Expr *E = Attr.getArg(0);
    llvm::APSInt Idx(32);
    if (E->isTypeDependent() || E->isValueDependent() ||
        !E->isIntegerConstantExpr(Idx, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_int)
        << "destructor" << 1 << E->getSourceRange();
      return;
    }
    priority = Idx.getZExtValue();
  }

  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return;
  }

  D->addAttr(::new (S.Context)
             DestructorAttr(Attr.getRange(), S.Context, priority,
                            Attr.getAttributeSpellingListIndex()));
}

template <typename AttrTy>
static void handleAttrWithMessage(Sema &S, Decl *D, const AttributeList &Attr,
                                  const char *Name) {
  unsigned NumArgs = Attr.getNumArgs();
  if (NumArgs > 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_too_many_arguments) << 1;
    return;
  }

  // Handle the case where the attribute has a text message.
  StringRef Str;
  if (NumArgs == 1) {
    StringLiteral *SE = dyn_cast<StringLiteral>(Attr.getArg(0));
    if (!SE) {
      S.Diag(Attr.getArg(0)->getLocStart(), diag::err_attribute_not_string)
        << Name;
      return;
    }
    Str = SE->getString();
  }

  D->addAttr(::new (S.Context) AttrTy(Attr.getRange(), S.Context, Str,
                                      Attr.getAttributeSpellingListIndex()));
}

static void handleArcWeakrefUnavailableAttr(Sema &S, Decl *D, 
                                            const AttributeList &Attr) {
  unsigned NumArgs = Attr.getNumArgs();
  if (NumArgs > 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_too_many_arguments) << 0;
    return;
  }
  
  D->addAttr(::new (S.Context)
             ArcWeakrefUnavailableAttr(Attr.getRange(), S.Context,
                                       Attr.getAttributeSpellingListIndex()));
}

static void handleObjCRootClassAttr(Sema &S, Decl *D, 
                                    const AttributeList &Attr) {
  if (!isa<ObjCInterfaceDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_requires_objc_interface);
    return;
  }
  
  unsigned NumArgs = Attr.getNumArgs();
  if (NumArgs > 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_too_many_arguments) << 0;
    return;
  }
  
  D->addAttr(::new (S.Context)
             ObjCRootClassAttr(Attr.getRange(), S.Context,
                               Attr.getAttributeSpellingListIndex()));
}

static void handleObjCRequiresPropertyDefsAttr(Sema &S, Decl *D,
                                               const AttributeList &Attr) {
  if (!isa<ObjCInterfaceDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_suppress_autosynthesis);
    return;
  }
  
  unsigned NumArgs = Attr.getNumArgs();
  if (NumArgs > 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_too_many_arguments) << 0;
    return;
  }
  
  D->addAttr(::new (S.Context)
             ObjCRequiresPropertyDefsAttr(Attr.getRange(), S.Context,
                                          Attr.getAttributeSpellingListIndex()));
}

static bool checkAvailabilityAttr(Sema &S, SourceRange Range,
                                  IdentifierInfo *Platform,
                                  VersionTuple Introduced,
                                  VersionTuple Deprecated,
                                  VersionTuple Obsoleted) {
  StringRef PlatformName
    = AvailabilityAttr::getPrettyPlatformName(Platform->getName());
  if (PlatformName.empty())
    PlatformName = Platform->getName();

  // Ensure that Introduced <= Deprecated <= Obsoleted (although not all
  // of these steps are needed).
  if (!Introduced.empty() && !Deprecated.empty() &&
      !(Introduced <= Deprecated)) {
    S.Diag(Range.getBegin(), diag::warn_availability_version_ordering)
      << 1 << PlatformName << Deprecated.getAsString()
      << 0 << Introduced.getAsString();
    return true;
  }

  if (!Introduced.empty() && !Obsoleted.empty() &&
      !(Introduced <= Obsoleted)) {
    S.Diag(Range.getBegin(), diag::warn_availability_version_ordering)
      << 2 << PlatformName << Obsoleted.getAsString()
      << 0 << Introduced.getAsString();
    return true;
  }

  if (!Deprecated.empty() && !Obsoleted.empty() &&
      !(Deprecated <= Obsoleted)) {
    S.Diag(Range.getBegin(), diag::warn_availability_version_ordering)
      << 2 << PlatformName << Obsoleted.getAsString()
      << 1 << Deprecated.getAsString();
    return true;
  }

  return false;
}

/// \brief Check whether the two versions match.
///
/// If either version tuple is empty, then they are assumed to match. If
/// \p BeforeIsOkay is true, then \p X can be less than or equal to \p Y.
static bool versionsMatch(const VersionTuple &X, const VersionTuple &Y,
                          bool BeforeIsOkay) {
  if (X.empty() || Y.empty())
    return true;

  if (X == Y)
    return true;

  if (BeforeIsOkay && X < Y)
    return true;

  return false;
}

AvailabilityAttr *Sema::mergeAvailabilityAttr(NamedDecl *D, SourceRange Range,
                                              IdentifierInfo *Platform,
                                              VersionTuple Introduced,
                                              VersionTuple Deprecated,
                                              VersionTuple Obsoleted,
                                              bool IsUnavailable,
                                              StringRef Message,
                                              bool Override,
                                              unsigned AttrSpellingListIndex) {
  VersionTuple MergedIntroduced = Introduced;
  VersionTuple MergedDeprecated = Deprecated;
  VersionTuple MergedObsoleted = Obsoleted;
  bool FoundAny = false;

  if (D->hasAttrs()) {
    AttrVec &Attrs = D->getAttrs();
    for (unsigned i = 0, e = Attrs.size(); i != e;) {
      const AvailabilityAttr *OldAA = dyn_cast<AvailabilityAttr>(Attrs[i]);
      if (!OldAA) {
        ++i;
        continue;
      }

      IdentifierInfo *OldPlatform = OldAA->getPlatform();
      if (OldPlatform != Platform) {
        ++i;
        continue;
      }

      FoundAny = true;
      VersionTuple OldIntroduced = OldAA->getIntroduced();
      VersionTuple OldDeprecated = OldAA->getDeprecated();
      VersionTuple OldObsoleted = OldAA->getObsoleted();
      bool OldIsUnavailable = OldAA->getUnavailable();

      if (!versionsMatch(OldIntroduced, Introduced, Override) ||
          !versionsMatch(Deprecated, OldDeprecated, Override) ||
          !versionsMatch(Obsoleted, OldObsoleted, Override) ||
          !(OldIsUnavailable == IsUnavailable ||
            (Override && !OldIsUnavailable && IsUnavailable))) {
        if (Override) {
          int Which = -1;
          VersionTuple FirstVersion;
          VersionTuple SecondVersion;
          if (!versionsMatch(OldIntroduced, Introduced, Override)) {
            Which = 0;
            FirstVersion = OldIntroduced;
            SecondVersion = Introduced;
          } else if (!versionsMatch(Deprecated, OldDeprecated, Override)) {
            Which = 1;
            FirstVersion = Deprecated;
            SecondVersion = OldDeprecated;
          } else if (!versionsMatch(Obsoleted, OldObsoleted, Override)) {
            Which = 2;
            FirstVersion = Obsoleted;
            SecondVersion = OldObsoleted;
          }

          if (Which == -1) {
            Diag(OldAA->getLocation(),
                 diag::warn_mismatched_availability_override_unavail)
              << AvailabilityAttr::getPrettyPlatformName(Platform->getName());
          } else {
            Diag(OldAA->getLocation(),
                 diag::warn_mismatched_availability_override)
              << Which
              << AvailabilityAttr::getPrettyPlatformName(Platform->getName())
              << FirstVersion.getAsString() << SecondVersion.getAsString();
          }
          Diag(Range.getBegin(), diag::note_overridden_method);
        } else {
          Diag(OldAA->getLocation(), diag::warn_mismatched_availability);
          Diag(Range.getBegin(), diag::note_previous_attribute);
        }

        Attrs.erase(Attrs.begin() + i);
        --e;
        continue;
      }

      VersionTuple MergedIntroduced2 = MergedIntroduced;
      VersionTuple MergedDeprecated2 = MergedDeprecated;
      VersionTuple MergedObsoleted2 = MergedObsoleted;

      if (MergedIntroduced2.empty())
        MergedIntroduced2 = OldIntroduced;
      if (MergedDeprecated2.empty())
        MergedDeprecated2 = OldDeprecated;
      if (MergedObsoleted2.empty())
        MergedObsoleted2 = OldObsoleted;

      if (checkAvailabilityAttr(*this, OldAA->getRange(), Platform,
                                MergedIntroduced2, MergedDeprecated2,
                                MergedObsoleted2)) {
        Attrs.erase(Attrs.begin() + i);
        --e;
        continue;
      }

      MergedIntroduced = MergedIntroduced2;
      MergedDeprecated = MergedDeprecated2;
      MergedObsoleted = MergedObsoleted2;
      ++i;
    }
  }

  if (FoundAny &&
      MergedIntroduced == Introduced &&
      MergedDeprecated == Deprecated &&
      MergedObsoleted == Obsoleted)
    return NULL;

  // Only create a new attribute if !Override, but we want to do
  // the checking.
  if (!checkAvailabilityAttr(*this, Range, Platform, MergedIntroduced,
                             MergedDeprecated, MergedObsoleted) &&
      !Override) {
    return ::new (Context) AvailabilityAttr(Range, Context, Platform,
                                            Introduced, Deprecated,
                                            Obsoleted, IsUnavailable, Message,
                                            AttrSpellingListIndex);
  }
  return NULL;
}

static void handleAvailabilityAttr(Sema &S, Decl *D,
                                   const AttributeList &Attr) {
  IdentifierInfo *Platform = Attr.getParameterName();
  SourceLocation PlatformLoc = Attr.getParameterLoc();
  unsigned Index = Attr.getAttributeSpellingListIndex();
  
  if (AvailabilityAttr::getPrettyPlatformName(Platform->getName()).empty())
    S.Diag(PlatformLoc, diag::warn_availability_unknown_platform)
      << Platform;

  NamedDecl *ND = dyn_cast<NamedDecl>(D);
  if (!ND) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << Attr.getName();
    return;
  }

  AvailabilityChange Introduced = Attr.getAvailabilityIntroduced();
  AvailabilityChange Deprecated = Attr.getAvailabilityDeprecated();
  AvailabilityChange Obsoleted = Attr.getAvailabilityObsoleted();
  bool IsUnavailable = Attr.getUnavailableLoc().isValid();
  StringRef Str;
  const StringLiteral *SE = 
    dyn_cast_or_null<const StringLiteral>(Attr.getMessageExpr());
  if (SE)
    Str = SE->getString();

  AvailabilityAttr *NewAttr = S.mergeAvailabilityAttr(ND, Attr.getRange(),
                                                      Platform,
                                                      Introduced.Version,
                                                      Deprecated.Version,
                                                      Obsoleted.Version,
                                                      IsUnavailable, Str,
                                                      /*Override=*/false,
                                                      Index);
  if (NewAttr)
    D->addAttr(NewAttr);
}

template <class T>
static T *mergeVisibilityAttr(Sema &S, Decl *D, SourceRange range,
                              typename T::VisibilityType value,
                              unsigned attrSpellingListIndex) {
  T *existingAttr = D->getAttr<T>();
  if (existingAttr) {
    typename T::VisibilityType existingValue = existingAttr->getVisibility();
    if (existingValue == value)
      return NULL;
    S.Diag(existingAttr->getLocation(), diag::err_mismatched_visibility);
    S.Diag(range.getBegin(), diag::note_previous_attribute);
    D->dropAttr<T>();
  }
  return ::new (S.Context) T(range, S.Context, value, attrSpellingListIndex);
}

VisibilityAttr *Sema::mergeVisibilityAttr(Decl *D, SourceRange Range,
                                          VisibilityAttr::VisibilityType Vis,
                                          unsigned AttrSpellingListIndex) {
  return ::mergeVisibilityAttr<VisibilityAttr>(*this, D, Range, Vis,
                                               AttrSpellingListIndex);
}

TypeVisibilityAttr *Sema::mergeTypeVisibilityAttr(Decl *D, SourceRange Range,
                                      TypeVisibilityAttr::VisibilityType Vis,
                                      unsigned AttrSpellingListIndex) {
  return ::mergeVisibilityAttr<TypeVisibilityAttr>(*this, D, Range, Vis,
                                                   AttrSpellingListIndex);
}

static void handleVisibilityAttr(Sema &S, Decl *D, const AttributeList &Attr,
                                 bool isTypeVisibility) {
  // Visibility attributes don't mean anything on a typedef.
  if (isa<TypedefNameDecl>(D)) {
    S.Diag(Attr.getRange().getBegin(), diag::warn_attribute_ignored)
      << Attr.getName();
    return;
  }

  // 'type_visibility' can only go on a type or namespace.
  if (isTypeVisibility &&
      !(isa<TagDecl>(D) ||
        isa<ObjCInterfaceDecl>(D) ||
        isa<NamespaceDecl>(D))) {
    S.Diag(Attr.getRange().getBegin(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedTypeOrNamespace;
    return;
  }

  // check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 1))
    return;

  Expr *Arg = Attr.getArg(0);
  Arg = Arg->IgnoreParenCasts();
  StringLiteral *Str = dyn_cast<StringLiteral>(Arg);

  if (!Str || !Str->isAscii()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string)
      << (isTypeVisibility ? "type_visibility" : "visibility") << 1;
    return;
  }

  StringRef TypeStr = Str->getString();
  VisibilityAttr::VisibilityType type;
  
  if (TypeStr == "default")
    type = VisibilityAttr::Default;
  else if (TypeStr == "hidden")
    type = VisibilityAttr::Hidden;
  else if (TypeStr == "internal")
    type = VisibilityAttr::Hidden; // FIXME
  else if (TypeStr == "protected") {
    // Complain about attempts to use protected visibility on targets
    // (like Darwin) that don't support it.
    if (!S.Context.getTargetInfo().hasProtectedVisibility()) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_protected_visibility);
      type = VisibilityAttr::Default;
    } else {
      type = VisibilityAttr::Protected;
    }
  } else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_unknown_visibility) << TypeStr;
    return;
  }

  unsigned Index = Attr.getAttributeSpellingListIndex();
  clang::Attr *newAttr;
  if (isTypeVisibility) {
    newAttr = S.mergeTypeVisibilityAttr(D, Attr.getRange(),
                                    (TypeVisibilityAttr::VisibilityType) type,
                                        Index);
  } else {
    newAttr = S.mergeVisibilityAttr(D, Attr.getRange(), type, Index);
  }
  if (newAttr)
    D->addAttr(newAttr);
}

static void handleObjCMethodFamilyAttr(Sema &S, Decl *decl,
                                       const AttributeList &Attr) {
  ObjCMethodDecl *method = dyn_cast<ObjCMethodDecl>(decl);
  if (!method) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << ExpectedMethod;
    return;
  }

  if (Attr.getNumArgs() != 0 || !Attr.getParameterName()) {
    if (!Attr.getParameterName() && Attr.getNumArgs() == 1) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string)
        << "objc_method_family" << 1;
    } else {
      S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    }
    Attr.setInvalid();
    return;
  }

  StringRef param = Attr.getParameterName()->getName();
  ObjCMethodFamilyAttr::FamilyKind family;
  if (param == "none")
    family = ObjCMethodFamilyAttr::OMF_None;
  else if (param == "alloc")
    family = ObjCMethodFamilyAttr::OMF_alloc;
  else if (param == "copy")
    family = ObjCMethodFamilyAttr::OMF_copy;
  else if (param == "init")
    family = ObjCMethodFamilyAttr::OMF_init;
  else if (param == "mutableCopy")
    family = ObjCMethodFamilyAttr::OMF_mutableCopy;
  else if (param == "new")
    family = ObjCMethodFamilyAttr::OMF_new;
  else {
    // Just warn and ignore it.  This is future-proof against new
    // families being used in system headers.
    S.Diag(Attr.getParameterLoc(), diag::warn_unknown_method_family);
    return;
  }

  if (family == ObjCMethodFamilyAttr::OMF_init && 
      !method->getResultType()->isObjCObjectPointerType()) {
    S.Diag(method->getLocation(), diag::err_init_method_bad_return_type)
      << method->getResultType();
    // Ignore the attribute.
    return;
  }

  method->addAttr(new (S.Context) ObjCMethodFamilyAttr(Attr.getRange(),
                                                       S.Context, family));
}

static void handleObjCExceptionAttr(Sema &S, Decl *D,
                                    const AttributeList &Attr) {
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  ObjCInterfaceDecl *OCI = dyn_cast<ObjCInterfaceDecl>(D);
  if (OCI == 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_requires_objc_interface);
    return;
  }

  D->addAttr(::new (S.Context)
             ObjCExceptionAttr(Attr.getRange(), S.Context,
                               Attr.getAttributeSpellingListIndex()));
}

static void handleObjCNSObject(Sema &S, Decl *D, const AttributeList &Attr) {
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }
  if (TypedefNameDecl *TD = dyn_cast<TypedefNameDecl>(D)) {
    QualType T = TD->getUnderlyingType();
    if (!T->isCARCBridgableType()) {
      S.Diag(TD->getLocation(), diag::err_nsobject_attribute);
      return;
    }
  }
  else if (ObjCPropertyDecl *PD = dyn_cast<ObjCPropertyDecl>(D)) {
    QualType T = PD->getType();
    if (!T->isCARCBridgableType()) {
      S.Diag(PD->getLocation(), diag::err_nsobject_attribute);
      return;
    }
  }
  else {
    // It is okay to include this attribute on properties, e.g.:
    //
    //  @property (retain, nonatomic) struct Bork *Q __attribute__((NSObject));
    //
    // In this case it follows tradition and suppresses an error in the above
    // case.    
    S.Diag(D->getLocation(), diag::warn_nsobject_attribute);
  }
  D->addAttr(::new (S.Context)
             ObjCNSObjectAttr(Attr.getRange(), S.Context,
                              Attr.getAttributeSpellingListIndex()));
}

static void
handleOverloadableAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_overloadable_not_function);
    return;
  }

  D->addAttr(::new (S.Context)
             OverloadableAttr(Attr.getRange(), S.Context,
                              Attr.getAttributeSpellingListIndex()));
}

static void handleBlocksAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (!Attr.getParameterName()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string)
      << "blocks" << 1;
    return;
  }

  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

  BlocksAttr::BlockType type;
  if (Attr.getParameterName()->isStr("byref"))
    type = BlocksAttr::ByRef;
  else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_type_not_supported)
      << "blocks" << Attr.getParameterName();
    return;
  }

  D->addAttr(::new (S.Context)
             BlocksAttr(Attr.getRange(), S.Context, type,
                        Attr.getAttributeSpellingListIndex()));
}

static void handleSentinelAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.getNumArgs() > 2) {
    S.Diag(Attr.getLoc(), diag::err_attribute_too_many_arguments) << 2;
    return;
  }

  unsigned sentinel = 0;
  if (Attr.getNumArgs() > 0) {
    Expr *E = Attr.getArg(0);
    llvm::APSInt Idx(32);
    if (E->isTypeDependent() || E->isValueDependent() ||
        !E->isIntegerConstantExpr(Idx, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_int)
       << "sentinel" << 1 << E->getSourceRange();
      return;
    }

    if (Idx.isSigned() && Idx.isNegative()) {
      S.Diag(Attr.getLoc(), diag::err_attribute_sentinel_less_than_zero)
        << E->getSourceRange();
      return;
    }

    sentinel = Idx.getZExtValue();
  }

  unsigned nullPos = 0;
  if (Attr.getNumArgs() > 1) {
    Expr *E = Attr.getArg(1);
    llvm::APSInt Idx(32);
    if (E->isTypeDependent() || E->isValueDependent() ||
        !E->isIntegerConstantExpr(Idx, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_int)
        << "sentinel" << 2 << E->getSourceRange();
      return;
    }
    nullPos = Idx.getZExtValue();

    if ((Idx.isSigned() && Idx.isNegative()) || nullPos > 1) {
      // FIXME: This error message could be improved, it would be nice
      // to say what the bounds actually are.
      S.Diag(Attr.getLoc(), diag::err_attribute_sentinel_not_zero_or_one)
        << E->getSourceRange();
      return;
    }
  }

  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    const FunctionType *FT = FD->getType()->castAs<FunctionType>();
    if (isa<FunctionNoProtoType>(FT)) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_sentinel_named_arguments);
      return;
    }

    if (!cast<FunctionProtoType>(FT)->isVariadic()) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_sentinel_not_variadic) << 0;
      return;
    }
  } else if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D)) {
    if (!MD->isVariadic()) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_sentinel_not_variadic) << 0;
      return;
    }
  } else if (BlockDecl *BD = dyn_cast<BlockDecl>(D)) {
    if (!BD->isVariadic()) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_sentinel_not_variadic) << 1;
      return;
    }
  } else if (const VarDecl *V = dyn_cast<VarDecl>(D)) {
    QualType Ty = V->getType();
    if (Ty->isBlockPointerType() || Ty->isFunctionPointerType()) {
      const FunctionType *FT = Ty->isFunctionPointerType() ? getFunctionType(D)
       : Ty->getAs<BlockPointerType>()->getPointeeType()->getAs<FunctionType>();
      if (!cast<FunctionProtoType>(FT)->isVariadic()) {
        int m = Ty->isFunctionPointerType() ? 0 : 1;
        S.Diag(Attr.getLoc(), diag::warn_attribute_sentinel_not_variadic) << m;
        return;
      }
    } else {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
        << Attr.getName() << ExpectedFunctionMethodOrBlock;
      return;
    }
  } else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionMethodOrBlock;
    return;
  }
  D->addAttr(::new (S.Context)
             SentinelAttr(Attr.getRange(), S.Context, sentinel, nullPos,
                          Attr.getAttributeSpellingListIndex()));
}

static void handleWarnUnusedResult(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  if (!isFunction(D) && !isa<ObjCMethodDecl>(D) && !isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionMethodOrClass;
    return;
  }

  if (isFunction(D) && getFunctionType(D)->getResultType()->isVoidType()) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_void_function_method)
      << Attr.getName() << 0;
    return;
  }
  if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D))
    if (MD->getResultType()->isVoidType()) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_void_function_method)
      << Attr.getName() << 1;
      return;
    }
  
  D->addAttr(::new (S.Context) 
             WarnUnusedResultAttr(Attr.getRange(), S.Context,
                                  Attr.getAttributeSpellingListIndex()));
}

static void handleWeakAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.hasParameterOrArguments()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  if (!isa<VarDecl>(D) && !isa<FunctionDecl>(D)) {
    if (isa<CXXRecordDecl>(D)) {
      D->addAttr(::new (S.Context) WeakAttr(Attr.getRange(), S.Context));
      return;
    }
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedVariableOrFunction;
    return;
  }

  NamedDecl *nd = cast<NamedDecl>(D);

  nd->addAttr(::new (S.Context)
              WeakAttr(Attr.getRange(), S.Context,
                       Attr.getAttributeSpellingListIndex()));
}

static void handleWeakImportAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;


  // weak_import only applies to variable & function declarations.
  bool isDef = false;
  if (!D->canBeWeakImported(isDef)) {
    if (isDef)
      S.Diag(Attr.getLoc(),
             diag::warn_attribute_weak_import_invalid_on_definition)
        << "weak_import" << 2 /*variable and function*/;
    else if (isa<ObjCPropertyDecl>(D) || isa<ObjCMethodDecl>(D) ||
             (S.Context.getTargetInfo().getTriple().isOSDarwin() &&
              (isa<ObjCInterfaceDecl>(D) || isa<EnumDecl>(D)))) {
      // Nothing to warn about here.
    } else
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
        << Attr.getName() << ExpectedVariableOrFunction;

    return;
  }

  D->addAttr(::new (S.Context)
             WeakImportAttr(Attr.getRange(), S.Context,
                            Attr.getAttributeSpellingListIndex()));
}

// Handles reqd_work_group_size and work_group_size_hint.
static void handleWorkGroupSize(Sema &S, Decl *D,
                                const AttributeList &Attr) {
  assert(Attr.getKind() == AttributeList::AT_ReqdWorkGroupSize 
      || Attr.getKind() == AttributeList::AT_WorkGroupSizeHint);

  // Attribute has 3 arguments.
  if (!checkAttributeNumArgs(S, Attr, 3)) return;

  unsigned WGSize[3];
  for (unsigned i = 0; i < 3; ++i) {
    Expr *E = Attr.getArg(i);
    llvm::APSInt ArgNum(32);
    if (E->isTypeDependent() || E->isValueDependent() ||
        !E->isIntegerConstantExpr(ArgNum, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_not_int)
        << Attr.getName()->getName() << E->getSourceRange();
      return;
    }
    WGSize[i] = (unsigned) ArgNum.getZExtValue();
  }

  if (Attr.getKind() == AttributeList::AT_ReqdWorkGroupSize
    && D->hasAttr<ReqdWorkGroupSizeAttr>()) {
      ReqdWorkGroupSizeAttr *A = D->getAttr<ReqdWorkGroupSizeAttr>();
      if (!(A->getXDim() == WGSize[0] &&
            A->getYDim() == WGSize[1] &&
            A->getZDim() == WGSize[2])) {
        S.Diag(Attr.getLoc(), diag::warn_duplicate_attribute) <<
          Attr.getName();
      }
  }

  if (Attr.getKind() == AttributeList::AT_WorkGroupSizeHint
    && D->hasAttr<WorkGroupSizeHintAttr>()) {
      WorkGroupSizeHintAttr *A = D->getAttr<WorkGroupSizeHintAttr>();
      if (!(A->getXDim() == WGSize[0] &&
            A->getYDim() == WGSize[1] &&
            A->getZDim() == WGSize[2])) {
        S.Diag(Attr.getLoc(), diag::warn_duplicate_attribute) <<
          Attr.getName();
      }
  }

  if (Attr.getKind() == AttributeList::AT_ReqdWorkGroupSize)
    D->addAttr(::new (S.Context)
                 ReqdWorkGroupSizeAttr(Attr.getRange(), S.Context,
                                       WGSize[0], WGSize[1], WGSize[2],
                                       Attr.getAttributeSpellingListIndex()));
  else
    D->addAttr(::new (S.Context)
                 WorkGroupSizeHintAttr(Attr.getRange(), S.Context,
                                       WGSize[0], WGSize[1], WGSize[2],
                                       Attr.getAttributeSpellingListIndex()));
}

static void handleVecTypeHint(Sema &S, Decl *D, const AttributeList &Attr) {
  assert(Attr.getKind() == AttributeList::AT_VecTypeHint);

  // Attribute has 1 argument.
  if (!checkAttributeNumArgs(S, Attr, 1))
    return;

  QualType ParmType = S.GetTypeFromParser(Attr.getTypeArg());

  if (!ParmType->isExtVectorType() && !ParmType->isFloatingType() &&
      (ParmType->isBooleanType() ||
       !ParmType->isIntegralType(S.getASTContext()))) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_vec_type_hint)
        << ParmType;
    return;
  }

  if (Attr.getKind() == AttributeList::AT_VecTypeHint &&
      D->hasAttr<VecTypeHintAttr>()) {
    VecTypeHintAttr *A = D->getAttr<VecTypeHintAttr>();
    if (A->getTypeHint() != ParmType) {
      S.Diag(Attr.getLoc(), diag::warn_duplicate_attribute) << Attr.getName();
      return;
    }
  }

  D->addAttr(::new (S.Context) VecTypeHintAttr(Attr.getLoc(), S.Context,
                                               ParmType, Attr.getLoc()));
}

static void handleEndianAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (!dyn_cast<VarDecl>(D))
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type) << "endian"
                                                                << 9;
  StringRef EndianType = Attr.getParameterName()->getName();
  if (EndianType != "host" && EndianType != "device")
    S.Diag(Attr.getLoc(), diag::warn_attribute_unknown_endian) << EndianType;
}

SectionAttr *Sema::mergeSectionAttr(Decl *D, SourceRange Range,
                                    StringRef Name,
                                    unsigned AttrSpellingListIndex) {
  if (SectionAttr *ExistingAttr = D->getAttr<SectionAttr>()) {
    if (ExistingAttr->getName() == Name)
      return NULL;
    Diag(ExistingAttr->getLocation(), diag::warn_mismatched_section);
    Diag(Range.getBegin(), diag::note_previous_attribute);
    return NULL;
  }
  return ::new (Context) SectionAttr(Range, Context, Name,
                                     AttrSpellingListIndex);
}

static void handleSectionAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // Attribute has no arguments.
  if (!checkAttributeNumArgs(S, Attr, 1))
    return;

  // Make sure that there is a string literal as the sections's single
  // argument.
  Expr *ArgExpr = Attr.getArg(0);
  StringLiteral *SE = dyn_cast<StringLiteral>(ArgExpr);
  if (!SE) {
    S.Diag(ArgExpr->getLocStart(), diag::err_attribute_not_string) << "section";
    return;
  }

  // If the target wants to validate the section specifier, make it happen.
  std::string Error = S.Context.getTargetInfo().isValidSectionSpecifier(SE->getString());
  if (!Error.empty()) {
    S.Diag(SE->getLocStart(), diag::err_attribute_section_invalid_for_target)
    << Error;
    return;
  }

  // This attribute cannot be applied to local variables.
  if (isa<VarDecl>(D) && cast<VarDecl>(D)->hasLocalStorage()) {
    S.Diag(SE->getLocStart(), diag::err_attribute_section_local_variable);
    return;
  }
  
  unsigned Index = Attr.getAttributeSpellingListIndex();
  SectionAttr *NewAttr = S.mergeSectionAttr(D, Attr.getRange(),
                                            SE->getString(), Index);
  if (NewAttr)
    D->addAttr(NewAttr);
}


static void handleNothrowAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.hasParameterOrArguments()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }
  
  if (NoThrowAttr *Existing = D->getAttr<NoThrowAttr>()) {
    if (Existing->getLocation().isInvalid())
      Existing->setRange(Attr.getRange());
  } else {
    D->addAttr(::new (S.Context)
               NoThrowAttr(Attr.getRange(), S.Context,
                           Attr.getAttributeSpellingListIndex()));
  }
}

static void handleConstAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.hasParameterOrArguments()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  if (ConstAttr *Existing = D->getAttr<ConstAttr>()) {
   if (Existing->getLocation().isInvalid())
     Existing->setRange(Attr.getRange());
  } else {
    D->addAttr(::new (S.Context)
               ConstAttr(Attr.getRange(), S.Context,
                         Attr.getAttributeSpellingListIndex() ));
  }
}

static void handlePureAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  D->addAttr(::new (S.Context)
             PureAttr(Attr.getRange(), S.Context,
                      Attr.getAttributeSpellingListIndex()));
}

static void handleCleanupAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (!Attr.getParameterName()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

  VarDecl *VD = dyn_cast<VarDecl>(D);

  if (!VD || !VD->hasLocalStorage()) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "cleanup";
    return;
  }

  // Look up the function
  // FIXME: Lookup probably isn't looking in the right place
  NamedDecl *CleanupDecl
    = S.LookupSingleName(S.TUScope, Attr.getParameterName(),
                         Attr.getParameterLoc(), Sema::LookupOrdinaryName);
  if (!CleanupDecl) {
    S.Diag(Attr.getParameterLoc(), diag::err_attribute_cleanup_arg_not_found) <<
      Attr.getParameterName();
    return;
  }

  FunctionDecl *FD = dyn_cast<FunctionDecl>(CleanupDecl);
  if (!FD) {
    S.Diag(Attr.getParameterLoc(),
           diag::err_attribute_cleanup_arg_not_function)
      << Attr.getParameterName();
    return;
  }

  if (FD->getNumParams() != 1) {
    S.Diag(Attr.getParameterLoc(),
           diag::err_attribute_cleanup_func_must_take_one_arg)
      << Attr.getParameterName();
    return;
  }

  // We're currently more strict than GCC about what function types we accept.
  // If this ever proves to be a problem it should be easy to fix.
  QualType Ty = S.Context.getPointerType(VD->getType());
  QualType ParamTy = FD->getParamDecl(0)->getType();
  if (S.CheckAssignmentConstraints(FD->getParamDecl(0)->getLocation(),
                                   ParamTy, Ty) != Sema::Compatible) {
    S.Diag(Attr.getParameterLoc(),
           diag::err_attribute_cleanup_func_arg_incompatible_type) <<
      Attr.getParameterName() << ParamTy << Ty;
    return;
  }

  D->addAttr(::new (S.Context)
             CleanupAttr(Attr.getRange(), S.Context, FD,
                         Attr.getAttributeSpellingListIndex()));
  S.MarkFunctionReferenced(Attr.getParameterLoc(), FD);
  S.DiagnoseUseOfDecl(FD, Attr.getParameterLoc());
}

/// Handle __attribute__((format_arg((idx)))) attribute based on
/// http://gcc.gnu.org/onlinedocs/gcc/Function-Attributes.html
static void handleFormatArgAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (!checkAttributeNumArgs(S, Attr, 1))
    return;

  if (!isFunctionOrMethod(D) || !hasFunctionProto(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return;
  }

  // In C++ the implicit 'this' function parameter also counts, and they are
  // counted from one.
  bool HasImplicitThisParam = isInstanceMethod(D);
  unsigned NumArgs  = getFunctionOrMethodNumArgs(D) + HasImplicitThisParam;
  unsigned FirstIdx = 1;

  // checks for the 2nd argument
  Expr *IdxExpr = Attr.getArg(0);
  llvm::APSInt Idx(32);
  if (IdxExpr->isTypeDependent() || IdxExpr->isValueDependent() ||
      !IdxExpr->isIntegerConstantExpr(Idx, S.Context)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_int)
    << "format" << 2 << IdxExpr->getSourceRange();
    return;
  }

  if (Idx.getZExtValue() < FirstIdx || Idx.getZExtValue() > NumArgs) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_out_of_bounds)
    << "format" << 2 << IdxExpr->getSourceRange();
    return;
  }

  unsigned ArgIdx = Idx.getZExtValue() - 1;

  if (HasImplicitThisParam) {
    if (ArgIdx == 0) {
      S.Diag(Attr.getLoc(), diag::err_attribute_invalid_implicit_this_argument)
        << "format_arg" << IdxExpr->getSourceRange();
      return;
    }
    ArgIdx--;
  }

  // make sure the format string is really a string
  QualType Ty = getFunctionOrMethodArgType(D, ArgIdx);

  bool not_nsstring_type = !isNSStringType(Ty, S.Context);
  if (not_nsstring_type &&
      !isCFStringType(Ty, S.Context) &&
      (!Ty->isPointerType() ||
       !Ty->getAs<PointerType>()->getPointeeType()->isCharType())) {
    // FIXME: Should highlight the actual expression that has the wrong type.
    S.Diag(Attr.getLoc(), diag::err_format_attribute_not)
    << (not_nsstring_type ? "a string type" : "an NSString")
       << IdxExpr->getSourceRange();
    return;
  }
  Ty = getFunctionOrMethodResultType(D);
  if (!isNSStringType(Ty, S.Context) &&
      !isCFStringType(Ty, S.Context) &&
      (!Ty->isPointerType() ||
       !Ty->getAs<PointerType>()->getPointeeType()->isCharType())) {
    // FIXME: Should highlight the actual expression that has the wrong type.
    S.Diag(Attr.getLoc(), diag::err_format_attribute_result_not)
    << (not_nsstring_type ? "string type" : "NSString")
       << IdxExpr->getSourceRange();
    return;
  }

  D->addAttr(::new (S.Context)
             FormatArgAttr(Attr.getRange(), S.Context, Idx.getZExtValue(),
                           Attr.getAttributeSpellingListIndex()));
}

enum FormatAttrKind {
  CFStringFormat,
  NSStringFormat,
  StrftimeFormat,
  SupportedFormat,
  IgnoredFormat,
  InvalidFormat
};

/// getFormatAttrKind - Map from format attribute names to supported format
/// types.
static FormatAttrKind getFormatAttrKind(StringRef Format) {
  return llvm::StringSwitch<FormatAttrKind>(Format)
    // Check for formats that get handled specially.
    .Case("NSString", NSStringFormat)
    .Case("CFString", CFStringFormat)
    .Case("strftime", StrftimeFormat)

    // Otherwise, check for supported formats.
    .Cases("scanf", "printf", "printf0", "strfmon", SupportedFormat)
    .Cases("cmn_err", "vcmn_err", "zcmn_err", SupportedFormat)
    .Case("kprintf", SupportedFormat) // OpenBSD.

    .Cases("gcc_diag", "gcc_cdiag", "gcc_cxxdiag", "gcc_tdiag", IgnoredFormat)
    .Default(InvalidFormat);
}

/// Handle __attribute__((init_priority(priority))) attributes based on
/// http://gcc.gnu.org/onlinedocs/gcc/C_002b_002b-Attributes.html
static void handleInitPriorityAttr(Sema &S, Decl *D,
                                   const AttributeList &Attr) {
  if (!S.getLangOpts().CPlusPlus) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << Attr.getName();
    return;
  }
  
  if (!isa<VarDecl>(D) || S.getCurFunctionOrMethodDecl()) {
    S.Diag(Attr.getLoc(), diag::err_init_priority_object_attr);
    Attr.setInvalid();
    return;
  }
  QualType T = dyn_cast<VarDecl>(D)->getType();
  if (S.Context.getAsArrayType(T))
    T = S.Context.getBaseElementType(T);
  if (!T->getAs<RecordType>()) {
    S.Diag(Attr.getLoc(), diag::err_init_priority_object_attr);
    Attr.setInvalid();
    return;
  }
  
  if (Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    Attr.setInvalid();
    return;
  }
  Expr *priorityExpr = Attr.getArg(0);
  
  llvm::APSInt priority(32);
  if (priorityExpr->isTypeDependent() || priorityExpr->isValueDependent() ||
      !priorityExpr->isIntegerConstantExpr(priority, S.Context)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_not_int)
    << "init_priority" << priorityExpr->getSourceRange();
    Attr.setInvalid();
    return;
  }
  unsigned prioritynum = priority.getZExtValue();
  if (prioritynum < 101 || prioritynum > 65535) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_outof_range)
    <<  priorityExpr->getSourceRange();
    Attr.setInvalid();
    return;
  }
  D->addAttr(::new (S.Context)
             InitPriorityAttr(Attr.getRange(), S.Context, prioritynum,
                              Attr.getAttributeSpellingListIndex()));
}

FormatAttr *Sema::mergeFormatAttr(Decl *D, SourceRange Range, StringRef Format,
                                  int FormatIdx, int FirstArg,
                                  unsigned AttrSpellingListIndex) {
  // Check whether we already have an equivalent format attribute.
  for (specific_attr_iterator<FormatAttr>
         i = D->specific_attr_begin<FormatAttr>(),
         e = D->specific_attr_end<FormatAttr>();
       i != e ; ++i) {
    FormatAttr *f = *i;
    if (f->getType() == Format &&
        f->getFormatIdx() == FormatIdx &&
        f->getFirstArg() == FirstArg) {
      // If we don't have a valid location for this attribute, adopt the
      // location.
      if (f->getLocation().isInvalid())
        f->setRange(Range);
      return NULL;
    }
  }

  return ::new (Context) FormatAttr(Range, Context, Format, FormatIdx, FirstArg,
                                    AttrSpellingListIndex);
}

/// Handle __attribute__((format(type,idx,firstarg))) attributes based on
/// http://gcc.gnu.org/onlinedocs/gcc/Function-Attributes.html
static void handleFormatAttr(Sema &S, Decl *D, const AttributeList &Attr) {

  if (!Attr.getParameterName()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string)
      << "format" << 1;
    return;
  }

  if (Attr.getNumArgs() != 2) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 3;
    return;
  }

  if (!isFunctionOrMethodOrBlock(D) || !hasFunctionProto(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return;
  }

  // In C++ the implicit 'this' function parameter also counts, and they are
  // counted from one.
  bool HasImplicitThisParam = isInstanceMethod(D);
  unsigned NumArgs  = getFunctionOrMethodNumArgs(D) + HasImplicitThisParam;
  unsigned FirstIdx = 1;

  StringRef Format = Attr.getParameterName()->getName();

  // Normalize the argument, __foo__ becomes foo.
  if (Format.startswith("__") && Format.endswith("__"))
    Format = Format.substr(2, Format.size() - 4);

  // Check for supported formats.
  FormatAttrKind Kind = getFormatAttrKind(Format);
  
  if (Kind == IgnoredFormat)
    return;
  
  if (Kind == InvalidFormat) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_type_not_supported)
      << "format" << Attr.getParameterName()->getName();
    return;
  }

  // checks for the 2nd argument
  Expr *IdxExpr = Attr.getArg(0);
  llvm::APSInt Idx(32);
  if (IdxExpr->isTypeDependent() || IdxExpr->isValueDependent() ||
      !IdxExpr->isIntegerConstantExpr(Idx, S.Context)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_int)
      << "format" << 2 << IdxExpr->getSourceRange();
    return;
  }

  if (Idx.getZExtValue() < FirstIdx || Idx.getZExtValue() > NumArgs) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_out_of_bounds)
      << "format" << 2 << IdxExpr->getSourceRange();
    return;
  }

  // FIXME: Do we need to bounds check?
  unsigned ArgIdx = Idx.getZExtValue() - 1;

  if (HasImplicitThisParam) {
    if (ArgIdx == 0) {
      S.Diag(Attr.getLoc(),
             diag::err_format_attribute_implicit_this_format_string)
        << IdxExpr->getSourceRange();
      return;
    }
    ArgIdx--;
  }

  // make sure the format string is really a string
  QualType Ty = getFunctionOrMethodArgType(D, ArgIdx);

  if (Kind == CFStringFormat) {
    if (!isCFStringType(Ty, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_format_attribute_not)
        << "a CFString" << IdxExpr->getSourceRange();
      return;
    }
  } else if (Kind == NSStringFormat) {
    // FIXME: do we need to check if the type is NSString*?  What are the
    // semantics?
    if (!isNSStringType(Ty, S.Context)) {
      // FIXME: Should highlight the actual expression that has the wrong type.
      S.Diag(Attr.getLoc(), diag::err_format_attribute_not)
        << "an NSString" << IdxExpr->getSourceRange();
      return;
    }
  } else if (!Ty->isPointerType() ||
             !Ty->getAs<PointerType>()->getPointeeType()->isCharType()) {
    // FIXME: Should highlight the actual expression that has the wrong type.
    S.Diag(Attr.getLoc(), diag::err_format_attribute_not)
      << "a string type" << IdxExpr->getSourceRange();
    return;
  }

  // check the 3rd argument
  Expr *FirstArgExpr = Attr.getArg(1);
  llvm::APSInt FirstArg(32);
  if (FirstArgExpr->isTypeDependent() || FirstArgExpr->isValueDependent() ||
      !FirstArgExpr->isIntegerConstantExpr(FirstArg, S.Context)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_int)
      << "format" << 3 << FirstArgExpr->getSourceRange();
    return;
  }

  // check if the function is variadic if the 3rd argument non-zero
  if (FirstArg != 0) {
    if (isFunctionOrMethodVariadic(D)) {
      ++NumArgs; // +1 for ...
    } else {
      S.Diag(D->getLocation(), diag::err_format_attribute_requires_variadic);
      return;
    }
  }

  // strftime requires FirstArg to be 0 because it doesn't read from any
  // variable the input is just the current time + the format string.
  if (Kind == StrftimeFormat) {
    if (FirstArg != 0) {
      S.Diag(Attr.getLoc(), diag::err_format_strftime_third_parameter)
        << FirstArgExpr->getSourceRange();
      return;
    }
  // if 0 it disables parameter checking (to use with e.g. va_list)
  } else if (FirstArg != 0 && FirstArg != NumArgs) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_out_of_bounds)
      << "format" << 3 << FirstArgExpr->getSourceRange();
    return;
  }

  FormatAttr *NewAttr = S.mergeFormatAttr(D, Attr.getRange(), Format,
                                          Idx.getZExtValue(),
                                          FirstArg.getZExtValue(),
                                          Attr.getAttributeSpellingListIndex());
  if (NewAttr)
    D->addAttr(NewAttr);
}

static void handleTransparentUnionAttr(Sema &S, Decl *D,
                                       const AttributeList &Attr) {
  // check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;


  // Try to find the underlying union declaration.
  RecordDecl *RD = 0;
  TypedefNameDecl *TD = dyn_cast<TypedefNameDecl>(D);
  if (TD && TD->getUnderlyingType()->isUnionType())
    RD = TD->getUnderlyingType()->getAsUnionType()->getDecl();
  else
    RD = dyn_cast<RecordDecl>(D);

  if (!RD || !RD->isUnion()) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedUnion;
    return;
  }

  if (!RD->isCompleteDefinition()) {
    S.Diag(Attr.getLoc(),
        diag::warn_transparent_union_attribute_not_definition);
    return;
  }

  RecordDecl::field_iterator Field = RD->field_begin(),
                          FieldEnd = RD->field_end();
  if (Field == FieldEnd) {
    S.Diag(Attr.getLoc(), diag::warn_transparent_union_attribute_zero_fields);
    return;
  }

  FieldDecl *FirstField = *Field;
  QualType FirstType = FirstField->getType();
  if (FirstType->hasFloatingRepresentation() || FirstType->isVectorType()) {
    S.Diag(FirstField->getLocation(),
           diag::warn_transparent_union_attribute_floating)
      << FirstType->isVectorType() << FirstType;
    return;
  }

  uint64_t FirstSize = S.Context.getTypeSize(FirstType);
  uint64_t FirstAlign = S.Context.getTypeAlign(FirstType);
  for (; Field != FieldEnd; ++Field) {
    QualType FieldType = Field->getType();
    if (S.Context.getTypeSize(FieldType) != FirstSize ||
        S.Context.getTypeAlign(FieldType) != FirstAlign) {
      // Warn if we drop the attribute.
      bool isSize = S.Context.getTypeSize(FieldType) != FirstSize;
      unsigned FieldBits = isSize? S.Context.getTypeSize(FieldType)
                                 : S.Context.getTypeAlign(FieldType);
      S.Diag(Field->getLocation(),
          diag::warn_transparent_union_attribute_field_size_align)
        << isSize << Field->getDeclName() << FieldBits;
      unsigned FirstBits = isSize? FirstSize : FirstAlign;
      S.Diag(FirstField->getLocation(),
             diag::note_transparent_union_first_field_size_align)
        << isSize << FirstBits;
      return;
    }
  }

  RD->addAttr(::new (S.Context)
              TransparentUnionAttr(Attr.getRange(), S.Context,
                                   Attr.getAttributeSpellingListIndex()));
}

static void handleAnnotateAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 1))
    return;

  Expr *ArgExpr = Attr.getArg(0);
  StringLiteral *SE = dyn_cast<StringLiteral>(ArgExpr);

  // Make sure that there is a string literal as the annotation's single
  // argument.
  if (!SE) {
    S.Diag(ArgExpr->getLocStart(), diag::err_attribute_not_string) <<"annotate";
    return;
  }

  // Don't duplicate annotations that are already set.
  for (specific_attr_iterator<AnnotateAttr>
       i = D->specific_attr_begin<AnnotateAttr>(),
       e = D->specific_attr_end<AnnotateAttr>(); i != e; ++i) {
      if ((*i)->getAnnotation() == SE->getString())
          return;
  }
  
  D->addAttr(::new (S.Context)
             AnnotateAttr(Attr.getRange(), S.Context, SE->getString(),
                          Attr.getAttributeSpellingListIndex()));
}

static void handleAlignedAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.getNumArgs() > 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

  if (Attr.getNumArgs() == 0) {
    D->addAttr(::new (S.Context) AlignedAttr(Attr.getRange(), S.Context,
               true, 0, Attr.getAttributeSpellingListIndex()));
    return;
  }

  Expr *E = Attr.getArg(0);
  if (Attr.isPackExpansion() && !E->containsUnexpandedParameterPack()) {
    S.Diag(Attr.getEllipsisLoc(),
           diag::err_pack_expansion_without_parameter_packs);
    return;
  }

  if (!Attr.isPackExpansion() && S.DiagnoseUnexpandedParameterPack(E))
    return;

  S.AddAlignedAttr(Attr.getRange(), D, E, Attr.getAttributeSpellingListIndex(),
                   Attr.isPackExpansion());
}

void Sema::AddAlignedAttr(SourceRange AttrRange, Decl *D, Expr *E,
                          unsigned SpellingListIndex, bool IsPackExpansion) {
  AlignedAttr TmpAttr(AttrRange, Context, true, E, SpellingListIndex);
  SourceLocation AttrLoc = AttrRange.getBegin();

  // C++11 alignas(...) and C11 _Alignas(...) have additional requirements.
  if (TmpAttr.isAlignas()) {
    // C++11 [dcl.align]p1:
    //   An alignment-specifier may be applied to a variable or to a class
    //   data member, but it shall not be applied to a bit-field, a function
    //   parameter, the formal parameter of a catch clause, or a variable
    //   declared with the register storage class specifier. An
    //   alignment-specifier may also be applied to the declaration of a class
    //   or enumeration type.
    // C11 6.7.5/2:
    //   An alignment attribute shall not be specified in a declaration of
    //   a typedef, or a bit-field, or a function, or a parameter, or an
    //   object declared with the register storage-class specifier.
    int DiagKind = -1;
    if (isa<ParmVarDecl>(D)) {
      DiagKind = 0;
    } else if (VarDecl *VD = dyn_cast<VarDecl>(D)) {
      if (VD->getStorageClass() == SC_Register)
        DiagKind = 1;
      if (VD->isExceptionVariable())
        DiagKind = 2;
    } else if (FieldDecl *FD = dyn_cast<FieldDecl>(D)) {
      if (FD->isBitField())
        DiagKind = 3;
    } else if (!isa<TagDecl>(D)) {
      Diag(AttrLoc, diag::err_attribute_wrong_decl_type)
        << (TmpAttr.isC11() ? "'_Alignas'" : "'alignas'")
        << (TmpAttr.isC11() ? ExpectedVariableOrField
                            : ExpectedVariableFieldOrTag);
      return;
    }
    if (DiagKind != -1) {
      Diag(AttrLoc, diag::err_alignas_attribute_wrong_decl_type)
        << TmpAttr.isC11() << DiagKind;
      return;
    }
  }

  if (E->isTypeDependent() || E->isValueDependent()) {
    // Save dependent expressions in the AST to be instantiated.
    AlignedAttr *AA = ::new (Context) AlignedAttr(TmpAttr);
    AA->setPackExpansion(IsPackExpansion);
    D->addAttr(AA);
    return;
  }

  // FIXME: Cache the number on the Attr object?
  llvm::APSInt Alignment(32);
  ExprResult ICE
    = VerifyIntegerConstantExpression(E, &Alignment,
        diag::err_aligned_attribute_argument_not_int,
        /*AllowFold*/ false);
  if (ICE.isInvalid())
    return;

  // C++11 [dcl.align]p2:
  //   -- if the constant expression evaluates to zero, the alignment
  //      specifier shall have no effect
  // C11 6.7.5p6:
  //   An alignment specification of zero has no effect.
  if (!(TmpAttr.isAlignas() && !Alignment) &&
      !llvm::isPowerOf2_64(Alignment.getZExtValue())) {
    Diag(AttrLoc, diag::err_attribute_aligned_not_power_of_two)
      << E->getSourceRange();
    return;
  }

  if (TmpAttr.isDeclspec()) {
    // We've already verified it's a power of 2, now let's make sure it's
    // 8192 or less.
    if (Alignment.getZExtValue() > 8192) {
      Diag(AttrLoc, diag::err_attribute_aligned_greater_than_8192)
        << E->getSourceRange();
      return;
    }
  }

  AlignedAttr *AA = ::new (Context) AlignedAttr(AttrRange, Context, true,
                                                ICE.take(), SpellingListIndex);
  AA->setPackExpansion(IsPackExpansion);
  D->addAttr(AA);
}

void Sema::AddAlignedAttr(SourceRange AttrRange, Decl *D, TypeSourceInfo *TS,
                          unsigned SpellingListIndex, bool IsPackExpansion) {
  // FIXME: Cache the number on the Attr object if non-dependent?
  // FIXME: Perform checking of type validity
  AlignedAttr *AA = ::new (Context) AlignedAttr(AttrRange, Context, false, TS,
                                                SpellingListIndex);
  AA->setPackExpansion(IsPackExpansion);
  D->addAttr(AA);
}

void Sema::CheckAlignasUnderalignment(Decl *D) {
  assert(D->hasAttrs() && "no attributes on decl");

  QualType Ty;
  if (ValueDecl *VD = dyn_cast<ValueDecl>(D))
    Ty = VD->getType();
  else
    Ty = Context.getTagDeclType(cast<TagDecl>(D));
  if (Ty->isDependentType() || Ty->isIncompleteType())
    return;

  // C++11 [dcl.align]p5, C11 6.7.5/4:
  //   The combined effect of all alignment attributes in a declaration shall
  //   not specify an alignment that is less strict than the alignment that
  //   would otherwise be required for the entity being declared.
  AlignedAttr *AlignasAttr = 0;
  unsigned Align = 0;
  for (specific_attr_iterator<AlignedAttr>
         I = D->specific_attr_begin<AlignedAttr>(),
         E = D->specific_attr_end<AlignedAttr>(); I != E; ++I) {
    if (I->isAlignmentDependent())
      return;
    if (I->isAlignas())
      AlignasAttr = *I;
    Align = std::max(Align, I->getAlignment(Context));
  }

  if (AlignasAttr && Align) {
    CharUnits RequestedAlign = Context.toCharUnitsFromBits(Align);
    CharUnits NaturalAlign = Context.getTypeAlignInChars(Ty);
    if (NaturalAlign > RequestedAlign)
      Diag(AlignasAttr->getLocation(), diag::err_alignas_underaligned)
        << Ty << (unsigned)NaturalAlign.getQuantity();
  }
}

/// handleModeAttr - This attribute modifies the width of a decl with primitive
/// type.
///
/// Despite what would be logical, the mode attribute is a decl attribute, not a
/// type attribute: 'int ** __attribute((mode(HI))) *G;' tries to make 'G' be
/// HImode, not an intermediate pointer.
static void handleModeAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // This attribute isn't documented, but glibc uses it.  It changes
  // the width of an int or unsigned int to the specified size.

  // Check that there aren't any arguments
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;


  IdentifierInfo *Name = Attr.getParameterName();
  if (!Name) {
    S.Diag(Attr.getLoc(), diag::err_attribute_missing_parameter_name);
    return;
  }

  StringRef Str = Attr.getParameterName()->getName();

  // Normalize the attribute name, __foo__ becomes foo.
  if (Str.startswith("__") && Str.endswith("__"))
    Str = Str.substr(2, Str.size() - 4);

  unsigned DestWidth = 0;
  bool IntegerMode = true;
  bool ComplexMode = false;
  switch (Str.size()) {
  case 2:
    switch (Str[0]) {
    case 'Q': DestWidth = 8; break;
    case 'H': DestWidth = 16; break;
    case 'S': DestWidth = 32; break;
    case 'D': DestWidth = 64; break;
    case 'X': DestWidth = 96; break;
    case 'T': DestWidth = 128; break;
    }
    if (Str[1] == 'F') {
      IntegerMode = false;
    } else if (Str[1] == 'C') {
      IntegerMode = false;
      ComplexMode = true;
    } else if (Str[1] != 'I') {
      DestWidth = 0;
    }
    break;
  case 4:
    // FIXME: glibc uses 'word' to define register_t; this is narrower than a
    // pointer on PIC16 and other embedded platforms.
    if (Str == "word")
      DestWidth = S.Context.getTargetInfo().getPointerWidth(0);
    else if (Str == "byte")
      DestWidth = S.Context.getTargetInfo().getCharWidth();
    break;
  case 7:
    if (Str == "pointer")
      DestWidth = S.Context.getTargetInfo().getPointerWidth(0);
    break;
  case 11:
    if (Str == "unwind_word")
      DestWidth = S.Context.getTargetInfo().getUnwindWordWidth();
    break;
  }

  QualType OldTy;
  if (TypedefNameDecl *TD = dyn_cast<TypedefNameDecl>(D))
    OldTy = TD->getUnderlyingType();
  else if (ValueDecl *VD = dyn_cast<ValueDecl>(D))
    OldTy = VD->getType();
  else {
    S.Diag(D->getLocation(), diag::err_attr_wrong_decl)
      << "mode" << Attr.getRange();
    return;
  }

  if (!OldTy->getAs<BuiltinType>() && !OldTy->isComplexType())
    S.Diag(Attr.getLoc(), diag::err_mode_not_primitive);
  else if (IntegerMode) {
    if (!OldTy->isIntegralOrEnumerationType())
      S.Diag(Attr.getLoc(), diag::err_mode_wrong_type);
  } else if (ComplexMode) {
    if (!OldTy->isComplexType())
      S.Diag(Attr.getLoc(), diag::err_mode_wrong_type);
  } else {
    if (!OldTy->isFloatingType())
      S.Diag(Attr.getLoc(), diag::err_mode_wrong_type);
  }

  // FIXME: Sync this with InitializePredefinedMacros; we need to match int8_t
  // and friends, at least with glibc.
  // FIXME: Make sure 32/64-bit integers don't get defined to types of the wrong
  // width on unusual platforms.
  // FIXME: Make sure floating-point mappings are accurate
  // FIXME: Support XF and TF types
  QualType NewTy;
  switch (DestWidth) {
  case 0:
    S.Diag(Attr.getLoc(), diag::err_unknown_machine_mode) << Name;
    return;
  default:
    S.Diag(Attr.getLoc(), diag::err_unsupported_machine_mode) << Name;
    return;
  case 8:
    if (!IntegerMode) {
      S.Diag(Attr.getLoc(), diag::err_unsupported_machine_mode) << Name;
      return;
    }
    if (OldTy->isSignedIntegerType())
      NewTy = S.Context.SignedCharTy;
    else
      NewTy = S.Context.UnsignedCharTy;
    break;
  case 16:
    if (!IntegerMode) {
      S.Diag(Attr.getLoc(), diag::err_unsupported_machine_mode) << Name;
      return;
    }
    if (OldTy->isSignedIntegerType())
      NewTy = S.Context.ShortTy;
    else
      NewTy = S.Context.UnsignedShortTy;
    break;
  case 32:
    if (!IntegerMode)
      NewTy = S.Context.FloatTy;
    else if (OldTy->isSignedIntegerType())
      NewTy = S.Context.IntTy;
    else
      NewTy = S.Context.UnsignedIntTy;
    break;
  case 64:
    if (!IntegerMode)
      NewTy = S.Context.DoubleTy;
    else if (OldTy->isSignedIntegerType())
      if (S.Context.getTargetInfo().getLongWidth() == 64)
        NewTy = S.Context.LongTy;
      else
        NewTy = S.Context.LongLongTy;
    else
      if (S.Context.getTargetInfo().getLongWidth() == 64)
        NewTy = S.Context.UnsignedLongTy;
      else
        NewTy = S.Context.UnsignedLongLongTy;
    break;
  case 96:
    NewTy = S.Context.LongDoubleTy;
    break;
  case 128:
    if (!IntegerMode) {
      S.Diag(Attr.getLoc(), diag::err_unsupported_machine_mode) << Name;
      return;
    }
    if (OldTy->isSignedIntegerType())
      NewTy = S.Context.Int128Ty;
    else
      NewTy = S.Context.UnsignedInt128Ty;
    break;
  }

  if (ComplexMode) {
    NewTy = S.Context.getComplexType(NewTy);
  }

  // Install the new type.
  if (TypedefNameDecl *TD = dyn_cast<TypedefNameDecl>(D)) {
    // FIXME: preserve existing source info.
    TD->setTypeSourceInfo(S.Context.getTrivialTypeSourceInfo(NewTy));
  } else
    cast<ValueDecl>(D)->setType(NewTy);
}

static void handleNoDebugAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
    if (!VD->hasGlobalStorage())
      S.Diag(Attr.getLoc(),
             diag::warn_attribute_requires_functions_or_static_globals)
        << Attr.getName();
  } else if (!isFunctionOrMethod(D)) {
    S.Diag(Attr.getLoc(),
           diag::warn_attribute_requires_functions_or_static_globals)
      << Attr.getName();
    return;
  }

  D->addAttr(::new (S.Context)
             NoDebugAttr(Attr.getRange(), S.Context,
                         Attr.getAttributeSpellingListIndex()));
}

static void handleNoInlineAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;


  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return;
  }

  D->addAttr(::new (S.Context)
             NoInlineAttr(Attr.getRange(), S.Context,
             Attr.getAttributeSpellingListIndex()));
}

static void handleNoInstrumentFunctionAttr(Sema &S, Decl *D,
                                           const AttributeList &Attr) {
  // check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;


  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return;
  }

  D->addAttr(::new (S.Context)
             NoInstrumentFunctionAttr(Attr.getRange(), S.Context,
                                      Attr.getAttributeSpellingListIndex()));
}

static void handleConstantAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (S.LangOpts.CUDA) {
    // check the attribute arguments.
    if (Attr.hasParameterOrArguments()) {
      S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
      return;
    }

    if (!isa<VarDecl>(D)) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
        << Attr.getName() << ExpectedVariable;
      return;
    }

    D->addAttr(::new (S.Context)
               CUDAConstantAttr(Attr.getRange(), S.Context,
                                Attr.getAttributeSpellingListIndex()));
  } else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "constant";
  }
}

static void handleDeviceAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (S.LangOpts.CUDA) {
    // check the attribute arguments.
    if (Attr.getNumArgs() != 0) {
      S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
      return;
    }

    if (!isa<FunctionDecl>(D) && !isa<VarDecl>(D)) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
        << Attr.getName() << ExpectedVariableOrFunction;
      return;
    }

    D->addAttr(::new (S.Context)
               CUDADeviceAttr(Attr.getRange(), S.Context,
                              Attr.getAttributeSpellingListIndex()));
  } else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "device";
  }
}

static void handleGlobalAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (S.LangOpts.CUDA) {
    // check the attribute arguments.
    if (!checkAttributeNumArgs(S, Attr, 0))
      return;

    if (!isa<FunctionDecl>(D)) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
        << Attr.getName() << ExpectedFunction;
      return;
    }

    FunctionDecl *FD = cast<FunctionDecl>(D);
    if (!FD->getResultType()->isVoidType()) {
      TypeLoc TL = FD->getTypeSourceInfo()->getTypeLoc().IgnoreParens();
      if (FunctionTypeLoc FTL = TL.getAs<FunctionTypeLoc>()) {
        S.Diag(FD->getTypeSpecStartLoc(), diag::err_kern_type_not_void_return)
          << FD->getType()
          << FixItHint::CreateReplacement(FTL.getResultLoc().getSourceRange(),
                                          "void");
      } else {
        S.Diag(FD->getTypeSpecStartLoc(), diag::err_kern_type_not_void_return)
          << FD->getType();
      }
      return;
    }

    D->addAttr(::new (S.Context)
               CUDAGlobalAttr(Attr.getRange(), S.Context,
                              Attr.getAttributeSpellingListIndex()));
  } else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "global";
  }
}

static void handleHostAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (S.LangOpts.CUDA) {
    // check the attribute arguments.
    if (!checkAttributeNumArgs(S, Attr, 0))
      return;


    if (!isa<FunctionDecl>(D)) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
        << Attr.getName() << ExpectedFunction;
      return;
    }

    D->addAttr(::new (S.Context)
               CUDAHostAttr(Attr.getRange(), S.Context,
                            Attr.getAttributeSpellingListIndex()));
  } else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "host";
  }
}

static void handleSharedAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (S.LangOpts.CUDA) {
    // check the attribute arguments.
    if (!checkAttributeNumArgs(S, Attr, 0))
      return;

    if (!isa<VarDecl>(D)) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
        << Attr.getName() << ExpectedVariable;
      return;
    }

    D->addAttr(::new (S.Context)
               CUDASharedAttr(Attr.getRange(), S.Context,
                              Attr.getAttributeSpellingListIndex()));
  } else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "shared";
  }
}

static void handleGNUInlineAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  FunctionDecl *Fn = dyn_cast<FunctionDecl>(D);
  if (Fn == 0) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return;
  }

  if (!Fn->isInlineSpecified()) {
    S.Diag(Attr.getLoc(), diag::warn_gnu_inline_attribute_requires_inline);
    return;
  }

  D->addAttr(::new (S.Context)
             GNUInlineAttr(Attr.getRange(), S.Context,
                           Attr.getAttributeSpellingListIndex()));
}

static void handleCallConvAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (hasDeclarator(D)) return;

  const FunctionDecl *FD = dyn_cast<FunctionDecl>(D);
  // Diagnostic is emitted elsewhere: here we store the (valid) Attr
  // in the Decl node for syntactic reasoning, e.g., pretty-printing.
  CallingConv CC;
  if (S.CheckCallingConvAttr(Attr, CC, FD))
    return;

  if (!isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrMethod;
    return;
  }

  switch (Attr.getKind()) {
  case AttributeList::AT_FastCall:
    D->addAttr(::new (S.Context)
               FastCallAttr(Attr.getRange(), S.Context,
                            Attr.getAttributeSpellingListIndex()));
    return;
  case AttributeList::AT_StdCall:
    D->addAttr(::new (S.Context)
               StdCallAttr(Attr.getRange(), S.Context,
                           Attr.getAttributeSpellingListIndex()));
    return;
  case AttributeList::AT_ThisCall:
    D->addAttr(::new (S.Context)
               ThisCallAttr(Attr.getRange(), S.Context,
                            Attr.getAttributeSpellingListIndex()));
    return;
  case AttributeList::AT_CDecl:
    D->addAttr(::new (S.Context)
               CDeclAttr(Attr.getRange(), S.Context,
                         Attr.getAttributeSpellingListIndex()));
    return;
  case AttributeList::AT_Pascal:
    D->addAttr(::new (S.Context)
               PascalAttr(Attr.getRange(), S.Context,
                          Attr.getAttributeSpellingListIndex()));
    return;
  case AttributeList::AT_Pcs: {
    PcsAttr::PCSType PCS;
    switch (CC) {
    case CC_AAPCS:
      PCS = PcsAttr::AAPCS;
      break;
    case CC_AAPCS_VFP:
      PCS = PcsAttr::AAPCS_VFP;
      break;
    default:
      llvm_unreachable("unexpected calling convention in pcs attribute");
    }

    D->addAttr(::new (S.Context)
               PcsAttr(Attr.getRange(), S.Context, PCS,
                       Attr.getAttributeSpellingListIndex()));
    return;
  }
  case AttributeList::AT_PnaclCall:
    D->addAttr(::new (S.Context)
               PnaclCallAttr(Attr.getRange(), S.Context,
                             Attr.getAttributeSpellingListIndex()));
    return;
  case AttributeList::AT_IntelOclBicc:
    D->addAttr(::new (S.Context)
               IntelOclBiccAttr(Attr.getRange(), S.Context,
                                Attr.getAttributeSpellingListIndex()));
    return;

  default:
    llvm_unreachable("unexpected attribute kind");
  }
}

static void handleOpenCLKernelAttr(Sema &S, Decl *D, const AttributeList &Attr){
  assert(!Attr.isInvalid());
  D->addAttr(::new (S.Context) OpenCLKernelAttr(Attr.getRange(), S.Context));
}

static void handleOpenCLImageAccessAttr(Sema &S, Decl *D, const AttributeList &Attr){
  assert(!Attr.isInvalid());

  Expr *E = Attr.getArg(0);
  llvm::APSInt ArgNum(32);
  if (E->isTypeDependent() || E->isValueDependent() ||
      !E->isIntegerConstantExpr(ArgNum, S.Context)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_not_int)
      << Attr.getName()->getName() << E->getSourceRange();
    return;
  }

  D->addAttr(::new (S.Context) OpenCLImageAccessAttr(
    Attr.getRange(), S.Context, ArgNum.getZExtValue()));
}

bool Sema::CheckCallingConvAttr(const AttributeList &attr, CallingConv &CC, 
                                const FunctionDecl *FD) {
  if (attr.isInvalid())
    return true;

  unsigned ReqArgs = attr.getKind() == AttributeList::AT_Pcs ? 1 : 0;
  if (attr.getNumArgs() != ReqArgs || attr.getParameterName()) {
    Diag(attr.getLoc(), diag::err_attribute_wrong_number_arguments) << ReqArgs;
    attr.setInvalid();
    return true;
  }

  // TODO: diagnose uses of these conventions on the wrong target. Or, better
  // move to TargetAttributesSema one day.
  switch (attr.getKind()) {
  case AttributeList::AT_CDecl: CC = CC_C; break;
  case AttributeList::AT_FastCall: CC = CC_X86FastCall; break;
  case AttributeList::AT_StdCall: CC = CC_X86StdCall; break;
  case AttributeList::AT_ThisCall: CC = CC_X86ThisCall; break;
  case AttributeList::AT_Pascal: CC = CC_X86Pascal; break;
  case AttributeList::AT_Pcs: {
    Expr *Arg = attr.getArg(0);
    StringLiteral *Str = dyn_cast<StringLiteral>(Arg);
    if (!Str || !Str->isAscii()) {
      Diag(attr.getLoc(), diag::err_attribute_argument_n_not_string)
        << "pcs" << 1;
      attr.setInvalid();
      return true;
    }

    StringRef StrRef = Str->getString();
    if (StrRef == "aapcs") {
      CC = CC_AAPCS;
      break;
    } else if (StrRef == "aapcs-vfp") {
      CC = CC_AAPCS_VFP;
      break;
    }

    attr.setInvalid();
    Diag(attr.getLoc(), diag::err_invalid_pcs);
    return true;
  }
  case AttributeList::AT_PnaclCall: CC = CC_PnaclCall; break;
  case AttributeList::AT_IntelOclBicc: CC = CC_IntelOclBicc; break;
  default: llvm_unreachable("unexpected attribute kind");
  }

  const TargetInfo &TI = Context.getTargetInfo();
  TargetInfo::CallingConvCheckResult A = TI.checkCallingConvention(CC);
  if (A == TargetInfo::CCCR_Warning) {
    Diag(attr.getLoc(), diag::warn_cconv_ignored) << attr.getName();

    TargetInfo::CallingConvMethodType MT = TargetInfo::CCMT_Unknown;
    if (FD)
      MT = FD->isCXXInstanceMember() ? TargetInfo::CCMT_Member : 
                                    TargetInfo::CCMT_NonMember;
    CC = TI.getDefaultCallingConv(MT);
  }

  return false;
}

static void handleRegparmAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (hasDeclarator(D)) return;

  unsigned numParams;
  if (S.CheckRegparmAttr(Attr, numParams))
    return;

  if (!isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrMethod;
    return;
  }

  D->addAttr(::new (S.Context)
             RegparmAttr(Attr.getRange(), S.Context, numParams,
                         Attr.getAttributeSpellingListIndex()));
}

/// Checks a regparm attribute, returning true if it is ill-formed and
/// otherwise setting numParams to the appropriate value.
bool Sema::CheckRegparmAttr(const AttributeList &Attr, unsigned &numParams) {
  if (Attr.isInvalid())
    return true;

  if (Attr.getNumArgs() != 1) {
    Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    Attr.setInvalid();
    return true;
  }

  Expr *NumParamsExpr = Attr.getArg(0);
  llvm::APSInt NumParams(32);
  if (NumParamsExpr->isTypeDependent() || NumParamsExpr->isValueDependent() ||
      !NumParamsExpr->isIntegerConstantExpr(NumParams, Context)) {
    Diag(Attr.getLoc(), diag::err_attribute_argument_not_int)
      << "regparm" << NumParamsExpr->getSourceRange();
    Attr.setInvalid();
    return true;
  }

  if (Context.getTargetInfo().getRegParmMax() == 0) {
    Diag(Attr.getLoc(), diag::err_attribute_regparm_wrong_platform)
      << NumParamsExpr->getSourceRange();
    Attr.setInvalid();
    return true;
  }

  numParams = NumParams.getZExtValue();
  if (numParams > Context.getTargetInfo().getRegParmMax()) {
    Diag(Attr.getLoc(), diag::err_attribute_regparm_invalid_number)
      << Context.getTargetInfo().getRegParmMax() << NumParamsExpr->getSourceRange();
    Attr.setInvalid();
    return true;
  }

  return false;
}

static void handleLaunchBoundsAttr(Sema &S, Decl *D, const AttributeList &Attr){
  if (S.LangOpts.CUDA) {
    // check the attribute arguments.
    if (Attr.getNumArgs() != 1 && Attr.getNumArgs() != 2) {
      // FIXME: 0 is not okay.
      S.Diag(Attr.getLoc(), diag::err_attribute_too_many_arguments) << 2;
      return;
    }

    if (!isFunctionOrMethod(D)) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
        << Attr.getName() << ExpectedFunctionOrMethod;
      return;
    }

    Expr *MaxThreadsExpr = Attr.getArg(0);
    llvm::APSInt MaxThreads(32);
    if (MaxThreadsExpr->isTypeDependent() ||
        MaxThreadsExpr->isValueDependent() ||
        !MaxThreadsExpr->isIntegerConstantExpr(MaxThreads, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_int)
        << "launch_bounds" << 1 << MaxThreadsExpr->getSourceRange();
      return;
    }

    llvm::APSInt MinBlocks(32);
    if (Attr.getNumArgs() > 1) {
      Expr *MinBlocksExpr = Attr.getArg(1);
      if (MinBlocksExpr->isTypeDependent() ||
          MinBlocksExpr->isValueDependent() ||
          !MinBlocksExpr->isIntegerConstantExpr(MinBlocks, S.Context)) {
        S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_int)
          << "launch_bounds" << 2 << MinBlocksExpr->getSourceRange();
        return;
      }
    }

    D->addAttr(::new (S.Context)
               CUDALaunchBoundsAttr(Attr.getRange(), S.Context,
                                    MaxThreads.getZExtValue(),
                                    MinBlocks.getZExtValue(),
                                    Attr.getAttributeSpellingListIndex()));
  } else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "launch_bounds";
  }
}

static void handleArgumentWithTypeTagAttr(Sema &S, Decl *D,
                                          const AttributeList &Attr) {
  StringRef AttrName = Attr.getName()->getName();
  if (!Attr.getParameterName()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_identifier)
      << Attr.getName() << /* arg num = */ 1;
    return;
  }

  if (Attr.getNumArgs() != 2) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments)
      << /* required args = */ 3;
    return;
  }

  IdentifierInfo *ArgumentKind = Attr.getParameterName();

  if (!isFunctionOrMethod(D) || !hasFunctionProto(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrMethod;
    return;
  }

  uint64_t ArgumentIdx;
  if (!checkFunctionOrMethodArgumentIndex(S, D, AttrName,
                                          Attr.getLoc(), 2,
                                          Attr.getArg(0), ArgumentIdx))
    return;

  uint64_t TypeTagIdx;
  if (!checkFunctionOrMethodArgumentIndex(S, D, AttrName,
                                          Attr.getLoc(), 3,
                                          Attr.getArg(1), TypeTagIdx))
    return;

  bool IsPointer = (AttrName == "pointer_with_type_tag");
  if (IsPointer) {
    // Ensure that buffer has a pointer type.
    QualType BufferTy = getFunctionOrMethodArgType(D, ArgumentIdx);
    if (!BufferTy->isPointerType()) {
      S.Diag(Attr.getLoc(), diag::err_attribute_pointers_only)
        << AttrName;
    }
  }

  D->addAttr(::new (S.Context)
             ArgumentWithTypeTagAttr(Attr.getRange(), S.Context, ArgumentKind,
                                     ArgumentIdx, TypeTagIdx, IsPointer,
                                     Attr.getAttributeSpellingListIndex()));
}

static void handleTypeTagForDatatypeAttr(Sema &S, Decl *D,
                                         const AttributeList &Attr) {
  IdentifierInfo *PointerKind = Attr.getParameterName();
  if (!PointerKind) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_identifier)
      << "type_tag_for_datatype" << 1;
    return;
  }

  QualType MatchingCType = S.GetTypeFromParser(Attr.getMatchingCType(), NULL);

  D->addAttr(::new (S.Context)
             TypeTagForDatatypeAttr(Attr.getRange(), S.Context, PointerKind,
                                    MatchingCType,
                                    Attr.getLayoutCompatible(),
                                    Attr.getMustBeNull(),
                                    Attr.getAttributeSpellingListIndex()));
}

//===----------------------------------------------------------------------===//
// Checker-specific attribute handlers.
//===----------------------------------------------------------------------===//

static bool isValidSubjectOfNSAttribute(Sema &S, QualType type) {
  return type->isDependentType() || 
         type->isObjCObjectPointerType() || 
         S.Context.isObjCNSObjectType(type);
}
static bool isValidSubjectOfCFAttribute(Sema &S, QualType type) {
  return type->isDependentType() || 
         type->isPointerType() || 
         isValidSubjectOfNSAttribute(S, type);
}

static void handleNSConsumedAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  ParmVarDecl *param = dyn_cast<ParmVarDecl>(D);
  if (!param) {
    S.Diag(D->getLocStart(), diag::warn_attribute_wrong_decl_type)
      << Attr.getRange() << Attr.getName() << ExpectedParameter;
    return;
  }

  bool typeOK, cf;
  if (Attr.getKind() == AttributeList::AT_NSConsumed) {
    typeOK = isValidSubjectOfNSAttribute(S, param->getType());
    cf = false;
  } else {
    typeOK = isValidSubjectOfCFAttribute(S, param->getType());
    cf = true;
  }

  if (!typeOK) {
    S.Diag(D->getLocStart(), diag::warn_ns_attribute_wrong_parameter_type)
      << Attr.getRange() << Attr.getName() << cf;
    return;
  }

  if (cf)
    param->addAttr(::new (S.Context)
                   CFConsumedAttr(Attr.getRange(), S.Context,
                                  Attr.getAttributeSpellingListIndex()));
  else
    param->addAttr(::new (S.Context)
                   NSConsumedAttr(Attr.getRange(), S.Context,
                                  Attr.getAttributeSpellingListIndex()));
}

static void handleNSConsumesSelfAttr(Sema &S, Decl *D,
                                     const AttributeList &Attr) {
  if (!isa<ObjCMethodDecl>(D)) {
    S.Diag(D->getLocStart(), diag::warn_attribute_wrong_decl_type)
      << Attr.getRange() << Attr.getName() << ExpectedMethod;
    return;
  }

  D->addAttr(::new (S.Context)
             NSConsumesSelfAttr(Attr.getRange(), S.Context,
                                Attr.getAttributeSpellingListIndex()));
}

static void handleNSReturnsRetainedAttr(Sema &S, Decl *D,
                                        const AttributeList &Attr) {

  QualType returnType;

  if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D))
    returnType = MD->getResultType();
  else if (S.getLangOpts().ObjCAutoRefCount && hasDeclarator(D) &&
           (Attr.getKind() == AttributeList::AT_NSReturnsRetained))
    return; // ignore: was handled as a type attribute
  else if (ObjCPropertyDecl *PD = dyn_cast<ObjCPropertyDecl>(D))
    returnType = PD->getType();
  else if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
    returnType = FD->getResultType();
  else {
    S.Diag(D->getLocStart(), diag::warn_attribute_wrong_decl_type)
        << Attr.getRange() << Attr.getName()
        << ExpectedFunctionOrMethod;
    return;
  }

  bool typeOK;
  bool cf;
  switch (Attr.getKind()) {
  default: llvm_unreachable("invalid ownership attribute");
  case AttributeList::AT_NSReturnsAutoreleased:
  case AttributeList::AT_NSReturnsRetained:
  case AttributeList::AT_NSReturnsNotRetained:
    typeOK = isValidSubjectOfNSAttribute(S, returnType);
    cf = false;
    break;

  case AttributeList::AT_CFReturnsRetained:
  case AttributeList::AT_CFReturnsNotRetained:
    typeOK = isValidSubjectOfCFAttribute(S, returnType);
    cf = true;
    break;
  }

  if (!typeOK) {
    S.Diag(D->getLocStart(), diag::warn_ns_attribute_wrong_return_type)
      << Attr.getRange() << Attr.getName() << isa<ObjCMethodDecl>(D) << cf;
    return;
  }

  switch (Attr.getKind()) {
    default:
      llvm_unreachable("invalid ownership attribute");
    case AttributeList::AT_NSReturnsAutoreleased:
      D->addAttr(::new (S.Context)
                 NSReturnsAutoreleasedAttr(Attr.getRange(), S.Context,
                                           Attr.getAttributeSpellingListIndex()));
      return;
    case AttributeList::AT_CFReturnsNotRetained:
      D->addAttr(::new (S.Context)
                 CFReturnsNotRetainedAttr(Attr.getRange(), S.Context,
                                          Attr.getAttributeSpellingListIndex()));
      return;
    case AttributeList::AT_NSReturnsNotRetained:
      D->addAttr(::new (S.Context)
                 NSReturnsNotRetainedAttr(Attr.getRange(), S.Context,
                                          Attr.getAttributeSpellingListIndex()));
      return;
    case AttributeList::AT_CFReturnsRetained:
      D->addAttr(::new (S.Context)
                 CFReturnsRetainedAttr(Attr.getRange(), S.Context,
                                       Attr.getAttributeSpellingListIndex()));
      return;
    case AttributeList::AT_NSReturnsRetained:
      D->addAttr(::new (S.Context)
                 NSReturnsRetainedAttr(Attr.getRange(), S.Context,
                                       Attr.getAttributeSpellingListIndex()));
      return;
  };
}

static void handleObjCReturnsInnerPointerAttr(Sema &S, Decl *D,
                                              const AttributeList &attr) {
  SourceLocation loc = attr.getLoc();

  ObjCMethodDecl *method = dyn_cast<ObjCMethodDecl>(D);

  if (!method) {
    S.Diag(D->getLocStart(), diag::err_attribute_wrong_decl_type)
      << SourceRange(loc, loc) << attr.getName() << ExpectedMethod;
    return;
  }

  // Check that the method returns a normal pointer.
  QualType resultType = method->getResultType();
    
  if (!resultType->isReferenceType() &&
      (!resultType->isPointerType() || resultType->isObjCRetainableType())) {
    S.Diag(method->getLocStart(), diag::warn_ns_attribute_wrong_return_type)
      << SourceRange(loc)
      << attr.getName() << /*method*/ 1 << /*non-retainable pointer*/ 2;

    // Drop the attribute.
    return;
  }

  method->addAttr(::new (S.Context)
                  ObjCReturnsInnerPointerAttr(attr.getRange(), S.Context,
                                              attr.getAttributeSpellingListIndex()));
}

static void handleObjCRequiresSuperAttr(Sema &S, Decl *D,
                                        const AttributeList &attr) {
  SourceLocation loc = attr.getLoc();
  ObjCMethodDecl *method = dyn_cast<ObjCMethodDecl>(D);
  
  if (!method) {
   S.Diag(D->getLocStart(), diag::err_attribute_wrong_decl_type)
   << SourceRange(loc, loc) << attr.getName() << ExpectedMethod;
    return;
  }
  DeclContext *DC = method->getDeclContext();
  if (const ObjCProtocolDecl *PDecl = dyn_cast_or_null<ObjCProtocolDecl>(DC)) {
    S.Diag(D->getLocStart(), diag::warn_objc_requires_super_protocol)
    << attr.getName() << 0;
    S.Diag(PDecl->getLocation(), diag::note_protocol_decl);
    return;
  }
  if (method->getMethodFamily() == OMF_dealloc) {
    S.Diag(D->getLocStart(), diag::warn_objc_requires_super_protocol)
    << attr.getName() << 1;
    return;
  }
  
  method->addAttr(::new (S.Context)
                  ObjCRequiresSuperAttr(attr.getRange(), S.Context,
                                        attr.getAttributeSpellingListIndex()));
}

/// Handle cf_audited_transfer and cf_unknown_transfer.
static void handleCFTransferAttr(Sema &S, Decl *D, const AttributeList &A) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(D->getLocStart(), diag::err_attribute_wrong_decl_type)
      << A.getRange() << A.getName() << ExpectedFunction;
    return;
  }

  bool IsAudited = (A.getKind() == AttributeList::AT_CFAuditedTransfer);

  // Check whether there's a conflicting attribute already present.
  Attr *Existing;
  if (IsAudited) {
    Existing = D->getAttr<CFUnknownTransferAttr>();
  } else {
    Existing = D->getAttr<CFAuditedTransferAttr>();
  }
  if (Existing) {
    S.Diag(D->getLocStart(), diag::err_attributes_are_not_compatible)
      << A.getName()
      << (IsAudited ? "cf_unknown_transfer" : "cf_audited_transfer")
      << A.getRange() << Existing->getRange();
    return;
  }

  // All clear;  add the attribute.
  if (IsAudited) {
    D->addAttr(::new (S.Context)
               CFAuditedTransferAttr(A.getRange(), S.Context,
                                     A.getAttributeSpellingListIndex()));
  } else {
    D->addAttr(::new (S.Context)
               CFUnknownTransferAttr(A.getRange(), S.Context,
                                     A.getAttributeSpellingListIndex()));
  }
}

static void handleNSBridgedAttr(Sema &S, Scope *Sc, Decl *D,
                                const AttributeList &Attr) {
  RecordDecl *RD = dyn_cast<RecordDecl>(D);
  if (!RD || RD->isUnion()) {
    S.Diag(D->getLocStart(), diag::err_attribute_wrong_decl_type)
      << Attr.getRange() << Attr.getName() << ExpectedStruct;
  }

  IdentifierInfo *ParmName = Attr.getParameterName();

  // In Objective-C, verify that the type names an Objective-C type.
  // We don't want to check this outside of ObjC because people sometimes
  // do crazy C declarations of Objective-C types.
  if (ParmName && S.getLangOpts().ObjC1) {
    // Check for an existing type with this name.
    LookupResult R(S, DeclarationName(ParmName), Attr.getParameterLoc(),
                   Sema::LookupOrdinaryName);
    if (S.LookupName(R, Sc)) {
      NamedDecl *Target = R.getFoundDecl();
      if (Target && !isa<ObjCInterfaceDecl>(Target)) {
        S.Diag(D->getLocStart(), diag::err_ns_bridged_not_interface);
        S.Diag(Target->getLocStart(), diag::note_declared_at);
      }
    }
  }

  D->addAttr(::new (S.Context)
             NSBridgedAttr(Attr.getRange(), S.Context, ParmName,
                           Attr.getAttributeSpellingListIndex()));
}

static void handleObjCOwnershipAttr(Sema &S, Decl *D,
                                    const AttributeList &Attr) {
  if (hasDeclarator(D)) return;

  S.Diag(D->getLocStart(), diag::err_attribute_wrong_decl_type)
    << Attr.getRange() << Attr.getName() << ExpectedVariable;
}

static void handleObjCPreciseLifetimeAttr(Sema &S, Decl *D,
                                          const AttributeList &Attr) {
  if (!isa<VarDecl>(D) && !isa<FieldDecl>(D)) {
    S.Diag(D->getLocStart(), diag::err_attribute_wrong_decl_type)
      << Attr.getRange() << Attr.getName() << ExpectedVariable;
    return;
  }

  ValueDecl *vd = cast<ValueDecl>(D);
  QualType type = vd->getType();

  if (!type->isDependentType() &&
      !type->isObjCLifetimeType()) {
    S.Diag(Attr.getLoc(), diag::err_objc_precise_lifetime_bad_type)
      << type;
    return;
  }

  Qualifiers::ObjCLifetime lifetime = type.getObjCLifetime();

  // If we have no lifetime yet, check the lifetime we're presumably
  // going to infer.
  if (lifetime == Qualifiers::OCL_None && !type->isDependentType())
    lifetime = type->getObjCARCImplicitLifetime();

  switch (lifetime) {
  case Qualifiers::OCL_None:
    assert(type->isDependentType() &&
           "didn't infer lifetime for non-dependent type?");
    break;

  case Qualifiers::OCL_Weak:   // meaningful
  case Qualifiers::OCL_Strong: // meaningful
    break;

  case Qualifiers::OCL_ExplicitNone:
  case Qualifiers::OCL_Autoreleasing:
    S.Diag(Attr.getLoc(), diag::warn_objc_precise_lifetime_meaningless)
      << (lifetime == Qualifiers::OCL_Autoreleasing);
    break;
  }

  D->addAttr(::new (S.Context)
             ObjCPreciseLifetimeAttr(Attr.getRange(), S.Context,
                                     Attr.getAttributeSpellingListIndex()));
}

//===----------------------------------------------------------------------===//
// Microsoft specific attribute handlers.
//===----------------------------------------------------------------------===//

static void handleUuidAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (S.LangOpts.MicrosoftExt || S.LangOpts.Borland) {
    // check the attribute arguments.
    if (!checkAttributeNumArgs(S, Attr, 1))
      return;

    Expr *Arg = Attr.getArg(0);
    StringLiteral *Str = dyn_cast<StringLiteral>(Arg);
    if (!Str || !Str->isAscii()) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string)
        << "uuid" << 1;
      return;
    }

    StringRef StrRef = Str->getString();

    bool IsCurly = StrRef.size() > 1 && StrRef.front() == '{' &&
                   StrRef.back() == '}';

    // Validate GUID length.
    if (IsCurly && StrRef.size() != 38) {
      S.Diag(Attr.getLoc(), diag::err_attribute_uuid_malformed_guid);
      return;
    }
    if (!IsCurly && StrRef.size() != 36) {
      S.Diag(Attr.getLoc(), diag::err_attribute_uuid_malformed_guid);
      return;
    }

    // GUID format is "XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX" or
    // "{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}"
    StringRef::iterator I = StrRef.begin();
    if (IsCurly) // Skip the optional '{'
       ++I;

    for (int i = 0; i < 36; ++i) {
      if (i == 8 || i == 13 || i == 18 || i == 23) {
        if (*I != '-') {
          S.Diag(Attr.getLoc(), diag::err_attribute_uuid_malformed_guid);
          return;
        }
      } else if (!isHexDigit(*I)) {
        S.Diag(Attr.getLoc(), diag::err_attribute_uuid_malformed_guid);
        return;
      }
      I++;
    }

    D->addAttr(::new (S.Context)
               UuidAttr(Attr.getRange(), S.Context, Str->getString(),
                        Attr.getAttributeSpellingListIndex()));
  } else
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "uuid";
}

static void handleInheritanceAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (!S.LangOpts.MicrosoftExt) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << Attr.getName();
    return;
  }

  AttributeList::Kind Kind = Attr.getKind();
  if (Kind == AttributeList::AT_SingleInheritance)
    D->addAttr(
        ::new (S.Context)
               SingleInheritanceAttr(Attr.getRange(), S.Context,
                                     Attr.getAttributeSpellingListIndex()));
  else if (Kind == AttributeList::AT_MultipleInheritance)
    D->addAttr(
        ::new (S.Context)
               MultipleInheritanceAttr(Attr.getRange(), S.Context,
                                       Attr.getAttributeSpellingListIndex()));
  else if (Kind == AttributeList::AT_VirtualInheritance)
    D->addAttr(
        ::new (S.Context)
               VirtualInheritanceAttr(Attr.getRange(), S.Context,
                                      Attr.getAttributeSpellingListIndex()));
}

static void handlePortabilityAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (S.LangOpts.MicrosoftExt) {
    AttributeList::Kind Kind = Attr.getKind();
    if (Kind == AttributeList::AT_Ptr32)
      D->addAttr(
          ::new (S.Context) Ptr32Attr(Attr.getRange(), S.Context,
                                      Attr.getAttributeSpellingListIndex()));
    else if (Kind == AttributeList::AT_Ptr64)
      D->addAttr(
          ::new (S.Context) Ptr64Attr(Attr.getRange(), S.Context,
                                      Attr.getAttributeSpellingListIndex()));
    else if (Kind == AttributeList::AT_Win64)
      D->addAttr(
          ::new (S.Context) Win64Attr(Attr.getRange(), S.Context,
                                      Attr.getAttributeSpellingListIndex()));
  } else
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << Attr.getName();
}

static void handleForceInlineAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (S.LangOpts.MicrosoftExt)
    D->addAttr(::new (S.Context)
               ForceInlineAttr(Attr.getRange(), S.Context,
                               Attr.getAttributeSpellingListIndex()));
  else
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << Attr.getName();
}

//===----------------------------------------------------------------------===//
// Top Level Sema Entry Points
//===----------------------------------------------------------------------===//

static void ProcessNonInheritableDeclAttr(Sema &S, Scope *scope, Decl *D,
                                          const AttributeList &Attr) {
  switch (Attr.getKind()) {
  case AttributeList::AT_CUDADevice:  handleDeviceAttr      (S, D, Attr); break;
  case AttributeList::AT_CUDAHost:    handleHostAttr        (S, D, Attr); break;
  case AttributeList::AT_Overloadable:handleOverloadableAttr(S, D, Attr); break;
  default:
    break;
  }
}

static void ProcessInheritableDeclAttr(Sema &S, Scope *scope, Decl *D,
                                       const AttributeList &Attr) {
  switch (Attr.getKind()) {
  case AttributeList::AT_IBAction:    handleIBAction(S, D, Attr); break;
  case AttributeList::AT_IBOutlet:    handleIBOutlet(S, D, Attr); break;
  case AttributeList::AT_IBOutletCollection:
    handleIBOutletCollection(S, D, Attr); break;
  case AttributeList::AT_AddressSpace:
  case AttributeList::AT_ObjCGC:
  case AttributeList::AT_VectorSize:
  case AttributeList::AT_NeonVectorType:
  case AttributeList::AT_NeonPolyVectorType:
    // Ignore these, these are type attributes, handled by
    // ProcessTypeAttributes.
    break;
  case AttributeList::AT_CUDADevice:
  case AttributeList::AT_CUDAHost:
  case AttributeList::AT_Overloadable:
    // Ignore, this is a non-inheritable attribute, handled
    // by ProcessNonInheritableDeclAttr.
    break;
  case AttributeList::AT_Alias:       handleAliasAttr       (S, D, Attr); break;
  case AttributeList::AT_Aligned:     handleAlignedAttr     (S, D, Attr); break;
  case AttributeList::AT_AllocSize:   handleAllocSizeAttr   (S, D, Attr); break;
  case AttributeList::AT_AlwaysInline:
    handleAlwaysInlineAttr  (S, D, Attr); break;
  case AttributeList::AT_AnalyzerNoReturn:
    handleAnalyzerNoReturnAttr  (S, D, Attr); break;
  case AttributeList::AT_TLSModel:    handleTLSModelAttr    (S, D, Attr); break;
  case AttributeList::AT_Annotate:    handleAnnotateAttr    (S, D, Attr); break;
  case AttributeList::AT_Availability:handleAvailabilityAttr(S, D, Attr); break;
  case AttributeList::AT_CarriesDependency:
    handleDependencyAttr(S, scope, D, Attr);
    break;
  case AttributeList::AT_Common:      handleCommonAttr      (S, D, Attr); break;
  case AttributeList::AT_CUDAConstant:handleConstantAttr    (S, D, Attr); break;
  case AttributeList::AT_Constructor: handleConstructorAttr (S, D, Attr); break;
  case AttributeList::AT_CXX11NoReturn:
    handleCXX11NoReturnAttr(S, D, Attr);
    break;
  case AttributeList::AT_Deprecated:
    handleAttrWithMessage<DeprecatedAttr>(S, D, Attr, "deprecated");
    break;
  case AttributeList::AT_Destructor:  handleDestructorAttr  (S, D, Attr); break;
  case AttributeList::AT_ExtVectorType:
    handleExtVectorTypeAttr(S, scope, D, Attr);
    break;
  case AttributeList::AT_MinSize:
    handleMinSizeAttr(S, D, Attr);
    break;
  case AttributeList::AT_Format:      handleFormatAttr      (S, D, Attr); break;
  case AttributeList::AT_FormatArg:   handleFormatArgAttr   (S, D, Attr); break;
  case AttributeList::AT_CUDAGlobal:  handleGlobalAttr      (S, D, Attr); break;
  case AttributeList::AT_GNUInline:   handleGNUInlineAttr   (S, D, Attr); break;
  case AttributeList::AT_CUDALaunchBounds:
    handleLaunchBoundsAttr(S, D, Attr);
    break;
  case AttributeList::AT_Mode:        handleModeAttr        (S, D, Attr); break;
  case AttributeList::AT_Malloc:      handleMallocAttr      (S, D, Attr); break;
  case AttributeList::AT_MayAlias:    handleMayAliasAttr    (S, D, Attr); break;
  case AttributeList::AT_NoCommon:    handleNoCommonAttr    (S, D, Attr); break;
  case AttributeList::AT_NonNull:     handleNonNullAttr     (S, D, Attr); break;
  case AttributeList::AT_ownership_returns:
  case AttributeList::AT_ownership_takes:
  case AttributeList::AT_ownership_holds:
      handleOwnershipAttr     (S, D, Attr); break;
  case AttributeList::AT_Cold:        handleColdAttr        (S, D, Attr); break;
  case AttributeList::AT_Hot:         handleHotAttr         (S, D, Attr); break;
  case AttributeList::AT_Naked:       handleNakedAttr       (S, D, Attr); break;
  case AttributeList::AT_NoReturn:    handleNoReturnAttr    (S, D, Attr); break;
  case AttributeList::AT_NoThrow:     handleNothrowAttr     (S, D, Attr); break;
  case AttributeList::AT_CUDAShared:  handleSharedAttr      (S, D, Attr); break;
  case AttributeList::AT_VecReturn:   handleVecReturnAttr   (S, D, Attr); break;

  case AttributeList::AT_ObjCOwnership:
    handleObjCOwnershipAttr(S, D, Attr); break;
  case AttributeList::AT_ObjCPreciseLifetime:
    handleObjCPreciseLifetimeAttr(S, D, Attr); break;

  case AttributeList::AT_ObjCReturnsInnerPointer:
    handleObjCReturnsInnerPointerAttr(S, D, Attr); break;

  case AttributeList::AT_ObjCRequiresSuper:
      handleObjCRequiresSuperAttr(S, D, Attr); break;
      
  case AttributeList::AT_NSBridged:
    handleNSBridgedAttr(S, scope, D, Attr); break;

  case AttributeList::AT_CFAuditedTransfer:
  case AttributeList::AT_CFUnknownTransfer:
    handleCFTransferAttr(S, D, Attr); break;

  // Checker-specific.
  case AttributeList::AT_CFConsumed:
  case AttributeList::AT_NSConsumed:  handleNSConsumedAttr  (S, D, Attr); break;
  case AttributeList::AT_NSConsumesSelf:
    handleNSConsumesSelfAttr(S, D, Attr); break;

  case AttributeList::AT_NSReturnsAutoreleased:
  case AttributeList::AT_NSReturnsNotRetained:
  case AttributeList::AT_CFReturnsNotRetained:
  case AttributeList::AT_NSReturnsRetained:
  case AttributeList::AT_CFReturnsRetained:
    handleNSReturnsRetainedAttr(S, D, Attr); break;

  case AttributeList::AT_WorkGroupSizeHint:
  case AttributeList::AT_ReqdWorkGroupSize:
    handleWorkGroupSize(S, D, Attr); break;

  case AttributeList::AT_VecTypeHint:
    handleVecTypeHint(S, D, Attr); break;

  case AttributeList::AT_Endian:
    handleEndianAttr(S, D, Attr);
    break;

  case AttributeList::AT_InitPriority: 
      handleInitPriorityAttr(S, D, Attr); break;
      
  case AttributeList::AT_Packed:      handlePackedAttr      (S, D, Attr); break;
  case AttributeList::AT_Section:     handleSectionAttr     (S, D, Attr); break;
  case AttributeList::AT_Unavailable:
    handleAttrWithMessage<UnavailableAttr>(S, D, Attr, "unavailable");
    break;
  case AttributeList::AT_ArcWeakrefUnavailable: 
    handleArcWeakrefUnavailableAttr (S, D, Attr); 
    break;
  case AttributeList::AT_ObjCRootClass:
    handleObjCRootClassAttr(S, D, Attr);
    break;
  case AttributeList::AT_ObjCRequiresPropertyDefs: 
    handleObjCRequiresPropertyDefsAttr (S, D, Attr); 
    break;
  case AttributeList::AT_Unused:      handleUnusedAttr      (S, D, Attr); break;
  case AttributeList::AT_ReturnsTwice:
    handleReturnsTwiceAttr(S, D, Attr);
    break;
  case AttributeList::AT_Used:        handleUsedAttr        (S, D, Attr); break;
  case AttributeList::AT_Visibility:
    handleVisibilityAttr(S, D, Attr, false);
    break;
  case AttributeList::AT_TypeVisibility:
    handleVisibilityAttr(S, D, Attr, true);
    break;
  case AttributeList::AT_WarnUnusedResult: handleWarnUnusedResult(S, D, Attr);
    break;
  case AttributeList::AT_Weak:        handleWeakAttr        (S, D, Attr); break;
  case AttributeList::AT_WeakRef:     handleWeakRefAttr     (S, D, Attr); break;
  case AttributeList::AT_WeakImport:  handleWeakImportAttr  (S, D, Attr); break;
  case AttributeList::AT_TransparentUnion:
    handleTransparentUnionAttr(S, D, Attr);
    break;
  case AttributeList::AT_ObjCException:
    handleObjCExceptionAttr(S, D, Attr);
    break;
  case AttributeList::AT_ObjCMethodFamily:
    handleObjCMethodFamilyAttr(S, D, Attr);
    break;
  case AttributeList::AT_ObjCNSObject:handleObjCNSObject    (S, D, Attr); break;
  case AttributeList::AT_Blocks:      handleBlocksAttr      (S, D, Attr); break;
  case AttributeList::AT_Sentinel:    handleSentinelAttr    (S, D, Attr); break;
  case AttributeList::AT_Const:       handleConstAttr       (S, D, Attr); break;
  case AttributeList::AT_Pure:        handlePureAttr        (S, D, Attr); break;
  case AttributeList::AT_Cleanup:     handleCleanupAttr     (S, D, Attr); break;
  case AttributeList::AT_NoDebug:     handleNoDebugAttr     (S, D, Attr); break;
  case AttributeList::AT_NoInline:    handleNoInlineAttr    (S, D, Attr); break;
  case AttributeList::AT_Regparm:     handleRegparmAttr     (S, D, Attr); break;
  case AttributeList::IgnoredAttribute:
    // Just ignore
    break;
  case AttributeList::AT_NoInstrumentFunction:  // Interacts with -pg.
    handleNoInstrumentFunctionAttr(S, D, Attr);
    break;
  case AttributeList::AT_StdCall:
  case AttributeList::AT_CDecl:
  case AttributeList::AT_FastCall:
  case AttributeList::AT_ThisCall:
  case AttributeList::AT_Pascal:
  case AttributeList::AT_Pcs:
  case AttributeList::AT_PnaclCall:
  case AttributeList::AT_IntelOclBicc:
    handleCallConvAttr(S, D, Attr);
    break;
  case AttributeList::AT_OpenCLKernel:
    handleOpenCLKernelAttr(S, D, Attr);
    break;
  case AttributeList::AT_OpenCLImageAccess:
    handleOpenCLImageAccessAttr(S, D, Attr);
    break;

  // Microsoft attributes:
  case AttributeList::AT_MsProperty: break;
  case AttributeList::AT_MsStruct:
    handleMsStructAttr(S, D, Attr);
    break;
  case AttributeList::AT_Uuid:
    handleUuidAttr(S, D, Attr);
    break;
  case AttributeList::AT_SingleInheritance:
  case AttributeList::AT_MultipleInheritance:
  case AttributeList::AT_VirtualInheritance:
    handleInheritanceAttr(S, D, Attr);
    break;
  case AttributeList::AT_Win64:
  case AttributeList::AT_Ptr32:
  case AttributeList::AT_Ptr64:
    handlePortabilityAttr(S, D, Attr);
    break;
  case AttributeList::AT_ForceInline:
    handleForceInlineAttr(S, D, Attr);
    break;

  // Thread safety attributes:
  case AttributeList::AT_GuardedVar:
    handleGuardedVarAttr(S, D, Attr);
    break;
  case AttributeList::AT_PtGuardedVar:
    handlePtGuardedVarAttr(S, D, Attr);
    break;
  case AttributeList::AT_ScopedLockable:
    handleScopedLockableAttr(S, D, Attr);
    break;
  case AttributeList::AT_NoSanitizeAddress:
    handleNoSanitizeAddressAttr(S, D, Attr);
    break;
  case AttributeList::AT_NoThreadSafetyAnalysis:
    handleNoThreadSafetyAnalysis(S, D, Attr);
    break;
  case AttributeList::AT_NoSanitizeThread:
    handleNoSanitizeThread(S, D, Attr);
    break;
  case AttributeList::AT_NoSanitizeMemory:
    handleNoSanitizeMemory(S, D, Attr);
    break;
  case AttributeList::AT_Lockable:
    handleLockableAttr(S, D, Attr);
    break;
  case AttributeList::AT_GuardedBy:
    handleGuardedByAttr(S, D, Attr);
    break;
  case AttributeList::AT_PtGuardedBy:
    handlePtGuardedByAttr(S, D, Attr);
    break;
  case AttributeList::AT_ExclusiveLockFunction:
    handleExclusiveLockFunctionAttr(S, D, Attr);
    break;
  case AttributeList::AT_ExclusiveLocksRequired:
    handleExclusiveLocksRequiredAttr(S, D, Attr);
    break;
  case AttributeList::AT_ExclusiveTrylockFunction:
    handleExclusiveTrylockFunctionAttr(S, D, Attr);
    break;
  case AttributeList::AT_LockReturned:
    handleLockReturnedAttr(S, D, Attr);
    break;
  case AttributeList::AT_LocksExcluded:
    handleLocksExcludedAttr(S, D, Attr);
    break;
  case AttributeList::AT_SharedLockFunction:
    handleSharedLockFunctionAttr(S, D, Attr);
    break;
  case AttributeList::AT_SharedLocksRequired:
    handleSharedLocksRequiredAttr(S, D, Attr);
    break;
  case AttributeList::AT_SharedTrylockFunction:
    handleSharedTrylockFunctionAttr(S, D, Attr);
    break;
  case AttributeList::AT_UnlockFunction:
    handleUnlockFunAttr(S, D, Attr);
    break;
  case AttributeList::AT_AcquiredBefore:
    handleAcquiredBeforeAttr(S, D, Attr);
    break;
  case AttributeList::AT_AcquiredAfter:
    handleAcquiredAfterAttr(S, D, Attr);
    break;

  // Type safety attributes.
  case AttributeList::AT_ArgumentWithTypeTag:
    handleArgumentWithTypeTagAttr(S, D, Attr);
    break;
  case AttributeList::AT_TypeTagForDatatype:
    handleTypeTagForDatatypeAttr(S, D, Attr);
    break;

  default:
    // Ask target about the attribute.
    const TargetAttributesSema &TargetAttrs = S.getTargetAttributesSema();
    if (!TargetAttrs.ProcessDeclAttribute(scope, D, Attr, S))
      S.Diag(Attr.getLoc(), Attr.isDeclspecAttribute() ? 
             diag::warn_unhandled_ms_attribute_ignored : 
             diag::warn_unknown_attribute_ignored) << Attr.getName();
    break;
  }
}

/// ProcessDeclAttribute - Apply the specific attribute to the specified decl if
/// the attribute applies to decls.  If the attribute is a type attribute, just
/// silently ignore it if a GNU attribute.
static void ProcessDeclAttribute(Sema &S, Scope *scope, Decl *D,
                                 const AttributeList &Attr,
                                 bool NonInheritable, bool Inheritable,
                                 bool IncludeCXX11Attributes) {
  if (Attr.isInvalid())
    return;

  // Ignore C++11 attributes on declarator chunks: they appertain to the type
  // instead.
  if (Attr.isCXX11Attribute() && !IncludeCXX11Attributes)
    return;

  if (NonInheritable)
    ProcessNonInheritableDeclAttr(S, scope, D, Attr);

  if (Inheritable)
    ProcessInheritableDeclAttr(S, scope, D, Attr);
}

/// ProcessDeclAttributeList - Apply all the decl attributes in the specified
/// attribute list to the specified decl, ignoring any type attributes.
void Sema::ProcessDeclAttributeList(Scope *S, Decl *D,
                                    const AttributeList *AttrList,
                                    bool NonInheritable, bool Inheritable,
                                    bool IncludeCXX11Attributes) {
  for (const AttributeList* l = AttrList; l; l = l->getNext())
    ProcessDeclAttribute(*this, S, D, *l, NonInheritable, Inheritable,
                         IncludeCXX11Attributes);

  // GCC accepts
  // static int a9 __attribute__((weakref));
  // but that looks really pointless. We reject it.
  if (Inheritable && D->hasAttr<WeakRefAttr>() && !D->hasAttr<AliasAttr>()) {
    Diag(AttrList->getLoc(), diag::err_attribute_weakref_without_alias) <<
    cast<NamedDecl>(D)->getNameAsString();
    D->dropAttr<WeakRefAttr>();
    return;
  }
}

// Annotation attributes are the only attributes allowed after an access
// specifier.
bool Sema::ProcessAccessDeclAttributeList(AccessSpecDecl *ASDecl,
                                          const AttributeList *AttrList) {
  for (const AttributeList* l = AttrList; l; l = l->getNext()) {
    if (l->getKind() == AttributeList::AT_Annotate) {
      handleAnnotateAttr(*this, ASDecl, *l);
    } else {
      Diag(l->getLoc(), diag::err_only_annotate_after_access_spec);
      return true;
    }
  }

  return false;
}

/// checkUnusedDeclAttributes - Check a list of attributes to see if it
/// contains any decl attributes that we should warn about.
static void checkUnusedDeclAttributes(Sema &S, const AttributeList *A) {
  for ( ; A; A = A->getNext()) {
    // Only warn if the attribute is an unignored, non-type attribute.
    if (A->isUsedAsTypeAttr() || A->isInvalid()) continue;
    if (A->getKind() == AttributeList::IgnoredAttribute) continue;

    if (A->getKind() == AttributeList::UnknownAttribute) {
      S.Diag(A->getLoc(), diag::warn_unknown_attribute_ignored)
        << A->getName() << A->getRange();
    } else {
      S.Diag(A->getLoc(), diag::warn_attribute_not_on_decl)
        << A->getName() << A->getRange();
    }
  }
}

/// checkUnusedDeclAttributes - Given a declarator which is not being
/// used to build a declaration, complain about any decl attributes
/// which might be lying around on it.
void Sema::checkUnusedDeclAttributes(Declarator &D) {
  ::checkUnusedDeclAttributes(*this, D.getDeclSpec().getAttributes().getList());
  ::checkUnusedDeclAttributes(*this, D.getAttributes());
  for (unsigned i = 0, e = D.getNumTypeObjects(); i != e; ++i)
    ::checkUnusedDeclAttributes(*this, D.getTypeObject(i).getAttrs());
}

/// DeclClonePragmaWeak - clone existing decl (maybe definition),
/// \#pragma weak needs a non-definition decl and source may not have one.
NamedDecl * Sema::DeclClonePragmaWeak(NamedDecl *ND, IdentifierInfo *II,
                                      SourceLocation Loc) {
  assert(isa<FunctionDecl>(ND) || isa<VarDecl>(ND));
  NamedDecl *NewD = 0;
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(ND)) {
    FunctionDecl *NewFD;
    // FIXME: Missing call to CheckFunctionDeclaration().
    // FIXME: Mangling?
    // FIXME: Is the qualifier info correct?
    // FIXME: Is the DeclContext correct?
    NewFD = FunctionDecl::Create(FD->getASTContext(), FD->getDeclContext(),
                                 Loc, Loc, DeclarationName(II),
                                 FD->getType(), FD->getTypeSourceInfo(),
                                 SC_None, false/*isInlineSpecified*/,
                                 FD->hasPrototype(),
                                 false/*isConstexprSpecified*/);
    NewD = NewFD;

    if (FD->getQualifier())
      NewFD->setQualifierInfo(FD->getQualifierLoc());

    // Fake up parameter variables; they are declared as if this were
    // a typedef.
    QualType FDTy = FD->getType();
    if (const FunctionProtoType *FT = FDTy->getAs<FunctionProtoType>()) {
      SmallVector<ParmVarDecl*, 16> Params;
      for (FunctionProtoType::arg_type_iterator AI = FT->arg_type_begin(),
           AE = FT->arg_type_end(); AI != AE; ++AI) {
        ParmVarDecl *Param = BuildParmVarDeclForTypedef(NewFD, Loc, *AI);
        Param->setScopeInfo(0, Params.size());
        Params.push_back(Param);
      }
      NewFD->setParams(Params);
    }
  } else if (VarDecl *VD = dyn_cast<VarDecl>(ND)) {
    NewD = VarDecl::Create(VD->getASTContext(), VD->getDeclContext(),
                           VD->getInnerLocStart(), VD->getLocation(), II,
                           VD->getType(), VD->getTypeSourceInfo(),
                           VD->getStorageClass());
    if (VD->getQualifier()) {
      VarDecl *NewVD = cast<VarDecl>(NewD);
      NewVD->setQualifierInfo(VD->getQualifierLoc());
    }
  }
  return NewD;
}

/// DeclApplyPragmaWeak - A declaration (maybe definition) needs \#pragma weak
/// applied to it, possibly with an alias.
void Sema::DeclApplyPragmaWeak(Scope *S, NamedDecl *ND, WeakInfo &W) {
  if (W.getUsed()) return; // only do this once
  W.setUsed(true);
  if (W.getAlias()) { // clone decl, impersonate __attribute(weak,alias(...))
    IdentifierInfo *NDId = ND->getIdentifier();
    NamedDecl *NewD = DeclClonePragmaWeak(ND, W.getAlias(), W.getLocation());
    NewD->addAttr(::new (Context) AliasAttr(W.getLocation(), Context,
                                            NDId->getName()));
    NewD->addAttr(::new (Context) WeakAttr(W.getLocation(), Context));
    WeakTopLevelDecl.push_back(NewD);
    // FIXME: "hideous" code from Sema::LazilyCreateBuiltin
    // to insert Decl at TU scope, sorry.
    DeclContext *SavedContext = CurContext;
    CurContext = Context.getTranslationUnitDecl();
    PushOnScopeChains(NewD, S);
    CurContext = SavedContext;
  } else { // just add weak to existing
    ND->addAttr(::new (Context) WeakAttr(W.getLocation(), Context));
  }
}

void Sema::ProcessPragmaWeak(Scope *S, Decl *D) {
  // It's valid to "forward-declare" #pragma weak, in which case we
  // have to do this.
  LoadExternalWeakUndeclaredIdentifiers();
  if (!WeakUndeclaredIdentifiers.empty()) {
    NamedDecl *ND = NULL;
    if (VarDecl *VD = dyn_cast<VarDecl>(D))
      if (VD->isExternC())
        ND = VD;
    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
      if (FD->isExternC())
        ND = FD;
    if (ND) {
      if (IdentifierInfo *Id = ND->getIdentifier()) {
        llvm::DenseMap<IdentifierInfo*,WeakInfo>::iterator I
          = WeakUndeclaredIdentifiers.find(Id);
        if (I != WeakUndeclaredIdentifiers.end()) {
          WeakInfo W = I->second;
          DeclApplyPragmaWeak(S, ND, W);
          WeakUndeclaredIdentifiers[Id] = W;
        }
      }
    }
  }
}

/// ProcessDeclAttributes - Given a declarator (PD) with attributes indicated in
/// it, apply them to D.  This is a bit tricky because PD can have attributes
/// specified in many different places, and we need to find and apply them all.
void Sema::ProcessDeclAttributes(Scope *S, Decl *D, const Declarator &PD,
                                 bool NonInheritable, bool Inheritable) {
  // Apply decl attributes from the DeclSpec if present.
  if (const AttributeList *Attrs = PD.getDeclSpec().getAttributes().getList())
    ProcessDeclAttributeList(S, D, Attrs, NonInheritable, Inheritable);

  // Walk the declarator structure, applying decl attributes that were in a type
  // position to the decl itself.  This handles cases like:
  //   int *__attr__(x)** D;
  // when X is a decl attribute.
  for (unsigned i = 0, e = PD.getNumTypeObjects(); i != e; ++i)
    if (const AttributeList *Attrs = PD.getTypeObject(i).getAttrs())
      ProcessDeclAttributeList(S, D, Attrs, NonInheritable, Inheritable,
                               /*IncludeCXX11Attributes=*/false);

  // Finally, apply any attributes on the decl itself.
  if (const AttributeList *Attrs = PD.getAttributes())
    ProcessDeclAttributeList(S, D, Attrs, NonInheritable, Inheritable);
}

/// Is the given declaration allowed to use a forbidden type?
static bool isForbiddenTypeAllowed(Sema &S, Decl *decl) {
  // Private ivars are always okay.  Unfortunately, people don't
  // always properly make their ivars private, even in system headers.
  // Plus we need to make fields okay, too.
  // Function declarations in sys headers will be marked unavailable.
  if (!isa<FieldDecl>(decl) && !isa<ObjCPropertyDecl>(decl) &&
      !isa<FunctionDecl>(decl))
    return false;

  // Require it to be declared in a system header.
  return S.Context.getSourceManager().isInSystemHeader(decl->getLocation());
}

/// Handle a delayed forbidden-type diagnostic.
static void handleDelayedForbiddenType(Sema &S, DelayedDiagnostic &diag,
                                       Decl *decl) {
  if (decl && isForbiddenTypeAllowed(S, decl)) {
    decl->addAttr(new (S.Context) UnavailableAttr(diag.Loc, S.Context,
                        "this system declaration uses an unsupported type"));
    return;
  }
  if (S.getLangOpts().ObjCAutoRefCount)
    if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(decl)) {
      // FIXME: we may want to suppress diagnostics for all
      // kind of forbidden type messages on unavailable functions. 
      if (FD->hasAttr<UnavailableAttr>() &&
          diag.getForbiddenTypeDiagnostic() == 
          diag::err_arc_array_param_no_ownership) {
        diag.Triggered = true;
        return;
      }
    }

  S.Diag(diag.Loc, diag.getForbiddenTypeDiagnostic())
    << diag.getForbiddenTypeOperand() << diag.getForbiddenTypeArgument();
  diag.Triggered = true;
}

void Sema::PopParsingDeclaration(ParsingDeclState state, Decl *decl) {
  assert(DelayedDiagnostics.getCurrentPool());
  DelayedDiagnosticPool &poppedPool = *DelayedDiagnostics.getCurrentPool();
  DelayedDiagnostics.popWithoutEmitting(state);

  // When delaying diagnostics to run in the context of a parsed
  // declaration, we only want to actually emit anything if parsing
  // succeeds.
  if (!decl) return;

  // We emit all the active diagnostics in this pool or any of its
  // parents.  In general, we'll get one pool for the decl spec
  // and a child pool for each declarator; in a decl group like:
  //   deprecated_typedef foo, *bar, baz();
  // only the declarator pops will be passed decls.  This is correct;
  // we really do need to consider delayed diagnostics from the decl spec
  // for each of the different declarations.
  const DelayedDiagnosticPool *pool = &poppedPool;
  do {
    for (DelayedDiagnosticPool::pool_iterator
           i = pool->pool_begin(), e = pool->pool_end(); i != e; ++i) {
      // This const_cast is a bit lame.  Really, Triggered should be mutable.
      DelayedDiagnostic &diag = const_cast<DelayedDiagnostic&>(*i);
      if (diag.Triggered)
        continue;

      switch (diag.Kind) {
      case DelayedDiagnostic::Deprecation:
        // Don't bother giving deprecation diagnostics if the decl is invalid.
        if (!decl->isInvalidDecl())
          HandleDelayedDeprecationCheck(diag, decl);
        break;

      case DelayedDiagnostic::Access:
        HandleDelayedAccessCheck(diag, decl);
        break;

      case DelayedDiagnostic::ForbiddenType:
        handleDelayedForbiddenType(*this, diag, decl);
        break;
      }
    }
  } while ((pool = pool->getParent()));
}

/// Given a set of delayed diagnostics, re-emit them as if they had
/// been delayed in the current context instead of in the given pool.
/// Essentially, this just moves them to the current pool.
void Sema::redelayDiagnostics(DelayedDiagnosticPool &pool) {
  DelayedDiagnosticPool *curPool = DelayedDiagnostics.getCurrentPool();
  assert(curPool && "re-emitting in undelayed context not supported");
  curPool->steal(pool);
}

static bool isDeclDeprecated(Decl *D) {
  do {
    if (D->isDeprecated())
      return true;
    // A category implicitly has the availability of the interface.
    if (const ObjCCategoryDecl *CatD = dyn_cast<ObjCCategoryDecl>(D))
      return CatD->getClassInterface()->isDeprecated();
  } while ((D = cast_or_null<Decl>(D->getDeclContext())));
  return false;
}

static void
DoEmitDeprecationWarning(Sema &S, const NamedDecl *D, StringRef Message,
                         SourceLocation Loc,
                         const ObjCInterfaceDecl *UnknownObjCClass,
                         const ObjCPropertyDecl *ObjCPropery) {
  DeclarationName Name = D->getDeclName();
  if (!Message.empty()) {
    S.Diag(Loc, diag::warn_deprecated_message) << Name << Message;
    S.Diag(D->getLocation(),
           isa<ObjCMethodDecl>(D) ? diag::note_method_declared_at
                                  : diag::note_previous_decl) << Name;
    if (ObjCPropery)
      S.Diag(ObjCPropery->getLocation(), diag::note_property_attribute)
        << ObjCPropery->getDeclName() << 0;
  } else if (!UnknownObjCClass) {
    S.Diag(Loc, diag::warn_deprecated) << D->getDeclName();
    S.Diag(D->getLocation(),
           isa<ObjCMethodDecl>(D) ? diag::note_method_declared_at
                                  : diag::note_previous_decl) << Name;
    if (ObjCPropery)
      S.Diag(ObjCPropery->getLocation(), diag::note_property_attribute)
        << ObjCPropery->getDeclName() << 0;
  } else {
    S.Diag(Loc, diag::warn_deprecated_fwdclass_message) << Name;
    S.Diag(UnknownObjCClass->getLocation(), diag::note_forward_class);
  }
}

void Sema::HandleDelayedDeprecationCheck(DelayedDiagnostic &DD,
                                         Decl *Ctx) {
  if (isDeclDeprecated(Ctx))
    return;

  DD.Triggered = true;
  DoEmitDeprecationWarning(*this, DD.getDeprecationDecl(),
                           DD.getDeprecationMessage(), DD.Loc,
                           DD.getUnknownObjCClass(),
                           DD.getObjCProperty());
}

void Sema::EmitDeprecationWarning(NamedDecl *D, StringRef Message,
                                  SourceLocation Loc,
                                  const ObjCInterfaceDecl *UnknownObjCClass,
                                  const ObjCPropertyDecl  *ObjCProperty) {
  // Delay if we're currently parsing a declaration.
  if (DelayedDiagnostics.shouldDelayDiagnostics()) {
    DelayedDiagnostics.add(DelayedDiagnostic::makeDeprecation(Loc, D, 
                                                              UnknownObjCClass,
                                                              ObjCProperty,
                                                              Message));
    return;
  }

  // Otherwise, don't warn if our current context is deprecated.
  if (isDeclDeprecated(cast<Decl>(getCurLexicalContext())))
    return;
  DoEmitDeprecationWarning(*this, D, Message, Loc, UnknownObjCClass, ObjCProperty);
}
