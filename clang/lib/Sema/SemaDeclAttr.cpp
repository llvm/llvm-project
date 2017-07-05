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

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/DelayedDiagnostic.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/SemaInternal.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MathExtras.h"

using namespace clang;
using namespace sema;

namespace AttributeLangSupport {
  enum LANG {
    C,
    Cpp,
    ObjC
  };
} // end namespace AttributeLangSupport

//===----------------------------------------------------------------------===//
//  Helper functions
//===----------------------------------------------------------------------===//

/// isFunctionOrMethod - Return true if the given decl has function
/// type (function or function-typed variable) or an Objective-C
/// method.
static bool isFunctionOrMethod(const Decl *D) {
  return (D->getFunctionType() != nullptr) || isa<ObjCMethodDecl>(D);
}

/// \brief Return true if the given decl has function type (function or
/// function-typed variable) or an Objective-C method or a block.
static bool isFunctionOrMethodOrBlock(const Decl *D) {
  return isFunctionOrMethod(D) || isa<BlockDecl>(D);
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
  if (const FunctionType *FnTy = D->getFunctionType())
    return isa<FunctionProtoType>(FnTy);
  return isa<ObjCMethodDecl>(D) || isa<BlockDecl>(D);
}

/// getFunctionOrMethodNumParams - Return number of function or method
/// parameters. It is an error to call this on a K&R function (use
/// hasFunctionProto first).
static unsigned getFunctionOrMethodNumParams(const Decl *D) {
  if (const FunctionType *FnTy = D->getFunctionType())
    return cast<FunctionProtoType>(FnTy)->getNumParams();
  if (const BlockDecl *BD = dyn_cast<BlockDecl>(D))
    return BD->getNumParams();
  return cast<ObjCMethodDecl>(D)->param_size();
}

static QualType getFunctionOrMethodParamType(const Decl *D, unsigned Idx) {
  if (const FunctionType *FnTy = D->getFunctionType())
    return cast<FunctionProtoType>(FnTy)->getParamType(Idx);
  if (const BlockDecl *BD = dyn_cast<BlockDecl>(D))
    return BD->getParamDecl(Idx)->getType();

  return cast<ObjCMethodDecl>(D)->parameters()[Idx]->getType();
}

static SourceRange getFunctionOrMethodParamRange(const Decl *D, unsigned Idx) {
  if (const auto *FD = dyn_cast<FunctionDecl>(D))
    return FD->getParamDecl(Idx)->getSourceRange();
  if (const auto *MD = dyn_cast<ObjCMethodDecl>(D))
    return MD->parameters()[Idx]->getSourceRange();
  if (const auto *BD = dyn_cast<BlockDecl>(D))
    return BD->getParamDecl(Idx)->getSourceRange();
  return SourceRange();
}

static QualType getFunctionOrMethodResultType(const Decl *D) {
  if (const FunctionType *FnTy = D->getFunctionType())
    return cast<FunctionType>(FnTy)->getReturnType();
  return cast<ObjCMethodDecl>(D)->getReturnType();
}

static SourceRange getFunctionOrMethodResultSourceRange(const Decl *D) {
  if (const auto *FD = dyn_cast<FunctionDecl>(D))
    return FD->getReturnTypeSourceRange();
  if (const auto *MD = dyn_cast<ObjCMethodDecl>(D))
    return MD->getReturnTypeSourceRange();
  return SourceRange();
}

static bool isFunctionOrMethodVariadic(const Decl *D) {
  if (const FunctionType *FnTy = D->getFunctionType()) {
    const FunctionProtoType *proto = cast<FunctionProtoType>(FnTy);
    return proto->isVariadic();
  }
  if (const BlockDecl *BD = dyn_cast<BlockDecl>(D))
    return BD->isVariadic();

  return cast<ObjCMethodDecl>(D)->isVariadic();
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

static unsigned getNumAttributeArgs(const AttributeList &Attr) {
  // FIXME: Include the type in the argument list.
  return Attr.getNumArgs() + Attr.hasParsedType();
}

template <typename Compare>
static bool checkAttributeNumArgsImpl(Sema &S, const AttributeList &Attr,
                                      unsigned Num, unsigned Diag,
                                      Compare Comp) {
  if (Comp(getNumAttributeArgs(Attr), Num)) {
    S.Diag(Attr.getLoc(), Diag) << Attr.getName() << Num;
    return false;
  }

  return true;
}

/// \brief Check if the attribute has exactly as many args as Num. May
/// output an error.
static bool checkAttributeNumArgs(Sema &S, const AttributeList &Attr,
                                  unsigned Num) {
  return checkAttributeNumArgsImpl(S, Attr, Num,
                                   diag::err_attribute_wrong_number_arguments,
                                   std::not_equal_to<unsigned>());
}

/// \brief Check if the attribute has at least as many args as Num. May
/// output an error.
static bool checkAttributeAtLeastNumArgs(Sema &S, const AttributeList &Attr,
                                         unsigned Num) {
  return checkAttributeNumArgsImpl(S, Attr, Num,
                                   diag::err_attribute_too_few_arguments,
                                   std::less<unsigned>());
}

/// \brief Check if the attribute has at most as many args as Num. May
/// output an error.
static bool checkAttributeAtMostNumArgs(Sema &S, const AttributeList &Attr,
                                         unsigned Num) {
  return checkAttributeNumArgsImpl(S, Attr, Num,
                                   diag::err_attribute_too_many_arguments,
                                   std::greater<unsigned>());
}

/// \brief A helper function to provide Attribute Location for the Attr types
/// AND the AttributeList.
template <typename AttrInfo>
static typename std::enable_if<std::is_base_of<clang::Attr, AttrInfo>::value,
                               SourceLocation>::type
getAttrLoc(const AttrInfo &Attr) {
  return Attr.getLocation();
}
static SourceLocation getAttrLoc(const clang::AttributeList &Attr) {
  return Attr.getLoc();
}

/// \brief A helper function to provide Attribute Name for the Attr types
/// AND the AttributeList.
template <typename AttrInfo>
static typename std::enable_if<std::is_base_of<clang::Attr, AttrInfo>::value,
                               const AttrInfo *>::type
getAttrName(const AttrInfo &Attr) {
  return &Attr;
}
static const IdentifierInfo *getAttrName(const clang::AttributeList &Attr) {
  return Attr.getName();
}

/// \brief If Expr is a valid integer constant, get the value of the integer
/// expression and return success or failure. May output an error.
template<typename AttrInfo>
static bool checkUInt32Argument(Sema &S, const AttrInfo& Attr, const Expr *Expr,
                                uint32_t &Val, unsigned Idx = UINT_MAX) {
  llvm::APSInt I(32);
  if (Expr->isTypeDependent() || Expr->isValueDependent() ||
      !Expr->isIntegerConstantExpr(I, S.Context)) {
    if (Idx != UINT_MAX)
      S.Diag(getAttrLoc(Attr), diag::err_attribute_argument_n_type)
        << getAttrName(Attr) << Idx << AANT_ArgumentIntegerConstant
        << Expr->getSourceRange();
    else
      S.Diag(getAttrLoc(Attr), diag::err_attribute_argument_type)
        << getAttrName(Attr) << AANT_ArgumentIntegerConstant
        << Expr->getSourceRange();
    return false;
  }

  if (!I.isIntN(32)) {
    S.Diag(Expr->getExprLoc(), diag::err_ice_too_large)
        << I.toString(10, false) << 32 << /* Unsigned */ 1;
    return false;
  }

  Val = (uint32_t)I.getZExtValue();
  return true;
}

/// \brief Wrapper around checkUInt32Argument, with an extra check to be sure
/// that the result will fit into a regular (signed) int. All args have the same
/// purpose as they do in checkUInt32Argument.
template<typename AttrInfo>
static bool checkPositiveIntArgument(Sema &S, const AttrInfo& Attr, const Expr *Expr,
                                     int &Val, unsigned Idx = UINT_MAX) {
  uint32_t UVal;
  if (!checkUInt32Argument(S, Attr, Expr, UVal, Idx))
    return false;

  if (UVal > (uint32_t)std::numeric_limits<int>::max()) {
    llvm::APSInt I(32); // for toString
    I = UVal;
    S.Diag(Expr->getExprLoc(), diag::err_ice_too_large)
        << I.toString(10, false) << 32 << /* Unsigned */ 0;
    return false;
  }

  Val = UVal;
  return true;
}

/// \brief Diagnose mutually exclusive attributes when present on a given
/// declaration. Returns true if diagnosed.
template <typename AttrTy>
static bool checkAttrMutualExclusion(Sema &S, Decl *D, SourceRange Range,
                                     IdentifierInfo *Ident) {
  if (AttrTy *A = D->getAttr<AttrTy>()) {
    S.Diag(Range.getBegin(), diag::err_attributes_are_not_compatible) << Ident
                                                                      << A;
    S.Diag(A->getLocation(), diag::note_conflicting_attribute);
    return true;
  }
  return false;
}

/// \brief Check if IdxExpr is a valid parameter index for a function or
/// instance method D.  May output an error.
///
/// \returns true if IdxExpr is a valid index.
template <typename AttrInfo>
static bool checkFunctionOrMethodParameterIndex(
    Sema &S, const Decl *D, const AttrInfo &Attr, unsigned AttrArgNum,
    const Expr *IdxExpr, uint64_t &Idx, bool AllowImplicitThis = false) {
  assert(isFunctionOrMethodOrBlock(D));

  // In C++ the implicit 'this' function parameter also counts.
  // Parameters are counted from one.
  bool HP = hasFunctionProto(D);
  bool HasImplicitThisParam = isInstanceMethod(D);
  bool IV = HP && isFunctionOrMethodVariadic(D);
  unsigned NumParams =
      (HP ? getFunctionOrMethodNumParams(D) : 0) + HasImplicitThisParam;

  llvm::APSInt IdxInt;
  if (IdxExpr->isTypeDependent() || IdxExpr->isValueDependent() ||
      !IdxExpr->isIntegerConstantExpr(IdxInt, S.Context)) {
    S.Diag(getAttrLoc(Attr), diag::err_attribute_argument_n_type)
      << getAttrName(Attr) << AttrArgNum << AANT_ArgumentIntegerConstant
      << IdxExpr->getSourceRange();
    return false;
  }

  Idx = IdxInt.getLimitedValue();
  if (Idx < 1 || (!IV && Idx > NumParams)) {
    S.Diag(getAttrLoc(Attr), diag::err_attribute_argument_out_of_bounds)
      << getAttrName(Attr) << AttrArgNum << IdxExpr->getSourceRange();
    return false;
  }
  Idx--; // Convert to zero-based.
  if (HasImplicitThisParam && !AllowImplicitThis) {
    if (Idx == 0) {
      S.Diag(getAttrLoc(Attr),
             diag::err_attribute_invalid_implicit_this_argument)
        << getAttrName(Attr) << IdxExpr->getSourceRange();
      return false;
    }
    --Idx;
  }

  return true;
}

/// \brief Check if the argument \p ArgNum of \p Attr is a ASCII string literal.
/// If not emit an error and return false. If the argument is an identifier it
/// will emit an error with a fixit hint and treat it as if it was a string
/// literal.
bool Sema::checkStringLiteralArgumentAttr(const AttributeList &Attr,
                                          unsigned ArgNum, StringRef &Str,
                                          SourceLocation *ArgLocation) {
  // Look for identifiers. If we have one emit a hint to fix it to a literal.
  if (Attr.isArgIdent(ArgNum)) {
    IdentifierLoc *Loc = Attr.getArgAsIdent(ArgNum);
    Diag(Loc->Loc, diag::err_attribute_argument_type)
        << Attr.getName() << AANT_ArgumentString
        << FixItHint::CreateInsertion(Loc->Loc, "\"")
        << FixItHint::CreateInsertion(getLocForEndOfToken(Loc->Loc), "\"");
    Str = Loc->Ident->getName();
    if (ArgLocation)
      *ArgLocation = Loc->Loc;
    return true;
  }

  // Now check for an actual string literal.
  Expr *ArgExpr = Attr.getArgAsExpr(ArgNum);
  StringLiteral *Literal = dyn_cast<StringLiteral>(ArgExpr->IgnoreParenCasts());
  if (ArgLocation)
    *ArgLocation = ArgExpr->getLocStart();

  if (!Literal || !Literal->isAscii()) {
    Diag(ArgExpr->getLocStart(), diag::err_attribute_argument_type)
        << Attr.getName() << AANT_ArgumentString;
    return false;
  }

  Str = Literal->getString();
  return true;
}

/// \brief Applies the given attribute to the Decl without performing any
/// additional semantic checking.
template <typename AttrType>
static void handleSimpleAttribute(Sema &S, Decl *D,
                                  const AttributeList &Attr) {
  D->addAttr(::new (S.Context) AttrType(Attr.getRange(), S.Context,
                                        Attr.getAttributeSpellingListIndex()));
}

template <typename AttrType>
static void handleSimpleAttributeWithExclusions(Sema &S, Decl *D,
                                                const AttributeList &Attr) {
  handleSimpleAttribute<AttrType>(S, D, Attr);
}

/// \brief Applies the given attribute to the Decl so long as the Decl doesn't
/// already have one of the given incompatible attributes.
template <typename AttrType, typename IncompatibleAttrType,
          typename... IncompatibleAttrTypes>
static void handleSimpleAttributeWithExclusions(Sema &S, Decl *D,
                                                const AttributeList &Attr) {
  if (checkAttrMutualExclusion<IncompatibleAttrType>(S, D, Attr.getRange(),
                                                     Attr.getName()))
    return;
  handleSimpleAttributeWithExclusions<AttrType, IncompatibleAttrTypes...>(S, D,
                                                                          Attr);
}

/// \brief Check if the passed-in expression is of type int or bool.
static bool isIntOrBool(Expr *Exp) {
  QualType QT = Exp->getType();
  return QT->isBooleanType() || QT->isIntegerType();
}


// Check to see if the type is a smart pointer of some kind.  We assume
// it's a smart pointer if it defines both operator-> and operator*.
static bool threadSafetyCheckIsSmartPointer(Sema &S, const RecordType* RT) {
  DeclContextLookupResult Res1 = RT->getDecl()->lookup(
      S.Context.DeclarationNames.getCXXOperatorName(OO_Star));
  if (Res1.empty())
    return false;

  DeclContextLookupResult Res2 = RT->getDecl()->lookup(
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
  const ValueDecl *vd = cast<ValueDecl>(D);
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
    << Attr.getName() << QT;
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

  return nullptr;
}

static bool checkRecordTypeForCapability(Sema &S, QualType Ty) {
  const RecordType *RT = getRecordType(Ty);

  if (!RT)
    return false;

  // Don't check for the capability if the class hasn't been defined yet.
  if (RT->isIncompleteType())
    return true;

  // Allow smart pointers to be used as capability objects.
  // FIXME -- Check the type that the smart pointer points to.
  if (threadSafetyCheckIsSmartPointer(S, RT))
    return true;

  // Check if the record itself has a capability.
  RecordDecl *RD = RT->getDecl();
  if (RD->hasAttr<CapabilityAttr>())
    return true;

  // Else check if any base classes have a capability.
  if (CXXRecordDecl *CRD = dyn_cast<CXXRecordDecl>(RD)) {
    CXXBasePaths BPaths(false, false);
    if (CRD->lookupInBases([](const CXXBaseSpecifier *BS, CXXBasePath &) {
          const auto *Type = BS->getType()->getAs<RecordType>();
          return Type->getDecl()->hasAttr<CapabilityAttr>();
        }, BPaths))
      return true;
  }
  return false;
}

static bool checkTypedefTypeForCapability(QualType Ty) {
  const auto *TD = Ty->getAs<TypedefType>();
  if (!TD)
    return false;

  TypedefNameDecl *TN = TD->getDecl();
  if (!TN)
    return false;

  return TN->hasAttr<CapabilityAttr>();
}

static bool typeHasCapability(Sema &S, QualType Ty) {
  if (checkTypedefTypeForCapability(Ty))
    return true;

  if (checkRecordTypeForCapability(S, Ty))
    return true;

  return false;
}

static bool isCapabilityExpr(Sema &S, const Expr *Ex) {
  // Capability expressions are simple expressions involving the boolean logic
  // operators &&, || or !, a simple DeclRefExpr, CastExpr or a ParenExpr. Once
  // a DeclRefExpr is found, its type should be checked to determine whether it
  // is a capability or not.

  if (const auto *E = dyn_cast<DeclRefExpr>(Ex))
    return typeHasCapability(S, E->getType());
  else if (const auto *E = dyn_cast<CastExpr>(Ex))
    return isCapabilityExpr(S, E->getSubExpr());
  else if (const auto *E = dyn_cast<ParenExpr>(Ex))
    return isCapabilityExpr(S, E->getSubExpr());
  else if (const auto *E = dyn_cast<UnaryOperator>(Ex)) {
    if (E->getOpcode() == UO_LNot)
      return isCapabilityExpr(S, E->getSubExpr());
    return false;
  } else if (const auto *E = dyn_cast<BinaryOperator>(Ex)) {
    if (E->getOpcode() == BO_LAnd || E->getOpcode() == BO_LOr)
      return isCapabilityExpr(S, E->getLHS()) &&
             isCapabilityExpr(S, E->getRHS());
    return false;
  }

  return false;
}

/// \brief Checks that all attribute arguments, starting from Sidx, resolve to
/// a capability object.
/// \param Sidx The attribute argument index to start checking with.
/// \param ParamIdxOk Whether an argument can be indexing into a function
/// parameter list.
static void checkAttrArgsAreCapabilityObjs(Sema &S, Decl *D,
                                           const AttributeList &Attr,
                                           SmallVectorImpl<Expr *> &Args,
                                           int Sidx = 0,
                                           bool ParamIdxOk = false) {
  for (unsigned Idx = Sidx; Idx < Attr.getNumArgs(); ++Idx) {
    Expr *ArgExp = Attr.getArgAsExpr(Idx);

    if (ArgExp->isTypeDependent()) {
      // FIXME -- need to check this again on template instantiation
      Args.push_back(ArgExp);
      continue;
    }

    if (StringLiteral *StrLit = dyn_cast<StringLiteral>(ArgExp)) {
      if (StrLit->getLength() == 0 ||
          (StrLit->isAscii() && StrLit->getString() == StringRef("*"))) {
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

    // First see if we can just cast to record type, or pointer to record type.
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

    // If the type does not have a capability, see if the components of the
    // expression have capabilities. This allows for writing C code where the
    // capability may be on the type, and the expression is a capability
    // boolean logic expression. Eg) requires_capability(A || B && !C)
    if (!typeHasCapability(S, ArgTy) && !isCapabilityExpr(S, ArgExp))
      S.Diag(Attr.getLoc(), diag::warn_thread_attribute_argument_not_lockable)
          << Attr.getName() << ArgTy;

    Args.push_back(ArgExp);
  }
}

//===----------------------------------------------------------------------===//
// Attribute Implementations
//===----------------------------------------------------------------------===//

static void handlePtGuardedVarAttr(Sema &S, Decl *D,
                                   const AttributeList &Attr) {
  if (!threadSafetyCheckIsPointer(S, D, Attr))
    return;

  D->addAttr(::new (S.Context)
             PtGuardedVarAttr(Attr.getRange(), S.Context,
                              Attr.getAttributeSpellingListIndex()));
}

static bool checkGuardedByAttrCommon(Sema &S, Decl *D,
                                     const AttributeList &Attr,
                                     Expr* &Arg) {
  SmallVector<Expr*, 1> Args;
  // check that all arguments are lockable objects
  checkAttrArgsAreCapabilityObjs(S, D, Attr, Args);
  unsigned Size = Args.size();
  if (Size != 1)
    return false;

  Arg = Args[0];

  return true;
}

static void handleGuardedByAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  Expr *Arg = nullptr;
  if (!checkGuardedByAttrCommon(S, D, Attr, Arg))
    return;

  D->addAttr(::new (S.Context) GuardedByAttr(Attr.getRange(), S.Context, Arg,
                                        Attr.getAttributeSpellingListIndex()));
}

static void handlePtGuardedByAttr(Sema &S, Decl *D,
                                  const AttributeList &Attr) {
  Expr *Arg = nullptr;
  if (!checkGuardedByAttrCommon(S, D, Attr, Arg))
    return;

  if (!threadSafetyCheckIsPointer(S, D, Attr))
    return;

  D->addAttr(::new (S.Context) PtGuardedByAttr(Attr.getRange(),
                                               S.Context, Arg,
                                        Attr.getAttributeSpellingListIndex()));
}

static bool checkAcquireOrderAttrCommon(Sema &S, Decl *D,
                                        const AttributeList &Attr,
                                        SmallVectorImpl<Expr *> &Args) {
  if (!checkAttributeAtLeastNumArgs(S, Attr, 1))
    return false;

  // Check that this attribute only applies to lockable types.
  QualType QT = cast<ValueDecl>(D)->getType();
  if (!QT->isDependentType() && !typeHasCapability(S, QT)) {
    S.Diag(Attr.getLoc(), diag::warn_thread_attribute_decl_not_lockable)
      << Attr.getName();
    return false;
  }

  // Check that all arguments are lockable objects.
  checkAttrArgsAreCapabilityObjs(S, D, Attr, Args);
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
                                   SmallVectorImpl<Expr *> &Args) {
  // zero or more arguments ok
  // check that all arguments are lockable objects
  checkAttrArgsAreCapabilityObjs(S, D, Attr, Args, 0, /*ParamIdxOk=*/true);

  return true;
}

static void handleAssertSharedLockAttr(Sema &S, Decl *D,
                                       const AttributeList &Attr) {
  SmallVector<Expr*, 1> Args;
  if (!checkLockFunAttrCommon(S, D, Attr, Args))
    return;

  unsigned Size = Args.size();
  Expr **StartArg = Size == 0 ? nullptr : &Args[0];
  D->addAttr(::new (S.Context)
             AssertSharedLockAttr(Attr.getRange(), S.Context, StartArg, Size,
                                  Attr.getAttributeSpellingListIndex()));
}

static void handleAssertExclusiveLockAttr(Sema &S, Decl *D,
                                          const AttributeList &Attr) {
  SmallVector<Expr*, 1> Args;
  if (!checkLockFunAttrCommon(S, D, Attr, Args))
    return;

  unsigned Size = Args.size();
  Expr **StartArg = Size == 0 ? nullptr : &Args[0];
  D->addAttr(::new (S.Context)
             AssertExclusiveLockAttr(Attr.getRange(), S.Context,
                                     StartArg, Size,
                                     Attr.getAttributeSpellingListIndex()));
}

/// \brief Checks to be sure that the given parameter number is in bounds, and is
/// an integral type. Will emit appropriate diagnostics if this returns
/// false.
///
/// FuncParamNo is expected to be from the user, so is base-1. AttrArgNo is used
/// to actually retrieve the argument, so it's base-0.
template <typename AttrInfo>
static bool checkParamIsIntegerType(Sema &S, const FunctionDecl *FD,
                                    const AttrInfo &Attr, Expr *AttrArg,
                                    unsigned FuncParamNo, unsigned AttrArgNo,
                                    bool AllowDependentType = false) {
  uint64_t Idx;
  if (!checkFunctionOrMethodParameterIndex(S, FD, Attr, FuncParamNo, AttrArg,
                                           Idx))
    return false;

  const ParmVarDecl *Param = FD->getParamDecl(Idx);
  if (AllowDependentType && Param->getType()->isDependentType())
    return true;
  if (!Param->getType()->isIntegerType() && !Param->getType()->isCharType()) {
    SourceLocation SrcLoc = AttrArg->getLocStart();
    S.Diag(SrcLoc, diag::err_attribute_integers_only)
        << getAttrName(Attr) << Param->getSourceRange();
    return false;
  }
  return true;
}

/// \brief Checks to be sure that the given parameter number is in bounds, and is
/// an integral type. Will emit appropriate diagnostics if this returns false.
///
/// FuncParamNo is expected to be from the user, so is base-1. AttrArgNo is used
/// to actually retrieve the argument, so it's base-0.
static bool checkParamIsIntegerType(Sema &S, const FunctionDecl *FD,
                                    const AttributeList &Attr,
                                    unsigned FuncParamNo, unsigned AttrArgNo,
                                    bool AllowDependentType = false) {
  assert(Attr.isArgExpr(AttrArgNo) && "Expected expression argument");
  return checkParamIsIntegerType(S, FD, Attr, Attr.getArgAsExpr(AttrArgNo),
                                 FuncParamNo, AttrArgNo, AllowDependentType);
}

static void handleAllocSizeAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (!checkAttributeAtLeastNumArgs(S, Attr, 1) ||
      !checkAttributeAtMostNumArgs(S, Attr, 2))
    return;

  const auto *FD = cast<FunctionDecl>(D);
  if (!FD->getReturnType()->isPointerType()) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_return_pointers_only)
        << Attr.getName();
    return;
  }

  const Expr *SizeExpr = Attr.getArgAsExpr(0);
  int SizeArgNo;
  // Parameter indices are 1-indexed, hence Index=1
  if (!checkPositiveIntArgument(S, Attr, SizeExpr, SizeArgNo, /*Index=*/1))
    return;

  if (!checkParamIsIntegerType(S, FD, Attr, SizeArgNo, /*AttrArgNo=*/0))
    return;

  // Args are 1-indexed, so 0 implies that the arg was not present
  int NumberArgNo = 0;
  if (Attr.getNumArgs() == 2) {
    const Expr *NumberExpr = Attr.getArgAsExpr(1);
    // Parameter indices are 1-based, hence Index=2
    if (!checkPositiveIntArgument(S, Attr, NumberExpr, NumberArgNo,
                                  /*Index=*/2))
      return;

    if (!checkParamIsIntegerType(S, FD, Attr, NumberArgNo, /*AttrArgNo=*/1))
      return;
  }

  D->addAttr(::new (S.Context) AllocSizeAttr(
      Attr.getRange(), S.Context, SizeArgNo, NumberArgNo,
      Attr.getAttributeSpellingListIndex()));
}

static bool checkTryLockFunAttrCommon(Sema &S, Decl *D,
                                      const AttributeList &Attr,
                                      SmallVectorImpl<Expr *> &Args) {
  if (!checkAttributeAtLeastNumArgs(S, Attr, 1))
    return false;

  if (!isIntOrBool(Attr.getArgAsExpr(0))) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_type)
      << Attr.getName() << 1 << AANT_ArgumentIntOrBool;
    return false;
  }

  // check that all arguments are lockable objects
  checkAttrArgsAreCapabilityObjs(S, D, Attr, Args, 1);

  return true;
}

static void handleSharedTrylockFunctionAttr(Sema &S, Decl *D,
                                            const AttributeList &Attr) {
  SmallVector<Expr*, 2> Args;
  if (!checkTryLockFunAttrCommon(S, D, Attr, Args))
    return;

  D->addAttr(::new (S.Context)
             SharedTrylockFunctionAttr(Attr.getRange(), S.Context,
                                       Attr.getArgAsExpr(0),
                                       Args.data(), Args.size(),
                                       Attr.getAttributeSpellingListIndex()));
}

static void handleExclusiveTrylockFunctionAttr(Sema &S, Decl *D,
                                               const AttributeList &Attr) {
  SmallVector<Expr*, 2> Args;
  if (!checkTryLockFunAttrCommon(S, D, Attr, Args))
    return;

  D->addAttr(::new (S.Context) ExclusiveTrylockFunctionAttr(
      Attr.getRange(), S.Context, Attr.getArgAsExpr(0), Args.data(),
      Args.size(), Attr.getAttributeSpellingListIndex()));
}

static void handleLockReturnedAttr(Sema &S, Decl *D,
                                   const AttributeList &Attr) {
  // check that the argument is lockable object
  SmallVector<Expr*, 1> Args;
  checkAttrArgsAreCapabilityObjs(S, D, Attr, Args);
  unsigned Size = Args.size();
  if (Size == 0)
    return;

  D->addAttr(::new (S.Context)
             LockReturnedAttr(Attr.getRange(), S.Context, Args[0],
                              Attr.getAttributeSpellingListIndex()));
}

static void handleLocksExcludedAttr(Sema &S, Decl *D,
                                    const AttributeList &Attr) {
  if (!checkAttributeAtLeastNumArgs(S, Attr, 1))
    return;

  // check that all arguments are lockable objects
  SmallVector<Expr*, 1> Args;
  checkAttrArgsAreCapabilityObjs(S, D, Attr, Args);
  unsigned Size = Args.size();
  if (Size == 0)
    return;
  Expr **StartArg = &Args[0];

  D->addAttr(::new (S.Context)
             LocksExcludedAttr(Attr.getRange(), S.Context, StartArg, Size,
                               Attr.getAttributeSpellingListIndex()));
}

static bool checkFunctionConditionAttr(Sema &S, Decl *D,
                                       const AttributeList &Attr,
                                       Expr *&Cond, StringRef &Msg) {
  Cond = Attr.getArgAsExpr(0);
  if (!Cond->isTypeDependent()) {
    ExprResult Converted = S.PerformContextuallyConvertToBool(Cond);
    if (Converted.isInvalid())
      return false;
    Cond = Converted.get();
  }

  if (!S.checkStringLiteralArgumentAttr(Attr, 1, Msg))
    return false;

  if (Msg.empty())
    Msg = "<no message provided>";

  SmallVector<PartialDiagnosticAt, 8> Diags;
  if (isa<FunctionDecl>(D) && !Cond->isValueDependent() &&
      !Expr::isPotentialConstantExprUnevaluated(Cond, cast<FunctionDecl>(D),
                                                Diags)) {
    S.Diag(Attr.getLoc(), diag::err_attr_cond_never_constant_expr)
        << Attr.getName();
    for (const PartialDiagnosticAt &PDiag : Diags)
      S.Diag(PDiag.first, PDiag.second);
    return false;
  }
  return true;
}

static void handleEnableIfAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  S.Diag(Attr.getLoc(), diag::ext_clang_enable_if);

  Expr *Cond;
  StringRef Msg;
  if (checkFunctionConditionAttr(S, D, Attr, Cond, Msg))
    D->addAttr(::new (S.Context)
                   EnableIfAttr(Attr.getRange(), S.Context, Cond, Msg,
                                Attr.getAttributeSpellingListIndex()));
}

namespace {
/// Determines if a given Expr references any of the given function's
/// ParmVarDecls, or the function's implicit `this` parameter (if applicable).
class ArgumentDependenceChecker
    : public RecursiveASTVisitor<ArgumentDependenceChecker> {
#ifndef NDEBUG
  const CXXRecordDecl *ClassType;
#endif
  llvm::SmallPtrSet<const ParmVarDecl *, 16> Parms;
  bool Result;

public:
  ArgumentDependenceChecker(const FunctionDecl *FD) {
#ifndef NDEBUG
    if (const auto *MD = dyn_cast<CXXMethodDecl>(FD))
      ClassType = MD->getParent();
    else
      ClassType = nullptr;
#endif
    Parms.insert(FD->param_begin(), FD->param_end());
  }

  bool referencesArgs(Expr *E) {
    Result = false;
    TraverseStmt(E);
    return Result;
  }

  bool VisitCXXThisExpr(CXXThisExpr *E) {
    assert(E->getType()->getPointeeCXXRecordDecl() == ClassType &&
           "`this` doesn't refer to the enclosing class?");
    Result = true;
    return false;
  }

  bool VisitDeclRefExpr(DeclRefExpr *DRE) {
    if (const auto *PVD = dyn_cast<ParmVarDecl>(DRE->getDecl()))
      if (Parms.count(PVD)) {
        Result = true;
        return false;
      }
    return true;
  }
};
}

static void handleDiagnoseIfAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  S.Diag(Attr.getLoc(), diag::ext_clang_diagnose_if);

  Expr *Cond;
  StringRef Msg;
  if (!checkFunctionConditionAttr(S, D, Attr, Cond, Msg))
    return;

  StringRef DiagTypeStr;
  if (!S.checkStringLiteralArgumentAttr(Attr, 2, DiagTypeStr))
    return;

  DiagnoseIfAttr::DiagnosticType DiagType;
  if (!DiagnoseIfAttr::ConvertStrToDiagnosticType(DiagTypeStr, DiagType)) {
    S.Diag(Attr.getArgAsExpr(2)->getLocStart(),
           diag::err_diagnose_if_invalid_diagnostic_type);
    return;
  }

  bool ArgDependent = false;
  if (const auto *FD = dyn_cast<FunctionDecl>(D))
    ArgDependent = ArgumentDependenceChecker(FD).referencesArgs(Cond);
  D->addAttr(::new (S.Context) DiagnoseIfAttr(
      Attr.getRange(), S.Context, Cond, Msg, DiagType, ArgDependent, cast<NamedDecl>(D),
      Attr.getAttributeSpellingListIndex()));
}

static void handlePassObjectSizeAttr(Sema &S, Decl *D,
                                     const AttributeList &Attr) {
  if (D->hasAttr<PassObjectSizeAttr>()) {
    S.Diag(D->getLocStart(), diag::err_attribute_only_once_per_parameter)
        << Attr.getName();
    return;
  }

  Expr *E = Attr.getArgAsExpr(0);
  uint32_t Type;
  if (!checkUInt32Argument(S, Attr, E, Type, /*Idx=*/1))
    return;

  // pass_object_size's argument is passed in as the second argument of
  // __builtin_object_size. So, it has the same constraints as that second
  // argument; namely, it must be in the range [0, 3].
  if (Type > 3) {
    S.Diag(E->getLocStart(), diag::err_attribute_argument_outof_range)
        << Attr.getName() << 0 << 3 << E->getSourceRange();
    return;
  }

  // pass_object_size is only supported on constant pointer parameters; as a
  // kindness to users, we allow the parameter to be non-const for declarations.
  // At this point, we have no clue if `D` belongs to a function declaration or
  // definition, so we defer the constness check until later.
  if (!cast<ParmVarDecl>(D)->getType()->isPointerType()) {
    S.Diag(D->getLocStart(), diag::err_attribute_pointers_only)
        << Attr.getName() << 1;
    return;
  }

  D->addAttr(::new (S.Context)
                 PassObjectSizeAttr(Attr.getRange(), S.Context, (int)Type,
                                    Attr.getAttributeSpellingListIndex()));
}

static void handleConsumableAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  ConsumableAttr::ConsumedState DefaultState;

  if (Attr.isArgIdent(0)) {
    IdentifierLoc *IL = Attr.getArgAsIdent(0);
    if (!ConsumableAttr::ConvertStrToConsumedState(IL->Ident->getName(),
                                                   DefaultState)) {
      S.Diag(IL->Loc, diag::warn_attribute_type_not_supported)
        << Attr.getName() << IL->Ident;
      return;
    }
  } else {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_type)
        << Attr.getName() << AANT_ArgumentIdentifier;
    return;
  }
  
  D->addAttr(::new (S.Context)
             ConsumableAttr(Attr.getRange(), S.Context, DefaultState,
                            Attr.getAttributeSpellingListIndex()));
}

static bool checkForConsumableClass(Sema &S, const CXXMethodDecl *MD,
                                        const AttributeList &Attr) {
  ASTContext &CurrContext = S.getASTContext();
  QualType ThisType = MD->getThisType(CurrContext)->getPointeeType();
  
  if (const CXXRecordDecl *RD = ThisType->getAsCXXRecordDecl()) {
    if (!RD->hasAttr<ConsumableAttr>()) {
      S.Diag(Attr.getLoc(), diag::warn_attr_on_unconsumable_class) <<
        RD->getNameAsString();
      
      return false;
    }
  }
  
  return true;
}

static void handleCallableWhenAttr(Sema &S, Decl *D,
                                   const AttributeList &Attr) {
  if (!checkAttributeAtLeastNumArgs(S, Attr, 1))
    return;
  
  if (!checkForConsumableClass(S, cast<CXXMethodDecl>(D), Attr))
    return;
  
  SmallVector<CallableWhenAttr::ConsumedState, 3> States;
  for (unsigned ArgIndex = 0; ArgIndex < Attr.getNumArgs(); ++ArgIndex) {
    CallableWhenAttr::ConsumedState CallableState;
    
    StringRef StateString;
    SourceLocation Loc;
    if (Attr.isArgIdent(ArgIndex)) {
      IdentifierLoc *Ident = Attr.getArgAsIdent(ArgIndex);
      StateString = Ident->Ident->getName();
      Loc = Ident->Loc;
    } else {
      if (!S.checkStringLiteralArgumentAttr(Attr, ArgIndex, StateString, &Loc))
        return;
    }

    if (!CallableWhenAttr::ConvertStrToConsumedState(StateString,
                                                     CallableState)) {
      S.Diag(Loc, diag::warn_attribute_type_not_supported)
        << Attr.getName() << StateString;
      return;
    }
      
    States.push_back(CallableState);
  }
  
  D->addAttr(::new (S.Context)
             CallableWhenAttr(Attr.getRange(), S.Context, States.data(),
               States.size(), Attr.getAttributeSpellingListIndex()));
}

static void handleParamTypestateAttr(Sema &S, Decl *D,
                                    const AttributeList &Attr) {
  ParamTypestateAttr::ConsumedState ParamState;
  
  if (Attr.isArgIdent(0)) {
    IdentifierLoc *Ident = Attr.getArgAsIdent(0);
    StringRef StateString = Ident->Ident->getName();

    if (!ParamTypestateAttr::ConvertStrToConsumedState(StateString,
                                                       ParamState)) {
      S.Diag(Ident->Loc, diag::warn_attribute_type_not_supported)
        << Attr.getName() << StateString;
      return;
    }
  } else {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_type) <<
      Attr.getName() << AANT_ArgumentIdentifier;
    return;
  }
  
  // FIXME: This check is currently being done in the analysis.  It can be
  //        enabled here only after the parser propagates attributes at
  //        template specialization definition, not declaration.
  //QualType ReturnType = cast<ParmVarDecl>(D)->getType();
  //const CXXRecordDecl *RD = ReturnType->getAsCXXRecordDecl();
  //
  //if (!RD || !RD->hasAttr<ConsumableAttr>()) {
  //    S.Diag(Attr.getLoc(), diag::warn_return_state_for_unconsumable_type) <<
  //      ReturnType.getAsString();
  //    return;
  //}
  
  D->addAttr(::new (S.Context)
             ParamTypestateAttr(Attr.getRange(), S.Context, ParamState,
                                Attr.getAttributeSpellingListIndex()));
}

static void handleReturnTypestateAttr(Sema &S, Decl *D,
                                      const AttributeList &Attr) {
  ReturnTypestateAttr::ConsumedState ReturnState;
  
  if (Attr.isArgIdent(0)) {
    IdentifierLoc *IL = Attr.getArgAsIdent(0);
    if (!ReturnTypestateAttr::ConvertStrToConsumedState(IL->Ident->getName(),
                                                        ReturnState)) {
      S.Diag(IL->Loc, diag::warn_attribute_type_not_supported)
        << Attr.getName() << IL->Ident;
      return;
    }
  } else {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_type) <<
      Attr.getName() << AANT_ArgumentIdentifier;
    return;
  }
  
  // FIXME: This check is currently being done in the analysis.  It can be
  //        enabled here only after the parser propagates attributes at
  //        template specialization definition, not declaration.
  //QualType ReturnType;
  //
  //if (const ParmVarDecl *Param = dyn_cast<ParmVarDecl>(D)) {
  //  ReturnType = Param->getType();
  //
  //} else if (const CXXConstructorDecl *Constructor =
  //             dyn_cast<CXXConstructorDecl>(D)) {
  //  ReturnType = Constructor->getThisType(S.getASTContext())->getPointeeType();
  //  
  //} else {
  //  
  //  ReturnType = cast<FunctionDecl>(D)->getCallResultType();
  //}
  //
  //const CXXRecordDecl *RD = ReturnType->getAsCXXRecordDecl();
  //
  //if (!RD || !RD->hasAttr<ConsumableAttr>()) {
  //    S.Diag(Attr.getLoc(), diag::warn_return_state_for_unconsumable_type) <<
  //      ReturnType.getAsString();
  //    return;
  //}
  
  D->addAttr(::new (S.Context)
             ReturnTypestateAttr(Attr.getRange(), S.Context, ReturnState,
                                 Attr.getAttributeSpellingListIndex()));
}

static void handleSetTypestateAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (!checkForConsumableClass(S, cast<CXXMethodDecl>(D), Attr))
    return;
  
  SetTypestateAttr::ConsumedState NewState;
  if (Attr.isArgIdent(0)) {
    IdentifierLoc *Ident = Attr.getArgAsIdent(0);
    StringRef Param = Ident->Ident->getName();
    if (!SetTypestateAttr::ConvertStrToConsumedState(Param, NewState)) {
      S.Diag(Ident->Loc, diag::warn_attribute_type_not_supported)
        << Attr.getName() << Param;
      return;
    }
  } else {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_type) <<
      Attr.getName() << AANT_ArgumentIdentifier;
    return;
  }
  
  D->addAttr(::new (S.Context)
             SetTypestateAttr(Attr.getRange(), S.Context, NewState,
                              Attr.getAttributeSpellingListIndex()));
}

static void handleTestTypestateAttr(Sema &S, Decl *D,
                                    const AttributeList &Attr) {
  if (!checkForConsumableClass(S, cast<CXXMethodDecl>(D), Attr))
    return;
  
  TestTypestateAttr::ConsumedState TestState;  
  if (Attr.isArgIdent(0)) {
    IdentifierLoc *Ident = Attr.getArgAsIdent(0);
    StringRef Param = Ident->Ident->getName();
    if (!TestTypestateAttr::ConvertStrToConsumedState(Param, TestState)) {
      S.Diag(Ident->Loc, diag::warn_attribute_type_not_supported)
        << Attr.getName() << Param;
      return;
    }
  } else {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_type) <<
      Attr.getName() << AANT_ArgumentIdentifier;
    return;
  }
  
  D->addAttr(::new (S.Context)
             TestTypestateAttr(Attr.getRange(), S.Context, TestState,
                                Attr.getAttributeSpellingListIndex()));
}

static void handleExtVectorTypeAttr(Sema &S, Scope *scope, Decl *D,
                                    const AttributeList &Attr) {
  // Remember this typedef decl, we will need it later for diagnostics.
  S.ExtVectorDecls.push_back(cast<TypedefNameDecl>(D));
}

static void handlePackedAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (TagDecl *TD = dyn_cast<TagDecl>(D))
    TD->addAttr(::new (S.Context) PackedAttr(Attr.getRange(), S.Context,
                                        Attr.getAttributeSpellingListIndex()));
  else if (FieldDecl *FD = dyn_cast<FieldDecl>(D)) {
    // Report warning about changed offset in the newer compiler versions.
    if (!FD->getType()->isDependentType() &&
        !FD->getType()->isIncompleteType() && FD->isBitField() &&
        S.Context.getTypeAlign(FD->getType()) <= 8)
      S.Diag(Attr.getLoc(), diag::warn_attribute_packed_for_bitfield);

    FD->addAttr(::new (S.Context) PackedAttr(
        Attr.getRange(), S.Context, Attr.getAttributeSpellingListIndex()));
  } else
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << Attr.getName();
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
  if (!checkIBOutletCommon(S, D, Attr))
    return;

  D->addAttr(::new (S.Context)
             IBOutletAttr(Attr.getRange(), S.Context,
                          Attr.getAttributeSpellingListIndex()));
}

static void handleIBOutletCollection(Sema &S, Decl *D,
                                     const AttributeList &Attr) {

  // The iboutletcollection attribute can have zero or one arguments.
  if (Attr.getNumArgs() > 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments)
      << Attr.getName() << 1;
    return;
  }

  if (!checkIBOutletCommon(S, D, Attr))
    return;

  ParsedType PT;

  if (Attr.hasParsedType())
    PT = Attr.getTypeArg();
  else {
    PT = S.getTypeName(S.Context.Idents.get("NSObject"), Attr.getLoc(),
                       S.getScopeForContext(D->getDeclContext()->getParent()));
    if (!PT) {
      S.Diag(Attr.getLoc(), diag::err_iboutletcollection_type) << "NSObject";
      return;
    }
  }

  TypeSourceInfo *QTLoc = nullptr;
  QualType QT = S.GetTypeFromParser(PT, &QTLoc);
  if (!QTLoc)
    QTLoc = S.Context.getTrivialTypeSourceInfo(QT, Attr.getLoc());

  // Diagnose use of non-object type in iboutletcollection attribute.
  // FIXME. Gnu attribute extension ignores use of builtin types in
  // attributes. So, __attribute__((iboutletcollection(char))) will be
  // treated as __attribute__((iboutletcollection())).
  if (!QT->isObjCIdType() && !QT->isObjCObjectType()) {
    S.Diag(Attr.getLoc(),
           QT->isBuiltinType() ? diag::err_iboutletcollection_builtintype
                               : diag::err_iboutletcollection_type) << QT;
    return;
  }

  D->addAttr(::new (S.Context)
             IBOutletCollectionAttr(Attr.getRange(), S.Context, QTLoc,
                                    Attr.getAttributeSpellingListIndex()));
}

bool Sema::isValidPointerAttrType(QualType T, bool RefOkay) {
  if (RefOkay) {
    if (T->isReferenceType())
      return true;
  } else {
    T = T.getNonReferenceType();
  }

  // The nonnull attribute, and other similar attributes, can be applied to a
  // transparent union that contains a pointer type.
  if (const RecordType *UT = T->getAsUnionType()) {
    if (UT && UT->getDecl()->hasAttr<TransparentUnionAttr>()) {
      RecordDecl *UD = UT->getDecl();
      for (const auto *I : UD->fields()) {
        QualType QT = I->getType();
        if (QT->isAnyPointerType() || QT->isBlockPointerType())
          return true;
      }
    }
  }

  return T->isAnyPointerType() || T->isBlockPointerType();
}

static bool attrNonNullArgCheck(Sema &S, QualType T, const AttributeList &Attr,
                                SourceRange AttrParmRange,
                                SourceRange TypeRange,
                                bool isReturnValue = false) {
  if (!S.isValidPointerAttrType(T)) {
    if (isReturnValue)
      S.Diag(Attr.getLoc(), diag::warn_attribute_return_pointers_only)
          << Attr.getName() << AttrParmRange << TypeRange;
    else
      S.Diag(Attr.getLoc(), diag::warn_attribute_pointers_only)
          << Attr.getName() << AttrParmRange << TypeRange << 0;
    return false;
  }
  return true;
}

static void handleNonNullAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  SmallVector<unsigned, 8> NonNullArgs;
  for (unsigned I = 0; I < Attr.getNumArgs(); ++I) {
    Expr *Ex = Attr.getArgAsExpr(I);
    uint64_t Idx;
    if (!checkFunctionOrMethodParameterIndex(S, D, Attr, I + 1, Ex, Idx))
      return;

    // Is the function argument a pointer type?
    if (Idx < getFunctionOrMethodNumParams(D) &&
        !attrNonNullArgCheck(S, getFunctionOrMethodParamType(D, Idx), Attr,
                             Ex->getSourceRange(),
                             getFunctionOrMethodParamRange(D, Idx)))
      continue;

    NonNullArgs.push_back(Idx);
  }

  // If no arguments were specified to __attribute__((nonnull)) then all pointer
  // arguments have a nonnull attribute; warn if there aren't any. Skip this
  // check if the attribute came from a macro expansion or a template
  // instantiation.
  if (NonNullArgs.empty() && Attr.getLoc().isFileID() &&
      !S.inTemplateInstantiation()) {
    bool AnyPointers = isFunctionOrMethodVariadic(D);
    for (unsigned I = 0, E = getFunctionOrMethodNumParams(D);
         I != E && !AnyPointers; ++I) {
      QualType T = getFunctionOrMethodParamType(D, I);
      if (T->isDependentType() || S.isValidPointerAttrType(T))
        AnyPointers = true;
    }

    if (!AnyPointers)
      S.Diag(Attr.getLoc(), diag::warn_attribute_nonnull_no_pointers);
  }

  unsigned *Start = NonNullArgs.data();
  unsigned Size = NonNullArgs.size();
  llvm::array_pod_sort(Start, Start + Size);
  D->addAttr(::new (S.Context)
             NonNullAttr(Attr.getRange(), S.Context, Start, Size,
                         Attr.getAttributeSpellingListIndex()));
}

static void handleNonNullAttrParameter(Sema &S, ParmVarDecl *D,
                                       const AttributeList &Attr) {
  if (Attr.getNumArgs() > 0) {
    if (D->getFunctionType()) {
      handleNonNullAttr(S, D, Attr);
    } else {
      S.Diag(Attr.getLoc(), diag::warn_attribute_nonnull_parm_no_args)
        << D->getSourceRange();
    }
    return;
  }

  // Is the argument a pointer type?
  if (!attrNonNullArgCheck(S, D->getType(), Attr, SourceRange(),
                           D->getSourceRange()))
    return;

  D->addAttr(::new (S.Context)
             NonNullAttr(Attr.getRange(), S.Context, nullptr, 0,
                         Attr.getAttributeSpellingListIndex()));
}

static void handleReturnsNonNullAttr(Sema &S, Decl *D,
                                     const AttributeList &Attr) {
  QualType ResultType = getFunctionOrMethodResultType(D);
  SourceRange SR = getFunctionOrMethodResultSourceRange(D);
  if (!attrNonNullArgCheck(S, ResultType, Attr, SourceRange(), SR,
                           /* isReturnValue */ true))
    return;

  D->addAttr(::new (S.Context)
            ReturnsNonNullAttr(Attr.getRange(), S.Context,
                               Attr.getAttributeSpellingListIndex()));
}

static void handleAssumeAlignedAttr(Sema &S, Decl *D,
                                    const AttributeList &Attr) {
  Expr *E = Attr.getArgAsExpr(0),
       *OE = Attr.getNumArgs() > 1 ? Attr.getArgAsExpr(1) : nullptr;
  S.AddAssumeAlignedAttr(Attr.getRange(), D, E, OE,
                         Attr.getAttributeSpellingListIndex());
}

static void handleAllocAlignAttr(Sema &S, Decl *D,
                                   const AttributeList &Attr) {
  S.AddAllocAlignAttr(Attr.getRange(), D, Attr.getArgAsExpr(0),
                      Attr.getAttributeSpellingListIndex());
}

void Sema::AddAssumeAlignedAttr(SourceRange AttrRange, Decl *D, Expr *E,
                                Expr *OE, unsigned SpellingListIndex) {
  QualType ResultType = getFunctionOrMethodResultType(D);
  SourceRange SR = getFunctionOrMethodResultSourceRange(D);

  AssumeAlignedAttr TmpAttr(AttrRange, Context, E, OE, SpellingListIndex);
  SourceLocation AttrLoc = AttrRange.getBegin();

  if (!isValidPointerAttrType(ResultType, /* RefOkay */ true)) {
    Diag(AttrLoc, diag::warn_attribute_return_pointers_refs_only)
      << &TmpAttr << AttrRange << SR;
    return;
  }

  if (!E->isValueDependent()) {
    llvm::APSInt I(64);
    if (!E->isIntegerConstantExpr(I, Context)) {
      if (OE)
        Diag(AttrLoc, diag::err_attribute_argument_n_type)
          << &TmpAttr << 1 << AANT_ArgumentIntegerConstant
          << E->getSourceRange();
      else
        Diag(AttrLoc, diag::err_attribute_argument_type)
          << &TmpAttr << AANT_ArgumentIntegerConstant
          << E->getSourceRange();
      return;
    }

    if (!I.isPowerOf2()) {
      Diag(AttrLoc, diag::err_alignment_not_power_of_two)
        << E->getSourceRange();
      return;
    }
  }

  if (OE) {
    if (!OE->isValueDependent()) {
      llvm::APSInt I(64);
      if (!OE->isIntegerConstantExpr(I, Context)) {
        Diag(AttrLoc, diag::err_attribute_argument_n_type)
          << &TmpAttr << 2 << AANT_ArgumentIntegerConstant
          << OE->getSourceRange();
        return;
      }
    }
  }

  D->addAttr(::new (Context)
            AssumeAlignedAttr(AttrRange, Context, E, OE, SpellingListIndex));
}

void Sema::AddAllocAlignAttr(SourceRange AttrRange, Decl *D, Expr *ParamExpr,
                             unsigned SpellingListIndex) {
  QualType ResultType = getFunctionOrMethodResultType(D);

  AllocAlignAttr TmpAttr(AttrRange, Context, 0, SpellingListIndex);
  SourceLocation AttrLoc = AttrRange.getBegin();

  if (!ResultType->isDependentType() &&
      !isValidPointerAttrType(ResultType, /* RefOkay */ true)) {
    Diag(AttrLoc, diag::warn_attribute_return_pointers_refs_only)
        << &TmpAttr << AttrRange << getFunctionOrMethodResultSourceRange(D);
    return;
  }

  uint64_t IndexVal;
  const auto *FuncDecl = cast<FunctionDecl>(D);
  if (!checkFunctionOrMethodParameterIndex(*this, FuncDecl, TmpAttr,
                                           /*AttrArgNo=*/1, ParamExpr,
                                           IndexVal))
    return;

  QualType Ty = getFunctionOrMethodParamType(D, IndexVal);
  if (!Ty->isDependentType() && !Ty->isIntegralType(Context)) {
    Diag(ParamExpr->getLocStart(), diag::err_attribute_integers_only)
        << &TmpAttr << FuncDecl->getParamDecl(IndexVal)->getSourceRange();
    return;
  }

  // We cannot use the Idx returned from checkFunctionOrMethodParameterIndex
  // because that has corrected for the implicit this parameter, and is zero-
  // based.  The attribute expects what the user wrote explicitly.
  llvm::APSInt Val;
  ParamExpr->EvaluateAsInt(Val, Context);

  D->addAttr(::new (Context) AllocAlignAttr(
      AttrRange, Context, Val.getZExtValue(), SpellingListIndex));
}

/// Normalize the attribute, __foo__ becomes foo.
/// Returns true if normalization was applied.
static bool normalizeName(StringRef &AttrName) {
  if (AttrName.size() > 4 && AttrName.startswith("__") &&
      AttrName.endswith("__")) {
    AttrName = AttrName.drop_front(2).drop_back(2);
    return true;
  }
  return false;
}

static void handleOwnershipAttr(Sema &S, Decl *D, const AttributeList &AL) {
  // This attribute must be applied to a function declaration. The first
  // argument to the attribute must be an identifier, the name of the resource,
  // for example: malloc. The following arguments must be argument indexes, the
  // arguments must be of integer type for Returns, otherwise of pointer type.
  // The difference between Holds and Takes is that a pointer may still be used
  // after being held. free() should be __attribute((ownership_takes)), whereas
  // a list append function may well be __attribute((ownership_holds)).

  if (!AL.isArgIdent(0)) {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_n_type)
      << AL.getName() << 1 << AANT_ArgumentIdentifier;
    return;
  }

  // Figure out our Kind.
  OwnershipAttr::OwnershipKind K =
      OwnershipAttr(AL.getLoc(), S.Context, nullptr, nullptr, 0,
                    AL.getAttributeSpellingListIndex()).getOwnKind();

  // Check arguments.
  switch (K) {
  case OwnershipAttr::Takes:
  case OwnershipAttr::Holds:
    if (AL.getNumArgs() < 2) {
      S.Diag(AL.getLoc(), diag::err_attribute_too_few_arguments)
        << AL.getName() << 2;
      return;
    }
    break;
  case OwnershipAttr::Returns:
    if (AL.getNumArgs() > 2) {
      S.Diag(AL.getLoc(), diag::err_attribute_too_many_arguments)
        << AL.getName() << 1;
      return;
    }
    break;
  }

  IdentifierInfo *Module = AL.getArgAsIdent(0)->Ident;

  StringRef ModuleName = Module->getName();
  if (normalizeName(ModuleName)) {
    Module = &S.PP.getIdentifierTable().get(ModuleName);
  }

  SmallVector<unsigned, 8> OwnershipArgs;
  for (unsigned i = 1; i < AL.getNumArgs(); ++i) {
    Expr *Ex = AL.getArgAsExpr(i);
    uint64_t Idx;
    if (!checkFunctionOrMethodParameterIndex(S, D, AL, i, Ex, Idx))
      return;

    // Is the function argument a pointer type?
    QualType T = getFunctionOrMethodParamType(D, Idx);
    int Err = -1;  // No error
    switch (K) {
      case OwnershipAttr::Takes:
      case OwnershipAttr::Holds:
        if (!T->isAnyPointerType() && !T->isBlockPointerType())
          Err = 0;
        break;
      case OwnershipAttr::Returns:
        if (!T->isIntegerType())
          Err = 1;
        break;
    }
    if (-1 != Err) {
      S.Diag(AL.getLoc(), diag::err_ownership_type) << AL.getName() << Err
        << Ex->getSourceRange();
      return;
    }

    // Check we don't have a conflict with another ownership attribute.
    for (const auto *I : D->specific_attrs<OwnershipAttr>()) {
      // Cannot have two ownership attributes of different kinds for the same
      // index.
      if (I->getOwnKind() != K && I->args_end() !=
          std::find(I->args_begin(), I->args_end(), Idx)) {
        S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible)
          << AL.getName() << I;
        return;
      } else if (K == OwnershipAttr::Returns &&
                 I->getOwnKind() == OwnershipAttr::Returns) {
        // A returns attribute conflicts with any other returns attribute using
        // a different index. Note, diagnostic reporting is 1-based, but stored
        // argument indexes are 0-based.
        if (std::find(I->args_begin(), I->args_end(), Idx) == I->args_end()) {
          S.Diag(I->getLocation(), diag::err_ownership_returns_index_mismatch)
              << *(I->args_begin()) + 1;
          if (I->args_size())
            S.Diag(AL.getLoc(), diag::note_ownership_returns_index_mismatch)
                << (unsigned)Idx + 1 << Ex->getSourceRange();
          return;
        }
      }
    }
    OwnershipArgs.push_back(Idx);
  }

  unsigned* start = OwnershipArgs.data();
  unsigned size = OwnershipArgs.size();
  llvm::array_pod_sort(start, start + size);

  D->addAttr(::new (S.Context)
             OwnershipAttr(AL.getLoc(), S.Context, Module, start, size,
                           AL.getAttributeSpellingListIndex()));
}

static void handleWeakRefAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // Check the attribute arguments.
  if (Attr.getNumArgs() > 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments)
      << Attr.getName() << 1;
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
    S.Diag(Attr.getLoc(), diag::err_attribute_weakref_not_global_context)
      << nd;
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

  // FIXME: it would be good for us to keep the WeakRefAttr as-written instead
  // of transforming it into an AliasAttr.  The WeakRefAttr never uses the
  // StringRef parameter it was given anyway.
  StringRef Str;
  if (Attr.getNumArgs() && S.checkStringLiteralArgumentAttr(Attr, 0, Str))
    // GCC will accept anything as the argument of weakref. Should we
    // check for an existing decl?
    D->addAttr(::new (S.Context) AliasAttr(Attr.getRange(), S.Context, Str,
                                        Attr.getAttributeSpellingListIndex()));

  D->addAttr(::new (S.Context)
             WeakRefAttr(Attr.getRange(), S.Context,
                         Attr.getAttributeSpellingListIndex()));
}

static void handleIFuncAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  StringRef Str;
  if (!S.checkStringLiteralArgumentAttr(Attr, 0, Str))
    return;

  // Aliases should be on declarations, not definitions.
  const auto *FD = cast<FunctionDecl>(D);
  if (FD->isThisDeclarationADefinition()) {
    S.Diag(Attr.getLoc(), diag::err_alias_is_definition) << FD << 1;
    return;
  }
  // FIXME: it should be handled as a target specific attribute.
  if (S.Context.getTargetInfo().getTriple().getObjectFormat() !=
          llvm::Triple::ELF) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << Attr.getName();
    return;
  }

  D->addAttr(::new (S.Context) IFuncAttr(Attr.getRange(), S.Context, Str,
                                         Attr.getAttributeSpellingListIndex()));
}

static void handleAliasAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  StringRef Str;
  if (!S.checkStringLiteralArgumentAttr(Attr, 0, Str))
    return;

  if (S.Context.getTargetInfo().getTriple().isOSDarwin()) {
    S.Diag(Attr.getLoc(), diag::err_alias_not_supported_on_darwin);
    return;
  }
  if (S.Context.getTargetInfo().getTriple().isNVPTX()) {
    S.Diag(Attr.getLoc(), diag::err_alias_not_supported_on_nvptx);
  }

  // Aliases should be on declarations, not definitions.
  if (const auto *FD = dyn_cast<FunctionDecl>(D)) {
    if (FD->isThisDeclarationADefinition()) {
      S.Diag(Attr.getLoc(), diag::err_alias_is_definition) << FD << 0;
      return;
    }
  } else {
    const auto *VD = cast<VarDecl>(D);
    if (VD->isThisDeclarationADefinition() && VD->isExternallyVisible()) {
      S.Diag(Attr.getLoc(), diag::err_alias_is_definition) << VD << 0;
      return;
    }
  }

  // FIXME: check if target symbol exists in current file

  D->addAttr(::new (S.Context) AliasAttr(Attr.getRange(), S.Context, Str,
                                         Attr.getAttributeSpellingListIndex()));
}

static void handleColdAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (checkAttrMutualExclusion<HotAttr>(S, D, Attr.getRange(), Attr.getName()))
    return;

  D->addAttr(::new (S.Context) ColdAttr(Attr.getRange(), S.Context,
                                        Attr.getAttributeSpellingListIndex()));
}

static void handleHotAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (checkAttrMutualExclusion<ColdAttr>(S, D, Attr.getRange(), Attr.getName()))
    return;

  D->addAttr(::new (S.Context) HotAttr(Attr.getRange(), S.Context,
                                       Attr.getAttributeSpellingListIndex()));
}

static void handleTLSModelAttr(Sema &S, Decl *D,
                               const AttributeList &Attr) {
  StringRef Model;
  SourceLocation LiteralLoc;
  // Check that it is a string.
  if (!S.checkStringLiteralArgumentAttr(Attr, 0, Model, &LiteralLoc))
    return;

  // Check that the value.
  if (Model != "global-dynamic" && Model != "local-dynamic"
      && Model != "initial-exec" && Model != "local-exec") {
    S.Diag(LiteralLoc, diag::err_attr_tlsmodel_arg);
    return;
  }

  D->addAttr(::new (S.Context)
             TLSModelAttr(Attr.getRange(), S.Context, Model,
                          Attr.getAttributeSpellingListIndex()));
}

static void handleRestrictAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  QualType ResultType = getFunctionOrMethodResultType(D);
  if (ResultType->isAnyPointerType() || ResultType->isBlockPointerType()) {
    D->addAttr(::new (S.Context) RestrictAttr(
        Attr.getRange(), S.Context, Attr.getAttributeSpellingListIndex()));
    return;
  }

  S.Diag(Attr.getLoc(), diag::warn_attribute_return_pointers_only)
      << Attr.getName() << getFunctionOrMethodResultSourceRange(D);
}

static void handleCommonAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (S.LangOpts.CPlusPlus) {
    S.Diag(Attr.getLoc(), diag::err_attribute_not_supported_in_lang)
        << Attr.getName() << AttributeLangSupport::Cpp;
    return;
  }

  if (CommonAttr *CA = S.mergeCommonAttr(D, Attr.getRange(), Attr.getName(),
                                         Attr.getAttributeSpellingListIndex()))
    D->addAttr(CA);
}

static void handleNakedAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (checkAttrMutualExclusion<DisableTailCallsAttr>(S, D, Attr.getRange(),
                                                     Attr.getName()))
    return;

  if (Attr.isDeclspecAttribute()) {
    const auto &Triple = S.getASTContext().getTargetInfo().getTriple();
    const auto &Arch = Triple.getArch();
    if (Arch != llvm::Triple::x86 &&
        (Arch != llvm::Triple::arm && Arch != llvm::Triple::thumb)) {
      S.Diag(Attr.getLoc(), diag::err_attribute_not_supported_on_arch)
          << Attr.getName() << Triple.getArchName();
      return;
    }
  }

  D->addAttr(::new (S.Context) NakedAttr(Attr.getRange(), S.Context,
                                         Attr.getAttributeSpellingListIndex()));
}

static void handleNoReturnAttr(Sema &S, Decl *D, const AttributeList &attr) {
  if (hasDeclarator(D)) return;

  if (S.CheckNoReturnAttr(attr))
    return;

  if (!isa<ObjCMethodDecl>(D)) {
    S.Diag(attr.getLoc(), diag::warn_attribute_wrong_decl_type)
        << attr.getName() << ExpectedFunctionOrMethod;
    return;
  }

  D->addAttr(::new (S.Context) NoReturnAttr(
      attr.getRange(), S.Context, attr.getAttributeSpellingListIndex()));
}

static void handleNoCallerSavedRegsAttr(Sema &S, Decl *D,
                                        const AttributeList &Attr) {
  if (S.CheckNoCallerSavedRegsAttr(Attr))
    return;

  D->addAttr(::new (S.Context) AnyX86NoCallerSavedRegistersAttr(
      Attr.getRange(), S.Context, Attr.getAttributeSpellingListIndex()));
}

bool Sema::CheckNoReturnAttr(const AttributeList &attr) {
  if (!checkAttributeNumArgs(*this, attr, 0)) {
    attr.setInvalid();
    return true;
  }

  return false;
}

bool Sema::CheckNoCallerSavedRegsAttr(const AttributeList &Attr) {
  // Check whether the attribute is valid on the current target.
  if (!Attr.existsInTarget(Context.getTargetInfo())) {
    Diag(Attr.getLoc(), diag::warn_unknown_attribute_ignored) << Attr.getName();
    Attr.setInvalid();
    return true;
  }

  if (!checkAttributeNumArgs(*this, Attr, 0)) {
    Attr.setInvalid();
    return true;
  }

  return false;
}

static void handleAnalyzerNoReturnAttr(Sema &S, Decl *D,
                                       const AttributeList &Attr) {
  
  // The checking path for 'noreturn' and 'analyzer_noreturn' are different
  // because 'analyzer_noreturn' does not impact the type.
  if (!isFunctionOrMethodOrBlock(D)) {
    ValueDecl *VD = dyn_cast<ValueDecl>(D);
    if (!VD || (!VD->getType()->isBlockPointerType() &&
                !VD->getType()->isFunctionPointerType())) {
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
  if (VecReturnAttr *A = D->getAttr<VecReturnAttr>()) {
    S.Diag(Attr.getLoc(), diag::err_repeat_attribute) << A;
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

  for (const auto *I : record->fields()) {
    if ((count == 1) || !I->getType()->isVectorType()) {
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
  }

  D->addAttr(::new (S.Context) CarriesDependencyAttr(
                                   Attr.getRange(), S.Context,
                                   Attr.getAttributeSpellingListIndex()));
}

static void handleNotTailCalledAttr(Sema &S, Decl *D,
                                    const AttributeList &Attr) {
  if (checkAttrMutualExclusion<AlwaysInlineAttr>(S, D, Attr.getRange(),
                                                 Attr.getName()))
    return;

  D->addAttr(::new (S.Context) NotTailCalledAttr(
      Attr.getRange(), S.Context, Attr.getAttributeSpellingListIndex()));
}

static void handleDisableTailCallsAttr(Sema &S, Decl *D,
                                       const AttributeList &Attr) {
  if (checkAttrMutualExclusion<NakedAttr>(S, D, Attr.getRange(),
                                          Attr.getName()))
    return;

  D->addAttr(::new (S.Context) DisableTailCallsAttr(
      Attr.getRange(), S.Context, Attr.getAttributeSpellingListIndex()));
}

static void handleUsedAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
    if (VD->hasLocalStorage()) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << Attr.getName();
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

static void handleUnusedAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  bool IsCXX1zAttr = Attr.isCXX11Attribute() && !Attr.getScopeName();

  if (IsCXX1zAttr && isa<VarDecl>(D)) {
    // The C++1z spelling of this attribute cannot be applied to a static data
    // member per [dcl.attr.unused]p2.
    if (cast<VarDecl>(D)->isStaticDataMember()) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
          << Attr.getName() << ExpectedForMaybeUnused;
      return;
    }
  }

  // If this is spelled as the standard C++1z attribute, but not in C++1z, warn
  // about using it as an extension.
  if (!S.getLangOpts().CPlusPlus1z && IsCXX1zAttr)
    S.Diag(Attr.getLoc(), diag::ext_cxx1z_attr) << Attr.getName();

  D->addAttr(::new (S.Context) UnusedAttr(
      Attr.getRange(), S.Context, Attr.getAttributeSpellingListIndex()));
}

static void handleConstructorAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  uint32_t priority = ConstructorAttr::DefaultPriority;
  if (Attr.getNumArgs() &&
      !checkUInt32Argument(S, Attr, Attr.getArgAsExpr(0), priority))
    return;

  D->addAttr(::new (S.Context)
             ConstructorAttr(Attr.getRange(), S.Context, priority,
                             Attr.getAttributeSpellingListIndex()));
}

static void handleDestructorAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  uint32_t priority = DestructorAttr::DefaultPriority;
  if (Attr.getNumArgs() &&
      !checkUInt32Argument(S, Attr, Attr.getArgAsExpr(0), priority))
    return;

  D->addAttr(::new (S.Context)
             DestructorAttr(Attr.getRange(), S.Context, priority,
                            Attr.getAttributeSpellingListIndex()));
}

template <typename AttrTy>
static void handleAttrWithMessage(Sema &S, Decl *D,
                                  const AttributeList &Attr) {
  // Handle the case where the attribute has a text message.
  StringRef Str;
  if (Attr.getNumArgs() == 1 && !S.checkStringLiteralArgumentAttr(Attr, 0, Str))
    return;

  D->addAttr(::new (S.Context) AttrTy(Attr.getRange(), S.Context, Str,
                                      Attr.getAttributeSpellingListIndex()));
}

static void handleObjCSuppresProtocolAttr(Sema &S, Decl *D,
                                          const AttributeList &Attr) {
  if (!cast<ObjCProtocolDecl>(D)->isThisDeclarationADefinition()) {
    S.Diag(Attr.getLoc(), diag::err_objc_attr_protocol_requires_definition)
      << Attr.getName() << Attr.getRange();
    return;
  }

  D->addAttr(::new (S.Context)
          ObjCExplicitProtocolImplAttr(Attr.getRange(), S.Context,
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
                                              bool Implicit,
                                              VersionTuple Introduced,
                                              VersionTuple Deprecated,
                                              VersionTuple Obsoleted,
                                              bool IsUnavailable,
                                              StringRef Message,
                                              bool IsStrict,
                                              StringRef Replacement,
                                              AvailabilityMergeKind AMK,
                                              unsigned AttrSpellingListIndex) {
  VersionTuple MergedIntroduced = Introduced;
  VersionTuple MergedDeprecated = Deprecated;
  VersionTuple MergedObsoleted = Obsoleted;
  bool FoundAny = false;
  bool OverrideOrImpl = false;
  switch (AMK) {
  case AMK_None:
  case AMK_Redeclaration:
    OverrideOrImpl = false;
    break;

  case AMK_Override:
  case AMK_ProtocolImplementation:
    OverrideOrImpl = true;
    break;
  }

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

      // If there is an existing availability attribute for this platform that
      // is explicit and the new one is implicit use the explicit one and
      // discard the new implicit attribute.
      if (!OldAA->isImplicit() && Implicit) {
        return nullptr;
      }

      // If there is an existing attribute for this platform that is implicit
      // and the new attribute is explicit then erase the old one and
      // continue processing the attributes.
      if (!Implicit && OldAA->isImplicit()) {
        Attrs.erase(Attrs.begin() + i);
        --e;
        continue;
      }

      FoundAny = true;
      VersionTuple OldIntroduced = OldAA->getIntroduced();
      VersionTuple OldDeprecated = OldAA->getDeprecated();
      VersionTuple OldObsoleted = OldAA->getObsoleted();
      bool OldIsUnavailable = OldAA->getUnavailable();

      if (!versionsMatch(OldIntroduced, Introduced, OverrideOrImpl) ||
          !versionsMatch(Deprecated, OldDeprecated, OverrideOrImpl) ||
          !versionsMatch(Obsoleted, OldObsoleted, OverrideOrImpl) ||
          !(OldIsUnavailable == IsUnavailable ||
            (OverrideOrImpl && !OldIsUnavailable && IsUnavailable))) {
        if (OverrideOrImpl) {
          int Which = -1;
          VersionTuple FirstVersion;
          VersionTuple SecondVersion;
          if (!versionsMatch(OldIntroduced, Introduced, OverrideOrImpl)) {
            Which = 0;
            FirstVersion = OldIntroduced;
            SecondVersion = Introduced;
          } else if (!versionsMatch(Deprecated, OldDeprecated, OverrideOrImpl)) {
            Which = 1;
            FirstVersion = Deprecated;
            SecondVersion = OldDeprecated;
          } else if (!versionsMatch(Obsoleted, OldObsoleted, OverrideOrImpl)) {
            Which = 2;
            FirstVersion = Obsoleted;
            SecondVersion = OldObsoleted;
          }

          if (Which == -1) {
            Diag(OldAA->getLocation(),
                 diag::warn_mismatched_availability_override_unavail)
              << AvailabilityAttr::getPrettyPlatformName(Platform->getName())
              << (AMK == AMK_Override);
          } else {
            Diag(OldAA->getLocation(),
                 diag::warn_mismatched_availability_override)
              << Which
              << AvailabilityAttr::getPrettyPlatformName(Platform->getName())
              << FirstVersion.getAsString() << SecondVersion.getAsString()
              << (AMK == AMK_Override);
          }
          if (AMK == AMK_Override)
            Diag(Range.getBegin(), diag::note_overridden_method);
          else
            Diag(Range.getBegin(), diag::note_protocol_method);
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
    return nullptr;

  // Only create a new attribute if !OverrideOrImpl, but we want to do
  // the checking.
  if (!checkAvailabilityAttr(*this, Range, Platform, MergedIntroduced,
                             MergedDeprecated, MergedObsoleted) &&
      !OverrideOrImpl) {
    auto *Avail =  ::new (Context) AvailabilityAttr(Range, Context, Platform,
                                            Introduced, Deprecated,
                                            Obsoleted, IsUnavailable, Message,
                                            IsStrict, Replacement,
                                            AttrSpellingListIndex);
    Avail->setImplicit(Implicit);
    return Avail;
  }
  return nullptr;
}

static void handleAvailabilityAttr(Sema &S, Decl *D,
                                   const AttributeList &Attr) {
  if (!checkAttributeNumArgs(S, Attr, 1))
    return;
  IdentifierLoc *Platform = Attr.getArgAsIdent(0);
  unsigned Index = Attr.getAttributeSpellingListIndex();
  
  IdentifierInfo *II = Platform->Ident;
  if (AvailabilityAttr::getPrettyPlatformName(II->getName()).empty())
    S.Diag(Platform->Loc, diag::warn_availability_unknown_platform)
      << Platform->Ident;

  NamedDecl *ND = dyn_cast<NamedDecl>(D);
  if (!ND) // We warned about this already, so just return.
    return;

  AvailabilityChange Introduced = Attr.getAvailabilityIntroduced();
  AvailabilityChange Deprecated = Attr.getAvailabilityDeprecated();
  AvailabilityChange Obsoleted = Attr.getAvailabilityObsoleted();
  bool IsUnavailable = Attr.getUnavailableLoc().isValid();
  bool IsStrict = Attr.getStrictLoc().isValid();
  StringRef Str;
  if (const StringLiteral *SE =
          dyn_cast_or_null<StringLiteral>(Attr.getMessageExpr()))
    Str = SE->getString();
  StringRef Replacement;
  if (const StringLiteral *SE =
          dyn_cast_or_null<StringLiteral>(Attr.getReplacementExpr()))
    Replacement = SE->getString();

  AvailabilityAttr *NewAttr = S.mergeAvailabilityAttr(ND, Attr.getRange(), II,
                                                      false/*Implicit*/,
                                                      Introduced.Version,
                                                      Deprecated.Version,
                                                      Obsoleted.Version,
                                                      IsUnavailable, Str,
                                                      IsStrict, Replacement,
                                                      Sema::AMK_None,
                                                      Index);
  if (NewAttr)
    D->addAttr(NewAttr);

  // Transcribe "ios" to "watchos" (and add a new attribute) if the versioning
  // matches before the start of the watchOS platform.
  if (S.Context.getTargetInfo().getTriple().isWatchOS()) {
    IdentifierInfo *NewII = nullptr;
    if (II->getName() == "ios")
      NewII = &S.Context.Idents.get("watchos");
    else if (II->getName() == "ios_app_extension")
      NewII = &S.Context.Idents.get("watchos_app_extension");

    if (NewII) {
        auto adjustWatchOSVersion = [](VersionTuple Version) -> VersionTuple {
          if (Version.empty())
            return Version;
          auto Major = Version.getMajor();
          auto NewMajor = Major >= 9 ? Major - 7 : 0;
          if (NewMajor >= 2) {
            if (Version.getMinor().hasValue()) {
              if (Version.getSubminor().hasValue())
                return VersionTuple(NewMajor, Version.getMinor().getValue(),
                                    Version.getSubminor().getValue());
              else
                return VersionTuple(NewMajor, Version.getMinor().getValue());
            }
          }

          return VersionTuple(2, 0);
        };

        auto NewIntroduced = adjustWatchOSVersion(Introduced.Version);
        auto NewDeprecated = adjustWatchOSVersion(Deprecated.Version);
        auto NewObsoleted = adjustWatchOSVersion(Obsoleted.Version);

        AvailabilityAttr *NewAttr = S.mergeAvailabilityAttr(ND,
                                                            Attr.getRange(),
                                                            NewII,
                                                            true/*Implicit*/,
                                                            NewIntroduced,
                                                            NewDeprecated,
                                                            NewObsoleted,
                                                            IsUnavailable, Str,
                                                            IsStrict,
                                                            Replacement,
                                                            Sema::AMK_None,
                                                            Index);
        if (NewAttr)
          D->addAttr(NewAttr);
      }
  } else if (S.Context.getTargetInfo().getTriple().isTvOS()) {
    // Transcribe "ios" to "tvos" (and add a new attribute) if the versioning
    // matches before the start of the tvOS platform.
    IdentifierInfo *NewII = nullptr;
    if (II->getName() == "ios")
      NewII = &S.Context.Idents.get("tvos");
    else if (II->getName() == "ios_app_extension")
      NewII = &S.Context.Idents.get("tvos_app_extension");

    if (NewII) {
        AvailabilityAttr *NewAttr = S.mergeAvailabilityAttr(ND,
                                                            Attr.getRange(),
                                                            NewII,
                                                            true/*Implicit*/,
                                                            Introduced.Version,
                                                            Deprecated.Version,
                                                            Obsoleted.Version,
                                                            IsUnavailable, Str,
                                                            IsStrict,
                                                            Replacement,
                                                            Sema::AMK_None,
                                                            Index);
        if (NewAttr)
          D->addAttr(NewAttr);
      }
  }
}

static void handleExternalSourceSymbolAttr(Sema &S, Decl *D,
                                           const AttributeList &Attr) {
  if (!checkAttributeAtLeastNumArgs(S, Attr, 1))
    return;
  assert(checkAttributeAtMostNumArgs(S, Attr, 3) &&
         "Invalid number of arguments in an external_source_symbol attribute");

  StringRef Language;
  if (const auto *SE = dyn_cast_or_null<StringLiteral>(Attr.getArgAsExpr(0)))
    Language = SE->getString();
  StringRef DefinedIn;
  if (const auto *SE = dyn_cast_or_null<StringLiteral>(Attr.getArgAsExpr(1)))
    DefinedIn = SE->getString();
  bool IsGeneratedDeclaration = Attr.getArgAsIdent(2) != nullptr;

  D->addAttr(::new (S.Context) ExternalSourceSymbolAttr(
      Attr.getRange(), S.Context, Language, DefinedIn, IsGeneratedDeclaration,
      Attr.getAttributeSpellingListIndex()));
}

template <class T>
static T *mergeVisibilityAttr(Sema &S, Decl *D, SourceRange range,
                              typename T::VisibilityType value,
                              unsigned attrSpellingListIndex) {
  T *existingAttr = D->getAttr<T>();
  if (existingAttr) {
    typename T::VisibilityType existingValue = existingAttr->getVisibility();
    if (existingValue == value)
      return nullptr;
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

  // Check that the argument is a string literal.
  StringRef TypeStr;
  SourceLocation LiteralLoc;
  if (!S.checkStringLiteralArgumentAttr(Attr, 0, TypeStr, &LiteralLoc))
    return;

  VisibilityAttr::VisibilityType type;
  if (!VisibilityAttr::ConvertStrToVisibilityType(TypeStr, type)) {
    S.Diag(LiteralLoc, diag::warn_attribute_type_not_supported)
      << Attr.getName() << TypeStr;
    return;
  }
  
  // Complain about attempts to use protected visibility on targets
  // (like Darwin) that don't support it.
  if (type == VisibilityAttr::Protected &&
      !S.Context.getTargetInfo().hasProtectedVisibility()) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_protected_visibility);
    type = VisibilityAttr::Default;
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
  ObjCMethodDecl *method = cast<ObjCMethodDecl>(decl);
  if (!Attr.isArgIdent(0)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_type)
      << Attr.getName() << 1 << AANT_ArgumentIdentifier;
    return;
  }

  IdentifierLoc *IL = Attr.getArgAsIdent(0);
  ObjCMethodFamilyAttr::FamilyKind F;
  if (!ObjCMethodFamilyAttr::ConvertStrToFamilyKind(IL->Ident->getName(), F)) {
    S.Diag(IL->Loc, diag::warn_attribute_type_not_supported) << Attr.getName()
      << IL->Ident;
    return;
  }

  if (F == ObjCMethodFamilyAttr::OMF_init &&
      !method->getReturnType()->isObjCObjectPointerType()) {
    S.Diag(method->getLocation(), diag::err_init_method_bad_return_type)
        << method->getReturnType();
    // Ignore the attribute.
    return;
  }

  method->addAttr(new (S.Context) ObjCMethodFamilyAttr(Attr.getRange(),
                                                       S.Context, F,
                                        Attr.getAttributeSpellingListIndex()));
}

static void handleObjCNSObject(Sema &S, Decl *D, const AttributeList &Attr) {
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

static void handleObjCIndependentClass(Sema &S, Decl *D, const AttributeList &Attr) {
  if (TypedefNameDecl *TD = dyn_cast<TypedefNameDecl>(D)) {
    QualType T = TD->getUnderlyingType();
    if (!T->isObjCObjectPointerType()) {
      S.Diag(TD->getLocation(), diag::warn_ptr_independentclass_attribute);
      return;
    }
  } else {
    S.Diag(D->getLocation(), diag::warn_independentclass_attribute);
    return;
  }
  D->addAttr(::new (S.Context)
             ObjCIndependentClassAttr(Attr.getRange(), S.Context,
                              Attr.getAttributeSpellingListIndex()));
}

static void handleBlocksAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (!Attr.isArgIdent(0)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_type)
      << Attr.getName() << 1 << AANT_ArgumentIdentifier;
    return;
  }

  IdentifierInfo *II = Attr.getArgAsIdent(0)->Ident;
  BlocksAttr::BlockType type;
  if (!BlocksAttr::ConvertStrToBlockType(II->getName(), type)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_type_not_supported)
      << Attr.getName() << II;
    return;
  }

  D->addAttr(::new (S.Context)
             BlocksAttr(Attr.getRange(), S.Context, type,
                        Attr.getAttributeSpellingListIndex()));
}

static void handleSentinelAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  unsigned sentinel = (unsigned)SentinelAttr::DefaultSentinel;
  if (Attr.getNumArgs() > 0) {
    Expr *E = Attr.getArgAsExpr(0);
    llvm::APSInt Idx(32);
    if (E->isTypeDependent() || E->isValueDependent() ||
        !E->isIntegerConstantExpr(Idx, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_type)
        << Attr.getName() << 1 << AANT_ArgumentIntegerConstant
        << E->getSourceRange();
      return;
    }

    if (Idx.isSigned() && Idx.isNegative()) {
      S.Diag(Attr.getLoc(), diag::err_attribute_sentinel_less_than_zero)
        << E->getSourceRange();
      return;
    }

    sentinel = Idx.getZExtValue();
  }

  unsigned nullPos = (unsigned)SentinelAttr::DefaultNullPos;
  if (Attr.getNumArgs() > 1) {
    Expr *E = Attr.getArgAsExpr(1);
    llvm::APSInt Idx(32);
    if (E->isTypeDependent() || E->isValueDependent() ||
        !E->isIntegerConstantExpr(Idx, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_type)
        << Attr.getName() << 2 << AANT_ArgumentIntegerConstant
        << E->getSourceRange();
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
      const FunctionType *FT = Ty->isFunctionPointerType()
       ? D->getFunctionType()
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
  if (D->getFunctionType() &&
      D->getFunctionType()->getReturnType()->isVoidType()) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_void_function_method)
      << Attr.getName() << 0;
    return;
  }
  if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D))
    if (MD->getReturnType()->isVoidType()) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_void_function_method)
      << Attr.getName() << 1;
      return;
    }
  
  // If this is spelled as the standard C++1z attribute, but not in C++1z, warn
  // about using it as an extension.
  if (!S.getLangOpts().CPlusPlus1z && Attr.isCXX11Attribute() &&
      !Attr.getScopeName())
    S.Diag(Attr.getLoc(), diag::ext_cxx1z_attr) << Attr.getName();

  D->addAttr(::new (S.Context) 
             WarnUnusedResultAttr(Attr.getRange(), S.Context,
                                  Attr.getAttributeSpellingListIndex()));
}

static void handleWeakImportAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // weak_import only applies to variable & function declarations.
  bool isDef = false;
  if (!D->canBeWeakImported(isDef)) {
    if (isDef)
      S.Diag(Attr.getLoc(), diag::warn_attribute_invalid_on_definition)
        << "weak_import";
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
template <typename WorkGroupAttr>
static void handleWorkGroupSize(Sema &S, Decl *D,
                                const AttributeList &Attr) {
  uint32_t WGSize[3];
  for (unsigned i = 0; i < 3; ++i) {
    const Expr *E = Attr.getArgAsExpr(i);
    if (!checkUInt32Argument(S, Attr, E, WGSize[i], i))
      return;
    if (WGSize[i] == 0) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_is_zero)
        << Attr.getName() << E->getSourceRange();
      return;
    }
  }

  WorkGroupAttr *Existing = D->getAttr<WorkGroupAttr>();
  if (Existing && !(Existing->getXDim() == WGSize[0] &&
                    Existing->getYDim() == WGSize[1] &&
                    Existing->getZDim() == WGSize[2]))
    S.Diag(Attr.getLoc(), diag::warn_duplicate_attribute) << Attr.getName();

  D->addAttr(::new (S.Context) WorkGroupAttr(Attr.getRange(), S.Context,
                                             WGSize[0], WGSize[1], WGSize[2],
                                       Attr.getAttributeSpellingListIndex()));
}

// Handles intel_reqd_sub_group_size.
static void handleSubGroupSize(Sema &S, Decl *D, const AttributeList &Attr) {
  uint32_t SGSize;
  const Expr *E = Attr.getArgAsExpr(0);
  if (!checkUInt32Argument(S, Attr, E, SGSize))
    return;
  if (SGSize == 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_is_zero)
        << Attr.getName() << E->getSourceRange();
    return;
  }

  OpenCLIntelReqdSubGroupSizeAttr *Existing =
      D->getAttr<OpenCLIntelReqdSubGroupSizeAttr>();
  if (Existing && Existing->getSubGroupSize() != SGSize)
    S.Diag(Attr.getLoc(), diag::warn_duplicate_attribute) << Attr.getName();

  D->addAttr(::new (S.Context) OpenCLIntelReqdSubGroupSizeAttr(
      Attr.getRange(), S.Context, SGSize,
      Attr.getAttributeSpellingListIndex()));
}

static void handleVecTypeHint(Sema &S, Decl *D, const AttributeList &Attr) {
  if (!Attr.hasParsedType()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments)
      << Attr.getName() << 1;
    return;
  }

  TypeSourceInfo *ParmTSI = nullptr;
  QualType ParmType = S.GetTypeFromParser(Attr.getTypeArg(), &ParmTSI);
  assert(ParmTSI && "no type source info for attribute argument");

  if (!ParmType->isExtVectorType() && !ParmType->isFloatingType() &&
      (ParmType->isBooleanType() ||
       !ParmType->isIntegralType(S.getASTContext()))) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_vec_type_hint)
        << ParmType;
    return;
  }

  if (VecTypeHintAttr *A = D->getAttr<VecTypeHintAttr>()) {
    if (!S.Context.hasSameType(A->getTypeHint(), ParmType)) {
      S.Diag(Attr.getLoc(), diag::warn_duplicate_attribute) << Attr.getName();
      return;
    }
  }

  D->addAttr(::new (S.Context) VecTypeHintAttr(Attr.getLoc(), S.Context,
                                               ParmTSI,
                                        Attr.getAttributeSpellingListIndex()));
}

SectionAttr *Sema::mergeSectionAttr(Decl *D, SourceRange Range,
                                    StringRef Name,
                                    unsigned AttrSpellingListIndex) {
  if (SectionAttr *ExistingAttr = D->getAttr<SectionAttr>()) {
    if (ExistingAttr->getName() == Name)
      return nullptr;
    Diag(ExistingAttr->getLocation(), diag::warn_mismatched_section);
    Diag(Range.getBegin(), diag::note_previous_attribute);
    return nullptr;
  }
  return ::new (Context) SectionAttr(Range, Context, Name,
                                     AttrSpellingListIndex);
}

bool Sema::checkSectionName(SourceLocation LiteralLoc, StringRef SecName) {
  std::string Error = Context.getTargetInfo().isValidSectionSpecifier(SecName);
  if (!Error.empty()) {
    Diag(LiteralLoc, diag::err_attribute_section_invalid_for_target) << Error;
    return false;
  }
  return true;
}

static void handleSectionAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // Make sure that there is a string literal as the sections's single
  // argument.
  StringRef Str;
  SourceLocation LiteralLoc;
  if (!S.checkStringLiteralArgumentAttr(Attr, 0, Str, &LiteralLoc))
    return;

  if (!S.checkSectionName(LiteralLoc, Str))
    return;

  // If the target wants to validate the section specifier, make it happen.
  std::string Error = S.Context.getTargetInfo().isValidSectionSpecifier(Str);
  if (!Error.empty()) {
    S.Diag(LiteralLoc, diag::err_attribute_section_invalid_for_target)
    << Error;
    return;
  }

  unsigned Index = Attr.getAttributeSpellingListIndex();
  SectionAttr *NewAttr = S.mergeSectionAttr(D, Attr.getRange(), Str, Index);
  if (NewAttr)
    D->addAttr(NewAttr);
}

// Check for things we'd like to warn about, no errors or validation for now.
// TODO: Validation should use a backend target library that specifies
// the allowable subtarget features and cpus. We could use something like a
// TargetCodeGenInfo hook here to do validation.
void Sema::checkTargetAttr(SourceLocation LiteralLoc, StringRef AttrStr) {
  for (auto Str : {"tune=", "fpmath="})
    if (AttrStr.find(Str) != StringRef::npos)
      Diag(LiteralLoc, diag::warn_unsupported_target_attribute) << Str;
}

static void handleTargetAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  StringRef Str;
  SourceLocation LiteralLoc;
  if (!S.checkStringLiteralArgumentAttr(Attr, 0, Str, &LiteralLoc))
    return;
  S.checkTargetAttr(LiteralLoc, Str);
  unsigned Index = Attr.getAttributeSpellingListIndex();
  TargetAttr *NewAttr =
      ::new (S.Context) TargetAttr(Attr.getRange(), S.Context, Str, Index);
  D->addAttr(NewAttr);
}

static void handleCleanupAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  VarDecl *VD = cast<VarDecl>(D);
  if (!VD->hasLocalStorage()) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << Attr.getName();
    return;
  }

  Expr *E = Attr.getArgAsExpr(0);
  SourceLocation Loc = E->getExprLoc();
  FunctionDecl *FD = nullptr;
  DeclarationNameInfo NI;

  // gcc only allows for simple identifiers. Since we support more than gcc, we
  // will warn the user.
  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    if (DRE->hasQualifier())
      S.Diag(Loc, diag::warn_cleanup_ext);
    FD = dyn_cast<FunctionDecl>(DRE->getDecl());
    NI = DRE->getNameInfo();
    if (!FD) {
      S.Diag(Loc, diag::err_attribute_cleanup_arg_not_function) << 1
        << NI.getName();
      return;
    }
  } else if (UnresolvedLookupExpr *ULE = dyn_cast<UnresolvedLookupExpr>(E)) {
    if (ULE->hasExplicitTemplateArgs())
      S.Diag(Loc, diag::warn_cleanup_ext);
    FD = S.ResolveSingleFunctionTemplateSpecialization(ULE, true);
    NI = ULE->getNameInfo();
    if (!FD) {
      S.Diag(Loc, diag::err_attribute_cleanup_arg_not_function) << 2
        << NI.getName();
      if (ULE->getType() == S.Context.OverloadTy)
        S.NoteAllOverloadCandidates(ULE);
      return;
    }
  } else {
    S.Diag(Loc, diag::err_attribute_cleanup_arg_not_function) << 0;
    return;
  }

  if (FD->getNumParams() != 1) {
    S.Diag(Loc, diag::err_attribute_cleanup_func_must_take_one_arg)
      << NI.getName();
    return;
  }

  // We're currently more strict than GCC about what function types we accept.
  // If this ever proves to be a problem it should be easy to fix.
  QualType Ty = S.Context.getPointerType(VD->getType());
  QualType ParamTy = FD->getParamDecl(0)->getType();
  if (S.CheckAssignmentConstraints(FD->getParamDecl(0)->getLocation(),
                                   ParamTy, Ty) != Sema::Compatible) {
    S.Diag(Loc, diag::err_attribute_cleanup_func_arg_incompatible_type)
      << NI.getName() << ParamTy << Ty;
    return;
  }

  D->addAttr(::new (S.Context)
             CleanupAttr(Attr.getRange(), S.Context, FD,
                         Attr.getAttributeSpellingListIndex()));
}

static void handleEnumExtensibilityAttr(Sema &S, Decl *D,
                                        const AttributeList &Attr) {
  if (!Attr.isArgIdent(0)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_type)
        << Attr.getName() << 0 << AANT_ArgumentIdentifier;
    return;
  }

  EnumExtensibilityAttr::Kind ExtensibilityKind;
  IdentifierInfo *II = Attr.getArgAsIdent(0)->Ident;
  if (!EnumExtensibilityAttr::ConvertStrToKind(II->getName(),
                                               ExtensibilityKind)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_type_not_supported)
        << Attr.getName() << II;
    return;
  }

  D->addAttr(::new (S.Context) EnumExtensibilityAttr(
      Attr.getRange(), S.Context, ExtensibilityKind,
      Attr.getAttributeSpellingListIndex()));
}

/// Handle __attribute__((format_arg((idx)))) attribute based on
/// http://gcc.gnu.org/onlinedocs/gcc/Function-Attributes.html
static void handleFormatArgAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  Expr *IdxExpr = Attr.getArgAsExpr(0);
  uint64_t Idx;
  if (!checkFunctionOrMethodParameterIndex(S, D, Attr, 1, IdxExpr, Idx))
    return;

  // Make sure the format string is really a string.
  QualType Ty = getFunctionOrMethodParamType(D, Idx);

  bool NotNSStringTy = !isNSStringType(Ty, S.Context);
  if (NotNSStringTy &&
      !isCFStringType(Ty, S.Context) &&
      (!Ty->isPointerType() ||
       !Ty->getAs<PointerType>()->getPointeeType()->isCharType())) {
    S.Diag(Attr.getLoc(), diag::err_format_attribute_not)
        << "a string type" << IdxExpr->getSourceRange()
        << getFunctionOrMethodParamRange(D, 0);
    return;
  }
  Ty = getFunctionOrMethodResultType(D);
  if (!isNSStringType(Ty, S.Context) &&
      !isCFStringType(Ty, S.Context) &&
      (!Ty->isPointerType() ||
       !Ty->getAs<PointerType>()->getPointeeType()->isCharType())) {
    S.Diag(Attr.getLoc(), diag::err_format_attribute_result_not)
        << (NotNSStringTy ? "string type" : "NSString")
        << IdxExpr->getSourceRange() << getFunctionOrMethodParamRange(D, 0);
    return;
  }

  // We cannot use the Idx returned from checkFunctionOrMethodParameterIndex
  // because that has corrected for the implicit this parameter, and is zero-
  // based.  The attribute expects what the user wrote explicitly.
  llvm::APSInt Val;
  IdxExpr->EvaluateAsInt(Val, S.Context);

  D->addAttr(::new (S.Context)
             FormatArgAttr(Attr.getRange(), S.Context, Val.getZExtValue(),
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
      .Case("kprintf", SupportedFormat)         // OpenBSD.
      .Case("freebsd_kprintf", SupportedFormat) // FreeBSD.
      .Case("os_trace", SupportedFormat)
      .Case("os_log", SupportedFormat)

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
  
  if (S.getCurFunctionOrMethodDecl()) {
    S.Diag(Attr.getLoc(), diag::err_init_priority_object_attr);
    Attr.setInvalid();
    return;
  }
  QualType T = cast<VarDecl>(D)->getType();
  if (S.Context.getAsArrayType(T))
    T = S.Context.getBaseElementType(T);
  if (!T->getAs<RecordType>()) {
    S.Diag(Attr.getLoc(), diag::err_init_priority_object_attr);
    Attr.setInvalid();
    return;
  }

  Expr *E = Attr.getArgAsExpr(0);
  uint32_t prioritynum;
  if (!checkUInt32Argument(S, Attr, E, prioritynum)) {
    Attr.setInvalid();
    return;
  }

  if (prioritynum < 101 || prioritynum > 65535) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_outof_range)
      << E->getSourceRange() << Attr.getName() << 101 << 65535;
    Attr.setInvalid();
    return;
  }
  D->addAttr(::new (S.Context)
             InitPriorityAttr(Attr.getRange(), S.Context, prioritynum,
                              Attr.getAttributeSpellingListIndex()));
}

FormatAttr *Sema::mergeFormatAttr(Decl *D, SourceRange Range,
                                  IdentifierInfo *Format, int FormatIdx,
                                  int FirstArg,
                                  unsigned AttrSpellingListIndex) {
  // Check whether we already have an equivalent format attribute.
  for (auto *F : D->specific_attrs<FormatAttr>()) {
    if (F->getType() == Format &&
        F->getFormatIdx() == FormatIdx &&
        F->getFirstArg() == FirstArg) {
      // If we don't have a valid location for this attribute, adopt the
      // location.
      if (F->getLocation().isInvalid())
        F->setRange(Range);
      return nullptr;
    }
  }

  return ::new (Context) FormatAttr(Range, Context, Format, FormatIdx,
                                    FirstArg, AttrSpellingListIndex);
}

/// Handle __attribute__((format(type,idx,firstarg))) attributes based on
/// http://gcc.gnu.org/onlinedocs/gcc/Function-Attributes.html
static void handleFormatAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (!Attr.isArgIdent(0)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_type)
      << Attr.getName() << 1 << AANT_ArgumentIdentifier;
    return;
  }

  // In C++ the implicit 'this' function parameter also counts, and they are
  // counted from one.
  bool HasImplicitThisParam = isInstanceMethod(D);
  unsigned NumArgs = getFunctionOrMethodNumParams(D) + HasImplicitThisParam;

  IdentifierInfo *II = Attr.getArgAsIdent(0)->Ident;
  StringRef Format = II->getName();

  if (normalizeName(Format)) {
    // If we've modified the string name, we need a new identifier for it.
    II = &S.Context.Idents.get(Format);
  }

  // Check for supported formats.
  FormatAttrKind Kind = getFormatAttrKind(Format);
  
  if (Kind == IgnoredFormat)
    return;
  
  if (Kind == InvalidFormat) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_type_not_supported)
      << Attr.getName() << II->getName();
    return;
  }

  // checks for the 2nd argument
  Expr *IdxExpr = Attr.getArgAsExpr(1);
  uint32_t Idx;
  if (!checkUInt32Argument(S, Attr, IdxExpr, Idx, 2))
    return;

  if (Idx < 1 || Idx > NumArgs) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_out_of_bounds)
      << Attr.getName() << 2 << IdxExpr->getSourceRange();
    return;
  }

  // FIXME: Do we need to bounds check?
  unsigned ArgIdx = Idx - 1;

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
  QualType Ty = getFunctionOrMethodParamType(D, ArgIdx);

  if (Kind == CFStringFormat) {
    if (!isCFStringType(Ty, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_format_attribute_not)
        << "a CFString" << IdxExpr->getSourceRange()
        << getFunctionOrMethodParamRange(D, ArgIdx);
      return;
    }
  } else if (Kind == NSStringFormat) {
    // FIXME: do we need to check if the type is NSString*?  What are the
    // semantics?
    if (!isNSStringType(Ty, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_format_attribute_not)
        << "an NSString" << IdxExpr->getSourceRange()
        << getFunctionOrMethodParamRange(D, ArgIdx);
      return;
    }
  } else if (!Ty->isPointerType() ||
             !Ty->getAs<PointerType>()->getPointeeType()->isCharType()) {
    S.Diag(Attr.getLoc(), diag::err_format_attribute_not)
      << "a string type" << IdxExpr->getSourceRange()
      << getFunctionOrMethodParamRange(D, ArgIdx);
    return;
  }

  // check the 3rd argument
  Expr *FirstArgExpr = Attr.getArgAsExpr(2);
  uint32_t FirstArg;
  if (!checkUInt32Argument(S, Attr, FirstArgExpr, FirstArg, 3))
    return;

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
      << Attr.getName() << 3 << FirstArgExpr->getSourceRange();
    return;
  }

  FormatAttr *NewAttr = S.mergeFormatAttr(D, Attr.getRange(), II,
                                          Idx, FirstArg,
                                          Attr.getAttributeSpellingListIndex());
  if (NewAttr)
    D->addAttr(NewAttr);
}

static void handleTransparentUnionAttr(Sema &S, Decl *D,
                                       const AttributeList &Attr) {
  // Try to find the underlying union declaration.
  RecordDecl *RD = nullptr;
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
    if (!RD->isBeingDefined())
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

  if (FirstType->isIncompleteType())
    return;
  uint64_t FirstSize = S.Context.getTypeSize(FirstType);
  uint64_t FirstAlign = S.Context.getTypeAlign(FirstType);
  for (; Field != FieldEnd; ++Field) {
    QualType FieldType = Field->getType();
    if (FieldType->isIncompleteType())
      return;
    // FIXME: this isn't fully correct; we also need to test whether the
    // members of the union would all have the same calling convention as the
    // first member of the union. Checking just the size and alignment isn't
    // sufficient (consider structs passed on the stack instead of in registers
    // as an example).
    if (S.Context.getTypeSize(FieldType) != FirstSize ||
        S.Context.getTypeAlign(FieldType) > FirstAlign) {
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
  // Make sure that there is a string literal as the annotation's single
  // argument.
  StringRef Str;
  if (!S.checkStringLiteralArgumentAttr(Attr, 0, Str))
    return;

  // Don't duplicate annotations that are already set.
  for (const auto *I : D->specific_attrs<AnnotateAttr>()) {
    if (I->getAnnotation() == Str)
      return;
  }
  
  D->addAttr(::new (S.Context)
             AnnotateAttr(Attr.getRange(), S.Context, Str,
                          Attr.getAttributeSpellingListIndex()));
}

static void handleAlignValueAttr(Sema &S, Decl *D,
                                 const AttributeList &Attr) {
  S.AddAlignValueAttr(Attr.getRange(), D, Attr.getArgAsExpr(0),
                      Attr.getAttributeSpellingListIndex());
}

void Sema::AddAlignValueAttr(SourceRange AttrRange, Decl *D, Expr *E,
                             unsigned SpellingListIndex) {
  AlignValueAttr TmpAttr(AttrRange, Context, E, SpellingListIndex);
  SourceLocation AttrLoc = AttrRange.getBegin();

  QualType T;
  if (TypedefNameDecl *TD = dyn_cast<TypedefNameDecl>(D))
    T = TD->getUnderlyingType();
  else if (ValueDecl *VD = dyn_cast<ValueDecl>(D))
    T = VD->getType();
  else
    llvm_unreachable("Unknown decl type for align_value");

  if (!T->isDependentType() && !T->isAnyPointerType() &&
      !T->isReferenceType() && !T->isMemberPointerType()) {
    Diag(AttrLoc, diag::warn_attribute_pointer_or_reference_only)
      << &TmpAttr /*TmpAttr.getName()*/ << T << D->getSourceRange();
    return;
  }

  if (!E->isValueDependent()) {
    llvm::APSInt Alignment;
    ExprResult ICE
      = VerifyIntegerConstantExpression(E, &Alignment,
          diag::err_align_value_attribute_argument_not_int,
            /*AllowFold*/ false);
    if (ICE.isInvalid())
      return;

    if (!Alignment.isPowerOf2()) {
      Diag(AttrLoc, diag::err_alignment_not_power_of_two)
        << E->getSourceRange();
      return;
    }

    D->addAttr(::new (Context)
               AlignValueAttr(AttrRange, Context, ICE.get(),
               SpellingListIndex));
    return;
  }

  // Save dependent expressions in the AST to be instantiated.
  D->addAttr(::new (Context) AlignValueAttr(TmpAttr));
}

static void handleAlignedAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.getNumArgs() > 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments)
      << Attr.getName() << 1;
    return;
  }

  if (Attr.getNumArgs() == 0) {
    D->addAttr(::new (S.Context) AlignedAttr(Attr.getRange(), S.Context,
               true, nullptr, Attr.getAttributeSpellingListIndex()));
    return;
  }

  Expr *E = Attr.getArgAsExpr(0);
  if (Attr.isPackExpansion() && !E->containsUnexpandedParameterPack()) {
    S.Diag(Attr.getEllipsisLoc(),
           diag::err_pack_expansion_without_parameter_packs);
    return;
  }

  if (!Attr.isPackExpansion() && S.DiagnoseUnexpandedParameterPack(E))
    return;

  if (E->isValueDependent()) {
    if (const auto *TND = dyn_cast<TypedefNameDecl>(D)) {
      if (!TND->getUnderlyingType()->isDependentType()) {
        S.Diag(Attr.getLoc(), diag::err_alignment_dependent_typedef_name)
            << E->getSourceRange();
        return;
      }
    }
  }

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
      Diag(AttrLoc, diag::err_attribute_wrong_decl_type) << &TmpAttr
        << (TmpAttr.isC11() ? ExpectedVariableOrField
                            : ExpectedVariableFieldOrTag);
      return;
    }
    if (DiagKind != -1) {
      Diag(AttrLoc, diag::err_alignas_attribute_wrong_decl_type)
        << &TmpAttr << DiagKind;
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
  llvm::APSInt Alignment;
  ExprResult ICE
    = VerifyIntegerConstantExpression(E, &Alignment,
        diag::err_aligned_attribute_argument_not_int,
        /*AllowFold*/ false);
  if (ICE.isInvalid())
    return;

  uint64_t AlignVal = Alignment.getZExtValue();

  // C++11 [dcl.align]p2:
  //   -- if the constant expression evaluates to zero, the alignment
  //      specifier shall have no effect
  // C11 6.7.5p6:
  //   An alignment specification of zero has no effect.
  if (!(TmpAttr.isAlignas() && !Alignment)) {
    if (!llvm::isPowerOf2_64(AlignVal)) {
      Diag(AttrLoc, diag::err_alignment_not_power_of_two)
        << E->getSourceRange();
      return;
    }
  }

  // Alignment calculations can wrap around if it's greater than 2**28.
  unsigned MaxValidAlignment =
      Context.getTargetInfo().getTriple().isOSBinFormatCOFF() ? 8192
                                                              : 268435456;
  if (AlignVal > MaxValidAlignment) {
    Diag(AttrLoc, diag::err_attribute_aligned_too_great) << MaxValidAlignment
                                                         << E->getSourceRange();
    return;
  }

  if (Context.getTargetInfo().isTLSSupported()) {
    unsigned MaxTLSAlign =
        Context.toCharUnitsFromBits(Context.getTargetInfo().getMaxTLSAlign())
            .getQuantity();
    auto *VD = dyn_cast<VarDecl>(D);
    if (MaxTLSAlign && AlignVal > MaxTLSAlign && VD &&
        VD->getTLSKind() != VarDecl::TLS_None) {
      Diag(VD->getLocation(), diag::err_tls_var_aligned_over_maximum)
          << (unsigned)AlignVal << VD << MaxTLSAlign;
      return;
    }
  }

  AlignedAttr *AA = ::new (Context) AlignedAttr(AttrRange, Context, true,
                                                ICE.get(), SpellingListIndex);
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

  QualType UnderlyingTy, DiagTy;
  if (ValueDecl *VD = dyn_cast<ValueDecl>(D)) {
    UnderlyingTy = DiagTy = VD->getType();
  } else {
    UnderlyingTy = DiagTy = Context.getTagDeclType(cast<TagDecl>(D));
    if (EnumDecl *ED = dyn_cast<EnumDecl>(D))
      UnderlyingTy = ED->getIntegerType();
  }
  if (DiagTy->isDependentType() || DiagTy->isIncompleteType())
    return;

  // C++11 [dcl.align]p5, C11 6.7.5/4:
  //   The combined effect of all alignment attributes in a declaration shall
  //   not specify an alignment that is less strict than the alignment that
  //   would otherwise be required for the entity being declared.
  AlignedAttr *AlignasAttr = nullptr;
  unsigned Align = 0;
  for (auto *I : D->specific_attrs<AlignedAttr>()) {
    if (I->isAlignmentDependent())
      return;
    if (I->isAlignas())
      AlignasAttr = I;
    Align = std::max(Align, I->getAlignment(Context));
  }

  if (AlignasAttr && Align) {
    CharUnits RequestedAlign = Context.toCharUnitsFromBits(Align);
    CharUnits NaturalAlign = Context.getTypeAlignInChars(UnderlyingTy);
    if (NaturalAlign > RequestedAlign)
      Diag(AlignasAttr->getLocation(), diag::err_alignas_underaligned)
        << DiagTy << (unsigned)NaturalAlign.getQuantity();
  }
}

bool Sema::checkMSInheritanceAttrOnDefinition(
    CXXRecordDecl *RD, SourceRange Range, bool BestCase,
    MSInheritanceAttr::Spelling SemanticSpelling) {
  assert(RD->hasDefinition() && "RD has no definition!");

  // We may not have seen base specifiers or any virtual methods yet.  We will
  // have to wait until the record is defined to catch any mismatches.
  if (!RD->getDefinition()->isCompleteDefinition())
    return false;

  // The unspecified model never matches what a definition could need.
  if (SemanticSpelling == MSInheritanceAttr::Keyword_unspecified_inheritance)
    return false;

  if (BestCase) {
    if (RD->calculateInheritanceModel() == SemanticSpelling)
      return false;
  } else {
    if (RD->calculateInheritanceModel() <= SemanticSpelling)
      return false;
  }

  Diag(Range.getBegin(), diag::err_mismatched_ms_inheritance)
      << 0 /*definition*/;
  Diag(RD->getDefinition()->getLocation(), diag::note_defined_here)
      << RD->getNameAsString();
  return true;
}

/// parseModeAttrArg - Parses attribute mode string and returns parsed type
/// attribute.
static void parseModeAttrArg(Sema &S, StringRef Str, unsigned &DestWidth,
                             bool &IntegerMode, bool &ComplexMode) {
  IntegerMode = true;
  ComplexMode = false;
  switch (Str.size()) {
  case 2:
    switch (Str[0]) {
    case 'Q':
      DestWidth = 8;
      break;
    case 'H':
      DestWidth = 16;
      break;
    case 'S':
      DestWidth = 32;
      break;
    case 'D':
      DestWidth = 64;
      break;
    case 'X':
      DestWidth = 96;
      break;
    case 'T':
      DestWidth = 128;
      break;
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
      DestWidth = S.Context.getTargetInfo().getRegisterWidth();
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
  if (!Attr.isArgIdent(0)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_type) << Attr.getName()
      << AANT_ArgumentIdentifier;
    return;
  }

  IdentifierInfo *Name = Attr.getArgAsIdent(0)->Ident;

  S.AddModeAttr(Attr.getRange(), D, Name, Attr.getAttributeSpellingListIndex());
}

void Sema::AddModeAttr(SourceRange AttrRange, Decl *D, IdentifierInfo *Name,
                       unsigned SpellingListIndex, bool InInstantiation) {
  StringRef Str = Name->getName();
  normalizeName(Str);
  SourceLocation AttrLoc = AttrRange.getBegin();

  unsigned DestWidth = 0;
  bool IntegerMode = true;
  bool ComplexMode = false;
  llvm::APInt VectorSize(64, 0);
  if (Str.size() >= 4 && Str[0] == 'V') {
    // Minimal length of vector mode is 4: 'V' + NUMBER(>=1) + TYPE(>=2).
    size_t StrSize = Str.size();
    size_t VectorStringLength = 0;
    while ((VectorStringLength + 1) < StrSize &&
           isdigit(Str[VectorStringLength + 1]))
      ++VectorStringLength;
    if (VectorStringLength &&
        !Str.substr(1, VectorStringLength).getAsInteger(10, VectorSize) &&
        VectorSize.isPowerOf2()) {
      parseModeAttrArg(*this, Str.substr(VectorStringLength + 1), DestWidth,
                       IntegerMode, ComplexMode);
      // Avoid duplicate warning from template instantiation.
      if (!InInstantiation)
        Diag(AttrLoc, diag::warn_vector_mode_deprecated);
    } else {
      VectorSize = 0;
    }
  }

  if (!VectorSize)
    parseModeAttrArg(*this, Str, DestWidth, IntegerMode, ComplexMode);

  // FIXME: Sync this with InitializePredefinedMacros; we need to match int8_t
  // and friends, at least with glibc.
  // FIXME: Make sure floating-point mappings are accurate
  // FIXME: Support XF and TF types
  if (!DestWidth) {
    Diag(AttrLoc, diag::err_machine_mode) << 0 /*Unknown*/ << Name;
    return;
  }

  QualType OldTy;
  if (TypedefNameDecl *TD = dyn_cast<TypedefNameDecl>(D))
    OldTy = TD->getUnderlyingType();
  else if (EnumDecl *ED = dyn_cast<EnumDecl>(D)) {
    // Something like 'typedef enum { X } __attribute__((mode(XX))) T;'.
    // Try to get type from enum declaration, default to int.
    OldTy = ED->getIntegerType();
    if (OldTy.isNull())
      OldTy = Context.IntTy;
  } else
    OldTy = cast<ValueDecl>(D)->getType();

  if (OldTy->isDependentType()) {
    D->addAttr(::new (Context)
               ModeAttr(AttrRange, Context, Name, SpellingListIndex));
    return;
  }

  // Base type can also be a vector type (see PR17453).
  // Distinguish between base type and base element type.
  QualType OldElemTy = OldTy;
  if (const VectorType *VT = OldTy->getAs<VectorType>())
    OldElemTy = VT->getElementType();

  // GCC allows 'mode' attribute on enumeration types (even incomplete), except
  // for vector modes. So, 'enum X __attribute__((mode(QI)));' forms a complete
  // type, 'enum { A } __attribute__((mode(V4SI)))' is rejected.
  if ((isa<EnumDecl>(D) || OldElemTy->getAs<EnumType>()) &&
      VectorSize.getBoolValue()) {
    Diag(AttrLoc, diag::err_enum_mode_vector_type) << Name << AttrRange;
    return;
  }
  bool IntegralOrAnyEnumType =
      OldElemTy->isIntegralOrEnumerationType() || OldElemTy->getAs<EnumType>();

  if (!OldElemTy->getAs<BuiltinType>() && !OldElemTy->isComplexType() &&
      !IntegralOrAnyEnumType)
    Diag(AttrLoc, diag::err_mode_not_primitive);
  else if (IntegerMode) {
    if (!IntegralOrAnyEnumType)
      Diag(AttrLoc, diag::err_mode_wrong_type);
  } else if (ComplexMode) {
    if (!OldElemTy->isComplexType())
      Diag(AttrLoc, diag::err_mode_wrong_type);
  } else {
    if (!OldElemTy->isFloatingType())
      Diag(AttrLoc, diag::err_mode_wrong_type);
  }

  QualType NewElemTy;

  if (IntegerMode)
    NewElemTy = Context.getIntTypeForBitwidth(DestWidth,
                                              OldElemTy->isSignedIntegerType());
  else
    NewElemTy = Context.getRealTypeForBitwidth(DestWidth);

  if (NewElemTy.isNull()) {
    Diag(AttrLoc, diag::err_machine_mode) << 1 /*Unsupported*/ << Name;
    return;
  }

  if (ComplexMode) {
    NewElemTy = Context.getComplexType(NewElemTy);
  }

  QualType NewTy = NewElemTy;
  if (VectorSize.getBoolValue()) {
    NewTy = Context.getVectorType(NewTy, VectorSize.getZExtValue(),
                                  VectorType::GenericVector);
  } else if (const VectorType *OldVT = OldTy->getAs<VectorType>()) {
    // Complex machine mode does not support base vector types.
    if (ComplexMode) {
      Diag(AttrLoc, diag::err_complex_mode_vector_type);
      return;
    }
    unsigned NumElements = Context.getTypeSize(OldElemTy) *
                           OldVT->getNumElements() /
                           Context.getTypeSize(NewElemTy);
    NewTy =
        Context.getVectorType(NewElemTy, NumElements, OldVT->getVectorKind());
  }

  if (NewTy.isNull()) {
    Diag(AttrLoc, diag::err_mode_wrong_type);
    return;
  }

  // Install the new type.
  if (TypedefNameDecl *TD = dyn_cast<TypedefNameDecl>(D))
    TD->setModedTypeSourceInfo(TD->getTypeSourceInfo(), NewTy);
  else if (EnumDecl *ED = dyn_cast<EnumDecl>(D))
    ED->setIntegerType(NewTy);
  else
    cast<ValueDecl>(D)->setType(NewTy);

  D->addAttr(::new (Context)
             ModeAttr(AttrRange, Context, Name, SpellingListIndex));
}

static void handleNoDebugAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  D->addAttr(::new (S.Context)
             NoDebugAttr(Attr.getRange(), S.Context,
                         Attr.getAttributeSpellingListIndex()));
}

AlwaysInlineAttr *Sema::mergeAlwaysInlineAttr(Decl *D, SourceRange Range,
                                              IdentifierInfo *Ident,
                                              unsigned AttrSpellingListIndex) {
  if (OptimizeNoneAttr *Optnone = D->getAttr<OptimizeNoneAttr>()) {
    Diag(Range.getBegin(), diag::warn_attribute_ignored) << Ident;
    Diag(Optnone->getLocation(), diag::note_conflicting_attribute);
    return nullptr;
  }

  if (D->hasAttr<AlwaysInlineAttr>())
    return nullptr;

  return ::new (Context) AlwaysInlineAttr(Range, Context,
                                          AttrSpellingListIndex);
}

CommonAttr *Sema::mergeCommonAttr(Decl *D, SourceRange Range,
                                  IdentifierInfo *Ident,
                                  unsigned AttrSpellingListIndex) {
  if (checkAttrMutualExclusion<InternalLinkageAttr>(*this, D, Range, Ident))
    return nullptr;

  return ::new (Context) CommonAttr(Range, Context, AttrSpellingListIndex);
}

InternalLinkageAttr *
Sema::mergeInternalLinkageAttr(Decl *D, SourceRange Range,
                               IdentifierInfo *Ident,
                               unsigned AttrSpellingListIndex) {
  if (auto VD = dyn_cast<VarDecl>(D)) {
    // Attribute applies to Var but not any subclass of it (like ParmVar,
    // ImplicitParm or VarTemplateSpecialization).
    if (VD->getKind() != Decl::Var) {
      Diag(Range.getBegin(), diag::warn_attribute_wrong_decl_type)
          << Ident << (getLangOpts().CPlusPlus ? ExpectedFunctionVariableOrClass
                                               : ExpectedVariableOrFunction);
      return nullptr;
    }
    // Attribute does not apply to non-static local variables.
    if (VD->hasLocalStorage()) {
      Diag(VD->getLocation(), diag::warn_internal_linkage_local_storage);
      return nullptr;
    }
  }

  if (checkAttrMutualExclusion<CommonAttr>(*this, D, Range, Ident))
    return nullptr;

  return ::new (Context)
      InternalLinkageAttr(Range, Context, AttrSpellingListIndex);
}

MinSizeAttr *Sema::mergeMinSizeAttr(Decl *D, SourceRange Range,
                                    unsigned AttrSpellingListIndex) {
  if (OptimizeNoneAttr *Optnone = D->getAttr<OptimizeNoneAttr>()) {
    Diag(Range.getBegin(), diag::warn_attribute_ignored) << "'minsize'";
    Diag(Optnone->getLocation(), diag::note_conflicting_attribute);
    return nullptr;
  }

  if (D->hasAttr<MinSizeAttr>())
    return nullptr;

  return ::new (Context) MinSizeAttr(Range, Context, AttrSpellingListIndex);
}

OptimizeNoneAttr *Sema::mergeOptimizeNoneAttr(Decl *D, SourceRange Range,
                                              unsigned AttrSpellingListIndex) {
  if (AlwaysInlineAttr *Inline = D->getAttr<AlwaysInlineAttr>()) {
    Diag(Inline->getLocation(), diag::warn_attribute_ignored) << Inline;
    Diag(Range.getBegin(), diag::note_conflicting_attribute);
    D->dropAttr<AlwaysInlineAttr>();
  }
  if (MinSizeAttr *MinSize = D->getAttr<MinSizeAttr>()) {
    Diag(MinSize->getLocation(), diag::warn_attribute_ignored) << MinSize;
    Diag(Range.getBegin(), diag::note_conflicting_attribute);
    D->dropAttr<MinSizeAttr>();
  }

  if (D->hasAttr<OptimizeNoneAttr>())
    return nullptr;

  return ::new (Context) OptimizeNoneAttr(Range, Context,
                                          AttrSpellingListIndex);
}

static void handleAlwaysInlineAttr(Sema &S, Decl *D,
                                   const AttributeList &Attr) {
  if (checkAttrMutualExclusion<NotTailCalledAttr>(S, D, Attr.getRange(),
                                                  Attr.getName()))
    return;

  if (AlwaysInlineAttr *Inline = S.mergeAlwaysInlineAttr(
          D, Attr.getRange(), Attr.getName(),
          Attr.getAttributeSpellingListIndex()))
    D->addAttr(Inline);
}

static void handleMinSizeAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (MinSizeAttr *MinSize = S.mergeMinSizeAttr(
          D, Attr.getRange(), Attr.getAttributeSpellingListIndex()))
    D->addAttr(MinSize);
}

static void handleOptimizeNoneAttr(Sema &S, Decl *D,
                                   const AttributeList &Attr) {
  if (OptimizeNoneAttr *Optnone = S.mergeOptimizeNoneAttr(
          D, Attr.getRange(), Attr.getAttributeSpellingListIndex()))
    D->addAttr(Optnone);
}

static void handleConstantAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (checkAttrMutualExclusion<CUDASharedAttr>(S, D, Attr.getRange(),
                                               Attr.getName()))
    return;
  auto *VD = cast<VarDecl>(D);
  if (!VD->hasGlobalStorage()) {
    S.Diag(Attr.getLoc(), diag::err_cuda_nonglobal_constant);
    return;
  }
  D->addAttr(::new (S.Context) CUDAConstantAttr(
      Attr.getRange(), S.Context, Attr.getAttributeSpellingListIndex()));
}

static void handleSharedAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (checkAttrMutualExclusion<CUDAConstantAttr>(S, D, Attr.getRange(),
                                                 Attr.getName()))
    return;
  auto *VD = cast<VarDecl>(D);
  // extern __shared__ is only allowed on arrays with no length (e.g.
  // "int x[]").
  if (VD->hasExternalStorage() && !isa<IncompleteArrayType>(VD->getType())) {
    S.Diag(Attr.getLoc(), diag::err_cuda_extern_shared) << VD;
    return;
  }
  if (S.getLangOpts().CUDA && VD->hasLocalStorage() &&
      S.CUDADiagIfHostCode(Attr.getLoc(), diag::err_cuda_host_shared)
          << S.CurrentCUDATarget())
    return;
  D->addAttr(::new (S.Context) CUDASharedAttr(
      Attr.getRange(), S.Context, Attr.getAttributeSpellingListIndex()));
}

static void handleGlobalAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (checkAttrMutualExclusion<CUDADeviceAttr>(S, D, Attr.getRange(),
                                               Attr.getName()) ||
      checkAttrMutualExclusion<CUDAHostAttr>(S, D, Attr.getRange(),
                                             Attr.getName())) {
    return;
  }
  FunctionDecl *FD = cast<FunctionDecl>(D);
  if (!FD->getReturnType()->isVoidType()) {
    SourceRange RTRange = FD->getReturnTypeSourceRange();
    S.Diag(FD->getTypeSpecStartLoc(), diag::err_kern_type_not_void_return)
        << FD->getType()
        << (RTRange.isValid() ? FixItHint::CreateReplacement(RTRange, "void")
                              : FixItHint());
    return;
  }
  if (const auto *Method = dyn_cast<CXXMethodDecl>(FD)) {
    if (Method->isInstance()) {
      S.Diag(Method->getLocStart(), diag::err_kern_is_nonstatic_method)
          << Method;
      return;
    }
    S.Diag(Method->getLocStart(), diag::warn_kern_is_method) << Method;
  }
  // Only warn for "inline" when compiling for host, to cut down on noise.
  if (FD->isInlineSpecified() && !S.getLangOpts().CUDAIsDevice)
    S.Diag(FD->getLocStart(), diag::warn_kern_is_inline) << FD;

  D->addAttr(::new (S.Context)
              CUDAGlobalAttr(Attr.getRange(), S.Context,
                             Attr.getAttributeSpellingListIndex()));
}

static void handleGNUInlineAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  FunctionDecl *Fn = cast<FunctionDecl>(D);
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

  // Diagnostic is emitted elsewhere: here we store the (valid) Attr
  // in the Decl node for syntactic reasoning, e.g., pretty-printing.
  CallingConv CC;
  if (S.CheckCallingConvAttr(Attr, CC, /*FD*/nullptr))
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
  case AttributeList::AT_SwiftCall:
    D->addAttr(::new (S.Context)
               SwiftCallAttr(Attr.getRange(), S.Context,
                             Attr.getAttributeSpellingListIndex()));
    return;
  case AttributeList::AT_VectorCall:
    D->addAttr(::new (S.Context)
               VectorCallAttr(Attr.getRange(), S.Context,
                              Attr.getAttributeSpellingListIndex()));
    return;
  case AttributeList::AT_MSABI:
    D->addAttr(::new (S.Context)
               MSABIAttr(Attr.getRange(), S.Context,
                         Attr.getAttributeSpellingListIndex()));
    return;
  case AttributeList::AT_SysVABI:
    D->addAttr(::new (S.Context)
               SysVABIAttr(Attr.getRange(), S.Context,
                           Attr.getAttributeSpellingListIndex()));
    return;
  case AttributeList::AT_RegCall:
    D->addAttr(::new (S.Context) RegCallAttr(
        Attr.getRange(), S.Context, Attr.getAttributeSpellingListIndex()));
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
  case AttributeList::AT_IntelOclBicc:
    D->addAttr(::new (S.Context)
               IntelOclBiccAttr(Attr.getRange(), S.Context,
                                Attr.getAttributeSpellingListIndex()));
    return;
  case AttributeList::AT_PreserveMost:
    D->addAttr(::new (S.Context) PreserveMostAttr(
        Attr.getRange(), S.Context, Attr.getAttributeSpellingListIndex()));
    return;
  case AttributeList::AT_PreserveAll:
    D->addAttr(::new (S.Context) PreserveAllAttr(
        Attr.getRange(), S.Context, Attr.getAttributeSpellingListIndex()));
    return;
  default:
    llvm_unreachable("unexpected attribute kind");
  }
}

static void handleSuppressAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (!checkAttributeAtLeastNumArgs(S, Attr, 1))
    return;

  std::vector<StringRef> DiagnosticIdentifiers;
  for (unsigned I = 0, E = Attr.getNumArgs(); I != E; ++I) {
    StringRef RuleName;

    if (!S.checkStringLiteralArgumentAttr(Attr, I, RuleName, nullptr))
      return;

    // FIXME: Warn if the rule name is unknown. This is tricky because only
    // clang-tidy knows about available rules.
    DiagnosticIdentifiers.push_back(RuleName);
  }
  D->addAttr(::new (S.Context) SuppressAttr(
      Attr.getRange(), S.Context, DiagnosticIdentifiers.data(),
      DiagnosticIdentifiers.size(), Attr.getAttributeSpellingListIndex()));
}

bool Sema::CheckCallingConvAttr(const AttributeList &attr, CallingConv &CC, 
                                const FunctionDecl *FD) {
  if (attr.isInvalid())
    return true;

  if (attr.hasProcessingCache()) {
    CC = (CallingConv) attr.getProcessingCache();
    return false;
  }

  unsigned ReqArgs = attr.getKind() == AttributeList::AT_Pcs ? 1 : 0;
  if (!checkAttributeNumArgs(*this, attr, ReqArgs)) {
    attr.setInvalid();
    return true;
  }

  // TODO: diagnose uses of these conventions on the wrong target.
  switch (attr.getKind()) {
  case AttributeList::AT_CDecl: CC = CC_C; break;
  case AttributeList::AT_FastCall: CC = CC_X86FastCall; break;
  case AttributeList::AT_StdCall: CC = CC_X86StdCall; break;
  case AttributeList::AT_ThisCall: CC = CC_X86ThisCall; break;
  case AttributeList::AT_Pascal: CC = CC_X86Pascal; break;
  case AttributeList::AT_SwiftCall: CC = CC_Swift; break;
  case AttributeList::AT_VectorCall: CC = CC_X86VectorCall; break;
  case AttributeList::AT_RegCall: CC = CC_X86RegCall; break;
  case AttributeList::AT_MSABI:
    CC = Context.getTargetInfo().getTriple().isOSWindows() ? CC_C :
                                                             CC_X86_64Win64;
    break;
  case AttributeList::AT_SysVABI:
    CC = Context.getTargetInfo().getTriple().isOSWindows() ? CC_X86_64SysV :
                                                             CC_C;
    break;
  case AttributeList::AT_Pcs: {
    StringRef StrRef;
    if (!checkStringLiteralArgumentAttr(attr, 0, StrRef)) {
      attr.setInvalid();
      return true;
    }
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
  case AttributeList::AT_IntelOclBicc: CC = CC_IntelOclBicc; break;
  case AttributeList::AT_PreserveMost: CC = CC_PreserveMost; break;
  case AttributeList::AT_PreserveAll: CC = CC_PreserveAll; break;
  default: llvm_unreachable("unexpected attribute kind");
  }

  const TargetInfo &TI = Context.getTargetInfo();
  TargetInfo::CallingConvCheckResult A = TI.checkCallingConvention(CC);
  if (A != TargetInfo::CCCR_OK) {
    if (A == TargetInfo::CCCR_Warning)
      Diag(attr.getLoc(), diag::warn_cconv_ignored) << attr.getName();

    // This convention is not valid for the target. Use the default function or
    // method calling convention.
    bool IsCXXMethod = false, IsVariadic = false;
    if (FD) {
      IsCXXMethod = FD->isCXXInstanceMember();
      IsVariadic = FD->isVariadic();
    }
    CC = Context.getDefaultCallingConvention(IsVariadic, IsCXXMethod);
  }

  attr.setProcessingCache((unsigned) CC);
  return false;
}

/// Pointer-like types in the default address space.
static bool isValidSwiftContextType(QualType type) {
  if (!type->hasPointerRepresentation())
    return type->isDependentType();
  return type->getPointeeType().getAddressSpace() == 0;
}

/// Pointers and references in the default address space.
static bool isValidSwiftIndirectResultType(QualType type) {
  if (auto ptrType = type->getAs<PointerType>()) {
    type = ptrType->getPointeeType();
  } else if (auto refType = type->getAs<ReferenceType>()) {
    type = refType->getPointeeType();
  } else {
    return type->isDependentType();
  }
  return type.getAddressSpace() == 0;
}

/// Pointers and references to pointers in the default address space.
static bool isValidSwiftErrorResultType(QualType type) {
  if (auto ptrType = type->getAs<PointerType>()) {
    type = ptrType->getPointeeType();
  } else if (auto refType = type->getAs<ReferenceType>()) {
    type = refType->getPointeeType();
  } else {
    return type->isDependentType();
  }
  if (!type.getQualifiers().empty())
    return false;
  return isValidSwiftContextType(type);
}

static void handleParameterABIAttr(Sema &S, Decl *D, const AttributeList &attr,
                                   ParameterABI abi) {
  S.AddParameterABIAttr(attr.getRange(), D, abi,
                        attr.getAttributeSpellingListIndex());
}

void Sema::AddParameterABIAttr(SourceRange range, Decl *D, ParameterABI abi,
                               unsigned spellingIndex) {

  QualType type = cast<ParmVarDecl>(D)->getType();

  if (auto existingAttr = D->getAttr<ParameterABIAttr>()) {
    if (existingAttr->getABI() != abi) {
      Diag(range.getBegin(), diag::err_attributes_are_not_compatible)
        << getParameterABISpelling(abi) << existingAttr;
      Diag(existingAttr->getLocation(), diag::note_conflicting_attribute);
      return;
    }
  }

  switch (abi) {
  case ParameterABI::Ordinary:
    llvm_unreachable("explicit attribute for ordinary parameter ABI?");

  case ParameterABI::SwiftContext:
    if (!isValidSwiftContextType(type)) {
      Diag(range.getBegin(), diag::err_swift_abi_parameter_wrong_type)
        << getParameterABISpelling(abi)
        << /*pointer to pointer */ 0 << type;
    }
    D->addAttr(::new (Context)
               SwiftContextAttr(range, Context, spellingIndex));
    return;

  case ParameterABI::SwiftErrorResult:
    if (!isValidSwiftErrorResultType(type)) {
      Diag(range.getBegin(), diag::err_swift_abi_parameter_wrong_type)
        << getParameterABISpelling(abi)
        << /*pointer to pointer */ 1 << type;
    }
    D->addAttr(::new (Context)
               SwiftErrorResultAttr(range, Context, spellingIndex));
    return;

  case ParameterABI::SwiftIndirectResult:
    if (!isValidSwiftIndirectResultType(type)) {
      Diag(range.getBegin(), diag::err_swift_abi_parameter_wrong_type)
        << getParameterABISpelling(abi)
        << /*pointer*/ 0 << type;
    }
    D->addAttr(::new (Context)
               SwiftIndirectResultAttr(range, Context, spellingIndex));
    return;
  }
  llvm_unreachable("bad parameter ABI attribute");
}

/// Checks a regparm attribute, returning true if it is ill-formed and
/// otherwise setting numParams to the appropriate value.
bool Sema::CheckRegparmAttr(const AttributeList &Attr, unsigned &numParams) {
  if (Attr.isInvalid())
    return true;

  if (!checkAttributeNumArgs(*this, Attr, 1)) {
    Attr.setInvalid();
    return true;
  }

  uint32_t NP;
  Expr *NumParamsExpr = Attr.getArgAsExpr(0);
  if (!checkUInt32Argument(*this, Attr, NumParamsExpr, NP)) {
    Attr.setInvalid();
    return true;
  }

  if (Context.getTargetInfo().getRegParmMax() == 0) {
    Diag(Attr.getLoc(), diag::err_attribute_regparm_wrong_platform)
      << NumParamsExpr->getSourceRange();
    Attr.setInvalid();
    return true;
  }

  numParams = NP;
  if (numParams > Context.getTargetInfo().getRegParmMax()) {
    Diag(Attr.getLoc(), diag::err_attribute_regparm_invalid_number)
      << Context.getTargetInfo().getRegParmMax() << NumParamsExpr->getSourceRange();
    Attr.setInvalid();
    return true;
  }

  return false;
}

// Checks whether an argument of launch_bounds attribute is
// acceptable, performs implicit conversion to Rvalue, and returns
// non-nullptr Expr result on success. Otherwise, it returns nullptr
// and may output an error.
static Expr *makeLaunchBoundsArgExpr(Sema &S, Expr *E,
                                     const CUDALaunchBoundsAttr &Attr,
                                     const unsigned Idx) {
  if (S.DiagnoseUnexpandedParameterPack(E))
    return nullptr;

  // Accept template arguments for now as they depend on something else.
  // We'll get to check them when they eventually get instantiated.
  if (E->isValueDependent())
    return E;

  llvm::APSInt I(64);
  if (!E->isIntegerConstantExpr(I, S.Context)) {
    S.Diag(E->getExprLoc(), diag::err_attribute_argument_n_type)
        << &Attr << Idx << AANT_ArgumentIntegerConstant << E->getSourceRange();
    return nullptr;
  }
  // Make sure we can fit it in 32 bits.
  if (!I.isIntN(32)) {
    S.Diag(E->getExprLoc(), diag::err_ice_too_large) << I.toString(10, false)
                                                     << 32 << /* Unsigned */ 1;
    return nullptr;
  }
  if (I < 0)
    S.Diag(E->getExprLoc(), diag::warn_attribute_argument_n_negative)
        << &Attr << Idx << E->getSourceRange();

  // We may need to perform implicit conversion of the argument.
  InitializedEntity Entity = InitializedEntity::InitializeParameter(
      S.Context, S.Context.getConstType(S.Context.IntTy), /*consume*/ false);
  ExprResult ValArg = S.PerformCopyInitialization(Entity, SourceLocation(), E);
  assert(!ValArg.isInvalid() &&
         "Unexpected PerformCopyInitialization() failure.");

  return ValArg.getAs<Expr>();
}

void Sema::AddLaunchBoundsAttr(SourceRange AttrRange, Decl *D, Expr *MaxThreads,
                               Expr *MinBlocks, unsigned SpellingListIndex) {
  CUDALaunchBoundsAttr TmpAttr(AttrRange, Context, MaxThreads, MinBlocks,
                               SpellingListIndex);
  MaxThreads = makeLaunchBoundsArgExpr(*this, MaxThreads, TmpAttr, 0);
  if (MaxThreads == nullptr)
    return;

  if (MinBlocks) {
    MinBlocks = makeLaunchBoundsArgExpr(*this, MinBlocks, TmpAttr, 1);
    if (MinBlocks == nullptr)
      return;
  }

  D->addAttr(::new (Context) CUDALaunchBoundsAttr(
      AttrRange, Context, MaxThreads, MinBlocks, SpellingListIndex));
}

static void handleLaunchBoundsAttr(Sema &S, Decl *D,
                                   const AttributeList &Attr) {
  if (!checkAttributeAtLeastNumArgs(S, Attr, 1) ||
      !checkAttributeAtMostNumArgs(S, Attr, 2))
    return;

  S.AddLaunchBoundsAttr(Attr.getRange(), D, Attr.getArgAsExpr(0),
                        Attr.getNumArgs() > 1 ? Attr.getArgAsExpr(1) : nullptr,
                        Attr.getAttributeSpellingListIndex());
}

static void handleArgumentWithTypeTagAttr(Sema &S, Decl *D,
                                          const AttributeList &Attr) {
  if (!Attr.isArgIdent(0)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_type)
      << Attr.getName() << /* arg num = */ 1 << AANT_ArgumentIdentifier;
    return;
  }
  
  if (!checkAttributeNumArgs(S, Attr, 3))
    return;

  IdentifierInfo *ArgumentKind = Attr.getArgAsIdent(0)->Ident;

  if (!isFunctionOrMethod(D) || !hasFunctionProto(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrMethod;
    return;
  }

  uint64_t ArgumentIdx;
  if (!checkFunctionOrMethodParameterIndex(S, D, Attr, 2, Attr.getArgAsExpr(1),
                                           ArgumentIdx))
    return;

  uint64_t TypeTagIdx;
  if (!checkFunctionOrMethodParameterIndex(S, D, Attr, 3, Attr.getArgAsExpr(2),
                                           TypeTagIdx))
    return;

  bool IsPointer = (Attr.getName()->getName() == "pointer_with_type_tag");
  if (IsPointer) {
    // Ensure that buffer has a pointer type.
    QualType BufferTy = getFunctionOrMethodParamType(D, ArgumentIdx);
    if (!BufferTy->isPointerType()) {
      S.Diag(Attr.getLoc(), diag::err_attribute_pointers_only)
        << Attr.getName() << 0;
    }
  }

  D->addAttr(::new (S.Context)
             ArgumentWithTypeTagAttr(Attr.getRange(), S.Context, ArgumentKind,
                                     ArgumentIdx, TypeTagIdx, IsPointer,
                                     Attr.getAttributeSpellingListIndex()));
}

static void handleTypeTagForDatatypeAttr(Sema &S, Decl *D,
                                         const AttributeList &Attr) {
  if (!Attr.isArgIdent(0)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_type)
      << Attr.getName() << 1 << AANT_ArgumentIdentifier;
    return;
  }
  
  if (!checkAttributeNumArgs(S, Attr, 1))
    return;

  if (!isa<VarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedVariable;
    return;
  }

  IdentifierInfo *PointerKind = Attr.getArgAsIdent(0)->Ident;
  TypeSourceInfo *MatchingCTypeLoc = nullptr;
  S.GetTypeFromParser(Attr.getMatchingCType(), &MatchingCTypeLoc);
  assert(MatchingCTypeLoc && "no type source info for attribute argument");

  D->addAttr(::new (S.Context)
             TypeTagForDatatypeAttr(Attr.getRange(), S.Context, PointerKind,
                                    MatchingCTypeLoc,
                                    Attr.getLayoutCompatible(),
                                    Attr.getMustBeNull(),
                                    Attr.getAttributeSpellingListIndex()));
}

static void handleXRayLogArgsAttr(Sema &S, Decl *D,
                                  const AttributeList &Attr) {
  uint64_t ArgCount;

  if (!checkFunctionOrMethodParameterIndex(S, D, Attr, 1, Attr.getArgAsExpr(0),
                                           ArgCount,
                                           true /* AllowImplicitThis*/))
    return;

  // ArgCount isn't a parameter index [0;n), it's a count [1;n] - hence + 1.
  D->addAttr(::new (S.Context)
                 XRayLogArgsAttr(Attr.getRange(), S.Context, ++ArgCount,
                                 Attr.getAttributeSpellingListIndex()));
}

//===----------------------------------------------------------------------===//
// Checker-specific attribute handlers.
//===----------------------------------------------------------------------===//

static bool isValidSubjectOfNSReturnsRetainedAttribute(QualType type) {
  return type->isDependentType() ||
         type->isObjCRetainableType();
}

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
  S.AddNSConsumedAttr(Attr.getRange(), D, Attr.getAttributeSpellingListIndex(),
                      Attr.getKind() == AttributeList::AT_NSConsumed,
                      /*template instantiation*/ false);
}

void Sema::AddNSConsumedAttr(SourceRange attrRange, Decl *D,
                             unsigned spellingIndex, bool isNSConsumed,
                             bool isTemplateInstantiation) {
  ParmVarDecl *param = cast<ParmVarDecl>(D);
  bool typeOK;

  if (isNSConsumed) {
    typeOK = isValidSubjectOfNSAttribute(*this, param->getType());
  } else {
    typeOK = isValidSubjectOfCFAttribute(*this, param->getType());
  }

  if (!typeOK) {
    // These attributes are normally just advisory, but in ARC, ns_consumed
    // is significant.  Allow non-dependent code to contain inappropriate
    // attributes even in ARC, but require template instantiations to be
    // set up correctly.
    Diag(D->getLocStart(),
         (isTemplateInstantiation && isNSConsumed &&
            getLangOpts().ObjCAutoRefCount
          ? diag::err_ns_attribute_wrong_parameter_type
          : diag::warn_ns_attribute_wrong_parameter_type))
      << attrRange
      << (isNSConsumed ? "ns_consumed" : "cf_consumed")
      << (isNSConsumed ? /*objc pointers*/ 0 : /*cf pointers*/ 1);
    return;
  }

  if (isNSConsumed)
    param->addAttr(::new (Context)
                   NSConsumedAttr(attrRange, Context, spellingIndex));
  else
    param->addAttr(::new (Context)
                   CFConsumedAttr(attrRange, Context, spellingIndex));
}

static void handleNSReturnsRetainedAttr(Sema &S, Decl *D,
                                        const AttributeList &Attr) {
  QualType returnType;

  if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D))
    returnType = MD->getReturnType();
  else if (S.getLangOpts().ObjCAutoRefCount && hasDeclarator(D) &&
           (Attr.getKind() == AttributeList::AT_NSReturnsRetained))
    return; // ignore: was handled as a type attribute
  else if (ObjCPropertyDecl *PD = dyn_cast<ObjCPropertyDecl>(D))
    returnType = PD->getType();
  else if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
    returnType = FD->getReturnType();
  else if (auto *Param = dyn_cast<ParmVarDecl>(D)) {
    returnType = Param->getType()->getPointeeType();
    if (returnType.isNull()) {
      S.Diag(D->getLocStart(), diag::warn_ns_attribute_wrong_parameter_type)
          << Attr.getName() << /*pointer-to-CF*/2
          << Attr.getRange();
      return;
    }
  } else {
    AttributeDeclKind ExpectedDeclKind;
    switch (Attr.getKind()) {
    default: llvm_unreachable("invalid ownership attribute");
    case AttributeList::AT_NSReturnsRetained:
    case AttributeList::AT_NSReturnsAutoreleased:
    case AttributeList::AT_NSReturnsNotRetained:
      ExpectedDeclKind = ExpectedFunctionOrMethod;
      break;

    case AttributeList::AT_CFReturnsRetained:
    case AttributeList::AT_CFReturnsNotRetained:
      ExpectedDeclKind = ExpectedFunctionMethodOrParameter;
      break;
    }
    S.Diag(D->getLocStart(), diag::warn_attribute_wrong_decl_type)
        << Attr.getRange() << Attr.getName() << ExpectedDeclKind;
    return;
  }

  bool typeOK;
  bool cf;
  switch (Attr.getKind()) {
  default: llvm_unreachable("invalid ownership attribute");
  case AttributeList::AT_NSReturnsRetained:
    typeOK = isValidSubjectOfNSReturnsRetainedAttribute(returnType);
    cf = false;
    break;
      
  case AttributeList::AT_NSReturnsAutoreleased:
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
    if (isa<ParmVarDecl>(D)) {
      S.Diag(D->getLocStart(), diag::warn_ns_attribute_wrong_parameter_type)
          << Attr.getName() << /*pointer-to-CF*/2
          << Attr.getRange();
    } else {
      // Needs to be kept in sync with warn_ns_attribute_wrong_return_type.
      enum : unsigned {
        Function,
        Method,
        Property
      } SubjectKind = Function;
      if (isa<ObjCMethodDecl>(D))
        SubjectKind = Method;
      else if (isa<ObjCPropertyDecl>(D))
        SubjectKind = Property;
      S.Diag(D->getLocStart(), diag::warn_ns_attribute_wrong_return_type)
          << Attr.getName() << SubjectKind << cf
          << Attr.getRange();
    }
    return;
  }

  switch (Attr.getKind()) {
    default:
      llvm_unreachable("invalid ownership attribute");
    case AttributeList::AT_NSReturnsAutoreleased:
      D->addAttr(::new (S.Context) NSReturnsAutoreleasedAttr(
          Attr.getRange(), S.Context, Attr.getAttributeSpellingListIndex()));
      return;
    case AttributeList::AT_CFReturnsNotRetained:
      D->addAttr(::new (S.Context) CFReturnsNotRetainedAttr(
          Attr.getRange(), S.Context, Attr.getAttributeSpellingListIndex()));
      return;
    case AttributeList::AT_NSReturnsNotRetained:
      D->addAttr(::new (S.Context) NSReturnsNotRetainedAttr(
          Attr.getRange(), S.Context, Attr.getAttributeSpellingListIndex()));
      return;
    case AttributeList::AT_CFReturnsRetained:
      D->addAttr(::new (S.Context) CFReturnsRetainedAttr(
          Attr.getRange(), S.Context, Attr.getAttributeSpellingListIndex()));
      return;
    case AttributeList::AT_NSReturnsRetained:
      D->addAttr(::new (S.Context) NSReturnsRetainedAttr(
          Attr.getRange(), S.Context, Attr.getAttributeSpellingListIndex()));
      return;
  };
}

static void handleObjCReturnsInnerPointerAttr(Sema &S, Decl *D,
                                              const AttributeList &attr) {
  const int EP_ObjCMethod = 1;
  const int EP_ObjCProperty = 2;
  
  SourceLocation loc = attr.getLoc();
  QualType resultType;
  if (isa<ObjCMethodDecl>(D))
    resultType = cast<ObjCMethodDecl>(D)->getReturnType();
  else
    resultType = cast<ObjCPropertyDecl>(D)->getType();

  if (!resultType->isReferenceType() &&
      (!resultType->isPointerType() || resultType->isObjCRetainableType())) {
    S.Diag(D->getLocStart(), diag::warn_ns_attribute_wrong_return_type)
      << SourceRange(loc)
    << attr.getName()
    << (isa<ObjCMethodDecl>(D) ? EP_ObjCMethod : EP_ObjCProperty)
    << /*non-retainable pointer*/ 2;

    // Drop the attribute.
    return;
  }

  D->addAttr(::new (S.Context) ObjCReturnsInnerPointerAttr(
      attr.getRange(), S.Context, attr.getAttributeSpellingListIndex()));
}

static void handleObjCRequiresSuperAttr(Sema &S, Decl *D,
                                        const AttributeList &attr) {
  ObjCMethodDecl *method = cast<ObjCMethodDecl>(D);
  
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

static void handleCFAuditedTransferAttr(Sema &S, Decl *D,
                                        const AttributeList &Attr) {
  if (checkAttrMutualExclusion<CFUnknownTransferAttr>(S, D, Attr.getRange(),
                                                      Attr.getName()))
    return;

  D->addAttr(::new (S.Context)
             CFAuditedTransferAttr(Attr.getRange(), S.Context,
                                   Attr.getAttributeSpellingListIndex()));
}

static void handleCFUnknownTransferAttr(Sema &S, Decl *D,
                                        const AttributeList &Attr) {
  if (checkAttrMutualExclusion<CFAuditedTransferAttr>(S, D, Attr.getRange(),
                                                      Attr.getName()))
    return;

  D->addAttr(::new (S.Context)
             CFUnknownTransferAttr(Attr.getRange(), S.Context,
             Attr.getAttributeSpellingListIndex()));
}

static void handleObjCBridgeAttr(Sema &S, Scope *Sc, Decl *D,
                                const AttributeList &Attr) {
  IdentifierLoc * Parm = Attr.isArgIdent(0) ? Attr.getArgAsIdent(0) : nullptr;

  if (!Parm) {
    S.Diag(D->getLocStart(), diag::err_objc_attr_not_id) << Attr.getName() << 0;
    return;
  }

  // Typedefs only allow objc_bridge(id) and have some additional checking.
  if (auto TD = dyn_cast<TypedefNameDecl>(D)) {
    if (!Parm->Ident->isStr("id")) {
      S.Diag(Attr.getLoc(), diag::err_objc_attr_typedef_not_id)
        << Attr.getName();
      return;
    }

    // Only allow 'cv void *'.
    QualType T = TD->getUnderlyingType();
    if (!T->isVoidPointerType()) {
      S.Diag(Attr.getLoc(), diag::err_objc_attr_typedef_not_void_pointer);
      return;
    }
  }
  
  D->addAttr(::new (S.Context)
             ObjCBridgeAttr(Attr.getRange(), S.Context, Parm->Ident,
                           Attr.getAttributeSpellingListIndex()));
}

static void handleObjCBridgeMutableAttr(Sema &S, Scope *Sc, Decl *D,
                                        const AttributeList &Attr) {
  IdentifierLoc * Parm = Attr.isArgIdent(0) ? Attr.getArgAsIdent(0) : nullptr;

  if (!Parm) {
    S.Diag(D->getLocStart(), diag::err_objc_attr_not_id) << Attr.getName() << 0;
    return;
  }
  
  D->addAttr(::new (S.Context)
             ObjCBridgeMutableAttr(Attr.getRange(), S.Context, Parm->Ident,
                            Attr.getAttributeSpellingListIndex()));
}

static void handleObjCBridgeRelatedAttr(Sema &S, Scope *Sc, Decl *D,
                                 const AttributeList &Attr) {
  IdentifierInfo *RelatedClass =
    Attr.isArgIdent(0) ? Attr.getArgAsIdent(0)->Ident : nullptr;
  if (!RelatedClass) {
    S.Diag(D->getLocStart(), diag::err_objc_attr_not_id) << Attr.getName() << 0;
    return;
  }
  IdentifierInfo *ClassMethod =
    Attr.getArgAsIdent(1) ? Attr.getArgAsIdent(1)->Ident : nullptr;
  IdentifierInfo *InstanceMethod =
    Attr.getArgAsIdent(2) ? Attr.getArgAsIdent(2)->Ident : nullptr;
  D->addAttr(::new (S.Context)
             ObjCBridgeRelatedAttr(Attr.getRange(), S.Context, RelatedClass,
                                   ClassMethod, InstanceMethod,
                                   Attr.getAttributeSpellingListIndex()));
}

static void handleObjCDesignatedInitializer(Sema &S, Decl *D,
                                            const AttributeList &Attr) {
  ObjCInterfaceDecl *IFace;
  if (ObjCCategoryDecl *CatDecl =
          dyn_cast<ObjCCategoryDecl>(D->getDeclContext()))
    IFace = CatDecl->getClassInterface();
  else
    IFace = cast<ObjCInterfaceDecl>(D->getDeclContext());

  if (!IFace)
    return;

  IFace->setHasDesignatedInitializers();
  D->addAttr(::new (S.Context)
                  ObjCDesignatedInitializerAttr(Attr.getRange(), S.Context,
                                         Attr.getAttributeSpellingListIndex()));
}

static void handleObjCRuntimeName(Sema &S, Decl *D,
                                  const AttributeList &Attr) {
  StringRef MetaDataName;
  if (!S.checkStringLiteralArgumentAttr(Attr, 0, MetaDataName))
    return;
  D->addAttr(::new (S.Context)
             ObjCRuntimeNameAttr(Attr.getRange(), S.Context,
                                 MetaDataName,
                                 Attr.getAttributeSpellingListIndex()));
}

// When a user wants to use objc_boxable with a union or struct
// but they don't have access to the declaration (legacy/third-party code)
// then they can 'enable' this feature with a typedef:
// typedef struct __attribute((objc_boxable)) legacy_struct legacy_struct;
static void handleObjCBoxable(Sema &S, Decl *D, const AttributeList &Attr) {
  bool notify = false;

  RecordDecl *RD = dyn_cast<RecordDecl>(D);
  if (RD && RD->getDefinition()) {
    RD = RD->getDefinition();
    notify = true;
  }

  if (RD) {
    ObjCBoxableAttr *BoxableAttr = ::new (S.Context)
                          ObjCBoxableAttr(Attr.getRange(), S.Context,
                                          Attr.getAttributeSpellingListIndex());
    RD->addAttr(BoxableAttr);
    if (notify) {
      // we need to notify ASTReader/ASTWriter about
      // modification of existing declaration
      if (ASTMutationListener *L = S.getASTMutationListener())
        L->AddedAttributeToRecord(BoxableAttr, RD);
    }
  }
}

static void handleObjCOwnershipAttr(Sema &S, Decl *D,
                                    const AttributeList &Attr) {
  if (hasDeclarator(D)) return;

  S.Diag(D->getLocStart(), diag::err_attribute_wrong_decl_type)
    << Attr.getRange() << Attr.getName() << ExpectedVariable;
}

static void handleObjCPreciseLifetimeAttr(Sema &S, Decl *D,
                                          const AttributeList &Attr) {
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

UuidAttr *Sema::mergeUuidAttr(Decl *D, SourceRange Range,
                              unsigned AttrSpellingListIndex, StringRef Uuid) {
  if (const auto *UA = D->getAttr<UuidAttr>()) {
    if (UA->getGuid().equals_lower(Uuid))
      return nullptr;
    Diag(UA->getLocation(), diag::err_mismatched_uuid);
    Diag(Range.getBegin(), diag::note_previous_uuid);
    D->dropAttr<UuidAttr>();
  }

  return ::new (Context) UuidAttr(Range, Context, Uuid, AttrSpellingListIndex);
}

static void handleUuidAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (!S.LangOpts.CPlusPlus) {
    S.Diag(Attr.getLoc(), diag::err_attribute_not_supported_in_lang)
      << Attr.getName() << AttributeLangSupport::C;
    return;
  }

  StringRef StrRef;
  SourceLocation LiteralLoc;
  if (!S.checkStringLiteralArgumentAttr(Attr, 0, StrRef, &LiteralLoc))
    return;

  // GUID format is "XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX" or
  // "{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}", normalize to the former.
  if (StrRef.size() == 38 && StrRef.front() == '{' && StrRef.back() == '}')
    StrRef = StrRef.drop_front().drop_back();

  // Validate GUID length.
  if (StrRef.size() != 36) {
    S.Diag(LiteralLoc, diag::err_attribute_uuid_malformed_guid);
    return;
  }

  for (unsigned i = 0; i < 36; ++i) {
    if (i == 8 || i == 13 || i == 18 || i == 23) {
      if (StrRef[i] != '-') {
        S.Diag(LiteralLoc, diag::err_attribute_uuid_malformed_guid);
        return;
      }
    } else if (!isHexDigit(StrRef[i])) {
      S.Diag(LiteralLoc, diag::err_attribute_uuid_malformed_guid);
      return;
    }
  }

  // FIXME: It'd be nice to also emit a fixit removing uuid(...) (and, if it's
  // the only thing in the [] list, the [] too), and add an insertion of
  // __declspec(uuid(...)).  But sadly, neither the SourceLocs of the commas
  // separating attributes nor of the [ and the ] are in the AST.
  // Cf "SourceLocations of attribute list delimiters - [[ ... , ... ]] etc"
  // on cfe-dev.
  if (Attr.isMicrosoftAttribute()) // Check for [uuid(...)] spelling.
    S.Diag(Attr.getLoc(), diag::warn_atl_uuid_deprecated);

  UuidAttr *UA = S.mergeUuidAttr(D, Attr.getRange(),
                                 Attr.getAttributeSpellingListIndex(), StrRef);
  if (UA)
    D->addAttr(UA);
}

static void handleMSInheritanceAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (!S.LangOpts.CPlusPlus) {
    S.Diag(Attr.getLoc(), diag::err_attribute_not_supported_in_lang)
      << Attr.getName() << AttributeLangSupport::C;
    return;
  }
  MSInheritanceAttr *IA = S.mergeMSInheritanceAttr(
      D, Attr.getRange(), /*BestCase=*/true,
      Attr.getAttributeSpellingListIndex(),
      (MSInheritanceAttr::Spelling)Attr.getSemanticSpelling());
  if (IA) {
    D->addAttr(IA);
    S.Consumer.AssignInheritanceModel(cast<CXXRecordDecl>(D));
  }
}

static void handleDeclspecThreadAttr(Sema &S, Decl *D,
                                     const AttributeList &Attr) {
  VarDecl *VD = cast<VarDecl>(D);
  if (!S.Context.getTargetInfo().isTLSSupported()) {
    S.Diag(Attr.getLoc(), diag::err_thread_unsupported);
    return;
  }
  if (VD->getTSCSpec() != TSCS_unspecified) {
    S.Diag(Attr.getLoc(), diag::err_declspec_thread_on_thread_variable);
    return;
  }
  if (VD->hasLocalStorage()) {
    S.Diag(Attr.getLoc(), diag::err_thread_non_global) << "__declspec(thread)";
    return;
  }
  VD->addAttr(::new (S.Context) ThreadAttr(
      Attr.getRange(), S.Context, Attr.getAttributeSpellingListIndex()));
}

static void handleAbiTagAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  SmallVector<StringRef, 4> Tags;
  for (unsigned I = 0, E = Attr.getNumArgs(); I != E; ++I) {
    StringRef Tag;
    if (!S.checkStringLiteralArgumentAttr(Attr, I, Tag))
      return;
    Tags.push_back(Tag);
  }

  if (const auto *NS = dyn_cast<NamespaceDecl>(D)) {
    if (!NS->isInline()) {
      S.Diag(Attr.getLoc(), diag::warn_attr_abi_tag_namespace) << 0;
      return;
    }
    if (NS->isAnonymousNamespace()) {
      S.Diag(Attr.getLoc(), diag::warn_attr_abi_tag_namespace) << 1;
      return;
    }
    if (Attr.getNumArgs() == 0)
      Tags.push_back(NS->getName());
  } else if (!checkAttributeAtLeastNumArgs(S, Attr, 1))
    return;

  // Store tags sorted and without duplicates.
  std::sort(Tags.begin(), Tags.end());
  Tags.erase(std::unique(Tags.begin(), Tags.end()), Tags.end());

  D->addAttr(::new (S.Context)
             AbiTagAttr(Attr.getRange(), S.Context, Tags.data(), Tags.size(),
                        Attr.getAttributeSpellingListIndex()));
}

static void handleARMInterruptAttr(Sema &S, Decl *D,
                                   const AttributeList &Attr) {
  // Check the attribute arguments.
  if (Attr.getNumArgs() > 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_too_many_arguments)
      << Attr.getName() << 1;
    return;
  }

  StringRef Str;
  SourceLocation ArgLoc;

  if (Attr.getNumArgs() == 0)
    Str = "";
  else if (!S.checkStringLiteralArgumentAttr(Attr, 0, Str, &ArgLoc))
    return;

  ARMInterruptAttr::InterruptType Kind;
  if (!ARMInterruptAttr::ConvertStrToInterruptType(Str, Kind)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_type_not_supported)
      << Attr.getName() << Str << ArgLoc;
    return;
  }

  unsigned Index = Attr.getAttributeSpellingListIndex();
  D->addAttr(::new (S.Context)
             ARMInterruptAttr(Attr.getLoc(), S.Context, Kind, Index));
}

static void handleMSP430InterruptAttr(Sema &S, Decl *D,
                                      const AttributeList &Attr) {
  if (!checkAttributeNumArgs(S, Attr, 1))
    return;

  if (!Attr.isArgExpr(0)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_type) << Attr.getName()
      << AANT_ArgumentIntegerConstant;
    return;    
  }

  // FIXME: Check for decl - it should be void ()(void).

  Expr *NumParamsExpr = static_cast<Expr *>(Attr.getArgAsExpr(0));
  llvm::APSInt NumParams(32);
  if (!NumParamsExpr->isIntegerConstantExpr(NumParams, S.Context)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_type)
      << Attr.getName() << AANT_ArgumentIntegerConstant
      << NumParamsExpr->getSourceRange();
    return;
  }

  unsigned Num = NumParams.getLimitedValue(255);
  if ((Num & 1) || Num > 30) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_out_of_bounds)
      << Attr.getName() << (int)NumParams.getSExtValue()
      << NumParamsExpr->getSourceRange();
    return;
  }

  D->addAttr(::new (S.Context)
              MSP430InterruptAttr(Attr.getLoc(), S.Context, Num,
                                  Attr.getAttributeSpellingListIndex()));
  D->addAttr(UsedAttr::CreateImplicit(S.Context));
}

static void handleMipsInterruptAttr(Sema &S, Decl *D,
                                    const AttributeList &Attr) {
  // Only one optional argument permitted.
  if (Attr.getNumArgs() > 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_too_many_arguments)
        << Attr.getName() << 1;
    return;
  }

  StringRef Str;
  SourceLocation ArgLoc;

  if (Attr.getNumArgs() == 0)
    Str = "";
  else if (!S.checkStringLiteralArgumentAttr(Attr, 0, Str, &ArgLoc))
    return;

  // Semantic checks for a function with the 'interrupt' attribute for MIPS:
  // a) Must be a function.
  // b) Must have no parameters.
  // c) Must have the 'void' return type.
  // d) Cannot have the 'mips16' attribute, as that instruction set
  //    lacks the 'eret' instruction.
  // e) The attribute itself must either have no argument or one of the
  //    valid interrupt types, see [MipsInterruptDocs].

  if (!isFunctionOrMethod(D)) {
    S.Diag(D->getLocation(), diag::warn_attribute_wrong_decl_type)
        << "'interrupt'" << ExpectedFunctionOrMethod;
    return;
  }

  if (hasFunctionProto(D) && getFunctionOrMethodNumParams(D) != 0) {
    S.Diag(D->getLocation(), diag::warn_mips_interrupt_attribute)
        << 0;
    return;
  }

  if (!getFunctionOrMethodResultType(D)->isVoidType()) {
    S.Diag(D->getLocation(), diag::warn_mips_interrupt_attribute)
        << 1;
    return;
  }

  if (checkAttrMutualExclusion<Mips16Attr>(S, D, Attr.getRange(),
                                           Attr.getName()))
    return;

  MipsInterruptAttr::InterruptType Kind;
  if (!MipsInterruptAttr::ConvertStrToInterruptType(Str, Kind)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_type_not_supported)
        << Attr.getName() << "'" + std::string(Str) + "'";
    return;
  }

  D->addAttr(::new (S.Context) MipsInterruptAttr(
      Attr.getLoc(), S.Context, Kind, Attr.getAttributeSpellingListIndex()));
}

static void handleAnyX86InterruptAttr(Sema &S, Decl *D,
                                      const AttributeList &Attr) {
  // Semantic checks for a function with the 'interrupt' attribute.
  // a) Must be a function.
  // b) Must have the 'void' return type.
  // c) Must take 1 or 2 arguments.
  // d) The 1st argument must be a pointer.
  // e) The 2nd argument (if any) must be an unsigned integer.
  if (!isFunctionOrMethod(D) || !hasFunctionProto(D) || isInstanceMethod(D) ||
      CXXMethodDecl::isStaticOverloadedOperator(
          cast<NamedDecl>(D)->getDeclName().getCXXOverloadedOperator())) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
        << Attr.getName() << ExpectedFunctionWithProtoType;
    return;
  }
  // Interrupt handler must have void return type.
  if (!getFunctionOrMethodResultType(D)->isVoidType()) {
    S.Diag(getFunctionOrMethodResultSourceRange(D).getBegin(),
           diag::err_anyx86_interrupt_attribute)
        << (S.Context.getTargetInfo().getTriple().getArch() == llvm::Triple::x86
                ? 0
                : 1)
        << 0;
    return;
  }
  // Interrupt handler must have 1 or 2 parameters.
  unsigned NumParams = getFunctionOrMethodNumParams(D);
  if (NumParams < 1 || NumParams > 2) {
    S.Diag(D->getLocStart(), diag::err_anyx86_interrupt_attribute)
        << (S.Context.getTargetInfo().getTriple().getArch() == llvm::Triple::x86
                ? 0
                : 1)
        << 1;
    return;
  }
  // The first argument must be a pointer.
  if (!getFunctionOrMethodParamType(D, 0)->isPointerType()) {
    S.Diag(getFunctionOrMethodParamRange(D, 0).getBegin(),
           diag::err_anyx86_interrupt_attribute)
        << (S.Context.getTargetInfo().getTriple().getArch() == llvm::Triple::x86
                ? 0
                : 1)
        << 2;
    return;
  }
  // The second argument, if present, must be an unsigned integer.
  unsigned TypeSize =
      S.Context.getTargetInfo().getTriple().getArch() == llvm::Triple::x86_64
          ? 64
          : 32;
  if (NumParams == 2 &&
      (!getFunctionOrMethodParamType(D, 1)->isUnsignedIntegerType() ||
       S.Context.getTypeSize(getFunctionOrMethodParamType(D, 1)) != TypeSize)) {
    S.Diag(getFunctionOrMethodParamRange(D, 1).getBegin(),
           diag::err_anyx86_interrupt_attribute)
        << (S.Context.getTargetInfo().getTriple().getArch() == llvm::Triple::x86
                ? 0
                : 1)
        << 3 << S.Context.getIntTypeForBitwidth(TypeSize, /*Signed=*/false);
    return;
  }
  D->addAttr(::new (S.Context) AnyX86InterruptAttr(
      Attr.getLoc(), S.Context, Attr.getAttributeSpellingListIndex()));
  D->addAttr(UsedAttr::CreateImplicit(S.Context));
}

static void handleAVRInterruptAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (!isFunctionOrMethod(D)) {
    S.Diag(D->getLocation(), diag::warn_attribute_wrong_decl_type)
        << "'interrupt'" << ExpectedFunction;
    return;
  }

  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  handleSimpleAttribute<AVRInterruptAttr>(S, D, Attr);
}

static void handleAVRSignalAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (!isFunctionOrMethod(D)) {
    S.Diag(D->getLocation(), diag::warn_attribute_wrong_decl_type)
        << "'signal'" << ExpectedFunction;
    return;
  }

  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  handleSimpleAttribute<AVRSignalAttr>(S, D, Attr);
}

static void handleInterruptAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // Dispatch the interrupt attribute based on the current target.
  switch (S.Context.getTargetInfo().getTriple().getArch()) {
  case llvm::Triple::msp430:
    handleMSP430InterruptAttr(S, D, Attr);
    break;
  case llvm::Triple::mipsel:
  case llvm::Triple::mips:
    handleMipsInterruptAttr(S, D, Attr);
    break;
  case llvm::Triple::x86:
  case llvm::Triple::x86_64:
    handleAnyX86InterruptAttr(S, D, Attr);
    break;
  case llvm::Triple::avr:
    handleAVRInterruptAttr(S, D, Attr);
    break;
  default:
    handleARMInterruptAttr(S, D, Attr);
    break;
  }
}

static void handleAMDGPUFlatWorkGroupSizeAttr(Sema &S, Decl *D,
                                              const AttributeList &Attr) {
  uint32_t Min = 0;
  Expr *MinExpr = Attr.getArgAsExpr(0);
  if (!checkUInt32Argument(S, Attr, MinExpr, Min))
    return;

  uint32_t Max = 0;
  Expr *MaxExpr = Attr.getArgAsExpr(1);
  if (!checkUInt32Argument(S, Attr, MaxExpr, Max))
    return;

  if (Min == 0 && Max != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_invalid)
      << Attr.getName() << 0;
    return;
  }
  if (Min > Max) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_invalid)
      << Attr.getName() << 1;
    return;
  }

  D->addAttr(::new (S.Context)
             AMDGPUFlatWorkGroupSizeAttr(Attr.getLoc(), S.Context, Min, Max,
                                         Attr.getAttributeSpellingListIndex()));
}

static void handleAMDGPUWavesPerEUAttr(Sema &S, Decl *D,
                                       const AttributeList &Attr) {
  uint32_t Min = 0;
  Expr *MinExpr = Attr.getArgAsExpr(0);
  if (!checkUInt32Argument(S, Attr, MinExpr, Min))
    return;

  uint32_t Max = 0;
  if (Attr.getNumArgs() == 2) {
    Expr *MaxExpr = Attr.getArgAsExpr(1);
    if (!checkUInt32Argument(S, Attr, MaxExpr, Max))
      return;
  }

  if (Min == 0 && Max != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_invalid)
      << Attr.getName() << 0;
    return;
  }
  if (Max != 0 && Min > Max) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_invalid)
      << Attr.getName() << 1;
    return;
  }

  D->addAttr(::new (S.Context)
             AMDGPUWavesPerEUAttr(Attr.getLoc(), S.Context, Min, Max,
                                  Attr.getAttributeSpellingListIndex()));
}

static void handleAMDGPUNumSGPRAttr(Sema &S, Decl *D,
                                    const AttributeList &Attr) {
  uint32_t NumSGPR = 0;
  Expr *NumSGPRExpr = Attr.getArgAsExpr(0);
  if (!checkUInt32Argument(S, Attr, NumSGPRExpr, NumSGPR))
    return;

  D->addAttr(::new (S.Context)
             AMDGPUNumSGPRAttr(Attr.getLoc(), S.Context, NumSGPR,
                               Attr.getAttributeSpellingListIndex()));
}

static void handleAMDGPUNumVGPRAttr(Sema &S, Decl *D,
                                    const AttributeList &Attr) {
  uint32_t NumVGPR = 0;
  Expr *NumVGPRExpr = Attr.getArgAsExpr(0);
  if (!checkUInt32Argument(S, Attr, NumVGPRExpr, NumVGPR))
    return;

  D->addAttr(::new (S.Context)
             AMDGPUNumVGPRAttr(Attr.getLoc(), S.Context, NumVGPR,
                               Attr.getAttributeSpellingListIndex()));
}

static void handleX86ForceAlignArgPointerAttr(Sema &S, Decl *D,
                                              const AttributeList& Attr) {
  // If we try to apply it to a function pointer, don't warn, but don't
  // do anything, either. It doesn't matter anyway, because there's nothing
  // special about calling a force_align_arg_pointer function.
  ValueDecl *VD = dyn_cast<ValueDecl>(D);
  if (VD && VD->getType()->isFunctionPointerType())
    return;
  // Also don't warn on function pointer typedefs.
  TypedefNameDecl *TD = dyn_cast<TypedefNameDecl>(D);
  if (TD && (TD->getUnderlyingType()->isFunctionPointerType() ||
    TD->getUnderlyingType()->isFunctionType()))
    return;
  // Attribute can only be applied to function types.
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << /* function */0;
    return;
  }

  D->addAttr(::new (S.Context)
              X86ForceAlignArgPointerAttr(Attr.getRange(), S.Context,
                                        Attr.getAttributeSpellingListIndex()));
}

static void handleLayoutVersion(Sema &S, Decl *D, const AttributeList &Attr) {
  uint32_t Version;
  Expr *VersionExpr = static_cast<Expr *>(Attr.getArgAsExpr(0));
  if (!checkUInt32Argument(S, Attr, Attr.getArgAsExpr(0), Version))
    return;

  // TODO: Investigate what happens with the next major version of MSVC.
  if (Version != LangOptions::MSVC2015) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_out_of_bounds)
        << Attr.getName() << Version << VersionExpr->getSourceRange();
    return;
  }

  D->addAttr(::new (S.Context)
                 LayoutVersionAttr(Attr.getRange(), S.Context, Version,
                                   Attr.getAttributeSpellingListIndex()));
}

DLLImportAttr *Sema::mergeDLLImportAttr(Decl *D, SourceRange Range,
                                        unsigned AttrSpellingListIndex) {
  if (D->hasAttr<DLLExportAttr>()) {
    Diag(Range.getBegin(), diag::warn_attribute_ignored) << "'dllimport'";
    return nullptr;
  }

  if (D->hasAttr<DLLImportAttr>())
    return nullptr;

  return ::new (Context) DLLImportAttr(Range, Context, AttrSpellingListIndex);
}

DLLExportAttr *Sema::mergeDLLExportAttr(Decl *D, SourceRange Range,
                                        unsigned AttrSpellingListIndex) {
  if (DLLImportAttr *Import = D->getAttr<DLLImportAttr>()) {
    Diag(Import->getLocation(), diag::warn_attribute_ignored) << Import;
    D->dropAttr<DLLImportAttr>();
  }

  if (D->hasAttr<DLLExportAttr>())
    return nullptr;

  return ::new (Context) DLLExportAttr(Range, Context, AttrSpellingListIndex);
}

static void handleDLLAttr(Sema &S, Decl *D, const AttributeList &A) {
  if (isa<ClassTemplatePartialSpecializationDecl>(D) &&
      S.Context.getTargetInfo().getCXXABI().isMicrosoft()) {
    S.Diag(A.getRange().getBegin(), diag::warn_attribute_ignored)
        << A.getName();
    return;
  }

  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    if (FD->isInlined() && A.getKind() == AttributeList::AT_DLLImport &&
        !S.Context.getTargetInfo().getCXXABI().isMicrosoft()) {
      // MinGW doesn't allow dllimport on inline functions.
      S.Diag(A.getRange().getBegin(), diag::warn_attribute_ignored_on_inline)
          << A.getName();
      return;
    }
  }

  if (auto *MD = dyn_cast<CXXMethodDecl>(D)) {
    if (S.Context.getTargetInfo().getCXXABI().isMicrosoft() &&
        MD->getParent()->isLambda()) {
      S.Diag(A.getRange().getBegin(), diag::err_attribute_dll_lambda) << A.getName();
      return;
    }
  }

  unsigned Index = A.getAttributeSpellingListIndex();
  Attr *NewAttr = A.getKind() == AttributeList::AT_DLLExport
                      ? (Attr *)S.mergeDLLExportAttr(D, A.getRange(), Index)
                      : (Attr *)S.mergeDLLImportAttr(D, A.getRange(), Index);
  if (NewAttr)
    D->addAttr(NewAttr);
}

MSInheritanceAttr *
Sema::mergeMSInheritanceAttr(Decl *D, SourceRange Range, bool BestCase,
                             unsigned AttrSpellingListIndex,
                             MSInheritanceAttr::Spelling SemanticSpelling) {
  if (MSInheritanceAttr *IA = D->getAttr<MSInheritanceAttr>()) {
    if (IA->getSemanticSpelling() == SemanticSpelling)
      return nullptr;
    Diag(IA->getLocation(), diag::err_mismatched_ms_inheritance)
        << 1 /*previous declaration*/;
    Diag(Range.getBegin(), diag::note_previous_ms_inheritance);
    D->dropAttr<MSInheritanceAttr>();
  }

  CXXRecordDecl *RD = cast<CXXRecordDecl>(D);
  if (RD->hasDefinition()) {
    if (checkMSInheritanceAttrOnDefinition(RD, Range, BestCase,
                                           SemanticSpelling)) {
      return nullptr;
    }
  } else {
    if (isa<ClassTemplatePartialSpecializationDecl>(RD)) {
      Diag(Range.getBegin(), diag::warn_ignored_ms_inheritance)
          << 1 /*partial specialization*/;
      return nullptr;
    }
    if (RD->getDescribedClassTemplate()) {
      Diag(Range.getBegin(), diag::warn_ignored_ms_inheritance)
          << 0 /*primary template*/;
      return nullptr;
    }
  }

  return ::new (Context)
      MSInheritanceAttr(Range, Context, BestCase, AttrSpellingListIndex);
}

static void handleCapabilityAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // The capability attributes take a single string parameter for the name of
  // the capability they represent. The lockable attribute does not take any
  // parameters. However, semantically, both attributes represent the same
  // concept, and so they use the same semantic attribute. Eventually, the
  // lockable attribute will be removed.
  //
  // For backward compatibility, any capability which has no specified string
  // literal will be considered a "mutex."
  StringRef N("mutex");
  SourceLocation LiteralLoc;
  if (Attr.getKind() == AttributeList::AT_Capability &&
      !S.checkStringLiteralArgumentAttr(Attr, 0, N, &LiteralLoc))
    return;

  // Currently, there are only two names allowed for a capability: role and
  // mutex (case insensitive). Diagnose other capability names.
  if (!N.equals_lower("mutex") && !N.equals_lower("role"))
    S.Diag(LiteralLoc, diag::warn_invalid_capability_name) << N;

  D->addAttr(::new (S.Context) CapabilityAttr(Attr.getRange(), S.Context, N,
                                        Attr.getAttributeSpellingListIndex()));
}

static void handleAssertCapabilityAttr(Sema &S, Decl *D,
                                       const AttributeList &Attr) {
  D->addAttr(::new (S.Context) AssertCapabilityAttr(Attr.getRange(), S.Context,
                                                    Attr.getArgAsExpr(0),
                                        Attr.getAttributeSpellingListIndex()));
}

static void handleAcquireCapabilityAttr(Sema &S, Decl *D,
                                        const AttributeList &Attr) {
  SmallVector<Expr*, 1> Args;
  if (!checkLockFunAttrCommon(S, D, Attr, Args))
    return;

  D->addAttr(::new (S.Context) AcquireCapabilityAttr(Attr.getRange(),
                                                     S.Context,
                                                     Args.data(), Args.size(),
                                        Attr.getAttributeSpellingListIndex()));
}

static void handleTryAcquireCapabilityAttr(Sema &S, Decl *D,
                                           const AttributeList &Attr) {
  SmallVector<Expr*, 2> Args;
  if (!checkTryLockFunAttrCommon(S, D, Attr, Args))
    return;

  D->addAttr(::new (S.Context) TryAcquireCapabilityAttr(Attr.getRange(),
                                                        S.Context,
                                                        Attr.getArgAsExpr(0),
                                                        Args.data(),
                                                        Args.size(),
                                        Attr.getAttributeSpellingListIndex()));
}

static void handleReleaseCapabilityAttr(Sema &S, Decl *D,
                                        const AttributeList &Attr) {
  // Check that all arguments are lockable objects.
  SmallVector<Expr *, 1> Args;
  checkAttrArgsAreCapabilityObjs(S, D, Attr, Args, 0, true);

  D->addAttr(::new (S.Context) ReleaseCapabilityAttr(
      Attr.getRange(), S.Context, Args.data(), Args.size(),
      Attr.getAttributeSpellingListIndex()));
}

static void handleRequiresCapabilityAttr(Sema &S, Decl *D,
                                         const AttributeList &Attr) {
  if (!checkAttributeAtLeastNumArgs(S, Attr, 1))
    return;

  // check that all arguments are lockable objects
  SmallVector<Expr*, 1> Args;
  checkAttrArgsAreCapabilityObjs(S, D, Attr, Args);
  if (Args.empty())
    return;

  RequiresCapabilityAttr *RCA = ::new (S.Context)
    RequiresCapabilityAttr(Attr.getRange(), S.Context, Args.data(),
                           Args.size(), Attr.getAttributeSpellingListIndex());

  D->addAttr(RCA);
}

static void handleDeprecatedAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (auto *NSD = dyn_cast<NamespaceDecl>(D)) {
    if (NSD->isAnonymousNamespace()) {
      S.Diag(Attr.getLoc(), diag::warn_deprecated_anonymous_namespace);
      // Do not want to attach the attribute to the namespace because that will
      // cause confusing diagnostic reports for uses of declarations within the
      // namespace.
      return;
    }
  }

  // Handle the cases where the attribute has a text message.
  StringRef Str, Replacement;
  if (Attr.isArgExpr(0) && Attr.getArgAsExpr(0) &&
      !S.checkStringLiteralArgumentAttr(Attr, 0, Str))
    return;

  // Only support a single optional message for Declspec and CXX11.
  if (Attr.isDeclspecAttribute() || Attr.isCXX11Attribute())
    checkAttributeAtMostNumArgs(S, Attr, 1);
  else if (Attr.isArgExpr(1) && Attr.getArgAsExpr(1) &&
           !S.checkStringLiteralArgumentAttr(Attr, 1, Replacement))
    return;

  if (!S.getLangOpts().CPlusPlus14)
    if (Attr.isCXX11Attribute() &&
        !(Attr.hasScope() && Attr.getScopeName()->isStr("gnu")))
      S.Diag(Attr.getLoc(), diag::ext_cxx14_attr) << Attr.getName();

  D->addAttr(::new (S.Context)
                 DeprecatedAttr(Attr.getRange(), S.Context, Str, Replacement,
                                Attr.getAttributeSpellingListIndex()));
}

static bool isGlobalVar(const Decl *D) {
  if (const auto *S = dyn_cast<VarDecl>(D))
    return S->hasGlobalStorage();
  return false;
}

static void handleNoSanitizeAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (!checkAttributeAtLeastNumArgs(S, Attr, 1))
    return;

  std::vector<StringRef> Sanitizers;

  for (unsigned I = 0, E = Attr.getNumArgs(); I != E; ++I) {
    StringRef SanitizerName;
    SourceLocation LiteralLoc;

    if (!S.checkStringLiteralArgumentAttr(Attr, I, SanitizerName, &LiteralLoc))
      return;

    if (parseSanitizerValue(SanitizerName, /*AllowGroups=*/true) == 0)
      S.Diag(LiteralLoc, diag::warn_unknown_sanitizer_ignored) << SanitizerName;
    else if (isGlobalVar(D) && SanitizerName != "address")
      S.Diag(D->getLocation(), diag::err_attribute_wrong_decl_type)
          << Attr.getName() << ExpectedFunctionOrMethod;
    Sanitizers.push_back(SanitizerName);
  }

  D->addAttr(::new (S.Context) NoSanitizeAttr(
      Attr.getRange(), S.Context, Sanitizers.data(), Sanitizers.size(),
      Attr.getAttributeSpellingListIndex()));
}

static void handleNoSanitizeSpecificAttr(Sema &S, Decl *D,
                                         const AttributeList &Attr) {
  StringRef AttrName = Attr.getName()->getName();
  normalizeName(AttrName);
  StringRef SanitizerName = llvm::StringSwitch<StringRef>(AttrName)
                                .Case("no_address_safety_analysis", "address")
                                .Case("no_sanitize_address", "address")
                                .Case("no_sanitize_thread", "thread")
                                .Case("no_sanitize_memory", "memory");
  if (isGlobalVar(D) && SanitizerName != "address")
    S.Diag(D->getLocation(), diag::err_attribute_wrong_decl_type)
        << Attr.getName() << ExpectedFunction;
  D->addAttr(::new (S.Context)
                 NoSanitizeAttr(Attr.getRange(), S.Context, &SanitizerName, 1,
                                Attr.getAttributeSpellingListIndex()));
}

static void handleInternalLinkageAttr(Sema &S, Decl *D,
                                      const AttributeList &Attr) {
  if (InternalLinkageAttr *Internal =
          S.mergeInternalLinkageAttr(D, Attr.getRange(), Attr.getName(),
                                     Attr.getAttributeSpellingListIndex()))
    D->addAttr(Internal);
}

static void handleOpenCLNoSVMAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (S.LangOpts.OpenCLVersion != 200)
    S.Diag(Attr.getLoc(), diag::err_attribute_requires_opencl_version)
        << Attr.getName() << "2.0" << 0;
  else
    S.Diag(Attr.getLoc(), diag::warn_opencl_attr_deprecated_ignored)
        << Attr.getName() << "2.0";
}

/// Handles semantic checking for features that are common to all attributes,
/// such as checking whether a parameter was properly specified, or the correct
/// number of arguments were passed, etc.
static bool handleCommonAttributeFeatures(Sema &S, Scope *scope, Decl *D,
                                          const AttributeList &Attr) {
  // Several attributes carry different semantics than the parsing requires, so
  // those are opted out of the common argument checks.
  //
  // We also bail on unknown and ignored attributes because those are handled
  // as part of the target-specific handling logic.
  if (Attr.getKind() == AttributeList::UnknownAttribute)
    return false;
  // Check whether the attribute requires specific language extensions to be
  // enabled.
  if (!Attr.diagnoseLangOpts(S))
    return true;
  // Check whether the attribute appertains to the given subject.
  if (!Attr.diagnoseAppertainsTo(S, D))
    return true;
  if (Attr.hasCustomParsing())
    return false;

  if (Attr.getMinArgs() == Attr.getMaxArgs()) {
    // If there are no optional arguments, then checking for the argument count
    // is trivial.
    if (!checkAttributeNumArgs(S, Attr, Attr.getMinArgs()))
      return true;
  } else {
    // There are optional arguments, so checking is slightly more involved.
    if (Attr.getMinArgs() &&
        !checkAttributeAtLeastNumArgs(S, Attr, Attr.getMinArgs()))
      return true;
    else if (!Attr.hasVariadicArg() && Attr.getMaxArgs() &&
             !checkAttributeAtMostNumArgs(S, Attr, Attr.getMaxArgs()))
      return true;
  }

  return false;
}

static void handleOpenCLAccessAttr(Sema &S, Decl *D,
                                   const AttributeList &Attr) {
  if (D->isInvalidDecl())
    return;

  // Check if there is only one access qualifier.
  if (D->hasAttr<OpenCLAccessAttr>()) {
    S.Diag(Attr.getLoc(), diag::err_opencl_multiple_access_qualifiers)
        << D->getSourceRange();
    D->setInvalidDecl(true);
    return;
  }

  // OpenCL v2.0 s6.6 - read_write can be used for image types to specify that an
  // image object can be read and written.
  // OpenCL v2.0 s6.13.6 - A kernel cannot read from and write to the same pipe
  // object. Using the read_write (or __read_write) qualifier with the pipe
  // qualifier is a compilation error.
  if (const ParmVarDecl *PDecl = dyn_cast<ParmVarDecl>(D)) {
    const Type *DeclTy = PDecl->getType().getCanonicalType().getTypePtr();
    if (Attr.getName()->getName().find("read_write") != StringRef::npos) {
      if (S.getLangOpts().OpenCLVersion < 200 || DeclTy->isPipeType()) {
        S.Diag(Attr.getLoc(), diag::err_opencl_invalid_read_write)
            << Attr.getName() << PDecl->getType() << DeclTy->isImageType();
        D->setInvalidDecl(true);
        return;
      }
    }
  }

  D->addAttr(::new (S.Context) OpenCLAccessAttr(
      Attr.getRange(), S.Context, Attr.getAttributeSpellingListIndex()));
}

//===----------------------------------------------------------------------===//
// Top Level Sema Entry Points
//===----------------------------------------------------------------------===//

/// ProcessDeclAttribute - Apply the specific attribute to the specified decl if
/// the attribute applies to decls.  If the attribute is a type attribute, just
/// silently ignore it if a GNU attribute.
static void ProcessDeclAttribute(Sema &S, Scope *scope, Decl *D,
                                 const AttributeList &Attr,
                                 bool IncludeCXX11Attributes) {
  if (Attr.isInvalid() || Attr.getKind() == AttributeList::IgnoredAttribute)
    return;

  // Ignore C++11 attributes on declarator chunks: they appertain to the type
  // instead.
  if (Attr.isCXX11Attribute() && !IncludeCXX11Attributes)
    return;

  // Unknown attributes are automatically warned on. Target-specific attributes
  // which do not apply to the current target architecture are treated as
  // though they were unknown attributes.
  if (Attr.getKind() == AttributeList::UnknownAttribute ||
      !Attr.existsInTarget(S.Context.getTargetInfo())) {
    S.Diag(Attr.getLoc(), Attr.isDeclspecAttribute()
                              ? diag::warn_unhandled_ms_attribute_ignored
                              : diag::warn_unknown_attribute_ignored)
        << Attr.getName();
    return;
  }

  if (handleCommonAttributeFeatures(S, scope, D, Attr))
    return;

  switch (Attr.getKind()) {
  default:
    if (!Attr.isStmtAttr()) {
      // Type attributes are handled elsewhere; silently move on.
      assert(Attr.isTypeAttr() && "Non-type attribute not handled");
      break;
    }
    S.Diag(Attr.getLoc(), diag::err_stmt_attribute_invalid_on_decl)
        << Attr.getName() << D->getLocation();
    break;
  case AttributeList::AT_Interrupt:
    handleInterruptAttr(S, D, Attr);
    break;
  case AttributeList::AT_X86ForceAlignArgPointer:
    handleX86ForceAlignArgPointerAttr(S, D, Attr);
    break;
  case AttributeList::AT_DLLExport:
  case AttributeList::AT_DLLImport:
    handleDLLAttr(S, D, Attr);
    break;
  case AttributeList::AT_Mips16:
    handleSimpleAttributeWithExclusions<Mips16Attr, MicroMipsAttr,
                                        MipsInterruptAttr>(S, D, Attr);
    break;
  case AttributeList::AT_NoMips16:
    handleSimpleAttribute<NoMips16Attr>(S, D, Attr);
    break;
  case AttributeList::AT_MicroMips:
    handleSimpleAttributeWithExclusions<MicroMipsAttr, Mips16Attr>(S, D, Attr);
    break;
  case AttributeList::AT_NoMicroMips:
    handleSimpleAttribute<NoMicroMipsAttr>(S, D, Attr);
    break;
  case AttributeList::AT_AMDGPUFlatWorkGroupSize:
    handleAMDGPUFlatWorkGroupSizeAttr(S, D, Attr);
    break;
  case AttributeList::AT_AMDGPUWavesPerEU:
    handleAMDGPUWavesPerEUAttr(S, D, Attr);
    break;
  case AttributeList::AT_AMDGPUNumSGPR:
    handleAMDGPUNumSGPRAttr(S, D, Attr);
    break;
  case AttributeList::AT_AMDGPUNumVGPR:
    handleAMDGPUNumVGPRAttr(S, D, Attr);
    break;
  case AttributeList::AT_AVRSignal:
    handleAVRSignalAttr(S, D, Attr);
    break;
  case AttributeList::AT_IBAction:
    handleSimpleAttribute<IBActionAttr>(S, D, Attr);
    break;
  case AttributeList::AT_IBOutlet:
    handleIBOutlet(S, D, Attr);
    break;
  case AttributeList::AT_IBOutletCollection:
    handleIBOutletCollection(S, D, Attr);
    break;
  case AttributeList::AT_IFunc:
    handleIFuncAttr(S, D, Attr);
    break;
  case AttributeList::AT_Alias:
    handleAliasAttr(S, D, Attr);
    break;
  case AttributeList::AT_Aligned:
    handleAlignedAttr(S, D, Attr);
    break;
  case AttributeList::AT_AlignValue:
    handleAlignValueAttr(S, D, Attr);
    break;
  case AttributeList::AT_AllocSize:
    handleAllocSizeAttr(S, D, Attr);
    break;
  case AttributeList::AT_AlwaysInline:
    handleAlwaysInlineAttr(S, D, Attr);
    break;
  case AttributeList::AT_AnalyzerNoReturn:
    handleAnalyzerNoReturnAttr(S, D, Attr);
    break;
  case AttributeList::AT_TLSModel:
    handleTLSModelAttr(S, D, Attr);
    break;
  case AttributeList::AT_Annotate:
    handleAnnotateAttr(S, D, Attr);
    break;
  case AttributeList::AT_Availability:
    handleAvailabilityAttr(S, D, Attr);
    break;
  case AttributeList::AT_CarriesDependency:
    handleDependencyAttr(S, scope, D, Attr);
    break;
  case AttributeList::AT_Common:
    handleCommonAttr(S, D, Attr);
    break;
  case AttributeList::AT_CUDAConstant:
    handleConstantAttr(S, D, Attr);
    break;
  case AttributeList::AT_PassObjectSize:
    handlePassObjectSizeAttr(S, D, Attr);
    break;
  case AttributeList::AT_Constructor:
    handleConstructorAttr(S, D, Attr);
    break;
  case AttributeList::AT_CXX11NoReturn:
    handleSimpleAttribute<CXX11NoReturnAttr>(S, D, Attr);
    break;
  case AttributeList::AT_Deprecated:
    handleDeprecatedAttr(S, D, Attr);
    break;
  case AttributeList::AT_Destructor:
    handleDestructorAttr(S, D, Attr);
    break;
  case AttributeList::AT_EnableIf:
    handleEnableIfAttr(S, D, Attr);
    break;
  case AttributeList::AT_DiagnoseIf:
    handleDiagnoseIfAttr(S, D, Attr);
    break;
  case AttributeList::AT_ExtVectorType:
    handleExtVectorTypeAttr(S, scope, D, Attr);
    break;
  case AttributeList::AT_ExternalSourceSymbol:
    handleExternalSourceSymbolAttr(S, D, Attr);
    break;
  case AttributeList::AT_MinSize:
    handleMinSizeAttr(S, D, Attr);
    break;
  case AttributeList::AT_OptimizeNone:
    handleOptimizeNoneAttr(S, D, Attr);
    break;
  case AttributeList::AT_FlagEnum:
    handleSimpleAttribute<FlagEnumAttr>(S, D, Attr);
    break;
  case AttributeList::AT_EnumExtensibility:
    handleEnumExtensibilityAttr(S, D, Attr);
    break;
  case AttributeList::AT_Flatten:
    handleSimpleAttribute<FlattenAttr>(S, D, Attr);
    break;
  case AttributeList::AT_Format:
    handleFormatAttr(S, D, Attr);
    break;
  case AttributeList::AT_FormatArg:
    handleFormatArgAttr(S, D, Attr);
    break;
  case AttributeList::AT_CUDAGlobal:
    handleGlobalAttr(S, D, Attr);
    break;
  case AttributeList::AT_CUDADevice:
    handleSimpleAttributeWithExclusions<CUDADeviceAttr, CUDAGlobalAttr>(S, D,
                                                                        Attr);
    break;
  case AttributeList::AT_CUDAHost:
    handleSimpleAttributeWithExclusions<CUDAHostAttr, CUDAGlobalAttr>(S, D,
                                                                      Attr);
    break;
  case AttributeList::AT_GNUInline:
    handleGNUInlineAttr(S, D, Attr);
    break;
  case AttributeList::AT_CUDALaunchBounds:
    handleLaunchBoundsAttr(S, D, Attr);
    break;
  case AttributeList::AT_Restrict:
    handleRestrictAttr(S, D, Attr);
    break;
  case AttributeList::AT_MayAlias:
    handleSimpleAttribute<MayAliasAttr>(S, D, Attr);
    break;
  case AttributeList::AT_Mode:
    handleModeAttr(S, D, Attr);
    break;
  case AttributeList::AT_NoAlias:
    handleSimpleAttribute<NoAliasAttr>(S, D, Attr);
    break;
  case AttributeList::AT_NoCommon:
    handleSimpleAttribute<NoCommonAttr>(S, D, Attr);
    break;
  case AttributeList::AT_NoSplitStack:
    handleSimpleAttribute<NoSplitStackAttr>(S, D, Attr);
    break;
  case AttributeList::AT_NonNull:
    if (ParmVarDecl *PVD = dyn_cast<ParmVarDecl>(D))
      handleNonNullAttrParameter(S, PVD, Attr);
    else
      handleNonNullAttr(S, D, Attr);
    break;
  case AttributeList::AT_ReturnsNonNull:
    handleReturnsNonNullAttr(S, D, Attr);
    break;
  case AttributeList::AT_AssumeAligned:
    handleAssumeAlignedAttr(S, D, Attr);
    break;
  case AttributeList::AT_AllocAlign:
    handleAllocAlignAttr(S, D, Attr);
    break;
  case AttributeList::AT_Overloadable:
    handleSimpleAttribute<OverloadableAttr>(S, D, Attr);
    break;
  case AttributeList::AT_Ownership:
    handleOwnershipAttr(S, D, Attr);
    break;
  case AttributeList::AT_Cold:
    handleColdAttr(S, D, Attr);
    break;
  case AttributeList::AT_Hot:
    handleHotAttr(S, D, Attr);
    break;
  case AttributeList::AT_Naked:
    handleNakedAttr(S, D, Attr);
    break;
  case AttributeList::AT_NoReturn:
    handleNoReturnAttr(S, D, Attr);
    break;
  case AttributeList::AT_NoThrow:
    handleSimpleAttribute<NoThrowAttr>(S, D, Attr);
    break;
  case AttributeList::AT_CUDAShared:
    handleSharedAttr(S, D, Attr);
    break;
  case AttributeList::AT_VecReturn:
    handleVecReturnAttr(S, D, Attr);
    break;
  case AttributeList::AT_ObjCOwnership:
    handleObjCOwnershipAttr(S, D, Attr);
    break;
  case AttributeList::AT_ObjCPreciseLifetime:
    handleObjCPreciseLifetimeAttr(S, D, Attr);
    break;
  case AttributeList::AT_ObjCReturnsInnerPointer:
    handleObjCReturnsInnerPointerAttr(S, D, Attr);
    break;
  case AttributeList::AT_ObjCRequiresSuper:
    handleObjCRequiresSuperAttr(S, D, Attr);
    break;
  case AttributeList::AT_ObjCBridge:
    handleObjCBridgeAttr(S, scope, D, Attr);
    break;
  case AttributeList::AT_ObjCBridgeMutable:
    handleObjCBridgeMutableAttr(S, scope, D, Attr);
    break;
  case AttributeList::AT_ObjCBridgeRelated:
    handleObjCBridgeRelatedAttr(S, scope, D, Attr);
    break;
  case AttributeList::AT_ObjCDesignatedInitializer:
    handleObjCDesignatedInitializer(S, D, Attr);
    break;
  case AttributeList::AT_ObjCRuntimeName:
    handleObjCRuntimeName(S, D, Attr);
    break;
   case AttributeList::AT_ObjCRuntimeVisible:
    handleSimpleAttribute<ObjCRuntimeVisibleAttr>(S, D, Attr);
    break;
  case AttributeList::AT_ObjCBoxable:
    handleObjCBoxable(S, D, Attr);
    break;
  case AttributeList::AT_CFAuditedTransfer:
    handleCFAuditedTransferAttr(S, D, Attr);
    break;
  case AttributeList::AT_CFUnknownTransfer:
    handleCFUnknownTransferAttr(S, D, Attr);
    break;
  case AttributeList::AT_CFConsumed:
  case AttributeList::AT_NSConsumed:
    handleNSConsumedAttr(S, D, Attr);
    break;
  case AttributeList::AT_NSConsumesSelf:
    handleSimpleAttribute<NSConsumesSelfAttr>(S, D, Attr);
    break;
  case AttributeList::AT_NSReturnsAutoreleased:
  case AttributeList::AT_NSReturnsNotRetained:
  case AttributeList::AT_CFReturnsNotRetained:
  case AttributeList::AT_NSReturnsRetained:
  case AttributeList::AT_CFReturnsRetained:
    handleNSReturnsRetainedAttr(S, D, Attr);
    break;
  case AttributeList::AT_WorkGroupSizeHint:
    handleWorkGroupSize<WorkGroupSizeHintAttr>(S, D, Attr);
    break;
  case AttributeList::AT_ReqdWorkGroupSize:
    handleWorkGroupSize<ReqdWorkGroupSizeAttr>(S, D, Attr);
    break;
  case AttributeList::AT_OpenCLIntelReqdSubGroupSize:
    handleSubGroupSize(S, D, Attr);
    break;
  case AttributeList::AT_VecTypeHint:
    handleVecTypeHint(S, D, Attr);
    break;
  case AttributeList::AT_RequireConstantInit:
    handleSimpleAttribute<RequireConstantInitAttr>(S, D, Attr);
    break;
  case AttributeList::AT_InitPriority:
    handleInitPriorityAttr(S, D, Attr);
    break;
  case AttributeList::AT_Packed:
    handlePackedAttr(S, D, Attr);
    break;
  case AttributeList::AT_Section:
    handleSectionAttr(S, D, Attr);
    break;
  case AttributeList::AT_Target:
    handleTargetAttr(S, D, Attr);
    break;
  case AttributeList::AT_Unavailable:
    handleAttrWithMessage<UnavailableAttr>(S, D, Attr);
    break;
  case AttributeList::AT_ArcWeakrefUnavailable:
    handleSimpleAttribute<ArcWeakrefUnavailableAttr>(S, D, Attr);
    break;
  case AttributeList::AT_ObjCRootClass:
    handleSimpleAttribute<ObjCRootClassAttr>(S, D, Attr);
    break;
  case AttributeList::AT_ObjCSubclassingRestricted:
    handleSimpleAttribute<ObjCSubclassingRestrictedAttr>(S, D, Attr);
    break;
  case AttributeList::AT_ObjCExplicitProtocolImpl:
    handleObjCSuppresProtocolAttr(S, D, Attr);
    break;
  case AttributeList::AT_ObjCRequiresPropertyDefs:
    handleSimpleAttribute<ObjCRequiresPropertyDefsAttr>(S, D, Attr);
    break;
  case AttributeList::AT_Unused:
    handleUnusedAttr(S, D, Attr);
    break;
  case AttributeList::AT_ReturnsTwice:
    handleSimpleAttribute<ReturnsTwiceAttr>(S, D, Attr);
    break;
  case AttributeList::AT_NotTailCalled:
    handleNotTailCalledAttr(S, D, Attr);
    break;
  case AttributeList::AT_DisableTailCalls:
    handleDisableTailCallsAttr(S, D, Attr);
    break;
  case AttributeList::AT_Used:
    handleUsedAttr(S, D, Attr);
    break;
  case AttributeList::AT_Visibility:
    handleVisibilityAttr(S, D, Attr, false);
    break;
  case AttributeList::AT_TypeVisibility:
    handleVisibilityAttr(S, D, Attr, true);
    break;
  case AttributeList::AT_WarnUnused:
    handleSimpleAttribute<WarnUnusedAttr>(S, D, Attr);
    break;
  case AttributeList::AT_WarnUnusedResult:
    handleWarnUnusedResult(S, D, Attr);
    break;
  case AttributeList::AT_Weak:
    handleSimpleAttribute<WeakAttr>(S, D, Attr);
    break;
  case AttributeList::AT_WeakRef:
    handleWeakRefAttr(S, D, Attr);
    break;
  case AttributeList::AT_WeakImport:
    handleWeakImportAttr(S, D, Attr);
    break;
  case AttributeList::AT_TransparentUnion:
    handleTransparentUnionAttr(S, D, Attr);
    break;
  case AttributeList::AT_ObjCException:
    handleSimpleAttribute<ObjCExceptionAttr>(S, D, Attr);
    break;
  case AttributeList::AT_ObjCMethodFamily:
    handleObjCMethodFamilyAttr(S, D, Attr);
    break;
  case AttributeList::AT_ObjCNSObject:
    handleObjCNSObject(S, D, Attr);
    break;
  case AttributeList::AT_ObjCIndependentClass:
    handleObjCIndependentClass(S, D, Attr);
    break;
  case AttributeList::AT_Blocks:
    handleBlocksAttr(S, D, Attr);
    break;
  case AttributeList::AT_Sentinel:
    handleSentinelAttr(S, D, Attr);
    break;
  case AttributeList::AT_Const:
    handleSimpleAttribute<ConstAttr>(S, D, Attr);
    break;
  case AttributeList::AT_Pure:
    handleSimpleAttribute<PureAttr>(S, D, Attr);
    break;
  case AttributeList::AT_Cleanup:
    handleCleanupAttr(S, D, Attr);
    break;
  case AttributeList::AT_NoDebug:
    handleNoDebugAttr(S, D, Attr);
    break;
  case AttributeList::AT_NoDuplicate:
    handleSimpleAttribute<NoDuplicateAttr>(S, D, Attr);
    break;
  case AttributeList::AT_Convergent:
    handleSimpleAttribute<ConvergentAttr>(S, D, Attr);
    break;
  case AttributeList::AT_NoInline:
    handleSimpleAttribute<NoInlineAttr>(S, D, Attr);
    break;
  case AttributeList::AT_NoInstrumentFunction: // Interacts with -pg.
    handleSimpleAttribute<NoInstrumentFunctionAttr>(S, D, Attr);
    break;
  case AttributeList::AT_StdCall:
  case AttributeList::AT_CDecl:
  case AttributeList::AT_FastCall:
  case AttributeList::AT_ThisCall:
  case AttributeList::AT_Pascal:
  case AttributeList::AT_RegCall:
  case AttributeList::AT_SwiftCall:
  case AttributeList::AT_VectorCall:
  case AttributeList::AT_MSABI:
  case AttributeList::AT_SysVABI:
  case AttributeList::AT_Pcs:
  case AttributeList::AT_IntelOclBicc:
  case AttributeList::AT_PreserveMost:
  case AttributeList::AT_PreserveAll:
    handleCallConvAttr(S, D, Attr);
    break;
  case AttributeList::AT_Suppress:
    handleSuppressAttr(S, D, Attr);
    break;
  case AttributeList::AT_OpenCLKernel:
    handleSimpleAttribute<OpenCLKernelAttr>(S, D, Attr);
    break;
  case AttributeList::AT_OpenCLAccess:
    handleOpenCLAccessAttr(S, D, Attr);
    break;
  case AttributeList::AT_OpenCLNoSVM:
    handleOpenCLNoSVMAttr(S, D, Attr);
    break;
  case AttributeList::AT_SwiftContext:
    handleParameterABIAttr(S, D, Attr, ParameterABI::SwiftContext);
    break;
  case AttributeList::AT_SwiftErrorResult:
    handleParameterABIAttr(S, D, Attr, ParameterABI::SwiftErrorResult);
    break;
  case AttributeList::AT_SwiftIndirectResult:
    handleParameterABIAttr(S, D, Attr, ParameterABI::SwiftIndirectResult);
    break;
  case AttributeList::AT_InternalLinkage:
    handleInternalLinkageAttr(S, D, Attr);
    break;
  case AttributeList::AT_LTOVisibilityPublic:
    handleSimpleAttribute<LTOVisibilityPublicAttr>(S, D, Attr);
    break;

  // Microsoft attributes:
  case AttributeList::AT_EmptyBases:
    handleSimpleAttribute<EmptyBasesAttr>(S, D, Attr);
    break;
  case AttributeList::AT_LayoutVersion:
    handleLayoutVersion(S, D, Attr);
    break;
  case AttributeList::AT_MSNoVTable:
    handleSimpleAttribute<MSNoVTableAttr>(S, D, Attr);
    break;
  case AttributeList::AT_MSStruct:
    handleSimpleAttribute<MSStructAttr>(S, D, Attr);
    break;
  case AttributeList::AT_Uuid:
    handleUuidAttr(S, D, Attr);
    break;
  case AttributeList::AT_MSInheritance:
    handleMSInheritanceAttr(S, D, Attr);
    break;
  case AttributeList::AT_SelectAny:
    handleSimpleAttribute<SelectAnyAttr>(S, D, Attr);
    break;
  case AttributeList::AT_Thread:
    handleDeclspecThreadAttr(S, D, Attr);
    break;

  case AttributeList::AT_AbiTag:
    handleAbiTagAttr(S, D, Attr);
    break;

  // Thread safety attributes:
  case AttributeList::AT_AssertExclusiveLock:
    handleAssertExclusiveLockAttr(S, D, Attr);
    break;
  case AttributeList::AT_AssertSharedLock:
    handleAssertSharedLockAttr(S, D, Attr);
    break;
  case AttributeList::AT_GuardedVar:
    handleSimpleAttribute<GuardedVarAttr>(S, D, Attr);
    break;
  case AttributeList::AT_PtGuardedVar:
    handlePtGuardedVarAttr(S, D, Attr);
    break;
  case AttributeList::AT_ScopedLockable:
    handleSimpleAttribute<ScopedLockableAttr>(S, D, Attr);
    break;
  case AttributeList::AT_NoSanitize:
    handleNoSanitizeAttr(S, D, Attr);
    break;
  case AttributeList::AT_NoSanitizeSpecific:
    handleNoSanitizeSpecificAttr(S, D, Attr);
    break;
  case AttributeList::AT_NoThreadSafetyAnalysis:
    handleSimpleAttribute<NoThreadSafetyAnalysisAttr>(S, D, Attr);
    break;
  case AttributeList::AT_GuardedBy:
    handleGuardedByAttr(S, D, Attr);
    break;
  case AttributeList::AT_PtGuardedBy:
    handlePtGuardedByAttr(S, D, Attr);
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
  case AttributeList::AT_SharedTrylockFunction:
    handleSharedTrylockFunctionAttr(S, D, Attr);
    break;
  case AttributeList::AT_AcquiredBefore:
    handleAcquiredBeforeAttr(S, D, Attr);
    break;
  case AttributeList::AT_AcquiredAfter:
    handleAcquiredAfterAttr(S, D, Attr);
    break;

  // Capability analysis attributes.
  case AttributeList::AT_Capability:
  case AttributeList::AT_Lockable:
    handleCapabilityAttr(S, D, Attr);
    break;
  case AttributeList::AT_RequiresCapability:
    handleRequiresCapabilityAttr(S, D, Attr);
    break;

  case AttributeList::AT_AssertCapability:
    handleAssertCapabilityAttr(S, D, Attr);
    break;
  case AttributeList::AT_AcquireCapability:
    handleAcquireCapabilityAttr(S, D, Attr);
    break;
  case AttributeList::AT_ReleaseCapability:
    handleReleaseCapabilityAttr(S, D, Attr);
    break;
  case AttributeList::AT_TryAcquireCapability:
    handleTryAcquireCapabilityAttr(S, D, Attr);
    break;

  // Consumed analysis attributes.
  case AttributeList::AT_Consumable:
    handleConsumableAttr(S, D, Attr);
    break;
  case AttributeList::AT_ConsumableAutoCast:
    handleSimpleAttribute<ConsumableAutoCastAttr>(S, D, Attr);
    break;
  case AttributeList::AT_ConsumableSetOnRead:
    handleSimpleAttribute<ConsumableSetOnReadAttr>(S, D, Attr);
    break;
  case AttributeList::AT_CallableWhen:
    handleCallableWhenAttr(S, D, Attr);
    break;
  case AttributeList::AT_ParamTypestate:
    handleParamTypestateAttr(S, D, Attr);
    break;
  case AttributeList::AT_ReturnTypestate:
    handleReturnTypestateAttr(S, D, Attr);
    break;
  case AttributeList::AT_SetTypestate:
    handleSetTypestateAttr(S, D, Attr);
    break;
  case AttributeList::AT_TestTypestate:
    handleTestTypestateAttr(S, D, Attr);
    break;

  // Type safety attributes.
  case AttributeList::AT_ArgumentWithTypeTag:
    handleArgumentWithTypeTagAttr(S, D, Attr);
    break;
  case AttributeList::AT_TypeTagForDatatype:
    handleTypeTagForDatatypeAttr(S, D, Attr);
    break;
  case AttributeList::AT_AnyX86NoCallerSavedRegisters:
    handleNoCallerSavedRegsAttr(S, D, Attr);
    break;
  case AttributeList::AT_RenderScriptKernel:
    handleSimpleAttribute<RenderScriptKernelAttr>(S, D, Attr);
    break;
  // XRay attributes.
  case AttributeList::AT_XRayInstrument:
    handleSimpleAttribute<XRayInstrumentAttr>(S, D, Attr);
    break;
  case AttributeList::AT_XRayLogArgs:
    handleXRayLogArgsAttr(S, D, Attr);
    break;
  }
}

/// ProcessDeclAttributeList - Apply all the decl attributes in the specified
/// attribute list to the specified decl, ignoring any type attributes.
void Sema::ProcessDeclAttributeList(Scope *S, Decl *D,
                                    const AttributeList *AttrList,
                                    bool IncludeCXX11Attributes) {
  for (const AttributeList* l = AttrList; l; l = l->getNext())
    ProcessDeclAttribute(*this, S, D, *l, IncludeCXX11Attributes);

  // FIXME: We should be able to handle these cases in TableGen.
  // GCC accepts
  // static int a9 __attribute__((weakref));
  // but that looks really pointless. We reject it.
  if (D->hasAttr<WeakRefAttr>() && !D->hasAttr<AliasAttr>()) {
    Diag(AttrList->getLoc(), diag::err_attribute_weakref_without_alias)
      << cast<NamedDecl>(D);
    D->dropAttr<WeakRefAttr>();
    return;
  }

  // FIXME: We should be able to handle this in TableGen as well. It would be
  // good to have a way to specify "these attributes must appear as a group",
  // for these. Additionally, it would be good to have a way to specify "these
  // attribute must never appear as a group" for attributes like cold and hot.
  if (!D->hasAttr<OpenCLKernelAttr>()) {
    // These attributes cannot be applied to a non-kernel function.
    if (Attr *A = D->getAttr<ReqdWorkGroupSizeAttr>()) {
      // FIXME: This emits a different error message than
      // diag::err_attribute_wrong_decl_type + ExpectedKernelFunction.
      Diag(D->getLocation(), diag::err_opencl_kernel_attr) << A;
      D->setInvalidDecl();
    } else if (Attr *A = D->getAttr<WorkGroupSizeHintAttr>()) {
      Diag(D->getLocation(), diag::err_opencl_kernel_attr) << A;
      D->setInvalidDecl();
    } else if (Attr *A = D->getAttr<VecTypeHintAttr>()) {
      Diag(D->getLocation(), diag::err_opencl_kernel_attr) << A;
      D->setInvalidDecl();
    } else if (Attr *A = D->getAttr<AMDGPUFlatWorkGroupSizeAttr>()) {
      Diag(D->getLocation(), diag::err_attribute_wrong_decl_type)
        << A << ExpectedKernelFunction;
      D->setInvalidDecl();
    } else if (Attr *A = D->getAttr<AMDGPUWavesPerEUAttr>()) {
      Diag(D->getLocation(), diag::err_attribute_wrong_decl_type)
        << A << ExpectedKernelFunction;
      D->setInvalidDecl();
    } else if (Attr *A = D->getAttr<AMDGPUNumSGPRAttr>()) {
      Diag(D->getLocation(), diag::err_attribute_wrong_decl_type)
        << A << ExpectedKernelFunction;
      D->setInvalidDecl();
    } else if (Attr *A = D->getAttr<AMDGPUNumVGPRAttr>()) {
      Diag(D->getLocation(), diag::err_attribute_wrong_decl_type)
        << A << ExpectedKernelFunction;
      D->setInvalidDecl();
    } else if (Attr *A = D->getAttr<OpenCLIntelReqdSubGroupSizeAttr>()) {
      Diag(D->getLocation(), diag::err_opencl_kernel_attr) << A;
      D->setInvalidDecl();
    }
  }
}

// Helper for delayed processing TransparentUnion attribute.
void Sema::ProcessDeclAttributeDelayed(Decl *D, const AttributeList *AttrList) {
  for (const AttributeList *Attr = AttrList; Attr; Attr = Attr->getNext())
    if (Attr->getKind() == AttributeList::AT_TransparentUnion) {
      handleTransparentUnionAttr(*this, D, *Attr);
      break;
    }
}

// Annotation attributes are the only attributes allowed after an access
// specifier.
bool Sema::ProcessAccessDeclAttributeList(AccessSpecDecl *ASDecl,
                                          const AttributeList *AttrList) {
  for (const AttributeList* l = AttrList; l; l = l->getNext()) {
    if (l->getKind() == AttributeList::AT_Annotate) {
      ProcessDeclAttribute(*this, nullptr, ASDecl, *l, l->isCXX11Attribute());
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
  NamedDecl *NewD = nullptr;
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
      for (const auto &AI : FT->param_types()) {
        ParmVarDecl *Param = BuildParmVarDeclForTypedef(NewFD, Loc, AI);
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
    NewD->addAttr(AliasAttr::CreateImplicit(Context, NDId->getName(),
                                            W.getLocation()));
    NewD->addAttr(WeakAttr::CreateImplicit(Context, W.getLocation()));
    WeakTopLevelDecl.push_back(NewD);
    // FIXME: "hideous" code from Sema::LazilyCreateBuiltin
    // to insert Decl at TU scope, sorry.
    DeclContext *SavedContext = CurContext;
    CurContext = Context.getTranslationUnitDecl();
    NewD->setDeclContext(CurContext);
    NewD->setLexicalDeclContext(CurContext);
    PushOnScopeChains(NewD, S);
    CurContext = SavedContext;
  } else { // just add weak to existing
    ND->addAttr(WeakAttr::CreateImplicit(Context, W.getLocation()));
  }
}

void Sema::ProcessPragmaWeak(Scope *S, Decl *D) {
  // It's valid to "forward-declare" #pragma weak, in which case we
  // have to do this.
  LoadExternalWeakUndeclaredIdentifiers();
  if (!WeakUndeclaredIdentifiers.empty()) {
    NamedDecl *ND = nullptr;
    if (VarDecl *VD = dyn_cast<VarDecl>(D))
      if (VD->isExternC())
        ND = VD;
    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
      if (FD->isExternC())
        ND = FD;
    if (ND) {
      if (IdentifierInfo *Id = ND->getIdentifier()) {
        auto I = WeakUndeclaredIdentifiers.find(Id);
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
void Sema::ProcessDeclAttributes(Scope *S, Decl *D, const Declarator &PD) {
  // Apply decl attributes from the DeclSpec if present.
  if (const AttributeList *Attrs = PD.getDeclSpec().getAttributes().getList())
    ProcessDeclAttributeList(S, D, Attrs);

  // Walk the declarator structure, applying decl attributes that were in a type
  // position to the decl itself.  This handles cases like:
  //   int *__attr__(x)** D;
  // when X is a decl attribute.
  for (unsigned i = 0, e = PD.getNumTypeObjects(); i != e; ++i)
    if (const AttributeList *Attrs = PD.getTypeObject(i).getAttrs())
      ProcessDeclAttributeList(S, D, Attrs, /*IncludeCXX11Attributes=*/false);

  // Finally, apply any attributes on the decl itself.
  if (const AttributeList *Attrs = PD.getAttributes())
    ProcessDeclAttributeList(S, D, Attrs);

  // Apply additional attributes specified by '#pragma clang attribute'.
  AddPragmaAttributes(S, D);
}

/// Is the given declaration allowed to use a forbidden type?
/// If so, it'll still be annotated with an attribute that makes it
/// illegal to actually use.
static bool isForbiddenTypeAllowed(Sema &S, Decl *decl,
                                   const DelayedDiagnostic &diag,
                                   UnavailableAttr::ImplicitReason &reason) {
  // Private ivars are always okay.  Unfortunately, people don't
  // always properly make their ivars private, even in system headers.
  // Plus we need to make fields okay, too.
  if (!isa<FieldDecl>(decl) && !isa<ObjCPropertyDecl>(decl) &&
      !isa<FunctionDecl>(decl))
    return false;

  // Silently accept unsupported uses of __weak in both user and system
  // declarations when it's been disabled, for ease of integration with
  // -fno-objc-arc files.  We do have to take some care against attempts
  // to define such things;  for now, we've only done that for ivars
  // and properties.
  if ((isa<ObjCIvarDecl>(decl) || isa<ObjCPropertyDecl>(decl))) {
    if (diag.getForbiddenTypeDiagnostic() == diag::err_arc_weak_disabled ||
        diag.getForbiddenTypeDiagnostic() == diag::err_arc_weak_no_runtime) {
      reason = UnavailableAttr::IR_ForbiddenWeak;
      return true;
    }
  }

  // Allow all sorts of things in system headers.
  if (S.Context.getSourceManager().isInSystemHeader(decl->getLocation())) {
    // Currently, all the failures dealt with this way are due to ARC
    // restrictions.
    reason = UnavailableAttr::IR_ARCForbiddenType;
    return true;
  }

  return false;
}

/// Handle a delayed forbidden-type diagnostic.
static void handleDelayedForbiddenType(Sema &S, DelayedDiagnostic &diag,
                                       Decl *decl) {
  auto reason = UnavailableAttr::IR_None;
  if (decl && isForbiddenTypeAllowed(S, decl, diag, reason)) {
    assert(reason && "didn't set reason?");
    decl->addAttr(UnavailableAttr::CreateImplicit(S.Context, "", reason,
                                                  diag.Loc));
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

static const AvailabilityAttr *getAttrForPlatform(ASTContext &Context,
                                                  const Decl *D) {
  // Check each AvailabilityAttr to find the one for this platform.
  for (const auto *A : D->attrs()) {
    if (const auto *Avail = dyn_cast<AvailabilityAttr>(A)) {
      // FIXME: this is copied from CheckAvailability. We should try to
      // de-duplicate.

      // Check if this is an App Extension "platform", and if so chop off
      // the suffix for matching with the actual platform.
      StringRef ActualPlatform = Avail->getPlatform()->getName();
      StringRef RealizedPlatform = ActualPlatform;
      if (Context.getLangOpts().AppExt) {
        size_t suffix = RealizedPlatform.rfind("_app_extension");
        if (suffix != StringRef::npos)
          RealizedPlatform = RealizedPlatform.slice(0, suffix);
      }

      StringRef TargetPlatform = Context.getTargetInfo().getPlatformName();

      // Match the platform name.
      if (RealizedPlatform == TargetPlatform)
        return Avail;
    }
  }
  return nullptr;
}

/// \brief whether we should emit a diagnostic for \c K and \c DeclVersion in
/// the context of \c Ctx. For example, we should emit an unavailable diagnostic
/// in a deprecated context, but not the other way around.
static bool ShouldDiagnoseAvailabilityInContext(Sema &S, AvailabilityResult K,
                                                VersionTuple DeclVersion,
                                                Decl *Ctx) {
  assert(K != AR_Available && "Expected an unavailable declaration here!");

  // Checks if we should emit the availability diagnostic in the context of C.
  auto CheckContext = [&](const Decl *C) {
    if (K == AR_NotYetIntroduced) {
      if (const AvailabilityAttr *AA = getAttrForPlatform(S.Context, C))
        if (AA->getIntroduced() >= DeclVersion)
          return true;
    } else if (K == AR_Deprecated)
      if (C->isDeprecated())
        return true;

    if (C->isUnavailable())
      return true;
    return false;
  };

  // FIXME: This is a temporary workaround! Some existing Apple headers depends
  // on nested declarations in an @interface having the availability of the
  // interface when they really shouldn't: they are members of the enclosing
  // context, and can referenced from there.
  if (S.OriginalLexicalContext && cast<Decl>(S.OriginalLexicalContext) != Ctx) {
    auto *OrigCtx = cast<Decl>(S.OriginalLexicalContext);
    if (CheckContext(OrigCtx))
      return false;

    // An implementation implicitly has the availability of the interface.
    if (auto *CatOrImpl = dyn_cast<ObjCImplDecl>(OrigCtx)) {
      if (const ObjCInterfaceDecl *Interface = CatOrImpl->getClassInterface())
        if (CheckContext(Interface))
          return false;
    }
    // A category implicitly has the availability of the interface.
    else if (auto *CatD = dyn_cast<ObjCCategoryDecl>(OrigCtx))
      if (const ObjCInterfaceDecl *Interface = CatD->getClassInterface())
        if (CheckContext(Interface))
          return false;
  }

  do {
    if (CheckContext(Ctx))
      return false;

    // An implementation implicitly has the availability of the interface.
    if (auto *CatOrImpl = dyn_cast<ObjCImplDecl>(Ctx)) {
      if (const ObjCInterfaceDecl *Interface = CatOrImpl->getClassInterface())
        if (CheckContext(Interface))
          return false;
    }
    // A category implicitly has the availability of the interface.
    else if (auto *CatD = dyn_cast<ObjCCategoryDecl>(Ctx))
      if (const ObjCInterfaceDecl *Interface = CatD->getClassInterface())
        if (CheckContext(Interface))
          return false;
  } while ((Ctx = cast_or_null<Decl>(Ctx->getDeclContext())));

  return true;
}

static bool
shouldDiagnoseAvailabilityByDefault(const ASTContext &Context,
                                    const VersionTuple &DeploymentVersion,
                                    const VersionTuple &DeclVersion) {
  const auto &Triple = Context.getTargetInfo().getTriple();
  VersionTuple ForceAvailabilityFromVersion;
  switch (Triple.getOS()) {
  case llvm::Triple::IOS:
  case llvm::Triple::TvOS:
    ForceAvailabilityFromVersion = VersionTuple(/*Major=*/11);
    break;
  case llvm::Triple::WatchOS:
    ForceAvailabilityFromVersion = VersionTuple(/*Major=*/4);
    break;
  case llvm::Triple::Darwin:
  case llvm::Triple::MacOSX:
    ForceAvailabilityFromVersion = VersionTuple(/*Major=*/10, /*Minor=*/13);
    break;
  default:
    // New targets should always warn about availability.
    return Triple.getVendor() == llvm::Triple::Apple;
  }
  return DeploymentVersion >= ForceAvailabilityFromVersion ||
         DeclVersion >= ForceAvailabilityFromVersion;
}

static NamedDecl *findEnclosingDeclToAnnotate(Decl *OrigCtx) {
  for (Decl *Ctx = OrigCtx; Ctx;
       Ctx = cast_or_null<Decl>(Ctx->getDeclContext())) {
    if (isa<TagDecl>(Ctx) || isa<FunctionDecl>(Ctx) || isa<ObjCMethodDecl>(Ctx))
      return cast<NamedDecl>(Ctx);
    if (auto *CD = dyn_cast<ObjCContainerDecl>(Ctx)) {
      if (auto *Imp = dyn_cast<ObjCImplDecl>(Ctx))
        return Imp->getClassInterface();
      return CD;
    }
  }

  return dyn_cast<NamedDecl>(OrigCtx);
}

/// Actually emit an availability diagnostic for a reference to an unavailable
/// decl.
///
/// \param Ctx The context that the reference occurred in
/// \param ReferringDecl The exact declaration that was referenced.
/// \param OffendingDecl A related decl to \c ReferringDecl that has an
/// availability attribute corrisponding to \c K attached to it. Note that this
/// may not be the same as ReferringDecl, i.e. if an EnumDecl is annotated and
/// we refer to a member EnumConstantDecl, ReferringDecl is the EnumConstantDecl
/// and OffendingDecl is the EnumDecl.
static void DoEmitAvailabilityWarning(Sema &S, AvailabilityResult K,
                                      Decl *Ctx, const NamedDecl *ReferringDecl,
                                      const NamedDecl *OffendingDecl,
                                      StringRef Message, SourceLocation Loc,
                                      const ObjCInterfaceDecl *UnknownObjCClass,
                                      const ObjCPropertyDecl *ObjCProperty,
                                      bool ObjCPropertyAccess) {
  // Diagnostics for deprecated or unavailable.
  unsigned diag, diag_message, diag_fwdclass_message;
  unsigned diag_available_here = diag::note_availability_specified_here;
  SourceLocation NoteLocation = OffendingDecl->getLocation();

  // Matches 'diag::note_property_attribute' options.
  unsigned property_note_select;

  // Matches diag::note_availability_specified_here.
  unsigned available_here_select_kind;

  VersionTuple DeclVersion;
  if (const AvailabilityAttr *AA = getAttrForPlatform(S.Context, OffendingDecl))
    DeclVersion = AA->getIntroduced();

  if (!ShouldDiagnoseAvailabilityInContext(S, K, DeclVersion, Ctx))
    return;

  switch (K) {
  case AR_Deprecated:
    diag = !ObjCPropertyAccess ? diag::warn_deprecated
                               : diag::warn_property_method_deprecated;
    diag_message = diag::warn_deprecated_message;
    diag_fwdclass_message = diag::warn_deprecated_fwdclass_message;
    property_note_select = /* deprecated */ 0;
    available_here_select_kind = /* deprecated */ 2;
    if (const auto *attr = OffendingDecl->getAttr<DeprecatedAttr>())
      NoteLocation = attr->getLocation();
    break;

  case AR_Unavailable:
    diag = !ObjCPropertyAccess ? diag::err_unavailable
                               : diag::err_property_method_unavailable;
    diag_message = diag::err_unavailable_message;
    diag_fwdclass_message = diag::warn_unavailable_fwdclass_message;
    property_note_select = /* unavailable */ 1;
    available_here_select_kind = /* unavailable */ 0;

    if (auto attr = OffendingDecl->getAttr<UnavailableAttr>()) {
      if (attr->isImplicit() && attr->getImplicitReason()) {
        // Most of these failures are due to extra restrictions in ARC;
        // reflect that in the primary diagnostic when applicable.
        auto flagARCError = [&] {
          if (S.getLangOpts().ObjCAutoRefCount &&
              S.getSourceManager().isInSystemHeader(
                  OffendingDecl->getLocation()))
            diag = diag::err_unavailable_in_arc;
        };

        switch (attr->getImplicitReason()) {
        case UnavailableAttr::IR_None: break;

        case UnavailableAttr::IR_ARCForbiddenType:
          flagARCError();
          diag_available_here = diag::note_arc_forbidden_type;
          break;

        case UnavailableAttr::IR_ForbiddenWeak:
          if (S.getLangOpts().ObjCWeakRuntime)
            diag_available_here = diag::note_arc_weak_disabled;
          else
            diag_available_here = diag::note_arc_weak_no_runtime;
          break;

        case UnavailableAttr::IR_ARCForbiddenConversion:
          flagARCError();
          diag_available_here = diag::note_performs_forbidden_arc_conversion;
          break;

        case UnavailableAttr::IR_ARCInitReturnsUnrelated:
          flagARCError();
          diag_available_here = diag::note_arc_init_returns_unrelated;
          break;

        case UnavailableAttr::IR_ARCFieldWithOwnership:
          flagARCError();
          diag_available_here = diag::note_arc_field_with_ownership;
          break;
        }
      }
    }
    break;

  case AR_NotYetIntroduced: {
    // We would like to emit the diagnostic even if -Wunguarded-availability is
    // not specified for deployment targets >= to iOS 11 or equivalent or
    // for declarations that were introduced in iOS 11 (macOS 10.13, ...) or
    // later.
    const AvailabilityAttr *AA =
        getAttrForPlatform(S.getASTContext(), OffendingDecl);
    VersionTuple Introduced = AA->getIntroduced();
    bool NewWarning = shouldDiagnoseAvailabilityByDefault(
        S.Context, S.Context.getTargetInfo().getPlatformMinVersion(),
        Introduced);
    diag = NewWarning ? diag::warn_partial_availability_new
                      : diag::warn_partial_availability;
    diag_message = NewWarning ? diag::warn_partial_message_new
                              : diag::warn_partial_message;
    diag_fwdclass_message = NewWarning ? diag::warn_partial_fwdclass_message_new
                                       : diag::warn_partial_fwdclass_message;
    property_note_select = /* partial */ 2;
    available_here_select_kind = /* partial */ 3;
    break;
  }

  case AR_Available:
    llvm_unreachable("Warning for availability of available declaration?");
  }

  CharSourceRange UseRange;
  StringRef Replacement;
  if (K == AR_Deprecated) {
    if (auto attr = OffendingDecl->getAttr<DeprecatedAttr>())
      Replacement = attr->getReplacement();
    if (auto attr = getAttrForPlatform(S.Context, OffendingDecl))
      Replacement = attr->getReplacement();

    if (!Replacement.empty())
      UseRange =
          CharSourceRange::getCharRange(Loc, S.getLocForEndOfToken(Loc));
  }

  if (!Message.empty()) {
    S.Diag(Loc, diag_message) << ReferringDecl << Message
      << (UseRange.isValid() ?
          FixItHint::CreateReplacement(UseRange, Replacement) : FixItHint());
    if (ObjCProperty)
      S.Diag(ObjCProperty->getLocation(), diag::note_property_attribute)
          << ObjCProperty->getDeclName() << property_note_select;
  } else if (!UnknownObjCClass) {
    S.Diag(Loc, diag) << ReferringDecl
      << (UseRange.isValid() ?
          FixItHint::CreateReplacement(UseRange, Replacement) : FixItHint());
    if (ObjCProperty)
      S.Diag(ObjCProperty->getLocation(), diag::note_property_attribute)
          << ObjCProperty->getDeclName() << property_note_select;
  } else {
    S.Diag(Loc, diag_fwdclass_message) << ReferringDecl
      << (UseRange.isValid() ?
          FixItHint::CreateReplacement(UseRange, Replacement) : FixItHint());
    S.Diag(UnknownObjCClass->getLocation(), diag::note_forward_class);
  }

  // The declaration can have multiple availability attributes, we are looking
  // at one of them.
  const AvailabilityAttr *A = getAttrForPlatform(S.Context, OffendingDecl);
  if (A && A->isInherited()) {
    for (const Decl *Redecl = OffendingDecl->getMostRecentDecl(); Redecl;
         Redecl = Redecl->getPreviousDecl()) {
      const AvailabilityAttr *AForRedecl = getAttrForPlatform(S.Context,
                                                              Redecl);
      if (AForRedecl && !AForRedecl->isInherited()) {
        // If D is a declaration with inherited attributes, the note should
        // point to the declaration with actual attributes.
        S.Diag(Redecl->getLocation(), diag_available_here) << OffendingDecl
            << available_here_select_kind;
        break;
      }
    }
  }
  else
    S.Diag(NoteLocation, diag_available_here)
        << OffendingDecl << available_here_select_kind;

  if (K == AR_NotYetIntroduced)
    if (const auto *Enclosing = findEnclosingDeclToAnnotate(Ctx)) {
      if (auto *TD = dyn_cast<TagDecl>(Enclosing))
        if (TD->getDeclName().isEmpty()) {
          S.Diag(TD->getLocation(), diag::note_partial_availability_silence)
              << /*Anonymous*/1 << TD->getKindName();
          return;
        }
      S.Diag(Enclosing->getLocation(), diag::note_partial_availability_silence)
          << /*Named*/0 << Enclosing;
    }
}

static void handleDelayedAvailabilityCheck(Sema &S, DelayedDiagnostic &DD,
                                           Decl *Ctx) {
  assert(DD.Kind == DelayedDiagnostic::Availability &&
         "Expected an availability diagnostic here");

  DD.Triggered = true;
  DoEmitAvailabilityWarning(
      S, DD.getAvailabilityResult(), Ctx, DD.getAvailabilityReferringDecl(),
      DD.getAvailabilityOffendingDecl(), DD.getAvailabilityMessage(), DD.Loc,
      DD.getUnknownObjCClass(), DD.getObjCProperty(), false);
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
      case DelayedDiagnostic::Availability:
        // Don't bother giving deprecation/unavailable diagnostics if
        // the decl is invalid.
        if (!decl->isInvalidDecl())
          handleDelayedAvailabilityCheck(*this, diag, decl);
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

void Sema::EmitAvailabilityWarning(AvailabilityResult AR,
                                   const NamedDecl *ReferringDecl,
                                   const NamedDecl *OffendingDecl,
                                   StringRef Message, SourceLocation Loc,
                                   const ObjCInterfaceDecl *UnknownObjCClass,
                                   const ObjCPropertyDecl *ObjCProperty,
                                   bool ObjCPropertyAccess) {
  // Delay if we're currently parsing a declaration.
  if (DelayedDiagnostics.shouldDelayDiagnostics()) {
    DelayedDiagnostics.add(
        DelayedDiagnostic::makeAvailability(
            AR, Loc, ReferringDecl, OffendingDecl, UnknownObjCClass,
            ObjCProperty, Message, ObjCPropertyAccess));
    return;
  }

  Decl *Ctx = cast<Decl>(getCurLexicalContext());
  DoEmitAvailabilityWarning(*this, AR, Ctx, ReferringDecl, OffendingDecl,
                            Message, Loc, UnknownObjCClass, ObjCProperty,
                            ObjCPropertyAccess);
}

namespace {

/// Returns true if the given statement can be a body-like child of \p Parent.
bool isBodyLikeChildStmt(const Stmt *S, const Stmt *Parent) {
  switch (Parent->getStmtClass()) {
  case Stmt::IfStmtClass:
    return cast<IfStmt>(Parent)->getThen() == S ||
           cast<IfStmt>(Parent)->getElse() == S;
  case Stmt::WhileStmtClass:
    return cast<WhileStmt>(Parent)->getBody() == S;
  case Stmt::DoStmtClass:
    return cast<DoStmt>(Parent)->getBody() == S;
  case Stmt::ForStmtClass:
    return cast<ForStmt>(Parent)->getBody() == S;
  case Stmt::CXXForRangeStmtClass:
    return cast<CXXForRangeStmt>(Parent)->getBody() == S;
  case Stmt::ObjCForCollectionStmtClass:
    return cast<ObjCForCollectionStmt>(Parent)->getBody() == S;
  case Stmt::CaseStmtClass:
  case Stmt::DefaultStmtClass:
    return cast<SwitchCase>(Parent)->getSubStmt() == S;
  default:
    return false;
  }
}

class StmtUSEFinder : public RecursiveASTVisitor<StmtUSEFinder> {
  const Stmt *Target;

public:
  bool VisitStmt(Stmt *S) { return S != Target; }

  /// Returns true if the given statement is present in the given declaration.
  static bool isContained(const Stmt *Target, const Decl *D) {
    StmtUSEFinder Visitor;
    Visitor.Target = Target;
    return !Visitor.TraverseDecl(const_cast<Decl *>(D));
  }
};

/// Traverses the AST and finds the last statement that used a given
/// declaration.
class LastDeclUSEFinder : public RecursiveASTVisitor<LastDeclUSEFinder> {
  const Decl *D;

public:
  bool VisitDeclRefExpr(DeclRefExpr *DRE) {
    if (DRE->getDecl() == D)
      return false;
    return true;
  }

  static const Stmt *findLastStmtThatUsesDecl(const Decl *D,
                                              const CompoundStmt *Scope) {
    LastDeclUSEFinder Visitor;
    Visitor.D = D;
    for (auto I = Scope->body_rbegin(), E = Scope->body_rend(); I != E; ++I) {
      const Stmt *S = *I;
      if (!Visitor.TraverseStmt(const_cast<Stmt *>(S)))
        return S;
    }
    return nullptr;
  }
};

/// \brief This class implements -Wunguarded-availability.
///
/// This is done with a traversal of the AST of a function that makes reference
/// to a partially available declaration. Whenever we encounter an \c if of the
/// form: \c if(@available(...)), we use the version from the condition to visit
/// the then statement.
class DiagnoseUnguardedAvailability
    : public RecursiveASTVisitor<DiagnoseUnguardedAvailability> {
  typedef RecursiveASTVisitor<DiagnoseUnguardedAvailability> Base;

  Sema &SemaRef;
  Decl *Ctx;

  /// Stack of potentially nested 'if (@available(...))'s.
  SmallVector<VersionTuple, 8> AvailabilityStack;
  SmallVector<const Stmt *, 16> StmtStack;

  void DiagnoseDeclAvailability(NamedDecl *D, SourceRange Range);

public:
  DiagnoseUnguardedAvailability(Sema &SemaRef, Decl *Ctx)
      : SemaRef(SemaRef), Ctx(Ctx) {
    AvailabilityStack.push_back(
        SemaRef.Context.getTargetInfo().getPlatformMinVersion());
  }

  bool TraverseDecl(Decl *D) {
    // Avoid visiting nested functions to prevent duplicate warnings.
    if (!D || isa<FunctionDecl>(D))
      return true;
    return Base::TraverseDecl(D);
  }

  bool TraverseStmt(Stmt *S) {
    if (!S)
      return true;
    StmtStack.push_back(S);
    bool Result = Base::TraverseStmt(S);
    StmtStack.pop_back();
    return Result;
  }

  void IssueDiagnostics(Stmt *S) { TraverseStmt(S); }

  bool TraverseIfStmt(IfStmt *If);

  bool TraverseLambdaExpr(LambdaExpr *E) { return true; }

  bool VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *PRE) {
    if (PRE->isClassReceiver())
      DiagnoseDeclAvailability(PRE->getClassReceiver(), PRE->getReceiverLocation());
    return true;
  }

  bool VisitObjCMessageExpr(ObjCMessageExpr *Msg) {
    if (ObjCMethodDecl *D = Msg->getMethodDecl())
      DiagnoseDeclAvailability(
          D, SourceRange(Msg->getSelectorStartLoc(), Msg->getLocEnd()));
    return true;
  }

  bool VisitDeclRefExpr(DeclRefExpr *DRE) {
    DiagnoseDeclAvailability(DRE->getDecl(),
                             SourceRange(DRE->getLocStart(), DRE->getLocEnd()));
    return true;
  }

  bool VisitMemberExpr(MemberExpr *ME) {
    DiagnoseDeclAvailability(ME->getMemberDecl(),
                             SourceRange(ME->getLocStart(), ME->getLocEnd()));
    return true;
  }

  bool VisitObjCAvailabilityCheckExpr(ObjCAvailabilityCheckExpr *E) {
    SemaRef.Diag(E->getLocStart(), diag::warn_at_available_unchecked_use)
        << (!SemaRef.getLangOpts().ObjC1);
    return true;
  }

  bool VisitTypeLoc(TypeLoc Ty);
};

void DiagnoseUnguardedAvailability::DiagnoseDeclAvailability(
    NamedDecl *D, SourceRange Range) {
  AvailabilityResult Result;
  const NamedDecl *OffendingDecl;
  std::tie(Result, OffendingDecl) =
      SemaRef.ShouldDiagnoseAvailabilityOfDecl(D, nullptr);
  if (Result != AR_Available) {
    // All other diagnostic kinds have already been handled in
    // DiagnoseAvailabilityOfDecl.
    if (Result != AR_NotYetIntroduced)
      return;

    const AvailabilityAttr *AA =
      getAttrForPlatform(SemaRef.getASTContext(), OffendingDecl);
    VersionTuple Introduced = AA->getIntroduced();

    if (AvailabilityStack.back() >= Introduced)
      return;

    // If the context of this function is less available than D, we should not
    // emit a diagnostic.
    if (!ShouldDiagnoseAvailabilityInContext(SemaRef, Result, Introduced, Ctx))
      return;

    // We would like to emit the diagnostic even if -Wunguarded-availability is
    // not specified for deployment targets >= to iOS 11 or equivalent or
    // for declarations that were introduced in iOS 11 (macOS 10.13, ...) or
    // later.
    unsigned DiagKind =
        shouldDiagnoseAvailabilityByDefault(
            SemaRef.Context,
            SemaRef.Context.getTargetInfo().getPlatformMinVersion(), Introduced)
            ? diag::warn_unguarded_availability_new
            : diag::warn_unguarded_availability;

    SemaRef.Diag(Range.getBegin(), DiagKind)
        << Range << D
        << AvailabilityAttr::getPrettyPlatformName(
               SemaRef.getASTContext().getTargetInfo().getPlatformName())
        << Introduced.getAsString();

    SemaRef.Diag(OffendingDecl->getLocation(),
                 diag::note_availability_specified_here)
        << OffendingDecl << /* partial */ 3;

    auto FixitDiag =
        SemaRef.Diag(Range.getBegin(), diag::note_unguarded_available_silence)
        << Range << D
        << (SemaRef.getLangOpts().ObjC1 ? /*@available*/ 0
                                        : /*__builtin_available*/ 1);

    // Find the statement which should be enclosed in the if @available check.
    if (StmtStack.empty())
      return;
    const Stmt *StmtOfUse = StmtStack.back();
    const CompoundStmt *Scope = nullptr;
    for (const Stmt *S : llvm::reverse(StmtStack)) {
      if (const auto *CS = dyn_cast<CompoundStmt>(S)) {
        Scope = CS;
        break;
      }
      if (isBodyLikeChildStmt(StmtOfUse, S)) {
        // The declaration won't be seen outside of the statement, so we don't
        // have to wrap the uses of any declared variables in if (@available).
        // Therefore we can avoid setting Scope here.
        break;
      }
      StmtOfUse = S;
    }
    const Stmt *LastStmtOfUse = nullptr;
    if (isa<DeclStmt>(StmtOfUse) && Scope) {
      for (const Decl *D : cast<DeclStmt>(StmtOfUse)->decls()) {
        if (StmtUSEFinder::isContained(StmtStack.back(), D)) {
          LastStmtOfUse = LastDeclUSEFinder::findLastStmtThatUsesDecl(D, Scope);
          break;
        }
      }
    }

    const SourceManager &SM = SemaRef.getSourceManager();
    SourceLocation IfInsertionLoc =
        SM.getExpansionLoc(StmtOfUse->getLocStart());
    SourceLocation StmtEndLoc =
        SM.getExpansionRange(
              (LastStmtOfUse ? LastStmtOfUse : StmtOfUse)->getLocEnd())
            .second;
    if (SM.getFileID(IfInsertionLoc) != SM.getFileID(StmtEndLoc))
      return;

    StringRef Indentation = Lexer::getIndentationForLine(IfInsertionLoc, SM);
    const char *ExtraIndentation = "    ";
    std::string FixItString;
    llvm::raw_string_ostream FixItOS(FixItString);
    FixItOS << "if (" << (SemaRef.getLangOpts().ObjC1 ? "@available"
                                                      : "__builtin_available")
            << "("
            << AvailabilityAttr::getPlatformNameSourceSpelling(
                   SemaRef.getASTContext().getTargetInfo().getPlatformName())
            << " " << Introduced.getAsString() << ", *)) {\n"
            << Indentation << ExtraIndentation;
    FixitDiag << FixItHint::CreateInsertion(IfInsertionLoc, FixItOS.str());
    SourceLocation ElseInsertionLoc = Lexer::findLocationAfterToken(
        StmtEndLoc, tok::semi, SM, SemaRef.getLangOpts(),
        /*SkipTrailingWhitespaceAndNewLine=*/false);
    if (ElseInsertionLoc.isInvalid())
      ElseInsertionLoc =
          Lexer::getLocForEndOfToken(StmtEndLoc, 0, SM, SemaRef.getLangOpts());
    FixItOS.str().clear();
    FixItOS << "\n"
            << Indentation << "} else {\n"
            << Indentation << ExtraIndentation
            << "// Fallback on earlier versions\n"
            << Indentation << "}";
    FixitDiag << FixItHint::CreateInsertion(ElseInsertionLoc, FixItOS.str());
  }
}

bool DiagnoseUnguardedAvailability::VisitTypeLoc(TypeLoc Ty) {
  const Type *TyPtr = Ty.getTypePtr();
  SourceRange Range{Ty.getBeginLoc(), Ty.getEndLoc()};

  if (Range.isInvalid())
    return true;

  if (const TagType *TT = dyn_cast<TagType>(TyPtr)) {
    TagDecl *TD = TT->getDecl();
    DiagnoseDeclAvailability(TD, Range);

  } else if (const TypedefType *TD = dyn_cast<TypedefType>(TyPtr)) {
    TypedefNameDecl *D = TD->getDecl();
    DiagnoseDeclAvailability(D, Range);

  } else if (const auto *ObjCO = dyn_cast<ObjCObjectType>(TyPtr)) {
    if (NamedDecl *D = ObjCO->getInterface())
      DiagnoseDeclAvailability(D, Range);
  }

  return true;
}

bool DiagnoseUnguardedAvailability::TraverseIfStmt(IfStmt *If) {
  VersionTuple CondVersion;
  if (auto *E = dyn_cast<ObjCAvailabilityCheckExpr>(If->getCond())) {
    CondVersion = E->getVersion();

    // If we're using the '*' case here or if this check is redundant, then we
    // use the enclosing version to check both branches.
    if (CondVersion.empty() || CondVersion <= AvailabilityStack.back())
      return Base::TraverseStmt(If->getThen()) &&
             Base::TraverseStmt(If->getElse());
  } else {
    // This isn't an availability checking 'if', we can just continue.
    return Base::TraverseIfStmt(If);
  }

  AvailabilityStack.push_back(CondVersion);
  bool ShouldContinue = TraverseStmt(If->getThen());
  AvailabilityStack.pop_back();

  return ShouldContinue && TraverseStmt(If->getElse());
}

} // end anonymous namespace

void Sema::DiagnoseUnguardedAvailabilityViolations(Decl *D) {
  Stmt *Body = nullptr;

  if (auto *FD = D->getAsFunction()) {
    // FIXME: We only examine the pattern decl for availability violations now,
    // but we should also examine instantiated templates.
    if (FD->isTemplateInstantiation())
      return;

    Body = FD->getBody();
  } else if (auto *MD = dyn_cast<ObjCMethodDecl>(D))
    Body = MD->getBody();
  else if (auto *BD = dyn_cast<BlockDecl>(D))
    Body = BD->getBody();

  assert(Body && "Need a body here!");

  DiagnoseUnguardedAvailability(*this, D).IssueDiagnostics(Body);
}
