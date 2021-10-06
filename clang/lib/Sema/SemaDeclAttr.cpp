//===--- SemaDeclAttr.cpp - Declaration Attribute Handling ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "clang/AST/Type.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/DarwinSDKInfo.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/DelayedDiagnostic.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/SemaInternal.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Assumptions.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

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

/// Return true if the given decl has function type (function or
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
  if (const auto *BD = dyn_cast<BlockDecl>(D))
    return BD->getNumParams();
  return cast<ObjCMethodDecl>(D)->param_size();
}

static const ParmVarDecl *getFunctionOrMethodParam(const Decl *D,
                                                   unsigned Idx) {
  if (const auto *FD = dyn_cast<FunctionDecl>(D))
    return FD->getParamDecl(Idx);
  if (const auto *MD = dyn_cast<ObjCMethodDecl>(D))
    return MD->getParamDecl(Idx);
  if (const auto *BD = dyn_cast<BlockDecl>(D))
    return BD->getParamDecl(Idx);
  return nullptr;
}

static QualType getFunctionOrMethodParamType(const Decl *D, unsigned Idx) {
  if (const FunctionType *FnTy = D->getFunctionType())
    return cast<FunctionProtoType>(FnTy)->getParamType(Idx);
  if (const auto *BD = dyn_cast<BlockDecl>(D))
    return BD->getParamDecl(Idx)->getType();

  return cast<ObjCMethodDecl>(D)->parameters()[Idx]->getType();
}

static SourceRange getFunctionOrMethodParamRange(const Decl *D, unsigned Idx) {
  if (auto *PVD = getFunctionOrMethodParam(D, Idx))
    return PVD->getSourceRange();
  return SourceRange();
}

static QualType getFunctionOrMethodResultType(const Decl *D) {
  if (const FunctionType *FnTy = D->getFunctionType())
    return FnTy->getReturnType();
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
  if (const FunctionType *FnTy = D->getFunctionType())
    return cast<FunctionProtoType>(FnTy)->isVariadic();
  if (const auto *BD = dyn_cast<BlockDecl>(D))
    return BD->isVariadic();
  return cast<ObjCMethodDecl>(D)->isVariadic();
}

static bool isInstanceMethod(const Decl *D) {
  if (const auto *MethodDecl = dyn_cast<CXXMethodDecl>(D))
    return MethodDecl->isInstance();
  return false;
}

static inline bool isNSStringType(QualType T, ASTContext &Ctx,
                                  bool AllowNSAttributedString = false) {
  const auto *PT = T->getAs<ObjCObjectPointerType>();
  if (!PT)
    return false;

  ObjCInterfaceDecl *Cls = PT->getObjectType()->getInterface();
  if (!Cls)
    return false;

  IdentifierInfo* ClsName = Cls->getIdentifier();

  if (AllowNSAttributedString &&
      ClsName == &Ctx.Idents.get("NSAttributedString"))
    return true;
  // FIXME: Should we walk the chain of classes?
  return ClsName == &Ctx.Idents.get("NSString") ||
         ClsName == &Ctx.Idents.get("NSMutableString");
}

static inline bool isCFStringType(QualType T, ASTContext &Ctx) {
  const auto *PT = T->getAs<PointerType>();
  if (!PT)
    return false;

  const auto *RT = PT->getPointeeType()->getAs<RecordType>();
  if (!RT)
    return false;

  const RecordDecl *RD = RT->getDecl();
  if (RD->getTagKind() != TTK_Struct)
    return false;

  return RD->getIdentifier() == &Ctx.Idents.get("__CFString");
}

static unsigned getNumAttributeArgs(const ParsedAttr &AL) {
  // FIXME: Include the type in the argument list.
  return AL.getNumArgs() + AL.hasParsedType();
}

/// A helper function to provide Attribute Location for the Attr types
/// AND the ParsedAttr.
template <typename AttrInfo>
static std::enable_if_t<std::is_base_of<Attr, AttrInfo>::value, SourceLocation>
getAttrLoc(const AttrInfo &AL) {
  return AL.getLocation();
}
static SourceLocation getAttrLoc(const ParsedAttr &AL) { return AL.getLoc(); }

/// If Expr is a valid integer constant, get the value of the integer
/// expression and return success or failure. May output an error.
///
/// Negative argument is implicitly converted to unsigned, unless
/// \p StrictlyUnsigned is true.
template <typename AttrInfo>
static bool checkUInt32Argument(Sema &S, const AttrInfo &AI, const Expr *Expr,
                                uint32_t &Val, unsigned Idx = UINT_MAX,
                                bool StrictlyUnsigned = false) {
  Optional<llvm::APSInt> I = llvm::APSInt(32);
  if (Expr->isTypeDependent() || Expr->isValueDependent() ||
      !(I = Expr->getIntegerConstantExpr(S.Context))) {
    if (Idx != UINT_MAX)
      S.Diag(getAttrLoc(AI), diag::err_attribute_argument_n_type)
          << &AI << Idx << AANT_ArgumentIntegerConstant
          << Expr->getSourceRange();
    else
      S.Diag(getAttrLoc(AI), diag::err_attribute_argument_type)
          << &AI << AANT_ArgumentIntegerConstant << Expr->getSourceRange();
    return false;
  }

  if (!I->isIntN(32)) {
    S.Diag(Expr->getExprLoc(), diag::err_ice_too_large)
        << toString(*I, 10, false) << 32 << /* Unsigned */ 1;
    return false;
  }

  if (StrictlyUnsigned && I->isSigned() && I->isNegative()) {
    S.Diag(getAttrLoc(AI), diag::err_attribute_requires_positive_integer)
        << &AI << /*non-negative*/ 1;
    return false;
  }

  Val = (uint32_t)I->getZExtValue();
  return true;
}

/// Wrapper around checkUInt32Argument, with an extra check to be sure
/// that the result will fit into a regular (signed) int. All args have the same
/// purpose as they do in checkUInt32Argument.
template <typename AttrInfo>
static bool checkPositiveIntArgument(Sema &S, const AttrInfo &AI, const Expr *Expr,
                                     int &Val, unsigned Idx = UINT_MAX) {
  uint32_t UVal;
  if (!checkUInt32Argument(S, AI, Expr, UVal, Idx))
    return false;

  if (UVal > (uint32_t)std::numeric_limits<int>::max()) {
    llvm::APSInt I(32); // for toString
    I = UVal;
    S.Diag(Expr->getExprLoc(), diag::err_ice_too_large)
        << toString(I, 10, false) << 32 << /* Unsigned */ 0;
    return false;
  }

  Val = UVal;
  return true;
}

/// Diagnose mutually exclusive attributes when present on a given
/// declaration. Returns true if diagnosed.
template <typename AttrTy>
static bool checkAttrMutualExclusion(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (const auto *A = D->getAttr<AttrTy>()) {
    S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A;
    S.Diag(A->getLocation(), diag::note_conflicting_attribute);
    return true;
  }
  return false;
}

template <typename AttrTy>
static bool checkAttrMutualExclusion(Sema &S, Decl *D, const Attr &AL) {
  if (const auto *A = D->getAttr<AttrTy>()) {
    S.Diag(AL.getLocation(), diag::err_attributes_are_not_compatible) << &AL
                                                                      << A;
    S.Diag(A->getLocation(), diag::note_conflicting_attribute);
    return true;
  }
  return false;
}

/// Check if IdxExpr is a valid parameter index for a function or
/// instance method D.  May output an error.
///
/// \returns true if IdxExpr is a valid index.
template <typename AttrInfo>
static bool checkFunctionOrMethodParameterIndex(
    Sema &S, const Decl *D, const AttrInfo &AI, unsigned AttrArgNum,
    const Expr *IdxExpr, ParamIdx &Idx, bool CanIndexImplicitThis = false) {
  assert(isFunctionOrMethodOrBlock(D));

  // In C++ the implicit 'this' function parameter also counts.
  // Parameters are counted from one.
  bool HP = hasFunctionProto(D);
  bool HasImplicitThisParam = isInstanceMethod(D);
  bool IV = HP && isFunctionOrMethodVariadic(D);
  unsigned NumParams =
      (HP ? getFunctionOrMethodNumParams(D) : 0) + HasImplicitThisParam;

  Optional<llvm::APSInt> IdxInt;
  if (IdxExpr->isTypeDependent() || IdxExpr->isValueDependent() ||
      !(IdxInt = IdxExpr->getIntegerConstantExpr(S.Context))) {
    S.Diag(getAttrLoc(AI), diag::err_attribute_argument_n_type)
        << &AI << AttrArgNum << AANT_ArgumentIntegerConstant
        << IdxExpr->getSourceRange();
    return false;
  }

  unsigned IdxSource = IdxInt->getLimitedValue(UINT_MAX);
  if (IdxSource < 1 || (!IV && IdxSource > NumParams)) {
    S.Diag(getAttrLoc(AI), diag::err_attribute_argument_out_of_bounds)
        << &AI << AttrArgNum << IdxExpr->getSourceRange();
    return false;
  }
  if (HasImplicitThisParam && !CanIndexImplicitThis) {
    if (IdxSource == 1) {
      S.Diag(getAttrLoc(AI), diag::err_attribute_invalid_implicit_this_argument)
          << &AI << IdxExpr->getSourceRange();
      return false;
    }
  }

  Idx = ParamIdx(IdxSource, D);
  return true;
}

/// Check if the argument \p ArgNum of \p Attr is a ASCII string literal.
/// If not emit an error and return false. If the argument is an identifier it
/// will emit an error with a fixit hint and treat it as if it was a string
/// literal.
bool Sema::checkStringLiteralArgumentAttr(const ParsedAttr &AL, unsigned ArgNum,
                                          StringRef &Str,
                                          SourceLocation *ArgLocation) {
  // Look for identifiers. If we have one emit a hint to fix it to a literal.
  if (AL.isArgIdent(ArgNum)) {
    IdentifierLoc *Loc = AL.getArgAsIdent(ArgNum);
    Diag(Loc->Loc, diag::err_attribute_argument_type)
        << AL << AANT_ArgumentString
        << FixItHint::CreateInsertion(Loc->Loc, "\"")
        << FixItHint::CreateInsertion(getLocForEndOfToken(Loc->Loc), "\"");
    Str = Loc->Ident->getName();
    if (ArgLocation)
      *ArgLocation = Loc->Loc;
    return true;
  }

  // Now check for an actual string literal.
  Expr *ArgExpr = AL.getArgAsExpr(ArgNum);
  const auto *Literal = dyn_cast<StringLiteral>(ArgExpr->IgnoreParenCasts());
  if (ArgLocation)
    *ArgLocation = ArgExpr->getBeginLoc();

  if (!Literal || !Literal->isAscii()) {
    Diag(ArgExpr->getBeginLoc(), diag::err_attribute_argument_type)
        << AL << AANT_ArgumentString;
    return false;
  }

  Str = Literal->getString();
  return true;
}

/// Applies the given attribute to the Decl without performing any
/// additional semantic checking.
template <typename AttrType>
static void handleSimpleAttribute(Sema &S, Decl *D,
                                  const AttributeCommonInfo &CI) {
  D->addAttr(::new (S.Context) AttrType(S.Context, CI));
}

template <typename... DiagnosticArgs>
static const Sema::SemaDiagnosticBuilder&
appendDiagnostics(const Sema::SemaDiagnosticBuilder &Bldr) {
  return Bldr;
}

template <typename T, typename... DiagnosticArgs>
static const Sema::SemaDiagnosticBuilder&
appendDiagnostics(const Sema::SemaDiagnosticBuilder &Bldr, T &&ExtraArg,
                  DiagnosticArgs &&... ExtraArgs) {
  return appendDiagnostics(Bldr << std::forward<T>(ExtraArg),
                           std::forward<DiagnosticArgs>(ExtraArgs)...);
}

/// Add an attribute @c AttrType to declaration @c D, provided that
/// @c PassesCheck is true.
/// Otherwise, emit diagnostic @c DiagID, passing in all parameters
/// specified in @c ExtraArgs.
template <typename AttrType, typename... DiagnosticArgs>
static void handleSimpleAttributeOrDiagnose(Sema &S, Decl *D,
                                            const AttributeCommonInfo &CI,
                                            bool PassesCheck, unsigned DiagID,
                                            DiagnosticArgs &&... ExtraArgs) {
  if (!PassesCheck) {
    Sema::SemaDiagnosticBuilder DB = S.Diag(D->getBeginLoc(), DiagID);
    appendDiagnostics(DB, std::forward<DiagnosticArgs>(ExtraArgs)...);
    return;
  }
  handleSimpleAttribute<AttrType>(S, D, CI);
}

/// Check if the passed-in expression is of type int or bool.
static bool isIntOrBool(Expr *Exp) {
  QualType QT = Exp->getType();
  return QT->isBooleanType() || QT->isIntegerType();
}


// Check to see if the type is a smart pointer of some kind.  We assume
// it's a smart pointer if it defines both operator-> and operator*.
static bool threadSafetyCheckIsSmartPointer(Sema &S, const RecordType* RT) {
  auto IsOverloadedOperatorPresent = [&S](const RecordDecl *Record,
                                          OverloadedOperatorKind Op) {
    DeclContextLookupResult Result =
        Record->lookup(S.Context.DeclarationNames.getCXXOperatorName(Op));
    return !Result.empty();
  };

  const RecordDecl *Record = RT->getDecl();
  bool foundStarOperator = IsOverloadedOperatorPresent(Record, OO_Star);
  bool foundArrowOperator = IsOverloadedOperatorPresent(Record, OO_Arrow);
  if (foundStarOperator && foundArrowOperator)
    return true;

  const CXXRecordDecl *CXXRecord = dyn_cast<CXXRecordDecl>(Record);
  if (!CXXRecord)
    return false;

  for (auto BaseSpecifier : CXXRecord->bases()) {
    if (!foundStarOperator)
      foundStarOperator = IsOverloadedOperatorPresent(
          BaseSpecifier.getType()->getAsRecordDecl(), OO_Star);
    if (!foundArrowOperator)
      foundArrowOperator = IsOverloadedOperatorPresent(
          BaseSpecifier.getType()->getAsRecordDecl(), OO_Arrow);
  }

  if (foundStarOperator && foundArrowOperator)
    return true;

  return false;
}

/// Check if passed in Decl is a pointer type.
/// Note that this function may produce an error message.
/// \return true if the Decl is a pointer type; false otherwise
static bool threadSafetyCheckIsPointer(Sema &S, const Decl *D,
                                       const ParsedAttr &AL) {
  const auto *VD = cast<ValueDecl>(D);
  QualType QT = VD->getType();
  if (QT->isAnyPointerType())
    return true;

  if (const auto *RT = QT->getAs<RecordType>()) {
    // If it's an incomplete type, it could be a smart pointer; skip it.
    // (We don't want to force template instantiation if we can avoid it,
    // since that would alter the order in which templates are instantiated.)
    if (RT->isIncompleteType())
      return true;

    if (threadSafetyCheckIsSmartPointer(S, RT))
      return true;
  }

  S.Diag(AL.getLoc(), diag::warn_thread_attribute_decl_not_pointer) << AL << QT;
  return false;
}

/// Checks that the passed in QualType either is of RecordType or points
/// to RecordType. Returns the relevant RecordType, null if it does not exit.
static const RecordType *getRecordType(QualType QT) {
  if (const auto *RT = QT->getAs<RecordType>())
    return RT;

  // Now check if we point to record type.
  if (const auto *PT = QT->getAs<PointerType>())
    return PT->getPointeeType()->getAs<RecordType>();

  return nullptr;
}

template <typename AttrType>
static bool checkRecordDeclForAttr(const RecordDecl *RD) {
  // Check if the record itself has the attribute.
  if (RD->hasAttr<AttrType>())
    return true;

  // Else check if any base classes have the attribute.
  if (const auto *CRD = dyn_cast<CXXRecordDecl>(RD)) {
    if (!CRD->forallBases([](const CXXRecordDecl *Base) {
          return !Base->hasAttr<AttrType>();
        }))
      return true;
  }
  return false;
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

  return checkRecordDeclForAttr<CapabilityAttr>(RT->getDecl());
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

  if (const auto *E = dyn_cast<CastExpr>(Ex))
    return isCapabilityExpr(S, E->getSubExpr());
  else if (const auto *E = dyn_cast<ParenExpr>(Ex))
    return isCapabilityExpr(S, E->getSubExpr());
  else if (const auto *E = dyn_cast<UnaryOperator>(Ex)) {
    if (E->getOpcode() == UO_LNot || E->getOpcode() == UO_AddrOf ||
        E->getOpcode() == UO_Deref)
      return isCapabilityExpr(S, E->getSubExpr());
    return false;
  } else if (const auto *E = dyn_cast<BinaryOperator>(Ex)) {
    if (E->getOpcode() == BO_LAnd || E->getOpcode() == BO_LOr)
      return isCapabilityExpr(S, E->getLHS()) &&
             isCapabilityExpr(S, E->getRHS());
    return false;
  }

  return typeHasCapability(S, Ex->getType());
}

/// Checks that all attribute arguments, starting from Sidx, resolve to
/// a capability object.
/// \param Sidx The attribute argument index to start checking with.
/// \param ParamIdxOk Whether an argument can be indexing into a function
/// parameter list.
static void checkAttrArgsAreCapabilityObjs(Sema &S, Decl *D,
                                           const ParsedAttr &AL,
                                           SmallVectorImpl<Expr *> &Args,
                                           unsigned Sidx = 0,
                                           bool ParamIdxOk = false) {
  if (Sidx == AL.getNumArgs()) {
    // If we don't have any capability arguments, the attribute implicitly
    // refers to 'this'. So we need to make sure that 'this' exists, i.e. we're
    // a non-static method, and that the class is a (scoped) capability.
    const auto *MD = dyn_cast<const CXXMethodDecl>(D);
    if (MD && !MD->isStatic()) {
      const CXXRecordDecl *RD = MD->getParent();
      // FIXME -- need to check this again on template instantiation
      if (!checkRecordDeclForAttr<CapabilityAttr>(RD) &&
          !checkRecordDeclForAttr<ScopedLockableAttr>(RD))
        S.Diag(AL.getLoc(),
               diag::warn_thread_attribute_not_on_capability_member)
            << AL << MD->getParent();
    } else {
      S.Diag(AL.getLoc(), diag::warn_thread_attribute_not_on_non_static_member)
          << AL;
    }
  }

  for (unsigned Idx = Sidx; Idx < AL.getNumArgs(); ++Idx) {
    Expr *ArgExp = AL.getArgAsExpr(Idx);

    if (ArgExp->isTypeDependent()) {
      // FIXME -- need to check this again on template instantiation
      Args.push_back(ArgExp);
      continue;
    }

    if (const auto *StrLit = dyn_cast<StringLiteral>(ArgExp)) {
      if (StrLit->getLength() == 0 ||
          (StrLit->isAscii() && StrLit->getString() == StringRef("*"))) {
        // Pass empty strings to the analyzer without warnings.
        // Treat "*" as the universal lock.
        Args.push_back(ArgExp);
        continue;
      }

      // We allow constant strings to be used as a placeholder for expressions
      // that are not valid C++ syntax, but warn that they are ignored.
      S.Diag(AL.getLoc(), diag::warn_thread_attribute_ignored) << AL;
      Args.push_back(ArgExp);
      continue;
    }

    QualType ArgTy = ArgExp->getType();

    // A pointer to member expression of the form  &MyClass::mu is treated
    // specially -- we need to look at the type of the member.
    if (const auto *UOp = dyn_cast<UnaryOperator>(ArgExp))
      if (UOp->getOpcode() == UO_AddrOf)
        if (const auto *DRE = dyn_cast<DeclRefExpr>(UOp->getSubExpr()))
          if (DRE->getDecl()->isCXXInstanceMember())
            ArgTy = DRE->getDecl()->getType();

    // First see if we can just cast to record type, or pointer to record type.
    const RecordType *RT = getRecordType(ArgTy);

    // Now check if we index into a record type function param.
    if(!RT && ParamIdxOk) {
      const auto *FD = dyn_cast<FunctionDecl>(D);
      const auto *IL = dyn_cast<IntegerLiteral>(ArgExp);
      if(FD && IL) {
        unsigned int NumParams = FD->getNumParams();
        llvm::APInt ArgValue = IL->getValue();
        uint64_t ParamIdxFromOne = ArgValue.getZExtValue();
        uint64_t ParamIdxFromZero = ParamIdxFromOne - 1;
        if (!ArgValue.isStrictlyPositive() || ParamIdxFromOne > NumParams) {
          S.Diag(AL.getLoc(),
                 diag::err_attribute_argument_out_of_bounds_extra_info)
              << AL << Idx + 1 << NumParams;
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
      S.Diag(AL.getLoc(), diag::warn_thread_attribute_argument_not_lockable)
          << AL << ArgTy;

    Args.push_back(ArgExp);
  }
}

//===----------------------------------------------------------------------===//
// Attribute Implementations
//===----------------------------------------------------------------------===//

static void handlePtGuardedVarAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (!threadSafetyCheckIsPointer(S, D, AL))
    return;

  D->addAttr(::new (S.Context) PtGuardedVarAttr(S.Context, AL));
}

static bool checkGuardedByAttrCommon(Sema &S, Decl *D, const ParsedAttr &AL,
                                     Expr *&Arg) {
  SmallVector<Expr *, 1> Args;
  // check that all arguments are lockable objects
  checkAttrArgsAreCapabilityObjs(S, D, AL, Args);
  unsigned Size = Args.size();
  if (Size != 1)
    return false;

  Arg = Args[0];

  return true;
}

static void handleGuardedByAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  Expr *Arg = nullptr;
  if (!checkGuardedByAttrCommon(S, D, AL, Arg))
    return;

  D->addAttr(::new (S.Context) GuardedByAttr(S.Context, AL, Arg));
}

static void handlePtGuardedByAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  Expr *Arg = nullptr;
  if (!checkGuardedByAttrCommon(S, D, AL, Arg))
    return;

  if (!threadSafetyCheckIsPointer(S, D, AL))
    return;

  D->addAttr(::new (S.Context) PtGuardedByAttr(S.Context, AL, Arg));
}

static bool checkAcquireOrderAttrCommon(Sema &S, Decl *D, const ParsedAttr &AL,
                                        SmallVectorImpl<Expr *> &Args) {
  if (!AL.checkAtLeastNumArgs(S, 1))
    return false;

  // Check that this attribute only applies to lockable types.
  QualType QT = cast<ValueDecl>(D)->getType();
  if (!QT->isDependentType() && !typeHasCapability(S, QT)) {
    S.Diag(AL.getLoc(), diag::warn_thread_attribute_decl_not_lockable) << AL;
    return false;
  }

  // Check that all arguments are lockable objects.
  checkAttrArgsAreCapabilityObjs(S, D, AL, Args);
  if (Args.empty())
    return false;

  return true;
}

static void handleAcquiredAfterAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  SmallVector<Expr *, 1> Args;
  if (!checkAcquireOrderAttrCommon(S, D, AL, Args))
    return;

  Expr **StartArg = &Args[0];
  D->addAttr(::new (S.Context)
                 AcquiredAfterAttr(S.Context, AL, StartArg, Args.size()));
}

static void handleAcquiredBeforeAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  SmallVector<Expr *, 1> Args;
  if (!checkAcquireOrderAttrCommon(S, D, AL, Args))
    return;

  Expr **StartArg = &Args[0];
  D->addAttr(::new (S.Context)
                 AcquiredBeforeAttr(S.Context, AL, StartArg, Args.size()));
}

static bool checkLockFunAttrCommon(Sema &S, Decl *D, const ParsedAttr &AL,
                                   SmallVectorImpl<Expr *> &Args) {
  // zero or more arguments ok
  // check that all arguments are lockable objects
  checkAttrArgsAreCapabilityObjs(S, D, AL, Args, 0, /*ParamIdxOk=*/true);

  return true;
}

static void handleAssertSharedLockAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  SmallVector<Expr *, 1> Args;
  if (!checkLockFunAttrCommon(S, D, AL, Args))
    return;

  unsigned Size = Args.size();
  Expr **StartArg = Size == 0 ? nullptr : &Args[0];
  D->addAttr(::new (S.Context)
                 AssertSharedLockAttr(S.Context, AL, StartArg, Size));
}

static void handleAssertExclusiveLockAttr(Sema &S, Decl *D,
                                          const ParsedAttr &AL) {
  SmallVector<Expr *, 1> Args;
  if (!checkLockFunAttrCommon(S, D, AL, Args))
    return;

  unsigned Size = Args.size();
  Expr **StartArg = Size == 0 ? nullptr : &Args[0];
  D->addAttr(::new (S.Context)
                 AssertExclusiveLockAttr(S.Context, AL, StartArg, Size));
}

/// Checks to be sure that the given parameter number is in bounds, and
/// is an integral type. Will emit appropriate diagnostics if this returns
/// false.
///
/// AttrArgNo is used to actually retrieve the argument, so it's base-0.
template <typename AttrInfo>
static bool checkParamIsIntegerType(Sema &S, const Decl *D, const AttrInfo &AI,
                                    unsigned AttrArgNo) {
  assert(AI.isArgExpr(AttrArgNo) && "Expected expression argument");
  Expr *AttrArg = AI.getArgAsExpr(AttrArgNo);
  ParamIdx Idx;
  if (!checkFunctionOrMethodParameterIndex(S, D, AI, AttrArgNo + 1, AttrArg,
                                           Idx))
    return false;

  QualType ParamTy = getFunctionOrMethodParamType(D, Idx.getASTIndex());
  if (!ParamTy->isIntegerType() && !ParamTy->isCharType()) {
    SourceLocation SrcLoc = AttrArg->getBeginLoc();
    S.Diag(SrcLoc, diag::err_attribute_integers_only)
        << AI << getFunctionOrMethodParamRange(D, Idx.getASTIndex());
    return false;
  }
  return true;
}

static void handleAllocSizeAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (!AL.checkAtLeastNumArgs(S, 1) || !AL.checkAtMostNumArgs(S, 2))
    return;

  assert(isFunctionOrMethod(D) && hasFunctionProto(D));

  QualType RetTy = getFunctionOrMethodResultType(D);
  if (!RetTy->isPointerType()) {
    S.Diag(AL.getLoc(), diag::warn_attribute_return_pointers_only) << AL;
    return;
  }

  const Expr *SizeExpr = AL.getArgAsExpr(0);
  int SizeArgNoVal;
  // Parameter indices are 1-indexed, hence Index=1
  if (!checkPositiveIntArgument(S, AL, SizeExpr, SizeArgNoVal, /*Idx=*/1))
    return;
  if (!checkParamIsIntegerType(S, D, AL, /*AttrArgNo=*/0))
    return;
  ParamIdx SizeArgNo(SizeArgNoVal, D);

  ParamIdx NumberArgNo;
  if (AL.getNumArgs() == 2) {
    const Expr *NumberExpr = AL.getArgAsExpr(1);
    int Val;
    // Parameter indices are 1-based, hence Index=2
    if (!checkPositiveIntArgument(S, AL, NumberExpr, Val, /*Idx=*/2))
      return;
    if (!checkParamIsIntegerType(S, D, AL, /*AttrArgNo=*/1))
      return;
    NumberArgNo = ParamIdx(Val, D);
  }

  D->addAttr(::new (S.Context)
                 AllocSizeAttr(S.Context, AL, SizeArgNo, NumberArgNo));
}

static bool checkTryLockFunAttrCommon(Sema &S, Decl *D, const ParsedAttr &AL,
                                      SmallVectorImpl<Expr *> &Args) {
  if (!AL.checkAtLeastNumArgs(S, 1))
    return false;

  if (!isIntOrBool(AL.getArgAsExpr(0))) {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_n_type)
        << AL << 1 << AANT_ArgumentIntOrBool;
    return false;
  }

  // check that all arguments are lockable objects
  checkAttrArgsAreCapabilityObjs(S, D, AL, Args, 1);

  return true;
}

static void handleSharedTrylockFunctionAttr(Sema &S, Decl *D,
                                            const ParsedAttr &AL) {
  SmallVector<Expr*, 2> Args;
  if (!checkTryLockFunAttrCommon(S, D, AL, Args))
    return;

  D->addAttr(::new (S.Context) SharedTrylockFunctionAttr(
      S.Context, AL, AL.getArgAsExpr(0), Args.data(), Args.size()));
}

static void handleExclusiveTrylockFunctionAttr(Sema &S, Decl *D,
                                               const ParsedAttr &AL) {
  SmallVector<Expr*, 2> Args;
  if (!checkTryLockFunAttrCommon(S, D, AL, Args))
    return;

  D->addAttr(::new (S.Context) ExclusiveTrylockFunctionAttr(
      S.Context, AL, AL.getArgAsExpr(0), Args.data(), Args.size()));
}

static void handleLockReturnedAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  // check that the argument is lockable object
  SmallVector<Expr*, 1> Args;
  checkAttrArgsAreCapabilityObjs(S, D, AL, Args);
  unsigned Size = Args.size();
  if (Size == 0)
    return;

  D->addAttr(::new (S.Context) LockReturnedAttr(S.Context, AL, Args[0]));
}

static void handleLocksExcludedAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (!AL.checkAtLeastNumArgs(S, 1))
    return;

  // check that all arguments are lockable objects
  SmallVector<Expr*, 1> Args;
  checkAttrArgsAreCapabilityObjs(S, D, AL, Args);
  unsigned Size = Args.size();
  if (Size == 0)
    return;
  Expr **StartArg = &Args[0];

  D->addAttr(::new (S.Context)
                 LocksExcludedAttr(S.Context, AL, StartArg, Size));
}

static bool checkFunctionConditionAttr(Sema &S, Decl *D, const ParsedAttr &AL,
                                       Expr *&Cond, StringRef &Msg) {
  Cond = AL.getArgAsExpr(0);
  if (!Cond->isTypeDependent()) {
    ExprResult Converted = S.PerformContextuallyConvertToBool(Cond);
    if (Converted.isInvalid())
      return false;
    Cond = Converted.get();
  }

  if (!S.checkStringLiteralArgumentAttr(AL, 1, Msg))
    return false;

  if (Msg.empty())
    Msg = "<no message provided>";

  SmallVector<PartialDiagnosticAt, 8> Diags;
  if (isa<FunctionDecl>(D) && !Cond->isValueDependent() &&
      !Expr::isPotentialConstantExprUnevaluated(Cond, cast<FunctionDecl>(D),
                                                Diags)) {
    S.Diag(AL.getLoc(), diag::err_attr_cond_never_constant_expr) << AL;
    for (const PartialDiagnosticAt &PDiag : Diags)
      S.Diag(PDiag.first, PDiag.second);
    return false;
  }
  return true;
}

static void handleEnableIfAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  S.Diag(AL.getLoc(), diag::ext_clang_enable_if);

  Expr *Cond;
  StringRef Msg;
  if (checkFunctionConditionAttr(S, D, AL, Cond, Msg))
    D->addAttr(::new (S.Context) EnableIfAttr(S.Context, AL, Cond, Msg));
}

static void handleErrorAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  StringRef NewUserDiagnostic;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, NewUserDiagnostic))
    return;
  if (ErrorAttr *EA = S.mergeErrorAttr(D, AL, NewUserDiagnostic))
    D->addAttr(EA);
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

static void handleDiagnoseIfAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  S.Diag(AL.getLoc(), diag::ext_clang_diagnose_if);

  Expr *Cond;
  StringRef Msg;
  if (!checkFunctionConditionAttr(S, D, AL, Cond, Msg))
    return;

  StringRef DiagTypeStr;
  if (!S.checkStringLiteralArgumentAttr(AL, 2, DiagTypeStr))
    return;

  DiagnoseIfAttr::DiagnosticType DiagType;
  if (!DiagnoseIfAttr::ConvertStrToDiagnosticType(DiagTypeStr, DiagType)) {
    S.Diag(AL.getArgAsExpr(2)->getBeginLoc(),
           diag::err_diagnose_if_invalid_diagnostic_type);
    return;
  }

  bool ArgDependent = false;
  if (const auto *FD = dyn_cast<FunctionDecl>(D))
    ArgDependent = ArgumentDependenceChecker(FD).referencesArgs(Cond);
  D->addAttr(::new (S.Context) DiagnoseIfAttr(
      S.Context, AL, Cond, Msg, DiagType, ArgDependent, cast<NamedDecl>(D)));
}

static void handleNoBuiltinAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  static constexpr const StringRef kWildcard = "*";

  llvm::SmallVector<StringRef, 16> Names;
  bool HasWildcard = false;

  const auto AddBuiltinName = [&Names, &HasWildcard](StringRef Name) {
    if (Name == kWildcard)
      HasWildcard = true;
    Names.push_back(Name);
  };

  // Add previously defined attributes.
  if (const auto *NBA = D->getAttr<NoBuiltinAttr>())
    for (StringRef BuiltinName : NBA->builtinNames())
      AddBuiltinName(BuiltinName);

  // Add current attributes.
  if (AL.getNumArgs() == 0)
    AddBuiltinName(kWildcard);
  else
    for (unsigned I = 0, E = AL.getNumArgs(); I != E; ++I) {
      StringRef BuiltinName;
      SourceLocation LiteralLoc;
      if (!S.checkStringLiteralArgumentAttr(AL, I, BuiltinName, &LiteralLoc))
        return;

      if (Builtin::Context::isBuiltinFunc(BuiltinName))
        AddBuiltinName(BuiltinName);
      else
        S.Diag(LiteralLoc, diag::warn_attribute_no_builtin_invalid_builtin_name)
            << BuiltinName << AL;
    }

  // Repeating the same attribute is fine.
  llvm::sort(Names);
  Names.erase(std::unique(Names.begin(), Names.end()), Names.end());

  // Empty no_builtin must be on its own.
  if (HasWildcard && Names.size() > 1)
    S.Diag(D->getLocation(),
           diag::err_attribute_no_builtin_wildcard_or_builtin_name)
        << AL;

  if (D->hasAttr<NoBuiltinAttr>())
    D->dropAttr<NoBuiltinAttr>();
  D->addAttr(::new (S.Context)
                 NoBuiltinAttr(S.Context, AL, Names.data(), Names.size()));
}

static void handlePassObjectSizeAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (D->hasAttr<PassObjectSizeAttr>()) {
    S.Diag(D->getBeginLoc(), diag::err_attribute_only_once_per_parameter) << AL;
    return;
  }

  Expr *E = AL.getArgAsExpr(0);
  uint32_t Type;
  if (!checkUInt32Argument(S, AL, E, Type, /*Idx=*/1))
    return;

  // pass_object_size's argument is passed in as the second argument of
  // __builtin_object_size. So, it has the same constraints as that second
  // argument; namely, it must be in the range [0, 3].
  if (Type > 3) {
    S.Diag(E->getBeginLoc(), diag::err_attribute_argument_out_of_range)
        << AL << 0 << 3 << E->getSourceRange();
    return;
  }

  // pass_object_size is only supported on constant pointer parameters; as a
  // kindness to users, we allow the parameter to be non-const for declarations.
  // At this point, we have no clue if `D` belongs to a function declaration or
  // definition, so we defer the constness check until later.
  if (!cast<ParmVarDecl>(D)->getType()->isPointerType()) {
    S.Diag(D->getBeginLoc(), diag::err_attribute_pointers_only) << AL << 1;
    return;
  }

  D->addAttr(::new (S.Context) PassObjectSizeAttr(S.Context, AL, (int)Type));
}

static void handleConsumableAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  ConsumableAttr::ConsumedState DefaultState;

  if (AL.isArgIdent(0)) {
    IdentifierLoc *IL = AL.getArgAsIdent(0);
    if (!ConsumableAttr::ConvertStrToConsumedState(IL->Ident->getName(),
                                                   DefaultState)) {
      S.Diag(IL->Loc, diag::warn_attribute_type_not_supported) << AL
                                                               << IL->Ident;
      return;
    }
  } else {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_type)
        << AL << AANT_ArgumentIdentifier;
    return;
  }

  D->addAttr(::new (S.Context) ConsumableAttr(S.Context, AL, DefaultState));
}

static bool checkForConsumableClass(Sema &S, const CXXMethodDecl *MD,
                                    const ParsedAttr &AL) {
  QualType ThisType = MD->getThisType()->getPointeeType();

  if (const CXXRecordDecl *RD = ThisType->getAsCXXRecordDecl()) {
    if (!RD->hasAttr<ConsumableAttr>()) {
      S.Diag(AL.getLoc(), diag::warn_attr_on_unconsumable_class) << RD;

      return false;
    }
  }

  return true;
}

static void handleCallableWhenAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (!AL.checkAtLeastNumArgs(S, 1))
    return;

  if (!checkForConsumableClass(S, cast<CXXMethodDecl>(D), AL))
    return;

  SmallVector<CallableWhenAttr::ConsumedState, 3> States;
  for (unsigned ArgIndex = 0; ArgIndex < AL.getNumArgs(); ++ArgIndex) {
    CallableWhenAttr::ConsumedState CallableState;

    StringRef StateString;
    SourceLocation Loc;
    if (AL.isArgIdent(ArgIndex)) {
      IdentifierLoc *Ident = AL.getArgAsIdent(ArgIndex);
      StateString = Ident->Ident->getName();
      Loc = Ident->Loc;
    } else {
      if (!S.checkStringLiteralArgumentAttr(AL, ArgIndex, StateString, &Loc))
        return;
    }

    if (!CallableWhenAttr::ConvertStrToConsumedState(StateString,
                                                     CallableState)) {
      S.Diag(Loc, diag::warn_attribute_type_not_supported) << AL << StateString;
      return;
    }

    States.push_back(CallableState);
  }

  D->addAttr(::new (S.Context)
                 CallableWhenAttr(S.Context, AL, States.data(), States.size()));
}

static void handleParamTypestateAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  ParamTypestateAttr::ConsumedState ParamState;

  if (AL.isArgIdent(0)) {
    IdentifierLoc *Ident = AL.getArgAsIdent(0);
    StringRef StateString = Ident->Ident->getName();

    if (!ParamTypestateAttr::ConvertStrToConsumedState(StateString,
                                                       ParamState)) {
      S.Diag(Ident->Loc, diag::warn_attribute_type_not_supported)
          << AL << StateString;
      return;
    }
  } else {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_type)
        << AL << AANT_ArgumentIdentifier;
    return;
  }

  // FIXME: This check is currently being done in the analysis.  It can be
  //        enabled here only after the parser propagates attributes at
  //        template specialization definition, not declaration.
  //QualType ReturnType = cast<ParmVarDecl>(D)->getType();
  //const CXXRecordDecl *RD = ReturnType->getAsCXXRecordDecl();
  //
  //if (!RD || !RD->hasAttr<ConsumableAttr>()) {
  //    S.Diag(AL.getLoc(), diag::warn_return_state_for_unconsumable_type) <<
  //      ReturnType.getAsString();
  //    return;
  //}

  D->addAttr(::new (S.Context) ParamTypestateAttr(S.Context, AL, ParamState));
}

static void handleReturnTypestateAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  ReturnTypestateAttr::ConsumedState ReturnState;

  if (AL.isArgIdent(0)) {
    IdentifierLoc *IL = AL.getArgAsIdent(0);
    if (!ReturnTypestateAttr::ConvertStrToConsumedState(IL->Ident->getName(),
                                                        ReturnState)) {
      S.Diag(IL->Loc, diag::warn_attribute_type_not_supported) << AL
                                                               << IL->Ident;
      return;
    }
  } else {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_type)
        << AL << AANT_ArgumentIdentifier;
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
  //  ReturnType = Constructor->getThisType()->getPointeeType();
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

  D->addAttr(::new (S.Context) ReturnTypestateAttr(S.Context, AL, ReturnState));
}

static void handleSetTypestateAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (!checkForConsumableClass(S, cast<CXXMethodDecl>(D), AL))
    return;

  SetTypestateAttr::ConsumedState NewState;
  if (AL.isArgIdent(0)) {
    IdentifierLoc *Ident = AL.getArgAsIdent(0);
    StringRef Param = Ident->Ident->getName();
    if (!SetTypestateAttr::ConvertStrToConsumedState(Param, NewState)) {
      S.Diag(Ident->Loc, diag::warn_attribute_type_not_supported) << AL
                                                                  << Param;
      return;
    }
  } else {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_type)
        << AL << AANT_ArgumentIdentifier;
    return;
  }

  D->addAttr(::new (S.Context) SetTypestateAttr(S.Context, AL, NewState));
}

static void handleTestTypestateAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (!checkForConsumableClass(S, cast<CXXMethodDecl>(D), AL))
    return;

  TestTypestateAttr::ConsumedState TestState;
  if (AL.isArgIdent(0)) {
    IdentifierLoc *Ident = AL.getArgAsIdent(0);
    StringRef Param = Ident->Ident->getName();
    if (!TestTypestateAttr::ConvertStrToConsumedState(Param, TestState)) {
      S.Diag(Ident->Loc, diag::warn_attribute_type_not_supported) << AL
                                                                  << Param;
      return;
    }
  } else {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_type)
        << AL << AANT_ArgumentIdentifier;
    return;
  }

  D->addAttr(::new (S.Context) TestTypestateAttr(S.Context, AL, TestState));
}

static void handleExtVectorTypeAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  // Remember this typedef decl, we will need it later for diagnostics.
  S.ExtVectorDecls.push_back(cast<TypedefNameDecl>(D));
}

static void handlePackedAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (auto *TD = dyn_cast<TagDecl>(D))
    TD->addAttr(::new (S.Context) PackedAttr(S.Context, AL));
  else if (auto *FD = dyn_cast<FieldDecl>(D)) {
    bool BitfieldByteAligned = (!FD->getType()->isDependentType() &&
                                !FD->getType()->isIncompleteType() &&
                                FD->isBitField() &&
                                S.Context.getTypeAlign(FD->getType()) <= 8);

    if (S.getASTContext().getTargetInfo().getTriple().isPS4()) {
      if (BitfieldByteAligned)
        // The PS4 target needs to maintain ABI backwards compatibility.
        S.Diag(AL.getLoc(), diag::warn_attribute_ignored_for_field_of_type)
            << AL << FD->getType();
      else
        FD->addAttr(::new (S.Context) PackedAttr(S.Context, AL));
    } else {
      // Report warning about changed offset in the newer compiler versions.
      if (BitfieldByteAligned)
        S.Diag(AL.getLoc(), diag::warn_attribute_packed_for_bitfield);

      FD->addAttr(::new (S.Context) PackedAttr(S.Context, AL));
    }

  } else
    S.Diag(AL.getLoc(), diag::warn_attribute_ignored) << AL;
}

static void handlePreferredName(Sema &S, Decl *D, const ParsedAttr &AL) {
  auto *RD = cast<CXXRecordDecl>(D);
  ClassTemplateDecl *CTD = RD->getDescribedClassTemplate();
  assert(CTD && "attribute does not appertain to this declaration");

  ParsedType PT = AL.getTypeArg();
  TypeSourceInfo *TSI = nullptr;
  QualType T = S.GetTypeFromParser(PT, &TSI);
  if (!TSI)
    TSI = S.Context.getTrivialTypeSourceInfo(T, AL.getLoc());

  if (!T.hasQualifiers() && T->isTypedefNameType()) {
    // Find the template name, if this type names a template specialization.
    const TemplateDecl *Template = nullptr;
    if (const auto *CTSD = dyn_cast_or_null<ClassTemplateSpecializationDecl>(
            T->getAsCXXRecordDecl())) {
      Template = CTSD->getSpecializedTemplate();
    } else if (const auto *TST = T->getAs<TemplateSpecializationType>()) {
      while (TST && TST->isTypeAlias())
        TST = TST->getAliasedType()->getAs<TemplateSpecializationType>();
      if (TST)
        Template = TST->getTemplateName().getAsTemplateDecl();
    }

    if (Template && declaresSameEntity(Template, CTD)) {
      D->addAttr(::new (S.Context) PreferredNameAttr(S.Context, AL, TSI));
      return;
    }
  }

  S.Diag(AL.getLoc(), diag::err_attribute_preferred_name_arg_invalid)
      << T << CTD;
  if (const auto *TT = T->getAs<TypedefType>())
    S.Diag(TT->getDecl()->getLocation(), diag::note_entity_declared_at)
        << TT->getDecl();
}

static bool checkIBOutletCommon(Sema &S, Decl *D, const ParsedAttr &AL) {
  // The IBOutlet/IBOutletCollection attributes only apply to instance
  // variables or properties of Objective-C classes.  The outlet must also
  // have an object reference type.
  if (const auto *VD = dyn_cast<ObjCIvarDecl>(D)) {
    if (!VD->getType()->getAs<ObjCObjectPointerType>()) {
      S.Diag(AL.getLoc(), diag::warn_iboutlet_object_type)
          << AL << VD->getType() << 0;
      return false;
    }
  }
  else if (const auto *PD = dyn_cast<ObjCPropertyDecl>(D)) {
    if (!PD->getType()->getAs<ObjCObjectPointerType>()) {
      S.Diag(AL.getLoc(), diag::warn_iboutlet_object_type)
          << AL << PD->getType() << 1;
      return false;
    }
  }
  else {
    S.Diag(AL.getLoc(), diag::warn_attribute_iboutlet) << AL;
    return false;
  }

  return true;
}

static void handleIBOutlet(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (!checkIBOutletCommon(S, D, AL))
    return;

  D->addAttr(::new (S.Context) IBOutletAttr(S.Context, AL));
}

static void handleIBOutletCollection(Sema &S, Decl *D, const ParsedAttr &AL) {

  // The iboutletcollection attribute can have zero or one arguments.
  if (AL.getNumArgs() > 1) {
    S.Diag(AL.getLoc(), diag::err_attribute_wrong_number_arguments) << AL << 1;
    return;
  }

  if (!checkIBOutletCommon(S, D, AL))
    return;

  ParsedType PT;

  if (AL.hasParsedType())
    PT = AL.getTypeArg();
  else {
    PT = S.getTypeName(S.Context.Idents.get("NSObject"), AL.getLoc(),
                       S.getScopeForContext(D->getDeclContext()->getParent()));
    if (!PT) {
      S.Diag(AL.getLoc(), diag::err_iboutletcollection_type) << "NSObject";
      return;
    }
  }

  TypeSourceInfo *QTLoc = nullptr;
  QualType QT = S.GetTypeFromParser(PT, &QTLoc);
  if (!QTLoc)
    QTLoc = S.Context.getTrivialTypeSourceInfo(QT, AL.getLoc());

  // Diagnose use of non-object type in iboutletcollection attribute.
  // FIXME. Gnu attribute extension ignores use of builtin types in
  // attributes. So, __attribute__((iboutletcollection(char))) will be
  // treated as __attribute__((iboutletcollection())).
  if (!QT->isObjCIdType() && !QT->isObjCObjectType()) {
    S.Diag(AL.getLoc(),
           QT->isBuiltinType() ? diag::err_iboutletcollection_builtintype
                               : diag::err_iboutletcollection_type) << QT;
    return;
  }

  D->addAttr(::new (S.Context) IBOutletCollectionAttr(S.Context, AL, QTLoc));
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

static bool attrNonNullArgCheck(Sema &S, QualType T, const ParsedAttr &AL,
                                SourceRange AttrParmRange,
                                SourceRange TypeRange,
                                bool isReturnValue = false) {
  if (!S.isValidPointerAttrType(T)) {
    if (isReturnValue)
      S.Diag(AL.getLoc(), diag::warn_attribute_return_pointers_only)
          << AL << AttrParmRange << TypeRange;
    else
      S.Diag(AL.getLoc(), diag::warn_attribute_pointers_only)
          << AL << AttrParmRange << TypeRange << 0;
    return false;
  }
  return true;
}

static void handleNonNullAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  SmallVector<ParamIdx, 8> NonNullArgs;
  for (unsigned I = 0; I < AL.getNumArgs(); ++I) {
    Expr *Ex = AL.getArgAsExpr(I);
    ParamIdx Idx;
    if (!checkFunctionOrMethodParameterIndex(S, D, AL, I + 1, Ex, Idx))
      return;

    // Is the function argument a pointer type?
    if (Idx.getASTIndex() < getFunctionOrMethodNumParams(D) &&
        !attrNonNullArgCheck(
            S, getFunctionOrMethodParamType(D, Idx.getASTIndex()), AL,
            Ex->getSourceRange(),
            getFunctionOrMethodParamRange(D, Idx.getASTIndex())))
      continue;

    NonNullArgs.push_back(Idx);
  }

  // If no arguments were specified to __attribute__((nonnull)) then all pointer
  // arguments have a nonnull attribute; warn if there aren't any. Skip this
  // check if the attribute came from a macro expansion or a template
  // instantiation.
  if (NonNullArgs.empty() && AL.getLoc().isFileID() &&
      !S.inTemplateInstantiation()) {
    bool AnyPointers = isFunctionOrMethodVariadic(D);
    for (unsigned I = 0, E = getFunctionOrMethodNumParams(D);
         I != E && !AnyPointers; ++I) {
      QualType T = getFunctionOrMethodParamType(D, I);
      if (T->isDependentType() || S.isValidPointerAttrType(T))
        AnyPointers = true;
    }

    if (!AnyPointers)
      S.Diag(AL.getLoc(), diag::warn_attribute_nonnull_no_pointers);
  }

  ParamIdx *Start = NonNullArgs.data();
  unsigned Size = NonNullArgs.size();
  llvm::array_pod_sort(Start, Start + Size);
  D->addAttr(::new (S.Context) NonNullAttr(S.Context, AL, Start, Size));
}

static void handleNonNullAttrParameter(Sema &S, ParmVarDecl *D,
                                       const ParsedAttr &AL) {
  if (AL.getNumArgs() > 0) {
    if (D->getFunctionType()) {
      handleNonNullAttr(S, D, AL);
    } else {
      S.Diag(AL.getLoc(), diag::warn_attribute_nonnull_parm_no_args)
        << D->getSourceRange();
    }
    return;
  }

  // Is the argument a pointer type?
  if (!attrNonNullArgCheck(S, D->getType(), AL, SourceRange(),
                           D->getSourceRange()))
    return;

  D->addAttr(::new (S.Context) NonNullAttr(S.Context, AL, nullptr, 0));
}

static void handleReturnsNonNullAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  QualType ResultType = getFunctionOrMethodResultType(D);
  SourceRange SR = getFunctionOrMethodResultSourceRange(D);
  if (!attrNonNullArgCheck(S, ResultType, AL, SourceRange(), SR,
                           /* isReturnValue */ true))
    return;

  D->addAttr(::new (S.Context) ReturnsNonNullAttr(S.Context, AL));
}

static void handleNoEscapeAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (D->isInvalidDecl())
    return;

  // noescape only applies to pointer types.
  QualType T = cast<ParmVarDecl>(D)->getType();
  if (!S.isValidPointerAttrType(T, /* RefOkay */ true)) {
    S.Diag(AL.getLoc(), diag::warn_attribute_pointers_only)
        << AL << AL.getRange() << 0;
    return;
  }

  D->addAttr(::new (S.Context) NoEscapeAttr(S.Context, AL));
}

static void handleAssumeAlignedAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  Expr *E = AL.getArgAsExpr(0),
       *OE = AL.getNumArgs() > 1 ? AL.getArgAsExpr(1) : nullptr;
  S.AddAssumeAlignedAttr(D, AL, E, OE);
}

static void handleAllocAlignAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  S.AddAllocAlignAttr(D, AL, AL.getArgAsExpr(0));
}

void Sema::AddAssumeAlignedAttr(Decl *D, const AttributeCommonInfo &CI, Expr *E,
                                Expr *OE) {
  QualType ResultType = getFunctionOrMethodResultType(D);
  SourceRange SR = getFunctionOrMethodResultSourceRange(D);

  AssumeAlignedAttr TmpAttr(Context, CI, E, OE);
  SourceLocation AttrLoc = TmpAttr.getLocation();

  if (!isValidPointerAttrType(ResultType, /* RefOkay */ true)) {
    Diag(AttrLoc, diag::warn_attribute_return_pointers_refs_only)
        << &TmpAttr << TmpAttr.getRange() << SR;
    return;
  }

  if (!E->isValueDependent()) {
    Optional<llvm::APSInt> I = llvm::APSInt(64);
    if (!(I = E->getIntegerConstantExpr(Context))) {
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

    if (!I->isPowerOf2()) {
      Diag(AttrLoc, diag::err_alignment_not_power_of_two)
        << E->getSourceRange();
      return;
    }

    if (*I > Sema::MaximumAlignment)
      Diag(CI.getLoc(), diag::warn_assume_aligned_too_great)
          << CI.getRange() << Sema::MaximumAlignment;
  }

  if (OE && !OE->isValueDependent() && !OE->isIntegerConstantExpr(Context)) {
    Diag(AttrLoc, diag::err_attribute_argument_n_type)
        << &TmpAttr << 2 << AANT_ArgumentIntegerConstant
        << OE->getSourceRange();
    return;
  }

  D->addAttr(::new (Context) AssumeAlignedAttr(Context, CI, E, OE));
}

void Sema::AddAllocAlignAttr(Decl *D, const AttributeCommonInfo &CI,
                             Expr *ParamExpr) {
  QualType ResultType = getFunctionOrMethodResultType(D);

  AllocAlignAttr TmpAttr(Context, CI, ParamIdx());
  SourceLocation AttrLoc = CI.getLoc();

  if (!ResultType->isDependentType() &&
      !isValidPointerAttrType(ResultType, /* RefOkay */ true)) {
    Diag(AttrLoc, diag::warn_attribute_return_pointers_refs_only)
        << &TmpAttr << CI.getRange() << getFunctionOrMethodResultSourceRange(D);
    return;
  }

  ParamIdx Idx;
  const auto *FuncDecl = cast<FunctionDecl>(D);
  if (!checkFunctionOrMethodParameterIndex(*this, FuncDecl, TmpAttr,
                                           /*AttrArgNum=*/1, ParamExpr, Idx))
    return;

  QualType Ty = getFunctionOrMethodParamType(D, Idx.getASTIndex());
  if (!Ty->isDependentType() && !Ty->isIntegralType(Context) &&
      !Ty->isAlignValT()) {
    Diag(ParamExpr->getBeginLoc(), diag::err_attribute_integers_only)
        << &TmpAttr
        << FuncDecl->getParamDecl(Idx.getASTIndex())->getSourceRange();
    return;
  }

  D->addAttr(::new (Context) AllocAlignAttr(Context, CI, Idx));
}

/// Check if \p AssumptionStr is a known assumption and warn if not.
static void checkAssumptionAttr(Sema &S, SourceLocation Loc,
                                StringRef AssumptionStr) {
  if (llvm::KnownAssumptionStrings.count(AssumptionStr))
    return;

  unsigned BestEditDistance = 3;
  StringRef Suggestion;
  for (const auto &KnownAssumptionIt : llvm::KnownAssumptionStrings) {
    unsigned EditDistance =
        AssumptionStr.edit_distance(KnownAssumptionIt.getKey());
    if (EditDistance < BestEditDistance) {
      Suggestion = KnownAssumptionIt.getKey();
      BestEditDistance = EditDistance;
    }
  }

  if (!Suggestion.empty())
    S.Diag(Loc, diag::warn_assume_attribute_string_unknown_suggested)
        << AssumptionStr << Suggestion;
  else
    S.Diag(Loc, diag::warn_assume_attribute_string_unknown) << AssumptionStr;
}

static void handleAssumumptionAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  // Handle the case where the attribute has a text message.
  StringRef Str;
  SourceLocation AttrStrLoc;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, Str, &AttrStrLoc))
    return;

  checkAssumptionAttr(S, AttrStrLoc, Str);

  D->addAttr(::new (S.Context) AssumptionAttr(S.Context, AL, Str));
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

static void handleOwnershipAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  // This attribute must be applied to a function declaration. The first
  // argument to the attribute must be an identifier, the name of the resource,
  // for example: malloc. The following arguments must be argument indexes, the
  // arguments must be of integer type for Returns, otherwise of pointer type.
  // The difference between Holds and Takes is that a pointer may still be used
  // after being held. free() should be __attribute((ownership_takes)), whereas
  // a list append function may well be __attribute((ownership_holds)).

  if (!AL.isArgIdent(0)) {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_n_type)
        << AL << 1 << AANT_ArgumentIdentifier;
    return;
  }

  // Figure out our Kind.
  OwnershipAttr::OwnershipKind K =
      OwnershipAttr(S.Context, AL, nullptr, nullptr, 0).getOwnKind();

  // Check arguments.
  switch (K) {
  case OwnershipAttr::Takes:
  case OwnershipAttr::Holds:
    if (AL.getNumArgs() < 2) {
      S.Diag(AL.getLoc(), diag::err_attribute_too_few_arguments) << AL << 2;
      return;
    }
    break;
  case OwnershipAttr::Returns:
    if (AL.getNumArgs() > 2) {
      S.Diag(AL.getLoc(), diag::err_attribute_too_many_arguments) << AL << 1;
      return;
    }
    break;
  }

  IdentifierInfo *Module = AL.getArgAsIdent(0)->Ident;

  StringRef ModuleName = Module->getName();
  if (normalizeName(ModuleName)) {
    Module = &S.PP.getIdentifierTable().get(ModuleName);
  }

  SmallVector<ParamIdx, 8> OwnershipArgs;
  for (unsigned i = 1; i < AL.getNumArgs(); ++i) {
    Expr *Ex = AL.getArgAsExpr(i);
    ParamIdx Idx;
    if (!checkFunctionOrMethodParameterIndex(S, D, AL, i, Ex, Idx))
      return;

    // Is the function argument a pointer type?
    QualType T = getFunctionOrMethodParamType(D, Idx.getASTIndex());
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
      S.Diag(AL.getLoc(), diag::err_ownership_type) << AL << Err
                                                    << Ex->getSourceRange();
      return;
    }

    // Check we don't have a conflict with another ownership attribute.
    for (const auto *I : D->specific_attrs<OwnershipAttr>()) {
      // Cannot have two ownership attributes of different kinds for the same
      // index.
      if (I->getOwnKind() != K && I->args_end() !=
          std::find(I->args_begin(), I->args_end(), Idx)) {
        S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << I;
        return;
      } else if (K == OwnershipAttr::Returns &&
                 I->getOwnKind() == OwnershipAttr::Returns) {
        // A returns attribute conflicts with any other returns attribute using
        // a different index.
        if (std::find(I->args_begin(), I->args_end(), Idx) == I->args_end()) {
          S.Diag(I->getLocation(), diag::err_ownership_returns_index_mismatch)
              << I->args_begin()->getSourceIndex();
          if (I->args_size())
            S.Diag(AL.getLoc(), diag::note_ownership_returns_index_mismatch)
                << Idx.getSourceIndex() << Ex->getSourceRange();
          return;
        }
      }
    }
    OwnershipArgs.push_back(Idx);
  }

  ParamIdx *Start = OwnershipArgs.data();
  unsigned Size = OwnershipArgs.size();
  llvm::array_pod_sort(Start, Start + Size);
  D->addAttr(::new (S.Context)
                 OwnershipAttr(S.Context, AL, Module, Start, Size));
}

static void handleWeakRefAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  // Check the attribute arguments.
  if (AL.getNumArgs() > 1) {
    S.Diag(AL.getLoc(), diag::err_attribute_wrong_number_arguments) << AL << 1;
    return;
  }

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
    S.Diag(AL.getLoc(), diag::err_attribute_weakref_not_global_context)
        << cast<NamedDecl>(D);
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
  if (AL.getNumArgs() && S.checkStringLiteralArgumentAttr(AL, 0, Str))
    // GCC will accept anything as the argument of weakref. Should we
    // check for an existing decl?
    D->addAttr(::new (S.Context) AliasAttr(S.Context, AL, Str));

  D->addAttr(::new (S.Context) WeakRefAttr(S.Context, AL));
}

static void handleIFuncAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  StringRef Str;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, Str))
    return;

  // Aliases should be on declarations, not definitions.
  const auto *FD = cast<FunctionDecl>(D);
  if (FD->isThisDeclarationADefinition()) {
    S.Diag(AL.getLoc(), diag::err_alias_is_definition) << FD << 1;
    return;
  }

  D->addAttr(::new (S.Context) IFuncAttr(S.Context, AL, Str));
}

static void handleAliasAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  StringRef Str;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, Str))
    return;

  if (S.Context.getTargetInfo().getTriple().isOSDarwin()) {
    S.Diag(AL.getLoc(), diag::err_alias_not_supported_on_darwin);
    return;
  }
  if (S.Context.getTargetInfo().getTriple().isNVPTX()) {
    S.Diag(AL.getLoc(), diag::err_alias_not_supported_on_nvptx);
  }

  // Aliases should be on declarations, not definitions.
  if (const auto *FD = dyn_cast<FunctionDecl>(D)) {
    if (FD->isThisDeclarationADefinition()) {
      S.Diag(AL.getLoc(), diag::err_alias_is_definition) << FD << 0;
      return;
    }
  } else {
    const auto *VD = cast<VarDecl>(D);
    if (VD->isThisDeclarationADefinition() && VD->isExternallyVisible()) {
      S.Diag(AL.getLoc(), diag::err_alias_is_definition) << VD << 0;
      return;
    }
  }

  // Mark target used to prevent unneeded-internal-declaration warnings.
  if (!S.LangOpts.CPlusPlus) {
    // FIXME: demangle Str for C++, as the attribute refers to the mangled
    // linkage name, not the pre-mangled identifier.
    const DeclarationNameInfo target(&S.Context.Idents.get(Str), AL.getLoc());
    LookupResult LR(S, target, Sema::LookupOrdinaryName);
    if (S.LookupQualifiedName(LR, S.getCurLexicalContext()))
      for (NamedDecl *ND : LR)
        ND->markUsed(S.Context);
  }

  D->addAttr(::new (S.Context) AliasAttr(S.Context, AL, Str));
}

static void handleTLSModelAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  StringRef Model;
  SourceLocation LiteralLoc;
  // Check that it is a string.
  if (!S.checkStringLiteralArgumentAttr(AL, 0, Model, &LiteralLoc))
    return;

  // Check that the value.
  if (Model != "global-dynamic" && Model != "local-dynamic"
      && Model != "initial-exec" && Model != "local-exec") {
    S.Diag(LiteralLoc, diag::err_attr_tlsmodel_arg);
    return;
  }

  if (S.Context.getTargetInfo().getTriple().isOSAIX() &&
      Model != "global-dynamic") {
    S.Diag(LiteralLoc, diag::err_aix_attr_unsupported_tls_model) << Model;
    return;
  }

  D->addAttr(::new (S.Context) TLSModelAttr(S.Context, AL, Model));
}

static void handleRestrictAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  QualType ResultType = getFunctionOrMethodResultType(D);
  if (ResultType->isAnyPointerType() || ResultType->isBlockPointerType()) {
    D->addAttr(::new (S.Context) RestrictAttr(S.Context, AL));
    return;
  }

  S.Diag(AL.getLoc(), diag::warn_attribute_return_pointers_only)
      << AL << getFunctionOrMethodResultSourceRange(D);
}

static void handleCPUSpecificAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  FunctionDecl *FD = cast<FunctionDecl>(D);

  if (const auto *MD = dyn_cast<CXXMethodDecl>(D)) {
    if (MD->getParent()->isLambda()) {
      S.Diag(AL.getLoc(), diag::err_attribute_dll_lambda) << AL;
      return;
    }
  }

  if (!AL.checkAtLeastNumArgs(S, 1))
    return;

  SmallVector<IdentifierInfo *, 8> CPUs;
  for (unsigned ArgNo = 0; ArgNo < getNumAttributeArgs(AL); ++ArgNo) {
    if (!AL.isArgIdent(ArgNo)) {
      S.Diag(AL.getLoc(), diag::err_attribute_argument_type)
          << AL << AANT_ArgumentIdentifier;
      return;
    }

    IdentifierLoc *CPUArg = AL.getArgAsIdent(ArgNo);
    StringRef CPUName = CPUArg->Ident->getName().trim();

    if (!S.Context.getTargetInfo().validateCPUSpecificCPUDispatch(CPUName)) {
      S.Diag(CPUArg->Loc, diag::err_invalid_cpu_specific_dispatch_value)
          << CPUName << (AL.getKind() == ParsedAttr::AT_CPUDispatch);
      return;
    }

    const TargetInfo &Target = S.Context.getTargetInfo();
    if (llvm::any_of(CPUs, [CPUName, &Target](const IdentifierInfo *Cur) {
          return Target.CPUSpecificManglingCharacter(CPUName) ==
                 Target.CPUSpecificManglingCharacter(Cur->getName());
        })) {
      S.Diag(AL.getLoc(), diag::warn_multiversion_duplicate_entries);
      return;
    }
    CPUs.push_back(CPUArg->Ident);
  }

  FD->setIsMultiVersion(true);
  if (AL.getKind() == ParsedAttr::AT_CPUSpecific)
    D->addAttr(::new (S.Context)
                   CPUSpecificAttr(S.Context, AL, CPUs.data(), CPUs.size()));
  else
    D->addAttr(::new (S.Context)
                   CPUDispatchAttr(S.Context, AL, CPUs.data(), CPUs.size()));
}

static void handleCommonAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (S.LangOpts.CPlusPlus) {
    S.Diag(AL.getLoc(), diag::err_attribute_not_supported_in_lang)
        << AL << AttributeLangSupport::Cpp;
    return;
  }

  D->addAttr(::new (S.Context) CommonAttr(S.Context, AL));
}

static void handleCmseNSEntryAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (S.LangOpts.CPlusPlus && !D->getDeclContext()->isExternCContext()) {
    S.Diag(AL.getLoc(), diag::err_attribute_not_clinkage) << AL;
    return;
  }

  const auto *FD = cast<FunctionDecl>(D);
  if (!FD->isExternallyVisible()) {
    S.Diag(AL.getLoc(), diag::warn_attribute_cmse_entry_static);
    return;
  }

  D->addAttr(::new (S.Context) CmseNSEntryAttr(S.Context, AL));
}

static void handleNakedAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (AL.isDeclspecAttribute()) {
    const auto &Triple = S.getASTContext().getTargetInfo().getTriple();
    const auto &Arch = Triple.getArch();
    if (Arch != llvm::Triple::x86 &&
        (Arch != llvm::Triple::arm && Arch != llvm::Triple::thumb)) {
      S.Diag(AL.getLoc(), diag::err_attribute_not_supported_on_arch)
          << AL << Triple.getArchName();
      return;
    }
  }

  D->addAttr(::new (S.Context) NakedAttr(S.Context, AL));
}

static void handleNoReturnAttr(Sema &S, Decl *D, const ParsedAttr &Attrs) {
  if (hasDeclarator(D)) return;

  if (!isa<ObjCMethodDecl>(D)) {
    S.Diag(Attrs.getLoc(), diag::warn_attribute_wrong_decl_type)
        << Attrs << ExpectedFunctionOrMethod;
    return;
  }

  D->addAttr(::new (S.Context) NoReturnAttr(S.Context, Attrs));
}

static void handleNoCfCheckAttr(Sema &S, Decl *D, const ParsedAttr &Attrs) {
  if (!S.getLangOpts().CFProtectionBranch)
    S.Diag(Attrs.getLoc(), diag::warn_nocf_check_attribute_ignored);
  else
    handleSimpleAttribute<AnyX86NoCfCheckAttr>(S, D, Attrs);
}

bool Sema::CheckAttrNoArgs(const ParsedAttr &Attrs) {
  if (!Attrs.checkExactlyNumArgs(*this, 0)) {
    Attrs.setInvalid();
    return true;
  }

  return false;
}

bool Sema::CheckAttrTarget(const ParsedAttr &AL) {
  // Check whether the attribute is valid on the current target.
  if (!AL.existsInTarget(Context.getTargetInfo())) {
    Diag(AL.getLoc(), diag::warn_unknown_attribute_ignored)
        << AL << AL.getRange();
    AL.setInvalid();
    return true;
  }

  return false;
}

static void handleAnalyzerNoReturnAttr(Sema &S, Decl *D, const ParsedAttr &AL) {

  // The checking path for 'noreturn' and 'analyzer_noreturn' are different
  // because 'analyzer_noreturn' does not impact the type.
  if (!isFunctionOrMethodOrBlock(D)) {
    ValueDecl *VD = dyn_cast<ValueDecl>(D);
    if (!VD || (!VD->getType()->isBlockPointerType() &&
                !VD->getType()->isFunctionPointerType())) {
      S.Diag(AL.getLoc(), AL.isStandardAttributeSyntax()
                              ? diag::err_attribute_wrong_decl_type
                              : diag::warn_attribute_wrong_decl_type)
          << AL << ExpectedFunctionMethodOrBlock;
      return;
    }
  }

  D->addAttr(::new (S.Context) AnalyzerNoReturnAttr(S.Context, AL));
}

// PS3 PPU-specific.
static void handleVecReturnAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  /*
    Returning a Vector Class in Registers

    According to the PPU ABI specifications, a class with a single member of
    vector type is returned in memory when used as the return value of a
    function.
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
    S.Diag(AL.getLoc(), diag::err_repeat_attribute) << A;
    return;
  }

  const auto *R = cast<RecordDecl>(D);
  int count = 0;

  if (!isa<CXXRecordDecl>(R)) {
    S.Diag(AL.getLoc(), diag::err_attribute_vecreturn_only_vector_member);
    return;
  }

  if (!cast<CXXRecordDecl>(R)->isPOD()) {
    S.Diag(AL.getLoc(), diag::err_attribute_vecreturn_only_pod_record);
    return;
  }

  for (const auto *I : R->fields()) {
    if ((count == 1) || !I->getType()->isVectorType()) {
      S.Diag(AL.getLoc(), diag::err_attribute_vecreturn_only_vector_member);
      return;
    }
    count++;
  }

  D->addAttr(::new (S.Context) VecReturnAttr(S.Context, AL));
}

static void handleDependencyAttr(Sema &S, Scope *Scope, Decl *D,
                                 const ParsedAttr &AL) {
  if (isa<ParmVarDecl>(D)) {
    // [[carries_dependency]] can only be applied to a parameter if it is a
    // parameter of a function declaration or lambda.
    if (!(Scope->getFlags() & clang::Scope::FunctionDeclarationScope)) {
      S.Diag(AL.getLoc(),
             diag::err_carries_dependency_param_not_function_decl);
      return;
    }
  }

  D->addAttr(::new (S.Context) CarriesDependencyAttr(S.Context, AL));
}

static void handleUnusedAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  bool IsCXX17Attr = AL.isCXX11Attribute() && !AL.getScopeName();

  // If this is spelled as the standard C++17 attribute, but not in C++17, warn
  // about using it as an extension.
  if (!S.getLangOpts().CPlusPlus17 && IsCXX17Attr)
    S.Diag(AL.getLoc(), diag::ext_cxx17_attr) << AL;

  D->addAttr(::new (S.Context) UnusedAttr(S.Context, AL));
}

static void handleConstructorAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  uint32_t priority = ConstructorAttr::DefaultPriority;
  if (AL.getNumArgs() &&
      !checkUInt32Argument(S, AL, AL.getArgAsExpr(0), priority))
    return;

  D->addAttr(::new (S.Context) ConstructorAttr(S.Context, AL, priority));
}

static void handleDestructorAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  uint32_t priority = DestructorAttr::DefaultPriority;
  if (AL.getNumArgs() &&
      !checkUInt32Argument(S, AL, AL.getArgAsExpr(0), priority))
    return;

  D->addAttr(::new (S.Context) DestructorAttr(S.Context, AL, priority));
}

template <typename AttrTy>
static void handleAttrWithMessage(Sema &S, Decl *D, const ParsedAttr &AL) {
  // Handle the case where the attribute has a text message.
  StringRef Str;
  if (AL.getNumArgs() == 1 && !S.checkStringLiteralArgumentAttr(AL, 0, Str))
    return;

  D->addAttr(::new (S.Context) AttrTy(S.Context, AL, Str));
}

static void handleObjCSuppresProtocolAttr(Sema &S, Decl *D,
                                          const ParsedAttr &AL) {
  if (!cast<ObjCProtocolDecl>(D)->isThisDeclarationADefinition()) {
    S.Diag(AL.getLoc(), diag::err_objc_attr_protocol_requires_definition)
        << AL << AL.getRange();
    return;
  }

  D->addAttr(::new (S.Context) ObjCExplicitProtocolImplAttr(S.Context, AL));
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

/// Check whether the two versions match.
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

AvailabilityAttr *Sema::mergeAvailabilityAttr(
    NamedDecl *D, const AttributeCommonInfo &CI, IdentifierInfo *Platform,
    bool Implicit, VersionTuple Introduced, VersionTuple Deprecated,
    VersionTuple Obsoleted, bool IsUnavailable, StringRef Message,
    bool IsStrict, StringRef Replacement, AvailabilityMergeKind AMK,
    int Priority) {
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
  case AMK_OptionalProtocolImplementation:
    OverrideOrImpl = true;
    break;
  }

  if (D->hasAttrs()) {
    AttrVec &Attrs = D->getAttrs();
    for (unsigned i = 0, e = Attrs.size(); i != e;) {
      const auto *OldAA = dyn_cast<AvailabilityAttr>(Attrs[i]);
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
      // has a lower priority use the existing one and discard the new
      // attribute.
      if (OldAA->getPriority() < Priority)
        return nullptr;

      // If there is an existing attribute for this platform that has a higher
      // priority than the new attribute then erase the old one and continue
      // processing the attributes.
      if (OldAA->getPriority() > Priority) {
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
          } else if (Which != 1 && AMK == AMK_OptionalProtocolImplementation) {
            // Allow different 'introduced' / 'obsoleted' availability versions
            // on a method that implements an optional protocol requirement. It
            // makes less sense to allow this for 'deprecated' as the user can't
            // see if the method is 'deprecated' as 'respondsToSelector' will
            // still return true when the method is deprecated.
            ++i;
            continue;
          } else {
            Diag(OldAA->getLocation(),
                 diag::warn_mismatched_availability_override)
              << Which
              << AvailabilityAttr::getPrettyPlatformName(Platform->getName())
              << FirstVersion.getAsString() << SecondVersion.getAsString()
              << (AMK == AMK_Override);
          }
          if (AMK == AMK_Override)
            Diag(CI.getLoc(), diag::note_overridden_method);
          else
            Diag(CI.getLoc(), diag::note_protocol_method);
        } else {
          Diag(OldAA->getLocation(), diag::warn_mismatched_availability);
          Diag(CI.getLoc(), diag::note_previous_attribute);
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
  if (!checkAvailabilityAttr(*this, CI.getRange(), Platform, MergedIntroduced,
                             MergedDeprecated, MergedObsoleted) &&
      !OverrideOrImpl) {
    auto *Avail = ::new (Context) AvailabilityAttr(
        Context, CI, Platform, Introduced, Deprecated, Obsoleted, IsUnavailable,
        Message, IsStrict, Replacement, Priority);
    Avail->setImplicit(Implicit);
    return Avail;
  }
  return nullptr;
}

static void handleAvailabilityAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (isa<UsingDecl, UnresolvedUsingTypenameDecl, UnresolvedUsingValueDecl>(
          D)) {
    S.Diag(AL.getRange().getBegin(), diag::warn_deprecated_ignored_on_using)
        << AL;
    return;
  }

  if (!AL.checkExactlyNumArgs(S, 1))
    return;
  IdentifierLoc *Platform = AL.getArgAsIdent(0);

  IdentifierInfo *II = Platform->Ident;
  if (AvailabilityAttr::getPrettyPlatformName(II->getName()).empty())
    S.Diag(Platform->Loc, diag::warn_availability_unknown_platform)
      << Platform->Ident;

  auto *ND = dyn_cast<NamedDecl>(D);
  if (!ND) // We warned about this already, so just return.
    return;

  AvailabilityChange Introduced = AL.getAvailabilityIntroduced();
  AvailabilityChange Deprecated = AL.getAvailabilityDeprecated();
  AvailabilityChange Obsoleted = AL.getAvailabilityObsoleted();
  bool IsUnavailable = AL.getUnavailableLoc().isValid();
  bool IsStrict = AL.getStrictLoc().isValid();
  StringRef Str;
  if (const auto *SE = dyn_cast_or_null<StringLiteral>(AL.getMessageExpr()))
    Str = SE->getString();
  StringRef Replacement;
  if (const auto *SE = dyn_cast_or_null<StringLiteral>(AL.getReplacementExpr()))
    Replacement = SE->getString();

  if (II->isStr("swift")) {
    if (Introduced.isValid() || Obsoleted.isValid() ||
        (!IsUnavailable && !Deprecated.isValid())) {
      S.Diag(AL.getLoc(),
             diag::warn_availability_swift_unavailable_deprecated_only);
      return;
    }
  }

  int PriorityModifier = AL.isPragmaClangAttribute()
                             ? Sema::AP_PragmaClangAttribute
                             : Sema::AP_Explicit;
  AvailabilityAttr *NewAttr = S.mergeAvailabilityAttr(
      ND, AL, II, false /*Implicit*/, Introduced.Version, Deprecated.Version,
      Obsoleted.Version, IsUnavailable, Str, IsStrict, Replacement,
      Sema::AMK_None, PriorityModifier);
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
            return VersionTuple(NewMajor);
          }

          return VersionTuple(2, 0);
        };

        auto NewIntroduced = adjustWatchOSVersion(Introduced.Version);
        auto NewDeprecated = adjustWatchOSVersion(Deprecated.Version);
        auto NewObsoleted = adjustWatchOSVersion(Obsoleted.Version);

        AvailabilityAttr *NewAttr = S.mergeAvailabilityAttr(
            ND, AL, NewII, true /*Implicit*/, NewIntroduced, NewDeprecated,
            NewObsoleted, IsUnavailable, Str, IsStrict, Replacement,
            Sema::AMK_None,
            PriorityModifier + Sema::AP_InferredFromOtherPlatform);
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
      AvailabilityAttr *NewAttr = S.mergeAvailabilityAttr(
          ND, AL, NewII, true /*Implicit*/, Introduced.Version,
          Deprecated.Version, Obsoleted.Version, IsUnavailable, Str, IsStrict,
          Replacement, Sema::AMK_None,
          PriorityModifier + Sema::AP_InferredFromOtherPlatform);
      if (NewAttr)
        D->addAttr(NewAttr);
      }
  } else if (S.Context.getTargetInfo().getTriple().getOS() ==
                 llvm::Triple::IOS &&
             S.Context.getTargetInfo().getTriple().isMacCatalystEnvironment()) {
    auto GetSDKInfo = [&]() {
      return S.getDarwinSDKInfoForAvailabilityChecking(AL.getRange().getBegin(),
                                                       "macOS");
    };

    // Transcribe "ios" to "maccatalyst" (and add a new attribute).
    IdentifierInfo *NewII = nullptr;
    if (II->getName() == "ios")
      NewII = &S.Context.Idents.get("maccatalyst");
    else if (II->getName() == "ios_app_extension")
      NewII = &S.Context.Idents.get("maccatalyst_app_extension");
    if (NewII) {
      auto MinMacCatalystVersion = [](const VersionTuple &V) {
        if (V.empty())
          return V;
        if (V.getMajor() < 13 ||
            (V.getMajor() == 13 && V.getMinor() && *V.getMinor() < 1))
          return VersionTuple(13, 1); // The min Mac Catalyst version is 13.1.
        return V;
      };
      AvailabilityAttr *NewAttr = S.mergeAvailabilityAttr(
          ND, AL.getRange(), NewII, true /*Implicit*/,
          MinMacCatalystVersion(Introduced.Version),
          MinMacCatalystVersion(Deprecated.Version),
          MinMacCatalystVersion(Obsoleted.Version), IsUnavailable, Str,
          IsStrict, Replacement, Sema::AMK_None,
          PriorityModifier + Sema::AP_InferredFromOtherPlatform);
      if (NewAttr)
        D->addAttr(NewAttr);
    } else if (II->getName() == "macos" && GetSDKInfo() &&
               (!Introduced.Version.empty() || !Deprecated.Version.empty() ||
                !Obsoleted.Version.empty())) {
      if (const auto *MacOStoMacCatalystMapping =
              GetSDKInfo()->getVersionMapping(
                  DarwinSDKInfo::OSEnvPair::macOStoMacCatalystPair())) {
        // Infer Mac Catalyst availability from the macOS availability attribute
        // if it has versioned availability. Don't infer 'unavailable'. This
        // inferred availability has lower priority than the other availability
        // attributes that are inferred from 'ios'.
        NewII = &S.Context.Idents.get("maccatalyst");
        auto RemapMacOSVersion =
            [&](const VersionTuple &V) -> Optional<VersionTuple> {
          if (V.empty())
            return None;
          // API_TO_BE_DEPRECATED is 100000.
          if (V.getMajor() == 100000)
            return VersionTuple(100000);
          // The minimum iosmac version is 13.1
          return MacOStoMacCatalystMapping->map(V, VersionTuple(13, 1), None);
        };
        Optional<VersionTuple> NewIntroduced =
                                   RemapMacOSVersion(Introduced.Version),
                               NewDeprecated =
                                   RemapMacOSVersion(Deprecated.Version),
                               NewObsoleted =
                                   RemapMacOSVersion(Obsoleted.Version);
        if (NewIntroduced || NewDeprecated || NewObsoleted) {
          auto VersionOrEmptyVersion =
              [](const Optional<VersionTuple> &V) -> VersionTuple {
            return V ? *V : VersionTuple();
          };
          AvailabilityAttr *NewAttr = S.mergeAvailabilityAttr(
              ND, AL.getRange(), NewII, true /*Implicit*/,
              VersionOrEmptyVersion(NewIntroduced),
              VersionOrEmptyVersion(NewDeprecated),
              VersionOrEmptyVersion(NewObsoleted), /*IsUnavailable=*/false, Str,
              IsStrict, Replacement, Sema::AMK_None,
              PriorityModifier + Sema::AP_InferredFromOtherPlatform +
                  Sema::AP_InferredFromOtherPlatform);
          if (NewAttr)
            D->addAttr(NewAttr);
        }
      }
    }
  }
}

static void handleExternalSourceSymbolAttr(Sema &S, Decl *D,
                                           const ParsedAttr &AL) {
  if (!AL.checkAtLeastNumArgs(S, 1) || !AL.checkAtMostNumArgs(S, 3))
    return;

  StringRef Language;
  if (const auto *SE = dyn_cast_or_null<StringLiteral>(AL.getArgAsExpr(0)))
    Language = SE->getString();
  StringRef DefinedIn;
  if (const auto *SE = dyn_cast_or_null<StringLiteral>(AL.getArgAsExpr(1)))
    DefinedIn = SE->getString();
  bool IsGeneratedDeclaration = AL.getArgAsIdent(2) != nullptr;

  D->addAttr(::new (S.Context) ExternalSourceSymbolAttr(
      S.Context, AL, Language, DefinedIn, IsGeneratedDeclaration));
}

template <class T>
static T *mergeVisibilityAttr(Sema &S, Decl *D, const AttributeCommonInfo &CI,
                              typename T::VisibilityType value) {
  T *existingAttr = D->getAttr<T>();
  if (existingAttr) {
    typename T::VisibilityType existingValue = existingAttr->getVisibility();
    if (existingValue == value)
      return nullptr;
    S.Diag(existingAttr->getLocation(), diag::err_mismatched_visibility);
    S.Diag(CI.getLoc(), diag::note_previous_attribute);
    D->dropAttr<T>();
  }
  return ::new (S.Context) T(S.Context, CI, value);
}

VisibilityAttr *Sema::mergeVisibilityAttr(Decl *D,
                                          const AttributeCommonInfo &CI,
                                          VisibilityAttr::VisibilityType Vis) {
  return ::mergeVisibilityAttr<VisibilityAttr>(*this, D, CI, Vis);
}

TypeVisibilityAttr *
Sema::mergeTypeVisibilityAttr(Decl *D, const AttributeCommonInfo &CI,
                              TypeVisibilityAttr::VisibilityType Vis) {
  return ::mergeVisibilityAttr<TypeVisibilityAttr>(*this, D, CI, Vis);
}

static void handleVisibilityAttr(Sema &S, Decl *D, const ParsedAttr &AL,
                                 bool isTypeVisibility) {
  // Visibility attributes don't mean anything on a typedef.
  if (isa<TypedefNameDecl>(D)) {
    S.Diag(AL.getRange().getBegin(), diag::warn_attribute_ignored) << AL;
    return;
  }

  // 'type_visibility' can only go on a type or namespace.
  if (isTypeVisibility &&
      !(isa<TagDecl>(D) ||
        isa<ObjCInterfaceDecl>(D) ||
        isa<NamespaceDecl>(D))) {
    S.Diag(AL.getRange().getBegin(), diag::err_attribute_wrong_decl_type)
        << AL << ExpectedTypeOrNamespace;
    return;
  }

  // Check that the argument is a string literal.
  StringRef TypeStr;
  SourceLocation LiteralLoc;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, TypeStr, &LiteralLoc))
    return;

  VisibilityAttr::VisibilityType type;
  if (!VisibilityAttr::ConvertStrToVisibilityType(TypeStr, type)) {
    S.Diag(LiteralLoc, diag::warn_attribute_type_not_supported) << AL
                                                                << TypeStr;
    return;
  }

  // Complain about attempts to use protected visibility on targets
  // (like Darwin) that don't support it.
  if (type == VisibilityAttr::Protected &&
      !S.Context.getTargetInfo().hasProtectedVisibility()) {
    S.Diag(AL.getLoc(), diag::warn_attribute_protected_visibility);
    type = VisibilityAttr::Default;
  }

  Attr *newAttr;
  if (isTypeVisibility) {
    newAttr = S.mergeTypeVisibilityAttr(
        D, AL, (TypeVisibilityAttr::VisibilityType)type);
  } else {
    newAttr = S.mergeVisibilityAttr(D, AL, type);
  }
  if (newAttr)
    D->addAttr(newAttr);
}

static void handleObjCDirectAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  // objc_direct cannot be set on methods declared in the context of a protocol
  if (isa<ObjCProtocolDecl>(D->getDeclContext())) {
    S.Diag(AL.getLoc(), diag::err_objc_direct_on_protocol) << false;
    return;
  }

  if (S.getLangOpts().ObjCRuntime.allowsDirectDispatch()) {
    handleSimpleAttribute<ObjCDirectAttr>(S, D, AL);
  } else {
    S.Diag(AL.getLoc(), diag::warn_objc_direct_ignored) << AL;
  }
}

static void handleObjCDirectMembersAttr(Sema &S, Decl *D,
                                        const ParsedAttr &AL) {
  if (S.getLangOpts().ObjCRuntime.allowsDirectDispatch()) {
    handleSimpleAttribute<ObjCDirectMembersAttr>(S, D, AL);
  } else {
    S.Diag(AL.getLoc(), diag::warn_objc_direct_ignored) << AL;
  }
}

static void handleObjCMethodFamilyAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  const auto *M = cast<ObjCMethodDecl>(D);
  if (!AL.isArgIdent(0)) {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_n_type)
        << AL << 1 << AANT_ArgumentIdentifier;
    return;
  }

  IdentifierLoc *IL = AL.getArgAsIdent(0);
  ObjCMethodFamilyAttr::FamilyKind F;
  if (!ObjCMethodFamilyAttr::ConvertStrToFamilyKind(IL->Ident->getName(), F)) {
    S.Diag(IL->Loc, diag::warn_attribute_type_not_supported) << AL << IL->Ident;
    return;
  }

  if (F == ObjCMethodFamilyAttr::OMF_init &&
      !M->getReturnType()->isObjCObjectPointerType()) {
    S.Diag(M->getLocation(), diag::err_init_method_bad_return_type)
        << M->getReturnType();
    // Ignore the attribute.
    return;
  }

  D->addAttr(new (S.Context) ObjCMethodFamilyAttr(S.Context, AL, F));
}

static void handleObjCNSObject(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (const auto *TD = dyn_cast<TypedefNameDecl>(D)) {
    QualType T = TD->getUnderlyingType();
    if (!T->isCARCBridgableType()) {
      S.Diag(TD->getLocation(), diag::err_nsobject_attribute);
      return;
    }
  }
  else if (const auto *PD = dyn_cast<ObjCPropertyDecl>(D)) {
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
  D->addAttr(::new (S.Context) ObjCNSObjectAttr(S.Context, AL));
}

static void handleObjCIndependentClass(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (const auto *TD = dyn_cast<TypedefNameDecl>(D)) {
    QualType T = TD->getUnderlyingType();
    if (!T->isObjCObjectPointerType()) {
      S.Diag(TD->getLocation(), diag::warn_ptr_independentclass_attribute);
      return;
    }
  } else {
    S.Diag(D->getLocation(), diag::warn_independentclass_attribute);
    return;
  }
  D->addAttr(::new (S.Context) ObjCIndependentClassAttr(S.Context, AL));
}

static void handleBlocksAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (!AL.isArgIdent(0)) {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_n_type)
        << AL << 1 << AANT_ArgumentIdentifier;
    return;
  }

  IdentifierInfo *II = AL.getArgAsIdent(0)->Ident;
  BlocksAttr::BlockType type;
  if (!BlocksAttr::ConvertStrToBlockType(II->getName(), type)) {
    S.Diag(AL.getLoc(), diag::warn_attribute_type_not_supported) << AL << II;
    return;
  }

  D->addAttr(::new (S.Context) BlocksAttr(S.Context, AL, type));
}

static void handleSentinelAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  unsigned sentinel = (unsigned)SentinelAttr::DefaultSentinel;
  if (AL.getNumArgs() > 0) {
    Expr *E = AL.getArgAsExpr(0);
    Optional<llvm::APSInt> Idx = llvm::APSInt(32);
    if (E->isTypeDependent() || E->isValueDependent() ||
        !(Idx = E->getIntegerConstantExpr(S.Context))) {
      S.Diag(AL.getLoc(), diag::err_attribute_argument_n_type)
          << AL << 1 << AANT_ArgumentIntegerConstant << E->getSourceRange();
      return;
    }

    if (Idx->isSigned() && Idx->isNegative()) {
      S.Diag(AL.getLoc(), diag::err_attribute_sentinel_less_than_zero)
        << E->getSourceRange();
      return;
    }

    sentinel = Idx->getZExtValue();
  }

  unsigned nullPos = (unsigned)SentinelAttr::DefaultNullPos;
  if (AL.getNumArgs() > 1) {
    Expr *E = AL.getArgAsExpr(1);
    Optional<llvm::APSInt> Idx = llvm::APSInt(32);
    if (E->isTypeDependent() || E->isValueDependent() ||
        !(Idx = E->getIntegerConstantExpr(S.Context))) {
      S.Diag(AL.getLoc(), diag::err_attribute_argument_n_type)
          << AL << 2 << AANT_ArgumentIntegerConstant << E->getSourceRange();
      return;
    }
    nullPos = Idx->getZExtValue();

    if ((Idx->isSigned() && Idx->isNegative()) || nullPos > 1) {
      // FIXME: This error message could be improved, it would be nice
      // to say what the bounds actually are.
      S.Diag(AL.getLoc(), diag::err_attribute_sentinel_not_zero_or_one)
        << E->getSourceRange();
      return;
    }
  }

  if (const auto *FD = dyn_cast<FunctionDecl>(D)) {
    const FunctionType *FT = FD->getType()->castAs<FunctionType>();
    if (isa<FunctionNoProtoType>(FT)) {
      S.Diag(AL.getLoc(), diag::warn_attribute_sentinel_named_arguments);
      return;
    }

    if (!cast<FunctionProtoType>(FT)->isVariadic()) {
      S.Diag(AL.getLoc(), diag::warn_attribute_sentinel_not_variadic) << 0;
      return;
    }
  } else if (const auto *MD = dyn_cast<ObjCMethodDecl>(D)) {
    if (!MD->isVariadic()) {
      S.Diag(AL.getLoc(), diag::warn_attribute_sentinel_not_variadic) << 0;
      return;
    }
  } else if (const auto *BD = dyn_cast<BlockDecl>(D)) {
    if (!BD->isVariadic()) {
      S.Diag(AL.getLoc(), diag::warn_attribute_sentinel_not_variadic) << 1;
      return;
    }
  } else if (const auto *V = dyn_cast<VarDecl>(D)) {
    QualType Ty = V->getType();
    if (Ty->isBlockPointerType() || Ty->isFunctionPointerType()) {
      const FunctionType *FT = Ty->isFunctionPointerType()
                                   ? D->getFunctionType()
                                   : Ty->castAs<BlockPointerType>()
                                         ->getPointeeType()
                                         ->castAs<FunctionType>();
      if (!cast<FunctionProtoType>(FT)->isVariadic()) {
        int m = Ty->isFunctionPointerType() ? 0 : 1;
        S.Diag(AL.getLoc(), diag::warn_attribute_sentinel_not_variadic) << m;
        return;
      }
    } else {
      S.Diag(AL.getLoc(), diag::warn_attribute_wrong_decl_type)
          << AL << ExpectedFunctionMethodOrBlock;
      return;
    }
  } else {
    S.Diag(AL.getLoc(), diag::warn_attribute_wrong_decl_type)
        << AL << ExpectedFunctionMethodOrBlock;
    return;
  }
  D->addAttr(::new (S.Context) SentinelAttr(S.Context, AL, sentinel, nullPos));
}

static void handleWarnUnusedResult(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (D->getFunctionType() &&
      D->getFunctionType()->getReturnType()->isVoidType() &&
      !isa<CXXConstructorDecl>(D)) {
    S.Diag(AL.getLoc(), diag::warn_attribute_void_function_method) << AL << 0;
    return;
  }
  if (const auto *MD = dyn_cast<ObjCMethodDecl>(D))
    if (MD->getReturnType()->isVoidType()) {
      S.Diag(AL.getLoc(), diag::warn_attribute_void_function_method) << AL << 1;
      return;
    }

  StringRef Str;
  if (AL.isStandardAttributeSyntax() && !AL.getScopeName()) {
    // The standard attribute cannot be applied to variable declarations such
    // as a function pointer.
    if (isa<VarDecl>(D))
      S.Diag(AL.getLoc(), diag::warn_attribute_wrong_decl_type_str)
          << AL << "functions, classes, or enumerations";

    // If this is spelled as the standard C++17 attribute, but not in C++17,
    // warn about using it as an extension. If there are attribute arguments,
    // then claim it's a C++2a extension instead.
    // FIXME: If WG14 does not seem likely to adopt the same feature, add an
    // extension warning for C2x mode.
    const LangOptions &LO = S.getLangOpts();
    if (AL.getNumArgs() == 1) {
      if (LO.CPlusPlus && !LO.CPlusPlus20)
        S.Diag(AL.getLoc(), diag::ext_cxx20_attr) << AL;

      // Since this this is spelled [[nodiscard]], get the optional string
      // literal. If in C++ mode, but not in C++2a mode, diagnose as an
      // extension.
      // FIXME: C2x should support this feature as well, even as an extension.
      if (!S.checkStringLiteralArgumentAttr(AL, 0, Str, nullptr))
        return;
    } else if (LO.CPlusPlus && !LO.CPlusPlus17)
      S.Diag(AL.getLoc(), diag::ext_cxx17_attr) << AL;
  }

  D->addAttr(::new (S.Context) WarnUnusedResultAttr(S.Context, AL, Str));
}

static void handleWeakImportAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  // weak_import only applies to variable & function declarations.
  bool isDef = false;
  if (!D->canBeWeakImported(isDef)) {
    if (isDef)
      S.Diag(AL.getLoc(), diag::warn_attribute_invalid_on_definition)
        << "weak_import";
    else if (isa<ObjCPropertyDecl>(D) || isa<ObjCMethodDecl>(D) ||
             (S.Context.getTargetInfo().getTriple().isOSDarwin() &&
              (isa<ObjCInterfaceDecl>(D) || isa<EnumDecl>(D)))) {
      // Nothing to warn about here.
    } else
      S.Diag(AL.getLoc(), diag::warn_attribute_wrong_decl_type)
          << AL << ExpectedVariableOrFunction;

    return;
  }

  D->addAttr(::new (S.Context) WeakImportAttr(S.Context, AL));
}

// Handles reqd_work_group_size and work_group_size_hint.
template <typename WorkGroupAttr>
static void handleWorkGroupSize(Sema &S, Decl *D, const ParsedAttr &AL) {
  uint32_t WGSize[3];
  for (unsigned i = 0; i < 3; ++i) {
    const Expr *E = AL.getArgAsExpr(i);
    if (!checkUInt32Argument(S, AL, E, WGSize[i], i,
                             /*StrictlyUnsigned=*/true))
      return;
    if (WGSize[i] == 0) {
      S.Diag(AL.getLoc(), diag::err_attribute_argument_is_zero)
          << AL << E->getSourceRange();
      return;
    }
  }

  WorkGroupAttr *Existing = D->getAttr<WorkGroupAttr>();
  if (Existing && !(Existing->getXDim() == WGSize[0] &&
                    Existing->getYDim() == WGSize[1] &&
                    Existing->getZDim() == WGSize[2]))
    S.Diag(AL.getLoc(), diag::warn_duplicate_attribute) << AL;

  D->addAttr(::new (S.Context)
                 WorkGroupAttr(S.Context, AL, WGSize[0], WGSize[1], WGSize[2]));
}

// Handles intel_reqd_sub_group_size.
static void handleSubGroupSize(Sema &S, Decl *D, const ParsedAttr &AL) {
  uint32_t SGSize;
  const Expr *E = AL.getArgAsExpr(0);
  if (!checkUInt32Argument(S, AL, E, SGSize))
    return;
  if (SGSize == 0) {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_is_zero)
        << AL << E->getSourceRange();
    return;
  }

  OpenCLIntelReqdSubGroupSizeAttr *Existing =
      D->getAttr<OpenCLIntelReqdSubGroupSizeAttr>();
  if (Existing && Existing->getSubGroupSize() != SGSize)
    S.Diag(AL.getLoc(), diag::warn_duplicate_attribute) << AL;

  D->addAttr(::new (S.Context)
                 OpenCLIntelReqdSubGroupSizeAttr(S.Context, AL, SGSize));
}

static void handleVecTypeHint(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (!AL.hasParsedType()) {
    S.Diag(AL.getLoc(), diag::err_attribute_wrong_number_arguments) << AL << 1;
    return;
  }

  TypeSourceInfo *ParmTSI = nullptr;
  QualType ParmType = S.GetTypeFromParser(AL.getTypeArg(), &ParmTSI);
  assert(ParmTSI && "no type source info for attribute argument");

  if (!ParmType->isExtVectorType() && !ParmType->isFloatingType() &&
      (ParmType->isBooleanType() ||
       !ParmType->isIntegralType(S.getASTContext()))) {
    S.Diag(AL.getLoc(), diag::err_attribute_invalid_argument) << 2 << AL;
    return;
  }

  if (VecTypeHintAttr *A = D->getAttr<VecTypeHintAttr>()) {
    if (!S.Context.hasSameType(A->getTypeHint(), ParmType)) {
      S.Diag(AL.getLoc(), diag::warn_duplicate_attribute) << AL;
      return;
    }
  }

  D->addAttr(::new (S.Context) VecTypeHintAttr(S.Context, AL, ParmTSI));
}

SectionAttr *Sema::mergeSectionAttr(Decl *D, const AttributeCommonInfo &CI,
                                    StringRef Name) {
  // Explicit or partial specializations do not inherit
  // the section attribute from the primary template.
  if (const auto *FD = dyn_cast<FunctionDecl>(D)) {
    if (CI.getAttributeSpellingListIndex() == SectionAttr::Declspec_allocate &&
        FD->isFunctionTemplateSpecialization())
      return nullptr;
  }
  if (SectionAttr *ExistingAttr = D->getAttr<SectionAttr>()) {
    if (ExistingAttr->getName() == Name)
      return nullptr;
    Diag(ExistingAttr->getLocation(), diag::warn_mismatched_section)
         << 1 /*section*/;
    Diag(CI.getLoc(), diag::note_previous_attribute);
    return nullptr;
  }
  return ::new (Context) SectionAttr(Context, CI, Name);
}

/// Used to implement to perform semantic checking on
/// attribute((section("foo"))) specifiers.
///
/// In this case, "foo" is passed in to be checked.  If the section
/// specifier is invalid, return an Error that indicates the problem.
///
/// This is a simple quality of implementation feature to catch errors
/// and give good diagnostics in cases when the assembler or code generator
/// would otherwise reject the section specifier.
llvm::Error Sema::isValidSectionSpecifier(StringRef SecName) {
  if (!Context.getTargetInfo().getTriple().isOSDarwin())
    return llvm::Error::success();

  // Let MCSectionMachO validate this.
  StringRef Segment, Section;
  unsigned TAA, StubSize;
  bool HasTAA;
  return llvm::MCSectionMachO::ParseSectionSpecifier(SecName, Segment, Section,
                                                     TAA, HasTAA, StubSize);
}

bool Sema::checkSectionName(SourceLocation LiteralLoc, StringRef SecName) {
  if (llvm::Error E = isValidSectionSpecifier(SecName)) {
    Diag(LiteralLoc, diag::err_attribute_section_invalid_for_target)
        << toString(std::move(E)) << 1 /*'section'*/;
    return false;
  }
  return true;
}

static void handleSectionAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  // Make sure that there is a string literal as the sections's single
  // argument.
  StringRef Str;
  SourceLocation LiteralLoc;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, Str, &LiteralLoc))
    return;

  if (!S.checkSectionName(LiteralLoc, Str))
    return;

  SectionAttr *NewAttr = S.mergeSectionAttr(D, AL, Str);
  if (NewAttr) {
    D->addAttr(NewAttr);
    if (isa<FunctionDecl, FunctionTemplateDecl, ObjCMethodDecl,
            ObjCPropertyDecl>(D))
      S.UnifySection(NewAttr->getName(),
                     ASTContext::PSF_Execute | ASTContext::PSF_Read,
                     cast<NamedDecl>(D));
  }
}

// This is used for `__declspec(code_seg("segname"))` on a decl.
// `#pragma code_seg("segname")` uses checkSectionName() instead.
static bool checkCodeSegName(Sema &S, SourceLocation LiteralLoc,
                             StringRef CodeSegName) {
  if (llvm::Error E = S.isValidSectionSpecifier(CodeSegName)) {
    S.Diag(LiteralLoc, diag::err_attribute_section_invalid_for_target)
        << toString(std::move(E)) << 0 /*'code-seg'*/;
    return false;
  }

  return true;
}

CodeSegAttr *Sema::mergeCodeSegAttr(Decl *D, const AttributeCommonInfo &CI,
                                    StringRef Name) {
  // Explicit or partial specializations do not inherit
  // the code_seg attribute from the primary template.
  if (const auto *FD = dyn_cast<FunctionDecl>(D)) {
    if (FD->isFunctionTemplateSpecialization())
      return nullptr;
  }
  if (const auto *ExistingAttr = D->getAttr<CodeSegAttr>()) {
    if (ExistingAttr->getName() == Name)
      return nullptr;
    Diag(ExistingAttr->getLocation(), diag::warn_mismatched_section)
         << 0 /*codeseg*/;
    Diag(CI.getLoc(), diag::note_previous_attribute);
    return nullptr;
  }
  return ::new (Context) CodeSegAttr(Context, CI, Name);
}

static void handleCodeSegAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  StringRef Str;
  SourceLocation LiteralLoc;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, Str, &LiteralLoc))
    return;
  if (!checkCodeSegName(S, LiteralLoc, Str))
    return;
  if (const auto *ExistingAttr = D->getAttr<CodeSegAttr>()) {
    if (!ExistingAttr->isImplicit()) {
      S.Diag(AL.getLoc(),
             ExistingAttr->getName() == Str
             ? diag::warn_duplicate_codeseg_attribute
             : diag::err_conflicting_codeseg_attribute);
      return;
    }
    D->dropAttr<CodeSegAttr>();
  }
  if (CodeSegAttr *CSA = S.mergeCodeSegAttr(D, AL, Str))
    D->addAttr(CSA);
}

// Check for things we'd like to warn about. Multiversioning issues are
// handled later in the process, once we know how many exist.
bool Sema::checkTargetAttr(SourceLocation LiteralLoc, StringRef AttrStr) {
  enum FirstParam { Unsupported, Duplicate, Unknown };
  enum SecondParam { None, Architecture, Tune };
  if (AttrStr.find("fpmath=") != StringRef::npos)
    return Diag(LiteralLoc, diag::warn_unsupported_target_attribute)
           << Unsupported << None << "fpmath=";

  // Diagnose use of tune if target doesn't support it.
  if (!Context.getTargetInfo().supportsTargetAttributeTune() &&
      AttrStr.find("tune=") != StringRef::npos)
    return Diag(LiteralLoc, diag::warn_unsupported_target_attribute)
           << Unsupported << None << "tune=";

  ParsedTargetAttr ParsedAttrs = TargetAttr::parse(AttrStr);

  if (!ParsedAttrs.Architecture.empty() &&
      !Context.getTargetInfo().isValidCPUName(ParsedAttrs.Architecture))
    return Diag(LiteralLoc, diag::warn_unsupported_target_attribute)
           << Unknown << Architecture << ParsedAttrs.Architecture;

  if (!ParsedAttrs.Tune.empty() &&
      !Context.getTargetInfo().isValidCPUName(ParsedAttrs.Tune))
    return Diag(LiteralLoc, diag::warn_unsupported_target_attribute)
           << Unknown << Tune << ParsedAttrs.Tune;

  if (ParsedAttrs.DuplicateArchitecture)
    return Diag(LiteralLoc, diag::warn_unsupported_target_attribute)
           << Duplicate << None << "arch=";
  if (ParsedAttrs.DuplicateTune)
    return Diag(LiteralLoc, diag::warn_unsupported_target_attribute)
           << Duplicate << None << "tune=";

  for (const auto &Feature : ParsedAttrs.Features) {
    auto CurFeature = StringRef(Feature).drop_front(); // remove + or -.
    if (!Context.getTargetInfo().isValidFeatureName(CurFeature))
      return Diag(LiteralLoc, diag::warn_unsupported_target_attribute)
             << Unsupported << None << CurFeature;
  }

  TargetInfo::BranchProtectionInfo BPI;
  StringRef Error;
  if (!ParsedAttrs.BranchProtection.empty() &&
      !Context.getTargetInfo().validateBranchProtection(
          ParsedAttrs.BranchProtection, BPI, Error)) {
    if (Error.empty())
      return Diag(LiteralLoc, diag::warn_unsupported_target_attribute)
             << Unsupported << None << "branch-protection";
    else
      return Diag(LiteralLoc, diag::err_invalid_branch_protection_spec)
             << Error;
  }

  return false;
}

static void handleTargetAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  StringRef Str;
  SourceLocation LiteralLoc;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, Str, &LiteralLoc) ||
      S.checkTargetAttr(LiteralLoc, Str))
    return;

  TargetAttr *NewAttr = ::new (S.Context) TargetAttr(S.Context, AL, Str);
  D->addAttr(NewAttr);
}

static void handleMinVectorWidthAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  Expr *E = AL.getArgAsExpr(0);
  uint32_t VecWidth;
  if (!checkUInt32Argument(S, AL, E, VecWidth)) {
    AL.setInvalid();
    return;
  }

  MinVectorWidthAttr *Existing = D->getAttr<MinVectorWidthAttr>();
  if (Existing && Existing->getVectorWidth() != VecWidth) {
    S.Diag(AL.getLoc(), diag::warn_duplicate_attribute) << AL;
    return;
  }

  D->addAttr(::new (S.Context) MinVectorWidthAttr(S.Context, AL, VecWidth));
}

static void handleCleanupAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  Expr *E = AL.getArgAsExpr(0);
  SourceLocation Loc = E->getExprLoc();
  FunctionDecl *FD = nullptr;
  DeclarationNameInfo NI;

  // gcc only allows for simple identifiers. Since we support more than gcc, we
  // will warn the user.
  if (auto *DRE = dyn_cast<DeclRefExpr>(E)) {
    if (DRE->hasQualifier())
      S.Diag(Loc, diag::warn_cleanup_ext);
    FD = dyn_cast<FunctionDecl>(DRE->getDecl());
    NI = DRE->getNameInfo();
    if (!FD) {
      S.Diag(Loc, diag::err_attribute_cleanup_arg_not_function) << 1
        << NI.getName();
      return;
    }
  } else if (auto *ULE = dyn_cast<UnresolvedLookupExpr>(E)) {
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
  QualType Ty = S.Context.getPointerType(cast<VarDecl>(D)->getType());
  QualType ParamTy = FD->getParamDecl(0)->getType();
  if (S.CheckAssignmentConstraints(FD->getParamDecl(0)->getLocation(),
                                   ParamTy, Ty) != Sema::Compatible) {
    S.Diag(Loc, diag::err_attribute_cleanup_func_arg_incompatible_type)
      << NI.getName() << ParamTy << Ty;
    return;
  }

  D->addAttr(::new (S.Context) CleanupAttr(S.Context, AL, FD));
}

static void handleEnumExtensibilityAttr(Sema &S, Decl *D,
                                        const ParsedAttr &AL) {
  if (!AL.isArgIdent(0)) {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_n_type)
        << AL << 0 << AANT_ArgumentIdentifier;
    return;
  }

  EnumExtensibilityAttr::Kind ExtensibilityKind;
  IdentifierInfo *II = AL.getArgAsIdent(0)->Ident;
  if (!EnumExtensibilityAttr::ConvertStrToKind(II->getName(),
                                               ExtensibilityKind)) {
    S.Diag(AL.getLoc(), diag::warn_attribute_type_not_supported) << AL << II;
    return;
  }

  D->addAttr(::new (S.Context)
                 EnumExtensibilityAttr(S.Context, AL, ExtensibilityKind));
}

/// Handle __attribute__((format_arg((idx)))) attribute based on
/// http://gcc.gnu.org/onlinedocs/gcc/Function-Attributes.html
static void handleFormatArgAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  Expr *IdxExpr = AL.getArgAsExpr(0);
  ParamIdx Idx;
  if (!checkFunctionOrMethodParameterIndex(S, D, AL, 1, IdxExpr, Idx))
    return;

  // Make sure the format string is really a string.
  QualType Ty = getFunctionOrMethodParamType(D, Idx.getASTIndex());

  bool NotNSStringTy = !isNSStringType(Ty, S.Context);
  if (NotNSStringTy &&
      !isCFStringType(Ty, S.Context) &&
      (!Ty->isPointerType() ||
       !Ty->castAs<PointerType>()->getPointeeType()->isCharType())) {
    S.Diag(AL.getLoc(), diag::err_format_attribute_not)
        << "a string type" << IdxExpr->getSourceRange()
        << getFunctionOrMethodParamRange(D, 0);
    return;
  }
  Ty = getFunctionOrMethodResultType(D);
  if (!isNSStringType(Ty, S.Context, /*AllowNSAttributedString=*/true) &&
      !isCFStringType(Ty, S.Context) &&
      (!Ty->isPointerType() ||
       !Ty->castAs<PointerType>()->getPointeeType()->isCharType())) {
    S.Diag(AL.getLoc(), diag::err_format_attribute_result_not)
        << (NotNSStringTy ? "string type" : "NSString")
        << IdxExpr->getSourceRange() << getFunctionOrMethodParamRange(D, 0);
    return;
  }

  D->addAttr(::new (S.Context) FormatArgAttr(S.Context, AL, Idx));
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
static void handleInitPriorityAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (!S.getLangOpts().CPlusPlus) {
    S.Diag(AL.getLoc(), diag::warn_attribute_ignored) << AL;
    return;
  }

  if (S.getCurFunctionOrMethodDecl()) {
    S.Diag(AL.getLoc(), diag::err_init_priority_object_attr);
    AL.setInvalid();
    return;
  }
  QualType T = cast<VarDecl>(D)->getType();
  if (S.Context.getAsArrayType(T))
    T = S.Context.getBaseElementType(T);
  if (!T->getAs<RecordType>()) {
    S.Diag(AL.getLoc(), diag::err_init_priority_object_attr);
    AL.setInvalid();
    return;
  }

  Expr *E = AL.getArgAsExpr(0);
  uint32_t prioritynum;
  if (!checkUInt32Argument(S, AL, E, prioritynum)) {
    AL.setInvalid();
    return;
  }

  // Only perform the priority check if the attribute is outside of a system
  // header. Values <= 100 are reserved for the implementation, and libc++
  // benefits from being able to specify values in that range.
  if ((prioritynum < 101 || prioritynum > 65535) &&
      !S.getSourceManager().isInSystemHeader(AL.getLoc())) {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_out_of_range)
        << E->getSourceRange() << AL << 101 << 65535;
    AL.setInvalid();
    return;
  }
  D->addAttr(::new (S.Context) InitPriorityAttr(S.Context, AL, prioritynum));
}

ErrorAttr *Sema::mergeErrorAttr(Decl *D, const AttributeCommonInfo &CI,
                                StringRef NewUserDiagnostic) {
  if (const auto *EA = D->getAttr<ErrorAttr>()) {
    std::string NewAttr = CI.getNormalizedFullName();
    assert((NewAttr == "error" || NewAttr == "warning") &&
           "unexpected normalized full name");
    bool Match = (EA->isError() && NewAttr == "error") ||
                 (EA->isWarning() && NewAttr == "warning");
    if (!Match) {
      Diag(EA->getLocation(), diag::err_attributes_are_not_compatible)
          << CI << EA;
      Diag(CI.getLoc(), diag::note_conflicting_attribute);
      return nullptr;
    }
    if (EA->getUserDiagnostic() != NewUserDiagnostic) {
      Diag(CI.getLoc(), diag::warn_duplicate_attribute) << EA;
      Diag(EA->getLoc(), diag::note_previous_attribute);
    }
    D->dropAttr<ErrorAttr>();
  }
  return ::new (Context) ErrorAttr(Context, CI, NewUserDiagnostic);
}

FormatAttr *Sema::mergeFormatAttr(Decl *D, const AttributeCommonInfo &CI,
                                  IdentifierInfo *Format, int FormatIdx,
                                  int FirstArg) {
  // Check whether we already have an equivalent format attribute.
  for (auto *F : D->specific_attrs<FormatAttr>()) {
    if (F->getType() == Format &&
        F->getFormatIdx() == FormatIdx &&
        F->getFirstArg() == FirstArg) {
      // If we don't have a valid location for this attribute, adopt the
      // location.
      if (F->getLocation().isInvalid())
        F->setRange(CI.getRange());
      return nullptr;
    }
  }

  return ::new (Context) FormatAttr(Context, CI, Format, FormatIdx, FirstArg);
}

/// Handle __attribute__((format(type,idx,firstarg))) attributes based on
/// http://gcc.gnu.org/onlinedocs/gcc/Function-Attributes.html
static void handleFormatAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (!AL.isArgIdent(0)) {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_n_type)
        << AL << 1 << AANT_ArgumentIdentifier;
    return;
  }

  // In C++ the implicit 'this' function parameter also counts, and they are
  // counted from one.
  bool HasImplicitThisParam = isInstanceMethod(D);
  unsigned NumArgs = getFunctionOrMethodNumParams(D) + HasImplicitThisParam;

  IdentifierInfo *II = AL.getArgAsIdent(0)->Ident;
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
    S.Diag(AL.getLoc(), diag::warn_attribute_type_not_supported)
        << AL << II->getName();
    return;
  }

  // checks for the 2nd argument
  Expr *IdxExpr = AL.getArgAsExpr(1);
  uint32_t Idx;
  if (!checkUInt32Argument(S, AL, IdxExpr, Idx, 2))
    return;

  if (Idx < 1 || Idx > NumArgs) {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_out_of_bounds)
        << AL << 2 << IdxExpr->getSourceRange();
    return;
  }

  // FIXME: Do we need to bounds check?
  unsigned ArgIdx = Idx - 1;

  if (HasImplicitThisParam) {
    if (ArgIdx == 0) {
      S.Diag(AL.getLoc(),
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
      S.Diag(AL.getLoc(), diag::err_format_attribute_not)
        << "a CFString" << IdxExpr->getSourceRange()
        << getFunctionOrMethodParamRange(D, ArgIdx);
      return;
    }
  } else if (Kind == NSStringFormat) {
    // FIXME: do we need to check if the type is NSString*?  What are the
    // semantics?
    if (!isNSStringType(Ty, S.Context, /*AllowNSAttributedString=*/true)) {
      S.Diag(AL.getLoc(), diag::err_format_attribute_not)
        << "an NSString" << IdxExpr->getSourceRange()
        << getFunctionOrMethodParamRange(D, ArgIdx);
      return;
    }
  } else if (!Ty->isPointerType() ||
             !Ty->castAs<PointerType>()->getPointeeType()->isCharType()) {
    S.Diag(AL.getLoc(), diag::err_format_attribute_not)
      << "a string type" << IdxExpr->getSourceRange()
      << getFunctionOrMethodParamRange(D, ArgIdx);
    return;
  }

  // check the 3rd argument
  Expr *FirstArgExpr = AL.getArgAsExpr(2);
  uint32_t FirstArg;
  if (!checkUInt32Argument(S, AL, FirstArgExpr, FirstArg, 3))
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
      S.Diag(AL.getLoc(), diag::err_format_strftime_third_parameter)
        << FirstArgExpr->getSourceRange();
      return;
    }
  // if 0 it disables parameter checking (to use with e.g. va_list)
  } else if (FirstArg != 0 && FirstArg != NumArgs) {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_out_of_bounds)
        << AL << 3 << FirstArgExpr->getSourceRange();
    return;
  }

  FormatAttr *NewAttr = S.mergeFormatAttr(D, AL, II, Idx, FirstArg);
  if (NewAttr)
    D->addAttr(NewAttr);
}

/// Handle __attribute__((callback(CalleeIdx, PayloadIdx0, ...))) attributes.
static void handleCallbackAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  // The index that identifies the callback callee is mandatory.
  if (AL.getNumArgs() == 0) {
    S.Diag(AL.getLoc(), diag::err_callback_attribute_no_callee)
        << AL.getRange();
    return;
  }

  bool HasImplicitThisParam = isInstanceMethod(D);
  int32_t NumArgs = getFunctionOrMethodNumParams(D);

  FunctionDecl *FD = D->getAsFunction();
  assert(FD && "Expected a function declaration!");

  llvm::StringMap<int> NameIdxMapping;
  NameIdxMapping["__"] = -1;

  NameIdxMapping["this"] = 0;

  int Idx = 1;
  for (const ParmVarDecl *PVD : FD->parameters())
    NameIdxMapping[PVD->getName()] = Idx++;

  auto UnknownName = NameIdxMapping.end();

  SmallVector<int, 8> EncodingIndices;
  for (unsigned I = 0, E = AL.getNumArgs(); I < E; ++I) {
    SourceRange SR;
    int32_t ArgIdx;

    if (AL.isArgIdent(I)) {
      IdentifierLoc *IdLoc = AL.getArgAsIdent(I);
      auto It = NameIdxMapping.find(IdLoc->Ident->getName());
      if (It == UnknownName) {
        S.Diag(AL.getLoc(), diag::err_callback_attribute_argument_unknown)
            << IdLoc->Ident << IdLoc->Loc;
        return;
      }

      SR = SourceRange(IdLoc->Loc);
      ArgIdx = It->second;
    } else if (AL.isArgExpr(I)) {
      Expr *IdxExpr = AL.getArgAsExpr(I);

      // If the expression is not parseable as an int32_t we have a problem.
      if (!checkUInt32Argument(S, AL, IdxExpr, (uint32_t &)ArgIdx, I + 1,
                               false)) {
        S.Diag(AL.getLoc(), diag::err_attribute_argument_out_of_bounds)
            << AL << (I + 1) << IdxExpr->getSourceRange();
        return;
      }

      // Check oob, excluding the special values, 0 and -1.
      if (ArgIdx < -1 || ArgIdx > NumArgs) {
        S.Diag(AL.getLoc(), diag::err_attribute_argument_out_of_bounds)
            << AL << (I + 1) << IdxExpr->getSourceRange();
        return;
      }

      SR = IdxExpr->getSourceRange();
    } else {
      llvm_unreachable("Unexpected ParsedAttr argument type!");
    }

    if (ArgIdx == 0 && !HasImplicitThisParam) {
      S.Diag(AL.getLoc(), diag::err_callback_implicit_this_not_available)
          << (I + 1) << SR;
      return;
    }

    // Adjust for the case we do not have an implicit "this" parameter. In this
    // case we decrease all positive values by 1 to get LLVM argument indices.
    if (!HasImplicitThisParam && ArgIdx > 0)
      ArgIdx -= 1;

    EncodingIndices.push_back(ArgIdx);
  }

  int CalleeIdx = EncodingIndices.front();
  // Check if the callee index is proper, thus not "this" and not "unknown".
  // This means the "CalleeIdx" has to be non-negative if "HasImplicitThisParam"
  // is false and positive if "HasImplicitThisParam" is true.
  if (CalleeIdx < (int)HasImplicitThisParam) {
    S.Diag(AL.getLoc(), diag::err_callback_attribute_invalid_callee)
        << AL.getRange();
    return;
  }

  // Get the callee type, note the index adjustment as the AST doesn't contain
  // the this type (which the callee cannot reference anyway!).
  const Type *CalleeType =
      getFunctionOrMethodParamType(D, CalleeIdx - HasImplicitThisParam)
          .getTypePtr();
  if (!CalleeType || !CalleeType->isFunctionPointerType()) {
    S.Diag(AL.getLoc(), diag::err_callback_callee_no_function_type)
        << AL.getRange();
    return;
  }

  const Type *CalleeFnType =
      CalleeType->getPointeeType()->getUnqualifiedDesugaredType();

  // TODO: Check the type of the callee arguments.

  const auto *CalleeFnProtoType = dyn_cast<FunctionProtoType>(CalleeFnType);
  if (!CalleeFnProtoType) {
    S.Diag(AL.getLoc(), diag::err_callback_callee_no_function_type)
        << AL.getRange();
    return;
  }

  if (CalleeFnProtoType->getNumParams() > EncodingIndices.size() - 1) {
    S.Diag(AL.getLoc(), diag::err_attribute_wrong_number_arguments)
        << AL << (unsigned)(EncodingIndices.size() - 1);
    return;
  }

  if (CalleeFnProtoType->getNumParams() < EncodingIndices.size() - 1) {
    S.Diag(AL.getLoc(), diag::err_attribute_wrong_number_arguments)
        << AL << (unsigned)(EncodingIndices.size() - 1);
    return;
  }

  if (CalleeFnProtoType->isVariadic()) {
    S.Diag(AL.getLoc(), diag::err_callback_callee_is_variadic) << AL.getRange();
    return;
  }

  // Do not allow multiple callback attributes.
  if (D->hasAttr<CallbackAttr>()) {
    S.Diag(AL.getLoc(), diag::err_callback_attribute_multiple) << AL.getRange();
    return;
  }

  D->addAttr(::new (S.Context) CallbackAttr(
      S.Context, AL, EncodingIndices.data(), EncodingIndices.size()));
}

static bool isFunctionLike(const Type &T) {
  // Check for explicit function types.
  // 'called_once' is only supported in Objective-C and it has
  // function pointers and block pointers.
  return T.isFunctionPointerType() || T.isBlockPointerType();
}

/// Handle 'called_once' attribute.
static void handleCalledOnceAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  // 'called_once' only applies to parameters representing functions.
  QualType T = cast<ParmVarDecl>(D)->getType();

  if (!isFunctionLike(*T)) {
    S.Diag(AL.getLoc(), diag::err_called_once_attribute_wrong_type);
    return;
  }

  D->addAttr(::new (S.Context) CalledOnceAttr(S.Context, AL));
}

static void handleTransparentUnionAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  // Try to find the underlying union declaration.
  RecordDecl *RD = nullptr;
  const auto *TD = dyn_cast<TypedefNameDecl>(D);
  if (TD && TD->getUnderlyingType()->isUnionType())
    RD = TD->getUnderlyingType()->getAsUnionType()->getDecl();
  else
    RD = dyn_cast<RecordDecl>(D);

  if (!RD || !RD->isUnion()) {
    S.Diag(AL.getLoc(), diag::warn_attribute_wrong_decl_type) << AL
                                                              << ExpectedUnion;
    return;
  }

  if (!RD->isCompleteDefinition()) {
    if (!RD->isBeingDefined())
      S.Diag(AL.getLoc(),
             diag::warn_transparent_union_attribute_not_definition);
    return;
  }

  RecordDecl::field_iterator Field = RD->field_begin(),
                          FieldEnd = RD->field_end();
  if (Field == FieldEnd) {
    S.Diag(AL.getLoc(), diag::warn_transparent_union_attribute_zero_fields);
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
      unsigned FieldBits = isSize ? S.Context.getTypeSize(FieldType)
                                  : S.Context.getTypeAlign(FieldType);
      S.Diag(Field->getLocation(),
             diag::warn_transparent_union_attribute_field_size_align)
          << isSize << *Field << FieldBits;
      unsigned FirstBits = isSize ? FirstSize : FirstAlign;
      S.Diag(FirstField->getLocation(),
             diag::note_transparent_union_first_field_size_align)
          << isSize << FirstBits;
      return;
    }
  }

  RD->addAttr(::new (S.Context) TransparentUnionAttr(S.Context, AL));
}

void Sema::AddAnnotationAttr(Decl *D, const AttributeCommonInfo &CI,
                             StringRef Str, MutableArrayRef<Expr *> Args) {
  auto *Attr = AnnotateAttr::Create(Context, Str, Args.data(), Args.size(), CI);
  llvm::SmallVector<PartialDiagnosticAt, 8> Notes;
  for (unsigned Idx = 0; Idx < Attr->args_size(); Idx++) {
    Expr *&E = Attr->args_begin()[Idx];
    assert(E && "error are handled before");
    if (E->isValueDependent() || E->isTypeDependent())
      continue;

    if (E->getType()->isArrayType())
      E = ImpCastExprToType(E, Context.getPointerType(E->getType()),
                            clang::CK_ArrayToPointerDecay)
              .get();
    if (E->getType()->isFunctionType())
      E = ImplicitCastExpr::Create(Context,
                                   Context.getPointerType(E->getType()),
                                   clang::CK_FunctionToPointerDecay, E, nullptr,
                                   VK_PRValue, FPOptionsOverride());
    if (E->isLValue())
      E = ImplicitCastExpr::Create(Context, E->getType().getNonReferenceType(),
                                   clang::CK_LValueToRValue, E, nullptr,
                                   VK_PRValue, FPOptionsOverride());

    Expr::EvalResult Eval;
    Notes.clear();
    Eval.Diag = &Notes;

    bool Result =
        E->EvaluateAsConstantExpr(Eval, Context);

    /// Result means the expression can be folded to a constant.
    /// Note.empty() means the expression is a valid constant expression in the
    /// current language mode.
    if (!Result || !Notes.empty()) {
      Diag(E->getBeginLoc(), diag::err_attribute_argument_n_type)
          << CI << (Idx + 1) << AANT_ArgumentConstantExpr;
      for (auto &Note : Notes)
        Diag(Note.first, Note.second);
      return;
    }
    assert(Eval.Val.hasValue());
    E = ConstantExpr::Create(Context, E, Eval.Val);
  }
  D->addAttr(Attr);
}

static void handleAnnotateAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  // Make sure that there is a string literal as the annotation's first
  // argument.
  StringRef Str;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, Str))
    return;

  llvm::SmallVector<Expr *, 4> Args;
  Args.reserve(AL.getNumArgs() - 1);
  for (unsigned Idx = 1; Idx < AL.getNumArgs(); Idx++) {
    assert(!AL.isArgIdent(Idx));
    Args.push_back(AL.getArgAsExpr(Idx));
  }

  S.AddAnnotationAttr(D, AL, Str, Args);
}

static void handleAlignValueAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  S.AddAlignValueAttr(D, AL, AL.getArgAsExpr(0));
}

void Sema::AddAlignValueAttr(Decl *D, const AttributeCommonInfo &CI, Expr *E) {
  AlignValueAttr TmpAttr(Context, CI, E);
  SourceLocation AttrLoc = CI.getLoc();

  QualType T;
  if (const auto *TD = dyn_cast<TypedefNameDecl>(D))
    T = TD->getUnderlyingType();
  else if (const auto *VD = dyn_cast<ValueDecl>(D))
    T = VD->getType();
  else
    llvm_unreachable("Unknown decl type for align_value");

  if (!T->isDependentType() && !T->isAnyPointerType() &&
      !T->isReferenceType() && !T->isMemberPointerType()) {
    Diag(AttrLoc, diag::warn_attribute_pointer_or_reference_only)
      << &TmpAttr << T << D->getSourceRange();
    return;
  }

  if (!E->isValueDependent()) {
    llvm::APSInt Alignment;
    ExprResult ICE = VerifyIntegerConstantExpression(
        E, &Alignment, diag::err_align_value_attribute_argument_not_int);
    if (ICE.isInvalid())
      return;

    if (!Alignment.isPowerOf2()) {
      Diag(AttrLoc, diag::err_alignment_not_power_of_two)
        << E->getSourceRange();
      return;
    }

    D->addAttr(::new (Context) AlignValueAttr(Context, CI, ICE.get()));
    return;
  }

  // Save dependent expressions in the AST to be instantiated.
  D->addAttr(::new (Context) AlignValueAttr(Context, CI, E));
}

static void handleAlignedAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  // check the attribute arguments.
  if (AL.getNumArgs() > 1) {
    S.Diag(AL.getLoc(), diag::err_attribute_wrong_number_arguments) << AL << 1;
    return;
  }

  if (AL.getNumArgs() == 0) {
    D->addAttr(::new (S.Context) AlignedAttr(S.Context, AL, true, nullptr));
    return;
  }

  Expr *E = AL.getArgAsExpr(0);
  if (AL.isPackExpansion() && !E->containsUnexpandedParameterPack()) {
    S.Diag(AL.getEllipsisLoc(),
           diag::err_pack_expansion_without_parameter_packs);
    return;
  }

  if (!AL.isPackExpansion() && S.DiagnoseUnexpandedParameterPack(E))
    return;

  S.AddAlignedAttr(D, AL, E, AL.isPackExpansion());
}

void Sema::AddAlignedAttr(Decl *D, const AttributeCommonInfo &CI, Expr *E,
                          bool IsPackExpansion) {
  AlignedAttr TmpAttr(Context, CI, true, E);
  SourceLocation AttrLoc = CI.getLoc();

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
    } else if (const auto *VD = dyn_cast<VarDecl>(D)) {
      if (VD->getStorageClass() == SC_Register)
        DiagKind = 1;
      if (VD->isExceptionVariable())
        DiagKind = 2;
    } else if (const auto *FD = dyn_cast<FieldDecl>(D)) {
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

  if (E->isValueDependent()) {
    // We can't support a dependent alignment on a non-dependent type,
    // because we have no way to model that a type is "alignment-dependent"
    // but not dependent in any other way.
    if (const auto *TND = dyn_cast<TypedefNameDecl>(D)) {
      if (!TND->getUnderlyingType()->isDependentType()) {
        Diag(AttrLoc, diag::err_alignment_dependent_typedef_name)
            << E->getSourceRange();
        return;
      }
    }

    // Save dependent expressions in the AST to be instantiated.
    AlignedAttr *AA = ::new (Context) AlignedAttr(Context, CI, true, E);
    AA->setPackExpansion(IsPackExpansion);
    D->addAttr(AA);
    return;
  }

  // FIXME: Cache the number on the AL object?
  llvm::APSInt Alignment;
  ExprResult ICE = VerifyIntegerConstantExpression(
      E, &Alignment, diag::err_aligned_attribute_argument_not_int);
  if (ICE.isInvalid())
    return;

  uint64_t AlignVal = Alignment.getZExtValue();
  // 16 byte ByVal alignment not due to a vector member is not honoured by XL
  // on AIX. Emit a warning here that users are generating binary incompatible
  // code to be safe.
  if (AlignVal >= 16 && isa<FieldDecl>(D) &&
      Context.getTargetInfo().getTriple().isOSAIX())
    Diag(AttrLoc, diag::warn_not_xl_compatible) << E->getSourceRange();

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

  uint64_t MaximumAlignment = Sema::MaximumAlignment;
  if (Context.getTargetInfo().getTriple().isOSBinFormatCOFF())
    MaximumAlignment = std::min(MaximumAlignment, uint64_t(8192));
  if (AlignVal > MaximumAlignment) {
    Diag(AttrLoc, diag::err_attribute_aligned_too_great)
        << MaximumAlignment << E->getSourceRange();
    return;
  }

  const auto *VD = dyn_cast<VarDecl>(D);
  if (VD && Context.getTargetInfo().isTLSSupported()) {
    unsigned MaxTLSAlign =
        Context.toCharUnitsFromBits(Context.getTargetInfo().getMaxTLSAlign())
            .getQuantity();
    if (MaxTLSAlign && AlignVal > MaxTLSAlign &&
        VD->getTLSKind() != VarDecl::TLS_None) {
      Diag(VD->getLocation(), diag::err_tls_var_aligned_over_maximum)
          << (unsigned)AlignVal << VD << MaxTLSAlign;
      return;
    }
  }

  // On AIX, an aligned attribute can not decrease the alignment when applied
  // to a variable declaration with vector type.
  if (VD && Context.getTargetInfo().getTriple().isOSAIX()) {
    const Type *Ty = VD->getType().getTypePtr();
    if (Ty->isVectorType() && AlignVal < 16) {
      Diag(VD->getLocation(), diag::warn_aligned_attr_underaligned)
          << VD->getType() << 16;
      return;
    }
  }

  AlignedAttr *AA = ::new (Context) AlignedAttr(Context, CI, true, ICE.get());
  AA->setPackExpansion(IsPackExpansion);
  D->addAttr(AA);
}

void Sema::AddAlignedAttr(Decl *D, const AttributeCommonInfo &CI,
                          TypeSourceInfo *TS, bool IsPackExpansion) {
  // FIXME: Cache the number on the AL object if non-dependent?
  // FIXME: Perform checking of type validity
  AlignedAttr *AA = ::new (Context) AlignedAttr(Context, CI, false, TS);
  AA->setPackExpansion(IsPackExpansion);
  D->addAttr(AA);
}

void Sema::CheckAlignasUnderalignment(Decl *D) {
  assert(D->hasAttrs() && "no attributes on decl");

  QualType UnderlyingTy, DiagTy;
  if (const auto *VD = dyn_cast<ValueDecl>(D)) {
    UnderlyingTy = DiagTy = VD->getType();
  } else {
    UnderlyingTy = DiagTy = Context.getTagDeclType(cast<TagDecl>(D));
    if (const auto *ED = dyn_cast<EnumDecl>(D))
      UnderlyingTy = ED->getIntegerType();
  }
  if (DiagTy->isDependentType() || DiagTy->isIncompleteType())
    return;

  // C++11 [dcl.align]p5, C11 6.7.5/4:
  //   The combined effect of all alignment attributes in a declaration shall
  //   not specify an alignment that is less strict than the alignment that
  //   would otherwise be required for the entity being declared.
  AlignedAttr *AlignasAttr = nullptr;
  AlignedAttr *LastAlignedAttr = nullptr;
  unsigned Align = 0;
  for (auto *I : D->specific_attrs<AlignedAttr>()) {
    if (I->isAlignmentDependent())
      return;
    if (I->isAlignas())
      AlignasAttr = I;
    Align = std::max(Align, I->getAlignment(Context));
    LastAlignedAttr = I;
  }

  if (Align && DiagTy->isSizelessType()) {
    Diag(LastAlignedAttr->getLocation(), diag::err_attribute_sizeless_type)
        << LastAlignedAttr << DiagTy;
  } else if (AlignasAttr && Align) {
    CharUnits RequestedAlign = Context.toCharUnitsFromBits(Align);
    CharUnits NaturalAlign = Context.getTypeAlignInChars(UnderlyingTy);
    if (NaturalAlign > RequestedAlign)
      Diag(AlignasAttr->getLocation(), diag::err_alignas_underaligned)
        << DiagTy << (unsigned)NaturalAlign.getQuantity();
  }
}

bool Sema::checkMSInheritanceAttrOnDefinition(
    CXXRecordDecl *RD, SourceRange Range, bool BestCase,
    MSInheritanceModel ExplicitModel) {
  assert(RD->hasDefinition() && "RD has no definition!");

  // We may not have seen base specifiers or any virtual methods yet.  We will
  // have to wait until the record is defined to catch any mismatches.
  if (!RD->getDefinition()->isCompleteDefinition())
    return false;

  // The unspecified model never matches what a definition could need.
  if (ExplicitModel == MSInheritanceModel::Unspecified)
    return false;

  if (BestCase) {
    if (RD->calculateInheritanceModel() == ExplicitModel)
      return false;
  } else {
    if (RD->calculateInheritanceModel() <= ExplicitModel)
      return false;
  }

  Diag(Range.getBegin(), diag::err_mismatched_ms_inheritance)
      << 0 /*definition*/;
  Diag(RD->getDefinition()->getLocation(), diag::note_defined_here) << RD;
  return true;
}

/// parseModeAttrArg - Parses attribute mode string and returns parsed type
/// attribute.
static void parseModeAttrArg(Sema &S, StringRef Str, unsigned &DestWidth,
                             bool &IntegerMode, bool &ComplexMode,
                             bool &ExplicitIEEE) {
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
    case 'K': // KFmode - IEEE quad precision (__float128)
      ExplicitIEEE = true;
      DestWidth = Str[1] == 'I' ? 0 : 128;
      break;
    case 'T':
      ExplicitIEEE = false;
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
static void handleModeAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  // This attribute isn't documented, but glibc uses it.  It changes
  // the width of an int or unsigned int to the specified size.
  if (!AL.isArgIdent(0)) {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_type)
        << AL << AANT_ArgumentIdentifier;
    return;
  }

  IdentifierInfo *Name = AL.getArgAsIdent(0)->Ident;

  S.AddModeAttr(D, AL, Name);
}

void Sema::AddModeAttr(Decl *D, const AttributeCommonInfo &CI,
                       IdentifierInfo *Name, bool InInstantiation) {
  StringRef Str = Name->getName();
  normalizeName(Str);
  SourceLocation AttrLoc = CI.getLoc();

  unsigned DestWidth = 0;
  bool IntegerMode = true;
  bool ComplexMode = false;
  bool ExplicitIEEE = false;
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
                       IntegerMode, ComplexMode, ExplicitIEEE);
      // Avoid duplicate warning from template instantiation.
      if (!InInstantiation)
        Diag(AttrLoc, diag::warn_vector_mode_deprecated);
    } else {
      VectorSize = 0;
    }
  }

  if (!VectorSize)
    parseModeAttrArg(*this, Str, DestWidth, IntegerMode, ComplexMode,
                     ExplicitIEEE);

  // FIXME: Sync this with InitializePredefinedMacros; we need to match int8_t
  // and friends, at least with glibc.
  // FIXME: Make sure floating-point mappings are accurate
  // FIXME: Support XF and TF types
  if (!DestWidth) {
    Diag(AttrLoc, diag::err_machine_mode) << 0 /*Unknown*/ << Name;
    return;
  }

  QualType OldTy;
  if (const auto *TD = dyn_cast<TypedefNameDecl>(D))
    OldTy = TD->getUnderlyingType();
  else if (const auto *ED = dyn_cast<EnumDecl>(D)) {
    // Something like 'typedef enum { X } __attribute__((mode(XX))) T;'.
    // Try to get type from enum declaration, default to int.
    OldTy = ED->getIntegerType();
    if (OldTy.isNull())
      OldTy = Context.IntTy;
  } else
    OldTy = cast<ValueDecl>(D)->getType();

  if (OldTy->isDependentType()) {
    D->addAttr(::new (Context) ModeAttr(Context, CI, Name));
    return;
  }

  // Base type can also be a vector type (see PR17453).
  // Distinguish between base type and base element type.
  QualType OldElemTy = OldTy;
  if (const auto *VT = OldTy->getAs<VectorType>())
    OldElemTy = VT->getElementType();

  // GCC allows 'mode' attribute on enumeration types (even incomplete), except
  // for vector modes. So, 'enum X __attribute__((mode(QI)));' forms a complete
  // type, 'enum { A } __attribute__((mode(V4SI)))' is rejected.
  if ((isa<EnumDecl>(D) || OldElemTy->getAs<EnumType>()) &&
      VectorSize.getBoolValue()) {
    Diag(AttrLoc, diag::err_enum_mode_vector_type) << Name << CI.getRange();
    return;
  }
  bool IntegralOrAnyEnumType = (OldElemTy->isIntegralOrEnumerationType() &&
                                !OldElemTy->isExtIntType()) ||
                               OldElemTy->getAs<EnumType>();

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
    NewElemTy = Context.getRealTypeForBitwidth(DestWidth, ExplicitIEEE);

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
  } else if (const auto *OldVT = OldTy->getAs<VectorType>()) {
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
  if (auto *TD = dyn_cast<TypedefNameDecl>(D))
    TD->setModedTypeSourceInfo(TD->getTypeSourceInfo(), NewTy);
  else if (auto *ED = dyn_cast<EnumDecl>(D))
    ED->setIntegerType(NewTy);
  else
    cast<ValueDecl>(D)->setType(NewTy);

  D->addAttr(::new (Context) ModeAttr(Context, CI, Name));
}

static void handleNoDebugAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  D->addAttr(::new (S.Context) NoDebugAttr(S.Context, AL));
}

AlwaysInlineAttr *Sema::mergeAlwaysInlineAttr(Decl *D,
                                              const AttributeCommonInfo &CI,
                                              const IdentifierInfo *Ident) {
  if (OptimizeNoneAttr *Optnone = D->getAttr<OptimizeNoneAttr>()) {
    Diag(CI.getLoc(), diag::warn_attribute_ignored) << Ident;
    Diag(Optnone->getLocation(), diag::note_conflicting_attribute);
    return nullptr;
  }

  if (D->hasAttr<AlwaysInlineAttr>())
    return nullptr;

  return ::new (Context) AlwaysInlineAttr(Context, CI);
}

InternalLinkageAttr *Sema::mergeInternalLinkageAttr(Decl *D,
                                                    const ParsedAttr &AL) {
  if (const auto *VD = dyn_cast<VarDecl>(D)) {
    // Attribute applies to Var but not any subclass of it (like ParmVar,
    // ImplicitParm or VarTemplateSpecialization).
    if (VD->getKind() != Decl::Var) {
      Diag(AL.getLoc(), diag::warn_attribute_wrong_decl_type)
          << AL << (getLangOpts().CPlusPlus ? ExpectedFunctionVariableOrClass
                                            : ExpectedVariableOrFunction);
      return nullptr;
    }
    // Attribute does not apply to non-static local variables.
    if (VD->hasLocalStorage()) {
      Diag(VD->getLocation(), diag::warn_internal_linkage_local_storage);
      return nullptr;
    }
  }

  return ::new (Context) InternalLinkageAttr(Context, AL);
}
InternalLinkageAttr *
Sema::mergeInternalLinkageAttr(Decl *D, const InternalLinkageAttr &AL) {
  if (const auto *VD = dyn_cast<VarDecl>(D)) {
    // Attribute applies to Var but not any subclass of it (like ParmVar,
    // ImplicitParm or VarTemplateSpecialization).
    if (VD->getKind() != Decl::Var) {
      Diag(AL.getLocation(), diag::warn_attribute_wrong_decl_type)
          << &AL << (getLangOpts().CPlusPlus ? ExpectedFunctionVariableOrClass
                                             : ExpectedVariableOrFunction);
      return nullptr;
    }
    // Attribute does not apply to non-static local variables.
    if (VD->hasLocalStorage()) {
      Diag(VD->getLocation(), diag::warn_internal_linkage_local_storage);
      return nullptr;
    }
  }

  return ::new (Context) InternalLinkageAttr(Context, AL);
}

MinSizeAttr *Sema::mergeMinSizeAttr(Decl *D, const AttributeCommonInfo &CI) {
  if (OptimizeNoneAttr *Optnone = D->getAttr<OptimizeNoneAttr>()) {
    Diag(CI.getLoc(), diag::warn_attribute_ignored) << "'minsize'";
    Diag(Optnone->getLocation(), diag::note_conflicting_attribute);
    return nullptr;
  }

  if (D->hasAttr<MinSizeAttr>())
    return nullptr;

  return ::new (Context) MinSizeAttr(Context, CI);
}

SwiftNameAttr *Sema::mergeSwiftNameAttr(Decl *D, const SwiftNameAttr &SNA,
                                        StringRef Name) {
  if (const auto *PrevSNA = D->getAttr<SwiftNameAttr>()) {
    if (PrevSNA->getName() != Name && !PrevSNA->isImplicit()) {
      Diag(PrevSNA->getLocation(), diag::err_attributes_are_not_compatible)
          << PrevSNA << &SNA;
      Diag(SNA.getLoc(), diag::note_conflicting_attribute);
    }

    D->dropAttr<SwiftNameAttr>();
  }
  return ::new (Context) SwiftNameAttr(Context, SNA, Name);
}

OptimizeNoneAttr *Sema::mergeOptimizeNoneAttr(Decl *D,
                                              const AttributeCommonInfo &CI) {
  if (AlwaysInlineAttr *Inline = D->getAttr<AlwaysInlineAttr>()) {
    Diag(Inline->getLocation(), diag::warn_attribute_ignored) << Inline;
    Diag(CI.getLoc(), diag::note_conflicting_attribute);
    D->dropAttr<AlwaysInlineAttr>();
  }
  if (MinSizeAttr *MinSize = D->getAttr<MinSizeAttr>()) {
    Diag(MinSize->getLocation(), diag::warn_attribute_ignored) << MinSize;
    Diag(CI.getLoc(), diag::note_conflicting_attribute);
    D->dropAttr<MinSizeAttr>();
  }

  if (D->hasAttr<OptimizeNoneAttr>())
    return nullptr;

  return ::new (Context) OptimizeNoneAttr(Context, CI);
}

static void handleAlwaysInlineAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (AlwaysInlineAttr *Inline =
          S.mergeAlwaysInlineAttr(D, AL, AL.getAttrName()))
    D->addAttr(Inline);
}

static void handleMinSizeAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (MinSizeAttr *MinSize = S.mergeMinSizeAttr(D, AL))
    D->addAttr(MinSize);
}

static void handleOptimizeNoneAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (OptimizeNoneAttr *Optnone = S.mergeOptimizeNoneAttr(D, AL))
    D->addAttr(Optnone);
}

static void handleConstantAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  const auto *VD = cast<VarDecl>(D);
  if (VD->hasLocalStorage()) {
    S.Diag(AL.getLoc(), diag::err_cuda_nonstatic_constdev);
    return;
  }
  // constexpr variable may already get an implicit constant attr, which should
  // be replaced by the explicit constant attr.
  if (auto *A = D->getAttr<CUDAConstantAttr>()) {
    if (!A->isImplicit())
      return;
    D->dropAttr<CUDAConstantAttr>();
  }
  D->addAttr(::new (S.Context) CUDAConstantAttr(S.Context, AL));
}

static void handleSharedAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  const auto *VD = cast<VarDecl>(D);
  // extern __shared__ is only allowed on arrays with no length (e.g.
  // "int x[]").
  if (!S.getLangOpts().GPURelocatableDeviceCode && VD->hasExternalStorage() &&
      !isa<IncompleteArrayType>(VD->getType())) {
    S.Diag(AL.getLoc(), diag::err_cuda_extern_shared) << VD;
    return;
  }
  if (S.getLangOpts().CUDA && VD->hasLocalStorage() &&
      S.CUDADiagIfHostCode(AL.getLoc(), diag::err_cuda_host_shared)
          << S.CurrentCUDATarget())
    return;
  D->addAttr(::new (S.Context) CUDASharedAttr(S.Context, AL));
}

static void handleGlobalAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  const auto *FD = cast<FunctionDecl>(D);
  if (!FD->getReturnType()->isVoidType() &&
      !FD->getReturnType()->getAs<AutoType>() &&
      !FD->getReturnType()->isInstantiationDependentType()) {
    SourceRange RTRange = FD->getReturnTypeSourceRange();
    S.Diag(FD->getTypeSpecStartLoc(), diag::err_kern_type_not_void_return)
        << FD->getType()
        << (RTRange.isValid() ? FixItHint::CreateReplacement(RTRange, "void")
                              : FixItHint());
    return;
  }
  if (const auto *Method = dyn_cast<CXXMethodDecl>(FD)) {
    if (Method->isInstance()) {
      S.Diag(Method->getBeginLoc(), diag::err_kern_is_nonstatic_method)
          << Method;
      return;
    }
    S.Diag(Method->getBeginLoc(), diag::warn_kern_is_method) << Method;
  }
  // Only warn for "inline" when compiling for host, to cut down on noise.
  if (FD->isInlineSpecified() && !S.getLangOpts().CUDAIsDevice)
    S.Diag(FD->getBeginLoc(), diag::warn_kern_is_inline) << FD;

  D->addAttr(::new (S.Context) CUDAGlobalAttr(S.Context, AL));
  // In host compilation the kernel is emitted as a stub function, which is
  // a helper function for launching the kernel. The instructions in the helper
  // function has nothing to do with the source code of the kernel. Do not emit
  // debug info for the stub function to avoid confusing the debugger.
  if (S.LangOpts.HIP && !S.LangOpts.CUDAIsDevice)
    D->addAttr(NoDebugAttr::CreateImplicit(S.Context));
}

static void handleDeviceAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (const auto *VD = dyn_cast<VarDecl>(D)) {
    if (VD->hasLocalStorage()) {
      S.Diag(AL.getLoc(), diag::err_cuda_nonstatic_constdev);
      return;
    }
  }

  if (auto *A = D->getAttr<CUDADeviceAttr>()) {
    if (!A->isImplicit())
      return;
    D->dropAttr<CUDADeviceAttr>();
  }
  D->addAttr(::new (S.Context) CUDADeviceAttr(S.Context, AL));
}

static void handleManagedAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (const auto *VD = dyn_cast<VarDecl>(D)) {
    if (VD->hasLocalStorage()) {
      S.Diag(AL.getLoc(), diag::err_cuda_nonstatic_constdev);
      return;
    }
  }
  if (!D->hasAttr<HIPManagedAttr>())
    D->addAttr(::new (S.Context) HIPManagedAttr(S.Context, AL));
  if (!D->hasAttr<CUDADeviceAttr>())
    D->addAttr(CUDADeviceAttr::CreateImplicit(S.Context));
}

static void handleGNUInlineAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  const auto *Fn = cast<FunctionDecl>(D);
  if (!Fn->isInlineSpecified()) {
    S.Diag(AL.getLoc(), diag::warn_gnu_inline_attribute_requires_inline);
    return;
  }

  if (S.LangOpts.CPlusPlus && Fn->getStorageClass() != SC_Extern)
    S.Diag(AL.getLoc(), diag::warn_gnu_inline_cplusplus_without_extern);

  D->addAttr(::new (S.Context) GNUInlineAttr(S.Context, AL));
}

static void handleCallConvAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (hasDeclarator(D)) return;

  // Diagnostic is emitted elsewhere: here we store the (valid) AL
  // in the Decl node for syntactic reasoning, e.g., pretty-printing.
  CallingConv CC;
  if (S.CheckCallingConvAttr(AL, CC, /*FD*/nullptr))
    return;

  if (!isa<ObjCMethodDecl>(D)) {
    S.Diag(AL.getLoc(), diag::warn_attribute_wrong_decl_type)
        << AL << ExpectedFunctionOrMethod;
    return;
  }

  switch (AL.getKind()) {
  case ParsedAttr::AT_FastCall:
    D->addAttr(::new (S.Context) FastCallAttr(S.Context, AL));
    return;
  case ParsedAttr::AT_StdCall:
    D->addAttr(::new (S.Context) StdCallAttr(S.Context, AL));
    return;
  case ParsedAttr::AT_ThisCall:
    D->addAttr(::new (S.Context) ThisCallAttr(S.Context, AL));
    return;
  case ParsedAttr::AT_CDecl:
    D->addAttr(::new (S.Context) CDeclAttr(S.Context, AL));
    return;
  case ParsedAttr::AT_Pascal:
    D->addAttr(::new (S.Context) PascalAttr(S.Context, AL));
    return;
  case ParsedAttr::AT_SwiftCall:
    D->addAttr(::new (S.Context) SwiftCallAttr(S.Context, AL));
    return;
  case ParsedAttr::AT_SwiftAsyncCall:
    D->addAttr(::new (S.Context) SwiftAsyncCallAttr(S.Context, AL));
    return;
  case ParsedAttr::AT_VectorCall:
    D->addAttr(::new (S.Context) VectorCallAttr(S.Context, AL));
    return;
  case ParsedAttr::AT_MSABI:
    D->addAttr(::new (S.Context) MSABIAttr(S.Context, AL));
    return;
  case ParsedAttr::AT_SysVABI:
    D->addAttr(::new (S.Context) SysVABIAttr(S.Context, AL));
    return;
  case ParsedAttr::AT_RegCall:
    D->addAttr(::new (S.Context) RegCallAttr(S.Context, AL));
    return;
  case ParsedAttr::AT_Pcs: {
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

    D->addAttr(::new (S.Context) PcsAttr(S.Context, AL, PCS));
    return;
  }
  case ParsedAttr::AT_AArch64VectorPcs:
    D->addAttr(::new (S.Context) AArch64VectorPcsAttr(S.Context, AL));
    return;
  case ParsedAttr::AT_IntelOclBicc:
    D->addAttr(::new (S.Context) IntelOclBiccAttr(S.Context, AL));
    return;
  case ParsedAttr::AT_PreserveMost:
    D->addAttr(::new (S.Context) PreserveMostAttr(S.Context, AL));
    return;
  case ParsedAttr::AT_PreserveAll:
    D->addAttr(::new (S.Context) PreserveAllAttr(S.Context, AL));
    return;
  default:
    llvm_unreachable("unexpected attribute kind");
  }
}

static void handleSuppressAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (!AL.checkAtLeastNumArgs(S, 1))
    return;

  std::vector<StringRef> DiagnosticIdentifiers;
  for (unsigned I = 0, E = AL.getNumArgs(); I != E; ++I) {
    StringRef RuleName;

    if (!S.checkStringLiteralArgumentAttr(AL, I, RuleName, nullptr))
      return;

    // FIXME: Warn if the rule name is unknown. This is tricky because only
    // clang-tidy knows about available rules.
    DiagnosticIdentifiers.push_back(RuleName);
  }
  D->addAttr(::new (S.Context)
                 SuppressAttr(S.Context, AL, DiagnosticIdentifiers.data(),
                              DiagnosticIdentifiers.size()));
}

static void handleLifetimeCategoryAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  TypeSourceInfo *DerefTypeLoc = nullptr;
  QualType ParmType;
  if (AL.hasParsedType()) {
    ParmType = S.GetTypeFromParser(AL.getTypeArg(), &DerefTypeLoc);

    unsigned SelectIdx = ~0U;
    if (ParmType->isReferenceType())
      SelectIdx = 0;
    else if (ParmType->isArrayType())
      SelectIdx = 1;

    if (SelectIdx != ~0U) {
      S.Diag(AL.getLoc(), diag::err_attribute_invalid_argument)
          << SelectIdx << AL;
      return;
    }
  }

  // To check if earlier decl attributes do not conflict the newly parsed ones
  // we always add (and check) the attribute to the canonical decl. We need
  // to repeat the check for attribute mutual exclusion because we're attaching
  // all of the attributes to the canonical declaration rather than the current
  // declaration.
  D = D->getCanonicalDecl();
  if (AL.getKind() == ParsedAttr::AT_Owner) {
    if (checkAttrMutualExclusion<PointerAttr>(S, D, AL))
      return;
    if (const auto *OAttr = D->getAttr<OwnerAttr>()) {
      const Type *ExistingDerefType = OAttr->getDerefTypeLoc()
                                          ? OAttr->getDerefType().getTypePtr()
                                          : nullptr;
      if (ExistingDerefType != ParmType.getTypePtrOrNull()) {
        S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible)
            << AL << OAttr;
        S.Diag(OAttr->getLocation(), diag::note_conflicting_attribute);
      }
      return;
    }
    for (Decl *Redecl : D->redecls()) {
      Redecl->addAttr(::new (S.Context) OwnerAttr(S.Context, AL, DerefTypeLoc));
    }
  } else {
    if (checkAttrMutualExclusion<OwnerAttr>(S, D, AL))
      return;
    if (const auto *PAttr = D->getAttr<PointerAttr>()) {
      const Type *ExistingDerefType = PAttr->getDerefTypeLoc()
                                          ? PAttr->getDerefType().getTypePtr()
                                          : nullptr;
      if (ExistingDerefType != ParmType.getTypePtrOrNull()) {
        S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible)
            << AL << PAttr;
        S.Diag(PAttr->getLocation(), diag::note_conflicting_attribute);
      }
      return;
    }
    for (Decl *Redecl : D->redecls()) {
      Redecl->addAttr(::new (S.Context)
                          PointerAttr(S.Context, AL, DerefTypeLoc));
    }
  }
}

bool Sema::CheckCallingConvAttr(const ParsedAttr &Attrs, CallingConv &CC,
                                const FunctionDecl *FD) {
  if (Attrs.isInvalid())
    return true;

  if (Attrs.hasProcessingCache()) {
    CC = (CallingConv) Attrs.getProcessingCache();
    return false;
  }

  unsigned ReqArgs = Attrs.getKind() == ParsedAttr::AT_Pcs ? 1 : 0;
  if (!Attrs.checkExactlyNumArgs(*this, ReqArgs)) {
    Attrs.setInvalid();
    return true;
  }

  // TODO: diagnose uses of these conventions on the wrong target.
  switch (Attrs.getKind()) {
  case ParsedAttr::AT_CDecl:
    CC = CC_C;
    break;
  case ParsedAttr::AT_FastCall:
    CC = CC_X86FastCall;
    break;
  case ParsedAttr::AT_StdCall:
    CC = CC_X86StdCall;
    break;
  case ParsedAttr::AT_ThisCall:
    CC = CC_X86ThisCall;
    break;
  case ParsedAttr::AT_Pascal:
    CC = CC_X86Pascal;
    break;
  case ParsedAttr::AT_SwiftCall:
    CC = CC_Swift;
    break;
  case ParsedAttr::AT_SwiftAsyncCall:
    CC = CC_SwiftAsync;
    break;
  case ParsedAttr::AT_VectorCall:
    CC = CC_X86VectorCall;
    break;
  case ParsedAttr::AT_AArch64VectorPcs:
    CC = CC_AArch64VectorCall;
    break;
  case ParsedAttr::AT_RegCall:
    CC = CC_X86RegCall;
    break;
  case ParsedAttr::AT_MSABI:
    CC = Context.getTargetInfo().getTriple().isOSWindows() ? CC_C :
                                                             CC_Win64;
    break;
  case ParsedAttr::AT_SysVABI:
    CC = Context.getTargetInfo().getTriple().isOSWindows() ? CC_X86_64SysV :
                                                             CC_C;
    break;
  case ParsedAttr::AT_Pcs: {
    StringRef StrRef;
    if (!checkStringLiteralArgumentAttr(Attrs, 0, StrRef)) {
      Attrs.setInvalid();
      return true;
    }
    if (StrRef == "aapcs") {
      CC = CC_AAPCS;
      break;
    } else if (StrRef == "aapcs-vfp") {
      CC = CC_AAPCS_VFP;
      break;
    }

    Attrs.setInvalid();
    Diag(Attrs.getLoc(), diag::err_invalid_pcs);
    return true;
  }
  case ParsedAttr::AT_IntelOclBicc:
    CC = CC_IntelOclBicc;
    break;
  case ParsedAttr::AT_PreserveMost:
    CC = CC_PreserveMost;
    break;
  case ParsedAttr::AT_PreserveAll:
    CC = CC_PreserveAll;
    break;
  default: llvm_unreachable("unexpected attribute kind");
  }

  TargetInfo::CallingConvCheckResult A = TargetInfo::CCCR_OK;
  const TargetInfo &TI = Context.getTargetInfo();
  // CUDA functions may have host and/or device attributes which indicate
  // their targeted execution environment, therefore the calling convention
  // of functions in CUDA should be checked against the target deduced based
  // on their host/device attributes.
  if (LangOpts.CUDA) {
    auto *Aux = Context.getAuxTargetInfo();
    auto CudaTarget = IdentifyCUDATarget(FD);
    bool CheckHost = false, CheckDevice = false;
    switch (CudaTarget) {
    case CFT_HostDevice:
      CheckHost = true;
      CheckDevice = true;
      break;
    case CFT_Host:
      CheckHost = true;
      break;
    case CFT_Device:
    case CFT_Global:
      CheckDevice = true;
      break;
    case CFT_InvalidTarget:
      llvm_unreachable("unexpected cuda target");
    }
    auto *HostTI = LangOpts.CUDAIsDevice ? Aux : &TI;
    auto *DeviceTI = LangOpts.CUDAIsDevice ? &TI : Aux;
    if (CheckHost && HostTI)
      A = HostTI->checkCallingConvention(CC);
    if (A == TargetInfo::CCCR_OK && CheckDevice && DeviceTI)
      A = DeviceTI->checkCallingConvention(CC);
  } else {
    A = TI.checkCallingConvention(CC);
  }

  switch (A) {
  case TargetInfo::CCCR_OK:
    break;

  case TargetInfo::CCCR_Ignore:
    // Treat an ignored convention as if it was an explicit C calling convention
    // attribute. For example, __stdcall on Win x64 functions as __cdecl, so
    // that command line flags that change the default convention to
    // __vectorcall don't affect declarations marked __stdcall.
    CC = CC_C;
    break;

  case TargetInfo::CCCR_Error:
    Diag(Attrs.getLoc(), diag::error_cconv_unsupported)
        << Attrs << (int)CallingConventionIgnoredReason::ForThisTarget;
    break;

  case TargetInfo::CCCR_Warning: {
    Diag(Attrs.getLoc(), diag::warn_cconv_unsupported)
        << Attrs << (int)CallingConventionIgnoredReason::ForThisTarget;

    // This convention is not valid for the target. Use the default function or
    // method calling convention.
    bool IsCXXMethod = false, IsVariadic = false;
    if (FD) {
      IsCXXMethod = FD->isCXXInstanceMember();
      IsVariadic = FD->isVariadic();
    }
    CC = Context.getDefaultCallingConvention(IsVariadic, IsCXXMethod);
    break;
  }
  }

  Attrs.setProcessingCache((unsigned) CC);
  return false;
}

/// Pointer-like types in the default address space.
static bool isValidSwiftContextType(QualType Ty) {
  if (!Ty->hasPointerRepresentation())
    return Ty->isDependentType();
  return Ty->getPointeeType().getAddressSpace() == LangAS::Default;
}

/// Pointers and references in the default address space.
static bool isValidSwiftIndirectResultType(QualType Ty) {
  if (const auto *PtrType = Ty->getAs<PointerType>()) {
    Ty = PtrType->getPointeeType();
  } else if (const auto *RefType = Ty->getAs<ReferenceType>()) {
    Ty = RefType->getPointeeType();
  } else {
    return Ty->isDependentType();
  }
  return Ty.getAddressSpace() == LangAS::Default;
}

/// Pointers and references to pointers in the default address space.
static bool isValidSwiftErrorResultType(QualType Ty) {
  if (const auto *PtrType = Ty->getAs<PointerType>()) {
    Ty = PtrType->getPointeeType();
  } else if (const auto *RefType = Ty->getAs<ReferenceType>()) {
    Ty = RefType->getPointeeType();
  } else {
    return Ty->isDependentType();
  }
  if (!Ty.getQualifiers().empty())
    return false;
  return isValidSwiftContextType(Ty);
}

void Sema::AddParameterABIAttr(Decl *D, const AttributeCommonInfo &CI,
                               ParameterABI abi) {

  QualType type = cast<ParmVarDecl>(D)->getType();

  if (auto existingAttr = D->getAttr<ParameterABIAttr>()) {
    if (existingAttr->getABI() != abi) {
      Diag(CI.getLoc(), diag::err_attributes_are_not_compatible)
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
      Diag(CI.getLoc(), diag::err_swift_abi_parameter_wrong_type)
          << getParameterABISpelling(abi) << /*pointer to pointer */ 0 << type;
    }
    D->addAttr(::new (Context) SwiftContextAttr(Context, CI));
    return;

  case ParameterABI::SwiftAsyncContext:
    if (!isValidSwiftContextType(type)) {
      Diag(CI.getLoc(), diag::err_swift_abi_parameter_wrong_type)
          << getParameterABISpelling(abi) << /*pointer to pointer */ 0 << type;
    }
    D->addAttr(::new (Context) SwiftAsyncContextAttr(Context, CI));
    return;

  case ParameterABI::SwiftErrorResult:
    if (!isValidSwiftErrorResultType(type)) {
      Diag(CI.getLoc(), diag::err_swift_abi_parameter_wrong_type)
          << getParameterABISpelling(abi) << /*pointer to pointer */ 1 << type;
    }
    D->addAttr(::new (Context) SwiftErrorResultAttr(Context, CI));
    return;

  case ParameterABI::SwiftIndirectResult:
    if (!isValidSwiftIndirectResultType(type)) {
      Diag(CI.getLoc(), diag::err_swift_abi_parameter_wrong_type)
          << getParameterABISpelling(abi) << /*pointer*/ 0 << type;
    }
    D->addAttr(::new (Context) SwiftIndirectResultAttr(Context, CI));
    return;
  }
  llvm_unreachable("bad parameter ABI attribute");
}

/// Checks a regparm attribute, returning true if it is ill-formed and
/// otherwise setting numParams to the appropriate value.
bool Sema::CheckRegparmAttr(const ParsedAttr &AL, unsigned &numParams) {
  if (AL.isInvalid())
    return true;

  if (!AL.checkExactlyNumArgs(*this, 1)) {
    AL.setInvalid();
    return true;
  }

  uint32_t NP;
  Expr *NumParamsExpr = AL.getArgAsExpr(0);
  if (!checkUInt32Argument(*this, AL, NumParamsExpr, NP)) {
    AL.setInvalid();
    return true;
  }

  if (Context.getTargetInfo().getRegParmMax() == 0) {
    Diag(AL.getLoc(), diag::err_attribute_regparm_wrong_platform)
      << NumParamsExpr->getSourceRange();
    AL.setInvalid();
    return true;
  }

  numParams = NP;
  if (numParams > Context.getTargetInfo().getRegParmMax()) {
    Diag(AL.getLoc(), diag::err_attribute_regparm_invalid_number)
      << Context.getTargetInfo().getRegParmMax() << NumParamsExpr->getSourceRange();
    AL.setInvalid();
    return true;
  }

  return false;
}

// Checks whether an argument of launch_bounds attribute is
// acceptable, performs implicit conversion to Rvalue, and returns
// non-nullptr Expr result on success. Otherwise, it returns nullptr
// and may output an error.
static Expr *makeLaunchBoundsArgExpr(Sema &S, Expr *E,
                                     const CUDALaunchBoundsAttr &AL,
                                     const unsigned Idx) {
  if (S.DiagnoseUnexpandedParameterPack(E))
    return nullptr;

  // Accept template arguments for now as they depend on something else.
  // We'll get to check them when they eventually get instantiated.
  if (E->isValueDependent())
    return E;

  Optional<llvm::APSInt> I = llvm::APSInt(64);
  if (!(I = E->getIntegerConstantExpr(S.Context))) {
    S.Diag(E->getExprLoc(), diag::err_attribute_argument_n_type)
        << &AL << Idx << AANT_ArgumentIntegerConstant << E->getSourceRange();
    return nullptr;
  }
  // Make sure we can fit it in 32 bits.
  if (!I->isIntN(32)) {
    S.Diag(E->getExprLoc(), diag::err_ice_too_large)
        << toString(*I, 10, false) << 32 << /* Unsigned */ 1;
    return nullptr;
  }
  if (*I < 0)
    S.Diag(E->getExprLoc(), diag::warn_attribute_argument_n_negative)
        << &AL << Idx << E->getSourceRange();

  // We may need to perform implicit conversion of the argument.
  InitializedEntity Entity = InitializedEntity::InitializeParameter(
      S.Context, S.Context.getConstType(S.Context.IntTy), /*consume*/ false);
  ExprResult ValArg = S.PerformCopyInitialization(Entity, SourceLocation(), E);
  assert(!ValArg.isInvalid() &&
         "Unexpected PerformCopyInitialization() failure.");

  return ValArg.getAs<Expr>();
}

void Sema::AddLaunchBoundsAttr(Decl *D, const AttributeCommonInfo &CI,
                               Expr *MaxThreads, Expr *MinBlocks) {
  CUDALaunchBoundsAttr TmpAttr(Context, CI, MaxThreads, MinBlocks);
  MaxThreads = makeLaunchBoundsArgExpr(*this, MaxThreads, TmpAttr, 0);
  if (MaxThreads == nullptr)
    return;

  if (MinBlocks) {
    MinBlocks = makeLaunchBoundsArgExpr(*this, MinBlocks, TmpAttr, 1);
    if (MinBlocks == nullptr)
      return;
  }

  D->addAttr(::new (Context)
                 CUDALaunchBoundsAttr(Context, CI, MaxThreads, MinBlocks));
}

static void handleLaunchBoundsAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (!AL.checkAtLeastNumArgs(S, 1) || !AL.checkAtMostNumArgs(S, 2))
    return;

  S.AddLaunchBoundsAttr(D, AL, AL.getArgAsExpr(0),
                        AL.getNumArgs() > 1 ? AL.getArgAsExpr(1) : nullptr);
}

static void handleArgumentWithTypeTagAttr(Sema &S, Decl *D,
                                          const ParsedAttr &AL) {
  if (!AL.isArgIdent(0)) {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_n_type)
        << AL << /* arg num = */ 1 << AANT_ArgumentIdentifier;
    return;
  }

  ParamIdx ArgumentIdx;
  if (!checkFunctionOrMethodParameterIndex(S, D, AL, 2, AL.getArgAsExpr(1),
                                           ArgumentIdx))
    return;

  ParamIdx TypeTagIdx;
  if (!checkFunctionOrMethodParameterIndex(S, D, AL, 3, AL.getArgAsExpr(2),
                                           TypeTagIdx))
    return;

  bool IsPointer = AL.getAttrName()->getName() == "pointer_with_type_tag";
  if (IsPointer) {
    // Ensure that buffer has a pointer type.
    unsigned ArgumentIdxAST = ArgumentIdx.getASTIndex();
    if (ArgumentIdxAST >= getFunctionOrMethodNumParams(D) ||
        !getFunctionOrMethodParamType(D, ArgumentIdxAST)->isPointerType())
      S.Diag(AL.getLoc(), diag::err_attribute_pointers_only) << AL << 0;
  }

  D->addAttr(::new (S.Context) ArgumentWithTypeTagAttr(
      S.Context, AL, AL.getArgAsIdent(0)->Ident, ArgumentIdx, TypeTagIdx,
      IsPointer));
}

static void handleTypeTagForDatatypeAttr(Sema &S, Decl *D,
                                         const ParsedAttr &AL) {
  if (!AL.isArgIdent(0)) {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_n_type)
        << AL << 1 << AANT_ArgumentIdentifier;
    return;
  }

  if (!AL.checkExactlyNumArgs(S, 1))
    return;

  if (!isa<VarDecl>(D)) {
    S.Diag(AL.getLoc(), diag::err_attribute_wrong_decl_type)
        << AL << ExpectedVariable;
    return;
  }

  IdentifierInfo *PointerKind = AL.getArgAsIdent(0)->Ident;
  TypeSourceInfo *MatchingCTypeLoc = nullptr;
  S.GetTypeFromParser(AL.getMatchingCType(), &MatchingCTypeLoc);
  assert(MatchingCTypeLoc && "no type source info for attribute argument");

  D->addAttr(::new (S.Context) TypeTagForDatatypeAttr(
      S.Context, AL, PointerKind, MatchingCTypeLoc, AL.getLayoutCompatible(),
      AL.getMustBeNull()));
}

static void handleXRayLogArgsAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  ParamIdx ArgCount;

  if (!checkFunctionOrMethodParameterIndex(S, D, AL, 1, AL.getArgAsExpr(0),
                                           ArgCount,
                                           true /* CanIndexImplicitThis */))
    return;

  // ArgCount isn't a parameter index [0;n), it's a count [1;n]
  D->addAttr(::new (S.Context)
                 XRayLogArgsAttr(S.Context, AL, ArgCount.getSourceIndex()));
}

static void handlePatchableFunctionEntryAttr(Sema &S, Decl *D,
                                             const ParsedAttr &AL) {
  uint32_t Count = 0, Offset = 0;
  if (!checkUInt32Argument(S, AL, AL.getArgAsExpr(0), Count, 0, true))
    return;
  if (AL.getNumArgs() == 2) {
    Expr *Arg = AL.getArgAsExpr(1);
    if (!checkUInt32Argument(S, AL, Arg, Offset, 1, true))
      return;
    if (Count < Offset) {
      S.Diag(getAttrLoc(AL), diag::err_attribute_argument_out_of_range)
          << &AL << 0 << Count << Arg->getBeginLoc();
      return;
    }
  }
  D->addAttr(::new (S.Context)
                 PatchableFunctionEntryAttr(S.Context, AL, Count, Offset));
}

namespace {
struct IntrinToName {
  uint32_t Id;
  int32_t FullName;
  int32_t ShortName;
};
} // unnamed namespace

static bool ArmBuiltinAliasValid(unsigned BuiltinID, StringRef AliasName,
                                 ArrayRef<IntrinToName> Map,
                                 const char *IntrinNames) {
  if (AliasName.startswith("__arm_"))
    AliasName = AliasName.substr(6);
  const IntrinToName *It = std::lower_bound(
      Map.begin(), Map.end(), BuiltinID,
      [](const IntrinToName &L, unsigned Id) { return L.Id < Id; });
  if (It == Map.end() || It->Id != BuiltinID)
    return false;
  StringRef FullName(&IntrinNames[It->FullName]);
  if (AliasName == FullName)
    return true;
  if (It->ShortName == -1)
    return false;
  StringRef ShortName(&IntrinNames[It->ShortName]);
  return AliasName == ShortName;
}

static bool ArmMveAliasValid(unsigned BuiltinID, StringRef AliasName) {
#include "clang/Basic/arm_mve_builtin_aliases.inc"
  // The included file defines:
  // - ArrayRef<IntrinToName> Map
  // - const char IntrinNames[]
  return ArmBuiltinAliasValid(BuiltinID, AliasName, Map, IntrinNames);
}

static bool ArmCdeAliasValid(unsigned BuiltinID, StringRef AliasName) {
#include "clang/Basic/arm_cde_builtin_aliases.inc"
  return ArmBuiltinAliasValid(BuiltinID, AliasName, Map, IntrinNames);
}

static bool ArmSveAliasValid(ASTContext &Context, unsigned BuiltinID,
                             StringRef AliasName) {
  if (Context.BuiltinInfo.isAuxBuiltinID(BuiltinID))
    BuiltinID = Context.BuiltinInfo.getAuxBuiltinID(BuiltinID);
  return BuiltinID >= AArch64::FirstSVEBuiltin &&
         BuiltinID <= AArch64::LastSVEBuiltin;
}

static void handleArmBuiltinAliasAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (!AL.isArgIdent(0)) {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_n_type)
        << AL << 1 << AANT_ArgumentIdentifier;
    return;
  }

  IdentifierInfo *Ident = AL.getArgAsIdent(0)->Ident;
  unsigned BuiltinID = Ident->getBuiltinID();
  StringRef AliasName = cast<FunctionDecl>(D)->getIdentifier()->getName();

  bool IsAArch64 = S.Context.getTargetInfo().getTriple().isAArch64();
  if ((IsAArch64 && !ArmSveAliasValid(S.Context, BuiltinID, AliasName)) ||
      (!IsAArch64 && !ArmMveAliasValid(BuiltinID, AliasName) &&
       !ArmCdeAliasValid(BuiltinID, AliasName))) {
    S.Diag(AL.getLoc(), diag::err_attribute_arm_builtin_alias);
    return;
  }

  D->addAttr(::new (S.Context) ArmBuiltinAliasAttr(S.Context, AL, Ident));
}

static bool RISCVAliasValid(unsigned BuiltinID, StringRef AliasName) {
  return BuiltinID >= Builtin::FirstTSBuiltin &&
         BuiltinID < RISCV::LastTSBuiltin;
}

static void handleBuiltinAliasAttr(Sema &S, Decl *D,
                                        const ParsedAttr &AL) {
  if (!AL.isArgIdent(0)) {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_n_type)
        << AL << 1 << AANT_ArgumentIdentifier;
    return;
  }

  IdentifierInfo *Ident = AL.getArgAsIdent(0)->Ident;
  unsigned BuiltinID = Ident->getBuiltinID();
  StringRef AliasName = cast<FunctionDecl>(D)->getIdentifier()->getName();

  bool IsAArch64 = S.Context.getTargetInfo().getTriple().isAArch64();
  bool IsARM = S.Context.getTargetInfo().getTriple().isARM();
  bool IsRISCV = S.Context.getTargetInfo().getTriple().isRISCV();
  if ((IsAArch64 && !ArmSveAliasValid(S.Context, BuiltinID, AliasName)) ||
      (IsARM && !ArmMveAliasValid(BuiltinID, AliasName) &&
       !ArmCdeAliasValid(BuiltinID, AliasName)) ||
      (IsRISCV && !RISCVAliasValid(BuiltinID, AliasName)) ||
      (!IsAArch64 && !IsARM && !IsRISCV)) {
    S.Diag(AL.getLoc(), diag::err_attribute_builtin_alias) << AL;
    return;
  }

  D->addAttr(::new (S.Context) BuiltinAliasAttr(S.Context, AL, Ident));
}

//===----------------------------------------------------------------------===//
// Checker-specific attribute handlers.
//===----------------------------------------------------------------------===//
static bool isValidSubjectOfNSReturnsRetainedAttribute(QualType QT) {
  return QT->isDependentType() || QT->isObjCRetainableType();
}

static bool isValidSubjectOfNSAttribute(QualType QT) {
  return QT->isDependentType() || QT->isObjCObjectPointerType() ||
         QT->isObjCNSObjectType();
}

static bool isValidSubjectOfCFAttribute(QualType QT) {
  return QT->isDependentType() || QT->isPointerType() ||
         isValidSubjectOfNSAttribute(QT);
}

static bool isValidSubjectOfOSAttribute(QualType QT) {
  if (QT->isDependentType())
    return true;
  QualType PT = QT->getPointeeType();
  return !PT.isNull() && PT->getAsCXXRecordDecl() != nullptr;
}

void Sema::AddXConsumedAttr(Decl *D, const AttributeCommonInfo &CI,
                            RetainOwnershipKind K,
                            bool IsTemplateInstantiation) {
  ValueDecl *VD = cast<ValueDecl>(D);
  switch (K) {
  case RetainOwnershipKind::OS:
    handleSimpleAttributeOrDiagnose<OSConsumedAttr>(
        *this, VD, CI, isValidSubjectOfOSAttribute(VD->getType()),
        diag::warn_ns_attribute_wrong_parameter_type,
        /*ExtraArgs=*/CI.getRange(), "os_consumed", /*pointers*/ 1);
    return;
  case RetainOwnershipKind::NS:
    handleSimpleAttributeOrDiagnose<NSConsumedAttr>(
        *this, VD, CI, isValidSubjectOfNSAttribute(VD->getType()),

        // These attributes are normally just advisory, but in ARC, ns_consumed
        // is significant.  Allow non-dependent code to contain inappropriate
        // attributes even in ARC, but require template instantiations to be
        // set up correctly.
        ((IsTemplateInstantiation && getLangOpts().ObjCAutoRefCount)
             ? diag::err_ns_attribute_wrong_parameter_type
             : diag::warn_ns_attribute_wrong_parameter_type),
        /*ExtraArgs=*/CI.getRange(), "ns_consumed", /*objc pointers*/ 0);
    return;
  case RetainOwnershipKind::CF:
    handleSimpleAttributeOrDiagnose<CFConsumedAttr>(
        *this, VD, CI, isValidSubjectOfCFAttribute(VD->getType()),
        diag::warn_ns_attribute_wrong_parameter_type,
        /*ExtraArgs=*/CI.getRange(), "cf_consumed", /*pointers*/ 1);
    return;
  }
}

static Sema::RetainOwnershipKind
parsedAttrToRetainOwnershipKind(const ParsedAttr &AL) {
  switch (AL.getKind()) {
  case ParsedAttr::AT_CFConsumed:
  case ParsedAttr::AT_CFReturnsRetained:
  case ParsedAttr::AT_CFReturnsNotRetained:
    return Sema::RetainOwnershipKind::CF;
  case ParsedAttr::AT_OSConsumesThis:
  case ParsedAttr::AT_OSConsumed:
  case ParsedAttr::AT_OSReturnsRetained:
  case ParsedAttr::AT_OSReturnsNotRetained:
  case ParsedAttr::AT_OSReturnsRetainedOnZero:
  case ParsedAttr::AT_OSReturnsRetainedOnNonZero:
    return Sema::RetainOwnershipKind::OS;
  case ParsedAttr::AT_NSConsumesSelf:
  case ParsedAttr::AT_NSConsumed:
  case ParsedAttr::AT_NSReturnsRetained:
  case ParsedAttr::AT_NSReturnsNotRetained:
  case ParsedAttr::AT_NSReturnsAutoreleased:
    return Sema::RetainOwnershipKind::NS;
  default:
    llvm_unreachable("Wrong argument supplied");
  }
}

bool Sema::checkNSReturnsRetainedReturnType(SourceLocation Loc, QualType QT) {
  if (isValidSubjectOfNSReturnsRetainedAttribute(QT))
    return false;

  Diag(Loc, diag::warn_ns_attribute_wrong_return_type)
      << "'ns_returns_retained'" << 0 << 0;
  return true;
}

/// \return whether the parameter is a pointer to OSObject pointer.
static bool isValidOSObjectOutParameter(const Decl *D) {
  const auto *PVD = dyn_cast<ParmVarDecl>(D);
  if (!PVD)
    return false;
  QualType QT = PVD->getType();
  QualType PT = QT->getPointeeType();
  return !PT.isNull() && isValidSubjectOfOSAttribute(PT);
}

static void handleXReturnsXRetainedAttr(Sema &S, Decl *D,
                                        const ParsedAttr &AL) {
  QualType ReturnType;
  Sema::RetainOwnershipKind K = parsedAttrToRetainOwnershipKind(AL);

  if (const auto *MD = dyn_cast<ObjCMethodDecl>(D)) {
    ReturnType = MD->getReturnType();
  } else if (S.getLangOpts().ObjCAutoRefCount && hasDeclarator(D) &&
             (AL.getKind() == ParsedAttr::AT_NSReturnsRetained)) {
    return; // ignore: was handled as a type attribute
  } else if (const auto *PD = dyn_cast<ObjCPropertyDecl>(D)) {
    ReturnType = PD->getType();
  } else if (const auto *FD = dyn_cast<FunctionDecl>(D)) {
    ReturnType = FD->getReturnType();
  } else if (const auto *Param = dyn_cast<ParmVarDecl>(D)) {
    // Attributes on parameters are used for out-parameters,
    // passed as pointers-to-pointers.
    unsigned DiagID = K == Sema::RetainOwnershipKind::CF
            ? /*pointer-to-CF-pointer*/2
            : /*pointer-to-OSObject-pointer*/3;
    ReturnType = Param->getType()->getPointeeType();
    if (ReturnType.isNull()) {
      S.Diag(D->getBeginLoc(), diag::warn_ns_attribute_wrong_parameter_type)
          << AL << DiagID << AL.getRange();
      return;
    }
  } else if (AL.isUsedAsTypeAttr()) {
    return;
  } else {
    AttributeDeclKind ExpectedDeclKind;
    switch (AL.getKind()) {
    default: llvm_unreachable("invalid ownership attribute");
    case ParsedAttr::AT_NSReturnsRetained:
    case ParsedAttr::AT_NSReturnsAutoreleased:
    case ParsedAttr::AT_NSReturnsNotRetained:
      ExpectedDeclKind = ExpectedFunctionOrMethod;
      break;

    case ParsedAttr::AT_OSReturnsRetained:
    case ParsedAttr::AT_OSReturnsNotRetained:
    case ParsedAttr::AT_CFReturnsRetained:
    case ParsedAttr::AT_CFReturnsNotRetained:
      ExpectedDeclKind = ExpectedFunctionMethodOrParameter;
      break;
    }
    S.Diag(D->getBeginLoc(), diag::warn_attribute_wrong_decl_type)
        << AL.getRange() << AL << ExpectedDeclKind;
    return;
  }

  bool TypeOK;
  bool Cf;
  unsigned ParmDiagID = 2; // Pointer-to-CF-pointer
  switch (AL.getKind()) {
  default: llvm_unreachable("invalid ownership attribute");
  case ParsedAttr::AT_NSReturnsRetained:
    TypeOK = isValidSubjectOfNSReturnsRetainedAttribute(ReturnType);
    Cf = false;
    break;

  case ParsedAttr::AT_NSReturnsAutoreleased:
  case ParsedAttr::AT_NSReturnsNotRetained:
    TypeOK = isValidSubjectOfNSAttribute(ReturnType);
    Cf = false;
    break;

  case ParsedAttr::AT_CFReturnsRetained:
  case ParsedAttr::AT_CFReturnsNotRetained:
    TypeOK = isValidSubjectOfCFAttribute(ReturnType);
    Cf = true;
    break;

  case ParsedAttr::AT_OSReturnsRetained:
  case ParsedAttr::AT_OSReturnsNotRetained:
    TypeOK = isValidSubjectOfOSAttribute(ReturnType);
    Cf = true;
    ParmDiagID = 3; // Pointer-to-OSObject-pointer
    break;
  }

  if (!TypeOK) {
    if (AL.isUsedAsTypeAttr())
      return;

    if (isa<ParmVarDecl>(D)) {
      S.Diag(D->getBeginLoc(), diag::warn_ns_attribute_wrong_parameter_type)
          << AL << ParmDiagID << AL.getRange();
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
      S.Diag(D->getBeginLoc(), diag::warn_ns_attribute_wrong_return_type)
          << AL << SubjectKind << Cf << AL.getRange();
    }
    return;
  }

  switch (AL.getKind()) {
    default:
      llvm_unreachable("invalid ownership attribute");
    case ParsedAttr::AT_NSReturnsAutoreleased:
      handleSimpleAttribute<NSReturnsAutoreleasedAttr>(S, D, AL);
      return;
    case ParsedAttr::AT_CFReturnsNotRetained:
      handleSimpleAttribute<CFReturnsNotRetainedAttr>(S, D, AL);
      return;
    case ParsedAttr::AT_NSReturnsNotRetained:
      handleSimpleAttribute<NSReturnsNotRetainedAttr>(S, D, AL);
      return;
    case ParsedAttr::AT_CFReturnsRetained:
      handleSimpleAttribute<CFReturnsRetainedAttr>(S, D, AL);
      return;
    case ParsedAttr::AT_NSReturnsRetained:
      handleSimpleAttribute<NSReturnsRetainedAttr>(S, D, AL);
      return;
    case ParsedAttr::AT_OSReturnsRetained:
      handleSimpleAttribute<OSReturnsRetainedAttr>(S, D, AL);
      return;
    case ParsedAttr::AT_OSReturnsNotRetained:
      handleSimpleAttribute<OSReturnsNotRetainedAttr>(S, D, AL);
      return;
  };
}

static void handleObjCReturnsInnerPointerAttr(Sema &S, Decl *D,
                                              const ParsedAttr &Attrs) {
  const int EP_ObjCMethod = 1;
  const int EP_ObjCProperty = 2;

  SourceLocation loc = Attrs.getLoc();
  QualType resultType;
  if (isa<ObjCMethodDecl>(D))
    resultType = cast<ObjCMethodDecl>(D)->getReturnType();
  else
    resultType = cast<ObjCPropertyDecl>(D)->getType();

  if (!resultType->isReferenceType() &&
      (!resultType->isPointerType() || resultType->isObjCRetainableType())) {
    S.Diag(D->getBeginLoc(), diag::warn_ns_attribute_wrong_return_type)
        << SourceRange(loc) << Attrs
        << (isa<ObjCMethodDecl>(D) ? EP_ObjCMethod : EP_ObjCProperty)
        << /*non-retainable pointer*/ 2;

    // Drop the attribute.
    return;
  }

  D->addAttr(::new (S.Context) ObjCReturnsInnerPointerAttr(S.Context, Attrs));
}

static void handleObjCRequiresSuperAttr(Sema &S, Decl *D,
                                        const ParsedAttr &Attrs) {
  const auto *Method = cast<ObjCMethodDecl>(D);

  const DeclContext *DC = Method->getDeclContext();
  if (const auto *PDecl = dyn_cast_or_null<ObjCProtocolDecl>(DC)) {
    S.Diag(D->getBeginLoc(), diag::warn_objc_requires_super_protocol) << Attrs
                                                                      << 0;
    S.Diag(PDecl->getLocation(), diag::note_protocol_decl);
    return;
  }
  if (Method->getMethodFamily() == OMF_dealloc) {
    S.Diag(D->getBeginLoc(), diag::warn_objc_requires_super_protocol) << Attrs
                                                                      << 1;
    return;
  }

  D->addAttr(::new (S.Context) ObjCRequiresSuperAttr(S.Context, Attrs));
}

static void handleNSErrorDomain(Sema &S, Decl *D, const ParsedAttr &AL) {
  auto *E = AL.getArgAsExpr(0);
  auto Loc = E ? E->getBeginLoc() : AL.getLoc();

  auto *DRE = dyn_cast<DeclRefExpr>(AL.getArgAsExpr(0));
  if (!DRE) {
    S.Diag(Loc, diag::err_nserrordomain_invalid_decl) << 0;
    return;
  }

  auto *VD = dyn_cast<VarDecl>(DRE->getDecl());
  if (!VD) {
    S.Diag(Loc, diag::err_nserrordomain_invalid_decl) << 1 << DRE->getDecl();
    return;
  }

  if (!isNSStringType(VD->getType(), S.Context) &&
      !isCFStringType(VD->getType(), S.Context)) {
    S.Diag(Loc, diag::err_nserrordomain_wrong_type) << VD;
    return;
  }

  D->addAttr(::new (S.Context) NSErrorDomainAttr(S.Context, AL, VD));
}

static void handleObjCBridgeAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  IdentifierLoc *Parm = AL.isArgIdent(0) ? AL.getArgAsIdent(0) : nullptr;

  if (!Parm) {
    S.Diag(D->getBeginLoc(), diag::err_objc_attr_not_id) << AL << 0;
    return;
  }

  // Typedefs only allow objc_bridge(id) and have some additional checking.
  if (const auto *TD = dyn_cast<TypedefNameDecl>(D)) {
    if (!Parm->Ident->isStr("id")) {
      S.Diag(AL.getLoc(), diag::err_objc_attr_typedef_not_id) << AL;
      return;
    }

    // Only allow 'cv void *'.
    QualType T = TD->getUnderlyingType();
    if (!T->isVoidPointerType()) {
      S.Diag(AL.getLoc(), diag::err_objc_attr_typedef_not_void_pointer);
      return;
    }
  }

  D->addAttr(::new (S.Context) ObjCBridgeAttr(S.Context, AL, Parm->Ident));
}

static void handleObjCBridgeMutableAttr(Sema &S, Decl *D,
                                        const ParsedAttr &AL) {
  IdentifierLoc *Parm = AL.isArgIdent(0) ? AL.getArgAsIdent(0) : nullptr;

  if (!Parm) {
    S.Diag(D->getBeginLoc(), diag::err_objc_attr_not_id) << AL << 0;
    return;
  }

  D->addAttr(::new (S.Context)
                 ObjCBridgeMutableAttr(S.Context, AL, Parm->Ident));
}

static void handleObjCBridgeRelatedAttr(Sema &S, Decl *D,
                                        const ParsedAttr &AL) {
  IdentifierInfo *RelatedClass =
      AL.isArgIdent(0) ? AL.getArgAsIdent(0)->Ident : nullptr;
  if (!RelatedClass) {
    S.Diag(D->getBeginLoc(), diag::err_objc_attr_not_id) << AL << 0;
    return;
  }
  IdentifierInfo *ClassMethod =
    AL.getArgAsIdent(1) ? AL.getArgAsIdent(1)->Ident : nullptr;
  IdentifierInfo *InstanceMethod =
    AL.getArgAsIdent(2) ? AL.getArgAsIdent(2)->Ident : nullptr;
  D->addAttr(::new (S.Context) ObjCBridgeRelatedAttr(
      S.Context, AL, RelatedClass, ClassMethod, InstanceMethod));
}

static void handleObjCDesignatedInitializer(Sema &S, Decl *D,
                                            const ParsedAttr &AL) {
  DeclContext *Ctx = D->getDeclContext();

  // This attribute can only be applied to methods in interfaces or class
  // extensions.
  if (!isa<ObjCInterfaceDecl>(Ctx) &&
      !(isa<ObjCCategoryDecl>(Ctx) &&
        cast<ObjCCategoryDecl>(Ctx)->IsClassExtension())) {
    S.Diag(D->getLocation(), diag::err_designated_init_attr_non_init);
    return;
  }

  ObjCInterfaceDecl *IFace;
  if (auto *CatDecl = dyn_cast<ObjCCategoryDecl>(Ctx))
    IFace = CatDecl->getClassInterface();
  else
    IFace = cast<ObjCInterfaceDecl>(Ctx);

  if (!IFace)
    return;

  IFace->setHasDesignatedInitializers();
  D->addAttr(::new (S.Context) ObjCDesignatedInitializerAttr(S.Context, AL));
}

static void handleObjCRuntimeName(Sema &S, Decl *D, const ParsedAttr &AL) {
  StringRef MetaDataName;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, MetaDataName))
    return;
  D->addAttr(::new (S.Context)
                 ObjCRuntimeNameAttr(S.Context, AL, MetaDataName));
}

// When a user wants to use objc_boxable with a union or struct
// but they don't have access to the declaration (legacy/third-party code)
// then they can 'enable' this feature with a typedef:
// typedef struct __attribute((objc_boxable)) legacy_struct legacy_struct;
static void handleObjCBoxable(Sema &S, Decl *D, const ParsedAttr &AL) {
  bool notify = false;

  auto *RD = dyn_cast<RecordDecl>(D);
  if (RD && RD->getDefinition()) {
    RD = RD->getDefinition();
    notify = true;
  }

  if (RD) {
    ObjCBoxableAttr *BoxableAttr =
        ::new (S.Context) ObjCBoxableAttr(S.Context, AL);
    RD->addAttr(BoxableAttr);
    if (notify) {
      // we need to notify ASTReader/ASTWriter about
      // modification of existing declaration
      if (ASTMutationListener *L = S.getASTMutationListener())
        L->AddedAttributeToRecord(BoxableAttr, RD);
    }
  }
}

static void handleObjCOwnershipAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (hasDeclarator(D)) return;

  S.Diag(D->getBeginLoc(), diag::err_attribute_wrong_decl_type)
      << AL.getRange() << AL << ExpectedVariable;
}

static void handleObjCPreciseLifetimeAttr(Sema &S, Decl *D,
                                          const ParsedAttr &AL) {
  const auto *VD = cast<ValueDecl>(D);
  QualType QT = VD->getType();

  if (!QT->isDependentType() &&
      !QT->isObjCLifetimeType()) {
    S.Diag(AL.getLoc(), diag::err_objc_precise_lifetime_bad_type)
      << QT;
    return;
  }

  Qualifiers::ObjCLifetime Lifetime = QT.getObjCLifetime();

  // If we have no lifetime yet, check the lifetime we're presumably
  // going to infer.
  if (Lifetime == Qualifiers::OCL_None && !QT->isDependentType())
    Lifetime = QT->getObjCARCImplicitLifetime();

  switch (Lifetime) {
  case Qualifiers::OCL_None:
    assert(QT->isDependentType() &&
           "didn't infer lifetime for non-dependent type?");
    break;

  case Qualifiers::OCL_Weak:   // meaningful
  case Qualifiers::OCL_Strong: // meaningful
    break;

  case Qualifiers::OCL_ExplicitNone:
  case Qualifiers::OCL_Autoreleasing:
    S.Diag(AL.getLoc(), diag::warn_objc_precise_lifetime_meaningless)
        << (Lifetime == Qualifiers::OCL_Autoreleasing);
    break;
  }

  D->addAttr(::new (S.Context) ObjCPreciseLifetimeAttr(S.Context, AL));
}

static void handleSwiftAttrAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  // Make sure that there is a string literal as the annotation's single
  // argument.
  StringRef Str;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, Str))
    return;

  D->addAttr(::new (S.Context) SwiftAttrAttr(S.Context, AL, Str));
}

static void handleSwiftBridge(Sema &S, Decl *D, const ParsedAttr &AL) {
  // Make sure that there is a string literal as the annotation's single
  // argument.
  StringRef BT;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, BT))
    return;

  // Warn about duplicate attributes if they have different arguments, but drop
  // any duplicate attributes regardless.
  if (const auto *Other = D->getAttr<SwiftBridgeAttr>()) {
    if (Other->getSwiftType() != BT)
      S.Diag(AL.getLoc(), diag::warn_duplicate_attribute) << AL;
    return;
  }

  D->addAttr(::new (S.Context) SwiftBridgeAttr(S.Context, AL, BT));
}

static bool isErrorParameter(Sema &S, QualType QT) {
  const auto *PT = QT->getAs<PointerType>();
  if (!PT)
    return false;

  QualType Pointee = PT->getPointeeType();

  // Check for NSError**.
  if (const auto *OPT = Pointee->getAs<ObjCObjectPointerType>())
    if (const auto *ID = OPT->getInterfaceDecl())
      if (ID->getIdentifier() == S.getNSErrorIdent())
        return true;

  // Check for CFError**.
  if (const auto *PT = Pointee->getAs<PointerType>())
    if (const auto *RT = PT->getPointeeType()->getAs<RecordType>())
      if (S.isCFError(RT->getDecl()))
        return true;

  return false;
}

static void handleSwiftError(Sema &S, Decl *D, const ParsedAttr &AL) {
  auto hasErrorParameter = [](Sema &S, Decl *D, const ParsedAttr &AL) -> bool {
    for (unsigned I = 0, E = getFunctionOrMethodNumParams(D); I != E; ++I) {
      if (isErrorParameter(S, getFunctionOrMethodParamType(D, I)))
        return true;
    }

    S.Diag(AL.getLoc(), diag::err_attr_swift_error_no_error_parameter)
        << AL << isa<ObjCMethodDecl>(D);
    return false;
  };

  auto hasPointerResult = [](Sema &S, Decl *D, const ParsedAttr &AL) -> bool {
    // - C, ObjC, and block pointers are definitely okay.
    // - References are definitely not okay.
    // - nullptr_t is weird, but acceptable.
    QualType RT = getFunctionOrMethodResultType(D);
    if (RT->hasPointerRepresentation() && !RT->isReferenceType())
      return true;

    S.Diag(AL.getLoc(), diag::err_attr_swift_error_return_type)
        << AL << AL.getArgAsIdent(0)->Ident->getName() << isa<ObjCMethodDecl>(D)
        << /*pointer*/ 1;
    return false;
  };

  auto hasIntegerResult = [](Sema &S, Decl *D, const ParsedAttr &AL) -> bool {
    QualType RT = getFunctionOrMethodResultType(D);
    if (RT->isIntegralType(S.Context))
      return true;

    S.Diag(AL.getLoc(), diag::err_attr_swift_error_return_type)
        << AL << AL.getArgAsIdent(0)->Ident->getName() << isa<ObjCMethodDecl>(D)
        << /*integral*/ 0;
    return false;
  };

  if (D->isInvalidDecl())
    return;

  IdentifierLoc *Loc = AL.getArgAsIdent(0);
  SwiftErrorAttr::ConventionKind Convention;
  if (!SwiftErrorAttr::ConvertStrToConventionKind(Loc->Ident->getName(),
                                                  Convention)) {
    S.Diag(AL.getLoc(), diag::warn_attribute_type_not_supported)
        << AL << Loc->Ident;
    return;
  }

  switch (Convention) {
  case SwiftErrorAttr::None:
    // No additional validation required.
    break;

  case SwiftErrorAttr::NonNullError:
    if (!hasErrorParameter(S, D, AL))
      return;
    break;

  case SwiftErrorAttr::NullResult:
    if (!hasErrorParameter(S, D, AL) || !hasPointerResult(S, D, AL))
      return;
    break;

  case SwiftErrorAttr::NonZeroResult:
  case SwiftErrorAttr::ZeroResult:
    if (!hasErrorParameter(S, D, AL) || !hasIntegerResult(S, D, AL))
      return;
    break;
  }

  D->addAttr(::new (S.Context) SwiftErrorAttr(S.Context, AL, Convention));
}

static void checkSwiftAsyncErrorBlock(Sema &S, Decl *D,
                                      const SwiftAsyncErrorAttr *ErrorAttr,
                                      const SwiftAsyncAttr *AsyncAttr) {
  if (AsyncAttr->getKind() == SwiftAsyncAttr::None) {
    if (ErrorAttr->getConvention() != SwiftAsyncErrorAttr::None) {
      S.Diag(AsyncAttr->getLocation(),
             diag::err_swift_async_error_without_swift_async)
          << AsyncAttr << isa<ObjCMethodDecl>(D);
    }
    return;
  }

  const ParmVarDecl *HandlerParam = getFunctionOrMethodParam(
      D, AsyncAttr->getCompletionHandlerIndex().getASTIndex());
  // handleSwiftAsyncAttr already verified the type is correct, so no need to
  // double-check it here.
  const auto *FuncTy = HandlerParam->getType()
                           ->castAs<BlockPointerType>()
                           ->getPointeeType()
                           ->getAs<FunctionProtoType>();
  ArrayRef<QualType> BlockParams;
  if (FuncTy)
    BlockParams = FuncTy->getParamTypes();

  switch (ErrorAttr->getConvention()) {
  case SwiftAsyncErrorAttr::ZeroArgument:
  case SwiftAsyncErrorAttr::NonZeroArgument: {
    uint32_t ParamIdx = ErrorAttr->getHandlerParamIdx();
    if (ParamIdx == 0 || ParamIdx > BlockParams.size()) {
      S.Diag(ErrorAttr->getLocation(),
             diag::err_attribute_argument_out_of_bounds) << ErrorAttr << 2;
      return;
    }
    QualType ErrorParam = BlockParams[ParamIdx - 1];
    if (!ErrorParam->isIntegralType(S.Context)) {
      StringRef ConvStr =
          ErrorAttr->getConvention() == SwiftAsyncErrorAttr::ZeroArgument
              ? "zero_argument"
              : "nonzero_argument";
      S.Diag(ErrorAttr->getLocation(), diag::err_swift_async_error_non_integral)
          << ErrorAttr << ConvStr << ParamIdx << ErrorParam;
      return;
    }
    break;
  }
  case SwiftAsyncErrorAttr::NonNullError: {
    bool AnyErrorParams = false;
    for (QualType Param : BlockParams) {
      // Check for NSError *.
      if (const auto *ObjCPtrTy = Param->getAs<ObjCObjectPointerType>()) {
        if (const auto *ID = ObjCPtrTy->getInterfaceDecl()) {
          if (ID->getIdentifier() == S.getNSErrorIdent()) {
            AnyErrorParams = true;
            break;
          }
        }
      }
      // Check for CFError *.
      if (const auto *PtrTy = Param->getAs<PointerType>()) {
        if (const auto *RT = PtrTy->getPointeeType()->getAs<RecordType>()) {
          if (S.isCFError(RT->getDecl())) {
            AnyErrorParams = true;
            break;
          }
        }
      }
    }

    if (!AnyErrorParams) {
      S.Diag(ErrorAttr->getLocation(),
             diag::err_swift_async_error_no_error_parameter)
          << ErrorAttr << isa<ObjCMethodDecl>(D);
      return;
    }
    break;
  }
  case SwiftAsyncErrorAttr::None:
    break;
  }
}

static void handleSwiftAsyncError(Sema &S, Decl *D, const ParsedAttr &AL) {
  IdentifierLoc *IDLoc = AL.getArgAsIdent(0);
  SwiftAsyncErrorAttr::ConventionKind ConvKind;
  if (!SwiftAsyncErrorAttr::ConvertStrToConventionKind(IDLoc->Ident->getName(),
                                                       ConvKind)) {
    S.Diag(AL.getLoc(), diag::warn_attribute_type_not_supported)
        << AL << IDLoc->Ident;
    return;
  }

  uint32_t ParamIdx = 0;
  switch (ConvKind) {
  case SwiftAsyncErrorAttr::ZeroArgument:
  case SwiftAsyncErrorAttr::NonZeroArgument: {
    if (!AL.checkExactlyNumArgs(S, 2))
      return;

    Expr *IdxExpr = AL.getArgAsExpr(1);
    if (!checkUInt32Argument(S, AL, IdxExpr, ParamIdx))
      return;
    break;
  }
  case SwiftAsyncErrorAttr::NonNullError:
  case SwiftAsyncErrorAttr::None: {
    if (!AL.checkExactlyNumArgs(S, 1))
      return;
    break;
  }
  }

  auto *ErrorAttr =
      ::new (S.Context) SwiftAsyncErrorAttr(S.Context, AL, ConvKind, ParamIdx);
  D->addAttr(ErrorAttr);

  if (auto *AsyncAttr = D->getAttr<SwiftAsyncAttr>())
    checkSwiftAsyncErrorBlock(S, D, ErrorAttr, AsyncAttr);
}

// For a function, this will validate a compound Swift name, e.g.
// <code>init(foo:bar:baz:)</code> or <code>controllerForName(_:)</code>, and
// the function will output the number of parameter names, and whether this is a
// single-arg initializer.
//
// For a type, enum constant, property, or variable declaration, this will
// validate either a simple identifier, or a qualified
// <code>context.identifier</code> name.
static bool
validateSwiftFunctionName(Sema &S, const ParsedAttr &AL, SourceLocation Loc,
                          StringRef Name, unsigned &SwiftParamCount,
                          bool &IsSingleParamInit) {
  SwiftParamCount = 0;
  IsSingleParamInit = false;

  // Check whether this will be mapped to a getter or setter of a property.
  bool IsGetter = false, IsSetter = false;
  if (Name.startswith("getter:")) {
    IsGetter = true;
    Name = Name.substr(7);
  } else if (Name.startswith("setter:")) {
    IsSetter = true;
    Name = Name.substr(7);
  }

  if (Name.back() != ')') {
    S.Diag(Loc, diag::warn_attr_swift_name_function) << AL;
    return false;
  }

  bool IsMember = false;
  StringRef ContextName, BaseName, Parameters;

  std::tie(BaseName, Parameters) = Name.split('(');

  // Split at the first '.', if it exists, which separates the context name
  // from the base name.
  std::tie(ContextName, BaseName) = BaseName.split('.');
  if (BaseName.empty()) {
    BaseName = ContextName;
    ContextName = StringRef();
  } else if (ContextName.empty() || !isValidAsciiIdentifier(ContextName)) {
    S.Diag(Loc, diag::warn_attr_swift_name_invalid_identifier)
        << AL << /*context*/ 1;
    return false;
  } else {
    IsMember = true;
  }

  if (!isValidAsciiIdentifier(BaseName) || BaseName == "_") {
    S.Diag(Loc, diag::warn_attr_swift_name_invalid_identifier)
        << AL << /*basename*/ 0;
    return false;
  }

  bool IsSubscript = BaseName == "subscript";
  // A subscript accessor must be a getter or setter.
  if (IsSubscript && !IsGetter && !IsSetter) {
    S.Diag(Loc, diag::warn_attr_swift_name_subscript_invalid_parameter)
        << AL << /* getter or setter */ 0;
    return false;
  }

  if (Parameters.empty()) {
    S.Diag(Loc, diag::warn_attr_swift_name_missing_parameters) << AL;
    return false;
  }

  assert(Parameters.back() == ')' && "expected ')'");
  Parameters = Parameters.drop_back(); // ')'

  if (Parameters.empty()) {
    // Setters and subscripts must have at least one parameter.
    if (IsSubscript) {
      S.Diag(Loc, diag::warn_attr_swift_name_subscript_invalid_parameter)
          << AL << /* have at least one parameter */1;
      return false;
    }

    if (IsSetter) {
      S.Diag(Loc, diag::warn_attr_swift_name_setter_parameters) << AL;
      return false;
    }

    return true;
  }

  if (Parameters.back() != ':') {
    S.Diag(Loc, diag::warn_attr_swift_name_function) << AL;
    return false;
  }

  StringRef CurrentParam;
  llvm::Optional<unsigned> SelfLocation;
  unsigned NewValueCount = 0;
  llvm::Optional<unsigned> NewValueLocation;
  do {
    std::tie(CurrentParam, Parameters) = Parameters.split(':');

    if (!isValidAsciiIdentifier(CurrentParam)) {
      S.Diag(Loc, diag::warn_attr_swift_name_invalid_identifier)
          << AL << /*parameter*/2;
      return false;
    }

    if (IsMember && CurrentParam == "self") {
      // "self" indicates the "self" argument for a member.

      // More than one "self"?
      if (SelfLocation) {
        S.Diag(Loc, diag::warn_attr_swift_name_multiple_selfs) << AL;
        return false;
      }

      // The "self" location is the current parameter.
      SelfLocation = SwiftParamCount;
    } else if (CurrentParam == "newValue") {
      // "newValue" indicates the "newValue" argument for a setter.

      // There should only be one 'newValue', but it's only significant for
      // subscript accessors, so don't error right away.
      ++NewValueCount;

      NewValueLocation = SwiftParamCount;
    }

    ++SwiftParamCount;
  } while (!Parameters.empty());

  // Only instance subscripts are currently supported.
  if (IsSubscript && !SelfLocation) {
    S.Diag(Loc, diag::warn_attr_swift_name_subscript_invalid_parameter)
        << AL << /*have a 'self:' parameter*/2;
    return false;
  }

  IsSingleParamInit =
        SwiftParamCount == 1 && BaseName == "init" && CurrentParam != "_";

  // Check the number of parameters for a getter/setter.
  if (IsGetter || IsSetter) {
    // Setters have one parameter for the new value.
    unsigned NumExpectedParams = IsGetter ? 0 : 1;
    unsigned ParamDiag =
        IsGetter ? diag::warn_attr_swift_name_getter_parameters
                 : diag::warn_attr_swift_name_setter_parameters;

    // Instance methods have one parameter for "self".
    if (SelfLocation)
      ++NumExpectedParams;

    // Subscripts may have additional parameters beyond the expected params for
    // the index.
    if (IsSubscript) {
      if (SwiftParamCount < NumExpectedParams) {
        S.Diag(Loc, ParamDiag) << AL;
        return false;
      }

      // A subscript setter must explicitly label its newValue parameter to
      // distinguish it from index parameters.
      if (IsSetter) {
        if (!NewValueLocation) {
          S.Diag(Loc, diag::warn_attr_swift_name_subscript_setter_no_newValue)
              << AL;
          return false;
        }
        if (NewValueCount > 1) {
          S.Diag(Loc, diag::warn_attr_swift_name_subscript_setter_multiple_newValues)
              << AL;
          return false;
        }
      } else {
        // Subscript getters should have no 'newValue:' parameter.
        if (NewValueLocation) {
          S.Diag(Loc, diag::warn_attr_swift_name_subscript_getter_newValue)
              << AL;
          return false;
        }
      }
    } else {
      // Property accessors must have exactly the number of expected params.
      if (SwiftParamCount != NumExpectedParams) {
        S.Diag(Loc, ParamDiag) << AL;
        return false;
      }
    }
  }

  return true;
}

bool Sema::DiagnoseSwiftName(Decl *D, StringRef Name, SourceLocation Loc,
                             const ParsedAttr &AL, bool IsAsync) {
  if (isa<ObjCMethodDecl>(D) || isa<FunctionDecl>(D)) {
    ArrayRef<ParmVarDecl*> Params;
    unsigned ParamCount;

    if (const auto *Method = dyn_cast<ObjCMethodDecl>(D)) {
      ParamCount = Method->getSelector().getNumArgs();
      Params = Method->parameters().slice(0, ParamCount);
    } else {
      const auto *F = cast<FunctionDecl>(D);

      ParamCount = F->getNumParams();
      Params = F->parameters();

      if (!F->hasWrittenPrototype()) {
        Diag(Loc, diag::warn_attribute_wrong_decl_type) << AL
            << ExpectedFunctionWithProtoType;
        return false;
      }
    }

    // The async name drops the last callback parameter.
    if (IsAsync) {
      if (ParamCount == 0) {
        Diag(Loc, diag::warn_attr_swift_name_decl_missing_params)
            << AL << isa<ObjCMethodDecl>(D);
        return false;
      }
      ParamCount -= 1;
    }

    unsigned SwiftParamCount;
    bool IsSingleParamInit;
    if (!validateSwiftFunctionName(*this, AL, Loc, Name,
                                   SwiftParamCount, IsSingleParamInit))
      return false;

    bool ParamCountValid;
    if (SwiftParamCount == ParamCount) {
      ParamCountValid = true;
    } else if (SwiftParamCount > ParamCount) {
      ParamCountValid = IsSingleParamInit && ParamCount == 0;
    } else {
      // We have fewer Swift parameters than Objective-C parameters, but that
      // might be because we've transformed some of them. Check for potential
      // "out" parameters and err on the side of not warning.
      unsigned MaybeOutParamCount =
          std::count_if(Params.begin(), Params.end(),
                        [](const ParmVarDecl *Param) -> bool {
        QualType ParamTy = Param->getType();
        if (ParamTy->isReferenceType() || ParamTy->isPointerType())
          return !ParamTy->getPointeeType().isConstQualified();
        return false;
      });

      ParamCountValid = SwiftParamCount + MaybeOutParamCount >= ParamCount;
    }

    if (!ParamCountValid) {
      Diag(Loc, diag::warn_attr_swift_name_num_params)
          << (SwiftParamCount > ParamCount) << AL << ParamCount
          << SwiftParamCount;
      return false;
    }
  } else if ((isa<EnumConstantDecl>(D) || isa<ObjCProtocolDecl>(D) ||
              isa<ObjCInterfaceDecl>(D) || isa<ObjCPropertyDecl>(D) ||
              isa<VarDecl>(D) || isa<TypedefNameDecl>(D) || isa<TagDecl>(D) ||
              isa<IndirectFieldDecl>(D) || isa<FieldDecl>(D)) &&
             !IsAsync) {
    StringRef ContextName, BaseName;

    std::tie(ContextName, BaseName) = Name.split('.');
    if (BaseName.empty()) {
      BaseName = ContextName;
      ContextName = StringRef();
    } else if (!isValidAsciiIdentifier(ContextName)) {
      Diag(Loc, diag::warn_attr_swift_name_invalid_identifier) << AL
          << /*context*/1;
      return false;
    }

    if (!isValidAsciiIdentifier(BaseName)) {
      Diag(Loc, diag::warn_attr_swift_name_invalid_identifier) << AL
          << /*basename*/0;
      return false;
    }
  } else {
    Diag(Loc, diag::warn_attr_swift_name_decl_kind) << AL;
    return false;
  }
  return true;
}

static void handleSwiftName(Sema &S, Decl *D, const ParsedAttr &AL) {
  StringRef Name;
  SourceLocation Loc;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, Name, &Loc))
    return;

  if (!S.DiagnoseSwiftName(D, Name, Loc, AL, /*IsAsync=*/false))
    return;

  D->addAttr(::new (S.Context) SwiftNameAttr(S.Context, AL, Name));
}

static void handleSwiftAsyncName(Sema &S, Decl *D, const ParsedAttr &AL) {
  StringRef Name;
  SourceLocation Loc;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, Name, &Loc))
    return;

  if (!S.DiagnoseSwiftName(D, Name, Loc, AL, /*IsAsync=*/true))
    return;

  D->addAttr(::new (S.Context) SwiftAsyncNameAttr(S.Context, AL, Name));
}

static void handleSwiftNewType(Sema &S, Decl *D, const ParsedAttr &AL) {
  // Make sure that there is an identifier as the annotation's single argument.
  if (!AL.checkExactlyNumArgs(S, 1))
    return;

  if (!AL.isArgIdent(0)) {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_type)
        << AL << AANT_ArgumentIdentifier;
    return;
  }

  SwiftNewTypeAttr::NewtypeKind Kind;
  IdentifierInfo *II = AL.getArgAsIdent(0)->Ident;
  if (!SwiftNewTypeAttr::ConvertStrToNewtypeKind(II->getName(), Kind)) {
    S.Diag(AL.getLoc(), diag::warn_attribute_type_not_supported) << AL << II;
    return;
  }

  if (!isa<TypedefNameDecl>(D)) {
    S.Diag(AL.getLoc(), diag::warn_attribute_wrong_decl_type_str)
        << AL << "typedefs";
    return;
  }

  D->addAttr(::new (S.Context) SwiftNewTypeAttr(S.Context, AL, Kind));
}

static void handleSwiftAsyncAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (!AL.isArgIdent(0)) {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_n_type)
        << AL << 1 << AANT_ArgumentIdentifier;
    return;
  }

  SwiftAsyncAttr::Kind Kind;
  IdentifierInfo *II = AL.getArgAsIdent(0)->Ident;
  if (!SwiftAsyncAttr::ConvertStrToKind(II->getName(), Kind)) {
    S.Diag(AL.getLoc(), diag::err_swift_async_no_access) << AL << II;
    return;
  }

  ParamIdx Idx;
  if (Kind == SwiftAsyncAttr::None) {
    // If this is 'none', then there shouldn't be any additional arguments.
    if (!AL.checkExactlyNumArgs(S, 1))
      return;
  } else {
    // Non-none swift_async requires a completion handler index argument.
    if (!AL.checkExactlyNumArgs(S, 2))
      return;

    Expr *HandlerIdx = AL.getArgAsExpr(1);
    if (!checkFunctionOrMethodParameterIndex(S, D, AL, 2, HandlerIdx, Idx))
      return;

    const ParmVarDecl *CompletionBlock =
        getFunctionOrMethodParam(D, Idx.getASTIndex());
    QualType CompletionBlockType = CompletionBlock->getType();
    if (!CompletionBlockType->isBlockPointerType()) {
      S.Diag(CompletionBlock->getLocation(),
             diag::err_swift_async_bad_block_type)
          << CompletionBlock->getType();
      return;
    }
    QualType BlockTy =
        CompletionBlockType->castAs<BlockPointerType>()->getPointeeType();
    if (!BlockTy->castAs<FunctionType>()->getReturnType()->isVoidType()) {
      S.Diag(CompletionBlock->getLocation(),
             diag::err_swift_async_bad_block_type)
          << CompletionBlock->getType();
      return;
    }
  }

  auto *AsyncAttr =
      ::new (S.Context) SwiftAsyncAttr(S.Context, AL, Kind, Idx);
  D->addAttr(AsyncAttr);

  if (auto *ErrorAttr = D->getAttr<SwiftAsyncErrorAttr>())
    checkSwiftAsyncErrorBlock(S, D, ErrorAttr, AsyncAttr);
}

//===----------------------------------------------------------------------===//
// Microsoft specific attribute handlers.
//===----------------------------------------------------------------------===//

UuidAttr *Sema::mergeUuidAttr(Decl *D, const AttributeCommonInfo &CI,
                              StringRef UuidAsWritten, MSGuidDecl *GuidDecl) {
  if (const auto *UA = D->getAttr<UuidAttr>()) {
    if (declaresSameEntity(UA->getGuidDecl(), GuidDecl))
      return nullptr;
    if (!UA->getGuid().empty()) {
      Diag(UA->getLocation(), diag::err_mismatched_uuid);
      Diag(CI.getLoc(), diag::note_previous_uuid);
      D->dropAttr<UuidAttr>();
    }
  }

  return ::new (Context) UuidAttr(Context, CI, UuidAsWritten, GuidDecl);
}

static void handleUuidAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (!S.LangOpts.CPlusPlus) {
    S.Diag(AL.getLoc(), diag::err_attribute_not_supported_in_lang)
        << AL << AttributeLangSupport::C;
    return;
  }

  StringRef OrigStrRef;
  SourceLocation LiteralLoc;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, OrigStrRef, &LiteralLoc))
    return;

  // GUID format is "XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX" or
  // "{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}", normalize to the former.
  StringRef StrRef = OrigStrRef;
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

  // Convert to our parsed format and canonicalize.
  MSGuidDecl::Parts Parsed;
  StrRef.substr(0, 8).getAsInteger(16, Parsed.Part1);
  StrRef.substr(9, 4).getAsInteger(16, Parsed.Part2);
  StrRef.substr(14, 4).getAsInteger(16, Parsed.Part3);
  for (unsigned i = 0; i != 8; ++i)
    StrRef.substr(19 + 2 * i + (i >= 2 ? 1 : 0), 2)
        .getAsInteger(16, Parsed.Part4And5[i]);
  MSGuidDecl *Guid = S.Context.getMSGuidDecl(Parsed);

  // FIXME: It'd be nice to also emit a fixit removing uuid(...) (and, if it's
  // the only thing in the [] list, the [] too), and add an insertion of
  // __declspec(uuid(...)).  But sadly, neither the SourceLocs of the commas
  // separating attributes nor of the [ and the ] are in the AST.
  // Cf "SourceLocations of attribute list delimiters - [[ ... , ... ]] etc"
  // on cfe-dev.
  if (AL.isMicrosoftAttribute()) // Check for [uuid(...)] spelling.
    S.Diag(AL.getLoc(), diag::warn_atl_uuid_deprecated);

  UuidAttr *UA = S.mergeUuidAttr(D, AL, OrigStrRef, Guid);
  if (UA)
    D->addAttr(UA);
}

static void handleMSInheritanceAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (!S.LangOpts.CPlusPlus) {
    S.Diag(AL.getLoc(), diag::err_attribute_not_supported_in_lang)
        << AL << AttributeLangSupport::C;
    return;
  }
  MSInheritanceAttr *IA = S.mergeMSInheritanceAttr(
      D, AL, /*BestCase=*/true, (MSInheritanceModel)AL.getSemanticSpelling());
  if (IA) {
    D->addAttr(IA);
    S.Consumer.AssignInheritanceModel(cast<CXXRecordDecl>(D));
  }
}

static void handleDeclspecThreadAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  const auto *VD = cast<VarDecl>(D);
  if (!S.Context.getTargetInfo().isTLSSupported()) {
    S.Diag(AL.getLoc(), diag::err_thread_unsupported);
    return;
  }
  if (VD->getTSCSpec() != TSCS_unspecified) {
    S.Diag(AL.getLoc(), diag::err_declspec_thread_on_thread_variable);
    return;
  }
  if (VD->hasLocalStorage()) {
    S.Diag(AL.getLoc(), diag::err_thread_non_global) << "__declspec(thread)";
    return;
  }
  D->addAttr(::new (S.Context) ThreadAttr(S.Context, AL));
}

static void handleAbiTagAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  SmallVector<StringRef, 4> Tags;
  for (unsigned I = 0, E = AL.getNumArgs(); I != E; ++I) {
    StringRef Tag;
    if (!S.checkStringLiteralArgumentAttr(AL, I, Tag))
      return;
    Tags.push_back(Tag);
  }

  if (const auto *NS = dyn_cast<NamespaceDecl>(D)) {
    if (!NS->isInline()) {
      S.Diag(AL.getLoc(), diag::warn_attr_abi_tag_namespace) << 0;
      return;
    }
    if (NS->isAnonymousNamespace()) {
      S.Diag(AL.getLoc(), diag::warn_attr_abi_tag_namespace) << 1;
      return;
    }
    if (AL.getNumArgs() == 0)
      Tags.push_back(NS->getName());
  } else if (!AL.checkAtLeastNumArgs(S, 1))
    return;

  // Store tags sorted and without duplicates.
  llvm::sort(Tags);
  Tags.erase(std::unique(Tags.begin(), Tags.end()), Tags.end());

  D->addAttr(::new (S.Context)
                 AbiTagAttr(S.Context, AL, Tags.data(), Tags.size()));
}

static void handleARMInterruptAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  // Check the attribute arguments.
  if (AL.getNumArgs() > 1) {
    S.Diag(AL.getLoc(), diag::err_attribute_too_many_arguments) << AL << 1;
    return;
  }

  StringRef Str;
  SourceLocation ArgLoc;

  if (AL.getNumArgs() == 0)
    Str = "";
  else if (!S.checkStringLiteralArgumentAttr(AL, 0, Str, &ArgLoc))
    return;

  ARMInterruptAttr::InterruptType Kind;
  if (!ARMInterruptAttr::ConvertStrToInterruptType(Str, Kind)) {
    S.Diag(AL.getLoc(), diag::warn_attribute_type_not_supported) << AL << Str
                                                                 << ArgLoc;
    return;
  }

  D->addAttr(::new (S.Context) ARMInterruptAttr(S.Context, AL, Kind));
}

static void handleMSP430InterruptAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  // MSP430 'interrupt' attribute is applied to
  // a function with no parameters and void return type.
  if (!isFunctionOrMethod(D)) {
    S.Diag(D->getLocation(), diag::warn_attribute_wrong_decl_type)
        << "'interrupt'" << ExpectedFunctionOrMethod;
    return;
  }

  if (hasFunctionProto(D) && getFunctionOrMethodNumParams(D) != 0) {
    S.Diag(D->getLocation(), diag::warn_interrupt_attribute_invalid)
        << /*MSP430*/ 1 << 0;
    return;
  }

  if (!getFunctionOrMethodResultType(D)->isVoidType()) {
    S.Diag(D->getLocation(), diag::warn_interrupt_attribute_invalid)
        << /*MSP430*/ 1 << 1;
    return;
  }

  // The attribute takes one integer argument.
  if (!AL.checkExactlyNumArgs(S, 1))
    return;

  if (!AL.isArgExpr(0)) {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_type)
        << AL << AANT_ArgumentIntegerConstant;
    return;
  }

  Expr *NumParamsExpr = static_cast<Expr *>(AL.getArgAsExpr(0));
  Optional<llvm::APSInt> NumParams = llvm::APSInt(32);
  if (!(NumParams = NumParamsExpr->getIntegerConstantExpr(S.Context))) {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_type)
        << AL << AANT_ArgumentIntegerConstant
        << NumParamsExpr->getSourceRange();
    return;
  }
  // The argument should be in range 0..63.
  unsigned Num = NumParams->getLimitedValue(255);
  if (Num > 63) {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_out_of_bounds)
        << AL << (int)NumParams->getSExtValue()
        << NumParamsExpr->getSourceRange();
    return;
  }

  D->addAttr(::new (S.Context) MSP430InterruptAttr(S.Context, AL, Num));
  D->addAttr(UsedAttr::CreateImplicit(S.Context));
}

static void handleMipsInterruptAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  // Only one optional argument permitted.
  if (AL.getNumArgs() > 1) {
    S.Diag(AL.getLoc(), diag::err_attribute_too_many_arguments) << AL << 1;
    return;
  }

  StringRef Str;
  SourceLocation ArgLoc;

  if (AL.getNumArgs() == 0)
    Str = "";
  else if (!S.checkStringLiteralArgumentAttr(AL, 0, Str, &ArgLoc))
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
    S.Diag(D->getLocation(), diag::warn_interrupt_attribute_invalid)
        << /*MIPS*/ 0 << 0;
    return;
  }

  if (!getFunctionOrMethodResultType(D)->isVoidType()) {
    S.Diag(D->getLocation(), diag::warn_interrupt_attribute_invalid)
        << /*MIPS*/ 0 << 1;
    return;
  }

  // We still have to do this manually because the Interrupt attributes are
  // a bit special due to sharing their spellings across targets.
  if (checkAttrMutualExclusion<Mips16Attr>(S, D, AL))
    return;

  MipsInterruptAttr::InterruptType Kind;
  if (!MipsInterruptAttr::ConvertStrToInterruptType(Str, Kind)) {
    S.Diag(AL.getLoc(), diag::warn_attribute_type_not_supported)
        << AL << "'" + std::string(Str) + "'";
    return;
  }

  D->addAttr(::new (S.Context) MipsInterruptAttr(S.Context, AL, Kind));
}

static void handleM68kInterruptAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (!AL.checkExactlyNumArgs(S, 1))
    return;

  if (!AL.isArgExpr(0)) {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_type)
        << AL << AANT_ArgumentIntegerConstant;
    return;
  }

  // FIXME: Check for decl - it should be void ()(void).

  Expr *NumParamsExpr = static_cast<Expr *>(AL.getArgAsExpr(0));
  auto MaybeNumParams = NumParamsExpr->getIntegerConstantExpr(S.Context);
  if (!MaybeNumParams) {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_type)
        << AL << AANT_ArgumentIntegerConstant
        << NumParamsExpr->getSourceRange();
    return;
  }

  unsigned Num = MaybeNumParams->getLimitedValue(255);
  if ((Num & 1) || Num > 30) {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_out_of_bounds)
        << AL << (int)MaybeNumParams->getSExtValue()
        << NumParamsExpr->getSourceRange();
    return;
  }

  D->addAttr(::new (S.Context) M68kInterruptAttr(S.Context, AL, Num));
  D->addAttr(UsedAttr::CreateImplicit(S.Context));
}

static void handleAnyX86InterruptAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  // Semantic checks for a function with the 'interrupt' attribute.
  // a) Must be a function.
  // b) Must have the 'void' return type.
  // c) Must take 1 or 2 arguments.
  // d) The 1st argument must be a pointer.
  // e) The 2nd argument (if any) must be an unsigned integer.
  if (!isFunctionOrMethod(D) || !hasFunctionProto(D) || isInstanceMethod(D) ||
      CXXMethodDecl::isStaticOverloadedOperator(
          cast<NamedDecl>(D)->getDeclName().getCXXOverloadedOperator())) {
    S.Diag(AL.getLoc(), diag::warn_attribute_wrong_decl_type)
        << AL << ExpectedFunctionWithProtoType;
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
    S.Diag(D->getBeginLoc(), diag::err_anyx86_interrupt_attribute)
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
  D->addAttr(::new (S.Context) AnyX86InterruptAttr(S.Context, AL));
  D->addAttr(UsedAttr::CreateImplicit(S.Context));
}

static void handleAVRInterruptAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (!isFunctionOrMethod(D)) {
    S.Diag(D->getLocation(), diag::warn_attribute_wrong_decl_type)
        << "'interrupt'" << ExpectedFunction;
    return;
  }

  if (!AL.checkExactlyNumArgs(S, 0))
    return;

  handleSimpleAttribute<AVRInterruptAttr>(S, D, AL);
}

static void handleAVRSignalAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (!isFunctionOrMethod(D)) {
    S.Diag(D->getLocation(), diag::warn_attribute_wrong_decl_type)
        << "'signal'" << ExpectedFunction;
    return;
  }

  if (!AL.checkExactlyNumArgs(S, 0))
    return;

  handleSimpleAttribute<AVRSignalAttr>(S, D, AL);
}

static void handleBPFPreserveAIRecord(Sema &S, RecordDecl *RD) {
  // Add preserve_access_index attribute to all fields and inner records.
  for (auto D : RD->decls()) {
    if (D->hasAttr<BPFPreserveAccessIndexAttr>())
      continue;

    D->addAttr(BPFPreserveAccessIndexAttr::CreateImplicit(S.Context));
    if (auto *Rec = dyn_cast<RecordDecl>(D))
      handleBPFPreserveAIRecord(S, Rec);
  }
}

static void handleBPFPreserveAccessIndexAttr(Sema &S, Decl *D,
    const ParsedAttr &AL) {
  auto *Rec = cast<RecordDecl>(D);
  handleBPFPreserveAIRecord(S, Rec);
  Rec->addAttr(::new (S.Context) BPFPreserveAccessIndexAttr(S.Context, AL));
}

static bool hasBTFTagAttr(Decl *D, StringRef Tag) {
  for (const auto *I : D->specific_attrs<BTFTagAttr>()) {
    if (I->getBTFTag() == Tag)
      return true;
  }
  return false;
}

static void handleBTFTagAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  StringRef Str;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, Str))
    return;
  if (hasBTFTagAttr(D, Str))
    return;

  D->addAttr(::new (S.Context) BTFTagAttr(S.Context, AL, Str));
}

BTFTagAttr *Sema::mergeBTFTagAttr(Decl *D, const BTFTagAttr &AL) {
  if (hasBTFTagAttr(D, AL.getBTFTag()))
    return nullptr;
  return ::new (Context) BTFTagAttr(Context, AL, AL.getBTFTag());
}

static void handleWebAssemblyExportNameAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (!isFunctionOrMethod(D)) {
    S.Diag(D->getLocation(), diag::warn_attribute_wrong_decl_type)
        << "'export_name'" << ExpectedFunction;
    return;
  }

  auto *FD = cast<FunctionDecl>(D);
  if (FD->isThisDeclarationADefinition()) {
    S.Diag(D->getLocation(), diag::err_alias_is_definition) << FD << 0;
    return;
  }

  StringRef Str;
  SourceLocation ArgLoc;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, Str, &ArgLoc))
    return;

  D->addAttr(::new (S.Context) WebAssemblyExportNameAttr(S.Context, AL, Str));
  D->addAttr(UsedAttr::CreateImplicit(S.Context));
}

WebAssemblyImportModuleAttr *
Sema::mergeImportModuleAttr(Decl *D, const WebAssemblyImportModuleAttr &AL) {
  auto *FD = cast<FunctionDecl>(D);

  if (const auto *ExistingAttr = FD->getAttr<WebAssemblyImportModuleAttr>()) {
    if (ExistingAttr->getImportModule() == AL.getImportModule())
      return nullptr;
    Diag(ExistingAttr->getLocation(), diag::warn_mismatched_import) << 0
      << ExistingAttr->getImportModule() << AL.getImportModule();
    Diag(AL.getLoc(), diag::note_previous_attribute);
    return nullptr;
  }
  if (FD->hasBody()) {
    Diag(AL.getLoc(), diag::warn_import_on_definition) << 0;
    return nullptr;
  }
  return ::new (Context) WebAssemblyImportModuleAttr(Context, AL,
                                                     AL.getImportModule());
}

WebAssemblyImportNameAttr *
Sema::mergeImportNameAttr(Decl *D, const WebAssemblyImportNameAttr &AL) {
  auto *FD = cast<FunctionDecl>(D);

  if (const auto *ExistingAttr = FD->getAttr<WebAssemblyImportNameAttr>()) {
    if (ExistingAttr->getImportName() == AL.getImportName())
      return nullptr;
    Diag(ExistingAttr->getLocation(), diag::warn_mismatched_import) << 1
      << ExistingAttr->getImportName() << AL.getImportName();
    Diag(AL.getLoc(), diag::note_previous_attribute);
    return nullptr;
  }
  if (FD->hasBody()) {
    Diag(AL.getLoc(), diag::warn_import_on_definition) << 1;
    return nullptr;
  }
  return ::new (Context) WebAssemblyImportNameAttr(Context, AL,
                                                   AL.getImportName());
}

static void
handleWebAssemblyImportModuleAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  auto *FD = cast<FunctionDecl>(D);

  StringRef Str;
  SourceLocation ArgLoc;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, Str, &ArgLoc))
    return;
  if (FD->hasBody()) {
    S.Diag(AL.getLoc(), diag::warn_import_on_definition) << 0;
    return;
  }

  FD->addAttr(::new (S.Context)
                  WebAssemblyImportModuleAttr(S.Context, AL, Str));
}

static void
handleWebAssemblyImportNameAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  auto *FD = cast<FunctionDecl>(D);

  StringRef Str;
  SourceLocation ArgLoc;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, Str, &ArgLoc))
    return;
  if (FD->hasBody()) {
    S.Diag(AL.getLoc(), diag::warn_import_on_definition) << 1;
    return;
  }

  FD->addAttr(::new (S.Context) WebAssemblyImportNameAttr(S.Context, AL, Str));
}

static void handleRISCVInterruptAttr(Sema &S, Decl *D,
                                     const ParsedAttr &AL) {
  // Warn about repeated attributes.
  if (const auto *A = D->getAttr<RISCVInterruptAttr>()) {
    S.Diag(AL.getRange().getBegin(),
      diag::warn_riscv_repeated_interrupt_attribute);
    S.Diag(A->getLocation(), diag::note_riscv_repeated_interrupt_attribute);
    return;
  }

  // Check the attribute argument. Argument is optional.
  if (!AL.checkAtMostNumArgs(S, 1))
    return;

  StringRef Str;
  SourceLocation ArgLoc;

  // 'machine'is the default interrupt mode.
  if (AL.getNumArgs() == 0)
    Str = "machine";
  else if (!S.checkStringLiteralArgumentAttr(AL, 0, Str, &ArgLoc))
    return;

  // Semantic checks for a function with the 'interrupt' attribute:
  // - Must be a function.
  // - Must have no parameters.
  // - Must have the 'void' return type.
  // - The attribute itself must either have no argument or one of the
  //   valid interrupt types, see [RISCVInterruptDocs].

  if (D->getFunctionType() == nullptr) {
    S.Diag(D->getLocation(), diag::warn_attribute_wrong_decl_type)
      << "'interrupt'" << ExpectedFunction;
    return;
  }

  if (hasFunctionProto(D) && getFunctionOrMethodNumParams(D) != 0) {
    S.Diag(D->getLocation(), diag::warn_interrupt_attribute_invalid)
      << /*RISC-V*/ 2 << 0;
    return;
  }

  if (!getFunctionOrMethodResultType(D)->isVoidType()) {
    S.Diag(D->getLocation(), diag::warn_interrupt_attribute_invalid)
      << /*RISC-V*/ 2 << 1;
    return;
  }

  RISCVInterruptAttr::InterruptType Kind;
  if (!RISCVInterruptAttr::ConvertStrToInterruptType(Str, Kind)) {
    S.Diag(AL.getLoc(), diag::warn_attribute_type_not_supported) << AL << Str
                                                                 << ArgLoc;
    return;
  }

  D->addAttr(::new (S.Context) RISCVInterruptAttr(S.Context, AL, Kind));
}

static void handleInterruptAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  // Dispatch the interrupt attribute based on the current target.
  switch (S.Context.getTargetInfo().getTriple().getArch()) {
  case llvm::Triple::msp430:
    handleMSP430InterruptAttr(S, D, AL);
    break;
  case llvm::Triple::mipsel:
  case llvm::Triple::mips:
    handleMipsInterruptAttr(S, D, AL);
    break;
  case llvm::Triple::m68k:
    handleM68kInterruptAttr(S, D, AL);
    break;
  case llvm::Triple::x86:
  case llvm::Triple::x86_64:
    handleAnyX86InterruptAttr(S, D, AL);
    break;
  case llvm::Triple::avr:
    handleAVRInterruptAttr(S, D, AL);
    break;
  case llvm::Triple::riscv32:
  case llvm::Triple::riscv64:
    handleRISCVInterruptAttr(S, D, AL);
    break;
  default:
    handleARMInterruptAttr(S, D, AL);
    break;
  }
}

static bool
checkAMDGPUFlatWorkGroupSizeArguments(Sema &S, Expr *MinExpr, Expr *MaxExpr,
                                      const AMDGPUFlatWorkGroupSizeAttr &Attr) {
  // Accept template arguments for now as they depend on something else.
  // We'll get to check them when they eventually get instantiated.
  if (MinExpr->isValueDependent() || MaxExpr->isValueDependent())
    return false;

  uint32_t Min = 0;
  if (!checkUInt32Argument(S, Attr, MinExpr, Min, 0))
    return true;

  uint32_t Max = 0;
  if (!checkUInt32Argument(S, Attr, MaxExpr, Max, 1))
    return true;

  if (Min == 0 && Max != 0) {
    S.Diag(Attr.getLocation(), diag::err_attribute_argument_invalid)
        << &Attr << 0;
    return true;
  }
  if (Min > Max) {
    S.Diag(Attr.getLocation(), diag::err_attribute_argument_invalid)
        << &Attr << 1;
    return true;
  }

  return false;
}

void Sema::addAMDGPUFlatWorkGroupSizeAttr(Decl *D,
                                          const AttributeCommonInfo &CI,
                                          Expr *MinExpr, Expr *MaxExpr) {
  AMDGPUFlatWorkGroupSizeAttr TmpAttr(Context, CI, MinExpr, MaxExpr);

  if (checkAMDGPUFlatWorkGroupSizeArguments(*this, MinExpr, MaxExpr, TmpAttr))
    return;

  D->addAttr(::new (Context)
                 AMDGPUFlatWorkGroupSizeAttr(Context, CI, MinExpr, MaxExpr));
}

static void handleAMDGPUFlatWorkGroupSizeAttr(Sema &S, Decl *D,
                                              const ParsedAttr &AL) {
  Expr *MinExpr = AL.getArgAsExpr(0);
  Expr *MaxExpr = AL.getArgAsExpr(1);

  S.addAMDGPUFlatWorkGroupSizeAttr(D, AL, MinExpr, MaxExpr);
}

static bool checkAMDGPUWavesPerEUArguments(Sema &S, Expr *MinExpr,
                                           Expr *MaxExpr,
                                           const AMDGPUWavesPerEUAttr &Attr) {
  if (S.DiagnoseUnexpandedParameterPack(MinExpr) ||
      (MaxExpr && S.DiagnoseUnexpandedParameterPack(MaxExpr)))
    return true;

  // Accept template arguments for now as they depend on something else.
  // We'll get to check them when they eventually get instantiated.
  if (MinExpr->isValueDependent() || (MaxExpr && MaxExpr->isValueDependent()))
    return false;

  uint32_t Min = 0;
  if (!checkUInt32Argument(S, Attr, MinExpr, Min, 0))
    return true;

  uint32_t Max = 0;
  if (MaxExpr && !checkUInt32Argument(S, Attr, MaxExpr, Max, 1))
    return true;

  if (Min == 0 && Max != 0) {
    S.Diag(Attr.getLocation(), diag::err_attribute_argument_invalid)
        << &Attr << 0;
    return true;
  }
  if (Max != 0 && Min > Max) {
    S.Diag(Attr.getLocation(), diag::err_attribute_argument_invalid)
        << &Attr << 1;
    return true;
  }

  return false;
}

void Sema::addAMDGPUWavesPerEUAttr(Decl *D, const AttributeCommonInfo &CI,
                                   Expr *MinExpr, Expr *MaxExpr) {
  AMDGPUWavesPerEUAttr TmpAttr(Context, CI, MinExpr, MaxExpr);

  if (checkAMDGPUWavesPerEUArguments(*this, MinExpr, MaxExpr, TmpAttr))
    return;

  D->addAttr(::new (Context)
                 AMDGPUWavesPerEUAttr(Context, CI, MinExpr, MaxExpr));
}

static void handleAMDGPUWavesPerEUAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (!AL.checkAtLeastNumArgs(S, 1) || !AL.checkAtMostNumArgs(S, 2))
    return;

  Expr *MinExpr = AL.getArgAsExpr(0);
  Expr *MaxExpr = (AL.getNumArgs() > 1) ? AL.getArgAsExpr(1) : nullptr;

  S.addAMDGPUWavesPerEUAttr(D, AL, MinExpr, MaxExpr);
}

static void handleAMDGPUNumSGPRAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  uint32_t NumSGPR = 0;
  Expr *NumSGPRExpr = AL.getArgAsExpr(0);
  if (!checkUInt32Argument(S, AL, NumSGPRExpr, NumSGPR))
    return;

  D->addAttr(::new (S.Context) AMDGPUNumSGPRAttr(S.Context, AL, NumSGPR));
}

static void handleAMDGPUNumVGPRAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  uint32_t NumVGPR = 0;
  Expr *NumVGPRExpr = AL.getArgAsExpr(0);
  if (!checkUInt32Argument(S, AL, NumVGPRExpr, NumVGPR))
    return;

  D->addAttr(::new (S.Context) AMDGPUNumVGPRAttr(S.Context, AL, NumVGPR));
}

static void handleX86ForceAlignArgPointerAttr(Sema &S, Decl *D,
                                              const ParsedAttr &AL) {
  // If we try to apply it to a function pointer, don't warn, but don't
  // do anything, either. It doesn't matter anyway, because there's nothing
  // special about calling a force_align_arg_pointer function.
  const auto *VD = dyn_cast<ValueDecl>(D);
  if (VD && VD->getType()->isFunctionPointerType())
    return;
  // Also don't warn on function pointer typedefs.
  const auto *TD = dyn_cast<TypedefNameDecl>(D);
  if (TD && (TD->getUnderlyingType()->isFunctionPointerType() ||
    TD->getUnderlyingType()->isFunctionType()))
    return;
  // Attribute can only be applied to function types.
  if (!isa<FunctionDecl>(D)) {
    S.Diag(AL.getLoc(), diag::warn_attribute_wrong_decl_type)
        << AL << ExpectedFunction;
    return;
  }

  D->addAttr(::new (S.Context) X86ForceAlignArgPointerAttr(S.Context, AL));
}

static void handleLayoutVersion(Sema &S, Decl *D, const ParsedAttr &AL) {
  uint32_t Version;
  Expr *VersionExpr = static_cast<Expr *>(AL.getArgAsExpr(0));
  if (!checkUInt32Argument(S, AL, AL.getArgAsExpr(0), Version))
    return;

  // TODO: Investigate what happens with the next major version of MSVC.
  if (Version != LangOptions::MSVC2015 / 100) {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_out_of_bounds)
        << AL << Version << VersionExpr->getSourceRange();
    return;
  }

  // The attribute expects a "major" version number like 19, but new versions of
  // MSVC have moved to updating the "minor", or less significant numbers, so we
  // have to multiply by 100 now.
  Version *= 100;

  D->addAttr(::new (S.Context) LayoutVersionAttr(S.Context, AL, Version));
}

DLLImportAttr *Sema::mergeDLLImportAttr(Decl *D,
                                        const AttributeCommonInfo &CI) {
  if (D->hasAttr<DLLExportAttr>()) {
    Diag(CI.getLoc(), diag::warn_attribute_ignored) << "'dllimport'";
    return nullptr;
  }

  if (D->hasAttr<DLLImportAttr>())
    return nullptr;

  return ::new (Context) DLLImportAttr(Context, CI);
}

DLLExportAttr *Sema::mergeDLLExportAttr(Decl *D,
                                        const AttributeCommonInfo &CI) {
  if (DLLImportAttr *Import = D->getAttr<DLLImportAttr>()) {
    Diag(Import->getLocation(), diag::warn_attribute_ignored) << Import;
    D->dropAttr<DLLImportAttr>();
  }

  if (D->hasAttr<DLLExportAttr>())
    return nullptr;

  return ::new (Context) DLLExportAttr(Context, CI);
}

static void handleDLLAttr(Sema &S, Decl *D, const ParsedAttr &A) {
  if (isa<ClassTemplatePartialSpecializationDecl>(D) &&
      (S.Context.getTargetInfo().shouldDLLImportComdatSymbols())) {
    S.Diag(A.getRange().getBegin(), diag::warn_attribute_ignored) << A;
    return;
  }

  if (const auto *FD = dyn_cast<FunctionDecl>(D)) {
    if (FD->isInlined() && A.getKind() == ParsedAttr::AT_DLLImport &&
        !(S.Context.getTargetInfo().shouldDLLImportComdatSymbols())) {
      // MinGW doesn't allow dllimport on inline functions.
      S.Diag(A.getRange().getBegin(), diag::warn_attribute_ignored_on_inline)
          << A;
      return;
    }
  }

  if (const auto *MD = dyn_cast<CXXMethodDecl>(D)) {
    if ((S.Context.getTargetInfo().shouldDLLImportComdatSymbols()) &&
        MD->getParent()->isLambda()) {
      S.Diag(A.getRange().getBegin(), diag::err_attribute_dll_lambda) << A;
      return;
    }
  }

  Attr *NewAttr = A.getKind() == ParsedAttr::AT_DLLExport
                      ? (Attr *)S.mergeDLLExportAttr(D, A)
                      : (Attr *)S.mergeDLLImportAttr(D, A);
  if (NewAttr)
    D->addAttr(NewAttr);
}

MSInheritanceAttr *
Sema::mergeMSInheritanceAttr(Decl *D, const AttributeCommonInfo &CI,
                             bool BestCase,
                             MSInheritanceModel Model) {
  if (MSInheritanceAttr *IA = D->getAttr<MSInheritanceAttr>()) {
    if (IA->getInheritanceModel() == Model)
      return nullptr;
    Diag(IA->getLocation(), diag::err_mismatched_ms_inheritance)
        << 1 /*previous declaration*/;
    Diag(CI.getLoc(), diag::note_previous_ms_inheritance);
    D->dropAttr<MSInheritanceAttr>();
  }

  auto *RD = cast<CXXRecordDecl>(D);
  if (RD->hasDefinition()) {
    if (checkMSInheritanceAttrOnDefinition(RD, CI.getRange(), BestCase,
                                           Model)) {
      return nullptr;
    }
  } else {
    if (isa<ClassTemplatePartialSpecializationDecl>(RD)) {
      Diag(CI.getLoc(), diag::warn_ignored_ms_inheritance)
          << 1 /*partial specialization*/;
      return nullptr;
    }
    if (RD->getDescribedClassTemplate()) {
      Diag(CI.getLoc(), diag::warn_ignored_ms_inheritance)
          << 0 /*primary template*/;
      return nullptr;
    }
  }

  return ::new (Context) MSInheritanceAttr(Context, CI, BestCase);
}

static void handleCapabilityAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
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
  if (AL.getKind() == ParsedAttr::AT_Capability &&
      !S.checkStringLiteralArgumentAttr(AL, 0, N, &LiteralLoc))
    return;

  D->addAttr(::new (S.Context) CapabilityAttr(S.Context, AL, N));
}

static void handleAssertCapabilityAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  SmallVector<Expr*, 1> Args;
  if (!checkLockFunAttrCommon(S, D, AL, Args))
    return;

  D->addAttr(::new (S.Context)
                 AssertCapabilityAttr(S.Context, AL, Args.data(), Args.size()));
}

static void handleAcquireCapabilityAttr(Sema &S, Decl *D,
                                        const ParsedAttr &AL) {
  SmallVector<Expr*, 1> Args;
  if (!checkLockFunAttrCommon(S, D, AL, Args))
    return;

  D->addAttr(::new (S.Context) AcquireCapabilityAttr(S.Context, AL, Args.data(),
                                                     Args.size()));
}

static void handleTryAcquireCapabilityAttr(Sema &S, Decl *D,
                                           const ParsedAttr &AL) {
  SmallVector<Expr*, 2> Args;
  if (!checkTryLockFunAttrCommon(S, D, AL, Args))
    return;

  D->addAttr(::new (S.Context) TryAcquireCapabilityAttr(
      S.Context, AL, AL.getArgAsExpr(0), Args.data(), Args.size()));
}

static void handleReleaseCapabilityAttr(Sema &S, Decl *D,
                                        const ParsedAttr &AL) {
  // Check that all arguments are lockable objects.
  SmallVector<Expr *, 1> Args;
  checkAttrArgsAreCapabilityObjs(S, D, AL, Args, 0, true);

  D->addAttr(::new (S.Context) ReleaseCapabilityAttr(S.Context, AL, Args.data(),
                                                     Args.size()));
}

static void handleRequiresCapabilityAttr(Sema &S, Decl *D,
                                         const ParsedAttr &AL) {
  if (!AL.checkAtLeastNumArgs(S, 1))
    return;

  // check that all arguments are lockable objects
  SmallVector<Expr*, 1> Args;
  checkAttrArgsAreCapabilityObjs(S, D, AL, Args);
  if (Args.empty())
    return;

  RequiresCapabilityAttr *RCA = ::new (S.Context)
      RequiresCapabilityAttr(S.Context, AL, Args.data(), Args.size());

  D->addAttr(RCA);
}

static void handleDeprecatedAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (const auto *NSD = dyn_cast<NamespaceDecl>(D)) {
    if (NSD->isAnonymousNamespace()) {
      S.Diag(AL.getLoc(), diag::warn_deprecated_anonymous_namespace);
      // Do not want to attach the attribute to the namespace because that will
      // cause confusing diagnostic reports for uses of declarations within the
      // namespace.
      return;
    }
  } else if (isa<UsingDecl, UnresolvedUsingTypenameDecl,
                 UnresolvedUsingValueDecl>(D)) {
    S.Diag(AL.getRange().getBegin(), diag::warn_deprecated_ignored_on_using)
        << AL;
    return;
  }

  // Handle the cases where the attribute has a text message.
  StringRef Str, Replacement;
  if (AL.isArgExpr(0) && AL.getArgAsExpr(0) &&
      !S.checkStringLiteralArgumentAttr(AL, 0, Str))
    return;

  // Support a single optional message only for Declspec and [[]] spellings.
  if (AL.isDeclspecAttribute() || AL.isStandardAttributeSyntax())
    AL.checkAtMostNumArgs(S, 1);
  else if (AL.isArgExpr(1) && AL.getArgAsExpr(1) &&
           !S.checkStringLiteralArgumentAttr(AL, 1, Replacement))
    return;

  if (!S.getLangOpts().CPlusPlus14 && AL.isCXX11Attribute() && !AL.isGNUScope())
    S.Diag(AL.getLoc(), diag::ext_cxx14_attr) << AL;

  D->addAttr(::new (S.Context) DeprecatedAttr(S.Context, AL, Str, Replacement));
}

static bool isGlobalVar(const Decl *D) {
  if (const auto *S = dyn_cast<VarDecl>(D))
    return S->hasGlobalStorage();
  return false;
}

static void handleNoSanitizeAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (!AL.checkAtLeastNumArgs(S, 1))
    return;

  std::vector<StringRef> Sanitizers;

  for (unsigned I = 0, E = AL.getNumArgs(); I != E; ++I) {
    StringRef SanitizerName;
    SourceLocation LiteralLoc;

    if (!S.checkStringLiteralArgumentAttr(AL, I, SanitizerName, &LiteralLoc))
      return;

    if (parseSanitizerValue(SanitizerName, /*AllowGroups=*/true) ==
            SanitizerMask() &&
        SanitizerName != "coverage")
      S.Diag(LiteralLoc, diag::warn_unknown_sanitizer_ignored) << SanitizerName;
    else if (isGlobalVar(D) && SanitizerName != "address")
      S.Diag(D->getLocation(), diag::err_attribute_wrong_decl_type)
          << AL << ExpectedFunctionOrMethod;
    Sanitizers.push_back(SanitizerName);
  }

  D->addAttr(::new (S.Context) NoSanitizeAttr(S.Context, AL, Sanitizers.data(),
                                              Sanitizers.size()));
}

static void handleNoSanitizeSpecificAttr(Sema &S, Decl *D,
                                         const ParsedAttr &AL) {
  StringRef AttrName = AL.getAttrName()->getName();
  normalizeName(AttrName);
  StringRef SanitizerName = llvm::StringSwitch<StringRef>(AttrName)
                                .Case("no_address_safety_analysis", "address")
                                .Case("no_sanitize_address", "address")
                                .Case("no_sanitize_thread", "thread")
                                .Case("no_sanitize_memory", "memory");
  if (isGlobalVar(D) && SanitizerName != "address")
    S.Diag(D->getLocation(), diag::err_attribute_wrong_decl_type)
        << AL << ExpectedFunction;

  // FIXME: Rather than create a NoSanitizeSpecificAttr, this creates a
  // NoSanitizeAttr object; but we need to calculate the correct spelling list
  // index rather than incorrectly assume the index for NoSanitizeSpecificAttr
  // has the same spellings as the index for NoSanitizeAttr. We don't have a
  // general way to "translate" between the two, so this hack attempts to work
  // around the issue with hard-coded indices. This is critical for calling
  // getSpelling() or prettyPrint() on the resulting semantic attribute object
  // without failing assertions.
  unsigned TranslatedSpellingIndex = 0;
  if (AL.isStandardAttributeSyntax())
    TranslatedSpellingIndex = 1;

  AttributeCommonInfo Info = AL;
  Info.setAttributeSpellingListIndex(TranslatedSpellingIndex);
  D->addAttr(::new (S.Context)
                 NoSanitizeAttr(S.Context, Info, &SanitizerName, 1));
}

static void handleInternalLinkageAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (InternalLinkageAttr *Internal = S.mergeInternalLinkageAttr(D, AL))
    D->addAttr(Internal);
}

static void handleOpenCLNoSVMAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (S.LangOpts.getOpenCLCompatibleVersion() < 200)
    S.Diag(AL.getLoc(), diag::err_attribute_requires_opencl_version)
        << AL << "2.0" << 1;
  else
    S.Diag(AL.getLoc(), diag::warn_opencl_attr_deprecated_ignored)
        << AL << S.LangOpts.getOpenCLVersionString();
}

static void handleOpenCLAccessAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (D->isInvalidDecl())
    return;

  // Check if there is only one access qualifier.
  if (D->hasAttr<OpenCLAccessAttr>()) {
    if (D->getAttr<OpenCLAccessAttr>()->getSemanticSpelling() ==
        AL.getSemanticSpelling()) {
      S.Diag(AL.getLoc(), diag::warn_duplicate_declspec)
          << AL.getAttrName()->getName() << AL.getRange();
    } else {
      S.Diag(AL.getLoc(), diag::err_opencl_multiple_access_qualifiers)
          << D->getSourceRange();
      D->setInvalidDecl(true);
      return;
    }
  }

  // OpenCL v2.0 s6.6 - read_write can be used for image types to specify that
  // an image object can be read and written. OpenCL v2.0 s6.13.6 - A kernel
  // cannot read from and write to the same pipe object. Using the read_write
  // (or __read_write) qualifier with the pipe qualifier is a compilation error.
  // OpenCL v3.0 s6.8 - For OpenCL C 2.0, or with the
  // __opencl_c_read_write_images feature, image objects specified as arguments
  // to a kernel can additionally be declared to be read-write.
  // C++ for OpenCL 1.0 inherits rule from OpenCL C v2.0.
  // C++ for OpenCL 2021 inherits rule from OpenCL C v3.0.
  if (const auto *PDecl = dyn_cast<ParmVarDecl>(D)) {
    const Type *DeclTy = PDecl->getType().getCanonicalType().getTypePtr();
    if (AL.getAttrName()->getName().find("read_write") != StringRef::npos) {
      bool ReadWriteImagesUnsupported =
          (S.getLangOpts().getOpenCLCompatibleVersion() < 200) ||
          (S.getLangOpts().getOpenCLCompatibleVersion() == 300 &&
           !S.getOpenCLOptions().isSupported("__opencl_c_read_write_images",
                                             S.getLangOpts()));
      if (ReadWriteImagesUnsupported || DeclTy->isPipeType()) {
        S.Diag(AL.getLoc(), diag::err_opencl_invalid_read_write)
            << AL << PDecl->getType() << DeclTy->isImageType();
        D->setInvalidDecl(true);
        return;
      }
    }
  }

  D->addAttr(::new (S.Context) OpenCLAccessAttr(S.Context, AL));
}

static void handleSYCLKernelAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  // The 'sycl_kernel' attribute applies only to function templates.
  const auto *FD = cast<FunctionDecl>(D);
  const FunctionTemplateDecl *FT = FD->getDescribedFunctionTemplate();
  assert(FT && "Function template is expected");

  // Function template must have at least two template parameters.
  const TemplateParameterList *TL = FT->getTemplateParameters();
  if (TL->size() < 2) {
    S.Diag(FT->getLocation(), diag::warn_sycl_kernel_num_of_template_params);
    return;
  }

  // Template parameters must be typenames.
  for (unsigned I = 0; I < 2; ++I) {
    const NamedDecl *TParam = TL->getParam(I);
    if (isa<NonTypeTemplateParmDecl>(TParam)) {
      S.Diag(FT->getLocation(),
             diag::warn_sycl_kernel_invalid_template_param_type);
      return;
    }
  }

  // Function must have at least one argument.
  if (getFunctionOrMethodNumParams(D) != 1) {
    S.Diag(FT->getLocation(), diag::warn_sycl_kernel_num_of_function_params);
    return;
  }

  // Function must return void.
  QualType RetTy = getFunctionOrMethodResultType(D);
  if (!RetTy->isVoidType()) {
    S.Diag(FT->getLocation(), diag::warn_sycl_kernel_return_type);
    return;
  }

  handleSimpleAttribute<SYCLKernelAttr>(S, D, AL);
}

static void handleDestroyAttr(Sema &S, Decl *D, const ParsedAttr &A) {
  if (!cast<VarDecl>(D)->hasGlobalStorage()) {
    S.Diag(D->getLocation(), diag::err_destroy_attr_on_non_static_var)
        << (A.getKind() == ParsedAttr::AT_AlwaysDestroy);
    return;
  }

  if (A.getKind() == ParsedAttr::AT_AlwaysDestroy)
    handleSimpleAttribute<AlwaysDestroyAttr>(S, D, A);
  else
    handleSimpleAttribute<NoDestroyAttr>(S, D, A);
}

static void handleUninitializedAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  assert(cast<VarDecl>(D)->getStorageDuration() == SD_Automatic &&
         "uninitialized is only valid on automatic duration variables");
  D->addAttr(::new (S.Context) UninitializedAttr(S.Context, AL));
}

static bool tryMakeVariablePseudoStrong(Sema &S, VarDecl *VD,
                                        bool DiagnoseFailure) {
  QualType Ty = VD->getType();
  if (!Ty->isObjCRetainableType()) {
    if (DiagnoseFailure) {
      S.Diag(VD->getBeginLoc(), diag::warn_ignored_objc_externally_retained)
          << 0;
    }
    return false;
  }

  Qualifiers::ObjCLifetime LifetimeQual = Ty.getQualifiers().getObjCLifetime();

  // Sema::inferObjCARCLifetime must run after processing decl attributes
  // (because __block lowers to an attribute), so if the lifetime hasn't been
  // explicitly specified, infer it locally now.
  if (LifetimeQual == Qualifiers::OCL_None)
    LifetimeQual = Ty->getObjCARCImplicitLifetime();

  // The attributes only really makes sense for __strong variables; ignore any
  // attempts to annotate a parameter with any other lifetime qualifier.
  if (LifetimeQual != Qualifiers::OCL_Strong) {
    if (DiagnoseFailure) {
      S.Diag(VD->getBeginLoc(), diag::warn_ignored_objc_externally_retained)
          << 1;
    }
    return false;
  }

  // Tampering with the type of a VarDecl here is a bit of a hack, but we need
  // to ensure that the variable is 'const' so that we can error on
  // modification, which can otherwise over-release.
  VD->setType(Ty.withConst());
  VD->setARCPseudoStrong(true);
  return true;
}

static void handleObjCExternallyRetainedAttr(Sema &S, Decl *D,
                                             const ParsedAttr &AL) {
  if (auto *VD = dyn_cast<VarDecl>(D)) {
    assert(!isa<ParmVarDecl>(VD) && "should be diagnosed automatically");
    if (!VD->hasLocalStorage()) {
      S.Diag(D->getBeginLoc(), diag::warn_ignored_objc_externally_retained)
          << 0;
      return;
    }

    if (!tryMakeVariablePseudoStrong(S, VD, /*DiagnoseFailure=*/true))
      return;

    handleSimpleAttribute<ObjCExternallyRetainedAttr>(S, D, AL);
    return;
  }

  // If D is a function-like declaration (method, block, or function), then we
  // make every parameter psuedo-strong.
  unsigned NumParams =
      hasFunctionProto(D) ? getFunctionOrMethodNumParams(D) : 0;
  for (unsigned I = 0; I != NumParams; ++I) {
    auto *PVD = const_cast<ParmVarDecl *>(getFunctionOrMethodParam(D, I));
    QualType Ty = PVD->getType();

    // If a user wrote a parameter with __strong explicitly, then assume they
    // want "real" strong semantics for that parameter. This works because if
    // the parameter was written with __strong, then the strong qualifier will
    // be non-local.
    if (Ty.getLocalUnqualifiedType().getQualifiers().getObjCLifetime() ==
        Qualifiers::OCL_Strong)
      continue;

    tryMakeVariablePseudoStrong(S, PVD, /*DiagnoseFailure=*/false);
  }
  handleSimpleAttribute<ObjCExternallyRetainedAttr>(S, D, AL);
}

static void handleMIGServerRoutineAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  // Check that the return type is a `typedef int kern_return_t` or a typedef
  // around it, because otherwise MIG convention checks make no sense.
  // BlockDecl doesn't store a return type, so it's annoying to check,
  // so let's skip it for now.
  if (!isa<BlockDecl>(D)) {
    QualType T = getFunctionOrMethodResultType(D);
    bool IsKernReturnT = false;
    while (const auto *TT = T->getAs<TypedefType>()) {
      IsKernReturnT = (TT->getDecl()->getName() == "kern_return_t");
      T = TT->desugar();
    }
    if (!IsKernReturnT || T.getCanonicalType() != S.getASTContext().IntTy) {
      S.Diag(D->getBeginLoc(),
             diag::warn_mig_server_routine_does_not_return_kern_return_t);
      return;
    }
  }

  handleSimpleAttribute<MIGServerRoutineAttr>(S, D, AL);
}

static void handleMSAllocatorAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  // Warn if the return type is not a pointer or reference type.
  if (auto *FD = dyn_cast<FunctionDecl>(D)) {
    QualType RetTy = FD->getReturnType();
    if (!RetTy->isPointerType() && !RetTy->isReferenceType()) {
      S.Diag(AL.getLoc(), diag::warn_declspec_allocator_nonpointer)
          << AL.getRange() << RetTy;
      return;
    }
  }

  handleSimpleAttribute<MSAllocatorAttr>(S, D, AL);
}

static void handleAcquireHandleAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (AL.isUsedAsTypeAttr())
    return;
  // Warn if the parameter is definitely not an output parameter.
  if (const auto *PVD = dyn_cast<ParmVarDecl>(D)) {
    if (PVD->getType()->isIntegerType()) {
      S.Diag(AL.getLoc(), diag::err_attribute_output_parameter)
          << AL.getRange();
      return;
    }
  }
  StringRef Argument;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, Argument))
    return;
  D->addAttr(AcquireHandleAttr::Create(S.Context, Argument, AL));
}

template<typename Attr>
static void handleHandleAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  StringRef Argument;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, Argument))
    return;
  D->addAttr(Attr::Create(S.Context, Argument, AL));
}

static void handleCFGuardAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  // The guard attribute takes a single identifier argument.

  if (!AL.isArgIdent(0)) {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_type)
        << AL << AANT_ArgumentIdentifier;
    return;
  }

  CFGuardAttr::GuardArg Arg;
  IdentifierInfo *II = AL.getArgAsIdent(0)->Ident;
  if (!CFGuardAttr::ConvertStrToGuardArg(II->getName(), Arg)) {
    S.Diag(AL.getLoc(), diag::warn_attribute_type_not_supported) << AL << II;
    return;
  }

  D->addAttr(::new (S.Context) CFGuardAttr(S.Context, AL, Arg));
}


template <typename AttrTy>
static const AttrTy *findEnforceTCBAttrByName(Decl *D, StringRef Name) {
  auto Attrs = D->specific_attrs<AttrTy>();
  auto I = llvm::find_if(Attrs,
                         [Name](const AttrTy *A) {
                           return A->getTCBName() == Name;
                         });
  return I == Attrs.end() ? nullptr : *I;
}

template <typename AttrTy, typename ConflictingAttrTy>
static void handleEnforceTCBAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  StringRef Argument;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, Argument))
    return;

  // A function cannot be have both regular and leaf membership in the same TCB.
  if (const ConflictingAttrTy *ConflictingAttr =
      findEnforceTCBAttrByName<ConflictingAttrTy>(D, Argument)) {
    // We could attach a note to the other attribute but in this case
    // there's no need given how the two are very close to each other.
    S.Diag(AL.getLoc(), diag::err_tcb_conflicting_attributes)
      << AL.getAttrName()->getName() << ConflictingAttr->getAttrName()->getName()
      << Argument;

    // Error recovery: drop the non-leaf attribute so that to suppress
    // all future warnings caused by erroneous attributes. The leaf attribute
    // needs to be kept because it can only suppresses warnings, not cause them.
    D->dropAttr<EnforceTCBAttr>();
    return;
  }

  D->addAttr(AttrTy::Create(S.Context, Argument, AL));
}

template <typename AttrTy, typename ConflictingAttrTy>
static AttrTy *mergeEnforceTCBAttrImpl(Sema &S, Decl *D, const AttrTy &AL) {
  // Check if the new redeclaration has different leaf-ness in the same TCB.
  StringRef TCBName = AL.getTCBName();
  if (const ConflictingAttrTy *ConflictingAttr =
      findEnforceTCBAttrByName<ConflictingAttrTy>(D, TCBName)) {
    S.Diag(ConflictingAttr->getLoc(), diag::err_tcb_conflicting_attributes)
      << ConflictingAttr->getAttrName()->getName()
      << AL.getAttrName()->getName() << TCBName;

    // Add a note so that the user could easily find the conflicting attribute.
    S.Diag(AL.getLoc(), diag::note_conflicting_attribute);

    // More error recovery.
    D->dropAttr<EnforceTCBAttr>();
    return nullptr;
  }

  ASTContext &Context = S.getASTContext();
  return ::new(Context) AttrTy(Context, AL, AL.getTCBName());
}

EnforceTCBAttr *Sema::mergeEnforceTCBAttr(Decl *D, const EnforceTCBAttr &AL) {
  return mergeEnforceTCBAttrImpl<EnforceTCBAttr, EnforceTCBLeafAttr>(
      *this, D, AL);
}

EnforceTCBLeafAttr *Sema::mergeEnforceTCBLeafAttr(
    Decl *D, const EnforceTCBLeafAttr &AL) {
  return mergeEnforceTCBAttrImpl<EnforceTCBLeafAttr, EnforceTCBAttr>(
      *this, D, AL);
}

//===----------------------------------------------------------------------===//
// Top Level Sema Entry Points
//===----------------------------------------------------------------------===//

/// ProcessDeclAttribute - Apply the specific attribute to the specified decl if
/// the attribute applies to decls.  If the attribute is a type attribute, just
/// silently ignore it if a GNU attribute.
static void ProcessDeclAttribute(Sema &S, Scope *scope, Decl *D,
                                 const ParsedAttr &AL,
                                 bool IncludeCXX11Attributes) {
  if (AL.isInvalid() || AL.getKind() == ParsedAttr::IgnoredAttribute)
    return;

  // Ignore C++11 attributes on declarator chunks: they appertain to the type
  // instead.
  if (AL.isCXX11Attribute() && !IncludeCXX11Attributes)
    return;

  // Unknown attributes are automatically warned on. Target-specific attributes
  // which do not apply to the current target architecture are treated as
  // though they were unknown attributes.
  if (AL.getKind() == ParsedAttr::UnknownAttribute ||
      !AL.existsInTarget(S.Context.getTargetInfo())) {
    S.Diag(AL.getLoc(),
           AL.isDeclspecAttribute()
               ? (unsigned)diag::warn_unhandled_ms_attribute_ignored
               : (unsigned)diag::warn_unknown_attribute_ignored)
        << AL << AL.getRange();
    return;
  }

  if (S.checkCommonAttributeFeatures(D, AL))
    return;

  switch (AL.getKind()) {
  default:
    if (AL.getInfo().handleDeclAttribute(S, D, AL) != ParsedAttrInfo::NotHandled)
      break;
    if (!AL.isStmtAttr()) {
      // Type attributes are handled elsewhere; silently move on.
      assert(AL.isTypeAttr() && "Non-type attribute not handled");
      break;
    }
    // N.B., ClangAttrEmitter.cpp emits a diagnostic helper that ensures a
    // statement attribute is not written on a declaration, but this code is
    // needed for attributes in Attr.td that do not list any subjects.
    S.Diag(AL.getLoc(), diag::err_stmt_attribute_invalid_on_decl)
        << AL << D->getLocation();
    break;
  case ParsedAttr::AT_Interrupt:
    handleInterruptAttr(S, D, AL);
    break;
  case ParsedAttr::AT_X86ForceAlignArgPointer:
    handleX86ForceAlignArgPointerAttr(S, D, AL);
    break;
  case ParsedAttr::AT_DLLExport:
  case ParsedAttr::AT_DLLImport:
    handleDLLAttr(S, D, AL);
    break;
  case ParsedAttr::AT_AMDGPUFlatWorkGroupSize:
    handleAMDGPUFlatWorkGroupSizeAttr(S, D, AL);
    break;
  case ParsedAttr::AT_AMDGPUWavesPerEU:
    handleAMDGPUWavesPerEUAttr(S, D, AL);
    break;
  case ParsedAttr::AT_AMDGPUNumSGPR:
    handleAMDGPUNumSGPRAttr(S, D, AL);
    break;
  case ParsedAttr::AT_AMDGPUNumVGPR:
    handleAMDGPUNumVGPRAttr(S, D, AL);
    break;
  case ParsedAttr::AT_AVRSignal:
    handleAVRSignalAttr(S, D, AL);
    break;
  case ParsedAttr::AT_BPFPreserveAccessIndex:
    handleBPFPreserveAccessIndexAttr(S, D, AL);
    break;
  case ParsedAttr::AT_BTFTag:
    handleBTFTagAttr(S, D, AL);
    break;
  case ParsedAttr::AT_WebAssemblyExportName:
    handleWebAssemblyExportNameAttr(S, D, AL);
    break;
  case ParsedAttr::AT_WebAssemblyImportModule:
    handleWebAssemblyImportModuleAttr(S, D, AL);
    break;
  case ParsedAttr::AT_WebAssemblyImportName:
    handleWebAssemblyImportNameAttr(S, D, AL);
    break;
  case ParsedAttr::AT_IBOutlet:
    handleIBOutlet(S, D, AL);
    break;
  case ParsedAttr::AT_IBOutletCollection:
    handleIBOutletCollection(S, D, AL);
    break;
  case ParsedAttr::AT_IFunc:
    handleIFuncAttr(S, D, AL);
    break;
  case ParsedAttr::AT_Alias:
    handleAliasAttr(S, D, AL);
    break;
  case ParsedAttr::AT_Aligned:
    handleAlignedAttr(S, D, AL);
    break;
  case ParsedAttr::AT_AlignValue:
    handleAlignValueAttr(S, D, AL);
    break;
  case ParsedAttr::AT_AllocSize:
    handleAllocSizeAttr(S, D, AL);
    break;
  case ParsedAttr::AT_AlwaysInline:
    handleAlwaysInlineAttr(S, D, AL);
    break;
  case ParsedAttr::AT_AnalyzerNoReturn:
    handleAnalyzerNoReturnAttr(S, D, AL);
    break;
  case ParsedAttr::AT_TLSModel:
    handleTLSModelAttr(S, D, AL);
    break;
  case ParsedAttr::AT_Annotate:
    handleAnnotateAttr(S, D, AL);
    break;
  case ParsedAttr::AT_Availability:
    handleAvailabilityAttr(S, D, AL);
    break;
  case ParsedAttr::AT_CarriesDependency:
    handleDependencyAttr(S, scope, D, AL);
    break;
  case ParsedAttr::AT_CPUDispatch:
  case ParsedAttr::AT_CPUSpecific:
    handleCPUSpecificAttr(S, D, AL);
    break;
  case ParsedAttr::AT_Common:
    handleCommonAttr(S, D, AL);
    break;
  case ParsedAttr::AT_CUDAConstant:
    handleConstantAttr(S, D, AL);
    break;
  case ParsedAttr::AT_PassObjectSize:
    handlePassObjectSizeAttr(S, D, AL);
    break;
  case ParsedAttr::AT_Constructor:
      handleConstructorAttr(S, D, AL);
    break;
  case ParsedAttr::AT_Deprecated:
    handleDeprecatedAttr(S, D, AL);
    break;
  case ParsedAttr::AT_Destructor:
      handleDestructorAttr(S, D, AL);
    break;
  case ParsedAttr::AT_EnableIf:
    handleEnableIfAttr(S, D, AL);
    break;
  case ParsedAttr::AT_Error:
    handleErrorAttr(S, D, AL);
    break;
  case ParsedAttr::AT_DiagnoseIf:
    handleDiagnoseIfAttr(S, D, AL);
    break;
  case ParsedAttr::AT_NoBuiltin:
    handleNoBuiltinAttr(S, D, AL);
    break;
  case ParsedAttr::AT_ExtVectorType:
    handleExtVectorTypeAttr(S, D, AL);
    break;
  case ParsedAttr::AT_ExternalSourceSymbol:
    handleExternalSourceSymbolAttr(S, D, AL);
    break;
  case ParsedAttr::AT_MinSize:
    handleMinSizeAttr(S, D, AL);
    break;
  case ParsedAttr::AT_OptimizeNone:
    handleOptimizeNoneAttr(S, D, AL);
    break;
  case ParsedAttr::AT_EnumExtensibility:
    handleEnumExtensibilityAttr(S, D, AL);
    break;
  case ParsedAttr::AT_SYCLKernel:
    handleSYCLKernelAttr(S, D, AL);
    break;
  case ParsedAttr::AT_Format:
    handleFormatAttr(S, D, AL);
    break;
  case ParsedAttr::AT_FormatArg:
    handleFormatArgAttr(S, D, AL);
    break;
  case ParsedAttr::AT_Callback:
    handleCallbackAttr(S, D, AL);
    break;
  case ParsedAttr::AT_CalledOnce:
    handleCalledOnceAttr(S, D, AL);
    break;
  case ParsedAttr::AT_CUDAGlobal:
    handleGlobalAttr(S, D, AL);
    break;
  case ParsedAttr::AT_CUDADevice:
    handleDeviceAttr(S, D, AL);
    break;
  case ParsedAttr::AT_HIPManaged:
    handleManagedAttr(S, D, AL);
    break;
  case ParsedAttr::AT_GNUInline:
    handleGNUInlineAttr(S, D, AL);
    break;
  case ParsedAttr::AT_CUDALaunchBounds:
    handleLaunchBoundsAttr(S, D, AL);
    break;
  case ParsedAttr::AT_Restrict:
    handleRestrictAttr(S, D, AL);
    break;
  case ParsedAttr::AT_Mode:
    handleModeAttr(S, D, AL);
    break;
  case ParsedAttr::AT_NonNull:
    if (auto *PVD = dyn_cast<ParmVarDecl>(D))
      handleNonNullAttrParameter(S, PVD, AL);
    else
      handleNonNullAttr(S, D, AL);
    break;
  case ParsedAttr::AT_ReturnsNonNull:
    handleReturnsNonNullAttr(S, D, AL);
    break;
  case ParsedAttr::AT_NoEscape:
    handleNoEscapeAttr(S, D, AL);
    break;
  case ParsedAttr::AT_AssumeAligned:
    handleAssumeAlignedAttr(S, D, AL);
    break;
  case ParsedAttr::AT_AllocAlign:
    handleAllocAlignAttr(S, D, AL);
    break;
  case ParsedAttr::AT_Ownership:
    handleOwnershipAttr(S, D, AL);
    break;
  case ParsedAttr::AT_Naked:
    handleNakedAttr(S, D, AL);
    break;
  case ParsedAttr::AT_NoReturn:
    handleNoReturnAttr(S, D, AL);
    break;
  case ParsedAttr::AT_AnyX86NoCfCheck:
    handleNoCfCheckAttr(S, D, AL);
    break;
  case ParsedAttr::AT_NoThrow:
    if (!AL.isUsedAsTypeAttr())
      handleSimpleAttribute<NoThrowAttr>(S, D, AL);
    break;
  case ParsedAttr::AT_CUDAShared:
    handleSharedAttr(S, D, AL);
    break;
  case ParsedAttr::AT_VecReturn:
    handleVecReturnAttr(S, D, AL);
    break;
  case ParsedAttr::AT_ObjCOwnership:
    handleObjCOwnershipAttr(S, D, AL);
    break;
  case ParsedAttr::AT_ObjCPreciseLifetime:
    handleObjCPreciseLifetimeAttr(S, D, AL);
    break;
  case ParsedAttr::AT_ObjCReturnsInnerPointer:
    handleObjCReturnsInnerPointerAttr(S, D, AL);
    break;
  case ParsedAttr::AT_ObjCRequiresSuper:
    handleObjCRequiresSuperAttr(S, D, AL);
    break;
  case ParsedAttr::AT_ObjCBridge:
    handleObjCBridgeAttr(S, D, AL);
    break;
  case ParsedAttr::AT_ObjCBridgeMutable:
    handleObjCBridgeMutableAttr(S, D, AL);
    break;
  case ParsedAttr::AT_ObjCBridgeRelated:
    handleObjCBridgeRelatedAttr(S, D, AL);
    break;
  case ParsedAttr::AT_ObjCDesignatedInitializer:
    handleObjCDesignatedInitializer(S, D, AL);
    break;
  case ParsedAttr::AT_ObjCRuntimeName:
    handleObjCRuntimeName(S, D, AL);
    break;
  case ParsedAttr::AT_ObjCBoxable:
    handleObjCBoxable(S, D, AL);
    break;
  case ParsedAttr::AT_NSErrorDomain:
    handleNSErrorDomain(S, D, AL);
    break;
  case ParsedAttr::AT_CFConsumed:
  case ParsedAttr::AT_NSConsumed:
  case ParsedAttr::AT_OSConsumed:
    S.AddXConsumedAttr(D, AL, parsedAttrToRetainOwnershipKind(AL),
                       /*IsTemplateInstantiation=*/false);
    break;
  case ParsedAttr::AT_OSReturnsRetainedOnZero:
    handleSimpleAttributeOrDiagnose<OSReturnsRetainedOnZeroAttr>(
        S, D, AL, isValidOSObjectOutParameter(D),
        diag::warn_ns_attribute_wrong_parameter_type,
        /*Extra Args=*/AL, /*pointer-to-OSObject-pointer*/ 3, AL.getRange());
    break;
  case ParsedAttr::AT_OSReturnsRetainedOnNonZero:
    handleSimpleAttributeOrDiagnose<OSReturnsRetainedOnNonZeroAttr>(
        S, D, AL, isValidOSObjectOutParameter(D),
        diag::warn_ns_attribute_wrong_parameter_type,
        /*Extra Args=*/AL, /*pointer-to-OSObject-poointer*/ 3, AL.getRange());
    break;
  case ParsedAttr::AT_NSReturnsAutoreleased:
  case ParsedAttr::AT_NSReturnsNotRetained:
  case ParsedAttr::AT_NSReturnsRetained:
  case ParsedAttr::AT_CFReturnsNotRetained:
  case ParsedAttr::AT_CFReturnsRetained:
  case ParsedAttr::AT_OSReturnsNotRetained:
  case ParsedAttr::AT_OSReturnsRetained:
    handleXReturnsXRetainedAttr(S, D, AL);
    break;
  case ParsedAttr::AT_WorkGroupSizeHint:
    handleWorkGroupSize<WorkGroupSizeHintAttr>(S, D, AL);
    break;
  case ParsedAttr::AT_ReqdWorkGroupSize:
    handleWorkGroupSize<ReqdWorkGroupSizeAttr>(S, D, AL);
    break;
  case ParsedAttr::AT_OpenCLIntelReqdSubGroupSize:
    handleSubGroupSize(S, D, AL);
    break;
  case ParsedAttr::AT_VecTypeHint:
    handleVecTypeHint(S, D, AL);
    break;
  case ParsedAttr::AT_InitPriority:
      handleInitPriorityAttr(S, D, AL);
    break;
  case ParsedAttr::AT_Packed:
    handlePackedAttr(S, D, AL);
    break;
  case ParsedAttr::AT_PreferredName:
    handlePreferredName(S, D, AL);
    break;
  case ParsedAttr::AT_Section:
    handleSectionAttr(S, D, AL);
    break;
  case ParsedAttr::AT_CodeSeg:
    handleCodeSegAttr(S, D, AL);
    break;
  case ParsedAttr::AT_Target:
    handleTargetAttr(S, D, AL);
    break;
  case ParsedAttr::AT_MinVectorWidth:
    handleMinVectorWidthAttr(S, D, AL);
    break;
  case ParsedAttr::AT_Unavailable:
    handleAttrWithMessage<UnavailableAttr>(S, D, AL);
    break;
  case ParsedAttr::AT_Assumption:
    handleAssumumptionAttr(S, D, AL);
    break;
  case ParsedAttr::AT_ObjCDirect:
    handleObjCDirectAttr(S, D, AL);
    break;
  case ParsedAttr::AT_ObjCDirectMembers:
    handleObjCDirectMembersAttr(S, D, AL);
    handleSimpleAttribute<ObjCDirectMembersAttr>(S, D, AL);
    break;
  case ParsedAttr::AT_ObjCExplicitProtocolImpl:
    handleObjCSuppresProtocolAttr(S, D, AL);
    break;
  case ParsedAttr::AT_Unused:
    handleUnusedAttr(S, D, AL);
    break;
  case ParsedAttr::AT_Visibility:
    handleVisibilityAttr(S, D, AL, false);
    break;
  case ParsedAttr::AT_TypeVisibility:
    handleVisibilityAttr(S, D, AL, true);
    break;
  case ParsedAttr::AT_WarnUnusedResult:
    handleWarnUnusedResult(S, D, AL);
    break;
  case ParsedAttr::AT_WeakRef:
    handleWeakRefAttr(S, D, AL);
    break;
  case ParsedAttr::AT_WeakImport:
    handleWeakImportAttr(S, D, AL);
    break;
  case ParsedAttr::AT_TransparentUnion:
    handleTransparentUnionAttr(S, D, AL);
    break;
  case ParsedAttr::AT_ObjCMethodFamily:
    handleObjCMethodFamilyAttr(S, D, AL);
    break;
  case ParsedAttr::AT_ObjCNSObject:
    handleObjCNSObject(S, D, AL);
    break;
  case ParsedAttr::AT_ObjCIndependentClass:
    handleObjCIndependentClass(S, D, AL);
    break;
  case ParsedAttr::AT_Blocks:
    handleBlocksAttr(S, D, AL);
    break;
  case ParsedAttr::AT_Sentinel:
    handleSentinelAttr(S, D, AL);
    break;
  case ParsedAttr::AT_Cleanup:
    handleCleanupAttr(S, D, AL);
    break;
  case ParsedAttr::AT_NoDebug:
    handleNoDebugAttr(S, D, AL);
    break;
  case ParsedAttr::AT_CmseNSEntry:
    handleCmseNSEntryAttr(S, D, AL);
    break;
  case ParsedAttr::AT_StdCall:
  case ParsedAttr::AT_CDecl:
  case ParsedAttr::AT_FastCall:
  case ParsedAttr::AT_ThisCall:
  case ParsedAttr::AT_Pascal:
  case ParsedAttr::AT_RegCall:
  case ParsedAttr::AT_SwiftCall:
  case ParsedAttr::AT_SwiftAsyncCall:
  case ParsedAttr::AT_VectorCall:
  case ParsedAttr::AT_MSABI:
  case ParsedAttr::AT_SysVABI:
  case ParsedAttr::AT_Pcs:
  case ParsedAttr::AT_IntelOclBicc:
  case ParsedAttr::AT_PreserveMost:
  case ParsedAttr::AT_PreserveAll:
  case ParsedAttr::AT_AArch64VectorPcs:
    handleCallConvAttr(S, D, AL);
    break;
  case ParsedAttr::AT_Suppress:
    handleSuppressAttr(S, D, AL);
    break;
  case ParsedAttr::AT_Owner:
  case ParsedAttr::AT_Pointer:
    handleLifetimeCategoryAttr(S, D, AL);
    break;
  case ParsedAttr::AT_OpenCLAccess:
    handleOpenCLAccessAttr(S, D, AL);
    break;
  case ParsedAttr::AT_OpenCLNoSVM:
    handleOpenCLNoSVMAttr(S, D, AL);
    break;
  case ParsedAttr::AT_SwiftContext:
    S.AddParameterABIAttr(D, AL, ParameterABI::SwiftContext);
    break;
  case ParsedAttr::AT_SwiftAsyncContext:
    S.AddParameterABIAttr(D, AL, ParameterABI::SwiftAsyncContext);
    break;
  case ParsedAttr::AT_SwiftErrorResult:
    S.AddParameterABIAttr(D, AL, ParameterABI::SwiftErrorResult);
    break;
  case ParsedAttr::AT_SwiftIndirectResult:
    S.AddParameterABIAttr(D, AL, ParameterABI::SwiftIndirectResult);
    break;
  case ParsedAttr::AT_InternalLinkage:
    handleInternalLinkageAttr(S, D, AL);
    break;

  // Microsoft attributes:
  case ParsedAttr::AT_LayoutVersion:
    handleLayoutVersion(S, D, AL);
    break;
  case ParsedAttr::AT_Uuid:
    handleUuidAttr(S, D, AL);
    break;
  case ParsedAttr::AT_MSInheritance:
    handleMSInheritanceAttr(S, D, AL);
    break;
  case ParsedAttr::AT_Thread:
    handleDeclspecThreadAttr(S, D, AL);
    break;

  case ParsedAttr::AT_AbiTag:
    handleAbiTagAttr(S, D, AL);
    break;
  case ParsedAttr::AT_CFGuard:
    handleCFGuardAttr(S, D, AL);
    break;

  // Thread safety attributes:
  case ParsedAttr::AT_AssertExclusiveLock:
    handleAssertExclusiveLockAttr(S, D, AL);
    break;
  case ParsedAttr::AT_AssertSharedLock:
    handleAssertSharedLockAttr(S, D, AL);
    break;
  case ParsedAttr::AT_PtGuardedVar:
    handlePtGuardedVarAttr(S, D, AL);
    break;
  case ParsedAttr::AT_NoSanitize:
    handleNoSanitizeAttr(S, D, AL);
    break;
  case ParsedAttr::AT_NoSanitizeSpecific:
    handleNoSanitizeSpecificAttr(S, D, AL);
    break;
  case ParsedAttr::AT_GuardedBy:
    handleGuardedByAttr(S, D, AL);
    break;
  case ParsedAttr::AT_PtGuardedBy:
    handlePtGuardedByAttr(S, D, AL);
    break;
  case ParsedAttr::AT_ExclusiveTrylockFunction:
    handleExclusiveTrylockFunctionAttr(S, D, AL);
    break;
  case ParsedAttr::AT_LockReturned:
    handleLockReturnedAttr(S, D, AL);
    break;
  case ParsedAttr::AT_LocksExcluded:
    handleLocksExcludedAttr(S, D, AL);
    break;
  case ParsedAttr::AT_SharedTrylockFunction:
    handleSharedTrylockFunctionAttr(S, D, AL);
    break;
  case ParsedAttr::AT_AcquiredBefore:
    handleAcquiredBeforeAttr(S, D, AL);
    break;
  case ParsedAttr::AT_AcquiredAfter:
    handleAcquiredAfterAttr(S, D, AL);
    break;

  // Capability analysis attributes.
  case ParsedAttr::AT_Capability:
  case ParsedAttr::AT_Lockable:
    handleCapabilityAttr(S, D, AL);
    break;
  case ParsedAttr::AT_RequiresCapability:
    handleRequiresCapabilityAttr(S, D, AL);
    break;

  case ParsedAttr::AT_AssertCapability:
    handleAssertCapabilityAttr(S, D, AL);
    break;
  case ParsedAttr::AT_AcquireCapability:
    handleAcquireCapabilityAttr(S, D, AL);
    break;
  case ParsedAttr::AT_ReleaseCapability:
    handleReleaseCapabilityAttr(S, D, AL);
    break;
  case ParsedAttr::AT_TryAcquireCapability:
    handleTryAcquireCapabilityAttr(S, D, AL);
    break;

  // Consumed analysis attributes.
  case ParsedAttr::AT_Consumable:
    handleConsumableAttr(S, D, AL);
    break;
  case ParsedAttr::AT_CallableWhen:
    handleCallableWhenAttr(S, D, AL);
    break;
  case ParsedAttr::AT_ParamTypestate:
    handleParamTypestateAttr(S, D, AL);
    break;
  case ParsedAttr::AT_ReturnTypestate:
    handleReturnTypestateAttr(S, D, AL);
    break;
  case ParsedAttr::AT_SetTypestate:
    handleSetTypestateAttr(S, D, AL);
    break;
  case ParsedAttr::AT_TestTypestate:
    handleTestTypestateAttr(S, D, AL);
    break;

  // Type safety attributes.
  case ParsedAttr::AT_ArgumentWithTypeTag:
    handleArgumentWithTypeTagAttr(S, D, AL);
    break;
  case ParsedAttr::AT_TypeTagForDatatype:
    handleTypeTagForDatatypeAttr(S, D, AL);
    break;

  // Swift attributes.
  case ParsedAttr::AT_SwiftAsyncName:
    handleSwiftAsyncName(S, D, AL);
    break;
  case ParsedAttr::AT_SwiftAttr:
    handleSwiftAttrAttr(S, D, AL);
    break;
  case ParsedAttr::AT_SwiftBridge:
    handleSwiftBridge(S, D, AL);
    break;
  case ParsedAttr::AT_SwiftError:
    handleSwiftError(S, D, AL);
    break;
  case ParsedAttr::AT_SwiftName:
    handleSwiftName(S, D, AL);
    break;
  case ParsedAttr::AT_SwiftNewType:
    handleSwiftNewType(S, D, AL);
    break;
  case ParsedAttr::AT_SwiftAsync:
    handleSwiftAsyncAttr(S, D, AL);
    break;
  case ParsedAttr::AT_SwiftAsyncError:
    handleSwiftAsyncError(S, D, AL);
    break;

  // XRay attributes.
  case ParsedAttr::AT_XRayLogArgs:
    handleXRayLogArgsAttr(S, D, AL);
    break;

  case ParsedAttr::AT_PatchableFunctionEntry:
    handlePatchableFunctionEntryAttr(S, D, AL);
    break;

  case ParsedAttr::AT_AlwaysDestroy:
  case ParsedAttr::AT_NoDestroy:
    handleDestroyAttr(S, D, AL);
    break;

  case ParsedAttr::AT_Uninitialized:
    handleUninitializedAttr(S, D, AL);
    break;

  case ParsedAttr::AT_ObjCExternallyRetained:
    handleObjCExternallyRetainedAttr(S, D, AL);
    break;

  case ParsedAttr::AT_MIGServerRoutine:
    handleMIGServerRoutineAttr(S, D, AL);
    break;

  case ParsedAttr::AT_MSAllocator:
    handleMSAllocatorAttr(S, D, AL);
    break;

  case ParsedAttr::AT_ArmBuiltinAlias:
    handleArmBuiltinAliasAttr(S, D, AL);
    break;

  case ParsedAttr::AT_AcquireHandle:
    handleAcquireHandleAttr(S, D, AL);
    break;

  case ParsedAttr::AT_ReleaseHandle:
    handleHandleAttr<ReleaseHandleAttr>(S, D, AL);
    break;

  case ParsedAttr::AT_UseHandle:
    handleHandleAttr<UseHandleAttr>(S, D, AL);
    break;

  case ParsedAttr::AT_EnforceTCB:
    handleEnforceTCBAttr<EnforceTCBAttr, EnforceTCBLeafAttr>(S, D, AL);
    break;

  case ParsedAttr::AT_EnforceTCBLeaf:
    handleEnforceTCBAttr<EnforceTCBLeafAttr, EnforceTCBAttr>(S, D, AL);
    break;

  case ParsedAttr::AT_BuiltinAlias:
    handleBuiltinAliasAttr(S, D, AL);
    break;

  case ParsedAttr::AT_UsingIfExists:
    handleSimpleAttribute<UsingIfExistsAttr>(S, D, AL);
    break;
  }
}

/// ProcessDeclAttributeList - Apply all the decl attributes in the specified
/// attribute list to the specified decl, ignoring any type attributes.
void Sema::ProcessDeclAttributeList(Scope *S, Decl *D,
                                    const ParsedAttributesView &AttrList,
                                    bool IncludeCXX11Attributes) {
  if (AttrList.empty())
    return;

  for (const ParsedAttr &AL : AttrList)
    ProcessDeclAttribute(*this, S, D, AL, IncludeCXX11Attributes);

  // FIXME: We should be able to handle these cases in TableGen.
  // GCC accepts
  // static int a9 __attribute__((weakref));
  // but that looks really pointless. We reject it.
  if (D->hasAttr<WeakRefAttr>() && !D->hasAttr<AliasAttr>()) {
    Diag(AttrList.begin()->getLoc(), diag::err_attribute_weakref_without_alias)
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
    if (const auto *A = D->getAttr<ReqdWorkGroupSizeAttr>()) {
      // FIXME: This emits a different error message than
      // diag::err_attribute_wrong_decl_type + ExpectedKernelFunction.
      Diag(D->getLocation(), diag::err_opencl_kernel_attr) << A;
      D->setInvalidDecl();
    } else if (const auto *A = D->getAttr<WorkGroupSizeHintAttr>()) {
      Diag(D->getLocation(), diag::err_opencl_kernel_attr) << A;
      D->setInvalidDecl();
    } else if (const auto *A = D->getAttr<VecTypeHintAttr>()) {
      Diag(D->getLocation(), diag::err_opencl_kernel_attr) << A;
      D->setInvalidDecl();
    } else if (const auto *A = D->getAttr<OpenCLIntelReqdSubGroupSizeAttr>()) {
      Diag(D->getLocation(), diag::err_opencl_kernel_attr) << A;
      D->setInvalidDecl();
    } else if (!D->hasAttr<CUDAGlobalAttr>()) {
      if (const auto *A = D->getAttr<AMDGPUFlatWorkGroupSizeAttr>()) {
        Diag(D->getLocation(), diag::err_attribute_wrong_decl_type)
            << A << ExpectedKernelFunction;
        D->setInvalidDecl();
      } else if (const auto *A = D->getAttr<AMDGPUWavesPerEUAttr>()) {
        Diag(D->getLocation(), diag::err_attribute_wrong_decl_type)
            << A << ExpectedKernelFunction;
        D->setInvalidDecl();
      } else if (const auto *A = D->getAttr<AMDGPUNumSGPRAttr>()) {
        Diag(D->getLocation(), diag::err_attribute_wrong_decl_type)
            << A << ExpectedKernelFunction;
        D->setInvalidDecl();
      } else if (const auto *A = D->getAttr<AMDGPUNumVGPRAttr>()) {
        Diag(D->getLocation(), diag::err_attribute_wrong_decl_type)
            << A << ExpectedKernelFunction;
        D->setInvalidDecl();
      }
    }
  }

  // Do this check after processing D's attributes because the attribute
  // objc_method_family can change whether the given method is in the init
  // family, and it can be applied after objc_designated_initializer. This is a
  // bit of a hack, but we need it to be compatible with versions of clang that
  // processed the attribute list in the wrong order.
  if (D->hasAttr<ObjCDesignatedInitializerAttr>() &&
      cast<ObjCMethodDecl>(D)->getMethodFamily() != OMF_init) {
    Diag(D->getLocation(), diag::err_designated_init_attr_non_init);
    D->dropAttr<ObjCDesignatedInitializerAttr>();
  }
}

// Helper for delayed processing TransparentUnion or BPFPreserveAccessIndexAttr
// attribute.
void Sema::ProcessDeclAttributeDelayed(Decl *D,
                                       const ParsedAttributesView &AttrList) {
  for (const ParsedAttr &AL : AttrList)
    if (AL.getKind() == ParsedAttr::AT_TransparentUnion) {
      handleTransparentUnionAttr(*this, D, AL);
      break;
    }

  // For BPFPreserveAccessIndexAttr, we want to populate the attributes
  // to fields and inner records as well.
  if (D && D->hasAttr<BPFPreserveAccessIndexAttr>())
    handleBPFPreserveAIRecord(*this, cast<RecordDecl>(D));
}

// Annotation attributes are the only attributes allowed after an access
// specifier.
bool Sema::ProcessAccessDeclAttributeList(
    AccessSpecDecl *ASDecl, const ParsedAttributesView &AttrList) {
  for (const ParsedAttr &AL : AttrList) {
    if (AL.getKind() == ParsedAttr::AT_Annotate) {
      ProcessDeclAttribute(*this, nullptr, ASDecl, AL, AL.isCXX11Attribute());
    } else {
      Diag(AL.getLoc(), diag::err_only_annotate_after_access_spec);
      return true;
    }
  }
  return false;
}

/// checkUnusedDeclAttributes - Check a list of attributes to see if it
/// contains any decl attributes that we should warn about.
static void checkUnusedDeclAttributes(Sema &S, const ParsedAttributesView &A) {
  for (const ParsedAttr &AL : A) {
    // Only warn if the attribute is an unignored, non-type attribute.
    if (AL.isUsedAsTypeAttr() || AL.isInvalid())
      continue;
    if (AL.getKind() == ParsedAttr::IgnoredAttribute)
      continue;

    if (AL.getKind() == ParsedAttr::UnknownAttribute) {
      S.Diag(AL.getLoc(), diag::warn_unknown_attribute_ignored)
          << AL << AL.getRange();
    } else {
      S.Diag(AL.getLoc(), diag::warn_attribute_not_on_decl) << AL
                                                            << AL.getRange();
    }
  }
}

/// checkUnusedDeclAttributes - Given a declarator which is not being
/// used to build a declaration, complain about any decl attributes
/// which might be lying around on it.
void Sema::checkUnusedDeclAttributes(Declarator &D) {
  ::checkUnusedDeclAttributes(*this, D.getDeclSpec().getAttributes());
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
  if (auto *FD = dyn_cast<FunctionDecl>(ND)) {
    FunctionDecl *NewFD;
    // FIXME: Missing call to CheckFunctionDeclaration().
    // FIXME: Mangling?
    // FIXME: Is the qualifier info correct?
    // FIXME: Is the DeclContext correct?
    NewFD = FunctionDecl::Create(
        FD->getASTContext(), FD->getDeclContext(), Loc, Loc,
        DeclarationName(II), FD->getType(), FD->getTypeSourceInfo(), SC_None,
        getCurFPFeatures().isFPConstrained(), false /*isInlineSpecified*/,
        FD->hasPrototype(), ConstexprSpecKind::Unspecified,
        FD->getTrailingRequiresClause());
    NewD = NewFD;

    if (FD->getQualifier())
      NewFD->setQualifierInfo(FD->getQualifierLoc());

    // Fake up parameter variables; they are declared as if this were
    // a typedef.
    QualType FDTy = FD->getType();
    if (const auto *FT = FDTy->getAs<FunctionProtoType>()) {
      SmallVector<ParmVarDecl*, 16> Params;
      for (const auto &AI : FT->param_types()) {
        ParmVarDecl *Param = BuildParmVarDeclForTypedef(NewFD, Loc, AI);
        Param->setScopeInfo(0, Params.size());
        Params.push_back(Param);
      }
      NewFD->setParams(Params);
    }
  } else if (auto *VD = dyn_cast<VarDecl>(ND)) {
    NewD = VarDecl::Create(VD->getASTContext(), VD->getDeclContext(),
                           VD->getInnerLocStart(), VD->getLocation(), II,
                           VD->getType(), VD->getTypeSourceInfo(),
                           VD->getStorageClass());
    if (VD->getQualifier())
      cast<VarDecl>(NewD)->setQualifierInfo(VD->getQualifierLoc());
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
    NewD->addAttr(
        AliasAttr::CreateImplicit(Context, NDId->getName(), W.getLocation()));
    NewD->addAttr(WeakAttr::CreateImplicit(Context, W.getLocation(),
                                           AttributeCommonInfo::AS_Pragma));
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
    ND->addAttr(WeakAttr::CreateImplicit(Context, W.getLocation(),
                                         AttributeCommonInfo::AS_Pragma));
  }
}

void Sema::ProcessPragmaWeak(Scope *S, Decl *D) {
  // It's valid to "forward-declare" #pragma weak, in which case we
  // have to do this.
  LoadExternalWeakUndeclaredIdentifiers();
  if (!WeakUndeclaredIdentifiers.empty()) {
    NamedDecl *ND = nullptr;
    if (auto *VD = dyn_cast<VarDecl>(D))
      if (VD->isExternC())
        ND = VD;
    if (auto *FD = dyn_cast<FunctionDecl>(D))
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
  if (!PD.getDeclSpec().getAttributes().empty())
    ProcessDeclAttributeList(S, D, PD.getDeclSpec().getAttributes());

  // Walk the declarator structure, applying decl attributes that were in a type
  // position to the decl itself.  This handles cases like:
  //   int *__attr__(x)** D;
  // when X is a decl attribute.
  for (unsigned i = 0, e = PD.getNumTypeObjects(); i != e; ++i)
    ProcessDeclAttributeList(S, D, PD.getTypeObject(i).getAttrs(),
                             /*IncludeCXX11Attributes=*/false);

  // Finally, apply any attributes on the decl itself.
  ProcessDeclAttributeList(S, D, PD.getAttributes());

  // Apply additional attributes specified by '#pragma clang attribute'.
  AddPragmaAttributes(S, D);
}

/// Is the given declaration allowed to use a forbidden type?
/// If so, it'll still be annotated with an attribute that makes it
/// illegal to actually use.
static bool isForbiddenTypeAllowed(Sema &S, Decl *D,
                                   const DelayedDiagnostic &diag,
                                   UnavailableAttr::ImplicitReason &reason) {
  // Private ivars are always okay.  Unfortunately, people don't
  // always properly make their ivars private, even in system headers.
  // Plus we need to make fields okay, too.
  if (!isa<FieldDecl>(D) && !isa<ObjCPropertyDecl>(D) &&
      !isa<FunctionDecl>(D))
    return false;

  // Silently accept unsupported uses of __weak in both user and system
  // declarations when it's been disabled, for ease of integration with
  // -fno-objc-arc files.  We do have to take some care against attempts
  // to define such things;  for now, we've only done that for ivars
  // and properties.
  if ((isa<ObjCIvarDecl>(D) || isa<ObjCPropertyDecl>(D))) {
    if (diag.getForbiddenTypeDiagnostic() == diag::err_arc_weak_disabled ||
        diag.getForbiddenTypeDiagnostic() == diag::err_arc_weak_no_runtime) {
      reason = UnavailableAttr::IR_ForbiddenWeak;
      return true;
    }
  }

  // Allow all sorts of things in system headers.
  if (S.Context.getSourceManager().isInSystemHeader(D->getLocation())) {
    // Currently, all the failures dealt with this way are due to ARC
    // restrictions.
    reason = UnavailableAttr::IR_ARCForbiddenType;
    return true;
  }

  return false;
}

/// Handle a delayed forbidden-type diagnostic.
static void handleDelayedForbiddenType(Sema &S, DelayedDiagnostic &DD,
                                       Decl *D) {
  auto Reason = UnavailableAttr::IR_None;
  if (D && isForbiddenTypeAllowed(S, D, DD, Reason)) {
    assert(Reason && "didn't set reason?");
    D->addAttr(UnavailableAttr::CreateImplicit(S.Context, "", Reason, DD.Loc));
    return;
  }
  if (S.getLangOpts().ObjCAutoRefCount)
    if (const auto *FD = dyn_cast<FunctionDecl>(D)) {
      // FIXME: we may want to suppress diagnostics for all
      // kind of forbidden type messages on unavailable functions.
      if (FD->hasAttr<UnavailableAttr>() &&
          DD.getForbiddenTypeDiagnostic() ==
              diag::err_arc_array_param_no_ownership) {
        DD.Triggered = true;
        return;
      }
    }

  S.Diag(DD.Loc, DD.getForbiddenTypeDiagnostic())
      << DD.getForbiddenTypeOperand() << DD.getForbiddenTypeArgument();
  DD.Triggered = true;
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
    bool AnyAccessFailures = false;
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
          handleDelayedAvailabilityCheck(diag, decl);
        break;

      case DelayedDiagnostic::Access:
        // Only produce one access control diagnostic for a structured binding
        // declaration: we don't need to tell the user that all the fields are
        // inaccessible one at a time.
        if (AnyAccessFailures && isa<DecompositionDecl>(decl))
          continue;
        HandleDelayedAccessCheck(diag, decl);
        if (diag.Triggered)
          AnyAccessFailures = true;
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
