//===----- CGCXXABI.cpp - Interface to C++ ABIs -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for C++ code generation. Concrete subclasses
// of this implement code generation for specific C++ ABIs.
//
//===----------------------------------------------------------------------===//

#include "CGCXXABI.h"

using namespace clang;
using namespace CodeGen;

CGCXXABI::~CGCXXABI() { }

static void ErrorUnsupportedABI(CodeGenFunction &CGF,
                                StringRef S) {
  DiagnosticsEngine &Diags = CGF.CGM.getDiags();
  unsigned DiagID = Diags.getCustomDiagID(DiagnosticsEngine::Error,
                                          "cannot yet compile %0 in this ABI");
  Diags.Report(CGF.getContext().getFullLoc(CGF.CurCodeDecl->getLocation()),
               DiagID)
    << S;
}

static llvm::Constant *GetBogusMemberPointer(CodeGenModule &CGM,
                                             QualType T) {
  return llvm::Constant::getNullValue(CGM.getTypes().ConvertType(T));
}

llvm::Type *
CGCXXABI::ConvertMemberPointerType(const MemberPointerType *MPT) {
  return CGM.getTypes().ConvertType(CGM.getContext().getPointerDiffType());
}

llvm::Value *CGCXXABI::EmitLoadOfMemberFunctionPointer(CodeGenFunction &CGF,
                                                       llvm::Value *&This,
                                                       llvm::Value *MemPtr,
                                                 const MemberPointerType *MPT) {
  ErrorUnsupportedABI(CGF, "calls through member pointers");

  const FunctionProtoType *FPT = 
    MPT->getPointeeType()->getAs<FunctionProtoType>();
  const CXXRecordDecl *RD = 
    cast<CXXRecordDecl>(MPT->getClass()->getAs<RecordType>()->getDecl());
  llvm::FunctionType *FTy = CGM.getTypes().GetFunctionType(
                              CGM.getTypes().arrangeCXXMethodType(RD, FPT));
  return llvm::Constant::getNullValue(FTy->getPointerTo());
}

llvm::Value *CGCXXABI::EmitMemberDataPointerAddress(CodeGenFunction &CGF,
                                                    llvm::Value *Base,
                                                    llvm::Value *MemPtr,
                                              const MemberPointerType *MPT) {
  ErrorUnsupportedABI(CGF, "loads of member pointers");
  llvm::Type *Ty = CGF.ConvertType(MPT->getPointeeType())->getPointerTo();
  return llvm::Constant::getNullValue(Ty);
}

llvm::Value *CGCXXABI::EmitMemberPointerConversion(CodeGenFunction &CGF,
                                                   const CastExpr *E,
                                                   llvm::Value *Src) {
  ErrorUnsupportedABI(CGF, "member function pointer conversions");
  return GetBogusMemberPointer(CGM, E->getType());
}

llvm::Constant *CGCXXABI::EmitMemberPointerConversion(const CastExpr *E,
                                                      llvm::Constant *Src) {
  return GetBogusMemberPointer(CGM, E->getType());
}

llvm::Value *
CGCXXABI::EmitMemberPointerComparison(CodeGenFunction &CGF,
                                      llvm::Value *L,
                                      llvm::Value *R,
                                      const MemberPointerType *MPT,
                                      bool Inequality) {
  ErrorUnsupportedABI(CGF, "member function pointer comparison");
  return CGF.Builder.getFalse();
}

llvm::Value *
CGCXXABI::EmitMemberPointerIsNotNull(CodeGenFunction &CGF,
                                     llvm::Value *MemPtr,
                                     const MemberPointerType *MPT) {
  ErrorUnsupportedABI(CGF, "member function pointer null testing");
  return CGF.Builder.getFalse();
}

llvm::Constant *
CGCXXABI::EmitNullMemberPointer(const MemberPointerType *MPT) {
  return GetBogusMemberPointer(CGM, QualType(MPT, 0));
}

llvm::Constant *CGCXXABI::EmitMemberPointer(const CXXMethodDecl *MD) {
  return GetBogusMemberPointer(CGM,
                         CGM.getContext().getMemberPointerType(MD->getType(),
                                         MD->getParent()->getTypeForDecl()));
}

llvm::Constant *CGCXXABI::EmitMemberDataPointer(const MemberPointerType *MPT,
                                                CharUnits offset) {
  return GetBogusMemberPointer(CGM, QualType(MPT, 0));
}

llvm::Constant *CGCXXABI::EmitMemberPointer(const APValue &MP, QualType MPT) {
  return GetBogusMemberPointer(CGM, MPT);
}

bool CGCXXABI::isZeroInitializable(const MemberPointerType *MPT) {
  // Fake answer.
  return true;
}

void CGCXXABI::BuildThisParam(CodeGenFunction &CGF, FunctionArgList &params) {
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(CGF.CurGD.getDecl());

  // FIXME: I'm not entirely sure I like using a fake decl just for code
  // generation. Maybe we can come up with a better way?
  ImplicitParamDecl *ThisDecl
    = ImplicitParamDecl::Create(CGM.getContext(), 0, MD->getLocation(),
                                &CGM.getContext().Idents.get("this"),
                                MD->getThisType(CGM.getContext()));
  params.push_back(ThisDecl);
  getThisDecl(CGF) = ThisDecl;
}

void CGCXXABI::EmitThisParam(CodeGenFunction &CGF) {
  /// Initialize the 'this' slot.
  assert(getThisDecl(CGF) && "no 'this' variable for function");
  getThisValue(CGF)
    = CGF.Builder.CreateLoad(CGF.GetAddrOfLocalVar(getThisDecl(CGF)),
                             "this");
}

void CGCXXABI::EmitReturnFromThunk(CodeGenFunction &CGF,
                                   RValue RV, QualType ResultType) {
  CGF.EmitReturnOfRValue(RV, ResultType);
}

CharUnits CGCXXABI::GetArrayCookieSize(const CXXNewExpr *expr) {
  if (!requiresArrayCookie(expr))
    return CharUnits::Zero();
  return getArrayCookieSizeImpl(expr->getAllocatedType());
}

CharUnits CGCXXABI::getArrayCookieSizeImpl(QualType elementType) {
  // BOGUS
  return CharUnits::Zero();
}

llvm::Value *CGCXXABI::InitializeArrayCookie(CodeGenFunction &CGF,
                                             llvm::Value *NewPtr,
                                             llvm::Value *NumElements,
                                             const CXXNewExpr *expr,
                                             QualType ElementType) {
  // Should never be called.
  ErrorUnsupportedABI(CGF, "array cookie initialization");
  return 0;
}

bool CGCXXABI::requiresArrayCookie(const CXXDeleteExpr *expr,
                                   QualType elementType) {
  // If the class's usual deallocation function takes two arguments,
  // it needs a cookie.
  if (expr->doesUsualArrayDeleteWantSize())
    return true;

  return elementType.isDestructedType();
}

bool CGCXXABI::requiresArrayCookie(const CXXNewExpr *expr) {
  // If the class's usual deallocation function takes two arguments,
  // it needs a cookie.
  if (expr->doesUsualArrayDeleteWantSize())
    return true;

  return expr->getAllocatedType().isDestructedType();
}

void CGCXXABI::ReadArrayCookie(CodeGenFunction &CGF, llvm::Value *ptr,
                               const CXXDeleteExpr *expr, QualType eltTy,
                               llvm::Value *&numElements,
                               llvm::Value *&allocPtr, CharUnits &cookieSize) {
  // Derive a char* in the same address space as the pointer.
  unsigned AS = ptr->getType()->getPointerAddressSpace();
  llvm::Type *charPtrTy = CGF.Int8Ty->getPointerTo(AS);
  ptr = CGF.Builder.CreateBitCast(ptr, charPtrTy);

  // If we don't need an array cookie, bail out early.
  if (!requiresArrayCookie(expr, eltTy)) {
    allocPtr = ptr;
    numElements = 0;
    cookieSize = CharUnits::Zero();
    return;
  }

  cookieSize = getArrayCookieSizeImpl(eltTy);
  allocPtr = CGF.Builder.CreateConstInBoundsGEP1_64(ptr,
                                                    -cookieSize.getQuantity());
  numElements = readArrayCookieImpl(CGF, allocPtr, cookieSize);
}

llvm::Value *CGCXXABI::readArrayCookieImpl(CodeGenFunction &CGF,
                                           llvm::Value *ptr,
                                           CharUnits cookieSize) {
  ErrorUnsupportedABI(CGF, "reading a new[] cookie");
  return llvm::ConstantInt::get(CGF.SizeTy, 0);
}

void CGCXXABI::EmitGuardedInit(CodeGenFunction &CGF,
                               const VarDecl &D,
                               llvm::GlobalVariable *GV,
                               bool PerformInit) {
  ErrorUnsupportedABI(CGF, "static local variable initialization");
}

void CGCXXABI::registerGlobalDtor(CodeGenFunction &CGF,
                                  llvm::Constant *dtor,
                                  llvm::Constant *addr) {
  // The default behavior is to use atexit.
  CGF.registerGlobalDtorWithAtExit(dtor, addr);
}

/// Returns the adjustment, in bytes, required for the given
/// member-pointer operation.  Returns null if no adjustment is
/// required.
llvm::Constant *CGCXXABI::getMemberPointerAdjustment(const CastExpr *E) {
  assert(E->getCastKind() == CK_DerivedToBaseMemberPointer ||
         E->getCastKind() == CK_BaseToDerivedMemberPointer);

  QualType derivedType;
  if (E->getCastKind() == CK_DerivedToBaseMemberPointer)
    derivedType = E->getSubExpr()->getType();
  else
    derivedType = E->getType();

  const CXXRecordDecl *derivedClass =
    derivedType->castAs<MemberPointerType>()->getClass()->getAsCXXRecordDecl();

  return CGM.GetNonVirtualBaseClassOffset(derivedClass,
                                          E->path_begin(),
                                          E->path_end());
}
