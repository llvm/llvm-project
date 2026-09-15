//===--- CIRGenMicrosoftCXXABI.cpp - Emit CIR Code for MS C++ ABI --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides C++ code generation targeting the Microsoft C++ ABI.
//
//===----------------------------------------------------------------------===//

#include "CIRGenCXXABI.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Mangle.h"

using namespace clang;
using namespace clang::CIRGen;

namespace {

class CIRGenMicrosoftCXXABI : public CIRGenCXXABI {
public:
  CIRGenMicrosoftCXXABI(CIRGenModule &cgm) : CIRGenCXXABI(cgm) {}

  AddedStructorArgCounts
  buildStructorSignature(GlobalDecl gd,
                         SmallVectorImpl<CanQualType> &argTys) override {
    cgm.errorNYI(gd.getDecl()->getSourceRange(),
                 "buildStructorSignature: MSVC ABI");
    return AddedStructorArgCounts{};
  }

  void addImplicitStructorParams(CIRGenFunction &cgf, QualType &resTy,
                                 FunctionArgList &params) override {
    cgf.cgm.errorNYI(cgf.curGD.getDecl()->getSourceRange(),
                     "addImplicitStructorParams: MSVC ABI");
  }

  void emitInstanceFunctionProlog(SourceLocation loc,
                                  CIRGenFunction &cgf) override {
    cgf.cgm.errorNYI(loc, "emitInstanceFunctionProlog: MSVC ABI");
  }

  AddedStructorArgs getImplicitConstructorArgs(CIRGenFunction &cgf,
                                               const CXXConstructorDecl *d,
                                               CXXCtorType type,
                                               bool forVirtualBase,
                                               bool delegating) override {
    cgf.cgm.errorNYI(d->getSourceRange(),
                     "getImplicitConstructorArgs: MSVC ABI");
    return AddedStructorArgs{};
  }

  mlir::Value getCXXDestructorImplicitParam(CIRGenFunction &cgf,
                                            const CXXDestructorDecl *dd,
                                            CXXDtorType type,
                                            bool forVirtualBase,
                                            bool delegating) override {
    cgf.cgm.errorNYI(dd->getSourceRange(),
                     "getCXXDestructorImplicitParam: MSVC ABI");
    return nullptr;
  }

  void emitCXXConstructors(const CXXConstructorDecl *d) override {
    cgm.errorNYI(d->getSourceRange(), "emitCXXConstructors: MSVC ABI");
  }

  void emitCXXDestructors(const CXXDestructorDecl *d) override {
    cgm.errorNYI(d->getSourceRange(), "emitCXXDestructors: MSVC ABI");
  }

  void emitCXXStructor(GlobalDecl gd) override {
    cgm.errorNYI(gd.getDecl()->getSourceRange(), "emitCXXStructor: MSVC ABI");
  }

  void emitDestructorCall(CIRGenFunction &cgf, const CXXDestructorDecl *dd,
                          CXXDtorType type, bool forVirtualBase,
                          bool delegating, Address thisAddr,
                          QualType thisTy) override {
    cgf.cgm.errorNYI(dd->getSourceRange(), "emitDestructorCall: MSVC ABI");
  }

  mlir::Value emitVirtualDestructorCall(CIRGenFunction &cgf,
                                        const CXXDestructorDecl *dtor,
                                        CXXDtorType dtorType, Address thisAddr,
                                        DeleteOrMemberCallExpr e) override {
    cgf.cgm.errorNYI(dtor->getSourceRange(),
                     "emitVirtualDestructorCall: MSVC ABI");
    return nullptr;
  }

  void emitVirtualObjectDelete(CIRGenFunction &cgf, const CXXDeleteExpr *de,
                               Address ptr, QualType elementType,
                               const CXXDestructorDecl *dtor) override {
    cgf.cgm.errorNYI(de->getSourceRange(), "emitVirtualObjectDelete: MSVC ABI");
  }

  size_t getSrcArgforCopyCtor(const CXXConstructorDecl *cd,
                              FunctionArgList &args) const override {
    cgm.errorNYI(cd->getSourceRange(), "getSrcArgforCopyCtor: MSVC ABI");
    assert(args.size() >= 2 &&
           "expected the arglist to have at least two args!");
    // The 'most_derived' parameter goes second if the ctor is variadic and
    // has v-bases.
    if (cd->getParent()->getNumVBases() > 0 &&
        cd->getType()->castAs<FunctionProtoType>()->isVariadic())
      return 2;
    return 1;
  }

  const CXXRecordDecl *
  getThisArgumentTypeForMethod(const CXXMethodDecl *md) override {
    cgm.errorNYI(md->getSourceRange(),
                 "getThisArgumentTypeForMethod: MSVC ABI");
    return md->getParent();
  }

  Address adjustThisArgumentForVirtualFunctionCall(CIRGenFunction &cgf,
                                                   GlobalDecl gd,
                                                   Address thisAddr,
                                                   bool virtualCall) override {
    cgf.cgm.errorNYI(gd.getDecl()->getSourceRange(),
                     "adjustThisArgumentForVirtualFunctionCall: MSVC ABI");
    return thisAddr;
  }

  bool isVirtualOffsetNeededForVTableField(CIRGenFunction &cgf,
                                           CIRGenFunction::VPtr vptr) override {
    cgf.cgm.errorNYI("isVirtualOffsetNeededForVTableField: MSVC ABI");
    return false;
  }

  cir::GlobalOp getAddrOfVTable(const CXXRecordDecl *rd,
                                CharUnits vptrOffset) override {
    cgm.errorNYI(rd->getSourceRange(), "getAddrOfVTable: MSVC ABI");
    return nullptr;
  }

  mlir::Value getVTableAddressPoint(BaseSubobject base,
                                    const CXXRecordDecl *vtableClass) override {
    cgm.errorNYI(vtableClass->getSourceRange(),
                 "getVTableAddressPoint: MSVC ABI");
    return nullptr;
  }

  mlir::Value getVTableAddressPointInStructor(
      CIRGenFunction &cgf, const CXXRecordDecl *vtableClass, BaseSubobject base,
      const CXXRecordDecl *nearestVBase) override {
    cgf.cgm.errorNYI(vtableClass->getSourceRange(),
                     "getVTableAddressPointInStructor: MSVC ABI");
    return nullptr;
  }

  CIRGenCallee getVirtualFunctionPointer(CIRGenFunction &cgf, GlobalDecl gd,
                                         Address thisAddr, mlir::Type ty,
                                         SourceLocation loc) override {
    cgf.cgm.errorNYI(loc, "getVirtualFunctionPointer: MSVC ABI");
    return CIRGenCallee();
  }

  void emitVTableDefinitions(CIRGenVTables &cgvt,
                             const CXXRecordDecl *rd) override {
    cgm.errorNYI(rd->getSourceRange(), "emitVTableDefinitions: MSVC ABI");
  }

  void emitVirtualInheritanceTables(const CXXRecordDecl *rd) override {
    cgm.errorNYI(rd->getSourceRange(),
                 "emitVirtualInheritanceTables: MSVC ABI");
  }

  void
  initializeHiddenVirtualInheritanceMembers(CIRGenFunction &cgf,
                                            const CXXRecordDecl *rd) override {
    cgf.cgm.errorNYI(rd->getSourceRange(),
                     "initializeHiddenVirtualInheritanceMembers: MSVC ABI");
  }

  mlir::Value
  getVirtualBaseClassOffset(mlir::Location loc, CIRGenFunction &cgf,
                            Address thisAddr, const CXXRecordDecl *classDecl,
                            const CXXRecordDecl *baseClassDecl) override {
    cgf.cgm.errorNYI(loc, "getVirtualBaseClassOffset: MSVC ABI");
    return nullptr;
  }

  cir::MethodAttr buildVirtualMethodAttr(cir::MethodType methodTy,
                                         const CXXMethodDecl *md) override {
    cgm.errorNYI(md->getSourceRange(), "buildVirtualMethodAttr: MSVC ABI");
    return cir::MethodAttr();
  }

  mlir::Value performThisAdjustment(CIRGenFunction &cgf, Address thisAddr,
                                    const CXXRecordDecl *unadjustedClass,
                                    const ThunkInfo &ti) override {
    cgf.cgm.errorNYI("performThisAdjustment: MSVC ABI");
    return thisAddr.emitRawPointer();
  }

  mlir::Value performReturnAdjustment(CIRGenFunction &cgf, Address ret,
                                      const CXXRecordDecl *unadjustedClass,
                                      const ReturnAdjustment &ra) override {
    cgf.cgm.errorNYI("performReturnAdjustment: MSVC ABI");
    return ret.emitRawPointer();
  }

  bool canSpeculativelyEmitVTable(const CXXRecordDecl *rd) const override {
    return false;
  }

  bool doStructorsInitializeVPtrs(const CXXRecordDecl *vtableClass) override {
    cgm.errorNYI(vtableClass->getSourceRange(),
                 "doStructorsInitializeVPtrs: MSVC ABI");
    return false;
  }

  bool exportThunk() override { return false; }

  bool useThunkForDtorVariant(const CXXDestructorDecl *dtor,
                              CXXDtorType dt) const override {
    cgm.errorNYI(dtor->getSourceRange(), "useThunkForDtorVariant: MSVC ABI");
    return false;
  }

  void setThunkLinkage(cir::FuncOp thunk, bool forVTable, GlobalDecl gd,
                       bool returnAdjustment) override {
    cgm.errorNYI(gd.getDecl()->getSourceRange(),
                 "setThunkLinkage: MSVC ABI");
  }

  StringRef getPureVirtualCallName() override { return "_purecall"; }
  StringRef getDeletedVirtualCallName() override { return "_purecall"; }

  bool isZeroInitializable(const MemberPointerType *mpt) override {
    cgm.errorNYI("isZeroInitializable: MSVC ABI");
    return true;
  }

  bool requiresArrayCookie(const CXXNewExpr *e) override {
    cgm.errorNYI(e->getSourceRange(), "requiresArrayCookie: MSVC ABI");
    return false;
  }

  CharUnits getArrayCookieSizeImpl(QualType elementType) override {
    cgm.errorNYI("getArrayCookieSizeImpl: MSVC ABI");
    return CharUnits::Zero();
  }

  Address initializeArrayCookie(CIRGenFunction &cgf, Address newPtr,
                                mlir::Value numElements, const CXXNewExpr *e,
                                QualType elementType) override {
    cgf.cgm.errorNYI(e->getSourceRange(), "initializeArrayCookie: MSVC ABI");
    return newPtr;
  }

  bool shouldTypeidBeNullChecked(QualType srcTy) override {
    cgm.errorNYI("shouldTypeidBeNullChecked: MSVC ABI");
    return false;
  }

  mlir::Value emitTypeid(CIRGenFunction &cgf, QualType srcTy, Address thisPtr,
                         mlir::Type typeInfoPtrTy) override {
    cgf.cgm.errorNYI(cgf.getLoc(srcTy->getAsCXXRecordDecl()->getLocation()),
                     "emitTypeid: MSVC ABI");
    return cgf.getBuilder().getNullPtr(
        typeInfoPtrTy, cgf.getLoc(srcTy->getAsCXXRecordDecl()->getLocation()));
  }

  void emitBadTypeidCall(CIRGenFunction &cgf, mlir::Location loc) override {
    cgf.cgm.errorNYI(loc, "emitBadTypeidCall: MSVC ABI");
  }

  void emitBadCastCall(CIRGenFunction &cgf, mlir::Location loc) override {
    cgm.errorNYI(loc, "emitBadCastCall: MSVC ABI");
  }

  mlir::Value emitDynamicCast(CIRGenFunction &cgf, mlir::Location loc,
                              QualType srcRecordTy, QualType destRecordTy,
                              cir::PointerType destCIRTy, bool isRefCast,
                              Address src) override {
    cgf.cgm.errorNYI(loc, "emitDynamicCast: MSVC ABI");
    return cgf.getBuilder().getNullPtr(destCIRTy, loc);
  }

  mlir::Attribute getAddrOfRTTIDescriptor(mlir::Location loc,
                                          QualType ty) override {
    cgm.errorNYI(loc, "getAddrOfRTTIDescriptor: MSVC ABI");
    return nullptr;
  }

  CatchTypeInfo getCatchAllTypeInfo() override {
    cgm.errorNYI("getCatchAllTypeInfo: MSVC ABI");
    return CatchTypeInfo{nullptr, 0};
  }

  CatchTypeInfo
  getAddrOfCXXCatchHandlerType(mlir::Location loc, QualType ty,
                               QualType catchHandlerType) override {
    cgm.errorNYI(loc, "getAddrOfCXXCatchHandlerType: MSVC ABI");
    return CatchTypeInfo{nullptr, 0};
  }

  void emitRethrow(CIRGenFunction &cgf, bool isNoReturn) override {
    cgm.errorNYI("emitRethrow: MSVC ABI");
  }

  void emitThrow(CIRGenFunction &cgf, const CXXThrowExpr *e) override {
    cgf.cgm.errorNYI(e->getSourceRange(), "emitThrow: MSVC ABI");
  }

  void registerGlobalDtor(const VarDecl *vd, cir::FuncOp dtor,
                          mlir::Value addr) override {
    cgm.errorNYI(vd->getSourceRange(), "registerGlobalDtor: MSVC ABI");
  }
};

} // namespace

CIRGenCXXABI *clang::CIRGen::CreateCIRGenMicrosoftCXXABI(CIRGenModule &cgm) {
  return new CIRGenMicrosoftCXXABI(cgm);
}
