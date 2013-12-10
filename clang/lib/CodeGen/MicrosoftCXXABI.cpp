//===--- MicrosoftCXXABI.cpp - Emit LLVM Code from ASTs for a Module ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides C++ code generation targeting the Microsoft Visual C++ ABI.
// The class in this file generates structures that follow the Microsoft
// Visual C++ ABI, which is actually not very well documented at all outside
// of Microsoft.
//
//===----------------------------------------------------------------------===//

#include "CGCXXABI.h"
#include "CodeGenModule.h"
#include "CGVTables.h"
#include "MicrosoftVBTables.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/VTableBuilder.h"
#include "llvm/ADT/StringSet.h"

using namespace clang;
using namespace CodeGen;

namespace {

class MicrosoftCXXABI : public CGCXXABI {
public:
  MicrosoftCXXABI(CodeGenModule &CGM) : CGCXXABI(CGM) {}

  bool HasThisReturn(GlobalDecl GD) const;

  bool isReturnTypeIndirect(const CXXRecordDecl *RD) const {
    // Structures that are not C++03 PODs are always indirect.
    return !RD->isPOD();
  }

  RecordArgABI getRecordArgABI(const CXXRecordDecl *RD) const {
    if (RD->hasNonTrivialCopyConstructor() || RD->hasNonTrivialDestructor())
      return RAA_DirectInMemory;
    return RAA_Default;
  }

  StringRef GetPureVirtualCallName() { return "_purecall"; }
  // No known support for deleted functions in MSVC yet, so this choice is
  // arbitrary.
  StringRef GetDeletedVirtualCallName() { return "_purecall"; }

  bool isInlineInitializedStaticDataMemberLinkOnce() { return true; }

  llvm::Value *adjustToCompleteObject(CodeGenFunction &CGF,
                                      llvm::Value *ptr,
                                      QualType type);

  llvm::Value *GetVirtualBaseClassOffset(CodeGenFunction &CGF,
                                         llvm::Value *This,
                                         const CXXRecordDecl *ClassDecl,
                                         const CXXRecordDecl *BaseClassDecl);

  void BuildConstructorSignature(const CXXConstructorDecl *Ctor,
                                 CXXCtorType Type,
                                 CanQualType &ResTy,
                                 SmallVectorImpl<CanQualType> &ArgTys);

  llvm::BasicBlock *EmitCtorCompleteObjectHandler(CodeGenFunction &CGF,
                                                  const CXXRecordDecl *RD);

  void initializeHiddenVirtualInheritanceMembers(CodeGenFunction &CGF,
                                                 const CXXRecordDecl *RD);

  void EmitCXXConstructors(const CXXConstructorDecl *D);

  // Background on MSVC destructors
  // ==============================
  //
  // Both Itanium and MSVC ABIs have destructor variants.  The variant names
  // roughly correspond in the following way:
  //   Itanium       Microsoft
  //   Base       -> no name, just ~Class
  //   Complete   -> vbase destructor
  //   Deleting   -> scalar deleting destructor
  //                 vector deleting destructor
  //
  // The base and complete destructors are the same as in Itanium, although the
  // complete destructor does not accept a VTT parameter when there are virtual
  // bases.  A separate mechanism involving vtordisps is used to ensure that
  // virtual methods of destroyed subobjects are not called.
  //
  // The deleting destructors accept an i32 bitfield as a second parameter.  Bit
  // 1 indicates if the memory should be deleted.  Bit 2 indicates if the this
  // pointer points to an array.  The scalar deleting destructor assumes that
  // bit 2 is zero, and therefore does not contain a loop.
  //
  // For virtual destructors, only one entry is reserved in the vftable, and it
  // always points to the vector deleting destructor.  The vector deleting
  // destructor is the most general, so it can be used to destroy objects in
  // place, delete single heap objects, or delete arrays.
  //
  // A TU defining a non-inline destructor is only guaranteed to emit a base
  // destructor, and all of the other variants are emitted on an as-needed basis
  // in COMDATs.  Because a non-base destructor can be emitted in a TU that
  // lacks a definition for the destructor, non-base destructors must always
  // delegate to or alias the base destructor.

  void BuildDestructorSignature(const CXXDestructorDecl *Dtor,
                                CXXDtorType Type,
                                CanQualType &ResTy,
                                SmallVectorImpl<CanQualType> &ArgTys);

  /// Non-base dtors should be emitted as delegating thunks in this ABI.
  bool useThunkForDtorVariant(const CXXDestructorDecl *Dtor,
                              CXXDtorType DT) const {
    return DT != Dtor_Base;
  }

  void EmitCXXDestructors(const CXXDestructorDecl *D);

  const CXXRecordDecl *getThisArgumentTypeForMethod(const CXXMethodDecl *MD) {
    MD = MD->getCanonicalDecl();
    if (MD->isVirtual() && !isa<CXXDestructorDecl>(MD)) {
      MicrosoftVTableContext::MethodVFTableLocation ML =
          CGM.getMicrosoftVTableContext().getMethodVFTableLocation(MD);
      // The vbases might be ordered differently in the final overrider object
      // and the complete object, so the "this" argument may sometimes point to
      // memory that has no particular type (e.g. past the complete object).
      // In this case, we just use a generic pointer type.
      // FIXME: might want to have a more precise type in the non-virtual
      // multiple inheritance case.
      if (ML.VBase || !ML.VFPtrOffset.isZero())
        return 0;
    }
    return MD->getParent();
  }

  llvm::Value *adjustThisArgumentForVirtualCall(CodeGenFunction &CGF,
                                                GlobalDecl GD,
                                                llvm::Value *This);

  void BuildInstanceFunctionParams(CodeGenFunction &CGF,
                                   QualType &ResTy,
                                   FunctionArgList &Params);

  llvm::Value *adjustThisParameterInVirtualFunctionPrologue(
      CodeGenFunction &CGF, GlobalDecl GD, llvm::Value *This);

  void EmitInstanceFunctionProlog(CodeGenFunction &CGF);

  void EmitConstructorCall(CodeGenFunction &CGF,
                           const CXXConstructorDecl *D, CXXCtorType Type,
                           bool ForVirtualBase, bool Delegating,
                           llvm::Value *This,
                           CallExpr::const_arg_iterator ArgBeg,
                           CallExpr::const_arg_iterator ArgEnd);

  void emitVTableDefinitions(CodeGenVTables &CGVT, const CXXRecordDecl *RD);

  llvm::Value *getVTableAddressPointInStructor(
      CodeGenFunction &CGF, const CXXRecordDecl *VTableClass,
      BaseSubobject Base, const CXXRecordDecl *NearestVBase,
      bool &NeedsVirtualOffset);

  llvm::Constant *
  getVTableAddressPointForConstExpr(BaseSubobject Base,
                                    const CXXRecordDecl *VTableClass);

  llvm::GlobalVariable *getAddrOfVTable(const CXXRecordDecl *RD,
                                        CharUnits VPtrOffset);

  llvm::Value *getVirtualFunctionPointer(CodeGenFunction &CGF, GlobalDecl GD,
                                         llvm::Value *This, llvm::Type *Ty);

  void EmitVirtualDestructorCall(CodeGenFunction &CGF,
                                 const CXXDestructorDecl *Dtor,
                                 CXXDtorType DtorType, SourceLocation CallLoc,
                                 llvm::Value *This);

  void adjustCallArgsForDestructorThunk(CodeGenFunction &CGF, GlobalDecl GD,
                                        CallArgList &CallArgs) {
    assert(GD.getDtorType() == Dtor_Deleting &&
           "Only deleting destructor thunks are available in this ABI");
    CallArgs.add(RValue::get(getStructorImplicitParamValue(CGF)),
                             CGM.getContext().IntTy);
  }

  void emitVirtualInheritanceTables(const CXXRecordDecl *RD);

  void setThunkLinkage(llvm::Function *Thunk, bool ForVTable) {
    Thunk->setLinkage(llvm::GlobalValue::WeakAnyLinkage);
  }

  llvm::Value *performThisAdjustment(CodeGenFunction &CGF, llvm::Value *This,
                                     const ThisAdjustment &TA);

  llvm::Value *performReturnAdjustment(CodeGenFunction &CGF, llvm::Value *Ret,
                                       const ReturnAdjustment &RA);

  void EmitGuardedInit(CodeGenFunction &CGF, const VarDecl &D,
                       llvm::GlobalVariable *DeclPtr,
                       bool PerformInit);

  // ==== Notes on array cookies =========
  //
  // MSVC seems to only use cookies when the class has a destructor; a
  // two-argument usual array deallocation function isn't sufficient.
  //
  // For example, this code prints "100" and "1":
  //   struct A {
  //     char x;
  //     void *operator new[](size_t sz) {
  //       printf("%u\n", sz);
  //       return malloc(sz);
  //     }
  //     void operator delete[](void *p, size_t sz) {
  //       printf("%u\n", sz);
  //       free(p);
  //     }
  //   };
  //   int main() {
  //     A *p = new A[100];
  //     delete[] p;
  //   }
  // Whereas it prints "104" and "104" if you give A a destructor.

  bool requiresArrayCookie(const CXXDeleteExpr *expr, QualType elementType);
  bool requiresArrayCookie(const CXXNewExpr *expr);
  CharUnits getArrayCookieSizeImpl(QualType type);
  llvm::Value *InitializeArrayCookie(CodeGenFunction &CGF,
                                     llvm::Value *NewPtr,
                                     llvm::Value *NumElements,
                                     const CXXNewExpr *expr,
                                     QualType ElementType);
  llvm::Value *readArrayCookieImpl(CodeGenFunction &CGF,
                                   llvm::Value *allocPtr,
                                   CharUnits cookieSize);

private:
  MicrosoftMangleContext &getMangleContext() {
    return cast<MicrosoftMangleContext>(CodeGen::CGCXXABI::getMangleContext());
  }

  llvm::Constant *getZeroInt() {
    return llvm::ConstantInt::get(CGM.IntTy, 0);
  }

  llvm::Constant *getAllOnesInt() {
    return  llvm::Constant::getAllOnesValue(CGM.IntTy);
  }

  llvm::Constant *getConstantOrZeroInt(llvm::Constant *C) {
    return C ? C : getZeroInt();
  }

  llvm::Value *getValueOrZeroInt(llvm::Value *C) {
    return C ? C : getZeroInt();
  }

  void
  GetNullMemberPointerFields(const MemberPointerType *MPT,
                             llvm::SmallVectorImpl<llvm::Constant *> &fields);

  /// \brief Finds the offset from the base of RD to the vbptr it uses, even if
  /// it is reusing a vbptr from a non-virtual base.  RD must have morally
  /// virtual bases.
  CharUnits GetVBPtrOffsetFromBases(const CXXRecordDecl *RD);

  /// \brief Shared code for virtual base adjustment.  Returns the offset from
  /// the vbptr to the virtual base.  Optionally returns the address of the
  /// vbptr itself.
  llvm::Value *GetVBaseOffsetFromVBPtr(CodeGenFunction &CGF,
                                       llvm::Value *Base,
                                       llvm::Value *VBPtrOffset,
                                       llvm::Value *VBTableOffset,
                                       llvm::Value **VBPtr = 0);

  llvm::Value *GetVBaseOffsetFromVBPtr(CodeGenFunction &CGF,
                                       llvm::Value *Base,
                                       int32_t VBPtrOffset,
                                       int32_t VBTableOffset,
                                       llvm::Value **VBPtr = 0) {
    llvm::Value *VBPOffset = llvm::ConstantInt::get(CGM.IntTy, VBPtrOffset),
                *VBTOffset = llvm::ConstantInt::get(CGM.IntTy, VBTableOffset);
    return GetVBaseOffsetFromVBPtr(CGF, Base, VBPOffset, VBTOffset, VBPtr);
  }

  /// \brief Performs a full virtual base adjustment.  Used to dereference
  /// pointers to members of virtual bases.
  llvm::Value *AdjustVirtualBase(CodeGenFunction &CGF, const CXXRecordDecl *RD,
                                 llvm::Value *Base,
                                 llvm::Value *VirtualBaseAdjustmentOffset,
                                 llvm::Value *VBPtrOffset /* optional */);

  /// \brief Emits a full member pointer with the fields common to data and
  /// function member pointers.
  llvm::Constant *EmitFullMemberPointer(llvm::Constant *FirstField,
                                        bool IsMemberFunction,
                                        const CXXRecordDecl *RD,
                                        CharUnits NonVirtualBaseAdjustment);

  llvm::Constant *BuildMemberPointer(const CXXRecordDecl *RD,
                                     const CXXMethodDecl *MD,
                                     CharUnits NonVirtualBaseAdjustment);

  bool MemberPointerConstantIsNull(const MemberPointerType *MPT,
                                   llvm::Constant *MP);

  /// \brief - Initialize all vbptrs of 'this' with RD as the complete type.
  void EmitVBPtrStores(CodeGenFunction &CGF, const CXXRecordDecl *RD);

  /// \brief Caching wrapper around VBTableBuilder::enumerateVBTables().
  const VBTableVector &EnumerateVBTables(const CXXRecordDecl *RD);

  /// \brief Generate a thunk for calling a virtual member function MD.
  llvm::Function *EmitVirtualMemPtrThunk(const CXXMethodDecl *MD,
                                         StringRef ThunkName);

public:
  virtual llvm::Type *ConvertMemberPointerType(const MemberPointerType *MPT);

  virtual bool isZeroInitializable(const MemberPointerType *MPT);

  virtual llvm::Constant *EmitNullMemberPointer(const MemberPointerType *MPT);

  virtual llvm::Constant *EmitMemberDataPointer(const MemberPointerType *MPT,
                                                CharUnits offset);
  virtual llvm::Constant *EmitMemberPointer(const CXXMethodDecl *MD);
  virtual llvm::Constant *EmitMemberPointer(const APValue &MP, QualType MPT);

  virtual llvm::Value *EmitMemberPointerComparison(CodeGenFunction &CGF,
                                                   llvm::Value *L,
                                                   llvm::Value *R,
                                                   const MemberPointerType *MPT,
                                                   bool Inequality);

  virtual llvm::Value *EmitMemberPointerIsNotNull(CodeGenFunction &CGF,
                                                  llvm::Value *MemPtr,
                                                  const MemberPointerType *MPT);

  virtual llvm::Value *EmitMemberDataPointerAddress(CodeGenFunction &CGF,
                                                    llvm::Value *Base,
                                                    llvm::Value *MemPtr,
                                                  const MemberPointerType *MPT);

  virtual llvm::Value *EmitMemberPointerConversion(CodeGenFunction &CGF,
                                                   const CastExpr *E,
                                                   llvm::Value *Src);

  virtual llvm::Constant *EmitMemberPointerConversion(const CastExpr *E,
                                                      llvm::Constant *Src);

  virtual llvm::Value *
  EmitLoadOfMemberFunctionPointer(CodeGenFunction &CGF,
                                  llvm::Value *&This,
                                  llvm::Value *MemPtr,
                                  const MemberPointerType *MPT);

private:
  typedef std::pair<const CXXRecordDecl *, CharUnits> VFTableIdTy;
  typedef llvm::DenseMap<VFTableIdTy, llvm::GlobalVariable *> VFTablesMapTy;
  /// \brief All the vftables that have been referenced.
  VFTablesMapTy VFTablesMap;

  /// \brief This set holds the record decls we've deferred vtable emission for.
  llvm::SmallPtrSet<const CXXRecordDecl *, 4> DeferredVFTables;


  /// \brief All the vbtables which have been referenced.
  llvm::DenseMap<const CXXRecordDecl *, VBTableVector> VBTablesMap;

  /// Info on the global variable used to guard initialization of static locals.
  /// The BitIndex field is only used for externally invisible declarations.
  struct GuardInfo {
    GuardInfo() : Guard(0), BitIndex(0) {}
    llvm::GlobalVariable *Guard;
    unsigned BitIndex;
  };

  /// Map from DeclContext to the current guard variable.  We assume that the
  /// AST is visited in source code order.
  llvm::DenseMap<const DeclContext *, GuardInfo> GuardVariableMap;
};

}

llvm::Value *MicrosoftCXXABI::adjustToCompleteObject(CodeGenFunction &CGF,
                                                     llvm::Value *ptr,
                                                     QualType type) {
  // FIXME: implement
  return ptr;
}

/// \brief Finds the first non-virtual base of RD that has virtual bases.  If RD
/// doesn't have a vbptr, it will reuse the vbptr of the returned class.
static const CXXRecordDecl *FindFirstNVBaseWithVBases(const CXXRecordDecl *RD) {
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    const CXXRecordDecl *Base = I->getType()->getAsCXXRecordDecl();
    if (!I->isVirtual() && Base->getNumVBases() > 0)
      return Base;
  }
  llvm_unreachable("RD must have an nv base with vbases");
}

CharUnits MicrosoftCXXABI::GetVBPtrOffsetFromBases(const CXXRecordDecl *RD) {
  assert(RD->getNumVBases());
  CharUnits Total = CharUnits::Zero();
  while (RD) {
    const ASTRecordLayout &RDLayout = getContext().getASTRecordLayout(RD);
    CharUnits VBPtrOffset = RDLayout.getVBPtrOffset();
    // -1 is the sentinel for no vbptr.
    if (VBPtrOffset != CharUnits::fromQuantity(-1)) {
      Total += VBPtrOffset;
      break;
    }
    RD = FindFirstNVBaseWithVBases(RD);
    Total += RDLayout.getBaseClassOffset(RD);
  }
  return Total;
}

llvm::Value *
MicrosoftCXXABI::GetVirtualBaseClassOffset(CodeGenFunction &CGF,
                                           llvm::Value *This,
                                           const CXXRecordDecl *ClassDecl,
                                           const CXXRecordDecl *BaseClassDecl) {
  int64_t VBPtrChars = GetVBPtrOffsetFromBases(ClassDecl).getQuantity();
  llvm::Value *VBPtrOffset = llvm::ConstantInt::get(CGM.PtrDiffTy, VBPtrChars);
  CharUnits IntSize = getContext().getTypeSizeInChars(getContext().IntTy);
  CharUnits VBTableChars =
      IntSize *
      CGM.getMicrosoftVTableContext().getVBTableIndex(ClassDecl, BaseClassDecl);
  llvm::Value *VBTableOffset =
    llvm::ConstantInt::get(CGM.IntTy, VBTableChars.getQuantity());

  llvm::Value *VBPtrToNewBase =
    GetVBaseOffsetFromVBPtr(CGF, This, VBPtrOffset, VBTableOffset);
  VBPtrToNewBase =
    CGF.Builder.CreateSExtOrBitCast(VBPtrToNewBase, CGM.PtrDiffTy);
  return CGF.Builder.CreateNSWAdd(VBPtrOffset, VBPtrToNewBase);
}

bool MicrosoftCXXABI::HasThisReturn(GlobalDecl GD) const {
  return isa<CXXConstructorDecl>(GD.getDecl());
}

void MicrosoftCXXABI::BuildConstructorSignature(const CXXConstructorDecl *Ctor,
                                 CXXCtorType Type,
                                 CanQualType &ResTy,
                                 SmallVectorImpl<CanQualType> &ArgTys) {
  // 'this' parameter and 'this' return are already in place

  const CXXRecordDecl *Class = Ctor->getParent();
  if (Class->getNumVBases()) {
    // Constructors of classes with virtual bases take an implicit parameter.
    ArgTys.push_back(CGM.getContext().IntTy);
  }
}

llvm::BasicBlock *
MicrosoftCXXABI::EmitCtorCompleteObjectHandler(CodeGenFunction &CGF,
                                               const CXXRecordDecl *RD) {
  llvm::Value *IsMostDerivedClass = getStructorImplicitParamValue(CGF);
  assert(IsMostDerivedClass &&
         "ctor for a class with virtual bases must have an implicit parameter");
  llvm::Value *IsCompleteObject =
    CGF.Builder.CreateIsNotNull(IsMostDerivedClass, "is_complete_object");

  llvm::BasicBlock *CallVbaseCtorsBB = CGF.createBasicBlock("ctor.init_vbases");
  llvm::BasicBlock *SkipVbaseCtorsBB = CGF.createBasicBlock("ctor.skip_vbases");
  CGF.Builder.CreateCondBr(IsCompleteObject,
                           CallVbaseCtorsBB, SkipVbaseCtorsBB);

  CGF.EmitBlock(CallVbaseCtorsBB);

  // Fill in the vbtable pointers here.
  EmitVBPtrStores(CGF, RD);

  // CGF will put the base ctor calls in this basic block for us later.

  return SkipVbaseCtorsBB;
}

void MicrosoftCXXABI::initializeHiddenVirtualInheritanceMembers(
    CodeGenFunction &CGF, const CXXRecordDecl *RD) {
  // In most cases, an override for a vbase virtual method can adjust
  // the "this" parameter by applying a constant offset.
  // However, this is not enough while a constructor or a destructor of some
  // class X is being executed if all the following conditions are met:
  //  - X has virtual bases, (1)
  //  - X overrides a virtual method M of a vbase Y, (2)
  //  - X itself is a vbase of the most derived class.
  //
  // If (1) and (2) are true, the vtorDisp for vbase Y is a hidden member of X
  // which holds the extra amount of "this" adjustment we must do when we use
  // the X vftables (i.e. during X ctor or dtor).
  // Outside the ctors and dtors, the values of vtorDisps are zero.

  const ASTRecordLayout &Layout = getContext().getASTRecordLayout(RD);
  typedef ASTRecordLayout::VBaseOffsetsMapTy VBOffsets;
  const VBOffsets &VBaseMap = Layout.getVBaseOffsetsMap();
  CGBuilderTy &Builder = CGF.Builder;

  unsigned AS =
      cast<llvm::PointerType>(getThisValue(CGF)->getType())->getAddressSpace();
  llvm::Value *Int8This = 0;  // Initialize lazily.

  for (VBOffsets::const_iterator I = VBaseMap.begin(), E = VBaseMap.end();
        I != E; ++I) {
    if (!I->second.hasVtorDisp())
      continue;

    llvm::Value *VBaseOffset =
        GetVirtualBaseClassOffset(CGF, getThisValue(CGF), RD, I->first);
    // FIXME: it doesn't look right that we SExt in GetVirtualBaseClassOffset()
    // just to Trunc back immediately.
    VBaseOffset = Builder.CreateTruncOrBitCast(VBaseOffset, CGF.Int32Ty);
    uint64_t ConstantVBaseOffset =
        Layout.getVBaseClassOffset(I->first).getQuantity();

    // vtorDisp_for_vbase = vbptr[vbase_idx] - offsetof(RD, vbase).
    llvm::Value *VtorDispValue = Builder.CreateSub(
        VBaseOffset, llvm::ConstantInt::get(CGM.Int32Ty, ConstantVBaseOffset),
        "vtordisp.value");

    if (!Int8This)
      Int8This = Builder.CreateBitCast(getThisValue(CGF),
                                       CGF.Int8Ty->getPointerTo(AS));
    llvm::Value *VtorDispPtr = Builder.CreateInBoundsGEP(Int8This, VBaseOffset);
    // vtorDisp is always the 32-bits before the vbase in the class layout.
    VtorDispPtr = Builder.CreateConstGEP1_32(VtorDispPtr, -4);
    VtorDispPtr = Builder.CreateBitCast(
        VtorDispPtr, CGF.Int32Ty->getPointerTo(AS), "vtordisp.ptr");

    Builder.CreateStore(VtorDispValue, VtorDispPtr);
  }
}

void MicrosoftCXXABI::EmitCXXConstructors(const CXXConstructorDecl *D) {
  // There's only one constructor type in this ABI.
  CGM.EmitGlobal(GlobalDecl(D, Ctor_Complete));
}

void MicrosoftCXXABI::EmitVBPtrStores(CodeGenFunction &CGF,
                                      const CXXRecordDecl *RD) {
  llvm::Value *ThisInt8Ptr =
    CGF.Builder.CreateBitCast(getThisValue(CGF), CGM.Int8PtrTy, "this.int8");

  const VBTableVector &VBTables = EnumerateVBTables(RD);
  for (VBTableVector::const_iterator I = VBTables.begin(), E = VBTables.end();
       I != E; ++I) {
    const ASTRecordLayout &SubobjectLayout =
      CGM.getContext().getASTRecordLayout(I->VBPtrSubobject.getBase());
    uint64_t Offs = (I->VBPtrSubobject.getBaseOffset() +
                     SubobjectLayout.getVBPtrOffset()).getQuantity();
    llvm::Value *VBPtr =
        CGF.Builder.CreateConstInBoundsGEP1_64(ThisInt8Ptr, Offs);
    VBPtr = CGF.Builder.CreateBitCast(VBPtr, I->GV->getType()->getPointerTo(0),
                                      "vbptr." + I->ReusingBase->getName());
    CGF.Builder.CreateStore(I->GV, VBPtr);
  }
}

void MicrosoftCXXABI::BuildDestructorSignature(const CXXDestructorDecl *Dtor,
                                               CXXDtorType Type,
                                               CanQualType &ResTy,
                                        SmallVectorImpl<CanQualType> &ArgTys) {
  // 'this' is already in place

  // TODO: 'for base' flag

  if (Type == Dtor_Deleting) {
    // The scalar deleting destructor takes an implicit int parameter.
    ArgTys.push_back(CGM.getContext().IntTy);
  }
}

void MicrosoftCXXABI::EmitCXXDestructors(const CXXDestructorDecl *D) {
  // The TU defining a dtor is only guaranteed to emit a base destructor.  All
  // other destructor variants are delegating thunks.
  CGM.EmitGlobal(GlobalDecl(D, Dtor_Base));
}

llvm::Value *MicrosoftCXXABI::adjustThisArgumentForVirtualCall(
    CodeGenFunction &CGF, GlobalDecl GD, llvm::Value *This) {
  GD = GD.getCanonicalDecl();
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());
  // FIXME: consider splitting the vdtor vs regular method code into two
  // functions.

  GlobalDecl LookupGD = GD;
  if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(MD)) {
    // Complete dtors take a pointer to the complete object,
    // thus don't need adjustment.
    if (GD.getDtorType() == Dtor_Complete)
      return This;

    // There's only Dtor_Deleting in vftable but it shares the this adjustment
    // with the base one, so look up the deleting one instead.
    LookupGD = GlobalDecl(DD, Dtor_Deleting);
  }
  MicrosoftVTableContext::MethodVFTableLocation ML =
      CGM.getMicrosoftVTableContext().getMethodVFTableLocation(LookupGD);

  unsigned AS = cast<llvm::PointerType>(This->getType())->getAddressSpace();
  llvm::Type *charPtrTy = CGF.Int8Ty->getPointerTo(AS);
  CharUnits StaticOffset = ML.VFPtrOffset;
  if (ML.VBase) {
    bool AvoidVirtualOffset = false;
    if (isa<CXXDestructorDecl>(MD) && GD.getDtorType() == Dtor_Base) {
      // A base destructor can only be called from a complete destructor of the
      // same record type or another destructor of a more derived type;
      // or a constructor of the same record type if an exception is thrown.
      assert(isa<CXXDestructorDecl>(CGF.CurGD.getDecl()) ||
             isa<CXXConstructorDecl>(CGF.CurGD.getDecl()));
      const CXXRecordDecl *CurRD =
          cast<CXXMethodDecl>(CGF.CurGD.getDecl())->getParent();

      if (MD->getParent() == CurRD) {
        if (isa<CXXDestructorDecl>(CGF.CurGD.getDecl()))
          assert(CGF.CurGD.getDtorType() == Dtor_Complete);
        if (isa<CXXConstructorDecl>(CGF.CurGD.getDecl()))
          assert(CGF.CurGD.getCtorType() == Ctor_Complete);
        // We're calling the main base dtor from a complete structor,
        // so we know the "this" offset statically.
        AvoidVirtualOffset = true;
      } else {
        // Let's see if we try to call a destructor of a non-virtual base.
        for (CXXRecordDecl::base_class_const_iterator I = CurRD->bases_begin(),
             E = CurRD->bases_end(); I != E; ++I) {
          if (I->getType()->getAsCXXRecordDecl() != MD->getParent())
            continue;
          // If we call a base destructor for a non-virtual base, we statically
          // know where it expects the vfptr and "this" to be.
          // The total offset should reflect the adjustment done by
          // adjustThisParameterInVirtualFunctionPrologue().
          AvoidVirtualOffset = true;
          break;
        }
      }
    }

    if (AvoidVirtualOffset) {
      const ASTRecordLayout &Layout =
          CGF.getContext().getASTRecordLayout(MD->getParent());
      StaticOffset += Layout.getVBaseClassOffset(ML.VBase);
    } else {
      This = CGF.Builder.CreateBitCast(This, charPtrTy);
      llvm::Value *VBaseOffset =
          GetVirtualBaseClassOffset(CGF, This, MD->getParent(), ML.VBase);
      This = CGF.Builder.CreateInBoundsGEP(This, VBaseOffset);
    }
  }
  if (!StaticOffset.isZero()) {
    assert(StaticOffset.isPositive());
    This = CGF.Builder.CreateBitCast(This, charPtrTy);
    if (ML.VBase) {
      // Non-virtual adjustment might result in a pointer outside the allocated
      // object, e.g. if the final overrider class is laid out after the virtual
      // base that declares a method in the most derived class.
      // FIXME: Update the code that emits this adjustment in thunks prologues.
      This = CGF.Builder.CreateConstGEP1_32(This, StaticOffset.getQuantity());
    } else {
      This = CGF.Builder.CreateConstInBoundsGEP1_32(This,
                                                    StaticOffset.getQuantity());
    }
  }
  return This;
}

static bool IsDeletingDtor(GlobalDecl GD) {
  const CXXMethodDecl* MD = cast<CXXMethodDecl>(GD.getDecl());
  if (isa<CXXDestructorDecl>(MD)) {
    return GD.getDtorType() == Dtor_Deleting;
  }
  return false;
}

void MicrosoftCXXABI::BuildInstanceFunctionParams(CodeGenFunction &CGF,
                                                  QualType &ResTy,
                                                  FunctionArgList &Params) {
  BuildThisParam(CGF, Params);

  ASTContext &Context = getContext();
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(CGF.CurGD.getDecl());
  if (isa<CXXConstructorDecl>(MD) && MD->getParent()->getNumVBases()) {
    ImplicitParamDecl *IsMostDerived
      = ImplicitParamDecl::Create(Context, 0,
                                  CGF.CurGD.getDecl()->getLocation(),
                                  &Context.Idents.get("is_most_derived"),
                                  Context.IntTy);
    Params.push_back(IsMostDerived);
    getStructorImplicitParamDecl(CGF) = IsMostDerived;
  } else if (IsDeletingDtor(CGF.CurGD)) {
    ImplicitParamDecl *ShouldDelete
      = ImplicitParamDecl::Create(Context, 0,
                                  CGF.CurGD.getDecl()->getLocation(),
                                  &Context.Idents.get("should_call_delete"),
                                  Context.IntTy);
    Params.push_back(ShouldDelete);
    getStructorImplicitParamDecl(CGF) = ShouldDelete;
  }
}

llvm::Value *MicrosoftCXXABI::adjustThisParameterInVirtualFunctionPrologue(
    CodeGenFunction &CGF, GlobalDecl GD, llvm::Value *This) {
  GD = GD.getCanonicalDecl();
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());

  GlobalDecl LookupGD = GD;
  if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(MD)) {
    // Complete destructors take a pointer to the complete object as a
    // parameter, thus don't need this adjustment.
    if (GD.getDtorType() == Dtor_Complete)
      return This;

    // There's no Dtor_Base in vftable but it shares the this adjustment with
    // the deleting one, so look it up instead.
    LookupGD = GlobalDecl(DD, Dtor_Deleting);
  }

  // In this ABI, every virtual function takes a pointer to one of the
  // subobjects that first defines it as the 'this' parameter, rather than a
  // pointer to ther final overrider subobject. Thus, we need to adjust it back
  // to the final overrider subobject before use.
  // See comments in the MicrosoftVFTableContext implementation for the details.

  MicrosoftVTableContext::MethodVFTableLocation ML =
      CGM.getMicrosoftVTableContext().getMethodVFTableLocation(LookupGD);
  CharUnits Adjustment = ML.VFPtrOffset;
  if (ML.VBase) {
    const ASTRecordLayout &DerivedLayout =
        CGF.getContext().getASTRecordLayout(MD->getParent());
    Adjustment += DerivedLayout.getVBaseClassOffset(ML.VBase);
  }

  if (Adjustment.isZero())
    return This;

  unsigned AS = cast<llvm::PointerType>(This->getType())->getAddressSpace();
  llvm::Type *charPtrTy = CGF.Int8Ty->getPointerTo(AS),
             *thisTy = This->getType();

  This = CGF.Builder.CreateBitCast(This, charPtrTy);
  assert(Adjustment.isPositive());
  This =
      CGF.Builder.CreateConstInBoundsGEP1_32(This, -Adjustment.getQuantity());
  return CGF.Builder.CreateBitCast(This, thisTy);
}

void MicrosoftCXXABI::EmitInstanceFunctionProlog(CodeGenFunction &CGF) {
  EmitThisParam(CGF);

  /// If this is a function that the ABI specifies returns 'this', initialize
  /// the return slot to 'this' at the start of the function.
  ///
  /// Unlike the setting of return types, this is done within the ABI
  /// implementation instead of by clients of CGCXXABI because:
  /// 1) getThisValue is currently protected
  /// 2) in theory, an ABI could implement 'this' returns some other way;
  ///    HasThisReturn only specifies a contract, not the implementation    
  if (HasThisReturn(CGF.CurGD))
    CGF.Builder.CreateStore(getThisValue(CGF), CGF.ReturnValue);

  const CXXMethodDecl *MD = cast<CXXMethodDecl>(CGF.CurGD.getDecl());
  if (isa<CXXConstructorDecl>(MD) && MD->getParent()->getNumVBases()) {
    assert(getStructorImplicitParamDecl(CGF) &&
           "no implicit parameter for a constructor with virtual bases?");
    getStructorImplicitParamValue(CGF)
      = CGF.Builder.CreateLoad(
          CGF.GetAddrOfLocalVar(getStructorImplicitParamDecl(CGF)),
          "is_most_derived");
  }

  if (IsDeletingDtor(CGF.CurGD)) {
    assert(getStructorImplicitParamDecl(CGF) &&
           "no implicit parameter for a deleting destructor?");
    getStructorImplicitParamValue(CGF)
      = CGF.Builder.CreateLoad(
          CGF.GetAddrOfLocalVar(getStructorImplicitParamDecl(CGF)),
          "should_call_delete");
  }
}

void MicrosoftCXXABI::EmitConstructorCall(CodeGenFunction &CGF,
                                          const CXXConstructorDecl *D,
                                          CXXCtorType Type, 
                                          bool ForVirtualBase,
                                          bool Delegating,
                                          llvm::Value *This,
                                          CallExpr::const_arg_iterator ArgBeg,
                                          CallExpr::const_arg_iterator ArgEnd) {
  assert(Type == Ctor_Complete || Type == Ctor_Base);
  llvm::Value *Callee = CGM.GetAddrOfCXXConstructor(D, Ctor_Complete);

  llvm::Value *ImplicitParam = 0;
  QualType ImplicitParamTy;
  if (D->getParent()->getNumVBases()) {
    ImplicitParam = llvm::ConstantInt::get(CGM.Int32Ty, Type == Ctor_Complete);
    ImplicitParamTy = getContext().IntTy;
  }

  // FIXME: Provide a source location here.
  CGF.EmitCXXMemberCall(D, SourceLocation(), Callee, ReturnValueSlot(), This,
                        ImplicitParam, ImplicitParamTy, ArgBeg, ArgEnd);
}

void MicrosoftCXXABI::emitVTableDefinitions(CodeGenVTables &CGVT,
                                            const CXXRecordDecl *RD) {
  MicrosoftVTableContext &VFTContext = CGM.getMicrosoftVTableContext();
  MicrosoftVTableContext::VFPtrListTy VFPtrs = VFTContext.getVFPtrOffsets(RD);
  llvm::GlobalVariable::LinkageTypes Linkage = CGM.getVTableLinkage(RD);

  for (MicrosoftVTableContext::VFPtrListTy::iterator I = VFPtrs.begin(),
       E = VFPtrs.end(); I != E; ++I) {
    llvm::GlobalVariable *VTable = getAddrOfVTable(RD, I->VFPtrFullOffset);
    if (VTable->hasInitializer())
      continue;

    const VTableLayout &VTLayout =
        VFTContext.getVFTableLayout(RD, I->VFPtrFullOffset);
    llvm::Constant *Init = CGVT.CreateVTableInitializer(
        RD, VTLayout.vtable_component_begin(),
        VTLayout.getNumVTableComponents(), VTLayout.vtable_thunk_begin(),
        VTLayout.getNumVTableThunks());
    VTable->setInitializer(Init);

    VTable->setLinkage(Linkage);
    CGM.setTypeVisibility(VTable, RD, CodeGenModule::TVK_ForVTable);
  }
}

llvm::Value *MicrosoftCXXABI::getVTableAddressPointInStructor(
    CodeGenFunction &CGF, const CXXRecordDecl *VTableClass, BaseSubobject Base,
    const CXXRecordDecl *NearestVBase, bool &NeedsVirtualOffset) {
  NeedsVirtualOffset = (NearestVBase != 0);

  llvm::Value *VTableAddressPoint =
      getAddrOfVTable(VTableClass, Base.getBaseOffset());
  if (!VTableAddressPoint) {
    assert(Base.getBase()->getNumVBases() &&
           !CGM.getContext().getASTRecordLayout(Base.getBase()).hasOwnVFPtr());
  }
  return VTableAddressPoint;
}

static void mangleVFTableName(MicrosoftMangleContext &MangleContext,
                              const CXXRecordDecl *RD, const VFPtrInfo &VFPtr,
                              SmallString<256> &Name) {
  llvm::raw_svector_ostream Out(Name);
  MangleContext.mangleCXXVFTable(RD, VFPtr.PathToMangle, Out);
}

llvm::Constant *MicrosoftCXXABI::getVTableAddressPointForConstExpr(
    BaseSubobject Base, const CXXRecordDecl *VTableClass) {
  llvm::Constant *VTable = getAddrOfVTable(VTableClass, Base.getBaseOffset());
  assert(VTable && "Couldn't find a vftable for the given base?");
  return VTable;
}

llvm::GlobalVariable *MicrosoftCXXABI::getAddrOfVTable(const CXXRecordDecl *RD,
                                                       CharUnits VPtrOffset) {
  // getAddrOfVTable may return 0 if asked to get an address of a vtable which
  // shouldn't be used in the given record type. We want to cache this result in
  // VFTablesMap, thus a simple zero check is not sufficient.
  VFTableIdTy ID(RD, VPtrOffset);
  VFTablesMapTy::iterator I;
  bool Inserted;
  llvm::tie(I, Inserted) = VFTablesMap.insert(
      std::make_pair(ID, static_cast<llvm::GlobalVariable *>(0)));
  if (!Inserted)
    return I->second;

  llvm::GlobalVariable *&VTable = I->second;

  MicrosoftVTableContext &VTContext = CGM.getMicrosoftVTableContext();
  const MicrosoftVTableContext::VFPtrListTy &VFPtrs =
      VTContext.getVFPtrOffsets(RD);

  if (DeferredVFTables.insert(RD)) {
    // We haven't processed this record type before.
    // Queue up this v-table for possible deferred emission.
    CGM.addDeferredVTable(RD);

#ifndef NDEBUG
    // Create all the vftables at once in order to make sure each vftable has
    // a unique mangled name.
    llvm::StringSet<> ObservedMangledNames;
    for (size_t J = 0, F = VFPtrs.size(); J != F; ++J) {
      SmallString<256> Name;
      mangleVFTableName(getMangleContext(), RD, VFPtrs[J], Name);
      if (!ObservedMangledNames.insert(Name.str()))
        llvm_unreachable("Already saw this mangling before?");
    }
#endif
  }

  for (size_t J = 0, F = VFPtrs.size(); J != F; ++J) {
    if (VFPtrs[J].VFPtrFullOffset != VPtrOffset)
      continue;

    llvm::ArrayType *ArrayType = llvm::ArrayType::get(
        CGM.Int8PtrTy,
        VTContext.getVFTableLayout(RD, VFPtrs[J].VFPtrFullOffset)
            .getNumVTableComponents());

    SmallString<256> Name;
    mangleVFTableName(getMangleContext(), RD, VFPtrs[J], Name);
    VTable = CGM.CreateOrReplaceCXXRuntimeVariable(
        Name.str(), ArrayType, llvm::GlobalValue::ExternalLinkage);
    VTable->setUnnamedAddr(true);
    break;
  }

  return VTable;
}

llvm::Value *MicrosoftCXXABI::getVirtualFunctionPointer(CodeGenFunction &CGF,
                                                        GlobalDecl GD,
                                                        llvm::Value *This,
                                                        llvm::Type *Ty) {
  GD = GD.getCanonicalDecl();
  CGBuilderTy &Builder = CGF.Builder;

  Ty = Ty->getPointerTo()->getPointerTo();
  llvm::Value *VPtr = adjustThisArgumentForVirtualCall(CGF, GD, This);
  llvm::Value *VTable = CGF.GetVTablePtr(VPtr, Ty);

  MicrosoftVTableContext::MethodVFTableLocation ML =
      CGM.getMicrosoftVTableContext().getMethodVFTableLocation(GD);
  llvm::Value *VFuncPtr =
      Builder.CreateConstInBoundsGEP1_64(VTable, ML.Index, "vfn");
  return Builder.CreateLoad(VFuncPtr);
}

void MicrosoftCXXABI::EmitVirtualDestructorCall(CodeGenFunction &CGF,
                                                const CXXDestructorDecl *Dtor,
                                                CXXDtorType DtorType,
                                                SourceLocation CallLoc,
                                                llvm::Value *This) {
  assert(DtorType == Dtor_Deleting || DtorType == Dtor_Complete);

  // We have only one destructor in the vftable but can get both behaviors
  // by passing an implicit int parameter.
  GlobalDecl GD(Dtor, Dtor_Deleting);
  const CGFunctionInfo *FInfo =
      &CGM.getTypes().arrangeCXXDestructor(Dtor, Dtor_Deleting);
  llvm::Type *Ty = CGF.CGM.getTypes().GetFunctionType(*FInfo);
  llvm::Value *Callee = getVirtualFunctionPointer(CGF, GD, This, Ty);

  ASTContext &Context = CGF.getContext();
  llvm::Value *ImplicitParam =
      llvm::ConstantInt::get(llvm::IntegerType::getInt32Ty(CGF.getLLVMContext()),
                             DtorType == Dtor_Deleting);

  This = adjustThisArgumentForVirtualCall(CGF, GD, This);
  CGF.EmitCXXMemberCall(Dtor, CallLoc, Callee, ReturnValueSlot(), This,
                        ImplicitParam, Context.IntTy, 0, 0);
}

const VBTableVector &
MicrosoftCXXABI::EnumerateVBTables(const CXXRecordDecl *RD) {
  // At this layer, we can key the cache off of a single class, which is much
  // easier than caching at the GlobalVariable layer.
  llvm::DenseMap<const CXXRecordDecl*, VBTableVector>::iterator I;
  bool added;
  llvm::tie(I, added) = VBTablesMap.insert(std::make_pair(RD, VBTableVector()));
  VBTableVector &VBTables = I->second;
  if (!added)
    return VBTables;

  VBTableBuilder(CGM, RD).enumerateVBTables(VBTables);

  return VBTables;
}

llvm::Function *
MicrosoftCXXABI::EmitVirtualMemPtrThunk(const CXXMethodDecl *MD,
                                        StringRef ThunkName) {
  // If the thunk has been generated previously, just return it.
  if (llvm::GlobalValue *GV = CGM.getModule().getNamedValue(ThunkName))
    return cast<llvm::Function>(GV);

  // Create the llvm::Function.
  const CGFunctionInfo &FnInfo = CGM.getTypes().arrangeGlobalDeclaration(MD);
  llvm::FunctionType *ThunkTy = CGM.getTypes().GetFunctionType(FnInfo);
  llvm::Function *ThunkFn =
      llvm::Function::Create(ThunkTy, llvm::Function::ExternalLinkage,
                             ThunkName.str(), &CGM.getModule());
  assert(ThunkFn->getName() == ThunkName && "name was uniqued!");

  ThunkFn->setLinkage(MD->isExternallyVisible()
                          ? llvm::GlobalValue::LinkOnceODRLinkage
                          : llvm::GlobalValue::InternalLinkage);

  CGM.SetLLVMFunctionAttributes(MD, FnInfo, ThunkFn);
  CGM.SetLLVMFunctionAttributesForDefinition(MD, ThunkFn);

  // Start codegen.
  CodeGenFunction CGF(CGM);
  CGF.StartThunk(ThunkFn, MD, FnInfo);

  // Get to the Callee.
  llvm::Value *This = CGF.LoadCXXThis();
  llvm::Value *Callee = getVirtualFunctionPointer(CGF, MD, This, ThunkTy);

  // Make the call and return the result.
  CGF.EmitCallAndReturnForThunk(MD, Callee, 0);

  return ThunkFn;
}

void MicrosoftCXXABI::emitVirtualInheritanceTables(const CXXRecordDecl *RD) {
  const VBTableVector &VBTables = EnumerateVBTables(RD);
  llvm::GlobalVariable::LinkageTypes Linkage = CGM.getVTableLinkage(RD);

  for (VBTableVector::const_iterator I = VBTables.begin(), E = VBTables.end();
       I != E; ++I) {
    I->EmitVBTableDefinition(CGM, RD, Linkage);
  }
}

llvm::Value *MicrosoftCXXABI::performThisAdjustment(CodeGenFunction &CGF,
                                                    llvm::Value *This,
                                                    const ThisAdjustment &TA) {
  if (TA.isEmpty())
    return This;

  llvm::Value *V = CGF.Builder.CreateBitCast(This, CGF.Int8PtrTy);

  if (!TA.Virtual.isEmpty()) {
    assert(TA.Virtual.Microsoft.VtordispOffset < 0);
    // Adjust the this argument based on the vtordisp value.
    llvm::Value *VtorDispPtr =
        CGF.Builder.CreateConstGEP1_32(V, TA.Virtual.Microsoft.VtordispOffset);
    VtorDispPtr =
        CGF.Builder.CreateBitCast(VtorDispPtr, CGF.Int32Ty->getPointerTo());
    llvm::Value *VtorDisp = CGF.Builder.CreateLoad(VtorDispPtr, "vtordisp");
    V = CGF.Builder.CreateGEP(V, CGF.Builder.CreateNeg(VtorDisp));

    if (TA.Virtual.Microsoft.VBPtrOffset) {
      // If the final overrider is defined in a virtual base other than the one
      // that holds the vfptr, we have to use a vtordispex thunk which looks up
      // the vbtable of the derived class.
      assert(TA.Virtual.Microsoft.VBPtrOffset > 0);
      assert(TA.Virtual.Microsoft.VBOffsetOffset >= 0);
      llvm::Value *VBPtr;
      llvm::Value *VBaseOffset =
          GetVBaseOffsetFromVBPtr(CGF, V, -TA.Virtual.Microsoft.VBPtrOffset,
                                  TA.Virtual.Microsoft.VBOffsetOffset, &VBPtr);
      V = CGF.Builder.CreateInBoundsGEP(VBPtr, VBaseOffset);
    }
  }

  if (TA.NonVirtual) {
    // Non-virtual adjustment might result in a pointer outside the allocated
    // object, e.g. if the final overrider class is laid out after the virtual
    // base that declares a method in the most derived class.
    V = CGF.Builder.CreateConstGEP1_32(V, TA.NonVirtual);
  }

  // Don't need to bitcast back, the call CodeGen will handle this.
  return V;
}

llvm::Value *
MicrosoftCXXABI::performReturnAdjustment(CodeGenFunction &CGF, llvm::Value *Ret,
                                         const ReturnAdjustment &RA) {
  if (RA.isEmpty())
    return Ret;

  llvm::Value *V = CGF.Builder.CreateBitCast(Ret, CGF.Int8PtrTy);

  if (RA.Virtual.Microsoft.VBIndex) {
    assert(RA.Virtual.Microsoft.VBIndex > 0);
    int32_t IntSize =
        getContext().getTypeSizeInChars(getContext().IntTy).getQuantity();
    llvm::Value *VBPtr;
    llvm::Value *VBaseOffset =
        GetVBaseOffsetFromVBPtr(CGF, V, RA.Virtual.Microsoft.VBPtrOffset,
                                IntSize * RA.Virtual.Microsoft.VBIndex, &VBPtr);
    V = CGF.Builder.CreateInBoundsGEP(VBPtr, VBaseOffset);
  }

  if (RA.NonVirtual)
    V = CGF.Builder.CreateConstInBoundsGEP1_32(V, RA.NonVirtual);

  // Cast back to the original type.
  return CGF.Builder.CreateBitCast(V, Ret->getType());
}

bool MicrosoftCXXABI::requiresArrayCookie(const CXXDeleteExpr *expr,
                                   QualType elementType) {
  // Microsoft seems to completely ignore the possibility of a
  // two-argument usual deallocation function.
  return elementType.isDestructedType();
}

bool MicrosoftCXXABI::requiresArrayCookie(const CXXNewExpr *expr) {
  // Microsoft seems to completely ignore the possibility of a
  // two-argument usual deallocation function.
  return expr->getAllocatedType().isDestructedType();
}

CharUnits MicrosoftCXXABI::getArrayCookieSizeImpl(QualType type) {
  // The array cookie is always a size_t; we then pad that out to the
  // alignment of the element type.
  ASTContext &Ctx = getContext();
  return std::max(Ctx.getTypeSizeInChars(Ctx.getSizeType()),
                  Ctx.getTypeAlignInChars(type));
}

llvm::Value *MicrosoftCXXABI::readArrayCookieImpl(CodeGenFunction &CGF,
                                                  llvm::Value *allocPtr,
                                                  CharUnits cookieSize) {
  unsigned AS = allocPtr->getType()->getPointerAddressSpace();
  llvm::Value *numElementsPtr =
    CGF.Builder.CreateBitCast(allocPtr, CGF.SizeTy->getPointerTo(AS));
  return CGF.Builder.CreateLoad(numElementsPtr);
}

llvm::Value* MicrosoftCXXABI::InitializeArrayCookie(CodeGenFunction &CGF,
                                                    llvm::Value *newPtr,
                                                    llvm::Value *numElements,
                                                    const CXXNewExpr *expr,
                                                    QualType elementType) {
  assert(requiresArrayCookie(expr));

  // The size of the cookie.
  CharUnits cookieSize = getArrayCookieSizeImpl(elementType);

  // Compute an offset to the cookie.
  llvm::Value *cookiePtr = newPtr;

  // Write the number of elements into the appropriate slot.
  unsigned AS = newPtr->getType()->getPointerAddressSpace();
  llvm::Value *numElementsPtr
    = CGF.Builder.CreateBitCast(cookiePtr, CGF.SizeTy->getPointerTo(AS));
  CGF.Builder.CreateStore(numElements, numElementsPtr);

  // Finally, compute a pointer to the actual data buffer by skipping
  // over the cookie completely.
  return CGF.Builder.CreateConstInBoundsGEP1_64(newPtr,
                                                cookieSize.getQuantity());
}

void MicrosoftCXXABI::EmitGuardedInit(CodeGenFunction &CGF, const VarDecl &D,
                                      llvm::GlobalVariable *GV,
                                      bool PerformInit) {
  // MSVC always uses an i32 bitfield to guard initialization, which is *not*
  // threadsafe.  Since the user may be linking in inline functions compiled by
  // cl.exe, there's no reason to provide a false sense of security by using
  // critical sections here.

  if (D.getTLSKind())
    CGM.ErrorUnsupported(&D, "dynamic TLS initialization");

  CGBuilderTy &Builder = CGF.Builder;
  llvm::IntegerType *GuardTy = CGF.Int32Ty;
  llvm::ConstantInt *Zero = llvm::ConstantInt::get(GuardTy, 0);

  // Get the guard variable for this function if we have one already.
  GuardInfo &GI = GuardVariableMap[D.getDeclContext()];

  unsigned BitIndex;
  if (D.isExternallyVisible()) {
    // Externally visible variables have to be numbered in Sema to properly
    // handle unreachable VarDecls.
    BitIndex = getContext().getManglingNumber(&D);
    assert(BitIndex > 0);
    BitIndex--;
  } else {
    // Non-externally visible variables are numbered here in CodeGen.
    BitIndex = GI.BitIndex++;
  }

  if (BitIndex >= 32) {
    if (D.isExternallyVisible())
      ErrorUnsupportedABI(CGF, "more than 32 guarded initializations");
    BitIndex %= 32;
    GI.Guard = 0;
  }

  // Lazily create the i32 bitfield for this function.
  if (!GI.Guard) {
    // Mangle the name for the guard.
    SmallString<256> GuardName;
    {
      llvm::raw_svector_ostream Out(GuardName);
      getMangleContext().mangleStaticGuardVariable(&D, Out);
      Out.flush();
    }

    // Create the guard variable with a zero-initializer.  Just absorb linkage
    // and visibility from the guarded variable.
    GI.Guard = new llvm::GlobalVariable(CGM.getModule(), GuardTy, false,
                                     GV->getLinkage(), Zero, GuardName.str());
    GI.Guard->setVisibility(GV->getVisibility());
  } else {
    assert(GI.Guard->getLinkage() == GV->getLinkage() &&
           "static local from the same function had different linkage");
  }

  // Pseudo code for the test:
  // if (!(GuardVar & MyGuardBit)) {
  //   GuardVar |= MyGuardBit;
  //   ... initialize the object ...;
  // }

  // Test our bit from the guard variable.
  llvm::ConstantInt *Bit = llvm::ConstantInt::get(GuardTy, 1U << BitIndex);
  llvm::LoadInst *LI = Builder.CreateLoad(GI.Guard);
  llvm::Value *IsInitialized =
      Builder.CreateICmpNE(Builder.CreateAnd(LI, Bit), Zero);
  llvm::BasicBlock *InitBlock = CGF.createBasicBlock("init");
  llvm::BasicBlock *EndBlock = CGF.createBasicBlock("init.end");
  Builder.CreateCondBr(IsInitialized, EndBlock, InitBlock);

  // Set our bit in the guard variable and emit the initializer and add a global
  // destructor if appropriate.
  CGF.EmitBlock(InitBlock);
  Builder.CreateStore(Builder.CreateOr(LI, Bit), GI.Guard);
  CGF.EmitCXXGlobalVarDeclInit(D, GV, PerformInit);
  Builder.CreateBr(EndBlock);

  // Continue.
  CGF.EmitBlock(EndBlock);
}

// Member pointer helpers.
static bool hasVBPtrOffsetField(MSInheritanceModel Inheritance) {
  return Inheritance == MSIM_Unspecified;
}

static bool hasOnlyOneField(bool IsMemberFunction,
                            MSInheritanceModel Inheritance) {
  return Inheritance <= MSIM_SinglePolymorphic ||
      (!IsMemberFunction && Inheritance <= MSIM_MultiplePolymorphic);
}

// Only member pointers to functions need a this adjustment, since it can be
// combined with the field offset for data pointers.
static bool hasNonVirtualBaseAdjustmentField(bool IsMemberFunction,
                                             MSInheritanceModel Inheritance) {
  return (IsMemberFunction && Inheritance >= MSIM_Multiple);
}

static bool hasVirtualBaseAdjustmentField(MSInheritanceModel Inheritance) {
  return Inheritance >= MSIM_Virtual;
}

// Use zero for the field offset of a null data member pointer if we can
// guarantee that zero is not a valid field offset, or if the member pointer has
// multiple fields.  Polymorphic classes have a vfptr at offset zero, so we can
// use zero for null.  If there are multiple fields, we can use zero even if it
// is a valid field offset because null-ness testing will check the other
// fields.
static bool nullFieldOffsetIsZero(MSInheritanceModel Inheritance) {
  return Inheritance != MSIM_Multiple && Inheritance != MSIM_Single;
}

bool MicrosoftCXXABI::isZeroInitializable(const MemberPointerType *MPT) {
  // Null-ness for function memptrs only depends on the first field, which is
  // the function pointer.  The rest don't matter, so we can zero initialize.
  if (MPT->isMemberFunctionPointer())
    return true;

  // The virtual base adjustment field is always -1 for null, so if we have one
  // we can't zero initialize.  The field offset is sometimes also -1 if 0 is a
  // valid field offset.
  const CXXRecordDecl *RD = MPT->getClass()->getAsCXXRecordDecl();
  MSInheritanceModel Inheritance = RD->getMSInheritanceModel();
  return (!hasVirtualBaseAdjustmentField(Inheritance) &&
          nullFieldOffsetIsZero(Inheritance));
}

llvm::Type *
MicrosoftCXXABI::ConvertMemberPointerType(const MemberPointerType *MPT) {
  const CXXRecordDecl *RD = MPT->getClass()->getAsCXXRecordDecl();
  MSInheritanceModel Inheritance = RD->getMSInheritanceModel();
  llvm::SmallVector<llvm::Type *, 4> fields;
  if (MPT->isMemberFunctionPointer())
    fields.push_back(CGM.VoidPtrTy);  // FunctionPointerOrVirtualThunk
  else
    fields.push_back(CGM.IntTy);  // FieldOffset

  if (hasNonVirtualBaseAdjustmentField(MPT->isMemberFunctionPointer(),
                                       Inheritance))
    fields.push_back(CGM.IntTy);
  if (hasVBPtrOffsetField(Inheritance))
    fields.push_back(CGM.IntTy);
  if (hasVirtualBaseAdjustmentField(Inheritance))
    fields.push_back(CGM.IntTy);  // VirtualBaseAdjustmentOffset

  if (fields.size() == 1)
    return fields[0];
  return llvm::StructType::get(CGM.getLLVMContext(), fields);
}

void MicrosoftCXXABI::
GetNullMemberPointerFields(const MemberPointerType *MPT,
                           llvm::SmallVectorImpl<llvm::Constant *> &fields) {
  assert(fields.empty());
  const CXXRecordDecl *RD = MPT->getClass()->getAsCXXRecordDecl();
  MSInheritanceModel Inheritance = RD->getMSInheritanceModel();
  if (MPT->isMemberFunctionPointer()) {
    // FunctionPointerOrVirtualThunk
    fields.push_back(llvm::Constant::getNullValue(CGM.VoidPtrTy));
  } else {
    if (nullFieldOffsetIsZero(Inheritance))
      fields.push_back(getZeroInt());  // FieldOffset
    else
      fields.push_back(getAllOnesInt());  // FieldOffset
  }

  if (hasNonVirtualBaseAdjustmentField(MPT->isMemberFunctionPointer(),
                                       Inheritance))
    fields.push_back(getZeroInt());
  if (hasVBPtrOffsetField(Inheritance))
    fields.push_back(getZeroInt());
  if (hasVirtualBaseAdjustmentField(Inheritance))
    fields.push_back(getAllOnesInt());
}

llvm::Constant *
MicrosoftCXXABI::EmitNullMemberPointer(const MemberPointerType *MPT) {
  llvm::SmallVector<llvm::Constant *, 4> fields;
  GetNullMemberPointerFields(MPT, fields);
  if (fields.size() == 1)
    return fields[0];
  llvm::Constant *Res = llvm::ConstantStruct::getAnon(fields);
  assert(Res->getType() == ConvertMemberPointerType(MPT));
  return Res;
}

llvm::Constant *
MicrosoftCXXABI::EmitFullMemberPointer(llvm::Constant *FirstField,
                                       bool IsMemberFunction,
                                       const CXXRecordDecl *RD,
                                       CharUnits NonVirtualBaseAdjustment)
{
  MSInheritanceModel Inheritance = RD->getMSInheritanceModel();

  // Single inheritance class member pointer are represented as scalars instead
  // of aggregates.
  if (hasOnlyOneField(IsMemberFunction, Inheritance))
    return FirstField;

  llvm::SmallVector<llvm::Constant *, 4> fields;
  fields.push_back(FirstField);

  if (hasNonVirtualBaseAdjustmentField(IsMemberFunction, Inheritance))
    fields.push_back(llvm::ConstantInt::get(
      CGM.IntTy, NonVirtualBaseAdjustment.getQuantity()));

  if (hasVBPtrOffsetField(Inheritance)) {
    CharUnits Offs = CharUnits::Zero();
    if (RD->getNumVBases())
      Offs = GetVBPtrOffsetFromBases(RD);
    fields.push_back(llvm::ConstantInt::get(CGM.IntTy, Offs.getQuantity()));
  }

  // The rest of the fields are adjusted by conversions to a more derived class.
  if (hasVirtualBaseAdjustmentField(Inheritance))
    fields.push_back(getZeroInt());

  return llvm::ConstantStruct::getAnon(fields);
}

llvm::Constant *
MicrosoftCXXABI::EmitMemberDataPointer(const MemberPointerType *MPT,
                                       CharUnits offset) {
  const CXXRecordDecl *RD = MPT->getClass()->getAsCXXRecordDecl();
  llvm::Constant *FirstField =
    llvm::ConstantInt::get(CGM.IntTy, offset.getQuantity());
  return EmitFullMemberPointer(FirstField, /*IsMemberFunction=*/false, RD,
                               CharUnits::Zero());
}

llvm::Constant *MicrosoftCXXABI::EmitMemberPointer(const CXXMethodDecl *MD) {
  return BuildMemberPointer(MD->getParent(), MD, CharUnits::Zero());
}

llvm::Constant *MicrosoftCXXABI::EmitMemberPointer(const APValue &MP,
                                                   QualType MPType) {
  const MemberPointerType *MPT = MPType->castAs<MemberPointerType>();
  const ValueDecl *MPD = MP.getMemberPointerDecl();
  if (!MPD)
    return EmitNullMemberPointer(MPT);

  CharUnits ThisAdjustment = getMemberPointerPathAdjustment(MP);

  // FIXME PR15713: Support virtual inheritance paths.

  if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(MPD))
    return BuildMemberPointer(MPT->getClass()->getAsCXXRecordDecl(),
                              MD, ThisAdjustment);

  CharUnits FieldOffset =
    getContext().toCharUnitsFromBits(getContext().getFieldOffset(MPD));
  return EmitMemberDataPointer(MPT, ThisAdjustment + FieldOffset);
}

llvm::Constant *
MicrosoftCXXABI::BuildMemberPointer(const CXXRecordDecl *RD,
                                    const CXXMethodDecl *MD,
                                    CharUnits NonVirtualBaseAdjustment) {
  assert(MD->isInstance() && "Member function must not be static!");
  MD = MD->getCanonicalDecl();
  CodeGenTypes &Types = CGM.getTypes();

  llvm::Constant *FirstField;
  if (!MD->isVirtual()) {
    const FunctionProtoType *FPT = MD->getType()->castAs<FunctionProtoType>();
    llvm::Type *Ty;
    // Check whether the function has a computable LLVM signature.
    if (Types.isFuncTypeConvertible(FPT)) {
      // The function has a computable LLVM signature; use the correct type.
      Ty = Types.GetFunctionType(Types.arrangeCXXMethodDeclaration(MD));
    } else {
      // Use an arbitrary non-function type to tell GetAddrOfFunction that the
      // function type is incomplete.
      Ty = CGM.PtrDiffTy;
    }
    FirstField = CGM.GetAddrOfFunction(MD, Ty);
    FirstField = llvm::ConstantExpr::getBitCast(FirstField, CGM.VoidPtrTy);
  } else {
    MicrosoftVTableContext::MethodVFTableLocation ML =
        CGM.getMicrosoftVTableContext().getMethodVFTableLocation(MD);
    if (MD->isVariadic()) {
      CGM.ErrorUnsupported(MD, "pointer to variadic virtual member function");
      FirstField = llvm::Constant::getNullValue(CGM.VoidPtrTy);
    } else if (!CGM.getTypes().isFuncTypeConvertible(
                    MD->getType()->castAs<FunctionType>())) {
      CGM.ErrorUnsupported(MD, "pointer to virtual member function with "
                               "incomplete return or parameter type");
      FirstField = llvm::Constant::getNullValue(CGM.VoidPtrTy);
    } else if (ML.VBase) {
      CGM.ErrorUnsupported(MD, "pointer to virtual member function overriding "
                               "member function in virtual base class");
      FirstField = llvm::Constant::getNullValue(CGM.VoidPtrTy);
    } else {
      SmallString<256> ThunkName;
      CharUnits PointerWidth = getContext().toCharUnitsFromBits(
          getContext().getTargetInfo().getPointerWidth(0));
      uint64_t OffsetInVFTable = ML.Index * PointerWidth.getQuantity();
      llvm::raw_svector_ostream Out(ThunkName);
      getMangleContext().mangleVirtualMemPtrThunk(MD, OffsetInVFTable, Out);
      Out.flush();

      llvm::Function *Thunk = EmitVirtualMemPtrThunk(MD, ThunkName.str());
      FirstField = llvm::ConstantExpr::getBitCast(Thunk, CGM.VoidPtrTy);
    }
  }

  // The rest of the fields are common with data member pointers.
  return EmitFullMemberPointer(FirstField, /*IsMemberFunction=*/true, RD,
                               NonVirtualBaseAdjustment);
}

/// Member pointers are the same if they're either bitwise identical *or* both
/// null.  Null-ness for function members is determined by the first field,
/// while for data member pointers we must compare all fields.
llvm::Value *
MicrosoftCXXABI::EmitMemberPointerComparison(CodeGenFunction &CGF,
                                             llvm::Value *L,
                                             llvm::Value *R,
                                             const MemberPointerType *MPT,
                                             bool Inequality) {
  CGBuilderTy &Builder = CGF.Builder;

  // Handle != comparisons by switching the sense of all boolean operations.
  llvm::ICmpInst::Predicate Eq;
  llvm::Instruction::BinaryOps And, Or;
  if (Inequality) {
    Eq = llvm::ICmpInst::ICMP_NE;
    And = llvm::Instruction::Or;
    Or = llvm::Instruction::And;
  } else {
    Eq = llvm::ICmpInst::ICMP_EQ;
    And = llvm::Instruction::And;
    Or = llvm::Instruction::Or;
  }

  // If this is a single field member pointer (single inheritance), this is a
  // single icmp.
  const CXXRecordDecl *RD = MPT->getClass()->getAsCXXRecordDecl();
  MSInheritanceModel Inheritance = RD->getMSInheritanceModel();
  if (hasOnlyOneField(MPT->isMemberFunctionPointer(), Inheritance))
    return Builder.CreateICmp(Eq, L, R);

  // Compare the first field.
  llvm::Value *L0 = Builder.CreateExtractValue(L, 0, "lhs.0");
  llvm::Value *R0 = Builder.CreateExtractValue(R, 0, "rhs.0");
  llvm::Value *Cmp0 = Builder.CreateICmp(Eq, L0, R0, "memptr.cmp.first");

  // Compare everything other than the first field.
  llvm::Value *Res = 0;
  llvm::StructType *LType = cast<llvm::StructType>(L->getType());
  for (unsigned I = 1, E = LType->getNumElements(); I != E; ++I) {
    llvm::Value *LF = Builder.CreateExtractValue(L, I);
    llvm::Value *RF = Builder.CreateExtractValue(R, I);
    llvm::Value *Cmp = Builder.CreateICmp(Eq, LF, RF, "memptr.cmp.rest");
    if (Res)
      Res = Builder.CreateBinOp(And, Res, Cmp);
    else
      Res = Cmp;
  }

  // Check if the first field is 0 if this is a function pointer.
  if (MPT->isMemberFunctionPointer()) {
    // (l1 == r1 && ...) || l0 == 0
    llvm::Value *Zero = llvm::Constant::getNullValue(L0->getType());
    llvm::Value *IsZero = Builder.CreateICmp(Eq, L0, Zero, "memptr.cmp.iszero");
    Res = Builder.CreateBinOp(Or, Res, IsZero);
  }

  // Combine the comparison of the first field, which must always be true for
  // this comparison to succeeed.
  return Builder.CreateBinOp(And, Res, Cmp0, "memptr.cmp");
}

llvm::Value *
MicrosoftCXXABI::EmitMemberPointerIsNotNull(CodeGenFunction &CGF,
                                            llvm::Value *MemPtr,
                                            const MemberPointerType *MPT) {
  CGBuilderTy &Builder = CGF.Builder;
  llvm::SmallVector<llvm::Constant *, 4> fields;
  // We only need one field for member functions.
  if (MPT->isMemberFunctionPointer())
    fields.push_back(llvm::Constant::getNullValue(CGM.VoidPtrTy));
  else
    GetNullMemberPointerFields(MPT, fields);
  assert(!fields.empty());
  llvm::Value *FirstField = MemPtr;
  if (MemPtr->getType()->isStructTy())
    FirstField = Builder.CreateExtractValue(MemPtr, 0);
  llvm::Value *Res = Builder.CreateICmpNE(FirstField, fields[0], "memptr.cmp0");

  // For function member pointers, we only need to test the function pointer
  // field.  The other fields if any can be garbage.
  if (MPT->isMemberFunctionPointer())
    return Res;

  // Otherwise, emit a series of compares and combine the results.
  for (int I = 1, E = fields.size(); I < E; ++I) {
    llvm::Value *Field = Builder.CreateExtractValue(MemPtr, I);
    llvm::Value *Next = Builder.CreateICmpNE(Field, fields[I], "memptr.cmp");
    Res = Builder.CreateAnd(Res, Next, "memptr.tobool");
  }
  return Res;
}

bool MicrosoftCXXABI::MemberPointerConstantIsNull(const MemberPointerType *MPT,
                                                  llvm::Constant *Val) {
  // Function pointers are null if the pointer in the first field is null.
  if (MPT->isMemberFunctionPointer()) {
    llvm::Constant *FirstField = Val->getType()->isStructTy() ?
      Val->getAggregateElement(0U) : Val;
    return FirstField->isNullValue();
  }

  // If it's not a function pointer and it's zero initializable, we can easily
  // check zero.
  if (isZeroInitializable(MPT) && Val->isNullValue())
    return true;

  // Otherwise, break down all the fields for comparison.  Hopefully these
  // little Constants are reused, while a big null struct might not be.
  llvm::SmallVector<llvm::Constant *, 4> Fields;
  GetNullMemberPointerFields(MPT, Fields);
  if (Fields.size() == 1) {
    assert(Val->getType()->isIntegerTy());
    return Val == Fields[0];
  }

  unsigned I, E;
  for (I = 0, E = Fields.size(); I != E; ++I) {
    if (Val->getAggregateElement(I) != Fields[I])
      break;
  }
  return I == E;
}

llvm::Value *
MicrosoftCXXABI::GetVBaseOffsetFromVBPtr(CodeGenFunction &CGF,
                                         llvm::Value *This,
                                         llvm::Value *VBPtrOffset,
                                         llvm::Value *VBTableOffset,
                                         llvm::Value **VBPtrOut) {
  CGBuilderTy &Builder = CGF.Builder;
  // Load the vbtable pointer from the vbptr in the instance.
  This = Builder.CreateBitCast(This, CGM.Int8PtrTy);
  llvm::Value *VBPtr =
    Builder.CreateInBoundsGEP(This, VBPtrOffset, "vbptr");
  if (VBPtrOut) *VBPtrOut = VBPtr;
  VBPtr = Builder.CreateBitCast(VBPtr, CGM.Int8PtrTy->getPointerTo(0));
  llvm::Value *VBTable = Builder.CreateLoad(VBPtr, "vbtable");

  // Load an i32 offset from the vb-table.
  llvm::Value *VBaseOffs = Builder.CreateInBoundsGEP(VBTable, VBTableOffset);
  VBaseOffs = Builder.CreateBitCast(VBaseOffs, CGM.Int32Ty->getPointerTo(0));
  return Builder.CreateLoad(VBaseOffs, "vbase_offs");
}

// Returns an adjusted base cast to i8*, since we do more address arithmetic on
// it.
llvm::Value *
MicrosoftCXXABI::AdjustVirtualBase(CodeGenFunction &CGF,
                                   const CXXRecordDecl *RD, llvm::Value *Base,
                                   llvm::Value *VBTableOffset,
                                   llvm::Value *VBPtrOffset) {
  CGBuilderTy &Builder = CGF.Builder;
  Base = Builder.CreateBitCast(Base, CGM.Int8PtrTy);
  llvm::BasicBlock *OriginalBB = 0;
  llvm::BasicBlock *SkipAdjustBB = 0;
  llvm::BasicBlock *VBaseAdjustBB = 0;

  // In the unspecified inheritance model, there might not be a vbtable at all,
  // in which case we need to skip the virtual base lookup.  If there is a
  // vbtable, the first entry is a no-op entry that gives back the original
  // base, so look for a virtual base adjustment offset of zero.
  if (VBPtrOffset) {
    OriginalBB = Builder.GetInsertBlock();
    VBaseAdjustBB = CGF.createBasicBlock("memptr.vadjust");
    SkipAdjustBB = CGF.createBasicBlock("memptr.skip_vadjust");
    llvm::Value *IsVirtual =
      Builder.CreateICmpNE(VBTableOffset, getZeroInt(),
                           "memptr.is_vbase");
    Builder.CreateCondBr(IsVirtual, VBaseAdjustBB, SkipAdjustBB);
    CGF.EmitBlock(VBaseAdjustBB);
  }

  // If we weren't given a dynamic vbptr offset, RD should be complete and we'll
  // know the vbptr offset.
  if (!VBPtrOffset) {
    CharUnits offs = CharUnits::Zero();
    if (RD->getNumVBases()) {
      offs = GetVBPtrOffsetFromBases(RD);
    }
    VBPtrOffset = llvm::ConstantInt::get(CGM.IntTy, offs.getQuantity());
  }
  llvm::Value *VBPtr = 0;
  llvm::Value *VBaseOffs =
    GetVBaseOffsetFromVBPtr(CGF, Base, VBPtrOffset, VBTableOffset, &VBPtr);
  llvm::Value *AdjustedBase = Builder.CreateInBoundsGEP(VBPtr, VBaseOffs);

  // Merge control flow with the case where we didn't have to adjust.
  if (VBaseAdjustBB) {
    Builder.CreateBr(SkipAdjustBB);
    CGF.EmitBlock(SkipAdjustBB);
    llvm::PHINode *Phi = Builder.CreatePHI(CGM.Int8PtrTy, 2, "memptr.base");
    Phi->addIncoming(Base, OriginalBB);
    Phi->addIncoming(AdjustedBase, VBaseAdjustBB);
    return Phi;
  }
  return AdjustedBase;
}

llvm::Value *
MicrosoftCXXABI::EmitMemberDataPointerAddress(CodeGenFunction &CGF,
                                              llvm::Value *Base,
                                              llvm::Value *MemPtr,
                                              const MemberPointerType *MPT) {
  assert(MPT->isMemberDataPointer());
  unsigned AS = Base->getType()->getPointerAddressSpace();
  llvm::Type *PType =
      CGF.ConvertTypeForMem(MPT->getPointeeType())->getPointerTo(AS);
  CGBuilderTy &Builder = CGF.Builder;
  const CXXRecordDecl *RD = MPT->getClass()->getAsCXXRecordDecl();
  MSInheritanceModel Inheritance = RD->getMSInheritanceModel();

  // Extract the fields we need, regardless of model.  We'll apply them if we
  // have them.
  llvm::Value *FieldOffset = MemPtr;
  llvm::Value *VirtualBaseAdjustmentOffset = 0;
  llvm::Value *VBPtrOffset = 0;
  if (MemPtr->getType()->isStructTy()) {
    // We need to extract values.
    unsigned I = 0;
    FieldOffset = Builder.CreateExtractValue(MemPtr, I++);
    if (hasVBPtrOffsetField(Inheritance))
      VBPtrOffset = Builder.CreateExtractValue(MemPtr, I++);
    if (hasVirtualBaseAdjustmentField(Inheritance))
      VirtualBaseAdjustmentOffset = Builder.CreateExtractValue(MemPtr, I++);
  }

  if (VirtualBaseAdjustmentOffset) {
    Base = AdjustVirtualBase(CGF, RD, Base, VirtualBaseAdjustmentOffset,
                             VBPtrOffset);
  }
  llvm::Value *Addr =
    Builder.CreateInBoundsGEP(Base, FieldOffset, "memptr.offset");

  // Cast the address to the appropriate pointer type, adopting the address
  // space of the base pointer.
  return Builder.CreateBitCast(Addr, PType);
}

static MSInheritanceModel
getInheritanceFromMemptr(const MemberPointerType *MPT) {
  return MPT->getClass()->getAsCXXRecordDecl()->getMSInheritanceModel();
}

llvm::Value *
MicrosoftCXXABI::EmitMemberPointerConversion(CodeGenFunction &CGF,
                                             const CastExpr *E,
                                             llvm::Value *Src) {
  assert(E->getCastKind() == CK_DerivedToBaseMemberPointer ||
         E->getCastKind() == CK_BaseToDerivedMemberPointer ||
         E->getCastKind() == CK_ReinterpretMemberPointer);

  // Use constant emission if we can.
  if (isa<llvm::Constant>(Src))
    return EmitMemberPointerConversion(E, cast<llvm::Constant>(Src));

  // We may be adding or dropping fields from the member pointer, so we need
  // both types and the inheritance models of both records.
  const MemberPointerType *SrcTy =
    E->getSubExpr()->getType()->castAs<MemberPointerType>();
  const MemberPointerType *DstTy = E->getType()->castAs<MemberPointerType>();
  MSInheritanceModel SrcInheritance = getInheritanceFromMemptr(SrcTy);
  MSInheritanceModel DstInheritance = getInheritanceFromMemptr(DstTy);
  bool IsFunc = SrcTy->isMemberFunctionPointer();

  // If the classes use the same null representation, reinterpret_cast is a nop.
  bool IsReinterpret = E->getCastKind() == CK_ReinterpretMemberPointer;
  if (IsReinterpret && (IsFunc ||
                        nullFieldOffsetIsZero(SrcInheritance) ==
                        nullFieldOffsetIsZero(DstInheritance)))
    return Src;

  CGBuilderTy &Builder = CGF.Builder;

  // Branch past the conversion if Src is null.
  llvm::Value *IsNotNull = EmitMemberPointerIsNotNull(CGF, Src, SrcTy);
  llvm::Constant *DstNull = EmitNullMemberPointer(DstTy);

  // C++ 5.2.10p9: The null member pointer value is converted to the null member
  //   pointer value of the destination type.
  if (IsReinterpret) {
    // For reinterpret casts, sema ensures that src and dst are both functions
    // or data and have the same size, which means the LLVM types should match.
    assert(Src->getType() == DstNull->getType());
    return Builder.CreateSelect(IsNotNull, Src, DstNull);
  }

  llvm::BasicBlock *OriginalBB = Builder.GetInsertBlock();
  llvm::BasicBlock *ConvertBB = CGF.createBasicBlock("memptr.convert");
  llvm::BasicBlock *ContinueBB = CGF.createBasicBlock("memptr.converted");
  Builder.CreateCondBr(IsNotNull, ConvertBB, ContinueBB);
  CGF.EmitBlock(ConvertBB);

  // Decompose src.
  llvm::Value *FirstField = Src;
  llvm::Value *NonVirtualBaseAdjustment = 0;
  llvm::Value *VirtualBaseAdjustmentOffset = 0;
  llvm::Value *VBPtrOffset = 0;
  if (!hasOnlyOneField(IsFunc, SrcInheritance)) {
    // We need to extract values.
    unsigned I = 0;
    FirstField = Builder.CreateExtractValue(Src, I++);
    if (hasNonVirtualBaseAdjustmentField(IsFunc, SrcInheritance))
      NonVirtualBaseAdjustment = Builder.CreateExtractValue(Src, I++);
    if (hasVBPtrOffsetField(SrcInheritance))
      VBPtrOffset = Builder.CreateExtractValue(Src, I++);
    if (hasVirtualBaseAdjustmentField(SrcInheritance))
      VirtualBaseAdjustmentOffset = Builder.CreateExtractValue(Src, I++);
  }

  // For data pointers, we adjust the field offset directly.  For functions, we
  // have a separate field.
  llvm::Constant *Adj = getMemberPointerAdjustment(E);
  if (Adj) {
    Adj = llvm::ConstantExpr::getTruncOrBitCast(Adj, CGM.IntTy);
    llvm::Value *&NVAdjustField = IsFunc ? NonVirtualBaseAdjustment : FirstField;
    bool isDerivedToBase = (E->getCastKind() == CK_DerivedToBaseMemberPointer);
    if (!NVAdjustField)  // If this field didn't exist in src, it's zero.
      NVAdjustField = getZeroInt();
    if (isDerivedToBase)
      NVAdjustField = Builder.CreateNSWSub(NVAdjustField, Adj, "adj");
    else
      NVAdjustField = Builder.CreateNSWAdd(NVAdjustField, Adj, "adj");
  }

  // FIXME PR15713: Support conversions through virtually derived classes.

  // Recompose dst from the null struct and the adjusted fields from src.
  llvm::Value *Dst;
  if (hasOnlyOneField(IsFunc, DstInheritance)) {
    Dst = FirstField;
  } else {
    Dst = llvm::UndefValue::get(DstNull->getType());
    unsigned Idx = 0;
    Dst = Builder.CreateInsertValue(Dst, FirstField, Idx++);
    if (hasNonVirtualBaseAdjustmentField(IsFunc, DstInheritance))
      Dst = Builder.CreateInsertValue(
        Dst, getValueOrZeroInt(NonVirtualBaseAdjustment), Idx++);
    if (hasVBPtrOffsetField(DstInheritance))
      Dst = Builder.CreateInsertValue(
        Dst, getValueOrZeroInt(VBPtrOffset), Idx++);
    if (hasVirtualBaseAdjustmentField(DstInheritance))
      Dst = Builder.CreateInsertValue(
        Dst, getValueOrZeroInt(VirtualBaseAdjustmentOffset), Idx++);
  }
  Builder.CreateBr(ContinueBB);

  // In the continuation, choose between DstNull and Dst.
  CGF.EmitBlock(ContinueBB);
  llvm::PHINode *Phi = Builder.CreatePHI(DstNull->getType(), 2, "memptr.converted");
  Phi->addIncoming(DstNull, OriginalBB);
  Phi->addIncoming(Dst, ConvertBB);
  return Phi;
}

llvm::Constant *
MicrosoftCXXABI::EmitMemberPointerConversion(const CastExpr *E,
                                             llvm::Constant *Src) {
  const MemberPointerType *SrcTy =
    E->getSubExpr()->getType()->castAs<MemberPointerType>();
  const MemberPointerType *DstTy = E->getType()->castAs<MemberPointerType>();

  // If src is null, emit a new null for dst.  We can't return src because dst
  // might have a new representation.
  if (MemberPointerConstantIsNull(SrcTy, Src))
    return EmitNullMemberPointer(DstTy);

  // We don't need to do anything for reinterpret_casts of non-null member
  // pointers.  We should only get here when the two type representations have
  // the same size.
  if (E->getCastKind() == CK_ReinterpretMemberPointer)
    return Src;

  MSInheritanceModel SrcInheritance = getInheritanceFromMemptr(SrcTy);
  MSInheritanceModel DstInheritance = getInheritanceFromMemptr(DstTy);

  // Decompose src.
  llvm::Constant *FirstField = Src;
  llvm::Constant *NonVirtualBaseAdjustment = 0;
  llvm::Constant *VirtualBaseAdjustmentOffset = 0;
  llvm::Constant *VBPtrOffset = 0;
  bool IsFunc = SrcTy->isMemberFunctionPointer();
  if (!hasOnlyOneField(IsFunc, SrcInheritance)) {
    // We need to extract values.
    unsigned I = 0;
    FirstField = Src->getAggregateElement(I++);
    if (hasNonVirtualBaseAdjustmentField(IsFunc, SrcInheritance))
      NonVirtualBaseAdjustment = Src->getAggregateElement(I++);
    if (hasVBPtrOffsetField(SrcInheritance))
      VBPtrOffset = Src->getAggregateElement(I++);
    if (hasVirtualBaseAdjustmentField(SrcInheritance))
      VirtualBaseAdjustmentOffset = Src->getAggregateElement(I++);
  }

  // For data pointers, we adjust the field offset directly.  For functions, we
  // have a separate field.
  llvm::Constant *Adj = getMemberPointerAdjustment(E);
  if (Adj) {
    Adj = llvm::ConstantExpr::getTruncOrBitCast(Adj, CGM.IntTy);
    llvm::Constant *&NVAdjustField =
      IsFunc ? NonVirtualBaseAdjustment : FirstField;
    bool IsDerivedToBase = (E->getCastKind() == CK_DerivedToBaseMemberPointer);
    if (!NVAdjustField)  // If this field didn't exist in src, it's zero.
      NVAdjustField = getZeroInt();
    if (IsDerivedToBase)
      NVAdjustField = llvm::ConstantExpr::getNSWSub(NVAdjustField, Adj);
    else
      NVAdjustField = llvm::ConstantExpr::getNSWAdd(NVAdjustField, Adj);
  }

  // FIXME PR15713: Support conversions through virtually derived classes.

  // Recompose dst from the null struct and the adjusted fields from src.
  if (hasOnlyOneField(IsFunc, DstInheritance))
    return FirstField;

  llvm::SmallVector<llvm::Constant *, 4> Fields;
  Fields.push_back(FirstField);
  if (hasNonVirtualBaseAdjustmentField(IsFunc, DstInheritance))
    Fields.push_back(getConstantOrZeroInt(NonVirtualBaseAdjustment));
  if (hasVBPtrOffsetField(DstInheritance))
    Fields.push_back(getConstantOrZeroInt(VBPtrOffset));
  if (hasVirtualBaseAdjustmentField(DstInheritance))
    Fields.push_back(getConstantOrZeroInt(VirtualBaseAdjustmentOffset));
  return llvm::ConstantStruct::getAnon(Fields);
}

llvm::Value *
MicrosoftCXXABI::EmitLoadOfMemberFunctionPointer(CodeGenFunction &CGF,
                                                 llvm::Value *&This,
                                                 llvm::Value *MemPtr,
                                                 const MemberPointerType *MPT) {
  assert(MPT->isMemberFunctionPointer());
  const FunctionProtoType *FPT =
    MPT->getPointeeType()->castAs<FunctionProtoType>();
  const CXXRecordDecl *RD = MPT->getClass()->getAsCXXRecordDecl();
  llvm::FunctionType *FTy =
    CGM.getTypes().GetFunctionType(
      CGM.getTypes().arrangeCXXMethodType(RD, FPT));
  CGBuilderTy &Builder = CGF.Builder;

  MSInheritanceModel Inheritance = RD->getMSInheritanceModel();

  // Extract the fields we need, regardless of model.  We'll apply them if we
  // have them.
  llvm::Value *FunctionPointer = MemPtr;
  llvm::Value *NonVirtualBaseAdjustment = NULL;
  llvm::Value *VirtualBaseAdjustmentOffset = NULL;
  llvm::Value *VBPtrOffset = NULL;
  if (MemPtr->getType()->isStructTy()) {
    // We need to extract values.
    unsigned I = 0;
    FunctionPointer = Builder.CreateExtractValue(MemPtr, I++);
    if (hasNonVirtualBaseAdjustmentField(MPT, Inheritance))
      NonVirtualBaseAdjustment = Builder.CreateExtractValue(MemPtr, I++);
    if (hasVBPtrOffsetField(Inheritance))
      VBPtrOffset = Builder.CreateExtractValue(MemPtr, I++);
    if (hasVirtualBaseAdjustmentField(Inheritance))
      VirtualBaseAdjustmentOffset = Builder.CreateExtractValue(MemPtr, I++);
  }

  if (VirtualBaseAdjustmentOffset) {
    This = AdjustVirtualBase(CGF, RD, This, VirtualBaseAdjustmentOffset,
                             VBPtrOffset);
  }

  if (NonVirtualBaseAdjustment) {
    // Apply the adjustment and cast back to the original struct type.
    llvm::Value *Ptr = Builder.CreateBitCast(This, Builder.getInt8PtrTy());
    Ptr = Builder.CreateInBoundsGEP(Ptr, NonVirtualBaseAdjustment);
    This = Builder.CreateBitCast(Ptr, This->getType(), "this.adjusted");
  }

  return Builder.CreateBitCast(FunctionPointer, FTy->getPointerTo());
}

CGCXXABI *clang::CodeGen::CreateMicrosoftCXXABI(CodeGenModule &CGM) {
  return new MicrosoftCXXABI(CGM);
}
