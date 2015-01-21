//===------- ItaniumCXXABI.cpp - Emit LLVM Code from ASTs for a Module ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides C++ code generation targeting the Itanium C++ ABI.  The class
// in this file generates structures that follow the Itanium C++ ABI, which is
// documented at:
//  http://www.codesourcery.com/public/cxx-abi/abi.html
//  http://www.codesourcery.com/public/cxx-abi/abi-eh.html
//
// It also supports the closely-related ARM ABI, documented at:
// http://infocenter.arm.com/help/topic/com.arm.doc.ihi0041c/IHI0041C_cppabi.pdf
//
//===----------------------------------------------------------------------===//

#include "CGCXXABI.h"
#include "CGRecordLayout.h"
#include "CGVTables.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/Type.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Value.h"

using namespace clang;
using namespace CodeGen;

namespace {
class ItaniumCXXABI : public CodeGen::CGCXXABI {
  /// VTables - All the vtables which have been defined.
  llvm::DenseMap<const CXXRecordDecl *, llvm::GlobalVariable *> VTables;

protected:
  bool UseARMMethodPtrABI;
  bool UseARMGuardVarABI;

  ItaniumMangleContext &getMangleContext() {
    return cast<ItaniumMangleContext>(CodeGen::CGCXXABI::getMangleContext());
  }

public:
  ItaniumCXXABI(CodeGen::CodeGenModule &CGM,
                bool UseARMMethodPtrABI = false,
                bool UseARMGuardVarABI = false) :
    CGCXXABI(CGM), UseARMMethodPtrABI(UseARMMethodPtrABI),
    UseARMGuardVarABI(UseARMGuardVarABI) { }

  bool classifyReturnType(CGFunctionInfo &FI) const override;

  RecordArgABI getRecordArgABI(const CXXRecordDecl *RD) const override {
    // Structures with either a non-trivial destructor or a non-trivial
    // copy constructor are always indirect.
    // FIXME: Use canCopyArgument() when it is fixed to handle lazily declared
    // special members.
    if (RD->hasNonTrivialDestructor() || RD->hasNonTrivialCopyConstructor())
      return RAA_Indirect;
    return RAA_Default;
  }

  bool isZeroInitializable(const MemberPointerType *MPT) override;

  llvm::Type *ConvertMemberPointerType(const MemberPointerType *MPT) override;

  llvm::Value *
    EmitLoadOfMemberFunctionPointer(CodeGenFunction &CGF,
                                    const Expr *E,
                                    llvm::Value *&This,
                                    llvm::Value *MemFnPtr,
                                    const MemberPointerType *MPT) override;

  llvm::Value *
    EmitMemberDataPointerAddress(CodeGenFunction &CGF, const Expr *E,
                                 llvm::Value *Base,
                                 llvm::Value *MemPtr,
                                 const MemberPointerType *MPT) override;

  llvm::Value *EmitMemberPointerConversion(CodeGenFunction &CGF,
                                           const CastExpr *E,
                                           llvm::Value *Src) override;
  llvm::Constant *EmitMemberPointerConversion(const CastExpr *E,
                                              llvm::Constant *Src) override;

  llvm::Constant *EmitNullMemberPointer(const MemberPointerType *MPT) override;

  llvm::Constant *EmitMemberPointer(const CXXMethodDecl *MD) override;
  llvm::Constant *EmitMemberDataPointer(const MemberPointerType *MPT,
                                        CharUnits offset) override;
  llvm::Constant *EmitMemberPointer(const APValue &MP, QualType MPT) override;
  llvm::Constant *BuildMemberPointer(const CXXMethodDecl *MD,
                                     CharUnits ThisAdjustment);

  llvm::Value *EmitMemberPointerComparison(CodeGenFunction &CGF,
                                           llvm::Value *L, llvm::Value *R,
                                           const MemberPointerType *MPT,
                                           bool Inequality) override;

  llvm::Value *EmitMemberPointerIsNotNull(CodeGenFunction &CGF,
                                         llvm::Value *Addr,
                                         const MemberPointerType *MPT) override;

  void emitVirtualObjectDelete(CodeGenFunction &CGF, const CXXDeleteExpr *DE,
                               llvm::Value *Ptr, QualType ElementType,
                               const CXXDestructorDecl *Dtor) override;

  void emitRethrow(CodeGenFunction &CGF, bool isNoReturn) override;

  void EmitFundamentalRTTIDescriptor(QualType Type);
  void EmitFundamentalRTTIDescriptors();
  llvm::Constant *getAddrOfRTTIDescriptor(QualType Ty) override;

  bool shouldTypeidBeNullChecked(bool IsDeref, QualType SrcRecordTy) override;
  void EmitBadTypeidCall(CodeGenFunction &CGF) override;
  llvm::Value *EmitTypeid(CodeGenFunction &CGF, QualType SrcRecordTy,
                          llvm::Value *ThisPtr,
                          llvm::Type *StdTypeInfoPtrTy) override;

  bool shouldDynamicCastCallBeNullChecked(bool SrcIsPtr,
                                          QualType SrcRecordTy) override;

  llvm::Value *EmitDynamicCastCall(CodeGenFunction &CGF, llvm::Value *Value,
                                   QualType SrcRecordTy, QualType DestTy,
                                   QualType DestRecordTy,
                                   llvm::BasicBlock *CastEnd) override;

  llvm::Value *EmitDynamicCastToVoid(CodeGenFunction &CGF, llvm::Value *Value,
                                     QualType SrcRecordTy,
                                     QualType DestTy) override;

  bool EmitBadCastCall(CodeGenFunction &CGF) override;

  llvm::Value *
    GetVirtualBaseClassOffset(CodeGenFunction &CGF, llvm::Value *This,
                              const CXXRecordDecl *ClassDecl,
                              const CXXRecordDecl *BaseClassDecl) override;

  void EmitCXXConstructors(const CXXConstructorDecl *D) override;

  void buildStructorSignature(const CXXMethodDecl *MD, StructorType T,
                              SmallVectorImpl<CanQualType> &ArgTys) override;

  bool useThunkForDtorVariant(const CXXDestructorDecl *Dtor,
                              CXXDtorType DT) const override {
    // Itanium does not emit any destructor variant as an inline thunk.
    // Delegating may occur as an optimization, but all variants are either
    // emitted with external linkage or as linkonce if they are inline and used.
    return false;
  }

  void EmitCXXDestructors(const CXXDestructorDecl *D) override;

  void addImplicitStructorParams(CodeGenFunction &CGF, QualType &ResTy,
                                 FunctionArgList &Params) override;

  void EmitInstanceFunctionProlog(CodeGenFunction &CGF) override;

  unsigned addImplicitConstructorArgs(CodeGenFunction &CGF,
                                      const CXXConstructorDecl *D,
                                      CXXCtorType Type, bool ForVirtualBase,
                                      bool Delegating,
                                      CallArgList &Args) override;

  void EmitDestructorCall(CodeGenFunction &CGF, const CXXDestructorDecl *DD,
                          CXXDtorType Type, bool ForVirtualBase,
                          bool Delegating, llvm::Value *This) override;

  void emitVTableDefinitions(CodeGenVTables &CGVT,
                             const CXXRecordDecl *RD) override;

  llvm::Value *getVTableAddressPointInStructor(
      CodeGenFunction &CGF, const CXXRecordDecl *VTableClass,
      BaseSubobject Base, const CXXRecordDecl *NearestVBase,
      bool &NeedsVirtualOffset) override;

  llvm::Constant *
  getVTableAddressPointForConstExpr(BaseSubobject Base,
                                    const CXXRecordDecl *VTableClass) override;

  llvm::GlobalVariable *getAddrOfVTable(const CXXRecordDecl *RD,
                                        CharUnits VPtrOffset) override;

  llvm::Value *getVirtualFunctionPointer(CodeGenFunction &CGF, GlobalDecl GD,
                                         llvm::Value *This,
                                         llvm::Type *Ty) override;

  llvm::Value *EmitVirtualDestructorCall(CodeGenFunction &CGF,
                                         const CXXDestructorDecl *Dtor,
                                         CXXDtorType DtorType,
                                         llvm::Value *This,
                                         const CXXMemberCallExpr *CE) override;

  void emitVirtualInheritanceTables(const CXXRecordDecl *RD) override;

  void setThunkLinkage(llvm::Function *Thunk, bool ForVTable, GlobalDecl GD,
                       bool ReturnAdjustment) override {
    // Allow inlining of thunks by emitting them with available_externally
    // linkage together with vtables when needed.
    if (ForVTable)
      Thunk->setLinkage(llvm::GlobalValue::AvailableExternallyLinkage);
  }

  llvm::Value *performThisAdjustment(CodeGenFunction &CGF, llvm::Value *This,
                                     const ThisAdjustment &TA) override;

  llvm::Value *performReturnAdjustment(CodeGenFunction &CGF, llvm::Value *Ret,
                                       const ReturnAdjustment &RA) override;

  size_t getSrcArgforCopyCtor(const CXXConstructorDecl *,
                              FunctionArgList &Args) const override {
    assert(!Args.empty() && "expected the arglist to not be empty!");
    return Args.size() - 1;
  }

  StringRef GetPureVirtualCallName() override { return "__cxa_pure_virtual"; }
  StringRef GetDeletedVirtualCallName() override
    { return "__cxa_deleted_virtual"; }

  CharUnits getArrayCookieSizeImpl(QualType elementType) override;
  llvm::Value *InitializeArrayCookie(CodeGenFunction &CGF,
                                     llvm::Value *NewPtr,
                                     llvm::Value *NumElements,
                                     const CXXNewExpr *expr,
                                     QualType ElementType) override;
  llvm::Value *readArrayCookieImpl(CodeGenFunction &CGF,
                                   llvm::Value *allocPtr,
                                   CharUnits cookieSize) override;

  void EmitGuardedInit(CodeGenFunction &CGF, const VarDecl &D,
                       llvm::GlobalVariable *DeclPtr,
                       bool PerformInit) override;
  void registerGlobalDtor(CodeGenFunction &CGF, const VarDecl &D,
                          llvm::Constant *dtor, llvm::Constant *addr) override;

  llvm::Function *getOrCreateThreadLocalWrapper(const VarDecl *VD,
                                                llvm::Value *Val);
  void EmitThreadLocalInitFuncs(
      CodeGenModule &CGM,
      ArrayRef<std::pair<const VarDecl *, llvm::GlobalVariable *>>
          CXXThreadLocals,
      ArrayRef<llvm::Function *> CXXThreadLocalInits,
      ArrayRef<llvm::GlobalVariable *> CXXThreadLocalInitVars) override;

  bool usesThreadWrapperFunction() const override { return true; }
  LValue EmitThreadLocalVarDeclLValue(CodeGenFunction &CGF, const VarDecl *VD,
                                      QualType LValType) override;

  bool NeedsVTTParameter(GlobalDecl GD) override;

  /**************************** RTTI Uniqueness ******************************/

protected:
  /// Returns true if the ABI requires RTTI type_info objects to be unique
  /// across a program.
  virtual bool shouldRTTIBeUnique() const { return true; }

public:
  /// What sort of unique-RTTI behavior should we use?
  enum RTTIUniquenessKind {
    /// We are guaranteeing, or need to guarantee, that the RTTI string
    /// is unique.
    RUK_Unique,

    /// We are not guaranteeing uniqueness for the RTTI string, so we
    /// can demote to hidden visibility but must use string comparisons.
    RUK_NonUniqueHidden,

    /// We are not guaranteeing uniqueness for the RTTI string, so we
    /// have to use string comparisons, but we also have to emit it with
    /// non-hidden visibility.
    RUK_NonUniqueVisible
  };

  /// Return the required visibility status for the given type and linkage in
  /// the current ABI.
  RTTIUniquenessKind
  classifyRTTIUniqueness(QualType CanTy,
                         llvm::GlobalValue::LinkageTypes Linkage) const;
  friend class ItaniumRTTIBuilder;

  void emitCXXStructor(const CXXMethodDecl *MD, StructorType Type) override;
};

class ARMCXXABI : public ItaniumCXXABI {
public:
  ARMCXXABI(CodeGen::CodeGenModule &CGM) :
    ItaniumCXXABI(CGM, /* UseARMMethodPtrABI = */ true,
                  /* UseARMGuardVarABI = */ true) {}

  bool HasThisReturn(GlobalDecl GD) const override {
    return (isa<CXXConstructorDecl>(GD.getDecl()) || (
              isa<CXXDestructorDecl>(GD.getDecl()) &&
              GD.getDtorType() != Dtor_Deleting));
  }

  void EmitReturnFromThunk(CodeGenFunction &CGF, RValue RV,
                           QualType ResTy) override;

  CharUnits getArrayCookieSizeImpl(QualType elementType) override;
  llvm::Value *InitializeArrayCookie(CodeGenFunction &CGF,
                                     llvm::Value *NewPtr,
                                     llvm::Value *NumElements,
                                     const CXXNewExpr *expr,
                                     QualType ElementType) override;
  llvm::Value *readArrayCookieImpl(CodeGenFunction &CGF, llvm::Value *allocPtr,
                                   CharUnits cookieSize) override;
};

class iOS64CXXABI : public ARMCXXABI {
public:
  iOS64CXXABI(CodeGen::CodeGenModule &CGM) : ARMCXXABI(CGM) {}

  // ARM64 libraries are prepared for non-unique RTTI.
  bool shouldRTTIBeUnique() const override { return false; }
};
}

CodeGen::CGCXXABI *CodeGen::CreateItaniumCXXABI(CodeGenModule &CGM) {
  switch (CGM.getTarget().getCXXABI().getKind()) {
  // For IR-generation purposes, there's no significant difference
  // between the ARM and iOS ABIs.
  case TargetCXXABI::GenericARM:
  case TargetCXXABI::iOS:
    return new ARMCXXABI(CGM);

  case TargetCXXABI::iOS64:
    return new iOS64CXXABI(CGM);

  // Note that AArch64 uses the generic ItaniumCXXABI class since it doesn't
  // include the other 32-bit ARM oddities: constructor/destructor return values
  // and array cookies.
  case TargetCXXABI::GenericAArch64:
    return new ItaniumCXXABI(CGM, /* UseARMMethodPtrABI = */ true,
                             /* UseARMGuardVarABI = */ true);

  case TargetCXXABI::GenericItanium:
    if (CGM.getContext().getTargetInfo().getTriple().getArch()
        == llvm::Triple::le32) {
      // For PNaCl, use ARM-style method pointers so that PNaCl code
      // does not assume anything about the alignment of function
      // pointers.
      return new ItaniumCXXABI(CGM, /* UseARMMethodPtrABI = */ true,
                               /* UseARMGuardVarABI = */ false);
    }
    return new ItaniumCXXABI(CGM);

  case TargetCXXABI::Microsoft:
    llvm_unreachable("Microsoft ABI is not Itanium-based");
  }
  llvm_unreachable("bad ABI kind");
}

llvm::Type *
ItaniumCXXABI::ConvertMemberPointerType(const MemberPointerType *MPT) {
  if (MPT->isMemberDataPointer())
    return CGM.PtrDiffTy;
  return llvm::StructType::get(CGM.PtrDiffTy, CGM.PtrDiffTy, nullptr);
}

/// In the Itanium and ARM ABIs, method pointers have the form:
///   struct { ptrdiff_t ptr; ptrdiff_t adj; } memptr;
///
/// In the Itanium ABI:
///  - method pointers are virtual if (memptr.ptr & 1) is nonzero
///  - the this-adjustment is (memptr.adj)
///  - the virtual offset is (memptr.ptr - 1)
///
/// In the ARM ABI:
///  - method pointers are virtual if (memptr.adj & 1) is nonzero
///  - the this-adjustment is (memptr.adj >> 1)
///  - the virtual offset is (memptr.ptr)
/// ARM uses 'adj' for the virtual flag because Thumb functions
/// may be only single-byte aligned.
///
/// If the member is virtual, the adjusted 'this' pointer points
/// to a vtable pointer from which the virtual offset is applied.
///
/// If the member is non-virtual, memptr.ptr is the address of
/// the function to call.
llvm::Value *ItaniumCXXABI::EmitLoadOfMemberFunctionPointer(
    CodeGenFunction &CGF, const Expr *E, llvm::Value *&This,
    llvm::Value *MemFnPtr, const MemberPointerType *MPT) {
  CGBuilderTy &Builder = CGF.Builder;

  const FunctionProtoType *FPT = 
    MPT->getPointeeType()->getAs<FunctionProtoType>();
  const CXXRecordDecl *RD = 
    cast<CXXRecordDecl>(MPT->getClass()->getAs<RecordType>()->getDecl());

  llvm::FunctionType *FTy = 
    CGM.getTypes().GetFunctionType(
      CGM.getTypes().arrangeCXXMethodType(RD, FPT));

  llvm::Constant *ptrdiff_1 = llvm::ConstantInt::get(CGM.PtrDiffTy, 1);

  llvm::BasicBlock *FnVirtual = CGF.createBasicBlock("memptr.virtual");
  llvm::BasicBlock *FnNonVirtual = CGF.createBasicBlock("memptr.nonvirtual");
  llvm::BasicBlock *FnEnd = CGF.createBasicBlock("memptr.end");

  // Extract memptr.adj, which is in the second field.
  llvm::Value *RawAdj = Builder.CreateExtractValue(MemFnPtr, 1, "memptr.adj");

  // Compute the true adjustment.
  llvm::Value *Adj = RawAdj;
  if (UseARMMethodPtrABI)
    Adj = Builder.CreateAShr(Adj, ptrdiff_1, "memptr.adj.shifted");

  // Apply the adjustment and cast back to the original struct type
  // for consistency.
  llvm::Value *Ptr = Builder.CreateBitCast(This, Builder.getInt8PtrTy());
  Ptr = Builder.CreateInBoundsGEP(Ptr, Adj);
  This = Builder.CreateBitCast(Ptr, This->getType(), "this.adjusted");
  
  // Load the function pointer.
  llvm::Value *FnAsInt = Builder.CreateExtractValue(MemFnPtr, 0, "memptr.ptr");
  
  // If the LSB in the function pointer is 1, the function pointer points to
  // a virtual function.
  llvm::Value *IsVirtual;
  if (UseARMMethodPtrABI)
    IsVirtual = Builder.CreateAnd(RawAdj, ptrdiff_1);
  else
    IsVirtual = Builder.CreateAnd(FnAsInt, ptrdiff_1);
  IsVirtual = Builder.CreateIsNotNull(IsVirtual, "memptr.isvirtual");
  Builder.CreateCondBr(IsVirtual, FnVirtual, FnNonVirtual);

  // In the virtual path, the adjustment left 'This' pointing to the
  // vtable of the correct base subobject.  The "function pointer" is an
  // offset within the vtable (+1 for the virtual flag on non-ARM).
  CGF.EmitBlock(FnVirtual);

  // Cast the adjusted this to a pointer to vtable pointer and load.
  llvm::Type *VTableTy = Builder.getInt8PtrTy();
  llvm::Value *VTable = CGF.GetVTablePtr(This, VTableTy);

  // Apply the offset.
  llvm::Value *VTableOffset = FnAsInt;
  if (!UseARMMethodPtrABI)
    VTableOffset = Builder.CreateSub(VTableOffset, ptrdiff_1);
  VTable = Builder.CreateGEP(VTable, VTableOffset);

  // Load the virtual function to call.
  VTable = Builder.CreateBitCast(VTable, FTy->getPointerTo()->getPointerTo());
  llvm::Value *VirtualFn = Builder.CreateLoad(VTable, "memptr.virtualfn");
  CGF.EmitBranch(FnEnd);

  // In the non-virtual path, the function pointer is actually a
  // function pointer.
  CGF.EmitBlock(FnNonVirtual);
  llvm::Value *NonVirtualFn =
    Builder.CreateIntToPtr(FnAsInt, FTy->getPointerTo(), "memptr.nonvirtualfn");
  
  // We're done.
  CGF.EmitBlock(FnEnd);
  llvm::PHINode *Callee = Builder.CreatePHI(FTy->getPointerTo(), 2);
  Callee->addIncoming(VirtualFn, FnVirtual);
  Callee->addIncoming(NonVirtualFn, FnNonVirtual);
  return Callee;
}

/// Compute an l-value by applying the given pointer-to-member to a
/// base object.
llvm::Value *ItaniumCXXABI::EmitMemberDataPointerAddress(
    CodeGenFunction &CGF, const Expr *E, llvm::Value *Base, llvm::Value *MemPtr,
    const MemberPointerType *MPT) {
  assert(MemPtr->getType() == CGM.PtrDiffTy);

  CGBuilderTy &Builder = CGF.Builder;

  unsigned AS = Base->getType()->getPointerAddressSpace();

  // Cast to char*.
  Base = Builder.CreateBitCast(Base, Builder.getInt8Ty()->getPointerTo(AS));

  // Apply the offset, which we assume is non-null.
  llvm::Value *Addr = Builder.CreateInBoundsGEP(Base, MemPtr, "memptr.offset");

  // Cast the address to the appropriate pointer type, adopting the
  // address space of the base pointer.
  llvm::Type *PType
    = CGF.ConvertTypeForMem(MPT->getPointeeType())->getPointerTo(AS);
  return Builder.CreateBitCast(Addr, PType);
}

/// Perform a bitcast, derived-to-base, or base-to-derived member pointer
/// conversion.
///
/// Bitcast conversions are always a no-op under Itanium.
///
/// Obligatory offset/adjustment diagram:
///         <-- offset -->          <-- adjustment -->
///   |--------------------------|----------------------|--------------------|
///   ^Derived address point     ^Base address point    ^Member address point
///
/// So when converting a base member pointer to a derived member pointer,
/// we add the offset to the adjustment because the address point has
/// decreased;  and conversely, when converting a derived MP to a base MP
/// we subtract the offset from the adjustment because the address point
/// has increased.
///
/// The standard forbids (at compile time) conversion to and from
/// virtual bases, which is why we don't have to consider them here.
///
/// The standard forbids (at run time) casting a derived MP to a base
/// MP when the derived MP does not point to a member of the base.
/// This is why -1 is a reasonable choice for null data member
/// pointers.
llvm::Value *
ItaniumCXXABI::EmitMemberPointerConversion(CodeGenFunction &CGF,
                                           const CastExpr *E,
                                           llvm::Value *src) {
  assert(E->getCastKind() == CK_DerivedToBaseMemberPointer ||
         E->getCastKind() == CK_BaseToDerivedMemberPointer ||
         E->getCastKind() == CK_ReinterpretMemberPointer);

  // Under Itanium, reinterprets don't require any additional processing.
  if (E->getCastKind() == CK_ReinterpretMemberPointer) return src;

  // Use constant emission if we can.
  if (isa<llvm::Constant>(src))
    return EmitMemberPointerConversion(E, cast<llvm::Constant>(src));

  llvm::Constant *adj = getMemberPointerAdjustment(E);
  if (!adj) return src;

  CGBuilderTy &Builder = CGF.Builder;
  bool isDerivedToBase = (E->getCastKind() == CK_DerivedToBaseMemberPointer);

  const MemberPointerType *destTy =
    E->getType()->castAs<MemberPointerType>();

  // For member data pointers, this is just a matter of adding the
  // offset if the source is non-null.
  if (destTy->isMemberDataPointer()) {
    llvm::Value *dst;
    if (isDerivedToBase)
      dst = Builder.CreateNSWSub(src, adj, "adj");
    else
      dst = Builder.CreateNSWAdd(src, adj, "adj");

    // Null check.
    llvm::Value *null = llvm::Constant::getAllOnesValue(src->getType());
    llvm::Value *isNull = Builder.CreateICmpEQ(src, null, "memptr.isnull");
    return Builder.CreateSelect(isNull, src, dst);
  }

  // The this-adjustment is left-shifted by 1 on ARM.
  if (UseARMMethodPtrABI) {
    uint64_t offset = cast<llvm::ConstantInt>(adj)->getZExtValue();
    offset <<= 1;
    adj = llvm::ConstantInt::get(adj->getType(), offset);
  }

  llvm::Value *srcAdj = Builder.CreateExtractValue(src, 1, "src.adj");
  llvm::Value *dstAdj;
  if (isDerivedToBase)
    dstAdj = Builder.CreateNSWSub(srcAdj, adj, "adj");
  else
    dstAdj = Builder.CreateNSWAdd(srcAdj, adj, "adj");

  return Builder.CreateInsertValue(src, dstAdj, 1);
}

llvm::Constant *
ItaniumCXXABI::EmitMemberPointerConversion(const CastExpr *E,
                                           llvm::Constant *src) {
  assert(E->getCastKind() == CK_DerivedToBaseMemberPointer ||
         E->getCastKind() == CK_BaseToDerivedMemberPointer ||
         E->getCastKind() == CK_ReinterpretMemberPointer);

  // Under Itanium, reinterprets don't require any additional processing.
  if (E->getCastKind() == CK_ReinterpretMemberPointer) return src;

  // If the adjustment is trivial, we don't need to do anything.
  llvm::Constant *adj = getMemberPointerAdjustment(E);
  if (!adj) return src;

  bool isDerivedToBase = (E->getCastKind() == CK_DerivedToBaseMemberPointer);

  const MemberPointerType *destTy =
    E->getType()->castAs<MemberPointerType>();

  // For member data pointers, this is just a matter of adding the
  // offset if the source is non-null.
  if (destTy->isMemberDataPointer()) {
    // null maps to null.
    if (src->isAllOnesValue()) return src;

    if (isDerivedToBase)
      return llvm::ConstantExpr::getNSWSub(src, adj);
    else
      return llvm::ConstantExpr::getNSWAdd(src, adj);
  }

  // The this-adjustment is left-shifted by 1 on ARM.
  if (UseARMMethodPtrABI) {
    uint64_t offset = cast<llvm::ConstantInt>(adj)->getZExtValue();
    offset <<= 1;
    adj = llvm::ConstantInt::get(adj->getType(), offset);
  }

  llvm::Constant *srcAdj = llvm::ConstantExpr::getExtractValue(src, 1);
  llvm::Constant *dstAdj;
  if (isDerivedToBase)
    dstAdj = llvm::ConstantExpr::getNSWSub(srcAdj, adj);
  else
    dstAdj = llvm::ConstantExpr::getNSWAdd(srcAdj, adj);

  return llvm::ConstantExpr::getInsertValue(src, dstAdj, 1);
}

llvm::Constant *
ItaniumCXXABI::EmitNullMemberPointer(const MemberPointerType *MPT) {
  // Itanium C++ ABI 2.3:
  //   A NULL pointer is represented as -1.
  if (MPT->isMemberDataPointer()) 
    return llvm::ConstantInt::get(CGM.PtrDiffTy, -1ULL, /*isSigned=*/true);

  llvm::Constant *Zero = llvm::ConstantInt::get(CGM.PtrDiffTy, 0);
  llvm::Constant *Values[2] = { Zero, Zero };
  return llvm::ConstantStruct::getAnon(Values);
}

llvm::Constant *
ItaniumCXXABI::EmitMemberDataPointer(const MemberPointerType *MPT,
                                     CharUnits offset) {
  // Itanium C++ ABI 2.3:
  //   A pointer to data member is an offset from the base address of
  //   the class object containing it, represented as a ptrdiff_t
  return llvm::ConstantInt::get(CGM.PtrDiffTy, offset.getQuantity());
}

llvm::Constant *ItaniumCXXABI::EmitMemberPointer(const CXXMethodDecl *MD) {
  return BuildMemberPointer(MD, CharUnits::Zero());
}

llvm::Constant *ItaniumCXXABI::BuildMemberPointer(const CXXMethodDecl *MD,
                                                  CharUnits ThisAdjustment) {
  assert(MD->isInstance() && "Member function must not be static!");
  MD = MD->getCanonicalDecl();

  CodeGenTypes &Types = CGM.getTypes();

  // Get the function pointer (or index if this is a virtual function).
  llvm::Constant *MemPtr[2];
  if (MD->isVirtual()) {
    uint64_t Index = CGM.getItaniumVTableContext().getMethodVTableIndex(MD);

    const ASTContext &Context = getContext();
    CharUnits PointerWidth =
      Context.toCharUnitsFromBits(Context.getTargetInfo().getPointerWidth(0));
    uint64_t VTableOffset = (Index * PointerWidth.getQuantity());

    if (UseARMMethodPtrABI) {
      // ARM C++ ABI 3.2.1:
      //   This ABI specifies that adj contains twice the this
      //   adjustment, plus 1 if the member function is virtual. The
      //   least significant bit of adj then makes exactly the same
      //   discrimination as the least significant bit of ptr does for
      //   Itanium.
      MemPtr[0] = llvm::ConstantInt::get(CGM.PtrDiffTy, VTableOffset);
      MemPtr[1] = llvm::ConstantInt::get(CGM.PtrDiffTy,
                                         2 * ThisAdjustment.getQuantity() + 1);
    } else {
      // Itanium C++ ABI 2.3:
      //   For a virtual function, [the pointer field] is 1 plus the
      //   virtual table offset (in bytes) of the function,
      //   represented as a ptrdiff_t.
      MemPtr[0] = llvm::ConstantInt::get(CGM.PtrDiffTy, VTableOffset + 1);
      MemPtr[1] = llvm::ConstantInt::get(CGM.PtrDiffTy,
                                         ThisAdjustment.getQuantity());
    }
  } else {
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
    llvm::Constant *addr = CGM.GetAddrOfFunction(MD, Ty);

    MemPtr[0] = llvm::ConstantExpr::getPtrToInt(addr, CGM.PtrDiffTy);
    MemPtr[1] = llvm::ConstantInt::get(CGM.PtrDiffTy,
                                       (UseARMMethodPtrABI ? 2 : 1) *
                                       ThisAdjustment.getQuantity());
  }
  
  return llvm::ConstantStruct::getAnon(MemPtr);
}

llvm::Constant *ItaniumCXXABI::EmitMemberPointer(const APValue &MP,
                                                 QualType MPType) {
  const MemberPointerType *MPT = MPType->castAs<MemberPointerType>();
  const ValueDecl *MPD = MP.getMemberPointerDecl();
  if (!MPD)
    return EmitNullMemberPointer(MPT);

  CharUnits ThisAdjustment = getMemberPointerPathAdjustment(MP);

  if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(MPD))
    return BuildMemberPointer(MD, ThisAdjustment);

  CharUnits FieldOffset =
    getContext().toCharUnitsFromBits(getContext().getFieldOffset(MPD));
  return EmitMemberDataPointer(MPT, ThisAdjustment + FieldOffset);
}

/// The comparison algorithm is pretty easy: the member pointers are
/// the same if they're either bitwise identical *or* both null.
///
/// ARM is different here only because null-ness is more complicated.
llvm::Value *
ItaniumCXXABI::EmitMemberPointerComparison(CodeGenFunction &CGF,
                                           llvm::Value *L,
                                           llvm::Value *R,
                                           const MemberPointerType *MPT,
                                           bool Inequality) {
  CGBuilderTy &Builder = CGF.Builder;

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

  // Member data pointers are easy because there's a unique null
  // value, so it just comes down to bitwise equality.
  if (MPT->isMemberDataPointer())
    return Builder.CreateICmp(Eq, L, R);

  // For member function pointers, the tautologies are more complex.
  // The Itanium tautology is:
  //   (L == R) <==> (L.ptr == R.ptr && (L.ptr == 0 || L.adj == R.adj))
  // The ARM tautology is:
  //   (L == R) <==> (L.ptr == R.ptr &&
  //                  (L.adj == R.adj ||
  //                   (L.ptr == 0 && ((L.adj|R.adj) & 1) == 0)))
  // The inequality tautologies have exactly the same structure, except
  // applying De Morgan's laws.
  
  llvm::Value *LPtr = Builder.CreateExtractValue(L, 0, "lhs.memptr.ptr");
  llvm::Value *RPtr = Builder.CreateExtractValue(R, 0, "rhs.memptr.ptr");

  // This condition tests whether L.ptr == R.ptr.  This must always be
  // true for equality to hold.
  llvm::Value *PtrEq = Builder.CreateICmp(Eq, LPtr, RPtr, "cmp.ptr");

  // This condition, together with the assumption that L.ptr == R.ptr,
  // tests whether the pointers are both null.  ARM imposes an extra
  // condition.
  llvm::Value *Zero = llvm::Constant::getNullValue(LPtr->getType());
  llvm::Value *EqZero = Builder.CreateICmp(Eq, LPtr, Zero, "cmp.ptr.null");

  // This condition tests whether L.adj == R.adj.  If this isn't
  // true, the pointers are unequal unless they're both null.
  llvm::Value *LAdj = Builder.CreateExtractValue(L, 1, "lhs.memptr.adj");
  llvm::Value *RAdj = Builder.CreateExtractValue(R, 1, "rhs.memptr.adj");
  llvm::Value *AdjEq = Builder.CreateICmp(Eq, LAdj, RAdj, "cmp.adj");

  // Null member function pointers on ARM clear the low bit of Adj,
  // so the zero condition has to check that neither low bit is set.
  if (UseARMMethodPtrABI) {
    llvm::Value *One = llvm::ConstantInt::get(LPtr->getType(), 1);

    // Compute (l.adj | r.adj) & 1 and test it against zero.
    llvm::Value *OrAdj = Builder.CreateOr(LAdj, RAdj, "or.adj");
    llvm::Value *OrAdjAnd1 = Builder.CreateAnd(OrAdj, One);
    llvm::Value *OrAdjAnd1EqZero = Builder.CreateICmp(Eq, OrAdjAnd1, Zero,
                                                      "cmp.or.adj");
    EqZero = Builder.CreateBinOp(And, EqZero, OrAdjAnd1EqZero);
  }

  // Tie together all our conditions.
  llvm::Value *Result = Builder.CreateBinOp(Or, EqZero, AdjEq);
  Result = Builder.CreateBinOp(And, PtrEq, Result,
                               Inequality ? "memptr.ne" : "memptr.eq");
  return Result;
}

llvm::Value *
ItaniumCXXABI::EmitMemberPointerIsNotNull(CodeGenFunction &CGF,
                                          llvm::Value *MemPtr,
                                          const MemberPointerType *MPT) {
  CGBuilderTy &Builder = CGF.Builder;

  /// For member data pointers, this is just a check against -1.
  if (MPT->isMemberDataPointer()) {
    assert(MemPtr->getType() == CGM.PtrDiffTy);
    llvm::Value *NegativeOne =
      llvm::Constant::getAllOnesValue(MemPtr->getType());
    return Builder.CreateICmpNE(MemPtr, NegativeOne, "memptr.tobool");
  }
  
  // In Itanium, a member function pointer is not null if 'ptr' is not null.
  llvm::Value *Ptr = Builder.CreateExtractValue(MemPtr, 0, "memptr.ptr");

  llvm::Constant *Zero = llvm::ConstantInt::get(Ptr->getType(), 0);
  llvm::Value *Result = Builder.CreateICmpNE(Ptr, Zero, "memptr.tobool");

  // On ARM, a member function pointer is also non-null if the low bit of 'adj'
  // (the virtual bit) is set.
  if (UseARMMethodPtrABI) {
    llvm::Constant *One = llvm::ConstantInt::get(Ptr->getType(), 1);
    llvm::Value *Adj = Builder.CreateExtractValue(MemPtr, 1, "memptr.adj");
    llvm::Value *VirtualBit = Builder.CreateAnd(Adj, One, "memptr.virtualbit");
    llvm::Value *IsVirtual = Builder.CreateICmpNE(VirtualBit, Zero,
                                                  "memptr.isvirtual");
    Result = Builder.CreateOr(Result, IsVirtual);
  }

  return Result;
}

bool ItaniumCXXABI::classifyReturnType(CGFunctionInfo &FI) const {
  const CXXRecordDecl *RD = FI.getReturnType()->getAsCXXRecordDecl();
  if (!RD)
    return false;

  // Return indirectly if we have a non-trivial copy ctor or non-trivial dtor.
  // FIXME: Use canCopyArgument() when it is fixed to handle lazily declared
  // special members.
  if (RD->hasNonTrivialDestructor() || RD->hasNonTrivialCopyConstructor()) {
    FI.getReturnInfo() = ABIArgInfo::getIndirect(0, /*ByVal=*/false);
    return true;
  }
  return false;
}

/// The Itanium ABI requires non-zero initialization only for data
/// member pointers, for which '0' is a valid offset.
bool ItaniumCXXABI::isZeroInitializable(const MemberPointerType *MPT) {
  return MPT->getPointeeType()->isFunctionType();
}

/// The Itanium ABI always places an offset to the complete object
/// at entry -2 in the vtable.
void ItaniumCXXABI::emitVirtualObjectDelete(CodeGenFunction &CGF,
                                            const CXXDeleteExpr *DE,
                                            llvm::Value *Ptr,
                                            QualType ElementType,
                                            const CXXDestructorDecl *Dtor) {
  bool UseGlobalDelete = DE->isGlobalDelete();
  if (UseGlobalDelete) {
    // Derive the complete-object pointer, which is what we need
    // to pass to the deallocation function.

    // Grab the vtable pointer as an intptr_t*.
    llvm::Value *VTable = CGF.GetVTablePtr(Ptr, CGF.IntPtrTy->getPointerTo());

    // Track back to entry -2 and pull out the offset there.
    llvm::Value *OffsetPtr = CGF.Builder.CreateConstInBoundsGEP1_64(
        VTable, -2, "complete-offset.ptr");
    llvm::LoadInst *Offset = CGF.Builder.CreateLoad(OffsetPtr);
    Offset->setAlignment(CGF.PointerAlignInBytes);

    // Apply the offset.
    llvm::Value *CompletePtr = CGF.Builder.CreateBitCast(Ptr, CGF.Int8PtrTy);
    CompletePtr = CGF.Builder.CreateInBoundsGEP(CompletePtr, Offset);

    // If we're supposed to call the global delete, make sure we do so
    // even if the destructor throws.
    CGF.pushCallObjectDeleteCleanup(DE->getOperatorDelete(), CompletePtr,
                                    ElementType);
  }

  // FIXME: Provide a source location here even though there's no
  // CXXMemberCallExpr for dtor call.
  CXXDtorType DtorType = UseGlobalDelete ? Dtor_Complete : Dtor_Deleting;
  EmitVirtualDestructorCall(CGF, Dtor, DtorType, Ptr, /*CE=*/nullptr);

  if (UseGlobalDelete)
    CGF.PopCleanupBlock();
}

void ItaniumCXXABI::emitRethrow(CodeGenFunction &CGF, bool isNoReturn) {
  // void __cxa_rethrow();

  llvm::FunctionType *FTy =
    llvm::FunctionType::get(CGM.VoidTy, /*IsVarArgs=*/false);

  llvm::Constant *Fn = CGM.CreateRuntimeFunction(FTy, "__cxa_rethrow");

  if (isNoReturn)
    CGF.EmitNoreturnRuntimeCallOrInvoke(Fn, None);
  else
    CGF.EmitRuntimeCallOrInvoke(Fn);
}

static llvm::Constant *getItaniumDynamicCastFn(CodeGenFunction &CGF) {
  // void *__dynamic_cast(const void *sub,
  //                      const abi::__class_type_info *src,
  //                      const abi::__class_type_info *dst,
  //                      std::ptrdiff_t src2dst_offset);
  
  llvm::Type *Int8PtrTy = CGF.Int8PtrTy;
  llvm::Type *PtrDiffTy = 
    CGF.ConvertType(CGF.getContext().getPointerDiffType());

  llvm::Type *Args[4] = { Int8PtrTy, Int8PtrTy, Int8PtrTy, PtrDiffTy };

  llvm::FunctionType *FTy = llvm::FunctionType::get(Int8PtrTy, Args, false);

  // Mark the function as nounwind readonly.
  llvm::Attribute::AttrKind FuncAttrs[] = { llvm::Attribute::NoUnwind,
                                            llvm::Attribute::ReadOnly };
  llvm::AttributeSet Attrs = llvm::AttributeSet::get(
      CGF.getLLVMContext(), llvm::AttributeSet::FunctionIndex, FuncAttrs);

  return CGF.CGM.CreateRuntimeFunction(FTy, "__dynamic_cast", Attrs);
}

static llvm::Constant *getBadCastFn(CodeGenFunction &CGF) {
  // void __cxa_bad_cast();
  llvm::FunctionType *FTy = llvm::FunctionType::get(CGF.VoidTy, false);
  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_bad_cast");
}

/// \brief Compute the src2dst_offset hint as described in the
/// Itanium C++ ABI [2.9.7]
static CharUnits computeOffsetHint(ASTContext &Context,
                                   const CXXRecordDecl *Src,
                                   const CXXRecordDecl *Dst) {
  CXXBasePaths Paths(/*FindAmbiguities=*/true, /*RecordPaths=*/true,
                     /*DetectVirtual=*/false);

  // If Dst is not derived from Src we can skip the whole computation below and
  // return that Src is not a public base of Dst.  Record all inheritance paths.
  if (!Dst->isDerivedFrom(Src, Paths))
    return CharUnits::fromQuantity(-2ULL);

  unsigned NumPublicPaths = 0;
  CharUnits Offset;

  // Now walk all possible inheritance paths.
  for (CXXBasePaths::paths_iterator I = Paths.begin(), E = Paths.end(); I != E;
       ++I) {
    if (I->Access != AS_public) // Ignore non-public inheritance.
      continue;

    ++NumPublicPaths;

    for (CXXBasePath::iterator J = I->begin(), JE = I->end(); J != JE; ++J) {
      // If the path contains a virtual base class we can't give any hint.
      // -1: no hint.
      if (J->Base->isVirtual())
        return CharUnits::fromQuantity(-1ULL);

      if (NumPublicPaths > 1) // Won't use offsets, skip computation.
        continue;

      // Accumulate the base class offsets.
      const ASTRecordLayout &L = Context.getASTRecordLayout(J->Class);
      Offset += L.getBaseClassOffset(J->Base->getType()->getAsCXXRecordDecl());
    }
  }

  // -2: Src is not a public base of Dst.
  if (NumPublicPaths == 0)
    return CharUnits::fromQuantity(-2ULL);

  // -3: Src is a multiple public base type but never a virtual base type.
  if (NumPublicPaths > 1)
    return CharUnits::fromQuantity(-3ULL);

  // Otherwise, the Src type is a unique public nonvirtual base type of Dst.
  // Return the offset of Src from the origin of Dst.
  return Offset;
}

static llvm::Constant *getBadTypeidFn(CodeGenFunction &CGF) {
  // void __cxa_bad_typeid();
  llvm::FunctionType *FTy = llvm::FunctionType::get(CGF.VoidTy, false);

  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_bad_typeid");
}

bool ItaniumCXXABI::shouldTypeidBeNullChecked(bool IsDeref,
                                              QualType SrcRecordTy) {
  return IsDeref;
}

void ItaniumCXXABI::EmitBadTypeidCall(CodeGenFunction &CGF) {
  llvm::Value *Fn = getBadTypeidFn(CGF);
  CGF.EmitRuntimeCallOrInvoke(Fn).setDoesNotReturn();
  CGF.Builder.CreateUnreachable();
}

llvm::Value *ItaniumCXXABI::EmitTypeid(CodeGenFunction &CGF,
                                       QualType SrcRecordTy,
                                       llvm::Value *ThisPtr,
                                       llvm::Type *StdTypeInfoPtrTy) {
  llvm::Value *Value =
      CGF.GetVTablePtr(ThisPtr, StdTypeInfoPtrTy->getPointerTo());

  // Load the type info.
  Value = CGF.Builder.CreateConstInBoundsGEP1_64(Value, -1ULL);
  return CGF.Builder.CreateLoad(Value);
}

bool ItaniumCXXABI::shouldDynamicCastCallBeNullChecked(bool SrcIsPtr,
                                                       QualType SrcRecordTy) {
  return SrcIsPtr;
}

llvm::Value *ItaniumCXXABI::EmitDynamicCastCall(
    CodeGenFunction &CGF, llvm::Value *Value, QualType SrcRecordTy,
    QualType DestTy, QualType DestRecordTy, llvm::BasicBlock *CastEnd) {
  llvm::Type *PtrDiffLTy =
      CGF.ConvertType(CGF.getContext().getPointerDiffType());
  llvm::Type *DestLTy = CGF.ConvertType(DestTy);

  llvm::Value *SrcRTTI =
      CGF.CGM.GetAddrOfRTTIDescriptor(SrcRecordTy.getUnqualifiedType());
  llvm::Value *DestRTTI =
      CGF.CGM.GetAddrOfRTTIDescriptor(DestRecordTy.getUnqualifiedType());

  // Compute the offset hint.
  const CXXRecordDecl *SrcDecl = SrcRecordTy->getAsCXXRecordDecl();
  const CXXRecordDecl *DestDecl = DestRecordTy->getAsCXXRecordDecl();
  llvm::Value *OffsetHint = llvm::ConstantInt::get(
      PtrDiffLTy,
      computeOffsetHint(CGF.getContext(), SrcDecl, DestDecl).getQuantity());

  // Emit the call to __dynamic_cast.
  Value = CGF.EmitCastToVoidPtr(Value);

  llvm::Value *args[] = {Value, SrcRTTI, DestRTTI, OffsetHint};
  Value = CGF.EmitNounwindRuntimeCall(getItaniumDynamicCastFn(CGF), args);
  Value = CGF.Builder.CreateBitCast(Value, DestLTy);

  /// C++ [expr.dynamic.cast]p9:
  ///   A failed cast to reference type throws std::bad_cast
  if (DestTy->isReferenceType()) {
    llvm::BasicBlock *BadCastBlock =
        CGF.createBasicBlock("dynamic_cast.bad_cast");

    llvm::Value *IsNull = CGF.Builder.CreateIsNull(Value);
    CGF.Builder.CreateCondBr(IsNull, BadCastBlock, CastEnd);

    CGF.EmitBlock(BadCastBlock);
    EmitBadCastCall(CGF);
  }

  return Value;
}

llvm::Value *ItaniumCXXABI::EmitDynamicCastToVoid(CodeGenFunction &CGF,
                                                  llvm::Value *Value,
                                                  QualType SrcRecordTy,
                                                  QualType DestTy) {
  llvm::Type *PtrDiffLTy =
      CGF.ConvertType(CGF.getContext().getPointerDiffType());
  llvm::Type *DestLTy = CGF.ConvertType(DestTy);

  // Get the vtable pointer.
  llvm::Value *VTable = CGF.GetVTablePtr(Value, PtrDiffLTy->getPointerTo());

  // Get the offset-to-top from the vtable.
  llvm::Value *OffsetToTop =
      CGF.Builder.CreateConstInBoundsGEP1_64(VTable, -2ULL);
  OffsetToTop = CGF.Builder.CreateLoad(OffsetToTop, "offset.to.top");

  // Finally, add the offset to the pointer.
  Value = CGF.EmitCastToVoidPtr(Value);
  Value = CGF.Builder.CreateInBoundsGEP(Value, OffsetToTop);

  return CGF.Builder.CreateBitCast(Value, DestLTy);
}

bool ItaniumCXXABI::EmitBadCastCall(CodeGenFunction &CGF) {
  llvm::Value *Fn = getBadCastFn(CGF);
  CGF.EmitRuntimeCallOrInvoke(Fn).setDoesNotReturn();
  CGF.Builder.CreateUnreachable();
  return true;
}

llvm::Value *
ItaniumCXXABI::GetVirtualBaseClassOffset(CodeGenFunction &CGF,
                                         llvm::Value *This,
                                         const CXXRecordDecl *ClassDecl,
                                         const CXXRecordDecl *BaseClassDecl) {
  llvm::Value *VTablePtr = CGF.GetVTablePtr(This, CGM.Int8PtrTy);
  CharUnits VBaseOffsetOffset =
      CGM.getItaniumVTableContext().getVirtualBaseOffsetOffset(ClassDecl,
                                                               BaseClassDecl);

  llvm::Value *VBaseOffsetPtr =
    CGF.Builder.CreateConstGEP1_64(VTablePtr, VBaseOffsetOffset.getQuantity(),
                                   "vbase.offset.ptr");
  VBaseOffsetPtr = CGF.Builder.CreateBitCast(VBaseOffsetPtr,
                                             CGM.PtrDiffTy->getPointerTo());

  llvm::Value *VBaseOffset =
    CGF.Builder.CreateLoad(VBaseOffsetPtr, "vbase.offset");

  return VBaseOffset;
}

void ItaniumCXXABI::EmitCXXConstructors(const CXXConstructorDecl *D) {
  // Just make sure we're in sync with TargetCXXABI.
  assert(CGM.getTarget().getCXXABI().hasConstructorVariants());

  // The constructor used for constructing this as a base class;
  // ignores virtual bases.
  CGM.EmitGlobal(GlobalDecl(D, Ctor_Base));

  // The constructor used for constructing this as a complete class;
  // constructs the virtual bases, then calls the base constructor.
  if (!D->getParent()->isAbstract()) {
    // We don't need to emit the complete ctor if the class is abstract.
    CGM.EmitGlobal(GlobalDecl(D, Ctor_Complete));
  }
}

void
ItaniumCXXABI::buildStructorSignature(const CXXMethodDecl *MD, StructorType T,
                                      SmallVectorImpl<CanQualType> &ArgTys) {
  ASTContext &Context = getContext();

  // All parameters are already in place except VTT, which goes after 'this'.
  // These are Clang types, so we don't need to worry about sret yet.

  // Check if we need to add a VTT parameter (which has type void **).
  if (T == StructorType::Base && MD->getParent()->getNumVBases() != 0)
    ArgTys.insert(ArgTys.begin() + 1,
                  Context.getPointerType(Context.VoidPtrTy));
}

void ItaniumCXXABI::EmitCXXDestructors(const CXXDestructorDecl *D) {
  // The destructor used for destructing this as a base class; ignores
  // virtual bases.
  CGM.EmitGlobal(GlobalDecl(D, Dtor_Base));

  // The destructor used for destructing this as a most-derived class;
  // call the base destructor and then destructs any virtual bases.
  CGM.EmitGlobal(GlobalDecl(D, Dtor_Complete));

  // The destructor in a virtual table is always a 'deleting'
  // destructor, which calls the complete destructor and then uses the
  // appropriate operator delete.
  if (D->isVirtual())
    CGM.EmitGlobal(GlobalDecl(D, Dtor_Deleting));
}

void ItaniumCXXABI::addImplicitStructorParams(CodeGenFunction &CGF,
                                              QualType &ResTy,
                                              FunctionArgList &Params) {
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(CGF.CurGD.getDecl());
  assert(isa<CXXConstructorDecl>(MD) || isa<CXXDestructorDecl>(MD));

  // Check if we need a VTT parameter as well.
  if (NeedsVTTParameter(CGF.CurGD)) {
    ASTContext &Context = getContext();

    // FIXME: avoid the fake decl
    QualType T = Context.getPointerType(Context.VoidPtrTy);
    ImplicitParamDecl *VTTDecl
      = ImplicitParamDecl::Create(Context, nullptr, MD->getLocation(),
                                  &Context.Idents.get("vtt"), T);
    Params.insert(Params.begin() + 1, VTTDecl);
    getStructorImplicitParamDecl(CGF) = VTTDecl;
  }
}

void ItaniumCXXABI::EmitInstanceFunctionProlog(CodeGenFunction &CGF) {
  /// Initialize the 'this' slot.
  EmitThisParam(CGF);

  /// Initialize the 'vtt' slot if needed.
  if (getStructorImplicitParamDecl(CGF)) {
    getStructorImplicitParamValue(CGF) = CGF.Builder.CreateLoad(
        CGF.GetAddrOfLocalVar(getStructorImplicitParamDecl(CGF)), "vtt");
  }

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
}

unsigned ItaniumCXXABI::addImplicitConstructorArgs(
    CodeGenFunction &CGF, const CXXConstructorDecl *D, CXXCtorType Type,
    bool ForVirtualBase, bool Delegating, CallArgList &Args) {
  if (!NeedsVTTParameter(GlobalDecl(D, Type)))
    return 0;

  // Insert the implicit 'vtt' argument as the second argument.
  llvm::Value *VTT =
      CGF.GetVTTParameter(GlobalDecl(D, Type), ForVirtualBase, Delegating);
  QualType VTTTy = getContext().getPointerType(getContext().VoidPtrTy);
  Args.insert(Args.begin() + 1,
              CallArg(RValue::get(VTT), VTTTy, /*needscopy=*/false));
  return 1;  // Added one arg.
}

void ItaniumCXXABI::EmitDestructorCall(CodeGenFunction &CGF,
                                       const CXXDestructorDecl *DD,
                                       CXXDtorType Type, bool ForVirtualBase,
                                       bool Delegating, llvm::Value *This) {
  GlobalDecl GD(DD, Type);
  llvm::Value *VTT = CGF.GetVTTParameter(GD, ForVirtualBase, Delegating);
  QualType VTTTy = getContext().getPointerType(getContext().VoidPtrTy);

  llvm::Value *Callee = nullptr;
  if (getContext().getLangOpts().AppleKext)
    Callee = CGF.BuildAppleKextVirtualDestructorCall(DD, Type, DD->getParent());

  if (!Callee)
    Callee = CGM.getAddrOfCXXStructor(DD, getFromDtorType(Type));

  CGF.EmitCXXMemberOrOperatorCall(DD, Callee, ReturnValueSlot(), This, VTT,
                                  VTTTy, nullptr);
}

void ItaniumCXXABI::emitVTableDefinitions(CodeGenVTables &CGVT,
                                          const CXXRecordDecl *RD) {
  llvm::GlobalVariable *VTable = getAddrOfVTable(RD, CharUnits());
  if (VTable->hasInitializer())
    return;

  ItaniumVTableContext &VTContext = CGM.getItaniumVTableContext();
  const VTableLayout &VTLayout = VTContext.getVTableLayout(RD);
  llvm::GlobalVariable::LinkageTypes Linkage = CGM.getVTableLinkage(RD);
  llvm::Constant *RTTI =
      CGM.GetAddrOfRTTIDescriptor(CGM.getContext().getTagDeclType(RD));

  // Create and set the initializer.
  llvm::Constant *Init = CGVT.CreateVTableInitializer(
      RD, VTLayout.vtable_component_begin(), VTLayout.getNumVTableComponents(),
      VTLayout.vtable_thunk_begin(), VTLayout.getNumVTableThunks(), RTTI);
  VTable->setInitializer(Init);

  // Set the correct linkage.
  VTable->setLinkage(Linkage);

  // Set the right visibility.
  CGM.setGlobalVisibility(VTable, RD);

  // Use pointer alignment for the vtable. Otherwise we would align them based
  // on the size of the initializer which doesn't make sense as only single
  // values are read.
  unsigned PAlign = CGM.getTarget().getPointerAlign(0);
  VTable->setAlignment(getContext().toCharUnitsFromBits(PAlign).getQuantity());

  // If this is the magic class __cxxabiv1::__fundamental_type_info,
  // we will emit the typeinfo for the fundamental types. This is the
  // same behaviour as GCC.
  const DeclContext *DC = RD->getDeclContext();
  if (RD->getIdentifier() &&
      RD->getIdentifier()->isStr("__fundamental_type_info") &&
      isa<NamespaceDecl>(DC) && cast<NamespaceDecl>(DC)->getIdentifier() &&
      cast<NamespaceDecl>(DC)->getIdentifier()->isStr("__cxxabiv1") &&
      DC->getParent()->isTranslationUnit())
    EmitFundamentalRTTIDescriptors();
}

llvm::Value *ItaniumCXXABI::getVTableAddressPointInStructor(
    CodeGenFunction &CGF, const CXXRecordDecl *VTableClass, BaseSubobject Base,
    const CXXRecordDecl *NearestVBase, bool &NeedsVirtualOffset) {
  bool NeedsVTTParam = CGM.getCXXABI().NeedsVTTParameter(CGF.CurGD);
  NeedsVirtualOffset = (NeedsVTTParam && NearestVBase);

  llvm::Value *VTableAddressPoint;
  if (NeedsVTTParam && (Base.getBase()->getNumVBases() || NearestVBase)) {
    // Get the secondary vpointer index.
    uint64_t VirtualPointerIndex =
        CGM.getVTables().getSecondaryVirtualPointerIndex(VTableClass, Base);

    /// Load the VTT.
    llvm::Value *VTT = CGF.LoadCXXVTT();
    if (VirtualPointerIndex)
      VTT = CGF.Builder.CreateConstInBoundsGEP1_64(VTT, VirtualPointerIndex);

    // And load the address point from the VTT.
    VTableAddressPoint = CGF.Builder.CreateLoad(VTT);
  } else {
    llvm::Constant *VTable =
        CGM.getCXXABI().getAddrOfVTable(VTableClass, CharUnits());
    uint64_t AddressPoint = CGM.getItaniumVTableContext()
                                .getVTableLayout(VTableClass)
                                .getAddressPoint(Base);
    VTableAddressPoint =
        CGF.Builder.CreateConstInBoundsGEP2_64(VTable, 0, AddressPoint);
  }

  return VTableAddressPoint;
}

llvm::Constant *ItaniumCXXABI::getVTableAddressPointForConstExpr(
    BaseSubobject Base, const CXXRecordDecl *VTableClass) {
  llvm::Constant *VTable = getAddrOfVTable(VTableClass, CharUnits());

  // Find the appropriate vtable within the vtable group.
  uint64_t AddressPoint = CGM.getItaniumVTableContext()
                              .getVTableLayout(VTableClass)
                              .getAddressPoint(Base);
  llvm::Value *Indices[] = {
    llvm::ConstantInt::get(CGM.Int64Ty, 0),
    llvm::ConstantInt::get(CGM.Int64Ty, AddressPoint)
  };

  return llvm::ConstantExpr::getInBoundsGetElementPtr(VTable, Indices);
}

llvm::GlobalVariable *ItaniumCXXABI::getAddrOfVTable(const CXXRecordDecl *RD,
                                                     CharUnits VPtrOffset) {
  assert(VPtrOffset.isZero() && "Itanium ABI only supports zero vptr offsets");

  llvm::GlobalVariable *&VTable = VTables[RD];
  if (VTable)
    return VTable;

  // Queue up this v-table for possible deferred emission.
  CGM.addDeferredVTable(RD);

  SmallString<256> OutName;
  llvm::raw_svector_ostream Out(OutName);
  getMangleContext().mangleCXXVTable(RD, Out);
  Out.flush();
  StringRef Name = OutName.str();

  ItaniumVTableContext &VTContext = CGM.getItaniumVTableContext();
  llvm::ArrayType *ArrayType = llvm::ArrayType::get(
      CGM.Int8PtrTy, VTContext.getVTableLayout(RD).getNumVTableComponents());

  VTable = CGM.CreateOrReplaceCXXRuntimeVariable(
      Name, ArrayType, llvm::GlobalValue::ExternalLinkage);
  VTable->setUnnamedAddr(true);

  if (RD->hasAttr<DLLImportAttr>())
    VTable->setDLLStorageClass(llvm::GlobalValue::DLLImportStorageClass);
  else if (RD->hasAttr<DLLExportAttr>())
    VTable->setDLLStorageClass(llvm::GlobalValue::DLLExportStorageClass);

  return VTable;
}

llvm::Value *ItaniumCXXABI::getVirtualFunctionPointer(CodeGenFunction &CGF,
                                                      GlobalDecl GD,
                                                      llvm::Value *This,
                                                      llvm::Type *Ty) {
  GD = GD.getCanonicalDecl();
  Ty = Ty->getPointerTo()->getPointerTo();
  llvm::Value *VTable = CGF.GetVTablePtr(This, Ty);

  uint64_t VTableIndex = CGM.getItaniumVTableContext().getMethodVTableIndex(GD);
  llvm::Value *VFuncPtr =
      CGF.Builder.CreateConstInBoundsGEP1_64(VTable, VTableIndex, "vfn");
  return CGF.Builder.CreateLoad(VFuncPtr);
}

llvm::Value *ItaniumCXXABI::EmitVirtualDestructorCall(
    CodeGenFunction &CGF, const CXXDestructorDecl *Dtor, CXXDtorType DtorType,
    llvm::Value *This, const CXXMemberCallExpr *CE) {
  assert(CE == nullptr || CE->arg_begin() == CE->arg_end());
  assert(DtorType == Dtor_Deleting || DtorType == Dtor_Complete);

  const CGFunctionInfo *FInfo = &CGM.getTypes().arrangeCXXStructorDeclaration(
      Dtor, getFromDtorType(DtorType));
  llvm::Type *Ty = CGF.CGM.getTypes().GetFunctionType(*FInfo);
  llvm::Value *Callee =
      getVirtualFunctionPointer(CGF, GlobalDecl(Dtor, DtorType), This, Ty);

  CGF.EmitCXXMemberOrOperatorCall(Dtor, Callee, ReturnValueSlot(), This,
                                  /*ImplicitParam=*/nullptr, QualType(), CE);
  return nullptr;
}

void ItaniumCXXABI::emitVirtualInheritanceTables(const CXXRecordDecl *RD) {
  CodeGenVTables &VTables = CGM.getVTables();
  llvm::GlobalVariable *VTT = VTables.GetAddrOfVTT(RD);
  VTables.EmitVTTDefinition(VTT, CGM.getVTableLinkage(RD), RD);
}

static llvm::Value *performTypeAdjustment(CodeGenFunction &CGF,
                                          llvm::Value *Ptr,
                                          int64_t NonVirtualAdjustment,
                                          int64_t VirtualAdjustment,
                                          bool IsReturnAdjustment) {
  if (!NonVirtualAdjustment && !VirtualAdjustment)
    return Ptr;

  llvm::Type *Int8PtrTy = CGF.Int8PtrTy;
  llvm::Value *V = CGF.Builder.CreateBitCast(Ptr, Int8PtrTy);

  if (NonVirtualAdjustment && !IsReturnAdjustment) {
    // Perform the non-virtual adjustment for a base-to-derived cast.
    V = CGF.Builder.CreateConstInBoundsGEP1_64(V, NonVirtualAdjustment);
  }

  if (VirtualAdjustment) {
    llvm::Type *PtrDiffTy =
        CGF.ConvertType(CGF.getContext().getPointerDiffType());

    // Perform the virtual adjustment.
    llvm::Value *VTablePtrPtr =
        CGF.Builder.CreateBitCast(V, Int8PtrTy->getPointerTo());

    llvm::Value *VTablePtr = CGF.Builder.CreateLoad(VTablePtrPtr);

    llvm::Value *OffsetPtr =
        CGF.Builder.CreateConstInBoundsGEP1_64(VTablePtr, VirtualAdjustment);

    OffsetPtr = CGF.Builder.CreateBitCast(OffsetPtr, PtrDiffTy->getPointerTo());

    // Load the adjustment offset from the vtable.
    llvm::Value *Offset = CGF.Builder.CreateLoad(OffsetPtr);

    // Adjust our pointer.
    V = CGF.Builder.CreateInBoundsGEP(V, Offset);
  }

  if (NonVirtualAdjustment && IsReturnAdjustment) {
    // Perform the non-virtual adjustment for a derived-to-base cast.
    V = CGF.Builder.CreateConstInBoundsGEP1_64(V, NonVirtualAdjustment);
  }

  // Cast back to the original type.
  return CGF.Builder.CreateBitCast(V, Ptr->getType());
}

llvm::Value *ItaniumCXXABI::performThisAdjustment(CodeGenFunction &CGF,
                                                  llvm::Value *This,
                                                  const ThisAdjustment &TA) {
  return performTypeAdjustment(CGF, This, TA.NonVirtual,
                               TA.Virtual.Itanium.VCallOffsetOffset,
                               /*IsReturnAdjustment=*/false);
}

llvm::Value *
ItaniumCXXABI::performReturnAdjustment(CodeGenFunction &CGF, llvm::Value *Ret,
                                       const ReturnAdjustment &RA) {
  return performTypeAdjustment(CGF, Ret, RA.NonVirtual,
                               RA.Virtual.Itanium.VBaseOffsetOffset,
                               /*IsReturnAdjustment=*/true);
}

void ARMCXXABI::EmitReturnFromThunk(CodeGenFunction &CGF,
                                    RValue RV, QualType ResultType) {
  if (!isa<CXXDestructorDecl>(CGF.CurGD.getDecl()))
    return ItaniumCXXABI::EmitReturnFromThunk(CGF, RV, ResultType);

  // Destructor thunks in the ARM ABI have indeterminate results.
  llvm::Type *T =
    cast<llvm::PointerType>(CGF.ReturnValue->getType())->getElementType();
  RValue Undef = RValue::get(llvm::UndefValue::get(T));
  return ItaniumCXXABI::EmitReturnFromThunk(CGF, Undef, ResultType);
}

/************************** Array allocation cookies **************************/

CharUnits ItaniumCXXABI::getArrayCookieSizeImpl(QualType elementType) {
  // The array cookie is a size_t; pad that up to the element alignment.
  // The cookie is actually right-justified in that space.
  return std::max(CharUnits::fromQuantity(CGM.SizeSizeInBytes),
                  CGM.getContext().getTypeAlignInChars(elementType));
}

llvm::Value *ItaniumCXXABI::InitializeArrayCookie(CodeGenFunction &CGF,
                                                  llvm::Value *NewPtr,
                                                  llvm::Value *NumElements,
                                                  const CXXNewExpr *expr,
                                                  QualType ElementType) {
  assert(requiresArrayCookie(expr));

  unsigned AS = NewPtr->getType()->getPointerAddressSpace();

  ASTContext &Ctx = getContext();
  QualType SizeTy = Ctx.getSizeType();
  CharUnits SizeSize = Ctx.getTypeSizeInChars(SizeTy);

  // The size of the cookie.
  CharUnits CookieSize =
    std::max(SizeSize, Ctx.getTypeAlignInChars(ElementType));
  assert(CookieSize == getArrayCookieSizeImpl(ElementType));

  // Compute an offset to the cookie.
  llvm::Value *CookiePtr = NewPtr;
  CharUnits CookieOffset = CookieSize - SizeSize;
  if (!CookieOffset.isZero())
    CookiePtr = CGF.Builder.CreateConstInBoundsGEP1_64(CookiePtr,
                                                 CookieOffset.getQuantity());

  // Write the number of elements into the appropriate slot.
  llvm::Type *NumElementsTy = CGF.ConvertType(SizeTy)->getPointerTo(AS);
  llvm::Value *NumElementsPtr =
      CGF.Builder.CreateBitCast(CookiePtr, NumElementsTy);
  llvm::Instruction *SI = CGF.Builder.CreateStore(NumElements, NumElementsPtr);
  if (CGM.getLangOpts().Sanitize.has(SanitizerKind::Address) && AS == 0 &&
      expr->getOperatorNew()->isReplaceableGlobalAllocationFunction()) {
    // The store to the CookiePtr does not need to be instrumented.
    CGM.getSanitizerMetadata()->disableSanitizerForInstruction(SI);
    llvm::FunctionType *FTy =
        llvm::FunctionType::get(CGM.VoidTy, NumElementsTy, false);
    llvm::Constant *F =
        CGM.CreateRuntimeFunction(FTy, "__asan_poison_cxx_array_cookie");
    CGF.Builder.CreateCall(F, NumElementsPtr);
  }

  // Finally, compute a pointer to the actual data buffer by skipping
  // over the cookie completely.
  return CGF.Builder.CreateConstInBoundsGEP1_64(NewPtr,
                                                CookieSize.getQuantity());  
}

llvm::Value *ItaniumCXXABI::readArrayCookieImpl(CodeGenFunction &CGF,
                                                llvm::Value *allocPtr,
                                                CharUnits cookieSize) {
  // The element size is right-justified in the cookie.
  llvm::Value *numElementsPtr = allocPtr;
  CharUnits numElementsOffset =
    cookieSize - CharUnits::fromQuantity(CGF.SizeSizeInBytes);
  if (!numElementsOffset.isZero())
    numElementsPtr =
      CGF.Builder.CreateConstInBoundsGEP1_64(numElementsPtr,
                                             numElementsOffset.getQuantity());

  unsigned AS = allocPtr->getType()->getPointerAddressSpace();
  numElementsPtr = 
    CGF.Builder.CreateBitCast(numElementsPtr, CGF.SizeTy->getPointerTo(AS));
  if (!CGM.getLangOpts().Sanitize.has(SanitizerKind::Address) || AS != 0)
    return CGF.Builder.CreateLoad(numElementsPtr);
  // In asan mode emit a function call instead of a regular load and let the
  // run-time deal with it: if the shadow is properly poisoned return the
  // cookie, otherwise return 0 to avoid an infinite loop calling DTORs.
  // We can't simply ignore this load using nosanitize metadata because
  // the metadata may be lost.
  llvm::FunctionType *FTy =
      llvm::FunctionType::get(CGF.SizeTy, CGF.SizeTy->getPointerTo(0), false);
  llvm::Constant *F =
      CGM.CreateRuntimeFunction(FTy, "__asan_load_cxx_array_cookie");
  return CGF.Builder.CreateCall(F, numElementsPtr);
}

CharUnits ARMCXXABI::getArrayCookieSizeImpl(QualType elementType) {
  // ARM says that the cookie is always:
  //   struct array_cookie {
  //     std::size_t element_size; // element_size != 0
  //     std::size_t element_count;
  //   };
  // But the base ABI doesn't give anything an alignment greater than
  // 8, so we can dismiss this as typical ABI-author blindness to
  // actual language complexity and round up to the element alignment.
  return std::max(CharUnits::fromQuantity(2 * CGM.SizeSizeInBytes),
                  CGM.getContext().getTypeAlignInChars(elementType));
}

llvm::Value *ARMCXXABI::InitializeArrayCookie(CodeGenFunction &CGF,
                                              llvm::Value *newPtr,
                                              llvm::Value *numElements,
                                              const CXXNewExpr *expr,
                                              QualType elementType) {
  assert(requiresArrayCookie(expr));

  // NewPtr is a char*, but we generalize to arbitrary addrspaces.
  unsigned AS = newPtr->getType()->getPointerAddressSpace();

  // The cookie is always at the start of the buffer.
  llvm::Value *cookie = newPtr;

  // The first element is the element size.
  cookie = CGF.Builder.CreateBitCast(cookie, CGF.SizeTy->getPointerTo(AS));
  llvm::Value *elementSize = llvm::ConstantInt::get(CGF.SizeTy,
                 getContext().getTypeSizeInChars(elementType).getQuantity());
  CGF.Builder.CreateStore(elementSize, cookie);

  // The second element is the element count.
  cookie = CGF.Builder.CreateConstInBoundsGEP1_32(cookie, 1);
  CGF.Builder.CreateStore(numElements, cookie);

  // Finally, compute a pointer to the actual data buffer by skipping
  // over the cookie completely.
  CharUnits cookieSize = ARMCXXABI::getArrayCookieSizeImpl(elementType);
  return CGF.Builder.CreateConstInBoundsGEP1_64(newPtr,
                                                cookieSize.getQuantity());
}

llvm::Value *ARMCXXABI::readArrayCookieImpl(CodeGenFunction &CGF,
                                            llvm::Value *allocPtr,
                                            CharUnits cookieSize) {
  // The number of elements is at offset sizeof(size_t) relative to
  // the allocated pointer.
  llvm::Value *numElementsPtr
    = CGF.Builder.CreateConstInBoundsGEP1_64(allocPtr, CGF.SizeSizeInBytes);

  unsigned AS = allocPtr->getType()->getPointerAddressSpace();
  numElementsPtr = 
    CGF.Builder.CreateBitCast(numElementsPtr, CGF.SizeTy->getPointerTo(AS));
  return CGF.Builder.CreateLoad(numElementsPtr);
}

/*********************** Static local initialization **************************/

static llvm::Constant *getGuardAcquireFn(CodeGenModule &CGM,
                                         llvm::PointerType *GuardPtrTy) {
  // int __cxa_guard_acquire(__guard *guard_object);
  llvm::FunctionType *FTy =
    llvm::FunctionType::get(CGM.getTypes().ConvertType(CGM.getContext().IntTy),
                            GuardPtrTy, /*isVarArg=*/false);
  return CGM.CreateRuntimeFunction(FTy, "__cxa_guard_acquire",
                                   llvm::AttributeSet::get(CGM.getLLVMContext(),
                                              llvm::AttributeSet::FunctionIndex,
                                                 llvm::Attribute::NoUnwind));
}

static llvm::Constant *getGuardReleaseFn(CodeGenModule &CGM,
                                         llvm::PointerType *GuardPtrTy) {
  // void __cxa_guard_release(__guard *guard_object);
  llvm::FunctionType *FTy =
    llvm::FunctionType::get(CGM.VoidTy, GuardPtrTy, /*isVarArg=*/false);
  return CGM.CreateRuntimeFunction(FTy, "__cxa_guard_release",
                                   llvm::AttributeSet::get(CGM.getLLVMContext(),
                                              llvm::AttributeSet::FunctionIndex,
                                                 llvm::Attribute::NoUnwind));
}

static llvm::Constant *getGuardAbortFn(CodeGenModule &CGM,
                                       llvm::PointerType *GuardPtrTy) {
  // void __cxa_guard_abort(__guard *guard_object);
  llvm::FunctionType *FTy =
    llvm::FunctionType::get(CGM.VoidTy, GuardPtrTy, /*isVarArg=*/false);
  return CGM.CreateRuntimeFunction(FTy, "__cxa_guard_abort",
                                   llvm::AttributeSet::get(CGM.getLLVMContext(),
                                              llvm::AttributeSet::FunctionIndex,
                                                 llvm::Attribute::NoUnwind));
}

namespace {
  struct CallGuardAbort : EHScopeStack::Cleanup {
    llvm::GlobalVariable *Guard;
    CallGuardAbort(llvm::GlobalVariable *Guard) : Guard(Guard) {}

    void Emit(CodeGenFunction &CGF, Flags flags) override {
      CGF.EmitNounwindRuntimeCall(getGuardAbortFn(CGF.CGM, Guard->getType()),
                                  Guard);
    }
  };
}

/// The ARM code here follows the Itanium code closely enough that we
/// just special-case it at particular places.
void ItaniumCXXABI::EmitGuardedInit(CodeGenFunction &CGF,
                                    const VarDecl &D,
                                    llvm::GlobalVariable *var,
                                    bool shouldPerformInit) {
  CGBuilderTy &Builder = CGF.Builder;

  // We only need to use thread-safe statics for local non-TLS variables;
  // global initialization is always single-threaded.
  bool threadsafe = getContext().getLangOpts().ThreadsafeStatics &&
                    D.isLocalVarDecl() && !D.getTLSKind();

  // If we have a global variable with internal linkage and thread-safe statics
  // are disabled, we can just let the guard variable be of type i8.
  bool useInt8GuardVariable = !threadsafe && var->hasInternalLinkage();

  llvm::IntegerType *guardTy;
  if (useInt8GuardVariable) {
    guardTy = CGF.Int8Ty;
  } else {
    // Guard variables are 64 bits in the generic ABI and size width on ARM
    // (i.e. 32-bit on AArch32, 64-bit on AArch64).
    guardTy = (UseARMGuardVarABI ? CGF.SizeTy : CGF.Int64Ty);
  }
  llvm::PointerType *guardPtrTy = guardTy->getPointerTo();

  // Create the guard variable if we don't already have it (as we
  // might if we're double-emitting this function body).
  llvm::GlobalVariable *guard = CGM.getStaticLocalDeclGuardAddress(&D);
  if (!guard) {
    // Mangle the name for the guard.
    SmallString<256> guardName;
    {
      llvm::raw_svector_ostream out(guardName);
      getMangleContext().mangleStaticGuardVariable(&D, out);
      out.flush();
    }

    // Create the guard variable with a zero-initializer.
    // Just absorb linkage and visibility from the guarded variable.
    guard = new llvm::GlobalVariable(CGM.getModule(), guardTy,
                                     false, var->getLinkage(),
                                     llvm::ConstantInt::get(guardTy, 0),
                                     guardName.str());
    guard->setVisibility(var->getVisibility());
    // If the variable is thread-local, so is its guard variable.
    guard->setThreadLocalMode(var->getThreadLocalMode());

    // The ABI says: It is suggested that it be emitted in the same COMDAT group
    // as the associated data object
    if (!D.isLocalVarDecl() && var->isWeakForLinker() && CGM.supportsCOMDAT()) {
      llvm::Comdat *C = CGM.getModule().getOrInsertComdat(var->getName());
      guard->setComdat(C);
      var->setComdat(C);
      CGF.CurFn->setComdat(C);
    }

    CGM.setStaticLocalDeclGuardAddress(&D, guard);
  }

  // Test whether the variable has completed initialization.
  //
  // Itanium C++ ABI 3.3.2:
  //   The following is pseudo-code showing how these functions can be used:
  //     if (obj_guard.first_byte == 0) {
  //       if ( __cxa_guard_acquire (&obj_guard) ) {
  //         try {
  //           ... initialize the object ...;
  //         } catch (...) {
  //            __cxa_guard_abort (&obj_guard);
  //            throw;
  //         }
  //         ... queue object destructor with __cxa_atexit() ...;
  //         __cxa_guard_release (&obj_guard);
  //       }
  //     }

  // Load the first byte of the guard variable.
  llvm::LoadInst *LI =
      Builder.CreateLoad(Builder.CreateBitCast(guard, CGM.Int8PtrTy));
  LI->setAlignment(1);

  // Itanium ABI:
  //   An implementation supporting thread-safety on multiprocessor
  //   systems must also guarantee that references to the initialized
  //   object do not occur before the load of the initialization flag.
  //
  // In LLVM, we do this by marking the load Acquire.
  if (threadsafe)
    LI->setAtomic(llvm::Acquire);

  // For ARM, we should only check the first bit, rather than the entire byte:
  //
  // ARM C++ ABI 3.2.3.1:
  //   To support the potential use of initialization guard variables
  //   as semaphores that are the target of ARM SWP and LDREX/STREX
  //   synchronizing instructions we define a static initialization
  //   guard variable to be a 4-byte aligned, 4-byte word with the
  //   following inline access protocol.
  //     #define INITIALIZED 1
  //     if ((obj_guard & INITIALIZED) != INITIALIZED) {
  //       if (__cxa_guard_acquire(&obj_guard))
  //         ...
  //     }
  //
  // and similarly for ARM64:
  //
  // ARM64 C++ ABI 3.2.2:
  //   This ABI instead only specifies the value bit 0 of the static guard
  //   variable; all other bits are platform defined. Bit 0 shall be 0 when the
  //   variable is not initialized and 1 when it is.
  llvm::Value *V =
      (UseARMGuardVarABI && !useInt8GuardVariable)
          ? Builder.CreateAnd(LI, llvm::ConstantInt::get(CGM.Int8Ty, 1))
          : LI;
  llvm::Value *isInitialized = Builder.CreateIsNull(V, "guard.uninitialized");

  llvm::BasicBlock *InitCheckBlock = CGF.createBasicBlock("init.check");
  llvm::BasicBlock *EndBlock = CGF.createBasicBlock("init.end");

  // Check if the first byte of the guard variable is zero.
  Builder.CreateCondBr(isInitialized, InitCheckBlock, EndBlock);

  CGF.EmitBlock(InitCheckBlock);

  // Variables used when coping with thread-safe statics and exceptions.
  if (threadsafe) {    
    // Call __cxa_guard_acquire.
    llvm::Value *V
      = CGF.EmitNounwindRuntimeCall(getGuardAcquireFn(CGM, guardPtrTy), guard);
               
    llvm::BasicBlock *InitBlock = CGF.createBasicBlock("init");
  
    Builder.CreateCondBr(Builder.CreateIsNotNull(V, "tobool"),
                         InitBlock, EndBlock);
  
    // Call __cxa_guard_abort along the exceptional edge.
    CGF.EHStack.pushCleanup<CallGuardAbort>(EHCleanup, guard);
    
    CGF.EmitBlock(InitBlock);
  }

  // Emit the initializer and add a global destructor if appropriate.
  CGF.EmitCXXGlobalVarDeclInit(D, var, shouldPerformInit);

  if (threadsafe) {
    // Pop the guard-abort cleanup if we pushed one.
    CGF.PopCleanupBlock();

    // Call __cxa_guard_release.  This cannot throw.
    CGF.EmitNounwindRuntimeCall(getGuardReleaseFn(CGM, guardPtrTy), guard);
  } else {
    Builder.CreateStore(llvm::ConstantInt::get(guardTy, 1), guard);
  }

  CGF.EmitBlock(EndBlock);
}

/// Register a global destructor using __cxa_atexit.
static void emitGlobalDtorWithCXAAtExit(CodeGenFunction &CGF,
                                        llvm::Constant *dtor,
                                        llvm::Constant *addr,
                                        bool TLS) {
  const char *Name = "__cxa_atexit";
  if (TLS) {
    const llvm::Triple &T = CGF.getTarget().getTriple();
    Name = T.isMacOSX() ?  "_tlv_atexit" : "__cxa_thread_atexit";
  }

  // We're assuming that the destructor function is something we can
  // reasonably call with the default CC.  Go ahead and cast it to the
  // right prototype.
  llvm::Type *dtorTy =
    llvm::FunctionType::get(CGF.VoidTy, CGF.Int8PtrTy, false)->getPointerTo();

  // extern "C" int __cxa_atexit(void (*f)(void *), void *p, void *d);
  llvm::Type *paramTys[] = { dtorTy, CGF.Int8PtrTy, CGF.Int8PtrTy };
  llvm::FunctionType *atexitTy =
    llvm::FunctionType::get(CGF.IntTy, paramTys, false);

  // Fetch the actual function.
  llvm::Constant *atexit = CGF.CGM.CreateRuntimeFunction(atexitTy, Name);
  if (llvm::Function *fn = dyn_cast<llvm::Function>(atexit))
    fn->setDoesNotThrow();

  // Create a variable that binds the atexit to this shared object.
  llvm::Constant *handle =
    CGF.CGM.CreateRuntimeVariable(CGF.Int8Ty, "__dso_handle");

  llvm::Value *args[] = {
    llvm::ConstantExpr::getBitCast(dtor, dtorTy),
    llvm::ConstantExpr::getBitCast(addr, CGF.Int8PtrTy),
    handle
  };
  CGF.EmitNounwindRuntimeCall(atexit, args);
}

/// Register a global destructor as best as we know how.
void ItaniumCXXABI::registerGlobalDtor(CodeGenFunction &CGF,
                                       const VarDecl &D,
                                       llvm::Constant *dtor,
                                       llvm::Constant *addr) {
  // Use __cxa_atexit if available.
  if (CGM.getCodeGenOpts().CXAAtExit)
    return emitGlobalDtorWithCXAAtExit(CGF, dtor, addr, D.getTLSKind());

  if (D.getTLSKind())
    CGM.ErrorUnsupported(&D, "non-trivial TLS destruction");

  // In Apple kexts, we want to add a global destructor entry.
  // FIXME: shouldn't this be guarded by some variable?
  if (CGM.getLangOpts().AppleKext) {
    // Generate a global destructor entry.
    return CGM.AddCXXDtorEntry(dtor, addr);
  }

  CGF.registerGlobalDtorWithAtExit(D, dtor, addr);
}

static bool isThreadWrapperReplaceable(const VarDecl *VD,
                                       CodeGen::CodeGenModule &CGM) {
  assert(!VD->isStaticLocal() && "static local VarDecls don't need wrappers!");
  // OS X prefers to have references to thread local variables to go through
  // the thread wrapper instead of directly referencing the backing variable.
  return VD->getTLSKind() == VarDecl::TLS_Dynamic &&
         CGM.getTarget().getTriple().isMacOSX();
}

/// Get the appropriate linkage for the wrapper function. This is essentially
/// the weak form of the variable's linkage; every translation unit which needs
/// the wrapper emits a copy, and we want the linker to merge them.
static llvm::GlobalValue::LinkageTypes
getThreadLocalWrapperLinkage(const VarDecl *VD, CodeGen::CodeGenModule &CGM) {
  llvm::GlobalValue::LinkageTypes VarLinkage =
      CGM.getLLVMLinkageVarDefinition(VD, /*isConstant=*/false);

  // For internal linkage variables, we don't need an external or weak wrapper.
  if (llvm::GlobalValue::isLocalLinkage(VarLinkage))
    return VarLinkage;

  // If the thread wrapper is replaceable, give it appropriate linkage.
  if (isThreadWrapperReplaceable(VD, CGM)) {
    if (llvm::GlobalVariable::isLinkOnceLinkage(VarLinkage) ||
        llvm::GlobalVariable::isWeakODRLinkage(VarLinkage))
      return llvm::GlobalVariable::WeakAnyLinkage;
    return VarLinkage;
  }
  return llvm::GlobalValue::WeakODRLinkage;
}

llvm::Function *
ItaniumCXXABI::getOrCreateThreadLocalWrapper(const VarDecl *VD,
                                             llvm::Value *Val) {
  // Mangle the name for the thread_local wrapper function.
  SmallString<256> WrapperName;
  {
    llvm::raw_svector_ostream Out(WrapperName);
    getMangleContext().mangleItaniumThreadLocalWrapper(VD, Out);
    Out.flush();
  }

  if (llvm::Value *V = CGM.getModule().getNamedValue(WrapperName))
    return cast<llvm::Function>(V);

  llvm::Type *RetTy = Val->getType();
  if (VD->getType()->isReferenceType())
    RetTy = RetTy->getPointerElementType();

  llvm::FunctionType *FnTy = llvm::FunctionType::get(RetTy, false);
  llvm::Function *Wrapper =
      llvm::Function::Create(FnTy, getThreadLocalWrapperLinkage(VD, CGM),
                             WrapperName.str(), &CGM.getModule());
  // Always resolve references to the wrapper at link time.
  if (!Wrapper->hasLocalLinkage() && !isThreadWrapperReplaceable(VD, CGM))
    Wrapper->setVisibility(llvm::GlobalValue::HiddenVisibility);
  return Wrapper;
}

void ItaniumCXXABI::EmitThreadLocalInitFuncs(
    CodeGenModule &CGM,
    ArrayRef<std::pair<const VarDecl *, llvm::GlobalVariable *>>
        CXXThreadLocals, ArrayRef<llvm::Function *> CXXThreadLocalInits,
    ArrayRef<llvm::GlobalVariable *> CXXThreadLocalInitVars) {
  llvm::Function *InitFunc = nullptr;
  if (!CXXThreadLocalInits.empty()) {
    // Generate a guarded initialization function.
    llvm::FunctionType *FTy =
        llvm::FunctionType::get(CGM.VoidTy, /*isVarArg=*/false);
    InitFunc = CGM.CreateGlobalInitOrDestructFunction(FTy, "__tls_init",
                                                      SourceLocation(),
                                                      /*TLS=*/true);
    llvm::GlobalVariable *Guard = new llvm::GlobalVariable(
        CGM.getModule(), CGM.Int8Ty, /*isConstant=*/false,
        llvm::GlobalVariable::InternalLinkage,
        llvm::ConstantInt::get(CGM.Int8Ty, 0), "__tls_guard");
    Guard->setThreadLocal(true);
    CodeGenFunction(CGM)
        .GenerateCXXGlobalInitFunc(InitFunc, CXXThreadLocalInits, Guard);
  }
  for (unsigned I = 0, N = CXXThreadLocals.size(); I != N; ++I) {
    const VarDecl *VD = CXXThreadLocals[I].first;
    llvm::GlobalVariable *Var = CXXThreadLocals[I].second;

    // Some targets require that all access to thread local variables go through
    // the thread wrapper.  This means that we cannot attempt to create a thread
    // wrapper or a thread helper.
    if (isThreadWrapperReplaceable(VD, CGM) && !VD->hasDefinition())
      continue;

    // Mangle the name for the thread_local initialization function.
    SmallString<256> InitFnName;
    {
      llvm::raw_svector_ostream Out(InitFnName);
      getMangleContext().mangleItaniumThreadLocalInit(VD, Out);
      Out.flush();
    }

    // If we have a definition for the variable, emit the initialization
    // function as an alias to the global Init function (if any). Otherwise,
    // produce a declaration of the initialization function.
    llvm::GlobalValue *Init = nullptr;
    bool InitIsInitFunc = false;
    if (VD->hasDefinition()) {
      InitIsInitFunc = true;
      if (InitFunc)
        Init = llvm::GlobalAlias::create(Var->getLinkage(), InitFnName.str(),
                                         InitFunc);
    } else {
      // Emit a weak global function referring to the initialization function.
      // This function will not exist if the TU defining the thread_local
      // variable in question does not need any dynamic initialization for
      // its thread_local variables.
      llvm::FunctionType *FnTy = llvm::FunctionType::get(CGM.VoidTy, false);
      Init = llvm::Function::Create(
          FnTy, llvm::GlobalVariable::ExternalWeakLinkage, InitFnName.str(),
          &CGM.getModule());
    }

    if (Init)
      Init->setVisibility(Var->getVisibility());

    llvm::Function *Wrapper = getOrCreateThreadLocalWrapper(VD, Var);
    llvm::LLVMContext &Context = CGM.getModule().getContext();
    llvm::BasicBlock *Entry = llvm::BasicBlock::Create(Context, "", Wrapper);
    CGBuilderTy Builder(Entry);
    if (InitIsInitFunc) {
      if (Init)
        Builder.CreateCall(Init);
    } else {
      // Don't know whether we have an init function. Call it if it exists.
      llvm::Value *Have = Builder.CreateIsNotNull(Init);
      llvm::BasicBlock *InitBB = llvm::BasicBlock::Create(Context, "", Wrapper);
      llvm::BasicBlock *ExitBB = llvm::BasicBlock::Create(Context, "", Wrapper);
      Builder.CreateCondBr(Have, InitBB, ExitBB);

      Builder.SetInsertPoint(InitBB);
      Builder.CreateCall(Init);
      Builder.CreateBr(ExitBB);

      Builder.SetInsertPoint(ExitBB);
    }

    // For a reference, the result of the wrapper function is a pointer to
    // the referenced object.
    llvm::Value *Val = Var;
    if (VD->getType()->isReferenceType()) {
      llvm::LoadInst *LI = Builder.CreateLoad(Val);
      LI->setAlignment(CGM.getContext().getDeclAlign(VD).getQuantity());
      Val = LI;
    }
    if (Val->getType() != Wrapper->getReturnType())
      Val = Builder.CreatePointerBitCastOrAddrSpaceCast(
          Val, Wrapper->getReturnType(), "");
    Builder.CreateRet(Val);
  }
}

LValue ItaniumCXXABI::EmitThreadLocalVarDeclLValue(CodeGenFunction &CGF,
                                                   const VarDecl *VD,
                                                   QualType LValType) {
  QualType T = VD->getType();
  llvm::Type *Ty = CGF.getTypes().ConvertTypeForMem(T);
  llvm::Value *Val = CGF.CGM.GetAddrOfGlobalVar(VD, Ty);
  llvm::Function *Wrapper = getOrCreateThreadLocalWrapper(VD, Val);

  Val = CGF.Builder.CreateCall(Wrapper);

  LValue LV;
  if (VD->getType()->isReferenceType())
    LV = CGF.MakeNaturalAlignAddrLValue(Val, LValType);
  else
    LV = CGF.MakeAddrLValue(Val, LValType, CGF.getContext().getDeclAlign(VD));
  // FIXME: need setObjCGCLValueClass?
  return LV;
}

/// Return whether the given global decl needs a VTT parameter, which it does
/// if it's a base constructor or destructor with virtual bases.
bool ItaniumCXXABI::NeedsVTTParameter(GlobalDecl GD) {
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());
  
  // We don't have any virtual bases, just return early.
  if (!MD->getParent()->getNumVBases())
    return false;
  
  // Check if we have a base constructor.
  if (isa<CXXConstructorDecl>(MD) && GD.getCtorType() == Ctor_Base)
    return true;

  // Check if we have a base destructor.
  if (isa<CXXDestructorDecl>(MD) && GD.getDtorType() == Dtor_Base)
    return true;
  
  return false;
}

namespace {
class ItaniumRTTIBuilder {
  CodeGenModule &CGM;  // Per-module state.
  llvm::LLVMContext &VMContext;
  const ItaniumCXXABI &CXXABI;  // Per-module state.

  /// Fields - The fields of the RTTI descriptor currently being built.
  SmallVector<llvm::Constant *, 16> Fields;

  /// GetAddrOfTypeName - Returns the mangled type name of the given type.
  llvm::GlobalVariable *
  GetAddrOfTypeName(QualType Ty, llvm::GlobalVariable::LinkageTypes Linkage);

  /// GetAddrOfExternalRTTIDescriptor - Returns the constant for the RTTI
  /// descriptor of the given type.
  llvm::Constant *GetAddrOfExternalRTTIDescriptor(QualType Ty);

  /// BuildVTablePointer - Build the vtable pointer for the given type.
  void BuildVTablePointer(const Type *Ty);

  /// BuildSIClassTypeInfo - Build an abi::__si_class_type_info, used for single
  /// inheritance, according to the Itanium C++ ABI, 2.9.5p6b.
  void BuildSIClassTypeInfo(const CXXRecordDecl *RD);

  /// BuildVMIClassTypeInfo - Build an abi::__vmi_class_type_info, used for
  /// classes with bases that do not satisfy the abi::__si_class_type_info
  /// constraints, according ti the Itanium C++ ABI, 2.9.5p5c.
  void BuildVMIClassTypeInfo(const CXXRecordDecl *RD);

  /// BuildPointerTypeInfo - Build an abi::__pointer_type_info struct, used
  /// for pointer types.
  void BuildPointerTypeInfo(QualType PointeeTy);

  /// BuildObjCObjectTypeInfo - Build the appropriate kind of
  /// type_info for an object type.
  void BuildObjCObjectTypeInfo(const ObjCObjectType *Ty);

  /// BuildPointerToMemberTypeInfo - Build an abi::__pointer_to_member_type_info
  /// struct, used for member pointer types.
  void BuildPointerToMemberTypeInfo(const MemberPointerType *Ty);

public:
  ItaniumRTTIBuilder(const ItaniumCXXABI &ABI)
      : CGM(ABI.CGM), VMContext(CGM.getModule().getContext()), CXXABI(ABI) {}

  // Pointer type info flags.
  enum {
    /// PTI_Const - Type has const qualifier.
    PTI_Const = 0x1,

    /// PTI_Volatile - Type has volatile qualifier.
    PTI_Volatile = 0x2,

    /// PTI_Restrict - Type has restrict qualifier.
    PTI_Restrict = 0x4,

    /// PTI_Incomplete - Type is incomplete.
    PTI_Incomplete = 0x8,

    /// PTI_ContainingClassIncomplete - Containing class is incomplete.
    /// (in pointer to member).
    PTI_ContainingClassIncomplete = 0x10
  };

  // VMI type info flags.
  enum {
    /// VMI_NonDiamondRepeat - Class has non-diamond repeated inheritance.
    VMI_NonDiamondRepeat = 0x1,

    /// VMI_DiamondShaped - Class is diamond shaped.
    VMI_DiamondShaped = 0x2
  };

  // Base class type info flags.
  enum {
    /// BCTI_Virtual - Base class is virtual.
    BCTI_Virtual = 0x1,

    /// BCTI_Public - Base class is public.
    BCTI_Public = 0x2
  };

  /// BuildTypeInfo - Build the RTTI type info struct for the given type.
  ///
  /// \param Force - true to force the creation of this RTTI value
  llvm::Constant *BuildTypeInfo(QualType Ty, bool Force = false);
};
}

llvm::GlobalVariable *ItaniumRTTIBuilder::GetAddrOfTypeName(
    QualType Ty, llvm::GlobalVariable::LinkageTypes Linkage) {
  SmallString<256> OutName;
  llvm::raw_svector_ostream Out(OutName);
  CGM.getCXXABI().getMangleContext().mangleCXXRTTIName(Ty, Out);
  Out.flush();
  StringRef Name = OutName.str();

  // We know that the mangled name of the type starts at index 4 of the
  // mangled name of the typename, so we can just index into it in order to
  // get the mangled name of the type.
  llvm::Constant *Init = llvm::ConstantDataArray::getString(VMContext,
                                                            Name.substr(4));

  llvm::GlobalVariable *GV =
    CGM.CreateOrReplaceCXXRuntimeVariable(Name, Init->getType(), Linkage);

  GV->setInitializer(Init);

  return GV;
}

llvm::Constant *
ItaniumRTTIBuilder::GetAddrOfExternalRTTIDescriptor(QualType Ty) {
  // Mangle the RTTI name.
  SmallString<256> OutName;
  llvm::raw_svector_ostream Out(OutName);
  CGM.getCXXABI().getMangleContext().mangleCXXRTTI(Ty, Out);
  Out.flush();
  StringRef Name = OutName.str();

  // Look for an existing global.
  llvm::GlobalVariable *GV = CGM.getModule().getNamedGlobal(Name);

  if (!GV) {
    // Create a new global variable.
    GV = new llvm::GlobalVariable(CGM.getModule(), CGM.Int8PtrTy,
                                  /*Constant=*/true,
                                  llvm::GlobalValue::ExternalLinkage, nullptr,
                                  Name);
    if (const RecordType *RecordTy = dyn_cast<RecordType>(Ty)) {
      const CXXRecordDecl *RD = cast<CXXRecordDecl>(RecordTy->getDecl());
      if (RD->hasAttr<DLLImportAttr>())
        GV->setDLLStorageClass(llvm::GlobalVariable::DLLImportStorageClass);
    }
  }

  return llvm::ConstantExpr::getBitCast(GV, CGM.Int8PtrTy);
}

/// TypeInfoIsInStandardLibrary - Given a builtin type, returns whether the type
/// info for that type is defined in the standard library.
static bool TypeInfoIsInStandardLibrary(const BuiltinType *Ty) {
  // Itanium C++ ABI 2.9.2:
  //   Basic type information (e.g. for "int", "bool", etc.) will be kept in
  //   the run-time support library. Specifically, the run-time support
  //   library should contain type_info objects for the types X, X* and
  //   X const*, for every X in: void, std::nullptr_t, bool, wchar_t, char,
  //   unsigned char, signed char, short, unsigned short, int, unsigned int,
  //   long, unsigned long, long long, unsigned long long, float, double,
  //   long double, char16_t, char32_t, and the IEEE 754r decimal and
  //   half-precision floating point types.
  switch (Ty->getKind()) {
    case BuiltinType::Void:
    case BuiltinType::NullPtr:
    case BuiltinType::Bool:
    case BuiltinType::WChar_S:
    case BuiltinType::WChar_U:
    case BuiltinType::Char_U:
    case BuiltinType::Char_S:
    case BuiltinType::UChar:
    case BuiltinType::SChar:
    case BuiltinType::Short:
    case BuiltinType::UShort:
    case BuiltinType::Int:
    case BuiltinType::UInt:
    case BuiltinType::Long:
    case BuiltinType::ULong:
    case BuiltinType::LongLong:
    case BuiltinType::ULongLong:
    case BuiltinType::Half:
    case BuiltinType::Float:
    case BuiltinType::Double:
    case BuiltinType::LongDouble:
    case BuiltinType::Char16:
    case BuiltinType::Char32:
    case BuiltinType::Int128:
    case BuiltinType::UInt128:
    case BuiltinType::OCLImage1d:
    case BuiltinType::OCLImage1dArray:
    case BuiltinType::OCLImage1dBuffer:
    case BuiltinType::OCLImage2d:
    case BuiltinType::OCLImage2dArray:
    case BuiltinType::OCLImage3d:
    case BuiltinType::OCLSampler:
    case BuiltinType::OCLEvent:
      return true;

    case BuiltinType::Dependent:
#define BUILTIN_TYPE(Id, SingletonId)
#define PLACEHOLDER_TYPE(Id, SingletonId) \
    case BuiltinType::Id:
#include "clang/AST/BuiltinTypes.def"
      llvm_unreachable("asking for RRTI for a placeholder type!");

    case BuiltinType::ObjCId:
    case BuiltinType::ObjCClass:
    case BuiltinType::ObjCSel:
      llvm_unreachable("FIXME: Objective-C types are unsupported!");
  }

  llvm_unreachable("Invalid BuiltinType Kind!");
}

static bool TypeInfoIsInStandardLibrary(const PointerType *PointerTy) {
  QualType PointeeTy = PointerTy->getPointeeType();
  const BuiltinType *BuiltinTy = dyn_cast<BuiltinType>(PointeeTy);
  if (!BuiltinTy)
    return false;

  // Check the qualifiers.
  Qualifiers Quals = PointeeTy.getQualifiers();
  Quals.removeConst();

  if (!Quals.empty())
    return false;

  return TypeInfoIsInStandardLibrary(BuiltinTy);
}

/// IsStandardLibraryRTTIDescriptor - Returns whether the type
/// information for the given type exists in the standard library.
static bool IsStandardLibraryRTTIDescriptor(QualType Ty) {
  // Type info for builtin types is defined in the standard library.
  if (const BuiltinType *BuiltinTy = dyn_cast<BuiltinType>(Ty))
    return TypeInfoIsInStandardLibrary(BuiltinTy);

  // Type info for some pointer types to builtin types is defined in the
  // standard library.
  if (const PointerType *PointerTy = dyn_cast<PointerType>(Ty))
    return TypeInfoIsInStandardLibrary(PointerTy);

  return false;
}

/// ShouldUseExternalRTTIDescriptor - Returns whether the type information for
/// the given type exists somewhere else, and that we should not emit the type
/// information in this translation unit.  Assumes that it is not a
/// standard-library type.
static bool ShouldUseExternalRTTIDescriptor(CodeGenModule &CGM,
                                            QualType Ty) {
  ASTContext &Context = CGM.getContext();

  // If RTTI is disabled, assume it might be disabled in the
  // translation unit that defines any potential key function, too.
  if (!Context.getLangOpts().RTTI) return false;

  if (const RecordType *RecordTy = dyn_cast<RecordType>(Ty)) {
    const CXXRecordDecl *RD = cast<CXXRecordDecl>(RecordTy->getDecl());
    if (!RD->hasDefinition())
      return false;

    if (!RD->isDynamicClass())
      return false;

    // FIXME: this may need to be reconsidered if the key function
    // changes.
    if (CGM.getVTables().isVTableExternal(RD))
      return true;

    if (RD->hasAttr<DLLImportAttr>())
      return true;
  }

  return false;
}

/// IsIncompleteClassType - Returns whether the given record type is incomplete.
static bool IsIncompleteClassType(const RecordType *RecordTy) {
  return !RecordTy->getDecl()->isCompleteDefinition();
}

/// ContainsIncompleteClassType - Returns whether the given type contains an
/// incomplete class type. This is true if
///
///   * The given type is an incomplete class type.
///   * The given type is a pointer type whose pointee type contains an
///     incomplete class type.
///   * The given type is a member pointer type whose class is an incomplete
///     class type.
///   * The given type is a member pointer type whoise pointee type contains an
///     incomplete class type.
/// is an indirect or direct pointer to an incomplete class type.
static bool ContainsIncompleteClassType(QualType Ty) {
  if (const RecordType *RecordTy = dyn_cast<RecordType>(Ty)) {
    if (IsIncompleteClassType(RecordTy))
      return true;
  }

  if (const PointerType *PointerTy = dyn_cast<PointerType>(Ty))
    return ContainsIncompleteClassType(PointerTy->getPointeeType());

  if (const MemberPointerType *MemberPointerTy =
      dyn_cast<MemberPointerType>(Ty)) {
    // Check if the class type is incomplete.
    const RecordType *ClassType = cast<RecordType>(MemberPointerTy->getClass());
    if (IsIncompleteClassType(ClassType))
      return true;

    return ContainsIncompleteClassType(MemberPointerTy->getPointeeType());
  }

  return false;
}

// CanUseSingleInheritance - Return whether the given record decl has a "single,
// public, non-virtual base at offset zero (i.e. the derived class is dynamic
// iff the base is)", according to Itanium C++ ABI, 2.95p6b.
static bool CanUseSingleInheritance(const CXXRecordDecl *RD) {
  // Check the number of bases.
  if (RD->getNumBases() != 1)
    return false;

  // Get the base.
  CXXRecordDecl::base_class_const_iterator Base = RD->bases_begin();

  // Check that the base is not virtual.
  if (Base->isVirtual())
    return false;

  // Check that the base is public.
  if (Base->getAccessSpecifier() != AS_public)
    return false;

  // Check that the class is dynamic iff the base is.
  const CXXRecordDecl *BaseDecl =
    cast<CXXRecordDecl>(Base->getType()->getAs<RecordType>()->getDecl());
  if (!BaseDecl->isEmpty() &&
      BaseDecl->isDynamicClass() != RD->isDynamicClass())
    return false;

  return true;
}

void ItaniumRTTIBuilder::BuildVTablePointer(const Type *Ty) {
  // abi::__class_type_info.
  static const char * const ClassTypeInfo =
    "_ZTVN10__cxxabiv117__class_type_infoE";
  // abi::__si_class_type_info.
  static const char * const SIClassTypeInfo =
    "_ZTVN10__cxxabiv120__si_class_type_infoE";
  // abi::__vmi_class_type_info.
  static const char * const VMIClassTypeInfo =
    "_ZTVN10__cxxabiv121__vmi_class_type_infoE";

  const char *VTableName = nullptr;

  switch (Ty->getTypeClass()) {
#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(Class, Base) case Type::Class:
#define NON_CANONICAL_TYPE(Class, Base) case Type::Class:
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#include "clang/AST/TypeNodes.def"
    llvm_unreachable("Non-canonical and dependent types shouldn't get here");

  case Type::LValueReference:
  case Type::RValueReference:
    llvm_unreachable("References shouldn't get here");

  case Type::Auto:
    llvm_unreachable("Undeduced auto type shouldn't get here");

  case Type::Builtin:
  // GCC treats vector and complex types as fundamental types.
  case Type::Vector:
  case Type::ExtVector:
  case Type::Complex:
  case Type::Atomic:
  // FIXME: GCC treats block pointers as fundamental types?!
  case Type::BlockPointer:
    // abi::__fundamental_type_info.
    VTableName = "_ZTVN10__cxxabiv123__fundamental_type_infoE";
    break;

  case Type::ConstantArray:
  case Type::IncompleteArray:
  case Type::VariableArray:
    // abi::__array_type_info.
    VTableName = "_ZTVN10__cxxabiv117__array_type_infoE";
    break;

  case Type::FunctionNoProto:
  case Type::FunctionProto:
    // abi::__function_type_info.
    VTableName = "_ZTVN10__cxxabiv120__function_type_infoE";
    break;

  case Type::Enum:
    // abi::__enum_type_info.
    VTableName = "_ZTVN10__cxxabiv116__enum_type_infoE";
    break;

  case Type::Record: {
    const CXXRecordDecl *RD =
      cast<CXXRecordDecl>(cast<RecordType>(Ty)->getDecl());

    if (!RD->hasDefinition() || !RD->getNumBases()) {
      VTableName = ClassTypeInfo;
    } else if (CanUseSingleInheritance(RD)) {
      VTableName = SIClassTypeInfo;
    } else {
      VTableName = VMIClassTypeInfo;
    }

    break;
  }

  case Type::ObjCObject:
    // Ignore protocol qualifiers.
    Ty = cast<ObjCObjectType>(Ty)->getBaseType().getTypePtr();

    // Handle id and Class.
    if (isa<BuiltinType>(Ty)) {
      VTableName = ClassTypeInfo;
      break;
    }

    assert(isa<ObjCInterfaceType>(Ty));
    // Fall through.

  case Type::ObjCInterface:
    if (cast<ObjCInterfaceType>(Ty)->getDecl()->getSuperClass()) {
      VTableName = SIClassTypeInfo;
    } else {
      VTableName = ClassTypeInfo;
    }
    break;

  case Type::ObjCObjectPointer:
  case Type::Pointer:
    // abi::__pointer_type_info.
    VTableName = "_ZTVN10__cxxabiv119__pointer_type_infoE";
    break;

  case Type::MemberPointer:
    // abi::__pointer_to_member_type_info.
    VTableName = "_ZTVN10__cxxabiv129__pointer_to_member_type_infoE";
    break;
  }

  llvm::Constant *VTable =
    CGM.getModule().getOrInsertGlobal(VTableName, CGM.Int8PtrTy);

  llvm::Type *PtrDiffTy =
    CGM.getTypes().ConvertType(CGM.getContext().getPointerDiffType());

  // The vtable address point is 2.
  llvm::Constant *Two = llvm::ConstantInt::get(PtrDiffTy, 2);
  VTable = llvm::ConstantExpr::getInBoundsGetElementPtr(VTable, Two);
  VTable = llvm::ConstantExpr::getBitCast(VTable, CGM.Int8PtrTy);

  Fields.push_back(VTable);
}

/// \brief Return the linkage that the type info and type info name constants
/// should have for the given type.
static llvm::GlobalVariable::LinkageTypes getTypeInfoLinkage(CodeGenModule &CGM,
                                                             QualType Ty) {
  // Itanium C++ ABI 2.9.5p7:
  //   In addition, it and all of the intermediate abi::__pointer_type_info
  //   structs in the chain down to the abi::__class_type_info for the
  //   incomplete class type must be prevented from resolving to the
  //   corresponding type_info structs for the complete class type, possibly
  //   by making them local static objects. Finally, a dummy class RTTI is
  //   generated for the incomplete type that will not resolve to the final
  //   complete class RTTI (because the latter need not exist), possibly by
  //   making it a local static object.
  if (ContainsIncompleteClassType(Ty))
    return llvm::GlobalValue::InternalLinkage;

  switch (Ty->getLinkage()) {
  case NoLinkage:
  case InternalLinkage:
  case UniqueExternalLinkage:
    return llvm::GlobalValue::InternalLinkage;

  case VisibleNoLinkage:
  case ExternalLinkage:
    if (!CGM.getLangOpts().RTTI) {
      // RTTI is not enabled, which means that this type info struct is going
      // to be used for exception handling. Give it linkonce_odr linkage.
      return llvm::GlobalValue::LinkOnceODRLinkage;
    }

    if (const RecordType *Record = dyn_cast<RecordType>(Ty)) {
      const CXXRecordDecl *RD = cast<CXXRecordDecl>(Record->getDecl());
      if (RD->hasAttr<WeakAttr>())
        return llvm::GlobalValue::WeakODRLinkage;
      if (RD->isDynamicClass())
        return CGM.getVTableLinkage(RD);
    }

    return llvm::GlobalValue::LinkOnceODRLinkage;
  }

  llvm_unreachable("Invalid linkage!");
}

llvm::Constant *ItaniumRTTIBuilder::BuildTypeInfo(QualType Ty, bool Force) {
  // We want to operate on the canonical type.
  Ty = CGM.getContext().getCanonicalType(Ty);

  // Check if we've already emitted an RTTI descriptor for this type.
  SmallString<256> OutName;
  llvm::raw_svector_ostream Out(OutName);
  CGM.getCXXABI().getMangleContext().mangleCXXRTTI(Ty, Out);
  Out.flush();
  StringRef Name = OutName.str();

  llvm::GlobalVariable *OldGV = CGM.getModule().getNamedGlobal(Name);
  if (OldGV && !OldGV->isDeclaration()) {
    assert(!OldGV->hasAvailableExternallyLinkage() &&
           "available_externally typeinfos not yet implemented");

    return llvm::ConstantExpr::getBitCast(OldGV, CGM.Int8PtrTy);
  }

  // Check if there is already an external RTTI descriptor for this type.
  bool IsStdLib = IsStandardLibraryRTTIDescriptor(Ty);
  if (!Force && (IsStdLib || ShouldUseExternalRTTIDescriptor(CGM, Ty)))
    return GetAddrOfExternalRTTIDescriptor(Ty);

  // Emit the standard library with external linkage.
  llvm::GlobalVariable::LinkageTypes Linkage;
  if (IsStdLib)
    Linkage = llvm::GlobalValue::ExternalLinkage;
  else
    Linkage = getTypeInfoLinkage(CGM, Ty);

  // Add the vtable pointer.
  BuildVTablePointer(cast<Type>(Ty));

  // And the name.
  llvm::GlobalVariable *TypeName = GetAddrOfTypeName(Ty, Linkage);
  llvm::Constant *TypeNameField;

  // If we're supposed to demote the visibility, be sure to set a flag
  // to use a string comparison for type_info comparisons.
  ItaniumCXXABI::RTTIUniquenessKind RTTIUniqueness =
      CXXABI.classifyRTTIUniqueness(Ty, Linkage);
  if (RTTIUniqueness != ItaniumCXXABI::RUK_Unique) {
    // The flag is the sign bit, which on ARM64 is defined to be clear
    // for global pointers.  This is very ARM64-specific.
    TypeNameField = llvm::ConstantExpr::getPtrToInt(TypeName, CGM.Int64Ty);
    llvm::Constant *flag =
        llvm::ConstantInt::get(CGM.Int64Ty, ((uint64_t)1) << 63);
    TypeNameField = llvm::ConstantExpr::getAdd(TypeNameField, flag);
    TypeNameField =
        llvm::ConstantExpr::getIntToPtr(TypeNameField, CGM.Int8PtrTy);
  } else {
    TypeNameField = llvm::ConstantExpr::getBitCast(TypeName, CGM.Int8PtrTy);
  }
  Fields.push_back(TypeNameField);

  switch (Ty->getTypeClass()) {
#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(Class, Base) case Type::Class:
#define NON_CANONICAL_TYPE(Class, Base) case Type::Class:
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#include "clang/AST/TypeNodes.def"
    llvm_unreachable("Non-canonical and dependent types shouldn't get here");

  // GCC treats vector types as fundamental types.
  case Type::Builtin:
  case Type::Vector:
  case Type::ExtVector:
  case Type::Complex:
  case Type::BlockPointer:
    // Itanium C++ ABI 2.9.5p4:
    // abi::__fundamental_type_info adds no data members to std::type_info.
    break;

  case Type::LValueReference:
  case Type::RValueReference:
    llvm_unreachable("References shouldn't get here");

  case Type::Auto:
    llvm_unreachable("Undeduced auto type shouldn't get here");

  case Type::ConstantArray:
  case Type::IncompleteArray:
  case Type::VariableArray:
    // Itanium C++ ABI 2.9.5p5:
    // abi::__array_type_info adds no data members to std::type_info.
    break;

  case Type::FunctionNoProto:
  case Type::FunctionProto:
    // Itanium C++ ABI 2.9.5p5:
    // abi::__function_type_info adds no data members to std::type_info.
    break;

  case Type::Enum:
    // Itanium C++ ABI 2.9.5p5:
    // abi::__enum_type_info adds no data members to std::type_info.
    break;

  case Type::Record: {
    const CXXRecordDecl *RD =
      cast<CXXRecordDecl>(cast<RecordType>(Ty)->getDecl());
    if (!RD->hasDefinition() || !RD->getNumBases()) {
      // We don't need to emit any fields.
      break;
    }

    if (CanUseSingleInheritance(RD))
      BuildSIClassTypeInfo(RD);
    else
      BuildVMIClassTypeInfo(RD);

    break;
  }

  case Type::ObjCObject:
  case Type::ObjCInterface:
    BuildObjCObjectTypeInfo(cast<ObjCObjectType>(Ty));
    break;

  case Type::ObjCObjectPointer:
    BuildPointerTypeInfo(cast<ObjCObjectPointerType>(Ty)->getPointeeType());
    break;

  case Type::Pointer:
    BuildPointerTypeInfo(cast<PointerType>(Ty)->getPointeeType());
    break;

  case Type::MemberPointer:
    BuildPointerToMemberTypeInfo(cast<MemberPointerType>(Ty));
    break;

  case Type::Atomic:
    // No fields, at least for the moment.
    break;
  }

  llvm::Constant *Init = llvm::ConstantStruct::getAnon(Fields);

  llvm::GlobalVariable *GV =
    new llvm::GlobalVariable(CGM.getModule(), Init->getType(),
                             /*Constant=*/true, Linkage, Init, Name);

  // If there's already an old global variable, replace it with the new one.
  if (OldGV) {
    GV->takeName(OldGV);
    llvm::Constant *NewPtr =
      llvm::ConstantExpr::getBitCast(GV, OldGV->getType());
    OldGV->replaceAllUsesWith(NewPtr);
    OldGV->eraseFromParent();
  }

  // The Itanium ABI specifies that type_info objects must be globally
  // unique, with one exception: if the type is an incomplete class
  // type or a (possibly indirect) pointer to one.  That exception
  // affects the general case of comparing type_info objects produced
  // by the typeid operator, which is why the comparison operators on
  // std::type_info generally use the type_info name pointers instead
  // of the object addresses.  However, the language's built-in uses
  // of RTTI generally require class types to be complete, even when
  // manipulating pointers to those class types.  This allows the
  // implementation of dynamic_cast to rely on address equality tests,
  // which is much faster.

  // All of this is to say that it's important that both the type_info
  // object and the type_info name be uniqued when weakly emitted.

  // Give the type_info object and name the formal visibility of the
  // type itself.
  llvm::GlobalValue::VisibilityTypes llvmVisibility;
  if (llvm::GlobalValue::isLocalLinkage(Linkage))
    // If the linkage is local, only default visibility makes sense.
    llvmVisibility = llvm::GlobalValue::DefaultVisibility;
  else if (RTTIUniqueness == ItaniumCXXABI::RUK_NonUniqueHidden)
    llvmVisibility = llvm::GlobalValue::HiddenVisibility;
  else
    llvmVisibility = CodeGenModule::GetLLVMVisibility(Ty->getVisibility());
  TypeName->setVisibility(llvmVisibility);
  GV->setVisibility(llvmVisibility);

  return llvm::ConstantExpr::getBitCast(GV, CGM.Int8PtrTy);
}

/// ComputeQualifierFlags - Compute the pointer type info flags from the
/// given qualifier.
static unsigned ComputeQualifierFlags(Qualifiers Quals) {
  unsigned Flags = 0;

  if (Quals.hasConst())
    Flags |= ItaniumRTTIBuilder::PTI_Const;
  if (Quals.hasVolatile())
    Flags |= ItaniumRTTIBuilder::PTI_Volatile;
  if (Quals.hasRestrict())
    Flags |= ItaniumRTTIBuilder::PTI_Restrict;

  return Flags;
}

/// BuildObjCObjectTypeInfo - Build the appropriate kind of type_info
/// for the given Objective-C object type.
void ItaniumRTTIBuilder::BuildObjCObjectTypeInfo(const ObjCObjectType *OT) {
  // Drop qualifiers.
  const Type *T = OT->getBaseType().getTypePtr();
  assert(isa<BuiltinType>(T) || isa<ObjCInterfaceType>(T));

  // The builtin types are abi::__class_type_infos and don't require
  // extra fields.
  if (isa<BuiltinType>(T)) return;

  ObjCInterfaceDecl *Class = cast<ObjCInterfaceType>(T)->getDecl();
  ObjCInterfaceDecl *Super = Class->getSuperClass();

  // Root classes are also __class_type_info.
  if (!Super) return;

  QualType SuperTy = CGM.getContext().getObjCInterfaceType(Super);

  // Everything else is single inheritance.
  llvm::Constant *BaseTypeInfo =
      ItaniumRTTIBuilder(CXXABI).BuildTypeInfo(SuperTy);
  Fields.push_back(BaseTypeInfo);
}

/// BuildSIClassTypeInfo - Build an abi::__si_class_type_info, used for single
/// inheritance, according to the Itanium C++ ABI, 2.95p6b.
void ItaniumRTTIBuilder::BuildSIClassTypeInfo(const CXXRecordDecl *RD) {
  // Itanium C++ ABI 2.9.5p6b:
  // It adds to abi::__class_type_info a single member pointing to the
  // type_info structure for the base type,
  llvm::Constant *BaseTypeInfo =
    ItaniumRTTIBuilder(CXXABI).BuildTypeInfo(RD->bases_begin()->getType());
  Fields.push_back(BaseTypeInfo);
}

namespace {
  /// SeenBases - Contains virtual and non-virtual bases seen when traversing
  /// a class hierarchy.
  struct SeenBases {
    llvm::SmallPtrSet<const CXXRecordDecl *, 16> NonVirtualBases;
    llvm::SmallPtrSet<const CXXRecordDecl *, 16> VirtualBases;
  };
}

/// ComputeVMIClassTypeInfoFlags - Compute the value of the flags member in
/// abi::__vmi_class_type_info.
///
static unsigned ComputeVMIClassTypeInfoFlags(const CXXBaseSpecifier *Base,
                                             SeenBases &Bases) {

  unsigned Flags = 0;

  const CXXRecordDecl *BaseDecl =
    cast<CXXRecordDecl>(Base->getType()->getAs<RecordType>()->getDecl());

  if (Base->isVirtual()) {
    // Mark the virtual base as seen.
    if (!Bases.VirtualBases.insert(BaseDecl).second) {
      // If this virtual base has been seen before, then the class is diamond
      // shaped.
      Flags |= ItaniumRTTIBuilder::VMI_DiamondShaped;
    } else {
      if (Bases.NonVirtualBases.count(BaseDecl))
        Flags |= ItaniumRTTIBuilder::VMI_NonDiamondRepeat;
    }
  } else {
    // Mark the non-virtual base as seen.
    if (!Bases.NonVirtualBases.insert(BaseDecl).second) {
      // If this non-virtual base has been seen before, then the class has non-
      // diamond shaped repeated inheritance.
      Flags |= ItaniumRTTIBuilder::VMI_NonDiamondRepeat;
    } else {
      if (Bases.VirtualBases.count(BaseDecl))
        Flags |= ItaniumRTTIBuilder::VMI_NonDiamondRepeat;
    }
  }

  // Walk all bases.
  for (const auto &I : BaseDecl->bases())
    Flags |= ComputeVMIClassTypeInfoFlags(&I, Bases);

  return Flags;
}

static unsigned ComputeVMIClassTypeInfoFlags(const CXXRecordDecl *RD) {
  unsigned Flags = 0;
  SeenBases Bases;

  // Walk all bases.
  for (const auto &I : RD->bases())
    Flags |= ComputeVMIClassTypeInfoFlags(&I, Bases);

  return Flags;
}

/// BuildVMIClassTypeInfo - Build an abi::__vmi_class_type_info, used for
/// classes with bases that do not satisfy the abi::__si_class_type_info
/// constraints, according ti the Itanium C++ ABI, 2.9.5p5c.
void ItaniumRTTIBuilder::BuildVMIClassTypeInfo(const CXXRecordDecl *RD) {
  llvm::Type *UnsignedIntLTy =
    CGM.getTypes().ConvertType(CGM.getContext().UnsignedIntTy);

  // Itanium C++ ABI 2.9.5p6c:
  //   __flags is a word with flags describing details about the class
  //   structure, which may be referenced by using the __flags_masks
  //   enumeration. These flags refer to both direct and indirect bases.
  unsigned Flags = ComputeVMIClassTypeInfoFlags(RD);
  Fields.push_back(llvm::ConstantInt::get(UnsignedIntLTy, Flags));

  // Itanium C++ ABI 2.9.5p6c:
  //   __base_count is a word with the number of direct proper base class
  //   descriptions that follow.
  Fields.push_back(llvm::ConstantInt::get(UnsignedIntLTy, RD->getNumBases()));

  if (!RD->getNumBases())
    return;

  llvm::Type *LongLTy =
    CGM.getTypes().ConvertType(CGM.getContext().LongTy);

  // Now add the base class descriptions.

  // Itanium C++ ABI 2.9.5p6c:
  //   __base_info[] is an array of base class descriptions -- one for every
  //   direct proper base. Each description is of the type:
  //
  //   struct abi::__base_class_type_info {
  //   public:
  //     const __class_type_info *__base_type;
  //     long __offset_flags;
  //
  //     enum __offset_flags_masks {
  //       __virtual_mask = 0x1,
  //       __public_mask = 0x2,
  //       __offset_shift = 8
  //     };
  //   };
  for (const auto &Base : RD->bases()) {
    // The __base_type member points to the RTTI for the base type.
    Fields.push_back(ItaniumRTTIBuilder(CXXABI).BuildTypeInfo(Base.getType()));

    const CXXRecordDecl *BaseDecl =
      cast<CXXRecordDecl>(Base.getType()->getAs<RecordType>()->getDecl());

    int64_t OffsetFlags = 0;

    // All but the lower 8 bits of __offset_flags are a signed offset.
    // For a non-virtual base, this is the offset in the object of the base
    // subobject. For a virtual base, this is the offset in the virtual table of
    // the virtual base offset for the virtual base referenced (negative).
    CharUnits Offset;
    if (Base.isVirtual())
      Offset =
        CGM.getItaniumVTableContext().getVirtualBaseOffsetOffset(RD, BaseDecl);
    else {
      const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);
      Offset = Layout.getBaseClassOffset(BaseDecl);
    };

    OffsetFlags = uint64_t(Offset.getQuantity()) << 8;

    // The low-order byte of __offset_flags contains flags, as given by the
    // masks from the enumeration __offset_flags_masks.
    if (Base.isVirtual())
      OffsetFlags |= BCTI_Virtual;
    if (Base.getAccessSpecifier() == AS_public)
      OffsetFlags |= BCTI_Public;

    Fields.push_back(llvm::ConstantInt::get(LongLTy, OffsetFlags));
  }
}

/// BuildPointerTypeInfo - Build an abi::__pointer_type_info struct,
/// used for pointer types.
void ItaniumRTTIBuilder::BuildPointerTypeInfo(QualType PointeeTy) {
  Qualifiers Quals;
  QualType UnqualifiedPointeeTy =
    CGM.getContext().getUnqualifiedArrayType(PointeeTy, Quals);

  // Itanium C++ ABI 2.9.5p7:
  //   __flags is a flag word describing the cv-qualification and other
  //   attributes of the type pointed to
  unsigned Flags = ComputeQualifierFlags(Quals);

  // Itanium C++ ABI 2.9.5p7:
  //   When the abi::__pbase_type_info is for a direct or indirect pointer to an
  //   incomplete class type, the incomplete target type flag is set.
  if (ContainsIncompleteClassType(UnqualifiedPointeeTy))
    Flags |= PTI_Incomplete;

  llvm::Type *UnsignedIntLTy =
    CGM.getTypes().ConvertType(CGM.getContext().UnsignedIntTy);
  Fields.push_back(llvm::ConstantInt::get(UnsignedIntLTy, Flags));

  // Itanium C++ ABI 2.9.5p7:
  //  __pointee is a pointer to the std::type_info derivation for the
  //  unqualified type being pointed to.
  llvm::Constant *PointeeTypeInfo =
    ItaniumRTTIBuilder(CXXABI).BuildTypeInfo(UnqualifiedPointeeTy);
  Fields.push_back(PointeeTypeInfo);
}

/// BuildPointerToMemberTypeInfo - Build an abi::__pointer_to_member_type_info
/// struct, used for member pointer types.
void
ItaniumRTTIBuilder::BuildPointerToMemberTypeInfo(const MemberPointerType *Ty) {
  QualType PointeeTy = Ty->getPointeeType();

  Qualifiers Quals;
  QualType UnqualifiedPointeeTy =
    CGM.getContext().getUnqualifiedArrayType(PointeeTy, Quals);

  // Itanium C++ ABI 2.9.5p7:
  //   __flags is a flag word describing the cv-qualification and other
  //   attributes of the type pointed to.
  unsigned Flags = ComputeQualifierFlags(Quals);

  const RecordType *ClassType = cast<RecordType>(Ty->getClass());

  // Itanium C++ ABI 2.9.5p7:
  //   When the abi::__pbase_type_info is for a direct or indirect pointer to an
  //   incomplete class type, the incomplete target type flag is set.
  if (ContainsIncompleteClassType(UnqualifiedPointeeTy))
    Flags |= PTI_Incomplete;

  if (IsIncompleteClassType(ClassType))
    Flags |= PTI_ContainingClassIncomplete;

  llvm::Type *UnsignedIntLTy =
    CGM.getTypes().ConvertType(CGM.getContext().UnsignedIntTy);
  Fields.push_back(llvm::ConstantInt::get(UnsignedIntLTy, Flags));

  // Itanium C++ ABI 2.9.5p7:
  //   __pointee is a pointer to the std::type_info derivation for the
  //   unqualified type being pointed to.
  llvm::Constant *PointeeTypeInfo =
    ItaniumRTTIBuilder(CXXABI).BuildTypeInfo(UnqualifiedPointeeTy);
  Fields.push_back(PointeeTypeInfo);

  // Itanium C++ ABI 2.9.5p9:
  //   __context is a pointer to an abi::__class_type_info corresponding to the
  //   class type containing the member pointed to
  //   (e.g., the "A" in "int A::*").
  Fields.push_back(
      ItaniumRTTIBuilder(CXXABI).BuildTypeInfo(QualType(ClassType, 0)));
}

llvm::Constant *ItaniumCXXABI::getAddrOfRTTIDescriptor(QualType Ty) {
  return ItaniumRTTIBuilder(*this).BuildTypeInfo(Ty);
}

void ItaniumCXXABI::EmitFundamentalRTTIDescriptor(QualType Type) {
  QualType PointerType = getContext().getPointerType(Type);
  QualType PointerTypeConst = getContext().getPointerType(Type.withConst());
  ItaniumRTTIBuilder(*this).BuildTypeInfo(Type, true);
  ItaniumRTTIBuilder(*this).BuildTypeInfo(PointerType, true);
  ItaniumRTTIBuilder(*this).BuildTypeInfo(PointerTypeConst, true);
}

void ItaniumCXXABI::EmitFundamentalRTTIDescriptors() {
  QualType FundamentalTypes[] = {
      getContext().VoidTy,             getContext().NullPtrTy,
      getContext().BoolTy,             getContext().WCharTy,
      getContext().CharTy,             getContext().UnsignedCharTy,
      getContext().SignedCharTy,       getContext().ShortTy,
      getContext().UnsignedShortTy,    getContext().IntTy,
      getContext().UnsignedIntTy,      getContext().LongTy,
      getContext().UnsignedLongTy,     getContext().LongLongTy,
      getContext().UnsignedLongLongTy, getContext().HalfTy,
      getContext().FloatTy,            getContext().DoubleTy,
      getContext().LongDoubleTy,       getContext().Char16Ty,
      getContext().Char32Ty,
  };
  for (const QualType &FundamentalType : FundamentalTypes)
    EmitFundamentalRTTIDescriptor(FundamentalType);
}

/// What sort of uniqueness rules should we use for the RTTI for the
/// given type?
ItaniumCXXABI::RTTIUniquenessKind ItaniumCXXABI::classifyRTTIUniqueness(
    QualType CanTy, llvm::GlobalValue::LinkageTypes Linkage) const {
  if (shouldRTTIBeUnique())
    return RUK_Unique;

  // It's only necessary for linkonce_odr or weak_odr linkage.
  if (Linkage != llvm::GlobalValue::LinkOnceODRLinkage &&
      Linkage != llvm::GlobalValue::WeakODRLinkage)
    return RUK_Unique;

  // It's only necessary with default visibility.
  if (CanTy->getVisibility() != DefaultVisibility)
    return RUK_Unique;

  // If we're not required to publish this symbol, hide it.
  if (Linkage == llvm::GlobalValue::LinkOnceODRLinkage)
    return RUK_NonUniqueHidden;

  // If we're required to publish this symbol, as we might be under an
  // explicit instantiation, leave it with default visibility but
  // enable string-comparisons.
  assert(Linkage == llvm::GlobalValue::WeakODRLinkage);
  return RUK_NonUniqueVisible;
}

// Find out how to codegen the complete destructor and constructor
namespace {
enum class StructorCodegen { Emit, RAUW, Alias, COMDAT };
}
static StructorCodegen getCodegenToUse(CodeGenModule &CGM,
                                       const CXXMethodDecl *MD) {
  if (!CGM.getCodeGenOpts().CXXCtorDtorAliases)
    return StructorCodegen::Emit;

  // The complete and base structors are not equivalent if there are any virtual
  // bases, so emit separate functions.
  if (MD->getParent()->getNumVBases())
    return StructorCodegen::Emit;

  GlobalDecl AliasDecl;
  if (const auto *DD = dyn_cast<CXXDestructorDecl>(MD)) {
    AliasDecl = GlobalDecl(DD, Dtor_Complete);
  } else {
    const auto *CD = cast<CXXConstructorDecl>(MD);
    AliasDecl = GlobalDecl(CD, Ctor_Complete);
  }
  llvm::GlobalValue::LinkageTypes Linkage = CGM.getFunctionLinkage(AliasDecl);

  if (llvm::GlobalValue::isDiscardableIfUnused(Linkage))
    return StructorCodegen::RAUW;

  // FIXME: Should we allow available_externally aliases?
  if (!llvm::GlobalAlias::isValidLinkage(Linkage))
    return StructorCodegen::RAUW;

  if (llvm::GlobalValue::isWeakForLinker(Linkage)) {
    // Only ELF supports COMDATs with arbitrary names (C5/D5).
    if (CGM.getTarget().getTriple().isOSBinFormatELF())
      return StructorCodegen::COMDAT;
    return StructorCodegen::Emit;
  }

  return StructorCodegen::Alias;
}

static void emitConstructorDestructorAlias(CodeGenModule &CGM,
                                           GlobalDecl AliasDecl,
                                           GlobalDecl TargetDecl) {
  llvm::GlobalValue::LinkageTypes Linkage = CGM.getFunctionLinkage(AliasDecl);

  StringRef MangledName = CGM.getMangledName(AliasDecl);
  llvm::GlobalValue *Entry = CGM.GetGlobalValue(MangledName);
  if (Entry && !Entry->isDeclaration())
    return;

  auto *Aliasee = cast<llvm::GlobalValue>(CGM.GetAddrOfGlobal(TargetDecl));
  llvm::PointerType *AliasType = Aliasee->getType();

  // Create the alias with no name.
  auto *Alias = llvm::GlobalAlias::create(
      AliasType->getElementType(), 0, Linkage, "", Aliasee, &CGM.getModule());

  // Switch any previous uses to the alias.
  if (Entry) {
    assert(Entry->getType() == AliasType &&
           "declaration exists with different type");
    Alias->takeName(Entry);
    Entry->replaceAllUsesWith(Alias);
    Entry->eraseFromParent();
  } else {
    Alias->setName(MangledName);
  }

  // Finally, set up the alias with its proper name and attributes.
  CGM.setAliasAttributes(cast<NamedDecl>(AliasDecl.getDecl()), Alias);
}

void ItaniumCXXABI::emitCXXStructor(const CXXMethodDecl *MD,
                                    StructorType Type) {
  auto *CD = dyn_cast<CXXConstructorDecl>(MD);
  const CXXDestructorDecl *DD = CD ? nullptr : cast<CXXDestructorDecl>(MD);

  StructorCodegen CGType = getCodegenToUse(CGM, MD);

  if (Type == StructorType::Complete) {
    GlobalDecl CompleteDecl;
    GlobalDecl BaseDecl;
    if (CD) {
      CompleteDecl = GlobalDecl(CD, Ctor_Complete);
      BaseDecl = GlobalDecl(CD, Ctor_Base);
    } else {
      CompleteDecl = GlobalDecl(DD, Dtor_Complete);
      BaseDecl = GlobalDecl(DD, Dtor_Base);
    }

    if (CGType == StructorCodegen::Alias || CGType == StructorCodegen::COMDAT) {
      emitConstructorDestructorAlias(CGM, CompleteDecl, BaseDecl);
      return;
    }

    if (CGType == StructorCodegen::RAUW) {
      StringRef MangledName = CGM.getMangledName(CompleteDecl);
      auto *Aliasee = cast<llvm::GlobalValue>(CGM.GetAddrOfGlobal(BaseDecl));
      CGM.addReplacement(MangledName, Aliasee);
      return;
    }
  }

  // The base destructor is equivalent to the base destructor of its
  // base class if there is exactly one non-virtual base class with a
  // non-trivial destructor, there are no fields with a non-trivial
  // destructor, and the body of the destructor is trivial.
  if (DD && Type == StructorType::Base && CGType != StructorCodegen::COMDAT &&
      !CGM.TryEmitBaseDestructorAsAlias(DD))
    return;

  llvm::Function *Fn = CGM.codegenCXXStructor(MD, Type);

  if (CGType == StructorCodegen::COMDAT) {
    SmallString<256> Buffer;
    llvm::raw_svector_ostream Out(Buffer);
    if (DD)
      getMangleContext().mangleCXXDtorComdat(DD, Out);
    else
      getMangleContext().mangleCXXCtorComdat(CD, Out);
    llvm::Comdat *C = CGM.getModule().getOrInsertComdat(Out.str());
    Fn->setComdat(C);
  }
}
