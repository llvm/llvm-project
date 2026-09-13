//===- ABIInfoImpl.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_ABIINFOIMPL_H
#define LLVM_CLANG_LIB_CODEGEN_ABIINFOIMPL_H

#include "ABIInfo.h"
#include "CGCXXABI.h"
#include "llvm/IR/DerivedTypes.h"

namespace clang::CodeGen {

/// DefaultABIInfo - The default implementation for ABI specific
/// details. This implementation provides information which results in
/// self-consistent and sensible LLVM IR generation, but does not
/// conform to any particular ABI.
class DefaultABIInfo : public ABIInfo {
public:
  DefaultABIInfo(CodeGen::CodeGenTypes &CGT) : ABIInfo(CGT) {}

  virtual ~DefaultABIInfo();

  ABIArgInfo classifyReturnType(QualType RetTy) const;
  ABIArgInfo classifyArgumentType(QualType RetTy) const;

  void computeInfo(CGFunctionInfo &FI) const override;

  RValue EmitVAArg(CodeGenFunction &CGF, Address VAListAddr, QualType Ty,
                   AggValueSlot Slot) const override;
};

void AssignToArrayRange(CodeGen::CGBuilderTy &Builder, llvm::Value *Array,
                        llvm::Value *Value, unsigned FirstIndex,
                        unsigned LastIndex);

bool isAggregateTypeForABI(QualType T);

llvm::Type *getVAListElementType(CodeGenFunction &CGF);

CGCXXABI::RecordArgABI getRecordArgABI(const RecordType *RT, CGCXXABI &CXXABI);

CGCXXABI::RecordArgABI getRecordArgABI(QualType T, CGCXXABI &CXXABI);

bool classifyReturnType(const CGCXXABI &CXXABI, CGFunctionInfo &FI,
                        const ABIInfo &Info);

/// Pass transparent unions as if they were the type of the first element. Sema
/// should ensure that all elements of the union have the same "machine type".
QualType useFirstFieldIfTransparentUnion(QualType Ty);

// Dynamically round a pointer up to a multiple of the given alignment.
llvm::Value *emitRoundPointerUpToAlignment(CodeGenFunction &CGF,
                                           llvm::Value *Ptr, CharUnits Align);

/// Emit va_arg for a platform using the common void* representation,
/// where arguments are simply emitted in an array of slots on the stack.
///
/// This version implements the core direct-value passing rules.
///
/// \param SlotSize - The size and alignment of a stack slot.
///   Each argument will be allocated to a multiple of this number of
///   slots, and all the slots will be aligned to this value.
/// \param AllowHigherAlign - The slot alignment is not a cap;
///   an argument type with an alignment greater than the slot size
///   will be emitted on a higher-alignment address, potentially
///   leaving one or more empty slots behind as padding.  If this
///   is false, the returned address might be less-aligned than
///   DirectAlign.
/// \param ForceRightAdjust - Default is false. On big-endian platform and
///   if the argument is smaller than a slot, set this flag will force
///   right-adjust the argument in its slot irrespective of the type.
Address emitVoidPtrDirectVAArg(CodeGenFunction &CGF, Address VAListAddr,
                               llvm::Type *DirectTy, CharUnits DirectSize,
                               CharUnits DirectAlign, CharUnits SlotSize,
                               bool AllowHigherAlign,
                               bool ForceRightAdjust = false);

/// Emit va_arg for a platform using the common void* representation,
/// where arguments are simply emitted in an array of slots on the stack.
///
/// \param IsIndirect - Values of this type are passed indirectly.
/// \param ValueInfo - The size and alignment of this type, generally
///   computed with getContext().getTypeInfoInChars(ValueTy).
/// \param SlotSizeAndAlign - The size and alignment of a stack slot.
///   Each argument will be allocated to a multiple of this number of
///   slots, and all the slots will be aligned to this value.
/// \param AllowHigherAlign - The slot alignment is not a cap;
///   an argument type with an alignment greater than the slot size
///   will be emitted on a higher-alignment address, potentially
///   leaving one or more empty slots behind as padding.
/// \param ForceRightAdjust - Default is false. On big-endian platform and
///   if the argument is smaller than a slot, set this flag will force
///   right-adjust the argument in its slot irrespective of the type.
RValue emitVoidPtrVAArg(CodeGenFunction &CGF, Address VAListAddr,
                        QualType ValueTy, bool IsIndirect,
                        TypeInfoChars ValueInfo, CharUnits SlotSizeAndAlign,
                        bool AllowHigherAlign, AggValueSlot Slot,
                        bool ForceRightAdjust = false);

Address emitMergePHI(CodeGenFunction &CGF, Address Addr1,
                     llvm::BasicBlock *Block1, Address Addr2,
                     llvm::BasicBlock *Block2, const llvm::Twine &Name = "");

/// isEmptyField - Return true iff a the field is "empty", that is it
/// is an unnamed bit-field or an (array of) empty record(s). If
/// AsIfNoUniqueAddr is true, then C++ record fields are considered empty if
/// the [[no_unique_address]] attribute would have made them empty.
bool isEmptyField(ASTContext &Context, const FieldDecl *FD, bool AllowArrays,
                  bool AsIfNoUniqueAddr = false);

/// isEmptyRecord - Return true iff a structure contains only empty
/// fields. Note that a structure with a flexible array member is not
/// considered empty. If AsIfNoUniqueAddr is true, then C++ record fields are
/// considered empty if the [[no_unique_address]] attribute would have made
/// them empty.
bool isEmptyRecord(ASTContext &Context, QualType T, bool AllowArrays,
                   bool AsIfNoUniqueAddr = false);

/// isEmptyFieldForLayout - Return true iff the field is "empty", that is,
/// either a zero-width bit-field or an \ref isEmptyRecordForLayout.
bool isEmptyFieldForLayout(const ASTContext &Context, const FieldDecl *FD);

/// isEmptyRecordForLayout - Return true iff a structure contains only empty
/// base classes (per \ref isEmptyRecordForLayout) and fields (per
/// \ref isEmptyFieldForLayout). Note, C++ record fields are considered empty
/// if the [[no_unique_address]] attribute would have made them empty.
bool isEmptyRecordForLayout(const ASTContext &Context, QualType T);

/// isSingleElementStruct - Determine if a structure is a "single
/// element struct", i.e. it has exactly one non-empty field or
/// exactly one field which is itself a single element
/// struct. Structures with flexible array members are never
/// considered single element structs.
///
/// \return The field declaration for the single non-empty field, if
/// it exists.
const Type *isSingleElementStruct(QualType T, ASTContext &Context);

/// Shared classification rules for AMDGPU and AMDGCN-SPIR-V, with \p Base as
/// the fallback ABIInfo for non-register-packed cases.
template <typename Base> class AMDGPUABIInfoCommon : public Base {
protected:
  static constexpr unsigned MaxNumRegsForArgsRet = 16; // 16 32-bit registers
  mutable unsigned NumRegsLeft = 0;

  using Base::Base;

  /// Estimate number of registers the type will use when passed in registers.
  uint64_t numRegsForType(QualType Ty) const {
    uint64_t NumRegs = 0;

    if (const VectorType *VT = Ty->template getAs<VectorType>()) {
      // Compute from the number of elements. The reported size is based on
      // the in-memory size, which includes the padding 4th element for
      // 3-vectors.
      QualType EltTy = VT->getElementType();
      uint64_t EltSize = this->getContext().getTypeSize(EltTy);

      // 16-bit element vectors should be passed as packed.
      if (EltSize == 16)
        return (VT->getNumElements() + 1) / 2;

      uint64_t EltNumRegs = (EltSize + 31) / 32;
      return EltNumRegs * VT->getNumElements();
    }

    if (const auto *RD = Ty->getAsRecordDecl()) {
      assert(!RD->hasFlexibleArrayMember());

      for (const FieldDecl *Field : RD->fields())
        NumRegs += numRegsForType(Field->getType());

      return NumRegs;
    }

    return (this->getContext().getTypeSize(Ty) + 31) / 32;
  }

  bool isHomogeneousAggregateBaseType(QualType Ty) const override {
    return true;
  }

  bool isHomogeneousAggregateSmallEnough(const Type *T,
                                         uint64_t Members) const override {
    uint32_t NumRegs = (this->getContext().getTypeSize(T) + 31) / 32;

    // Homogeneous Aggregates may occupy at most 16 registers.
    return Members * NumRegs <= MaxNumRegsForArgsRet;
  }

  // Coerce scalar pointer arguments from generic pointers to a fixed AS.
  llvm::Type *coerceKernelArgumentType(llvm::Type *Ty, unsigned FromAS,
                                       unsigned ToAS) const {
    // Single value types.
    auto *PtrTy = llvm::dyn_cast<llvm::PointerType>(Ty);
    if (PtrTy && PtrTy->getAddressSpace() == FromAS)
      return llvm::PointerType::get(Ty->getContext(), ToAS);
    return Ty;
  }

  ABIArgInfo classifyReturnType(QualType RetTy) const {
    if (!isAggregateTypeForABI(RetTy) ||
        getRecordArgABI(RetTy, this->getCXXABI()))
      return Base::classifyReturnType(RetTy);

    // Ignore empty structs/unions.
    if (isEmptyRecord(this->getContext(), RetTy, true))
      return ABIArgInfo::getIgnore();

    // Lower single-element structs to just return a regular value.
    if (const Type *SeltTy = isSingleElementStruct(RetTy, this->getContext()))
      return ABIArgInfo::getDirect(this->CGT.ConvertType(QualType(SeltTy, 0)));

    if (const auto *RD = RetTy->getAsRecordDecl();
        RD && RD->hasFlexibleArrayMember())
      return Base::classifyReturnType(RetTy);

    // Pack aggregates <= 4 bytes into single VGPR or pair.
    uint64_t Size = this->getContext().getTypeSize(RetTy);
    if (Size <= 16)
      return ABIArgInfo::getDirect(
          llvm::Type::getInt16Ty(this->getVMContext()));

    if (Size <= 32)
      return ABIArgInfo::getDirect(
          llvm::Type::getInt32Ty(this->getVMContext()));

    if (Size <= 64) {
      llvm::Type *I32Ty = llvm::Type::getInt32Ty(this->getVMContext());
      return ABIArgInfo::getDirect(llvm::ArrayType::get(I32Ty, 2));
    }

    if (numRegsForType(RetTy) <= MaxNumRegsForArgsRet)
      return ABIArgInfo::getDirect();

    return Base::classifyReturnType(RetTy);
  }

  ABIArgInfo classifyArgumentType(QualType Ty, bool Variadic) const {
    assert(NumRegsLeft <= MaxNumRegsForArgsRet &&
           "register estimate underflow");

    Ty = useFirstFieldIfTransparentUnion(Ty);

    if (Variadic) {
      return ABIArgInfo::getDirect(/*T=*/nullptr,
                                   /*Offset=*/0,
                                   /*Padding=*/nullptr,
                                   /*CanBeFlattened=*/false,
                                   /*Align=*/0);
    }

    if (!isAggregateTypeForABI(Ty)) {
      ABIArgInfo ArgInfo = Base::classifyArgumentType(Ty);
      if (!ArgInfo.isIndirect()) {
        uint64_t NumRegs = numRegsForType(Ty);
        NumRegsLeft -= std::min(NumRegs, uint64_t{NumRegsLeft});
      }

      return ArgInfo;
    }

    // Records with non-trivial destructors/copy-constructors should not be
    // passed by value.
    if (auto RAA = getRecordArgABI(Ty, this->getCXXABI()))
      return this->getNaturalAlignIndirect(
          Ty, this->getDataLayout().getAllocaAddrSpace(),
          RAA == CGCXXABI::RAA_DirectInMemory);

    // Ignore empty structs/unions.
    if (isEmptyRecord(this->getContext(), Ty, true))
      return ABIArgInfo::getIgnore();

    // Lower single-element structs to just pass a regular value. TODO: We
    // could do reasonable-size multiple-element structs too, using
    // getExpand(), though watch out for things like bitfields.
    if (const Type *SeltTy = isSingleElementStruct(Ty, this->getContext()))
      return ABIArgInfo::getDirect(this->CGT.ConvertType(QualType(SeltTy, 0)));

    if (const auto *RD = Ty->getAsRecordDecl();
        RD && RD->hasFlexibleArrayMember())
      return Base::classifyArgumentType(Ty);

    // Pack aggregates <= 8 bytes into single VGPR or pair.
    uint64_t Size = this->getContext().getTypeSize(Ty);
    if (Size <= 64) {
      unsigned NumRegs = (Size + 31) / 32;
      NumRegsLeft -= std::min(NumRegsLeft, NumRegs);

      if (Size <= 16)
        return ABIArgInfo::getDirect(
            llvm::Type::getInt16Ty(this->getVMContext()));

      if (Size <= 32)
        return ABIArgInfo::getDirect(
            llvm::Type::getInt32Ty(this->getVMContext()));

      // XXX: Should this be i64 instead, and should the limit increase?
      llvm::Type *I32Ty = llvm::Type::getInt32Ty(this->getVMContext());
      return ABIArgInfo::getDirect(llvm::ArrayType::get(I32Ty, 2));
    }

    if (NumRegsLeft > 0) {
      uint64_t NumRegs = numRegsForType(Ty);
      if (NumRegsLeft >= NumRegs) {
        NumRegsLeft -= NumRegs;
        return ABIArgInfo::getDirect();
      }
    }

    // Use pass-by-reference instead of pass-by-value for struct arguments in
    // function ABI.
    return ABIArgInfo::getIndirectAliased(
        this->getContext().getTypeAlignInChars(Ty),
        this->getContext().getTargetAddressSpace(LangAS::opencl_private));
  }

  llvm::FixedVectorType *
  getOptimalVectorMemoryType(llvm::FixedVectorType *Ty,
                             const LangOptions &LangOpt) const override {
    // We have legal instructions for 96-bit so 3x32 can be supported.
    // FIXME: This check should be a subtarget feature as technically SI
    // doesn't support it.
    if (Ty->getNumElements() == 3 &&
        this->getDataLayout().getTypeSizeInBits(Ty) == 96)
      return Ty;
    return Base::getOptimalVectorMemoryType(Ty, LangOpt);
  }
};

Address EmitVAArgInstr(CodeGenFunction &CGF, Address VAListAddr, QualType Ty,
                       const ABIArgInfo &AI);

bool isSIMDVectorType(ASTContext &Context, QualType Ty);

bool isRecordWithSIMDVectorType(ASTContext &Context, QualType Ty);

} // namespace clang::CodeGen

#endif // LLVM_CLANG_LIB_CODEGEN_ABIINFOIMPL_H
