//===------- ItaniumCXXABI.cpp - Emit CIR code Itanium-specific code  -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides CIR lowering logic targeting the Itanium C++ ABI. The class in
// this file generates structures that follow the Itanium C++ ABI, which is
// documented at:
//  https://itanium-cxx-abi.github.io/cxx-abi/abi.html
//  https://itanium-cxx-abi.github.io/cxx-abi/abi-eh.html
//
// It also supports the closely-related ARM ABI, documented at:
// https://developer.arm.com/documentation/ihi0041/g/
//
// This file partially mimics clang/lib/CodeGen/ItaniumCXXABI.cpp. The queries
// are adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#include "../LoweringPrepareCXXABI.h"
#include "CIRCXXABI.h"
#include "LowerModule.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/ErrorHandling.h"

namespace cir {

namespace {

class ItaniumCXXABI : public CIRCXXABI {

protected:
  enum class VTableComponentLayout {
    /// Components in the vtable are pointers to other structs/functions.
    Pointer,

    /// Components in the vtable are relative offsets between the vtable and the
    /// other structs/functions.
    Relative,
  };

  bool UseARMMethodPtrABI;
  bool UseARMGuardVarABI;
  bool Use32BitVTableOffsetABI;
  VTableComponentLayout VTComponentLayout;

public:
  ItaniumCXXABI(
      LowerModule &LM, bool UseARMMethodPtrABI = false,
      bool UseARMGuardVarABI = false,
      VTableComponentLayout VTComponentLayout = VTableComponentLayout::Pointer)
      : CIRCXXABI(LM), UseARMMethodPtrABI(UseARMMethodPtrABI),
        UseARMGuardVarABI(UseARMGuardVarABI), Use32BitVTableOffsetABI(false),
        VTComponentLayout(VTComponentLayout) {}

  bool classifyReturnType(LowerFunctionInfo &FI) const override;

  // FIXME(cir): This expects a CXXRecordDecl! Not any record type.
  RecordArgABI getRecordArgABI(const StructType RD) const override {
    cir_cconv_assert(!cir::MissingFeatures::recordDeclIsCXXDecl());
    // If C++ prohibits us from making a copy, pass by address.
    cir_cconv_assert(!cir::MissingFeatures::recordDeclCanPassInRegisters());
    return RAA_Default;
  }

  mlir::Type
  lowerDataMemberType(cir::DataMemberType type,
                      const mlir::TypeConverter &typeConverter) const override;

  mlir::Type
  lowerMethodType(cir::MethodType type,
                  const mlir::TypeConverter &typeConverter) const override;

  mlir::TypedAttr lowerDataMemberConstant(
      cir::DataMemberAttr attr, const mlir::DataLayout &layout,
      const mlir::TypeConverter &typeConverter) const override;

  mlir::TypedAttr
  lowerMethodConstant(cir::MethodAttr attr, const mlir::DataLayout &layout,
                      const mlir::TypeConverter &typeConverter) const override;

  mlir::Operation *
  lowerGetRuntimeMember(cir::GetRuntimeMemberOp op, mlir::Type loweredResultTy,
                        mlir::Value loweredAddr, mlir::Value loweredMember,
                        mlir::OpBuilder &builder) const override;

  void lowerGetMethod(cir::GetMethodOp op, mlir::Value (&loweredResults)[2],
                      mlir::Value loweredMethod, mlir::Value loweredObjectPtr,
                      mlir::ConversionPatternRewriter &rewriter) const override;

  mlir::Value lowerBaseDataMember(cir::BaseDataMemberOp op,
                                  mlir::Value loweredSrc,
                                  mlir::OpBuilder &builder) const override;

  mlir::Value lowerDerivedDataMember(cir::DerivedDataMemberOp op,
                                     mlir::Value loweredSrc,
                                     mlir::OpBuilder &builder) const override;

  mlir::Value lowerBaseMethod(cir::BaseMethodOp op, mlir::Value loweredSrc,
                              mlir::OpBuilder &builder) const override;

  mlir::Value lowerDerivedMethod(cir::DerivedMethodOp op,
                                 mlir::Value loweredSrc,
                                 mlir::OpBuilder &builder) const override;

  mlir::Value lowerDataMemberCmp(cir::CmpOp op, mlir::Value loweredLhs,
                                 mlir::Value loweredRhs,
                                 mlir::OpBuilder &builder) const override;

  mlir::Value lowerMethodCmp(cir::CmpOp op, mlir::Value loweredLhs,
                             mlir::Value loweredRhs,
                             mlir::OpBuilder &builder) const override;

  mlir::Value lowerDataMemberBitcast(cir::CastOp op, mlir::Type loweredDstTy,
                                     mlir::Value loweredSrc,
                                     mlir::OpBuilder &builder) const override;

  mlir::Value
  lowerDataMemberToBoolCast(cir::CastOp op, mlir::Value loweredSrc,
                            mlir::OpBuilder &builder) const override;

  mlir::Value lowerMethodBitcast(cir::CastOp op, mlir::Type loweredDstTy,
                                 mlir::Value loweredSrc,
                                 mlir::OpBuilder &builder) const override;

  mlir::Value lowerMethodToBoolCast(cir::CastOp op, mlir::Value loweredSrc,
                                    mlir::OpBuilder &builder) const override;
};

} // namespace

bool ItaniumCXXABI::classifyReturnType(LowerFunctionInfo &FI) const {
  const StructType RD = mlir::dyn_cast<StructType>(FI.getReturnType());
  if (!RD)
    return false;

  // If C++ prohibits us from making a copy, return by address.
  if (cir::MissingFeatures::recordDeclCanPassInRegisters())
    cir_cconv_unreachable("NYI");

  return false;
}

static cir::IntType getPtrDiffCIRTy(LowerModule &lowerMod) {
  const clang::TargetInfo &target = lowerMod.getTarget();
  clang::TargetInfo::IntType ptrdiffTy =
      target.getPtrDiffType(clang::LangAS::Default);
  return cir::IntType::get(lowerMod.getMLIRContext(),
                           target.getTypeWidth(ptrdiffTy),
                           target.isTypeSigned(ptrdiffTy));
}

mlir::Type ItaniumCXXABI::lowerDataMemberType(
    cir::DataMemberType type, const mlir::TypeConverter &typeConverter) const {
  // Itanium C++ ABI 2.3.1:
  //   A data member pointer is represented as the data member's offset in bytes
  //   from the address point of an object of the base type, as a ptrdiff_t.
  return getPtrDiffCIRTy(LM);
}

mlir::Type
ItaniumCXXABI::lowerMethodType(cir::MethodType type,
                               const mlir::TypeConverter &typeConverter) const {
  // Itanium C++ ABI 2.3.2:
  //    In all representations, the basic ABI properties of member function
  //    pointer types are those of the following class, where fnptr_t is the
  //    appropriate function-pointer type for a member function of this type:
  //
  //    struct {
  //      fnptr_t ptr;
  //      ptrdiff_t adj;
  //    };

  cir::IntType ptrdiffCIRTy = getPtrDiffCIRTy(LM);

  // Note that clang CodeGen emits struct{ptrdiff_t, ptrdiff_t} for member
  // function pointers. Let's follow this approach.
  return cir::StructType::get(type.getContext(), {ptrdiffCIRTy, ptrdiffCIRTy},
                              /*packed=*/false, /*padded=*/false,
                              cir::StructType::Struct);
}

mlir::TypedAttr ItaniumCXXABI::lowerDataMemberConstant(
    cir::DataMemberAttr attr, const mlir::DataLayout &layout,
    const mlir::TypeConverter &typeConverter) const {
  uint64_t memberOffset;
  if (attr.isNullPtr()) {
    // Itanium C++ ABI 2.3:
    //   A NULL pointer is represented as -1.
    memberOffset = -1ull;
  } else {
    // Itanium C++ ABI 2.3:
    //   A pointer to data member is an offset from the base address of
    //   the class object containing it, represented as a ptrdiff_t
    auto memberIndex = attr.getMemberIndex().value();
    memberOffset =
        attr.getType().getClsTy().getElementOffset(layout, memberIndex);
  }

  mlir::Type abiTy = lowerDataMemberType(attr.getType(), typeConverter);
  return cir::IntAttr::get(abiTy, memberOffset);
}

mlir::TypedAttr ItaniumCXXABI::lowerMethodConstant(
    cir::MethodAttr attr, const mlir::DataLayout &layout,
    const mlir::TypeConverter &typeConverter) const {
  cir::IntType ptrdiffCIRTy = getPtrDiffCIRTy(LM);
  auto loweredMethodTy = mlir::cast<cir::StructType>(
      lowerMethodType(attr.getType(), typeConverter));

  auto zero = cir::IntAttr::get(ptrdiffCIRTy, 0);

  // Itanium C++ ABI 2.3.2:
  //   In all representations, the basic ABI properties of member function
  //   pointer types are those of the following class, where fnptr_t is the
  //   appropriate function-pointer type for a member function of this type:
  //
  //   struct {
  //     fnptr_t ptr;
  //     ptrdiff_t adj;
  //   };

  if (attr.isNull()) {
    // Itanium C++ ABI 2.3.2:
    //
    //   In the standard representation, a null member function pointer is
    //   represented with ptr set to a null pointer. The value of adj is
    //   unspecified for null member function pointers.
    //
    // clang CodeGen emits struct{null, null} for null member function pointers.
    // Let's do the same here.
    return cir::ConstStructAttr::get(
        loweredMethodTy, mlir::ArrayAttr::get(attr.getContext(), {zero, zero}));
  }

  if (attr.isVirtual()) {
    if (UseARMMethodPtrABI) {
      // ARM C++ ABI 3.2.1:
      //   This ABI specifies that adj contains twice the this
      //   adjustment, plus 1 if the member function is virtual. The
      //   least significant bit of adj then makes exactly the same
      //   discrimination as the least significant bit of ptr does for
      //   Itanium.
      llvm_unreachable("ARM method ptr abi NYI");
    }

    // Itanium C++ ABI 2.3.2:
    //
    //   In the standard representation, a member function pointer for a
    //   virtual function is represented with ptr set to 1 plus the function's
    //   v-table entry offset (in bytes), converted to a function pointer as if
    //   by reinterpret_cast<fnptr_t>(uintfnptr_t(1 + offset)), where
    //   uintfnptr_t is an unsigned integer of the same size as fnptr_t.
    auto ptr =
        cir::IntAttr::get(ptrdiffCIRTy, 1 + attr.getVtableOffset().value());
    return cir::ConstStructAttr::get(
        loweredMethodTy, mlir::ArrayAttr::get(attr.getContext(), {ptr, zero}));
  }

  // Itanium C++ ABI 2.3.2:
  //
  //   A member function pointer for a non-virtual member function is
  //   represented with ptr set to a pointer to the function, using the base
  //   ABI's representation of function pointers.
  auto ptr = cir::GlobalViewAttr::get(ptrdiffCIRTy, attr.getSymbol().value());
  return cir::ConstStructAttr::get(
      loweredMethodTy, mlir::ArrayAttr::get(attr.getContext(), {ptr, zero}));
}

mlir::Operation *ItaniumCXXABI::lowerGetRuntimeMember(
    cir::GetRuntimeMemberOp op, mlir::Type loweredResultTy,
    mlir::Value loweredAddr, mlir::Value loweredMember,
    mlir::OpBuilder &builder) const {
  auto byteTy = IntType::get(op.getContext(), 8, true);
  auto bytePtrTy = PointerType::get(
      byteTy, mlir::cast<PointerType>(op.getAddr().getType()).getAddrSpace());
  auto objectBytesPtr = builder.create<CastOp>(op.getLoc(), bytePtrTy,
                                               CastKind::bitcast, op.getAddr());
  auto memberBytesPtr = builder.create<PtrStrideOp>(
      op.getLoc(), bytePtrTy, objectBytesPtr, loweredMember);
  return builder.create<CastOp>(op.getLoc(), op.getType(), CastKind::bitcast,
                                memberBytesPtr);
}

void ItaniumCXXABI::lowerGetMethod(
    cir::GetMethodOp op, mlir::Value (&loweredResults)[2],
    mlir::Value loweredMethod, mlir::Value loweredObjectPtr,
    mlir::ConversionPatternRewriter &rewriter) const {
  // In the Itanium and ARM ABIs, method pointers have the form:
  //   struct { ptrdiff_t ptr; ptrdiff_t adj; } memptr;
  //
  // In the Itanium ABI:
  //  - method pointers are virtual if (memptr.ptr & 1) is nonzero
  //  - the this-adjustment is (memptr.adj)
  //  - the virtual offset is (memptr.ptr - 1)
  //
  // In the ARM ABI:
  //  - method pointers are virtual if (memptr.adj & 1) is nonzero
  //  - the this-adjustment is (memptr.adj >> 1)
  //  - the virtual offset is (memptr.ptr)
  // ARM uses 'adj' for the virtual flag because Thumb functions
  // may be only single-byte aligned.
  //
  // If the member is virtual, the adjusted 'this' pointer points
  // to a vtable pointer from which the virtual offset is applied.
  //
  // If the member is non-virtual, memptr.ptr is the address of
  // the function to call.

  mlir::Value &callee = loweredResults[0];
  mlir::Value &adjustedThis = loweredResults[1];
  mlir::Type calleePtrTy = op.getCallee().getType();

  cir::IntType ptrdiffCIRTy = getPtrDiffCIRTy(LM);
  mlir::Value ptrdiffOne = rewriter.create<cir::ConstantOp>(
      op.getLoc(), cir::IntAttr::get(ptrdiffCIRTy, 1));

  mlir::Value adj = rewriter.create<cir::ExtractMemberOp>(
      op.getLoc(), ptrdiffCIRTy, loweredMethod, 1);
  if (UseARMMethodPtrABI)
    llvm_unreachable("ARM method ptr abi NYI");

  // Apply the adjustment to the 'this' pointer.
  mlir::Type thisVoidPtrTy = cir::PointerType::get(
      cir::VoidType::get(rewriter.getContext()),
      mlir::cast<cir::PointerType>(op.getObject().getType()).getAddrSpace());
  mlir::Value thisVoidPtr = rewriter.create<cir::CastOp>(
      op.getLoc(), thisVoidPtrTy, cir::CastKind::bitcast, loweredObjectPtr);
  adjustedThis = rewriter.create<cir::PtrStrideOp>(op.getLoc(), thisVoidPtrTy,
                                                   thisVoidPtr, adj);

  // Load the "ptr" field of the member function pointer and determine if it
  // points to a virtual function.
  mlir::Value methodPtrField = rewriter.create<cir::ExtractMemberOp>(
      op.getLoc(), ptrdiffCIRTy, loweredMethod, 0);
  mlir::Value virtualBit = rewriter.create<cir::BinOp>(
      op.getLoc(), cir::BinOpKind::And, methodPtrField, ptrdiffOne);
  mlir::Value isVirtual;
  if (UseARMMethodPtrABI)
    llvm_unreachable("ARM method ptr abi NYI");
  else
    isVirtual = rewriter.create<cir::CmpOp>(op.getLoc(), cir::CmpOpKind::eq,
                                            virtualBit, ptrdiffOne);

  assert(!MissingFeatures::emitCFICheck());
  assert(!MissingFeatures::emitVFEInfo());
  assert(!MissingFeatures::emitWPDInfo());

  // See their original definitions in
  // ItaniumCXXABI::EmitLoadOfMemberFunctionPointer in file
  // clang/lib/CodeGen/ItaniumCXXABI.cpp.
  bool shouldEmitCFICheck = false;
  bool shouldEmitVFEInfo =
      LM.getContext().getCodeGenOpts().VirtualFunctionElimination;
  bool shouldEmitWPDInfo = LM.getContext().getCodeGenOpts().WholeProgramVTables;

  mlir::Block *currBlock = rewriter.getInsertionBlock();
  mlir::Block *continueBlock =
      rewriter.splitBlock(currBlock, rewriter.getInsertionPoint());
  continueBlock->addArgument(calleePtrTy, op.getLoc());

  mlir::Block *virtualBlock = rewriter.createBlock(continueBlock);
  mlir::Block *nonVirtualBlock = rewriter.createBlock(continueBlock);
  rewriter.setInsertionPointToEnd(currBlock);
  rewriter.create<cir::BrCondOp>(op.getLoc(), isVirtual, virtualBlock,
                                 nonVirtualBlock);

  auto buildVirtualBranch = [&] {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(virtualBlock);

    // Load vtable pointer.
    // Note that vtable pointer always point to the global address space.
    auto vtablePtrTy = cir::PointerType::get(
        rewriter.getContext(),
        cir::IntType::get(rewriter.getContext(), 8, true));
    auto vtablePtrPtrTy = cir::PointerType::get(
        rewriter.getContext(), vtablePtrTy,
        mlir::cast<cir::PointerType>(op.getObject().getType()).getAddrSpace());
    auto vtablePtrPtr = rewriter.create<cir::CastOp>(
        op.getLoc(), vtablePtrPtrTy, cir::CastKind::bitcast, loweredObjectPtr);
    mlir::Value vtablePtr = rewriter.create<cir::LoadOp>(
        op.getLoc(), vtablePtrPtr, /*isDeref=*/false, /*isVolatile=*/false,
        /*alignment=*/mlir::IntegerAttr(), /*mem_order=*/cir::MemOrderAttr(),
        /*tbaa=*/mlir::ArrayAttr());

    // Get the vtable offset.
    mlir::Value vtableOffset = methodPtrField;
    if (!UseARMMethodPtrABI)
      vtableOffset = rewriter.create<cir::BinOp>(
          op.getLoc(), cir::BinOpKind::Sub, vtableOffset, ptrdiffOne);
    if (Use32BitVTableOffsetABI)
      llvm_unreachable("NYI");

    if (shouldEmitCFICheck || shouldEmitVFEInfo || shouldEmitWPDInfo)
      llvm_unreachable("NYI");

    // Apply the offset to the vtable pointer and get the pointer to the target
    // virtual function. Then load that pointer to get the callee.
    mlir::Value funcPtr;
    if (shouldEmitVFEInfo)
      llvm_unreachable("NYI");
    else {
      if (shouldEmitCFICheck || shouldEmitWPDInfo)
        llvm_unreachable("NYI");

      if (VTComponentLayout == VTableComponentLayout::Relative)
        llvm_unreachable("NYI");
      else {
        mlir::Value vfpAddr = rewriter.create<cir::PtrStrideOp>(
            op.getLoc(), vtablePtrTy, vtablePtr, vtableOffset);
        auto vfpPtrTy =
            cir::PointerType::get(rewriter.getContext(), calleePtrTy);
        mlir::Value vfpPtr = rewriter.create<cir::CastOp>(
            op.getLoc(), vfpPtrTy, cir::CastKind::bitcast, vfpAddr);
        funcPtr = rewriter.create<cir::LoadOp>(
            op.getLoc(), vfpPtr, /*isDeref=*/false, /*isVolatile=*/false,
            /*alignment=*/mlir::IntegerAttr(),
            /*mem_order=*/cir::MemOrderAttr(),
            /*tbaa=*/mlir::ArrayAttr());
      }
    }

    if (shouldEmitCFICheck)
      llvm_unreachable("NYI");

    rewriter.create<cir::BrOp>(op.getLoc(), continueBlock, funcPtr);
  };

  auto buildNonVirtualBranch = [&] {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(nonVirtualBlock);

    mlir::Value funcPtr = rewriter.create<cir::CastOp>(
        op.getLoc(), calleePtrTy, cir::CastKind::int_to_ptr, methodPtrField);

    if (shouldEmitCFICheck)
      llvm_unreachable("NYI");

    rewriter.create<cir::BrOp>(op.getLoc(), continueBlock, funcPtr);
  };

  buildVirtualBranch();
  buildNonVirtualBranch();

  rewriter.setInsertionPointToStart(continueBlock);
  callee = continueBlock->getArgument(0);
}

static mlir::Value lowerDataMemberCast(mlir::Operation *op,
                                       mlir::Value loweredSrc,
                                       std::int64_t offset,
                                       bool isDerivedToBase,
                                       mlir::OpBuilder &builder) {
  if (offset == 0)
    return loweredSrc;

  auto nullValue = builder.create<cir::ConstantOp>(
      op->getLoc(), mlir::IntegerAttr::get(loweredSrc.getType(), -1));
  auto isNull = builder.create<cir::CmpOp>(op->getLoc(), cir::CmpOpKind::eq,
                                           loweredSrc, nullValue);

  auto offsetValue = builder.create<cir::ConstantOp>(
      op->getLoc(), mlir::IntegerAttr::get(loweredSrc.getType(), offset));
  auto binOpKind = isDerivedToBase ? cir::BinOpKind::Sub : cir::BinOpKind::Add;
  auto adjustedPtr = builder.create<cir::BinOp>(
      op->getLoc(), loweredSrc.getType(), binOpKind, loweredSrc, offsetValue);

  return builder.create<cir::SelectOp>(op->getLoc(), loweredSrc.getType(),
                                       isNull, nullValue, adjustedPtr);
}

static mlir::Value lowerMethodCast(mlir::Operation *op, mlir::Value loweredSrc,
                                   std::int64_t offset, bool isDerivedToBase,
                                   LowerModule &lowerMod,
                                   mlir::OpBuilder &builder) {
  if (offset == 0)
    return loweredSrc;

  cir::IntType ptrdiffCIRTy = getPtrDiffCIRTy(lowerMod);
  auto adjField = builder.create<cir::ExtractMemberOp>(
      op->getLoc(), ptrdiffCIRTy, loweredSrc, 1);

  auto offsetValue = builder.create<cir::ConstantOp>(
      op->getLoc(), cir::IntAttr::get(ptrdiffCIRTy, offset));
  auto binOpKind = isDerivedToBase ? cir::BinOpKind::Sub : cir::BinOpKind::Add;
  auto adjustedAdjField = builder.create<cir::BinOp>(
      op->getLoc(), ptrdiffCIRTy, binOpKind, adjField, offsetValue);

  return builder.create<cir::InsertMemberOp>(op->getLoc(), loweredSrc, 1,
                                             adjustedAdjField);
}

mlir::Value ItaniumCXXABI::lowerBaseDataMember(cir::BaseDataMemberOp op,
                                               mlir::Value loweredSrc,
                                               mlir::OpBuilder &builder) const {
  return lowerDataMemberCast(op, loweredSrc, op.getOffset().getSExtValue(),
                             /*isDerivedToBase=*/true, builder);
}

mlir::Value
ItaniumCXXABI::lowerDerivedDataMember(cir::DerivedDataMemberOp op,
                                      mlir::Value loweredSrc,
                                      mlir::OpBuilder &builder) const {
  return lowerDataMemberCast(op, loweredSrc, op.getOffset().getSExtValue(),
                             /*isDerivedToBase=*/false, builder);
}

mlir::Value ItaniumCXXABI::lowerBaseMethod(cir::BaseMethodOp op,
                                           mlir::Value loweredSrc,
                                           mlir::OpBuilder &builder) const {
  return lowerMethodCast(op, loweredSrc, op.getOffset().getSExtValue(),
                         /*isDerivedToBase=*/true, LM, builder);
}

mlir::Value ItaniumCXXABI::lowerDerivedMethod(cir::DerivedMethodOp op,
                                              mlir::Value loweredSrc,
                                              mlir::OpBuilder &builder) const {
  return lowerMethodCast(op, loweredSrc, op.getOffset().getSExtValue(),
                         /*isDerivedToBase=*/false, LM, builder);
}

mlir::Value ItaniumCXXABI::lowerDataMemberCmp(cir::CmpOp op,
                                              mlir::Value loweredLhs,
                                              mlir::Value loweredRhs,
                                              mlir::OpBuilder &builder) const {
  return builder.create<cir::CmpOp>(op.getLoc(), op.getKind(), loweredLhs,
                                    loweredRhs);
}

mlir::Value ItaniumCXXABI::lowerMethodCmp(cir::CmpOp op, mlir::Value loweredLhs,
                                          mlir::Value loweredRhs,
                                          mlir::OpBuilder &builder) const {
  assert(op.getKind() == cir::CmpOpKind::eq ||
         op.getKind() == cir::CmpOpKind::ne);

  cir::IntType ptrdiffCIRTy = getPtrDiffCIRTy(LM);
  mlir::Value ptrdiffZero = builder.create<cir::ConstantOp>(
      op.getLoc(), ptrdiffCIRTy, cir::IntAttr::get(ptrdiffCIRTy, 0));

  mlir::Value lhsPtrField = builder.create<cir::ExtractMemberOp>(
      op.getLoc(), ptrdiffCIRTy, loweredLhs, 0);
  mlir::Value rhsPtrField = builder.create<cir::ExtractMemberOp>(
      op.getLoc(), ptrdiffCIRTy, loweredRhs, 0);
  mlir::Value ptrCmp = builder.create<cir::CmpOp>(op.getLoc(), op.getKind(),
                                                  lhsPtrField, rhsPtrField);
  mlir::Value ptrCmpToNull = builder.create<cir::CmpOp>(
      op.getLoc(), op.getKind(), lhsPtrField, ptrdiffZero);

  mlir::Value lhsAdjField = builder.create<cir::ExtractMemberOp>(
      op.getLoc(), ptrdiffCIRTy, loweredLhs, 1);
  mlir::Value rhsAdjField = builder.create<cir::ExtractMemberOp>(
      op.getLoc(), ptrdiffCIRTy, loweredRhs, 1);
  mlir::Value adjCmp = builder.create<cir::CmpOp>(op.getLoc(), op.getKind(),
                                                  lhsAdjField, rhsAdjField);

  // We use cir.select to represent "||" and "&&" operations below:
  //   - cir.select if %a then %b else false => %a && %b
  //   - cir.select if %a then true else %b  => %a || %b
  // TODO: Do we need to invent dedicated "cir.logical_or" and "cir.logical_and"
  // operations for this?
  auto boolTy = cir::BoolType::get(op.getContext());
  mlir::Value trueValue = builder.create<cir::ConstantOp>(
      op.getLoc(), boolTy, cir::BoolAttr::get(op.getContext(), boolTy, true));
  mlir::Value falseValue = builder.create<cir::ConstantOp>(
      op.getLoc(), boolTy, cir::BoolAttr::get(op.getContext(), boolTy, false));
  auto create_and = [&](mlir::Value lhs, mlir::Value rhs) {
    return builder.create<cir::SelectOp>(op.getLoc(), lhs, rhs, falseValue);
  };
  auto create_or = [&](mlir::Value lhs, mlir::Value rhs) {
    return builder.create<cir::SelectOp>(op.getLoc(), lhs, trueValue, rhs);
  };

  mlir::Value result;
  if (op.getKind() == cir::CmpOpKind::eq) {
    // (lhs.ptr == null || lhs.adj == rhs.adj) && lhs.ptr == rhs.ptr
    result = create_and(create_or(ptrCmpToNull, adjCmp), ptrCmp);
  } else {
    // (lhs.ptr != null && lhs.adj != rhs.adj) || lhs.ptr != rhs.ptr
    result = create_or(create_and(ptrCmpToNull, adjCmp), ptrCmp);
  }

  return result;
}

mlir::Value
ItaniumCXXABI::lowerDataMemberBitcast(cir::CastOp op, mlir::Type loweredDstTy,
                                      mlir::Value loweredSrc,
                                      mlir::OpBuilder &builder) const {
  return builder.create<cir::CastOp>(op.getLoc(), loweredDstTy,
                                     cir::CastKind::bitcast, loweredSrc);
}

mlir::Value
ItaniumCXXABI::lowerDataMemberToBoolCast(cir::CastOp op, mlir::Value loweredSrc,
                                         mlir::OpBuilder &builder) const {
  // Itanium C++ ABI 2.3:
  //   A NULL pointer is represented as -1.
  auto nullAttr = cir::IntAttr::get(getPtrDiffCIRTy(LM), -1);
  auto nullValue = builder.create<cir::ConstantOp>(op.getLoc(), nullAttr);
  return builder.create<cir::CmpOp>(op.getLoc(), cir::CmpOpKind::ne, loweredSrc,
                                    nullValue);
}

mlir::Value ItaniumCXXABI::lowerMethodBitcast(cir::CastOp op,
                                              mlir::Type loweredDstTy,
                                              mlir::Value loweredSrc,
                                              mlir::OpBuilder &builder) const {
  return loweredSrc;
}

mlir::Value
ItaniumCXXABI::lowerMethodToBoolCast(cir::CastOp op, mlir::Value loweredSrc,
                                     mlir::OpBuilder &builder) const {
  // Itanium C++ ABI 2.3.2:
  //
  //   In the standard representation, a null member function pointer is
  //   represented with ptr set to a null pointer. The value of adj is
  //   unspecified for null member function pointers.
  cir::IntType ptrdiffCIRTy = getPtrDiffCIRTy(LM);
  mlir::Value ptrdiffZero = builder.create<cir::ConstantOp>(
      op.getLoc(), ptrdiffCIRTy, cir::IntAttr::get(ptrdiffCIRTy, 0));
  mlir::Value ptrField = builder.create<cir::ExtractMemberOp>(
      op.getLoc(), ptrdiffCIRTy, loweredSrc, 0);
  return builder.create<cir::CmpOp>(op.getLoc(), cir::CmpOpKind::ne, ptrField,
                                    ptrdiffZero);
}

CIRCXXABI *CreateItaniumCXXABI(LowerModule &LM) {
  switch (LM.getCXXABIKind()) {
  // Note that AArch64 uses the generic ItaniumCXXABI class since it doesn't
  // include the other 32-bit ARM oddities: constructor/destructor return values
  // and array cookies.
  case clang::TargetCXXABI::GenericAArch64:
  case clang::TargetCXXABI::AppleARM64:
    // TODO: this isn't quite right, clang uses AppleARM64CXXABI which inherits
    // from ARMCXXABI. We'll have to follow suit.
    cir_cconv_assert(!cir::MissingFeatures::appleArm64CXXABI());
    return new ItaniumCXXABI(LM, /*UseARMMethodPtrABI=*/true,
                             /*UseARMGuardVarABI=*/true);

  case clang::TargetCXXABI::GenericItanium:
    return new ItaniumCXXABI(LM);

  case clang::TargetCXXABI::Microsoft:
    cir_cconv_unreachable("Microsoft ABI is not Itanium-based");
  default:
    cir_cconv_unreachable("NYI");
  }

  cir_cconv_unreachable("bad ABI kind");
}

} // namespace cir

// FIXME(cir): Merge this into the CIRCXXABI class above.
class LoweringPrepareItaniumCXXABI : public cir::LoweringPrepareCXXABI {
public:
  mlir::Value lowerDynamicCast(cir::CIRBaseBuilderTy &builder,
                               clang::ASTContext &astCtx,
                               cir::DynamicCastOp op) override;
  mlir::Value lowerVAArg(cir::CIRBaseBuilderTy &builder, cir::VAArgOp op,
                         const cir::CIRDataLayout &datalayout) override;
  mlir::Value lowerDeleteArray(cir::CIRBaseBuilderTy &builder,
                               cir::DeleteArrayOp op,
                               const cir::CIRDataLayout &datalayout) override;
};
