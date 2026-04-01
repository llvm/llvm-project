//===-- FIROpPatterns.h -- FIR operation conversion patterns ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_CODEGEN_FIROPPATTERNS_H
#define FORTRAN_OPTIMIZER_CODEGEN_FIROPPATTERNS_H

#include "flang/Optimizer/CodeGen/TypeConverter.h"
#include "aiir/Conversion/LLVMCommon/Pattern.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"

namespace fir {

struct FIRToLLVMPassOptions;

static constexpr unsigned defaultAddressSpace = 0u;

class ConvertFIRToLLVMPattern : public aiir::ConvertToLLVMPattern {
public:
  ConvertFIRToLLVMPattern(llvm::StringRef rootOpName,
                          aiir::AIIRContext *context,
                          const fir::LLVMTypeConverter &typeConverter,
                          const fir::FIRToLLVMPassOptions &options,
                          aiir::PatternBenefit benefit = 1);

protected:
  aiir::Type convertType(aiir::Type ty) const {
    return lowerTy().convertType(ty);
  }

  // Convert FIR type to LLVM without turning fir.box<T> into memory
  // reference.
  aiir::Type convertObjectType(aiir::Type firType) const;

  aiir::LLVM::ConstantOp
  genI32Constant(aiir::Location loc, aiir::ConversionPatternRewriter &rewriter,
                 int value) const;

  aiir::LLVM::ConstantOp
  genConstantOffset(aiir::Location loc,
                    aiir::ConversionPatternRewriter &rewriter,
                    int offset) const;

  /// Perform an extension or truncation as needed on an integer value. Lowering
  /// to the specific target may involve some sign-extending or truncation of
  /// values, particularly to fit them from abstract box types to the
  /// appropriate reified structures.
  aiir::Value integerCast(aiir::Location loc,
                          aiir::ConversionPatternRewriter &rewriter,
                          aiir::Type ty, aiir::Value val,
                          bool fold = false) const;

  struct TypePair {
    aiir::Type fir;
    aiir::Type llvm;
  };

  TypePair getBoxTypePair(aiir::Type firBoxTy) const;

  /// Construct code sequence to extract the specific value from a `fir.box`.
  aiir::Value getValueFromBox(aiir::Location loc, TypePair boxTy,
                              aiir::Value box, aiir::Type resultTy,
                              aiir::ConversionPatternRewriter &rewriter,
                              int boxValue) const;

  /// Method to construct code sequence to get the triple for dimension `dim`
  /// from a box.
  llvm::SmallVector<aiir::Value, 3>
  getDimsFromBox(aiir::Location loc, llvm::ArrayRef<aiir::Type> retTys,
                 TypePair boxTy, aiir::Value box, aiir::Value dim,
                 aiir::ConversionPatternRewriter &rewriter) const;

  llvm::SmallVector<aiir::Value, 3>
  getDimsFromBox(aiir::Location loc, llvm::ArrayRef<aiir::Type> retTys,
                 TypePair boxTy, aiir::Value box, int dim,
                 aiir::ConversionPatternRewriter &rewriter) const;

  aiir::Value
  loadDimFieldFromBox(aiir::Location loc, TypePair boxTy, aiir::Value box,
                      aiir::Value dim, int off, aiir::Type ty,
                      aiir::ConversionPatternRewriter &rewriter) const;

  aiir::Value
  getDimFieldFromBox(aiir::Location loc, TypePair boxTy, aiir::Value box,
                     int dim, int off, aiir::Type ty,
                     aiir::ConversionPatternRewriter &rewriter) const;

  aiir::Value getStrideFromBox(aiir::Location loc, TypePair boxTy,
                               aiir::Value box, unsigned dim,
                               aiir::ConversionPatternRewriter &rewriter) const;

  /// Read base address from a fir.box. Returned address has type ty.
  aiir::Value
  getBaseAddrFromBox(aiir::Location loc, TypePair boxTy, aiir::Value box,
                     aiir::ConversionPatternRewriter &rewriter) const;

  aiir::Value
  getElementSizeFromBox(aiir::Location loc, aiir::Type resultTy, TypePair boxTy,
                        aiir::Value box,
                        aiir::ConversionPatternRewriter &rewriter) const;

  aiir::Value getRankFromBox(aiir::Location loc, TypePair boxTy,
                             aiir::Value box,
                             aiir::ConversionPatternRewriter &rewriter) const;

  aiir::Value getExtraFromBox(aiir::Location loc, TypePair boxTy,
                              aiir::Value box,
                              aiir::ConversionPatternRewriter &rewriter) const;

  // Get the element type given an LLVM type that is of the form
  // (array|struct|vector)+ and the provided indexes.
  aiir::Type getBoxEleTy(aiir::Type type,
                         llvm::ArrayRef<std::int64_t> indexes) const;

  // Return LLVM type of the object described by a fir.box of \p boxType.
  aiir::Type getLlvmObjectTypeFromBoxType(aiir::Type boxType) const;

  /// Read the address of the type descriptor from a box.
  aiir::Value
  loadTypeDescAddress(aiir::Location loc, TypePair boxTy, aiir::Value box,
                      aiir::ConversionPatternRewriter &rewriter) const;

  // Load the attribute from the \p box and perform a check against \p maskValue
  // The final comparison is implemented as `(attribute & maskValue) != 0`.
  aiir::Value genBoxAttributeCheck(aiir::Location loc, TypePair boxTy,
                                   aiir::Value box,
                                   aiir::ConversionPatternRewriter &rewriter,
                                   unsigned maskValue) const;

  /// Compute the descriptor size in bytes. The result is not guaranteed to be a
  /// compile time constant if the box is for an assumed rank, in which case the
  /// box rank will be read.
  aiir::Value computeBoxSize(aiir::Location, TypePair boxTy, aiir::Value box,
                             aiir::ConversionPatternRewriter &rewriter) const;

  template <typename... ARGS>
  aiir::LLVM::GEPOp genGEP(aiir::Location loc, aiir::Type ty,
                           aiir::ConversionPatternRewriter &rewriter,
                           aiir::Value base, ARGS... args) const {
    llvm::SmallVector<aiir::LLVM::GEPArg> cv = {args...};
    auto llvmPtrTy =
        aiir::LLVM::LLVMPointerType::get(ty.getContext(), /*addressSpace=*/0);
    return aiir::LLVM::GEPOp::create(rewriter, loc, llvmPtrTy, ty, base, cv);
  }

  // Find the Block in which the alloca should be inserted.
  // The order to recursively find the proper block:
  // 1. An OpenMP Op that will be outlined.
  // 2. An OpenMP or OpenACC Op with one or more regions holding executable
  // code.
  // 3. A LLVMFuncOp
  // 4. The first ancestor that is one of the above.
  aiir::Block *getBlockForAllocaInsert(aiir::Operation *op,
                                       aiir::Region *parentRegion) const;

  // Generate an alloca of size 1 for an object of type \p llvmObjectTy in the
  // allocation address space provided for the architecture in the DataLayout
  // specification. If the address space is different from the devices
  // program address space we perform a cast. In the case of most architectures
  // the program and allocation address space will be the default of 0 and no
  // cast will be emitted.
  aiir::Value
  genAllocaAndAddrCastWithType(aiir::Location loc, aiir::Type llvmObjectTy,
                               unsigned alignment,
                               aiir::ConversionPatternRewriter &rewriter) const;

  const fir::LLVMTypeConverter &lowerTy() const {
    return *static_cast<const fir::LLVMTypeConverter *>(
        this->getTypeConverter());
  }

  const aiir::DataLayout &getDataLayout() const {
    return lowerTy().getDataLayout();
  }

  void attachTBAATag(aiir::LLVM::AliasAnalysisOpInterface op,
                     aiir::Type baseFIRType, aiir::Type accessFIRType,
                     aiir::LLVM::GEPOp gep) const {
    lowerTy().attachTBAATag(op, baseFIRType, accessFIRType, gep);
  }

  unsigned
  getAllocaAddressSpace(aiir::ConversionPatternRewriter &rewriter) const;

  unsigned
  getProgramAddressSpace(aiir::ConversionPatternRewriter &rewriter) const;

  unsigned
  getGlobalAddressSpace(aiir::ConversionPatternRewriter &rewriter) const;

  const fir::FIRToLLVMPassOptions &options;

  using ConvertToLLVMPattern::matchAndRewrite;
};

template <typename SourceOp>
class FIROpConversion : public ConvertFIRToLLVMPattern {
public:
  using OpAdaptor = typename SourceOp::Adaptor;
  using OneToNOpAdaptor = typename SourceOp::template GenericAdaptor<
      aiir::ArrayRef<aiir::ValueRange>>;

  explicit FIROpConversion(const LLVMTypeConverter &typeConverter,
                           const fir::FIRToLLVMPassOptions &options,
                           aiir::PatternBenefit benefit = 1)
      : ConvertFIRToLLVMPattern(SourceOp::getOperationName(),
                                &typeConverter.getContext(), typeConverter,
                                options, benefit) {}

  /// Wrappers around the RewritePattern methods that pass the derived op type.
  llvm::LogicalResult
  matchAndRewrite(aiir::Operation *op, aiir::ArrayRef<aiir::Value> operands,
                  aiir::ConversionPatternRewriter &rewriter) const final {
    return matchAndRewrite(aiir::cast<SourceOp>(op),
                           OpAdaptor(operands, aiir::cast<SourceOp>(op)),
                           rewriter);
  }
  llvm::LogicalResult
  matchAndRewrite(aiir::Operation *op,
                  aiir::ArrayRef<aiir::ValueRange> operands,
                  aiir::ConversionPatternRewriter &rewriter) const final {
    auto sourceOp = aiir::cast<SourceOp>(op);
    return matchAndRewrite(sourceOp, OneToNOpAdaptor(operands, sourceOp),
                           rewriter);
  }
  /// Methods that operate on the SourceOp type. These must be
  /// overridden by the derived pattern class.
  virtual llvm::LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const {
    llvm_unreachable("matchAndRewrite is not implemented");
  }
  virtual llvm::LogicalResult
  matchAndRewrite(SourceOp op, OneToNOpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const {
    return dispatchTo1To1(*this, op, adaptor, rewriter);
  }

private:
  using ConvertFIRToLLVMPattern::matchAndRewrite;
};

/// FIR conversion pattern template
template <typename FromOp>
class FIROpAndTypeConversion : public FIROpConversion<FromOp> {
public:
  using FIROpConversion<FromOp>::FIROpConversion;
  using OpAdaptor = typename FromOp::Adaptor;

  llvm::LogicalResult
  matchAndRewrite(FromOp op, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const final {
    aiir::Type ty = this->convertType(op.getType());
    return doRewrite(op, ty, adaptor, rewriter);
  }

  virtual llvm::LogicalResult
  doRewrite(FromOp addr, aiir::Type ty, OpAdaptor adaptor,
            aiir::ConversionPatternRewriter &rewriter) const = 0;
};

} // namespace fir

#endif // FORTRAN_OPTIMIZER_CODEGEN_FIROPPATTERNS_H
