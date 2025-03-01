//====- LowerToLLVM.h - Lowering from CIR to LLVMIR -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LowerModule.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

namespace cir {
namespace direct {

/// Convert a CIR attribute to an LLVM attribute. May use the datalayout for
/// lowering attributes to-be-stored in memory.
mlir::Value lowerCirAttrAsValue(mlir::Operation *parentOp,
                                const mlir::Attribute attr,
                                mlir::ConversionPatternRewriter &rewriter,
                                const mlir::TypeConverter *converter,
                                mlir::DataLayout const &dataLayout);

mlir::LLVM::Linkage convertLinkage(cir::GlobalLinkageKind linkage);

mlir::LLVM::CConv convertCallingConv(cir::CallingConv callinvConv);

void convertSideEffectForCall(mlir::Operation *callOp,
                              cir::SideEffect sideEffect,
                              mlir::LLVM::MemoryEffectsAttr &memoryEffect,
                              bool &noUnwind, bool &willReturn);

void buildCtorDtorList(
    mlir::ModuleOp module, mlir::StringRef globalXtorName,
    mlir::StringRef llvmXtorName,
    llvm::function_ref<std::pair<mlir::StringRef, int>(mlir::Attribute)>
        createXtor);

void populateCIRToLLVMConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::TypeConverter &converter,
    mlir::DataLayout &dataLayout,
    llvm::StringMap<mlir::LLVM::GlobalOp> &stringGlobalsMap,
    llvm::StringMap<mlir::LLVM::GlobalOp> &argStringGlobalsMap,
    llvm::MapVector<mlir::ArrayAttr, mlir::LLVM::GlobalOp> &argsVarMap);

std::unique_ptr<cir::LowerModule> prepareLowerModule(mlir::ModuleOp module);

void prepareTypeConverter(mlir::LLVMTypeConverter &converter,
                          mlir::DataLayout &dataLayout,
                          cir::LowerModule *lowerModule);

mlir::LLVM::AtomicOrdering
getLLVMMemOrder(std::optional<cir::MemOrder> &memorder);

mlir::LLVM::AtomicOrdering getLLVMAtomicOrder(cir::MemOrder memo);

mlir::LLVM::CallIntrinsicOp
createCallLLVMIntrinsicOp(mlir::ConversionPatternRewriter &rewriter,
                          mlir::Location loc, const llvm::Twine &intrinsicName,
                          mlir::Type resultTy, mlir::ValueRange operands);

mlir::LLVM::CallIntrinsicOp replaceOpWithCallLLVMIntrinsicOp(
    mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op,
    const llvm::Twine &intrinsicName, mlir::Type resultTy,
    mlir::ValueRange operands);

mlir::Value createLLVMBitOp(mlir::Location loc,
                            const llvm::Twine &llvmIntrinBaseName,
                            mlir::Type resultTy, mlir::Value operand,
                            std::optional<bool> poisonZeroInputFlag,
                            mlir::ConversionPatternRewriter &rewriter);

class CIRToLLVMCopyOpLowering : public mlir::OpConversionPattern<cir::CopyOp> {
public:
  using mlir::OpConversionPattern<cir::CopyOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::CopyOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMMemCpyOpLowering
    : public mlir::OpConversionPattern<cir::MemCpyOp> {
public:
  using mlir::OpConversionPattern<cir::MemCpyOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::MemCpyOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMMemChrOpLowering
    : public mlir::OpConversionPattern<cir::MemChrOp> {
public:
  using mlir::OpConversionPattern<cir::MemChrOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::MemChrOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMMemMoveOpLowering
    : public mlir::OpConversionPattern<cir::MemMoveOp> {
public:
  using mlir::OpConversionPattern<cir::MemMoveOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::MemMoveOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMMemCpyInlineOpLowering
    : public mlir::OpConversionPattern<cir::MemCpyInlineOp> {
public:
  using mlir::OpConversionPattern<cir::MemCpyInlineOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::MemCpyInlineOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

class CIRToLLVMMemSetOpLowering
    : public mlir::OpConversionPattern<cir::MemSetOp> {
public:
  using mlir::OpConversionPattern<cir::MemSetOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::MemSetOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMMemSetInlineOpLowering
    : public mlir::OpConversionPattern<cir::MemSetInlineOp> {
public:
  using mlir::OpConversionPattern<cir::MemSetInlineOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::MemSetInlineOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

class CIRToLLVMPtrStrideOpLowering
    : public mlir::OpConversionPattern<cir::PtrStrideOp> {
  mlir::DataLayout const &dataLayout;

public:
  CIRToLLVMPtrStrideOpLowering(const mlir::TypeConverter &typeConverter,
                               mlir::MLIRContext *context,
                               mlir::DataLayout const &dataLayout)
      : OpConversionPattern(typeConverter, context), dataLayout(dataLayout) {}
  using mlir::OpConversionPattern<cir::PtrStrideOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::PtrStrideOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMBaseClassAddrOpLowering
    : public mlir::OpConversionPattern<cir::BaseClassAddrOp> {
public:
  using mlir::OpConversionPattern<cir::BaseClassAddrOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BaseClassAddrOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMDerivedClassAddrOpLowering
    : public mlir::OpConversionPattern<cir::DerivedClassAddrOp> {
public:
  using mlir::OpConversionPattern<cir::DerivedClassAddrOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::DerivedClassAddrOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMBaseDataMemberOpLowering
    : public mlir::OpConversionPattern<cir::BaseDataMemberOp> {
  cir::LowerModule *lowerMod;

public:
  CIRToLLVMBaseDataMemberOpLowering(const mlir::TypeConverter &typeConverter,
                                    mlir::MLIRContext *context,
                                    cir::LowerModule *lowerModule)
      : OpConversionPattern(typeConverter, context), lowerMod(lowerModule) {}

  mlir::LogicalResult
  matchAndRewrite(cir::BaseDataMemberOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMDerivedDataMemberOpLowering
    : public mlir::OpConversionPattern<cir::DerivedDataMemberOp> {
  cir::LowerModule *lowerMod;

public:
  CIRToLLVMDerivedDataMemberOpLowering(const mlir::TypeConverter &typeConverter,
                                       mlir::MLIRContext *context,
                                       cir::LowerModule *lowerModule)
      : OpConversionPattern(typeConverter, context), lowerMod(lowerModule) {}

  mlir::LogicalResult
  matchAndRewrite(cir::DerivedDataMemberOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMBaseMethodOpLowering
    : public mlir::OpConversionPattern<cir::BaseMethodOp> {
  cir::LowerModule *lowerMod;

public:
  CIRToLLVMBaseMethodOpLowering(const mlir::TypeConverter &typeConverter,
                                mlir::MLIRContext *context,
                                cir::LowerModule *lowerModule)
      : OpConversionPattern(typeConverter, context), lowerMod(lowerModule) {}

  mlir::LogicalResult
  matchAndRewrite(cir::BaseMethodOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMDerivedMethodOpLowering
    : public mlir::OpConversionPattern<cir::DerivedMethodOp> {
  cir::LowerModule *lowerMod;

public:
  CIRToLLVMDerivedMethodOpLowering(const mlir::TypeConverter &typeConverter,
                                   mlir::MLIRContext *context,
                                   cir::LowerModule *lowerModule)
      : OpConversionPattern(typeConverter, context), lowerMod(lowerModule) {}

  mlir::LogicalResult
  matchAndRewrite(cir::DerivedMethodOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMVTTAddrPointOpLowering
    : public mlir::OpConversionPattern<cir::VTTAddrPointOp> {
public:
  using mlir::OpConversionPattern<cir::VTTAddrPointOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VTTAddrPointOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMBrCondOpLowering
    : public mlir::OpConversionPattern<cir::BrCondOp> {
public:
  using mlir::OpConversionPattern<cir::BrCondOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BrCondOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMCastOpLowering : public mlir::OpConversionPattern<cir::CastOp> {
  cir::LowerModule *lowerMod;
  mlir::DataLayout const &dataLayout;

  mlir::Type convertTy(mlir::Type ty) const;

public:
  CIRToLLVMCastOpLowering(const mlir::TypeConverter &typeConverter,
                          mlir::MLIRContext *context,
                          cir::LowerModule *lowerModule,
                          mlir::DataLayout const &dataLayout)
      : OpConversionPattern(typeConverter, context), lowerMod(lowerModule),
        dataLayout(dataLayout) {}

  mlir::LogicalResult
  matchAndRewrite(cir::CastOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMReturnOpLowering
    : public mlir::OpConversionPattern<cir::ReturnOp> {
public:
  using mlir::OpConversionPattern<cir::ReturnOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ReturnOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMCallOpLowering : public mlir::OpConversionPattern<cir::CallOp> {
public:
  using mlir::OpConversionPattern<cir::CallOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::CallOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMTryCallOpLowering
    : public mlir::OpConversionPattern<cir::TryCallOp> {
public:
  using mlir::OpConversionPattern<cir::TryCallOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::TryCallOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMEhInflightOpLowering
    : public mlir::OpConversionPattern<cir::EhInflightOp> {
public:
  using mlir::OpConversionPattern<cir::EhInflightOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::EhInflightOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMAllocaOpLowering
    : public mlir::OpConversionPattern<cir::AllocaOp> {
  mlir::DataLayout const &dataLayout;
  // Track globals created for annotation related strings
  llvm::StringMap<mlir::LLVM::GlobalOp> &stringGlobalsMap;
  // Track globals created for annotation arg related strings.
  // They are different from annotation strings, as strings used in args
  // are not in llvmMetadataSectionName, and also has aligment 1.
  llvm::StringMap<mlir::LLVM::GlobalOp> &argStringGlobalsMap;
  // Track globals created for annotation args.
  llvm::MapVector<mlir::ArrayAttr, mlir::LLVM::GlobalOp> &argsVarMap;

public:
  CIRToLLVMAllocaOpLowering(
      mlir::TypeConverter const &typeConverter,
      mlir::DataLayout const &dataLayout,
      llvm::StringMap<mlir::LLVM::GlobalOp> &stringGlobalsMap,
      llvm::StringMap<mlir::LLVM::GlobalOp> &argStringGlobalsMap,
      llvm::MapVector<mlir::ArrayAttr, mlir::LLVM::GlobalOp> &argsVarMap,
      mlir::MLIRContext *context)
      : OpConversionPattern<cir::AllocaOp>(typeConverter, context),
        dataLayout(dataLayout), stringGlobalsMap(stringGlobalsMap),
        argStringGlobalsMap(argStringGlobalsMap), argsVarMap(argsVarMap) {}

  using mlir::OpConversionPattern<cir::AllocaOp>::OpConversionPattern;

  void buildAllocaAnnotations(mlir::LLVM::AllocaOp op, OpAdaptor adaptor,
                              mlir::ConversionPatternRewriter &rewriter,
                              mlir::ArrayAttr annotationValuesArray) const;

  mlir::LogicalResult
  matchAndRewrite(cir::AllocaOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMLoadOpLowering : public mlir::OpConversionPattern<cir::LoadOp> {
  cir::LowerModule *lowerMod;
  mlir::DataLayout const &dataLayout;

public:
  CIRToLLVMLoadOpLowering(const mlir::TypeConverter &typeConverter,
                          mlir::MLIRContext *context,
                          cir::LowerModule *lowerModule,
                          mlir::DataLayout const &dataLayout)
      : OpConversionPattern(typeConverter, context), lowerMod(lowerModule),
        dataLayout(dataLayout) {}

  mlir::LogicalResult
  matchAndRewrite(cir::LoadOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMStoreOpLowering
    : public mlir::OpConversionPattern<cir::StoreOp> {
  cir::LowerModule *lowerMod;
  mlir::DataLayout const &dataLayout;

public:
  CIRToLLVMStoreOpLowering(const mlir::TypeConverter &typeConverter,
                           mlir::MLIRContext *context,
                           cir::LowerModule *lowerModule,
                           mlir::DataLayout const &dataLayout)
      : OpConversionPattern(typeConverter, context), lowerMod(lowerModule),
        dataLayout(dataLayout) {}

  mlir::LogicalResult
  matchAndRewrite(cir::StoreOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMConstantOpLowering
    : public mlir::OpConversionPattern<cir::ConstantOp> {
  cir::LowerModule *lowerMod;
  mlir::DataLayout const &dataLayout;

public:
  CIRToLLVMConstantOpLowering(const mlir::TypeConverter &typeConverter,
                              mlir::MLIRContext *context,
                              cir::LowerModule *lowerModule,
                              mlir::DataLayout const &dataLayout)
      : OpConversionPattern(typeConverter, context), lowerMod(lowerModule),
        dataLayout(dataLayout) {
    setHasBoundedRewriteRecursion();
  }

  mlir::LogicalResult
  matchAndRewrite(cir::ConstantOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMVecCreateOpLowering
    : public mlir::OpConversionPattern<cir::VecCreateOp> {
public:
  using mlir::OpConversionPattern<cir::VecCreateOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VecCreateOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMVecCmpOpLowering
    : public mlir::OpConversionPattern<cir::VecCmpOp> {
public:
  using mlir::OpConversionPattern<cir::VecCmpOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VecCmpOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMVecSplatOpLowering
    : public mlir::OpConversionPattern<cir::VecSplatOp> {
public:
  using mlir::OpConversionPattern<cir::VecSplatOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VecSplatOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMVecTernaryOpLowering
    : public mlir::OpConversionPattern<cir::VecTernaryOp> {
public:
  using mlir::OpConversionPattern<cir::VecTernaryOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VecTernaryOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMVecShuffleOpLowering
    : public mlir::OpConversionPattern<cir::VecShuffleOp> {
public:
  using mlir::OpConversionPattern<cir::VecShuffleOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VecShuffleOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMVecShuffleDynamicOpLowering
    : public mlir::OpConversionPattern<cir::VecShuffleDynamicOp> {
public:
  using mlir::OpConversionPattern<
      cir::VecShuffleDynamicOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VecShuffleDynamicOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMVAStartOpLowering
    : public mlir::OpConversionPattern<cir::VAStartOp> {
public:
  using mlir::OpConversionPattern<cir::VAStartOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VAStartOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMVAEndOpLowering
    : public mlir::OpConversionPattern<cir::VAEndOp> {
public:
  using mlir::OpConversionPattern<cir::VAEndOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VAEndOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMVACopyOpLowering
    : public mlir::OpConversionPattern<cir::VACopyOp> {
public:
  using mlir::OpConversionPattern<cir::VACopyOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VACopyOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMVAArgOpLowering
    : public mlir::OpConversionPattern<cir::VAArgOp> {
public:
  using mlir::OpConversionPattern<cir::VAArgOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VAArgOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMFuncOpLowering : public mlir::OpConversionPattern<cir::FuncOp> {
  static mlir::StringRef getLinkageAttrNameString();

  void lowerFuncAttributes(
      cir::FuncOp func, bool filterArgAndResAttrs,
      mlir::SmallVectorImpl<mlir::NamedAttribute> &result) const;

  void
  lowerFuncOpenCLKernelMetadata(mlir::NamedAttribute &extraAttrsEntry) const;

public:
  using mlir::OpConversionPattern<cir::FuncOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::FuncOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMGetGlobalOpLowering
    : public mlir::OpConversionPattern<cir::GetGlobalOp> {
public:
  using mlir::OpConversionPattern<cir::GetGlobalOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::GetGlobalOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMComplexCreateOpLowering
    : public mlir::OpConversionPattern<cir::ComplexCreateOp> {
public:
  using mlir::OpConversionPattern<cir::ComplexCreateOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ComplexCreateOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMComplexRealOpLowering
    : public mlir::OpConversionPattern<cir::ComplexRealOp> {
public:
  using mlir::OpConversionPattern<cir::ComplexRealOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ComplexRealOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMComplexImagOpLowering
    : public mlir::OpConversionPattern<cir::ComplexImagOp> {
public:
  using mlir::OpConversionPattern<cir::ComplexImagOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ComplexImagOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMComplexRealPtrOpLowering
    : public mlir::OpConversionPattern<cir::ComplexRealPtrOp> {
public:
  using mlir::OpConversionPattern<cir::ComplexRealPtrOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ComplexRealPtrOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMComplexImagPtrOpLowering
    : public mlir::OpConversionPattern<cir::ComplexImagPtrOp> {
public:
  using mlir::OpConversionPattern<cir::ComplexImagPtrOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ComplexImagPtrOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMSwitchFlatOpLowering
    : public mlir::OpConversionPattern<cir::SwitchFlatOp> {
public:
  using mlir::OpConversionPattern<cir::SwitchFlatOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::SwitchFlatOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMGlobalOpLowering
    : public mlir::OpConversionPattern<cir::GlobalOp> {
  cir::LowerModule *lowerMod;
  mlir::DataLayout const &dataLayout;

public:
  CIRToLLVMGlobalOpLowering(const mlir::TypeConverter &typeConverter,
                            mlir::MLIRContext *context,
                            cir::LowerModule *lowerModule,
                            mlir::DataLayout const &dataLayout)
      : OpConversionPattern(typeConverter, context), lowerMod(lowerModule),
        dataLayout(dataLayout) {
    setHasBoundedRewriteRecursion();
  }

  mlir::LogicalResult
  matchAndRewrite(cir::GlobalOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;

private:
  void createRegionInitializedLLVMGlobalOp(
      cir::GlobalOp op, mlir::Attribute attr,
      mlir::ConversionPatternRewriter &rewriter) const;

  mutable mlir::LLVM::ComdatOp comdatOp = nullptr;
  static void addComdat(mlir::LLVM::GlobalOp &op,
                        mlir::LLVM::ComdatOp &comdatOp,
                        mlir::OpBuilder &builder, mlir::ModuleOp &module);
};

class CIRToLLVMUnaryOpLowering
    : public mlir::OpConversionPattern<cir::UnaryOp> {
public:
  using mlir::OpConversionPattern<cir::UnaryOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::UnaryOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMBinOpLowering : public mlir::OpConversionPattern<cir::BinOp> {
  mlir::LLVM::IntegerOverflowFlags getIntOverflowFlag(cir::BinOp op) const;

public:
  using mlir::OpConversionPattern<cir::BinOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BinOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMBinOpOverflowOpLowering
    : public mlir::OpConversionPattern<cir::BinOpOverflowOp> {
public:
  using mlir::OpConversionPattern<cir::BinOpOverflowOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BinOpOverflowOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;

private:
  static std::string getLLVMIntrinName(cir::BinOpOverflowKind opKind,
                                       bool isSigned, unsigned width);

  struct EncompassedTypeInfo {
    bool sign;
    unsigned width;
  };

  static EncompassedTypeInfo computeEncompassedTypeWidth(cir::IntType operandTy,
                                                         cir::IntType resultTy);
};

class CIRToLLVMShiftOpLowering
    : public mlir::OpConversionPattern<cir::ShiftOp> {
public:
  using mlir::OpConversionPattern<cir::ShiftOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ShiftOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMCmpOpLowering : public mlir::OpConversionPattern<cir::CmpOp> {
  cir::LowerModule *lowerMod;

public:
  CIRToLLVMCmpOpLowering(const mlir::TypeConverter &typeConverter,
                         mlir::MLIRContext *context,
                         cir::LowerModule *lowerModule)
      : OpConversionPattern(typeConverter, context), lowerMod(lowerModule) {
    setHasBoundedRewriteRecursion();
  }

  mlir::LogicalResult
  matchAndRewrite(cir::CmpOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMLLVMIntrinsicCallOpLowering
    : public mlir::OpConversionPattern<cir::LLVMIntrinsicCallOp> {
public:
  using mlir::OpConversionPattern<
      cir::LLVMIntrinsicCallOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::LLVMIntrinsicCallOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMAssumeOpLowering
    : public mlir::OpConversionPattern<cir::AssumeOp> {
public:
  using mlir::OpConversionPattern<cir::AssumeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::AssumeOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMAssumeAlignedOpLowering
    : public mlir::OpConversionPattern<cir::AssumeAlignedOp> {
public:
  using mlir::OpConversionPattern<cir::AssumeAlignedOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::AssumeAlignedOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMAssumeSepStorageOpLowering
    : public mlir::OpConversionPattern<cir::AssumeSepStorageOp> {
public:
  using mlir::OpConversionPattern<cir::AssumeSepStorageOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::AssumeSepStorageOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMBitClrsbOpLowering
    : public mlir::OpConversionPattern<cir::BitClrsbOp> {
public:
  using mlir::OpConversionPattern<cir::BitClrsbOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BitClrsbOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMObjSizeOpLowering
    : public mlir::OpConversionPattern<cir::ObjSizeOp> {
public:
  using mlir::OpConversionPattern<cir::ObjSizeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ObjSizeOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMBitClzOpLowering
    : public mlir::OpConversionPattern<cir::BitClzOp> {
public:
  using mlir::OpConversionPattern<cir::BitClzOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BitClzOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMBitCtzOpLowering
    : public mlir::OpConversionPattern<cir::BitCtzOp> {
public:
  using mlir::OpConversionPattern<cir::BitCtzOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BitCtzOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMBitFfsOpLowering
    : public mlir::OpConversionPattern<cir::BitFfsOp> {
public:
  using mlir::OpConversionPattern<cir::BitFfsOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BitFfsOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMBitParityOpLowering
    : public mlir::OpConversionPattern<cir::BitParityOp> {
public:
  using mlir::OpConversionPattern<cir::BitParityOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BitParityOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMBitPopcountOpLowering
    : public mlir::OpConversionPattern<cir::BitPopcountOp> {
public:
  using mlir::OpConversionPattern<cir::BitPopcountOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BitPopcountOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMAtomicCmpXchgLowering
    : public mlir::OpConversionPattern<cir::AtomicCmpXchg> {
public:
  using mlir::OpConversionPattern<cir::AtomicCmpXchg>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::AtomicCmpXchg op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMAtomicXchgLowering
    : public mlir::OpConversionPattern<cir::AtomicXchg> {
public:
  using mlir::OpConversionPattern<cir::AtomicXchg>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::AtomicXchg op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMAtomicFetchLowering
    : public mlir::OpConversionPattern<cir::AtomicFetch> {
  mlir::Value buildPostOp(cir::AtomicFetch op, OpAdaptor adaptor,
                          mlir::ConversionPatternRewriter &rewriter,
                          mlir::Value rmwVal, bool isInt) const;

  mlir::Value buildMinMaxPostOp(cir::AtomicFetch op, OpAdaptor adaptor,
                                mlir::ConversionPatternRewriter &rewriter,
                                mlir::Value rmwVal, bool isSigned) const;

  llvm::StringLiteral getLLVMBinop(cir::AtomicFetchKind k, bool isInt) const;

  mlir::LLVM::AtomicBinOp getLLVMAtomicBinOp(cir::AtomicFetchKind k, bool isInt,
                                             bool isSignedInt) const;

public:
  using mlir::OpConversionPattern<cir::AtomicFetch>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::AtomicFetch op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMAtomicFenceLowering
    : public mlir::OpConversionPattern<cir::AtomicFence> {
public:
  using mlir::OpConversionPattern<cir::AtomicFence>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::AtomicFence op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMByteswapOpLowering
    : public mlir::OpConversionPattern<cir::ByteswapOp> {
public:
  using mlir::OpConversionPattern<cir::ByteswapOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ByteswapOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMRotateOpLowering
    : public mlir::OpConversionPattern<cir::RotateOp> {
public:
  using mlir::OpConversionPattern<cir::RotateOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::RotateOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMSelectOpLowering
    : public mlir::OpConversionPattern<cir::SelectOp> {
public:
  using mlir::OpConversionPattern<cir::SelectOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::SelectOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMBrOpLowering : public mlir::OpConversionPattern<cir::BrOp> {
public:
  using mlir::OpConversionPattern<cir::BrOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BrOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMGetMemberOpLowering
    : public mlir::OpConversionPattern<cir::GetMemberOp> {
public:
  using mlir::OpConversionPattern<cir::GetMemberOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::GetMemberOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMExtractMemberOpLowering
    : public mlir::OpConversionPattern<cir::ExtractMemberOp> {
public:
  using mlir::OpConversionPattern<cir::ExtractMemberOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ExtractMemberOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMInsertMemberOpLowering
    : public mlir::OpConversionPattern<cir::InsertMemberOp> {
public:
  using mlir::OpConversionPattern<cir::InsertMemberOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::InsertMemberOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMGetMethodOpLowering
    : public mlir::OpConversionPattern<cir::GetMethodOp> {
  cir::LowerModule *lowerMod;

public:
  CIRToLLVMGetMethodOpLowering(const mlir::TypeConverter &typeConverter,
                               mlir::MLIRContext *context,
                               cir::LowerModule *lowerModule)
      : OpConversionPattern(typeConverter, context), lowerMod(lowerModule) {}

  mlir::LogicalResult
  matchAndRewrite(cir::GetMethodOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMGetRuntimeMemberOpLowering
    : public mlir::OpConversionPattern<cir::GetRuntimeMemberOp> {
  cir::LowerModule *lowerMod;

public:
  CIRToLLVMGetRuntimeMemberOpLowering(const mlir::TypeConverter &typeConverter,
                                      mlir::MLIRContext *context,
                                      cir::LowerModule *lowerModule)
      : OpConversionPattern(typeConverter, context), lowerMod(lowerModule) {}

  mlir::LogicalResult
  matchAndRewrite(cir::GetRuntimeMemberOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMPtrDiffOpLowering
    : public mlir::OpConversionPattern<cir::PtrDiffOp> {
  uint64_t getTypeSize(mlir::Type type, mlir::Operation &op) const;

public:
  using mlir::OpConversionPattern<cir::PtrDiffOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::PtrDiffOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMExpectOpLowering
    : public mlir::OpConversionPattern<cir::ExpectOp> {
public:
  using mlir::OpConversionPattern<cir::ExpectOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ExpectOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMVTableAddrPointOpLowering
    : public mlir::OpConversionPattern<cir::VTableAddrPointOp> {
public:
  using mlir::OpConversionPattern<cir::VTableAddrPointOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VTableAddrPointOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMStackSaveOpLowering
    : public mlir::OpConversionPattern<cir::StackSaveOp> {
public:
  using mlir::OpConversionPattern<cir::StackSaveOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::StackSaveOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMUnreachableOpLowering
    : public mlir::OpConversionPattern<cir::UnreachableOp> {
public:
  using mlir::OpConversionPattern<cir::UnreachableOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::UnreachableOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMTrapOpLowering : public mlir::OpConversionPattern<cir::TrapOp> {
public:
  using mlir::OpConversionPattern<cir::TrapOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::TrapOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMInlineAsmOpLowering
    : public mlir::OpConversionPattern<cir::InlineAsmOp> {
  mlir::DataLayout const &dataLayout;

public:
  CIRToLLVMInlineAsmOpLowering(const mlir::TypeConverter &typeConverter,
                               mlir::MLIRContext *context,
                               mlir::DataLayout const &dataLayout)
      : OpConversionPattern(typeConverter, context), dataLayout(dataLayout) {}

  using mlir::OpConversionPattern<cir::InlineAsmOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::InlineAsmOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMInvariantGroupOpLowering
    : public mlir::OpConversionPattern<cir::InvariantGroupOp> {
  cir::LowerModule *lowerMod;

public:
  CIRToLLVMInvariantGroupOpLowering(const mlir::TypeConverter &typeConverter,
                                    mlir::MLIRContext *context,
                                    cir::LowerModule *lowerModule)
      : OpConversionPattern(typeConverter, context), lowerMod(lowerModule) {}

  mlir::LogicalResult
  matchAndRewrite(cir::InvariantGroupOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMPrefetchOpLowering
    : public mlir::OpConversionPattern<cir::PrefetchOp> {
public:
  using mlir::OpConversionPattern<cir::PrefetchOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::PrefetchOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMSetBitfieldOpLowering
    : public mlir::OpConversionPattern<cir::SetBitfieldOp> {
public:
  using mlir::OpConversionPattern<cir::SetBitfieldOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::SetBitfieldOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMGetBitfieldOpLowering
    : public mlir::OpConversionPattern<cir::GetBitfieldOp> {
public:
  using mlir::OpConversionPattern<cir::GetBitfieldOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::GetBitfieldOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMIsConstantOpLowering
    : public mlir::OpConversionPattern<cir::IsConstantOp> {
public:
  using mlir::OpConversionPattern<cir::IsConstantOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::IsConstantOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMCmpThreeWayOpLowering
    : public mlir::OpConversionPattern<cir::CmpThreeWayOp> {
public:
  using mlir::OpConversionPattern<cir::CmpThreeWayOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::CmpThreeWayOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;

private:
  static std::string getLLVMIntrinsicName(bool signedCmp, unsigned operandWidth,
                                          unsigned resultWidth);
};

class CIRToLLVMReturnAddrOpLowering
    : public mlir::OpConversionPattern<cir::ReturnAddrOp> {
public:
  using mlir::OpConversionPattern<cir::ReturnAddrOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ReturnAddrOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMFrameAddrOpLowering
    : public mlir::OpConversionPattern<cir::FrameAddrOp> {
public:
  using mlir::OpConversionPattern<cir::FrameAddrOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::FrameAddrOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMClearCacheOpLowering
    : public mlir::OpConversionPattern<cir::ClearCacheOp> {
public:
  using mlir::OpConversionPattern<cir::ClearCacheOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ClearCacheOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMEhTypeIdOpLowering
    : public mlir::OpConversionPattern<cir::EhTypeIdOp> {
public:
  using mlir::OpConversionPattern<cir::EhTypeIdOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::EhTypeIdOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMCatchParamOpLowering
    : public mlir::OpConversionPattern<cir::CatchParamOp> {
public:
  using mlir::OpConversionPattern<cir::CatchParamOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::CatchParamOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMResumeOpLowering
    : public mlir::OpConversionPattern<cir::ResumeOp> {
public:
  using mlir::OpConversionPattern<cir::ResumeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ResumeOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMAllocExceptionOpLowering
    : public mlir::OpConversionPattern<cir::AllocExceptionOp> {
public:
  using mlir::OpConversionPattern<cir::AllocExceptionOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::AllocExceptionOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMFreeExceptionOpLowering
    : public mlir::OpConversionPattern<cir::FreeExceptionOp> {
public:
  using mlir::OpConversionPattern<cir::FreeExceptionOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::FreeExceptionOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMThrowOpLowering
    : public mlir::OpConversionPattern<cir::ThrowOp> {
public:
  using mlir::OpConversionPattern<cir::ThrowOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ThrowOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMIsFPClassOpLowering
    : public mlir::OpConversionPattern<cir::IsFPClassOp> {
public:
  using mlir::OpConversionPattern<cir::IsFPClassOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::IsFPClassOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMPtrMaskOpLowering
    : public mlir::OpConversionPattern<cir::PtrMaskOp> {
public:
  using mlir::OpConversionPattern<cir::PtrMaskOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::PtrMaskOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMAbsOpLowering : public mlir::OpConversionPattern<cir::AbsOp> {
public:
  using mlir::OpConversionPattern<cir::AbsOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::AbsOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMSignBitOpLowering
    : public mlir::OpConversionPattern<cir::SignBitOp> {
public:
  using OpConversionPattern<cir::SignBitOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::SignBitOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};
mlir::ArrayAttr lowerCIRTBAAAttr(mlir::Attribute tbaa,
                                 mlir::ConversionPatternRewriter &rewriter,
                                 cir::LowerModule *lowerMod);

#define GET_BUILTIN_LOWERING_CLASSES_DECLARE
#include "clang/CIR/Dialect/IR/CIRBuiltinsLowering.inc"
#undef GET_BUILTIN_LOWERING_CLASSES_DECLARE

} // namespace direct
} // namespace cir
