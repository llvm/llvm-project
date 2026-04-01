//===- TestEmulateNarrowType.cpp - Test Narrow Type Emulation  ------*- c++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Affine/IR/AffineOps.h"
#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/Arith/Transforms/NarrowTypeEmulationConverter.h"
#include "aiir/Dialect/Arith/Transforms/Passes.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/MemRef/IR/MemRef.h"
#include "aiir/Dialect/MemRef/Transforms/Transforms.h"
#include "aiir/Dialect/Vector/IR/VectorOps.h"
#include "aiir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/DialectConversion.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"

using namespace aiir;

namespace {

struct TestEmulateNarrowTypePass
    : public PassWrapper<TestEmulateNarrowTypePass,
                         OperationPass<func::FuncOp>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestEmulateNarrowTypePass)

  TestEmulateNarrowTypePass() = default;
  TestEmulateNarrowTypePass(const TestEmulateNarrowTypePass &pass)
      : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                vector::VectorDialect, affine::AffineDialect>();
  }
  StringRef getArgument() const final { return "test-emulate-narrow-int"; }
  StringRef getDescription() const final {
    return "Function pass to test Narrow Integer Emulation";
  }

  void runOnOperation() override {
    if (!llvm::isPowerOf2_32(loadStoreEmulateBitwidth) ||
        loadStoreEmulateBitwidth < 8) {
      signalPassFailure();
      return;
    }

    Operation *op = getOperation();
    AIIRContext *ctx = op->getContext();

    arith::NarrowTypeEmulationConverter typeConverter(loadStoreEmulateBitwidth);

    // Convert scalar type.
    typeConverter.addConversion([this](IntegerType ty) -> std::optional<Type> {
      unsigned width = ty.getWidth();
      if (width >= arithComputeBitwidth)
        return ty;

      return IntegerType::get(ty.getContext(), arithComputeBitwidth);
    });

    // Convert vector type.
    typeConverter.addConversion([this](VectorType ty) -> std::optional<Type> {
      auto intTy = dyn_cast<IntegerType>(ty.getElementType());
      if (!intTy)
        return ty;

      unsigned width = intTy.getWidth();
      if (width >= arithComputeBitwidth)
        return ty;

      return VectorType::get(
          to_vector(ty.getShape()),
          IntegerType::get(ty.getContext(), arithComputeBitwidth));
    });

    // With the type converter enabled, we are effectively unable to write
    // negative tests. This is a workaround specifically for negative tests.
    if (!disableMemrefTypeConversion)
      memref::populateMemRefNarrowTypeEmulationConversions(typeConverter);

    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<func::FuncOp>([&typeConverter](Operation *op) {
      return typeConverter.isLegal(cast<func::FuncOp>(op).getFunctionType());
    });
    auto opLegalCallback = [&typeConverter](Operation *op) {
      return typeConverter.isLegal(op);
    };
    target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(opLegalCallback);
    target.addDynamicallyLegalDialect<
        arith::ArithDialect, vector::VectorDialect, memref::MemRefDialect,
        affine::AffineDialect>(opLegalCallback);

    RewritePatternSet patterns(ctx);

    arith::populateArithNarrowTypeEmulationPatterns(typeConverter, patterns);
    memref::populateMemRefNarrowTypeEmulationPatterns(typeConverter, patterns,
                                                      disableAtomicRMW);
    vector::populateVectorNarrowTypeEmulationPatterns(
        typeConverter, patterns, disableAtomicRMW, assumeAligned);

    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }

  Option<unsigned> loadStoreEmulateBitwidth{
      *this, "memref-load-bitwidth",
      llvm::cl::desc("memref load/store emulation bit width"),
      llvm::cl::init(8)};

  Option<unsigned> arithComputeBitwidth{
      *this, "arith-compute-bitwidth",
      llvm::cl::desc("arith computation bit width"), llvm::cl::init(4)};

  Option<bool> disableMemrefTypeConversion{
      *this, "skip-memref-type-conversion",
      llvm::cl::desc("disable memref type conversion (to test failures)"),
      llvm::cl::init(false)};

  Option<bool> disableAtomicRMW{
      *this, "disable-atomic-rmw",
      llvm::cl::desc("disable atomic read-modify-write and prefer generating "
                     "normal sequence"),
      llvm::cl::init(false)};

  Option<bool> assumeAligned{
      *this, "assume-aligned",
      llvm::cl::desc("assume store offsets are aligned to container element "
                     "boundaries"),
      llvm::cl::init(false)};
};

struct TestMemRefFlattenAndVectorNarrowTypeEmulationPass
    : public PassWrapper<TestMemRefFlattenAndVectorNarrowTypeEmulationPass,
                         OperationPass<func::FuncOp>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestMemRefFlattenAndVectorNarrowTypeEmulationPass)

  TestMemRefFlattenAndVectorNarrowTypeEmulationPass() = default;
  TestMemRefFlattenAndVectorNarrowTypeEmulationPass(
      const TestMemRefFlattenAndVectorNarrowTypeEmulationPass &pass)
      : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                vector::VectorDialect, affine::AffineDialect>();
  }

  StringRef getArgument() const final {
    return "test-memref-flatten-and-vector-narrow-type-emulation";
  }

  StringRef getDescription() const final {
    return "Test MemRef flattening and vector narrow type emulation patterns";
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    AIIRContext *ctx = &getContext();

    // Create a type converter for narrow type emulation (8-bit)
    arith::NarrowTypeEmulationConverter typeConverter(8);

    // Add conversions for memref types with i4 elements
    memref::populateMemRefNarrowTypeEmulationConversions(typeConverter);

    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<func::FuncOp>([&typeConverter](Operation *op) {
      return typeConverter.isLegal(cast<func::FuncOp>(op).getFunctionType());
    });
    auto opLegalCallback = [&typeConverter](Operation *op) {
      return typeConverter.isLegal(op);
    };
    target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(opLegalCallback);
    target.addDynamicallyLegalDialect<
        arith::ArithDialect, vector::VectorDialect, memref::MemRefDialect,
        affine::AffineDialect>(opLegalCallback);

    RewritePatternSet patterns(ctx);

    // This is necessary for the purpose of emulating `memref.alloc` and
    // function boundaries.
    memref::populateMemRefNarrowTypeEmulationPatterns(typeConverter, patterns);

    vector::populateMemRefFlattenAndVectorNarrowTypeEmulationPatterns(
        typeConverter, patterns);

    // Apply partial conversion
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

namespace aiir::test {
void registerTestEmulateNarrowTypePass() {
  PassRegistration<TestEmulateNarrowTypePass>();
  PassRegistration<TestMemRefFlattenAndVectorNarrowTypeEmulationPass>();
}
} // namespace aiir::test
