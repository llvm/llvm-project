//===- X86TransformOps.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/X86/TransformOps/X86TransformOps.h"
#include "aiir/Conversion/LLVMCommon/TypeConverter.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/Dialect/Transform/IR/TransformDialect.h"
#include "aiir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "aiir/Dialect/Vector/IR/VectorOps.h"
#include "aiir/Dialect/X86/Transforms.h"
#include "aiir/Dialect/X86/X86Dialect.h"

#include "aiir/IR/OpImplementation.h"
#include "aiir/IR/RegionKindInterface.h"

using namespace aiir;
using namespace aiir::x86;
using namespace aiir::transform;

void aiir::transform::ApplyVectorContractToFMAPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  x86::populateVectorContractToFMAPatterns(patterns);
}

void aiir::transform::ApplyVectorContractToPackedTypeDotProductPatternsOp::
    populatePatterns(RewritePatternSet &patterns) {
  x86::populateVectorContractToPackedTypeDotProductPatterns(patterns);
}

void aiir::transform::ApplyVectorContractBF16ToFMAPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  x86::populateVectorContractBF16ToFMAPatterns(patterns);
}

void aiir::transform::ApplySinkVectorProducerOpsPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  x86::populateSinkVectorProducerOpsPatterns(patterns);
}

void aiir::transform::ApplyShuffleVectorFMAOpsPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  x86::populateShuffleVectorFMAOpsPatterns(patterns);
}

void aiir::transform::ApplyVectorContractToAMXDotProductPatternsOp::
    populatePatterns(RewritePatternSet &patterns) {
  x86::populateVectorContractToAMXDotProductPatterns(patterns);
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class X86TransformDialectExtension
    : public transform::TransformDialectExtension<
          X86TransformDialectExtension> {
public:
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(X86TransformDialectExtension)

  X86TransformDialectExtension() {
    declareGeneratedDialect<x86::X86Dialect>();
    declareGeneratedDialect<LLVM::LLVMDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "aiir/Dialect/X86/TransformOps/X86TransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "aiir/Dialect/X86/TransformOps/X86TransformOps.cpp.inc"

void aiir::x86::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<X86TransformDialectExtension>();
}
