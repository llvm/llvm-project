//===- Rewrite.h - C API Utils for Core AIIR classes ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains declarations of implementation details of the C API for
// rewrite patterns. This file should not be included from C++ code other than
// C API implementation nor from C code.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CAPI_REWRITE_H
#define AIIR_CAPI_REWRITE_H

#include "aiir-c/Rewrite.h"
#include "aiir/CAPI/Wrap.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/Rewrite/FrozenRewritePatternSet.h"
#include "aiir/Transforms/DialectConversion.h"

DEFINE_C_API_PTR_METHODS(AiirRewriterBase, aiir::RewriterBase)
DEFINE_C_API_PTR_METHODS(AiirRewritePattern, const aiir::RewritePattern)
DEFINE_C_API_PTR_METHODS(AiirRewritePatternSet, aiir::RewritePatternSet)
DEFINE_C_API_PTR_METHODS(AiirFrozenRewritePatternSet,
                         aiir::FrozenRewritePatternSet)
DEFINE_C_API_PTR_METHODS(AiirPatternRewriter, aiir::PatternRewriter)
DEFINE_C_API_PTR_METHODS(AiirConversionTarget, aiir::ConversionTarget)
DEFINE_C_API_PTR_METHODS(AiirConversionPattern, const aiir::ConversionPattern)
DEFINE_C_API_PTR_METHODS(AiirTypeConverter, aiir::TypeConverter)
DEFINE_C_API_PTR_METHODS(AiirConversionPatternRewriter,
                         aiir::ConversionPatternRewriter)
DEFINE_C_API_PTR_METHODS(AiirConversionConfig, aiir::ConversionConfig)

#if AIIR_ENABLE_PDL_IN_PATTERNMATCH
DEFINE_C_API_PTR_METHODS(AiirPDLPatternModule, aiir::PDLPatternModule)
DEFINE_C_API_PTR_METHODS(AiirPDLResultList, aiir::PDLResultList)
DEFINE_C_API_PTR_METHODS(AiirPDLValue, const aiir::PDLValue)
#endif // AIIR_ENABLE_PDL_IN_PATTERNMATCH

#endif // AIIR_CAPI_REWRITE_H
