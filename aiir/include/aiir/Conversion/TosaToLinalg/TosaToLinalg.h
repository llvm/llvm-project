//===- TosaToLinalg.h - TOSA optimization pass declarations ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the passes for the TOSA Linalg Dialect in AIIR.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_TOSATOLINALG_TOSATOLINALG_H
#define AIIR_CONVERSION_TOSATOLINALG_TOSATOLINALG_H

#include "aiir/Dialect/Tosa/Transforms/Passes.h"
#include "aiir/Pass/Pass.h"

namespace aiir {

#define GEN_PASS_DECL_TOSATOLINALG
#define GEN_PASS_DECL_TOSATOLINALGNAMED
#include "aiir/Conversion/Passes.h.inc"

namespace tosa {

std::unique_ptr<Pass> createTosaToLinalg();
std::unique_ptr<Pass> createTosaToLinalgNamed(
    const TosaToLinalgNamedOptions &options = TosaToLinalgNamedOptions());

/// Populates passes to convert from TOSA to Linalg. At the end of
/// the pass, the function will only contain linalg ops or standard ops if the
/// pipeline succeeds.  The option to disable decompositions is available for
/// benchmarking performance improvements from the canonicalizations.
void addTosaToLinalgPasses(
    OpPassManager &pm, const TosaToLinalgOptions &options,
    const TosaToLinalgNamedOptions &tosaToLinalgNamedOptions =
        TosaToLinalgNamedOptions(),
    // Note: Default to 'none' level unless otherwise specified.
    std::optional<tosa::TosaValidationOptions> validationOptions =
        tosa::TosaValidationOptions{false, false},
    std::optional<TosaAttachTargetOptions> attachTargetOptions = std::nullopt);

/// Populates TOSA to linalg pipelines
/// Currently, this includes only the "tosa-to-linalg-pipeline".
void registerTosaToLinalgPipelines();

/// Populates conversion passes from TOSA dialect to Linalg dialect.
void populateTosaToLinalgConversionPatterns(const TypeConverter &converter,
                                            RewritePatternSet *patterns);

/// Populates conversion passes from TOSA dialect to Linalg named operations.
void populateTosaToLinalgNamedConversionPatterns(
    const TypeConverter &converter, RewritePatternSet *patterns,
    const TosaToLinalgNamedOptions &options);

} // namespace tosa
} // namespace aiir

#endif // AIIR_CONVERSION_TOSATOLINALG_TOSATOLINALG_H
