//===- PreprocessorExcludedConditionalDirectiveSkipMapping.h - --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_PREPROCESSOR_EXCLUDED_COND_DIRECTIVE_SKIP_MAPPING_H
#define LLVM_CLANG_LEX_PREPROCESSOR_EXCLUDED_COND_DIRECTIVE_SKIP_MAPPING_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MemoryBuffer.h"

namespace clang {

/// A mapping from an offset into a buffer to the number of bytes that can be
/// skipped by the preprocessor when skipping over excluded conditional
/// directive ranges.
using PreprocessorSkippedRangeMapping = llvm::DenseMap<unsigned, unsigned>;

/// The datastructure that holds the mapping between the active memory buffers
/// and the individual skip mappings.
using ExcludedPreprocessorDirectiveSkipMapping =
    llvm::DenseMap<const char *, const PreprocessorSkippedRangeMapping *>;

} // end namespace clang

#endif // LLVM_CLANG_LEX_PREPROCESSOR_EXCLUDED_COND_DIRECTIVE_SKIP_MAPPING_H
