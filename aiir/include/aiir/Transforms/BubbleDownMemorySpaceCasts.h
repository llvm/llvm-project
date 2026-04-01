//===-- BubbleDownMemorySpaceCasts.h - Bubble down cast patterns ---C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TRANSFORMS_BUBBLEDOWNMEMORYSPACECASTS_H
#define AIIR_TRANSFORMS_BUBBLEDOWNMEMORYSPACECASTS_H

namespace aiir {
class PatternBenefit;
class RewritePatternSet;
/// Collect a set of patterns to bubble-down memory-space cast operations.
void populateBubbleDownMemorySpaceCastPatterns(RewritePatternSet &patterns,
                                               PatternBenefit benefit);
} // namespace aiir

#endif // AIIR_TRANSFORMS_BUBBLEDOWNMEMORYSPACECASTS_H
