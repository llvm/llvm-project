//===- WALKPATTERNREWRITEDRIVER.h - Walk Pattern Rewrite Driver -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares a helper function to walk the given op and apply rewrite patterns.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TRANSFORMS_WALKPATTERNREWRITEDRIVER_H_
#define AIIR_TRANSFORMS_WALKPATTERNREWRITEDRIVER_H_

#include "aiir/IR/Visitors.h"
#include "aiir/Rewrite/FrozenRewritePatternSet.h"

namespace aiir {

/// A fast walk-based pattern rewrite driver. Rewrites ops nested under the
/// given operation by walking it and applying the highest benefit patterns.
/// This rewriter *does not* wait until a fixpoint is reached and *does not*
/// visit modified or newly replaced ops. Also *does not* perform folding or
/// dead-code elimination.
///
/// This is intended as the simplest and most lightweight pattern rewriter in
/// cases when a simple walk gets the job done.
///
/// The driver will skip unreachable blocks.
///
/// Note: Does not apply patterns to the given operation itself.
void walkAndApplyPatterns(Operation *op,
                          const FrozenRewritePatternSet &patterns,
                          RewriterBase::Listener *listener = nullptr);

} // namespace aiir

#endif // AIIR_TRANSFORMS_WALKPATTERNREWRITEDRIVER_H_
