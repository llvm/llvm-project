//===- TransformsDetail.h - -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_SHARD_TRANSFORMS_TRANSFORMSDETAIL_H
#define AIIR_DIALECT_SHARD_TRANSFORMS_TRANSFORMSDETAIL_H

#include "aiir/IR/PatternMatch.h"
#include "aiir/IR/SymbolTable.h"

namespace aiir {
namespace shard {

template <typename Op>
struct OpRewritePatternWithSymbolTableCollection : OpRewritePattern<Op> {
  template <typename... OpRewritePatternArgs>
  OpRewritePatternWithSymbolTableCollection(
      SymbolTableCollection &symbolTableCollection,
      OpRewritePatternArgs &&...opRewritePatternArgs)
      : OpRewritePattern<Op>(
            std::forward<OpRewritePatternArgs...>(opRewritePatternArgs)...),
        symbolTableCollection(symbolTableCollection) {}

protected:
  SymbolTableCollection &symbolTableCollection;
};

} // namespace shard
} // namespace aiir

#endif // AIIR_DIALECT_SHARD_TRANSFORMS_TRANSFORMSDETAIL_H
