//===- TUSummaryExtractor.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_TUSUMMARY_TUSUMMARYEXTRACTOR_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_TUSUMMARY_TUSUMMARYEXTRACTOR_H

#include "clang/AST/ASTConsumer.h"

namespace clang::ssaf {
class TUSummaryBuilder;

class TUSummaryExtractor : public ASTConsumer {
public:
  explicit TUSummaryExtractor(TUSummaryBuilder &Builder)
      : SummaryBuilder(Builder) {}

protected:
  TUSummaryBuilder &SummaryBuilder;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_TUSUMMARY_TUSUMMARYEXTRACTOR_H
