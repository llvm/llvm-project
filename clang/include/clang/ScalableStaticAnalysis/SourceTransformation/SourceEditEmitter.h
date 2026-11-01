//===- SourceEditEmitter.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract accumulator for source edits.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_SOURCEEDITEMITTER_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_SOURCEEDITEMITTER_H

#include "clang/Tooling/Core/Replacement.h"

namespace clang::ssaf {

class SourceEditEmitter {
public:
  virtual ~SourceEditEmitter() = default;

  virtual void addReplacement(clang::tooling::Replacement R) = 0;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_SOURCEEDITEMITTER_H
