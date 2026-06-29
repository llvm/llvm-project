//===- TransformationReportEmitter.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract accumulator for the transformation report.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_TRANSFORMATIONREPORTEMITTER_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_TRANSFORMATIONREPORTEMITTER_H

#include "clang/Basic/Sarif.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/StringRef.h"

namespace clang::ssaf {

class TransformationReportEmitter {
public:
  virtual ~TransformationReportEmitter() = default;

  virtual void addResult(llvm::StringRef RuleId, clang::SarifResultLevel Level,
                         clang::CharSourceRange Range,
                         llvm::StringRef Message) = 0;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_TRANSFORMATIONREPORTEMITTER_H
