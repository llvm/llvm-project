//===- Transformation.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract base class for source transformations. A Transformation is an
// ASTConsumer that consumes a previously computed WPASuite.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_TRANSFORMATION_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_TRANSFORMATION_H

#include "clang/AST/ASTConsumer.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/WPASuite.h"
#include "clang/ScalableStaticAnalysis/SourceTransformation/SourceEditEmitter.h"
#include "clang/ScalableStaticAnalysis/SourceTransformation/TransformationReportEmitter.h"

namespace clang::ssaf {

class Transformation : public clang::ASTConsumer {
public:
  Transformation(const WPASuite &Suite, SourceEditEmitter &Edits,
                 TransformationReportEmitter &Report)
      : Suite(Suite), Edits(Edits), Report(Report) {}

protected:
  const WPASuite &Suite;
  SourceEditEmitter &Edits;
  TransformationReportEmitter &Report;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_TRANSFORMATION_H
