//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emit OpenACC Loop Stmt node as CIR code.
//
//===----------------------------------------------------------------------===//

#include "CIRGenBuilder.h"
#include "CIRGenFunction.h"
#include "CIRGenOpenACCClause.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "clang/AST/OpenACCClause.h"
#include "clang/AST/StmtOpenACC.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace cir;
using namespace mlir::acc;

mlir::LogicalResult
CIRGenFunction::emitOpenACCLoopConstruct(const OpenACCLoopConstruct &s) {
  cgm.errorNYI(s.getSourceRange(), "OpenACC Loop Construct");
  return mlir::failure();
}
