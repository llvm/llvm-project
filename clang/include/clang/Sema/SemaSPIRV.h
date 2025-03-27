//===----- SemaSPIRV.h ----- Semantic Analysis for SPIRV constructs--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares semantic analysis for SPIRV constructs.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMASPIRV_H
#define LLVM_CLANG_SEMA_SEMASPIRV_H

#include "clang/AST/ASTFwd.h"
#include "clang/Sema/SemaBase.h"

namespace clang {
class SemaSPIRV : public SemaBase {
public:
  SemaSPIRV(Sema &S);

  bool CheckSPIRVBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall);
};
} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMASPIRV_H
