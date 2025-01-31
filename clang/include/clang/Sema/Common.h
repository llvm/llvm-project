//===--- Common.h ----- Semantic Analysis common header file --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file declares common functions used in SPIRV and HLSL semantic
// analysis constructs.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_COMMON_H
#define LLVM_CLANG_SEMA_COMMON_H

#include "clang/Sema/Sema.h"

namespace clang {

bool CheckArgTypeIsCorrect(
    Sema *S, Expr *Arg, QualType ExpectedType,
    llvm::function_ref<bool(clang::QualType PassedType)> Check);

bool CheckAllArgTypesAreCorrect(
    Sema *S, CallExpr *TheCall, QualType ExpectedType,
    llvm::function_ref<bool(clang::QualType PassedType)> Check);

bool CheckAllArgTypesAreCorrect(Sema *SemaPtr, CallExpr *TheCall,
                                unsigned int NumOfElts,
                                unsigned int expectedNumOfElts);

} // namespace clang

#endif
