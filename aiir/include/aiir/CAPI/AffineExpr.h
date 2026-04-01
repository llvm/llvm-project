//===- AffineExpr.h - C API Utils for Affine Expressions --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains declarations of implementation details of the C API for
// AIIR Affine Expression. This file should not be included from C++ code other
// than C API implementation nor from C code.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CAPI_AFFINEEXPR_H
#define AIIR_CAPI_AFFINEEXPR_H

#include "aiir-c/AffineExpr.h"
#include "aiir/CAPI/Wrap.h"
#include "aiir/IR/AffineExpr.h"

DEFINE_C_API_METHODS(AiirAffineExpr, aiir::AffineExpr)

#endif // AIIR_CAPI_AFFINEEXPR_H
