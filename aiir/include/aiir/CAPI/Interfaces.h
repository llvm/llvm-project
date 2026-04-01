//===- Interfaces.h - C API Utils for AIIR interfaces -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains declarations of implementation details of the C API for
// AIIR interface classes. This file should not be included from C++ code other
// than C API implementation nor from C code.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CAPI_INTERFACES_H
#define AIIR_CAPI_INTERFACES_H

#include "aiir-c/Interfaces.h"
#include "aiir/CAPI/Wrap.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"

DEFINE_C_API_PTR_METHODS(
    AiirMemoryEffectInstancesList,
    llvm::SmallVectorImpl<aiir::MemoryEffects::EffectInstance>)

#endif // AIIR_CAPI_INTERFACES_H
