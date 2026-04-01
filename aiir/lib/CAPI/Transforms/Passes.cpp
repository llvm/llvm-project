//===- CAPIPasses.cpp - C API for Transformations Passes ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Transforms/Passes.h"
#include "aiir/CAPI/Pass.h"
#include "aiir/Pass/Pass.h"

// Must include the declarations as they carry important visibility attributes.
#include "aiir/Transforms/Transforms.capi.h.inc"

using namespace aiir;

#ifdef __cplusplus
extern "C" {
#endif

#include "aiir/Transforms/Transforms.capi.cpp.inc"

#ifdef __cplusplus
}
#endif
