//===----- hlsl.h - HLSL definitions --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _HLSL_H_
#define _HLSL_H_

#if defined(__clang__)
// Don't warn about any of the DXC compatibility warnings in the clang-only
// headers since these will never be used with DXC anyways.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Whlsl-dxc-compatability"
#endif

#include "hlsl/hlsl_basic_types.h"
#include "hlsl/hlsl_intrinsics.h"

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#endif //_HLSL_H_
