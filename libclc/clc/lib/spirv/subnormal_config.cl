//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// FIXME: These overrides are a workaround for canonicalize not working.
//
//===----------------------------------------------------------------------===//

#include "clc/math/clc_subnormal_config.h"

_CLC_DEF bool __clc_denormals_are_zero_fp16() { return false; }

_CLC_DEF bool __clc_denormals_are_zero_fp32() { return true; }

_CLC_DEF bool __clc_denormals_are_zero_fp64() { return false; }
