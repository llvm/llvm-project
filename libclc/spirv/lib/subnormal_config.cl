//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>
#include <clc/math/clc_subnormal_config.h>

_CLC_DEF bool __clc_fp16_subnormals_supported() { return false; }

_CLC_DEF bool __clc_fp32_subnormals_supported() { return false; }

_CLC_DEF bool __clc_fp64_subnormals_supported() { return false; }
