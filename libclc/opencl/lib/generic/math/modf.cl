//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/math/clc_modf.h>
#include <clc/opencl/math/modf.h>

#define __CLC_FUNCTION modf
#define __CLC_BODY <clc/math/unary_def_with_ptr.inc>
#include <clc/math/gentype.inc>
