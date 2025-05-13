//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>
#include <clc/math/clc_fmin.h>

#define FUNCTION fmin
#define __CLC_BODY <clc/shared/binary_def_with_scalar_second_arg.inc>
#include <clc/math/gentype.inc>
