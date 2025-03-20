//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/internal/clc.h>
#include <clc/math/clc_copysign.h>
#include <clc/math/clc_trunc.h>
#include <clc/math/math.h>
#include <clc/relational/clc_isinf.h>

#define __CLC_BODY <clc_modf.inc>
#include <clc/math/gentype.inc>
