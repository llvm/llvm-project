//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/clc_convert.h"
#include "clc/integer/clc_clz.h"
#include "clc/math/clc_get_twobypi_bits.h"
#include "clc/math/tables.h"

#define __CLC_DOUBLE_ONLY
#define __CLC_FUNCTION __clc_get_twobypi_bits
#define __CLC_BODY <clc_get_twobypi_bits.inc>
#include "clc/math/gentype.inc"
