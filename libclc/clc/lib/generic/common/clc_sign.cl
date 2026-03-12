//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/math/clc_copysign.h>
#include <clc/relational/clc_isnan.h>
#include <clc/relational/clc_select.h>

#define __CLC_BODY <clc_sign.inc>
#include <clc/math/gentype.inc>
