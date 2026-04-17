//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/clc_convert.h"
#include "clc/math/clc_fract.h"
#include "clc/math/clc_mad.h"
#include "clc/math/clc_rint.h"
#include "clc/math/clc_trigpi_helpers.h"

#define __CLC_BODY "clc_trigpi_helpers.inc"
#include "clc/math/gentype.inc"
