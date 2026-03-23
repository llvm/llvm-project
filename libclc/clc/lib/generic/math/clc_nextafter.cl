//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/clc_convert.h"
#include "clc/float/definitions.h"
#include "clc/math/clc_flush_if_daz.h"
#include "clc/math/clc_nextdown.h"
#include "clc/math/clc_nextup.h"
#include "clc/math/clc_subnormal_config.h"
#include "clc/relational/clc_isunordered.h"

#define __CLC_BODY "clc_nextafter.inc"
#include "clc/math/gentype.inc"
