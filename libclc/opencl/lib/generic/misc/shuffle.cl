//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/misc/clc_shuffle.h>
#include <clc/opencl/misc/shuffle.h>

#define __CLC_FUNCTION shuffle

#define __CLC_BODY <clc/misc/shuffle_def.inc>
#include <clc/integer/gentype.inc>

#define __CLC_BODY <clc/misc/shuffle_def.inc>
#include <clc/math/gentype.inc>
