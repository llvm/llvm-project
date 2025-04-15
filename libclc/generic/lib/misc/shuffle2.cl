//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>
#include <clc/misc/clc_shuffle2.h>

#define FUNCTION shuffle2

#define __CLC_BODY <clc/misc/shuffle2_def.inc>
#include <clc/integer/gentype.inc>

#define __CLC_BODY <clc/misc/shuffle2_def.inc>
#include <clc/math/gentype.inc>
