//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>
#include <clc/math/clc_remquo.h>

#define __CLC_BODY <remquo.inc>
#define __CLC_ADDRESS_SPACE global
#include <clc/math/gentype.inc>
#undef __CLC_ADDRESS_SPACE

#define __CLC_BODY <remquo.inc>
#define __CLC_ADDRESS_SPACE local
#include <clc/math/gentype.inc>
#undef __CLC_ADDRESS_SPACE

#define __CLC_BODY <remquo.inc>
#define __CLC_ADDRESS_SPACE private
#include <clc/math/gentype.inc>
#undef __CLC_ADDRESS_SPACE
