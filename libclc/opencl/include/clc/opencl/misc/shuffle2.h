//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_MISC_SHUFFLE2_H__
#define __CLC_OPENCL_MISC_SHUFFLE2_H__

#define FUNCTION shuffle2

// Integer-type decls
#define __CLC_BODY <clc/misc/shuffle2_decl.inc>
#include <clc/integer/gentype.inc>

// Floating-point decls
#define __CLC_BODY <clc/misc/shuffle2_decl.inc>
#include <clc/math/gentype.inc>

#undef FUNCTION

#endif // __CLC_OPENCL_MISC_SHUFFLE2_H__
