//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __CLC_FUNCTION shuffle2

// Integer-type decls
#define __CLC_BODY <clc/misc/shuffle2_decl.inc>
#include <clc/integer/gentype.inc>

// Floating-point decls
#define __CLC_BODY <clc/misc/shuffle2_decl.inc>
#include <clc/math/gentype.inc>

#undef __CLC_FUNCTION
