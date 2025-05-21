//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __CLC_FUNCTION frexp
#define __CLC_BODY <clc/math/unary_decl_with_int_ptr.inc>
#include <clc/math/gentype.inc>

#undef __CLC_FUNCTION
