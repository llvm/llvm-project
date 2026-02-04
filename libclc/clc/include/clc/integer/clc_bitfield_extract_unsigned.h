//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_INTEGER_CLC_BITFIELD_EXTRACT_SIGNED_H__
#define __CLC_INTEGER_CLC_BITFIELD_EXTRACT_SIGNED_H__

#include <clc/internal/clc.h>

#define __CLC_FUNCTION __clc_bitfield_extract_unsigned
#define __CLC_RETTYPE __CLC_U_GENTYPE

#define __CLC_BODY <clc/integer/clc_bitfield_extract_decl.inc>
#include <clc/integer/gentype.inc>

#undef __CLC_RETTYPE
#undef __CLC_FUNCTION

#endif // __CLC_INTEGER_CLC_BITFIELD_EXTRACT_SIGNED_H__
