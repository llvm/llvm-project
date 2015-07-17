//===-------------------------- unwind_ext.h ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//
//  Extensions to unwind API.
//
//===----------------------------------------------------------------------===//

#ifndef __UNWIND_EXT__
#define __UNWIND_EXT__

#include "unwind.h"

#ifdef __cplusplus
extern "C" {
#endif

// These platform specific functions to get and set the top context are
// implemented elsewhere.

extern struct _Unwind_FunctionContext *
__Unwind_SjLj_GetTopOfFunctionStack();

extern void
__Unwind_SjLj_SetTopOfFunctionStack(struct _Unwind_FunctionContext *fc);

#ifdef __cplusplus
}
#endif

#endif // __UNWIND_EXT__


