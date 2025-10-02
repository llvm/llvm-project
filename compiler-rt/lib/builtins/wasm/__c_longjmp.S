//===-- __c_longjmp.S - Implement __c_longjmp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __c_longjmp which LLVM uses to implenmet setjmp/longjmp
// when Wasm EH is enabled.
//
//===----------------------------------------------------------------------===//

#ifdef __wasm_exception_handling__

#ifdef __wasm64__
#define PTR i64
#else
#define PTR i32
#endif

.globl __c_longjmp
.tagtype __c_longjmp PTR
__c_longjmp:

#endif // !__wasm_exception_handling__
