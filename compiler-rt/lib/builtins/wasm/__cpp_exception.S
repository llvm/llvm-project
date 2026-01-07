//===-- __cpp_exception.S - Implement __cpp_exception ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __cpp_exception which LLVM uses to implement exception
// handling when Wasm EH is enabled.
//
//===----------------------------------------------------------------------===//

#ifdef __wasm_exception_handling__

#ifdef __wasm64__
#define PTR i64
#else
#define PTR i32
#endif

.globl __cpp_exception
.tagtype __cpp_exception PTR
__cpp_exception:

#endif // !__wasm_exception_handling__
