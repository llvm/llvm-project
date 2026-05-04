//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if !defined(__wasm__)

#include "assembly.h"

NO_EXEC_STACK_DIRECTIVE

#endif // !defined(__wasm__)

#ifdef __WASM_EXCEPTIONS__

// Define the `__cpp_exception` symbol which `__builtin_wasm_throw` will
// reference. This is defined here in `libunwind` as the single canonical
// definition for this API and it's required for users to ensure that there's
// only one copy of `libunwind` within a wasm module to ensure this is only
// defined once and exactly once.
.globl __cpp_exception
#if defined(__wasm32__)
.tagtype __cpp_exception i32
#elif defined(__wasm64__)
.tagtype __cpp_exception i64
#else
#error "Unsupported Wasm architecture"
#endif
__cpp_exception:

#endif // defined(__WASM_EXCEPTIONS__)
