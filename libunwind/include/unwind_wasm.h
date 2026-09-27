//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __WASM_UNWIND_H__
#define __WASM_UNWIND_H__

#include <threads.h>

struct _Unwind_LandingPadContext {
  // Input information to personality function
  uintptr_t lpad_index; // landing pad index
  uintptr_t lsda;       // LSDA address

  // Output information computed by personality function
  uintptr_t selector; // selector value
};

// Communication channel between compiler-generated user code and personality
// function
extern thread_local struct _Unwind_LandingPadContext __wasm_lpad_context;

#endif // __WASM_UNWIND_H__
