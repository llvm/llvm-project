//===-- Linux implementation of the mtx_init function ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/threads/mtx_init.h"
#include "src/__support/CPP/new.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/mutex.h"

#include <threads.h> // For mtx_t definition.

namespace LIBC_NAMESPACE_DECL {

static_assert(sizeof(Mutex) == sizeof(mtx_t) &&
                  alignof(Mutex) == alignof(mtx_t),
              "The public mtx_t type must exactly match the internal mutex "
              "type.");

LLVM_LIBC_FUNCTION(int, mtx_init, (mtx_t * m, int type)) {
  new (m) Mutex(/*is_priority_inherit=*/false,
                /*is_recursive=*/static_cast<bool>(type & mtx_recursive),
                /*is_robust=*/false, /*is_pshared=*/false);
  return thrd_success;
}

} // namespace LIBC_NAMESPACE_DECL
