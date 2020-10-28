//===-- Loader test to check if tls size is read correctly ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/errno.h"
#include "include/sys/mman.h"

#undef NDEBUG
#include "src/assert/assert.h"

#include "src/errno/llvmlibc_errno.h"
#include "src/sys/mman/mmap.h"

constexpr int threadLocalDataSize = 101;
_Thread_local int a[threadLocalDataSize] = {123};

int main(int argc, char **argv, char **envp) {
  assert(a[0] == 123);

  for (int i = 1; i < threadLocalDataSize; ++i)
    a[i] = i;
  for (int i = 1; i < threadLocalDataSize; ++i)
    assert(a[i] == i);

  // Call mmap with bad params so that an error value is
  // set in errno. Since errno is implemented using a thread
  // local var, this helps us test setting of errno and
  // reading it back.
  assert(llvmlibc_errno == 0);
  void *addr = __llvm_libc::mmap(nullptr, 0, PROT_READ,
                                 MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  assert(addr == MAP_FAILED);
  assert(llvmlibc_errno == EINVAL);

  return 0;
}
