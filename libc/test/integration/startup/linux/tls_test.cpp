//===-- Loader test to check if tls size is read correctly ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/sys/mman/mmap.h"
#include "test/IntegrationTest/test.h"

#include <errno.h>
#include <sys/mman.h>

constexpr int threadLocalDataSize = 101;
_Thread_local int a[threadLocalDataSize] = {123};

TEST_MAIN(int argc, char **argv, char **envp) {
  ASSERT_TRUE(a[0] == 123);

  for (int i = 1; i < threadLocalDataSize; ++i)
    a[i] = i;
  for (int i = 1; i < threadLocalDataSize; ++i)
    ASSERT_TRUE(a[i] == i);

  // Call mmap with bad params so that an error value is
  // set in errno. Since errno is implemented using a thread
  // local var, this helps us test setting of errno and
  // reading it back.
  ASSERT_ERRNO_SUCCESS();
  void *addr = LIBC_NAMESPACE::mmap(nullptr, 0, PROT_READ,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  ASSERT_TRUE(addr == MAP_FAILED);
  ASSERT_ERRNO_EQ(EINVAL);

  return 0;
}
