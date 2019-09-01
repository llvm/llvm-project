//===-- main.cpp ------------------------------------------------*- C++ -*-===//
////
//// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
//// See https://llvm.org/LICENSE.txt for license information.
//// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
////
////===----------------------------------------------------------------------===//
//

#include <cstddef>
#include <sys/prctl.h>

static void violate_upper_bound(int *ptr, int size)
{
  int i;
  i = *(ptr + size);
}

static void violate_lower_bound (int *ptr, int size)
{
  int i;
  i = *(ptr - size);
}

int
main(int argc, char const *argv[])
{
  unsigned int rax, rbx, rcx, rdx;
  int array[5];

// PR_MPX_ENABLE_MANAGEMENT won't be defined on linux kernel versions below 3.19
#ifndef PR_MPX_ENABLE_MANAGEMENT
    return -1;
#endif

  // This call returns 0 only if the CPU and the kernel support Intel(R) MPX.
  if (prctl(PR_MPX_ENABLE_MANAGEMENT, 0, 0, 0, 0) != 0)
    return -1;

  violate_upper_bound(array, 5);
  violate_lower_bound(array, 5);

  return 0;
}
