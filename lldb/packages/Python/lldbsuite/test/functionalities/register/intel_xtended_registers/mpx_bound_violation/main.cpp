//===-- main.cpp ------------------------------------------------*- C++ -*-===//
////
////                     The LLVM Compiler Infrastructure
////
//// This file is distributed under the University of Illinois Open Source
//// License. See LICENSE.TXT for details.
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

  // This call returns 0 only if the CPU and the kernel support Intel(R) MPX.
  if (prctl(PR_MPX_ENABLE_MANAGEMENT, 0, 0, 0, 0) != 0)
    return -1;

  violate_upper_bound(array, 5);
  violate_lower_bound(array, 5);

  return 0;
}
