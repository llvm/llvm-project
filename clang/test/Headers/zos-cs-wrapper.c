// REQUIRES: systemz-registered-target
// RUN: %clang -E --target=s390x-ibm-zos %s -mzos-sys-include=%S/Inputs/zos/usr/include | FileCheck %s --check-prefix=NOCS1
// RUN: %clang -E --target=s390x-ibm-zos -D__CS1 %s -mzos-sys-include=%S/Inputs/zos/usr/include | FileCheck %s --check-prefix=CS1

#define _EXT
#include <stdlib.h>

int func(unsigned int *a, unsigned int *b, unsigned int c) {
  return __cs(a, b, c);
}

// NOCS1: return __cs(a, b, c);
// CS1: return __cs1(a, b, &(c));
