// RUN: %clang_cc1 -nostdsysteminc -nobuiltininc -isystem Inputs -emit-llvm-only %s

// This used to cause a read past the end of a global variable.

#include <stdio.h>

void testcase(void) {
  vprintf(0, 0);
}

