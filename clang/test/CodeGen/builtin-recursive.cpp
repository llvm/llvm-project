// RUN: %clang_cc1 -triple x86_64-unknown-unknown -nostdsysteminc -nobuiltininc -isystem %S/Inputs -emit-llvm-only %s

// This used to cause a read past the end of a global variable.

#include <stdio.h>

void testcase(void) {
  vprintf(0, 0);
}

