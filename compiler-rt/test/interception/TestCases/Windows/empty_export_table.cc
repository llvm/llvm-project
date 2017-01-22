// RUN: %clang_cl %s %inter_lib -o %t
// RUN: %run %t 2>&1 | FileCheck %s
// CHECK: OK

#include "interception/interception.h"
#include <stdio.h>
#include <windows.h>

// We try to get a pointer to a function from an executable that doesn't export
// any symbol (empty export table).
int main() {
  __sanitizer::uptr FunPtr = __interception::InternalGetProcAddress(
      (void *)GetModuleHandleA(0), "exampleFun");
  if (FunPtr == 0)
    printf("OK");
  return 0;
}
