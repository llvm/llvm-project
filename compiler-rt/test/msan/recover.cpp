// RUN: %clangxx_msan -O0 %s -o %t && not %run %t >%t.out 2>&1
// FileCheck %s <%t.out
// RUN: %clangxx_msan -O0 %s -o %t && env MSAN_OPTIONS=keep_going=0 not %run %t >%t.out 2>&1
// FileCheck %s <%t.out
// RUN: %clangxx_msan -O0 %s -o %t && env MSAN_OPTIONS=keep_going=1 not %run %t >%t.out 2>&1
// FileCheck %s <%t.out

// Test behavior of -fsanitize-recover=memory and MSAN_OPTIONS=keep_going.
// -fsanitize-recover=memory provides the default value of keep_going flag; value
// of 1 can be overwritten by MSAN_OPTIONS, value of 0 can not.

// RUN: %clangxx_msan -fsanitize-recover=memory -O0 %s -o %t && not %run %t >%t.out 2>&1
// FileCheck --check-prefix=CHECK-RECOVER %s <%t.out
// RUN: %clangxx_msan -fsanitize-recover=memory -O0 %s -o %t && env MSAN_OPTIONS=keep_going=0 not %run %t >%t.out 2>&1
// FileCheck %s <%t.out
// RUN: %clangxx_msan -fsanitize-recover=memory -O0 %s -o %t && env MSAN_OPTIONS=keep_going=1 not %run %t >%t.out 2>&1
// FileCheck --check-prefix=CHECK-RECOVER %s <%t.out
// RUN: %clangxx_msan -fsanitize-recover=memory -O0 %s -o %t && env MSAN_OPTIONS=halt_on_error=1 not %run %t >%t.out 2>&1
// FileCheck %s <%t.out
// RUN: %clangxx_msan -fsanitize-recover=memory -O0 %s -o %t && env MSAN_OPTIONS=halt_on_error=0 not %run %t >%t.out 2>&1
// FileCheck --check-prefix=CHECK-RECOVER %s <%t.out

// Basic test of legacy -mllvm -msan-keep-going and MSAN_OPTIONS=keep_going.

// RUN: %clangxx_msan -mllvm -msan-keep-going=1 -O0 %s -o %t && not %run %t >%t.out 2>&1
// FileCheck --check-prefix=CHECK-RECOVER %s <%t.out
// RUN: %clangxx_msan -mllvm -msan-keep-going=1 -O0 %s -o %t && env MSAN_OPTIONS=keep_going=0 not %run %t >%t.out 2>&1
// FileCheck %s <%t.out

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  char *volatile x = (char*)malloc(5 * sizeof(char));
  if (x[0])
    exit(0);
  fprintf(stderr, "Done\n");
  // CHECK-NOT: Done
  // CHECK-RECOVER: Done
  return 0;
}
