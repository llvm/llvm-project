// RUN: %clangxx_msan -O0 %s -o %t && not %run %t >%t.out 2>&1
// FileCheck --check-prefix=CHECK-RECOVER %s <%t.out
// RUN: %clangxx_msan -O0 %s -o %t && env MSAN_OPTIONS=keep_going=0 not %run %t >%t.out 2>&1
// FileCheck %s <%t.out
// RUN: %clangxx_msan -O0 %s -o %t && env MSAN_OPTIONS=keep_going=1 not %run %t >%t.out 2>&1
// FileCheck --check-prefix=CHECK-RECOVER %s <%t.out

// Test how -fsanitize-recover=memory and MSAN_OPTIONS=keep_going affect reports
// from interceptors.
// -fsanitize-recover=memory provides the default value of keep_going flag, but is
// always overwritten by MSAN_OPTIONS

// RUN: %clangxx_msan -fsanitize-recover=memory -O0 %s -o %t && not %run %t >%t.out 2>&1
// FileCheck --check-prefix=CHECK-RECOVER %s <%t.out
// RUN: %clangxx_msan -fsanitize-recover=memory -O0 %s -o %t && env MSAN_OPTIONS=keep_going=0 not %run %t >%t.out 2>&1
// FileCheck %s <%t.out
// RUN: %clangxx_msan -fsanitize-recover=memory -O0 %s -o %t && env MSAN_OPTIONS=keep_going=1 not %run %t >%t.out 2>&1
// FileCheck --check-prefix=CHECK-RECOVER %s <%t.out

// Test how legacy -mllvm -msan-keep-going and MSAN_OPTIONS=keep_going affect
// reports from interceptors.

// RUN: %clangxx_msan -mllvm -msan-keep-going=1 -O0 %s -o %t && not %run %t >%t.out 2>&1
// FileCheck --check-prefix=CHECK-RECOVER %s <%t.out

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  char *volatile x = (char*)malloc(5 * sizeof(char));
  x[4] = 0;
  if (strlen(x) < 3)
    exit(0);
  fprintf(stderr, "Done\n");
  // CHECK-NOT: Done
  // CHECK-RECOVER: Done
  return 0;
}
