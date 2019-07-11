// RUN: %clang_cl_asan -Od %p/dll_host.cc -Fe%t
// RUN: %clang_cl_asan -LD -Od %s -Fe%t.dll
// RUN: not %run %t %t.dll 2>&1 | FileCheck %s

#include <stdio.h>
#include <string.h>

extern "C" __declspec(dllexport)
int test_function() {
  char str[] = "Hello!";
  if (6 != strlen(str))
    return 1;
  printf("Initial test OK\n");
  fflush(0);
// CHECK: Initial test OK

  str[6] = '!';  // Removes '\0' at the end!
  int len = strlen(str);
// CHECK: AddressSanitizer: stack-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// FIXME: Should be READ of size 1, see issue 155.
// CHECK: READ of size {{[0-9]+}} at [[ADDR]] thread T0
// CHECK-NEXT: {{#0 .*}}strlen
// CHECK-NEXT: {{#1 .* test_function .*}}dll_intercept_strlen.cc:[[@LINE-5]]
//
// CHECK: Address [[ADDR]] is located in stack of thread T0 at offset {{.*}} in frame
// CHECK-NEXT: test_function {{.*}}dll_intercept_strlen.cc:
  return len > 42;
}
