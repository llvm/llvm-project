// RUN: %clang_hwasan -g %s -o %t && not %run %t 2>&1 | FileCheck %s
// Only implemented for interceptor ABI on AArch64.
// REQUIRES: aarch64-target-arch

#include <setjmp.h>
#include <stdio.h>

/* Testing longjmp/setjmp should test that accesses to scopes jmp'd over are
   caught.  */
int __attribute__((noinline))
uses_longjmp(int **other_array, int num, jmp_buf env) {
  int internal_array[100] = {0};
  *other_array = &internal_array[0];
  if (num % 2)
    longjmp(env, num);
  else
    return num % 8;
}

int __attribute__((noinline)) uses_setjmp(int num) {
  int big_array[100];
  int *other_array = NULL;
  sigjmp_buf cur_env;
  int temp = 0;
  if ((temp = sigsetjmp(cur_env, 1)) != 0) {
    // We're testing that our longjmp interceptor untagged the previous stack.
    // Hence the tag in memory should be zero.
    if (other_array != NULL)
      return other_array[0];
    // CHECK: READ of size 4 at{{.*}}tags: {{..}}/00
    return 100;
  } else
    return uses_longjmp(&other_array, num, cur_env);
}

int __attribute__((noinline)) main() {
  uses_setjmp(1);
  return 0;
}
