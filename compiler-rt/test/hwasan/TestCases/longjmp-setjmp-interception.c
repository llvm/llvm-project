// RUN: %clang_hwasan -g %s -o %t
// RUN: not %run %t 0 2>&1 | FileCheck %s
// RUN: not %run %t -33 2>&1 | FileCheck %s
// REQUIRES: pointer-tagging

#include <assert.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>

/* Testing longjmp/setjmp should test that accesses to scopes jmp'd over are
   caught.  */
int __attribute__((noinline))
uses_longjmp(int **other_array, int num, jmp_buf env) {
  int internal_array[100] = {0};
  *other_array = &internal_array[0];
  longjmp(env, num);
}

int __attribute__((noinline)) uses_setjmp(int num) {
  int big_array[100];
  int *other_array = NULL;
  sigjmp_buf cur_env;
  int temp = 0;
  if ((temp = sigsetjmp(cur_env, 1)) != 0) {
    assert((num == 0 && temp == 1) || (num != 0 && temp == num));
    // We're testing that our longjmp interceptor untagged the previous stack.
    // Hence the tag in memory should be zero.
    if (other_array != NULL)
      return other_array[0];
    // CHECK: READ of size 4 at{{.*}}tags: {{..}}/00
    return 100;
  } else
    return uses_longjmp(&other_array, num, cur_env);
}

int __attribute__((noinline)) main(int argc, char *argv[]) {
  assert(argc == 2);
  int longjmp_retval = atoi(argv[1]);
  uses_setjmp(longjmp_retval);
  return 0;
}
