// RUN: %clangxx_tsan -O0 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

#include "test.h"
#include <setjmp.h>

__attribute__((noinline)) void throws_int() {
  throw 42;
}

__attribute__((noinline)) void callee_throws() {
  try {
    throws_int();
  } catch (int) {  // NOLINT
    fprintf(stderr, "callee_throws caught exception\n");
  }
}

__attribute__((noinline)) void throws_catches_rethrows() {
  try {
    throws_int();
  } catch (int) {  // NOLINT
    fprintf(stderr, "throws_catches_rethrows caught exception\n");
    throw;
  }
}

__attribute__((noinline)) void callee_rethrows() {
  try {
    throws_catches_rethrows();
  } catch (int) {  // NOLINT
    fprintf(stderr, "callee_rethrows caught exception\n");
  }
}

__attribute__((noinline)) void throws_and_catches() {
  try {
    throws_int();
  } catch (int) {  // NOLINT
    fprintf(stderr, "throws_and_catches caught exception\n");
  }
}

__attribute__((noinline)) void nested_try() {
  try {
    try {
      throws_int();
    } catch (double) {  // NOLINT
      fprintf(stderr, "nested_try inner block caught exception\n");
    }
  } catch (int) {  // NOLINT
    fprintf(stderr, "nested_try outer block caught exception\n");
  }
}

__attribute__((noinline)) void nested_try2() {
  try {
    try {
      throws_int();
    } catch (int) {  // NOLINT
      fprintf(stderr, "nested_try inner block caught exception\n");
    }
  } catch (double) {  // NOLINT
    fprintf(stderr, "nested_try outer block caught exception\n");
  }
}

class ClassWithDestructor {
 public:
  ClassWithDestructor() {
    fprintf(stderr, "ClassWithDestructor\n");
  }
  ~ClassWithDestructor() {
    fprintf(stderr, "~ClassWithDestructor\n");
  }
};

__attribute__((noinline)) void local_object_then_throw() {
  ClassWithDestructor obj;
  throws_int();
}

__attribute__((noinline)) void cpp_object_with_destructor() {
  try {
    local_object_then_throw();
  } catch (int) {  // NOLINT
    fprintf(stderr, "cpp_object_with_destructor caught exception\n");
  }
}

__attribute__((noinline)) void recursive_call(long n) {
  if (n > 0) {
    recursive_call(n - 1);
  } else {
    throws_int();
  }
}

__attribute__((noinline)) void multiframe_unwind() {
  try {
    recursive_call(5);
  } catch (int) {  // NOLINT
    fprintf(stderr, "multiframe_unwind caught exception\n");
  }
}

__attribute__((noinline)) void longjmp_unwind() {
  jmp_buf env;
  int i = setjmp(env);
  if (i != 0) {
    fprintf(stderr, "longjmp_unwind jumped\n");
    return;
  }

  try {
    longjmp(env, 42);
  } catch (int) {  // NOLINT
    fprintf(stderr, "longjmp_unwind caught exception\n");
  }
}

__attribute__((noinline)) void recursive_call_longjmp(jmp_buf env, long n) {
  if (n > 0) {
    recursive_call_longjmp(env, n - 1);
  } else {
    longjmp(env, 42);
  }
}

__attribute__((noinline)) void longjmp_unwind_multiple_frames() {
  jmp_buf env;
  int i = setjmp(env);
  if (i != 0) {
    fprintf(stderr, "longjmp_unwind_multiple_frames jumped\n");
    return;
  }

  try {
    recursive_call_longjmp(env, 5);
  } catch (int) {  // NOLINT
    fprintf(stderr, "longjmp_unwind_multiple_frames caught exception\n");
  }
}

#define CHECK_SHADOW_STACK(val)                                                \
  fprintf(stderr, (val == __tsan_testonly_shadow_stack_current_size()          \
                       ? "OK.\n"                                               \
                       : "Shadow stack leak!\n"));

int main(int argc, const char * argv[]) {
  fprintf(stderr, "Hello, World!\n");
  unsigned long shadow_stack_size = __tsan_testonly_shadow_stack_current_size();

  throws_and_catches();
  CHECK_SHADOW_STACK(shadow_stack_size);

  callee_throws();
  CHECK_SHADOW_STACK(shadow_stack_size);

  callee_rethrows();
  CHECK_SHADOW_STACK(shadow_stack_size);

  nested_try();
  CHECK_SHADOW_STACK(shadow_stack_size);

  nested_try2();
  CHECK_SHADOW_STACK(shadow_stack_size);

  cpp_object_with_destructor();
  CHECK_SHADOW_STACK(shadow_stack_size);

  multiframe_unwind();
  CHECK_SHADOW_STACK(shadow_stack_size);

  longjmp_unwind();
  CHECK_SHADOW_STACK(shadow_stack_size);

  longjmp_unwind_multiple_frames();
  CHECK_SHADOW_STACK(shadow_stack_size);

  return 0;
}

// CHECK: Hello, World!
// CHECK-NOT: Shadow stack leak
