// Tests trace pc guard coverage collection.
//
// REQUIRES: has_sancovcc,stable-runtime,x86_64-linux
// XFAIL: tsan
//
// RUN: DIR=%t_workdir
// RUN: CLANG_ARGS="-O0 -fsanitize-coverage=trace-pc-guard"
// RUN: rm -rf $DIR
// RUN: mkdir -p $DIR
// RUN: cd $DIR
// RUN: %clangxx -DSHARED1 $CLANG_ARGS -shared %s -o %t_1.so -fPIC
// RUN: %clangxx -DSTATIC1 $CLANG_ARGS %s -c -o %t_2.o
// RUN: %clangxx -DMAIN $CLANG_ARGS %s -o %t %t_1.so %t_2.o
// RUN: %env_tool_opts=coverage=1 %t 2>&1 | FileCheck %s
// RUN: rm -rf $DIR

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

extern "C" {
  int bar();
  int baz();
}

#ifdef MAIN

extern "C" void __sanitizer_cov_trace_pc_guard_init(uint32_t *start, uint32_t *stop) {
  fprintf(stderr, "__sanitizer_cov_trace_pc_guard_init\n");
}

extern "C" void __sanitizer_cov_trace_pc_guard(uint32_t *guard) { }


int foo() {
  fprintf(stderr, "foo\n");
  return 1;
}

int main() {
  fprintf(stderr, "main\n");
  foo();
  bar();
  baz();
}

#endif // MAIN

extern "C" {

#ifdef SHARED1
int bar() {
  fprintf(stderr, "bar\n");
  return 1;
}
#endif

#ifdef STATIC1
int baz() {
  fprintf(stderr, "baz\n");
  return 1;
}
#endif

} // extern "C"

// Init is called once per DSO.
// CHECK: __sanitizer_cov_trace_pc_guard_init
// CHECK-NEXT: __sanitizer_cov_trace_pc_guard_init
// CHECK-NEXT: main
// CHECK-NEXT: foo
// CHECK-NEXT: bar
// CHECK-NEXT: baz
