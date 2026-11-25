// Tests trace pc guard coverage collection.
//
// REQUIRES: has_sancovcc,stable-runtime,x86_64-linux
//
// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: cd %t.dir
// RUN: %clangxx -DSHARED1 -O0 -fsanitize-coverage=trace-pc-guard -shared %s -o %dynamiclib1 -fPIC %ld_flags_rpath_so1
// RUN: %clangxx -DSTATIC1 -O0 -fsanitize-coverage=trace-pc-guard %s -c -o %t_2.o
// RUN: %clangxx -DMAIN -O0 -fsanitize-coverage=trace-pc-guard %s -o %t.dir/exe %ld_flags_rpath_exe1 %t_2.o
// RUN: %env_tool_opts=coverage=1 %run %t.dir/exe 2>&1 | FileCheck %s
// RUN: rm -rf %t.dir

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
