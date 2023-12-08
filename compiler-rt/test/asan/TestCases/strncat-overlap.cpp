// RUN: %clangxx_asan -O0 -fno-builtin %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: echo "interceptor_via_fun:bad_function" > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"' %run %t
// RUN: echo "interceptor_name:strncat" > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"' %run %t
//
// RUN: %clangxx_asan -O1 -fno-builtin %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: echo "interceptor_via_fun:bad_function" > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"' %run %t
// RUN: echo "interceptor_name:strncat" > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"' %run %t
//
// RUN: %clangxx_asan -O2 -fno-builtin %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: echo "interceptor_via_fun:bad_function" > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"' %run %t
// RUN: echo "interceptor_name:strncat" > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"' %run %t
//
// RUN: %clangxx_asan -O3 -fno-builtin %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: echo "interceptor_via_fun:bad_function" > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"' %run %t
// RUN: echo "interceptor_name:strncat" > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"' %run %t

// UNSUPPORTED: android

#include <string.h>


// Don't inline function otherwise stacktrace changes.
__attribute__((noinline)) void bad_function() {
  char buffer[] = "hello\0XXX";
  // CHECK: strncat-param-overlap: memory ranges
  // CHECK: [{{0x.*,[ ]*0x.*}}) and [{{0x.*,[ ]*0x.*}}) overlap
  // CHECK: {{#0 0x.* in .*strncat}}
  // CHECK: {{#1 0x.* in bad_function.*strncat-overlap.cpp:}}[[@LINE+2]]
  // CHECK: {{#2 0x.* in main .*strncat-overlap.cpp:}}[[@LINE+5]]
  strncat(buffer, buffer + 1, 3); // BOOM
}

int main(int argc, char **argv) {
  bad_function();
  return 0;
}
