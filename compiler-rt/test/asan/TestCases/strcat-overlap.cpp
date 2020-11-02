// RUN: %clangxx_asan -O0 -fno-builtin %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: echo "interceptor_via_fun:bad_function" > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"' %run %t
// RUN: echo "interceptor_name:strcat" > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"' %run %t
//
// RUN: %clangxx_asan -O1 -fno-builtin %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: echo "interceptor_via_fun:bad_function" > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"' %run %t
// RUN: echo "interceptor_name:strcat" > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"' %run %t
//
// RUN: %clangxx_asan -O2 -fno-builtin %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: echo "interceptor_via_fun:bad_function" > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"' %run %t
// RUN: echo "interceptor_name:strcat" > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"' %run %t
//
// RUN: %clangxx_asan -O3 -fno-builtin %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: echo "interceptor_via_fun:bad_function" > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"' %run %t
// RUN: echo "interceptor_name:strcat" > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"' %run %t

// This test when run with suppressions invokes undefined
// behavior which can cause all sorts of bad things to happen
// depending on how strcat() is implemented. For now only run
// on platforms where we know the test passes.
// REQUIRES: x86_64h-darwin || x86_64-darwin || i386-darwin || x86_64-linux || i386-linux
// UNSUPPORTED: windows-msvc
// UNSUPPORTED: android

#include <string.h>


// Don't inline function otherwise stacktrace changes.
__attribute__((noinline)) void bad_function() {
  char buffer[] = "hello\0XXX";
  // CHECK: strcat-param-overlap: memory ranges
  // CHECK: [{{0x.*,[ ]*0x.*}}) and [{{0x.*,[ ]*0x.*}}) overlap
  // CHECK: {{#0 0x.* in .*strcat}}
  // CHECK: {{#1 0x.* in bad_function.*strcat-overlap.cpp:}}[[@LINE+2]]
  // CHECK: {{#2 0x.* in main .*strcat-overlap.cpp:}}[[@LINE+5]]
  strcat(buffer, buffer + 1); // BOOM
}

int main(int argc, char **argv) {
  bad_function();
  return 0;
}
