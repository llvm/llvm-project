// RUN: %clangxx_asan -O0 -fno-builtin %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s %if system-darwin %{ --check-prefixes=CHECK,DARWIN %} %else %{ --check-prefixes=CHECK,NON_DARWIN %}
// RUN: echo "interceptor_via_fun:bad_function" > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"' %run %t
// RUN: echo "interceptor_name:strncpy" > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"' %run %t
//
// RUN: %clangxx_asan -O1 -fno-builtin %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s %if system-darwin %{ --check-prefixes=CHECK,DARWIN %} %else %{ --check-prefixes=CHECK,NON_DARWIN %}
// RUN: echo "interceptor_via_fun:bad_function" > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"' %run %t
// RUN: echo "interceptor_name:strncpy" > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"' %run %t
//
// RUN: %clangxx_asan -O2 -fno-builtin %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s %if system-darwin %{ --check-prefixes=CHECK,DARWIN %} %else %{ --check-prefixes=CHECK,NON_DARWIN %}
// RUN: echo "interceptor_via_fun:bad_function" > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"' %run %t
// RUN: echo "interceptor_name:strncpy" > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"' %run %t
//
// RUN: %clangxx_asan -O3 -fno-builtin %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s %if system-darwin %{ --check-prefixes=CHECK,DARWIN %} %else %{ --check-prefixes=CHECK,NON_DARWIN %}
// RUN: echo "interceptor_via_fun:bad_function" > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"' %run %t
// RUN: echo "interceptor_name:strncpy" > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"' %run %t

// UNSUPPORTED: android

#include "defines.h"
#include <string.h>


// Don't inline function otherwise stacktrace changes.
ATTRIBUTE_NOINLINE void bad_function() {
  char buffer[] = "hello";
  // CHECK: strncpy-param-overlap: memory ranges
  // CHECK: [{{0x.*,[ ]*0x.*}}) and [{{0x.*,[ ]*0x.*}}) overlap
  // DARWIN: {{#0 0x.* in .*strncpy.cold}}
  // DARWIN: {{#1 0x.* in .*strncpy}}
  // DARWIN: {{#2 0x.* in bad_function.*strncpy-overlap.cpp:}}[[@LINE+5]]
  // DARWIN: {{#3 0x.* in main .*strncpy-overlap.cpp:}}[[@LINE+8]]
  // NON_DARWIN: {{#0 0x.* in .*strncpy}}
  // NON_DARWIN: {{#1 0x.* in bad_function.*strncpy-overlap.cpp:}}[[@LINE+2]]
  // NON_DARWIN: {{#2 0x.* in main .*strncpy-overlap.cpp:}}[[@LINE+5]]
  strncpy(buffer, buffer + 1, 5); // BOOM
}

int main(int argc, char **argv) {
  bad_function();
  return 0;
}
