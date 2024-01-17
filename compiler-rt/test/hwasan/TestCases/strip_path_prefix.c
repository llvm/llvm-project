// RUN: %clang_hwasan -O0 -g %s -o %t && %env_hwasan_opts=strip_path_prefix=/TestCases/ not %run %t 2>&1 | FileCheck %s

// Stack histories currently are not recorded on x86.
// XFAIL: target=x86_64{{.*}}

#include <assert.h>
#include <sanitizer/hwasan_interface.h>
#include <stdio.h>

int t;

__attribute__((noinline)) char *buggy() {
  char *volatile p;
  char zzz = {};
  char yyy = {};
  p = t ? &yyy : &zzz;
  return p;
}

int main() {
  char *p = buggy();
  return *p;
  // CHECK: READ of size 1 at
  // CHECK: #0 {{.*}} in main strip_path_prefix.c:[[@LINE-2]]
  // CHECK: Potentially referenced stack objects:
  // CHECK: in buggy strip_path_prefix.c:[[@LINE-12]]
}
