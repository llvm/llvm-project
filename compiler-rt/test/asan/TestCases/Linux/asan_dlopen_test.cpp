// Test that dlopen of dynamic runtime is prohibited.
//
// RUN: %clangxx %s -DRT=\"%shared_libasan\" -o %t -ldl
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: %env_asan_opts=verify_asan_link_order=true not %run %t 2>&1 | FileCheck %s
// RUN: %env_asan_opts=verify_asan_link_order=false %run %t 2>&1
// REQUIRES: asan-dynamic-runtime
// XFAIL: android

#include <dlfcn.h>

int main(int argc, char **argv) {
  dlopen(RT, RTLD_LAZY);
  return 0;
}

// CHECK: ASan runtime does not come first in initial library list
