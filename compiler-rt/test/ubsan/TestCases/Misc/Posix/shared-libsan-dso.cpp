// RUN: mkdir -p %t.dir
// RUN: %clangxx -DBUILD_SO -fsanitize=signed-integer-overflow -shared-libsan \
// RUN:   %gmlt -fPIC -shared %s -o %dynamiclib %ld_flags_rpath_so
// RUN: %clangxx %s -o %t.dir/exe %ld_flags_rpath_exe
// RUN: %env_ubsan_opts=print_stacktrace=1 %run %t.dir/exe 2>&1 | FileCheck %s

// REQUIRES: target={{.*linux.*}}, ubsan-standalone

#include <limits.h>

#ifdef BUILD_SO
extern "C" __attribute__((noinline)) int overflow(int x) {
  // CHECK: runtime error: signed integer overflow
  // CHECK-NEXT: #0 {{.*}} in overflow {{.*}}shared-libsan-dso.cpp:[[#@LINE+1]]
  return x + 1;
}
#else
extern "C" int overflow(int x);
int main() {
  overflow(INT_MAX);
  return 0;
}
#endif
