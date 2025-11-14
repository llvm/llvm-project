// This test simply checks that the "Invalid dyld module map" warning is not printed
// in the output of a backtrace.

// RUN: %clangxx_asan -DSHARED_LIB -g %s -dynamiclib -o %t.dylib
// RUN: %clangxx_asan -O0 -g %s %t.dylib -o %t.executable
// RUN: %env_asan_opts="print_module_map=2" not %run %t.executable 2>&1 | FileCheck %s -DDYLIB=%{t:stem}.tmp.dylib

// CHECK-NOT: WARN: Invalid dyld module map
// CHECK-DAG: 0x{{.*}}-0x{{.*}} {{.*}}[[DYLIB]]
// CHECK-DAG: 0x{{.*}}-0x{{.*}} {{.*}}libsystem

#ifdef SHARED_LIB
extern "C" void foo(int *a) { *a = 5; }
#else
#  include <cstdlib>

extern "C" void foo(int *a);

int main() {
  int *a = (int *)malloc(sizeof(int));
  free(a);
  foo(a);
  return 0;
}
#endif