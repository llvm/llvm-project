// UNSUPPORTED: ios
// RUN: %clangxx_asan -O0 -g %s -o %t.executable

// Deliberately don't produce the module map and then check that offline symbolization fails
// when we try to look for it.
// RUN: %env_asan_opts="symbolize=0,print_module_map=0" not %run %t.executable > %t_no_module_map.log 2>&1
// RUN: not %asan_symbolize --module-map %t_no_module_map.log --force-system-symbolizer < %t_no_module_map.log 2>&1 | FileCheck -check-prefix=CHECK-NO-MM %s
// CHECK-NO-MM: ERROR:{{.*}} Failed to find module map

// Now produce the module map and check we can symbolize.
// RUN: %env_asan_opts="symbolize=0,print_module_map=2" not %run %t.executable > %t_with_module_map.log 2>&1
// RUN: %asan_symbolize --module-map %t_with_module_map.log --force-system-symbolizer < %t_with_module_map.log 2>&1 | FileCheck -check-prefix=CHECK-MM %s

#include <cstdlib>

// CHECK-MM: WRITE of size 4

extern "C" void foo(int* a) {
  // CHECK-MM: #0 0x{{.+}} in foo {{.*}}asan-symbolize-with-module-map.cc:[[@LINE+1]]
  *a = 5;
}

int main() {
  int* a = (int*) malloc(sizeof(int));
  if (!a)
    return 0;
  free(a);
  // CHECK-MM: #1 0x{{.+}} in main {{.*}}asan-symbolize-with-module-map.cc:[[@LINE+1]]
  foo(a);
  return 0;
}
