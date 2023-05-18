// UNSUPPORTED: ios
// FIXME(dliew): We currently have to use module map for this test due to the atos
// symbolizer changing the module name from an absolute path to just the file name.
// rdar://problem/49784442
//
// Simulate partial symbolication (can happen with %L specifier) by printing
// out %L's fallback which will print the module name and offset instead of a
// source location.
// RUN: %clangxx_asan -O0 -g %s -o %t.executable
// RUN: %env_asan_opts=symbolize=1,print_module_map=1,stack_trace_format='"    #%%n %%p %%F %%M"' not %run %t.executable > %t.log 2>&1
// RUN: FileCheck -input-file=%t.log -check-prefix=CHECK-PS %s
// Now try to full symbolicate using the module map.
// RUN: %asan_symbolize --module-map %t.log --force-system-symbolizer < %t.log > %t.fully_symbolized
// RUN: FileCheck -input-file=%t.fully_symbolized -check-prefix=CHECK-FS %s

#include <cstdlib>

// Partially symbolicated back-trace where symbol is available but
// source location is not and instead module name and offset are
// printed.
// CHECK-PS: WRITE of size 4
// CHECK-PS: #0 0x{{.+}} in foo ({{.+}}.executable:{{.+}}+0x{{.+}})
// CHECK-PS: #1 0x{{.+}} in main ({{.+}}.executable:{{.+}}+0x{{.+}})

// CHECK-FS: WRITE of size 4

extern "C" void foo(int* a) {
  // CHECK-FS: #0 0x{{.+}} in foo {{.*}}asan-symbolize-partial-report-with-module-map.cpp:[[@LINE+1]]
  *a = 5;
}

int main() {
  int* a = (int*) malloc(sizeof(int));
  if (!a)
    return 0;
  free(a);
  // CHECK-FS: #1 0x{{.+}} in main {{.*}}asan-symbolize-partial-report-with-module-map.cpp:[[@LINE+1]]
  foo(a);
  return 0;
}
