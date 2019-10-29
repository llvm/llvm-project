// UNSUPPORTED: ios
// When `external_symbolizer_path` is empty on Darwin we fallback on using
// dladdr as the symbolizer which means we get the symbol name
// but no source location. The current implementation also doesn't try to
// change the module name so we end up with the full name so we actually don't
// need the module map here.

// RUN: %clangxx_asan -O0 -g %s -o %t.executable
// RUN: %env_asan_opts=symbolize=1,print_module_map=0,external_symbolizer_path= not %run %t.executable > %t2.log 2>&1
// RUN: FileCheck -input-file=%t2.log -check-prefix=CHECK-PS %s
// RUN: %asan_symbolize --force-system-symbolizer < %t2.log > %t2.fully_symbolized
// RUN: FileCheck -input-file=%t2.fully_symbolized -check-prefix=CHECK-FS %s

#include <cstdlib>

// Partially symbolicated back-trace where symbol is available but
// source location is not and instead module name and offset are
// printed.
// CHECK-PS: WRITE of size 4
// CHECK-PS: #0 0x{{.+}} in foo{{(\+0x[0-9a-f]+)?}} ({{.+}}.executable:{{.+}}+0x{{.+}})
// CHECK-PS: #1 0x{{.+}} in main{{(\+0x[0-9a-f]+)?}} ({{.+}}.executable:{{.+}}+0x{{.+}})

// CHECK-FS: WRITE of size 4

extern "C" void foo(int* a) {
  // CHECK-FS: #0 0x{{.+}} in foo {{.*}}asan-symbolize-partial-report-no-external-symbolizer.cpp:[[@LINE+1]]
  *a = 5;
}

int main() {
  int* a = (int*) malloc(sizeof(int));
  if (!a)
    return 0;
  free(a);
  // CHECK-FS: #1 0x{{.+}} in main {{.*}}asan-symbolize-partial-report-no-external-symbolizer.cpp:[[@LINE+1]]
  foo(a);
  return 0;
}
