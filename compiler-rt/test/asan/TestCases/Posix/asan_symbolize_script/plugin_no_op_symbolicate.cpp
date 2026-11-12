// UNSUPPORTED: ios, android
// Check plugin command line args get parsed and that plugin functions get called as expected.

// RUN: %clangxx_asan -O0 -g %s -o %t.executable
// RUN: not %env_asan_opts=symbolize=0 %run %t.executable > %t.log 2>&1
// RUN: %asan_symbolize --plugins %S/plugin_no_op.py --log-level info -l %t.log --unlikely-option-name-XXX=15 2>&1 | FileCheck %s

// CHECK: GOT --unlikely-option-name-XXX=15
// CHECK: filter_binary_path called in NoOpPlugin
// CHECK: destroy() called on NoOpPlugin

#include <cstdlib>
extern "C" void foo(int* a) {
  *a = 5;
}

int main() {
  int* a = (int*) malloc(sizeof(int));
  if (!a)
    return 0;
  free(a);
  foo(a);
  return 0;
}
