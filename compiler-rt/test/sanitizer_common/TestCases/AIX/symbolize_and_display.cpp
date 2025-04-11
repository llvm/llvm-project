// On AIX, the module path passed to the symbolizer is /proc/<pid>/object/<object_id>, while the
// path given for stack traces is not the full path in the case of executables, so here we test both
// the symbolization and the path display.

// RUN: %clangxx -O0 %s -o %t
// RUN:   %run %t 2>&1 | FileCheck %s

#include <sanitizer/common_interface_defs.h>

int main() {
  __sanitizer_print_stack_trace();
  // CHECK: #2 0x{{.*}} in .__start (symbolize_and_display.cpp.tmp+0x{{.*}})
}
