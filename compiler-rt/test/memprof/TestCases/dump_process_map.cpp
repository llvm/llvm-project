// Check print_module_map option.

// RUN: %clangxx_memprof -O0 %s -o %t
// RUN: %env_memprof_opts=print_text=true:log_path=stderr:print_module_map=1 %run %t 2>&1 | FileCheck %s
// RUN: %env_memprof_opts=print_text=true:log_path=stderr:print_module_map=0 %run %t 2>&1 | FileCheck %s --check-prefix=NOMAP

// CHECK: Process memory map follows:
// CHECK: dump_process_map.cpp.tmp
// CHECK: End of process memory map.
// NOMAP-NOT: memory map

int main() {
  return 0;
}
