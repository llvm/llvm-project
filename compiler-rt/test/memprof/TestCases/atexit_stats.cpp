// Check atexit option.

// RUN: %clangxx_memprof -O0 %s -o %t
// RUN: %env_memprof_opts=print_text=true:log_path=stderr:atexit=1 %run %t 2>&1 | FileCheck %s
// RUN: %env_memprof_opts=print_text=true:log_path=stderr:atexit=0 %run %t 2>&1 | FileCheck %s --check-prefix=NOATEXIT

// CHECK: MemProfiler exit stats:
// CHECK: Stats: {{[0-9]+}}M malloced ({{[0-9]+}}M for overhead) by {{[0-9]+}} calls
// CHECK: Stats: {{[0-9]+}}M realloced by {{[0-9]+}} calls
// CHECK: Stats: {{[0-9]+}}M freed by {{[0-9]+}} calls
// CHECK: Stats: {{[0-9]+}}M really freed by {{[0-9]+}} calls
// CHECK: Stats: {{[0-9]+}}M ({{[0-9]+}}M-{{[0-9]+}}M) mmaped; {{[0-9]+}} maps, {{[0-9]+}} unmaps
// CHECK:   mallocs by size class:
// CHECK: Stats: malloc large: {{[0-9]+}}

// NOATEXIT-NOT: MemProfiler exit stats

int main() {
  return 0;
}
