// RUN: %clangxx_memprof -O0 %s -o %t && %env_memprof_opts=help=1 %run %t 2>&1 | FileCheck %s

int main() {
}

// CHECK: Available flags for MemProfiler:
// CHECK-DAG: atexit
