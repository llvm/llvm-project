// RUN: %clangxx_asan %s -o %t
// RUN: %env_asan_opts=strict_init_order=true %run %t 2>&1 | FileCheck %s

// CHECK: WARNING: strict_init_order is not supported on AIX.

int main() { return 0; }
