// RUN: mkdir -p %T/coverage
// RUN: %clangxx -O1 -fsanitize-coverage=func  %s -o %t
// RUN: %env_ubsan_opts=coverage=1:coverage_bitset=1:verbosity=1:coverage_dir=%T/coverage %run %t 2>&1 | FileCheck %s --check-prefix=CHECK1
// RUN: %clangxx -O1 -fsanitize-coverage=bb  %s -o %t
// RUN: %env_ubsan_opts=coverage=1:coverage_bitset=1:verbosity=1:coverage_dir=%T/coverage %run %t 2>&1 | FileCheck %s --check-prefix=CHECK2
//
// UNSUPPORTED: ubsan-tsan
// UNSUPPORTED: android
volatile int sink;
int main(int argc, char **argv) {
  if (argc == 0)
    sink = 0;
}

// CHECK1: CovDump: bitset of 1 bits written for '{{.*}}', 1 bits are set
// CHECK1:  1 PCs written
// CHECK2: CovDump: bitset of 2 bits written for '{{.*}}', 1 bits are set
// CHECK2:  1 PCs written
