// Test various levels of coverage
//
// RUN: %clangxx_asan -O1 -fsanitize-coverage=func  %s -o %t
// RUN: %env_asan_opts=coverage=1:coverage_bitset=1:verbosity=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK1
// RUN: %clangxx_asan -O1 -fsanitize-coverage=bb  %s -o %t
// RUN: %env_asan_opts=coverage=1:coverage_bitset=1:verbosity=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK2
// RUN: %clangxx_asan -O1 -fsanitize-coverage=edge  %s -o %t
// RUN: %env_asan_opts=coverage=1:coverage_bitset=1:verbosity=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK3
// RUN: %clangxx_asan -O1 -fsanitize-coverage=edge -mllvm -sanitizer-coverage-block-threshold=0 %s -o %t
// RUN: %env_asan_opts=coverage=1:coverage_bitset=1:verbosity=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK3

// RUN: %env_asan_opts=coverage=1:coverage_bitset=0:verbosity=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK3_NOBITSET
// RUN: %env_asan_opts=coverage=1:verbosity=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK3_NOBITSET
// RUN: %env_asan_opts=coverage=1:coverage_pcs=0:verbosity=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK3_NOPCS
//
// REQUIRES: asan-64-bits
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
// CHECK3: CovDump: bitset of 3 bits written for '{{.*}}', 2 bits are set
// CHECK3:  2 PCs written
// CHECK3_NOBITSET-NOT: bitset of
// CHECK3_NOPCS-NOT: PCs written
