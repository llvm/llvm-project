// Test various levels of coverage
//
// RUN: %clangxx_msan -DINIT_VAR=1 -O1 -fsanitize-coverage=func  %s -o %t
// RUN: mkdir -p %T/coverage-levels
// RUN: MSAN_OPTIONS=coverage=1:verbosity=1:coverage_dir=%T/coverage-levels %run %t 2>&1 | FileCheck %s --check-prefix=CHECK1 --check-prefix=CHECK_NOWARN
// RUN: %clangxx_msan -O1 -fsanitize-coverage=func  %s -o %t
// RUN: MSAN_OPTIONS=coverage=1:verbosity=1:coverage_dir=%T/coverage-levels not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK1 --check-prefix=CHECK_WARN
// RUN: %clangxx_msan -O1 -fsanitize-coverage=bb  %s -o %t
// RUN: MSAN_OPTIONS=coverage=1:verbosity=1:coverage_dir=%T/coverage-levels not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK2 --check-prefix=CHECK_WARN
// RUN: %clangxx_msan -O1 -fsanitize-coverage=edge  %s -o %t
// RUN: MSAN_OPTIONS=coverage=1:verbosity=1:coverage_dir=%T/coverage-levels not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK3 --check-prefix=CHECK_WARN

volatile int sink;
int main(int argc, char **argv) {
  int var;
#if INIT_VAR
  var = 0;
#endif
  if (argc == 0)
    sink = 0;
  return *(volatile int*)&var;
}

// CHECK_WARN: WARNING: MemorySanitizer: use-of-uninitialized-value
// CHECK_NOWARN-NOT: ERROR
// CHECK1:  1 PCs written
// CHECK2:  1 PCs written
// CHECK3:  2 PCs written
