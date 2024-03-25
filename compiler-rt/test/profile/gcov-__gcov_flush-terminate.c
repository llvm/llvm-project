/// https://bugs.llvm.org/show_bug.cgi?id=38067
/// An abnormal exit does not clear execution counts of subsequent instructions.
// RUN: mkdir -p %t.dir && cd %t.dir
// RUN: %clang --coverage %s -o %t -dumpdir ./
// RUN: test -f gcov-__gcov_flush-terminate.gcno

// RUN: rm -f gcov-__gcov_flush-terminate.gcda && %expect_crash %run %t
// RUN: llvm-cov gcov -t gcov-__gcov_flush-terminate.gcda | FileCheck %s

// CHECK:             -:    0:Runs:1
// CHECK-NEXT:        -:    0:Programs:1

void __gcov_dump(void);
void __gcov_reset(void);

int main(void) {                   // CHECK:      1: [[#@LINE]]:int main(void)
  int i = 22;                      // CHECK-NEXT: 1: [[#@LINE]]:
  __gcov_dump();                   // CHECK-NEXT: 1: [[#@LINE]]:
  __gcov_reset();                  // CHECK-NEXT: 1: [[#@LINE]]:
  i = 42;                          // CHECK-NEXT: 1: [[#@LINE]]:
  __builtin_trap();                // CHECK-NEXT: 1: [[#@LINE]]:
  i = 84;                          // CHECK-NEXT: 1: [[#@LINE]]:
  return 0;                        // CHECK-NEXT: 1: [[#@LINE]]:
}
