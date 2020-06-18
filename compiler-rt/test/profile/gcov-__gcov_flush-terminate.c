/// https://bugs.llvm.org/show_bug.cgi?id=38067
/// An abnormal exit does not clear execution counts of subsequent instructions.
// RUN: mkdir -p %t.dir && cd %t.dir
// RUN: %clang --coverage %s -o %t
// RUN: test -f gcov-__gcov_flush-terminate.gcno

// RUN: rm -f gcov-__gcov_flush-terminate.gcda && %expect_crash %run %t
// RUN: llvm-cov gcov -t gcov-__gcov_flush-terminate.gcda | FileCheck %s

// CHECK:             -:    0:Runs:1
// CHECK-NEXT:        -:    0:Programs:1
// CHECK:             -:    1:void __gcov_flush(void);
// CHECK-NEXT:        -:    2:
// CHECK-NEXT:        1:    3:int main(void) {
// CHECK-NEXT:        1:    4:  int i = 22;
// CHECK-NEXT:        1:    5:  __gcov_flush();
// CHECK-NEXT:        1:    6:  i = 42;
// CHECK-NEXT:        1:    7:  __builtin_trap();
// CHECK-NEXT:        1:    8:  i = 84;
// CHECK-NEXT:        1:    9:  return 0;

void __gcov_flush(void);

int main(void) {
  int i = 22;
  __gcov_flush();
  i = 42;
  __builtin_trap();
  i = 84;
  return 0;
}
