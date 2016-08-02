// REQUIRES: x86-registered-target

// RUN: true
// UN: not %clang_cc1 -triple i386-apple-darwin10 -emit-obj %s -o /dev/null > %t 2>&1
// UN: FileCheck %s < %t
// RUN: %clang_cc1 -triple i386-apple-darwin10 -emit-llvm-bc %s -o %t.bc
// RUN: %clang_cc1 -triple i386-apple-darwin10 -emit-obj %t.bc -o /dev/null 2>&1 | \
// RUN:   FileCheck --check-prefix=CRASH-REPORT %s
// CRASH-REPORT: <inline asm>:
// CRASH-REPORT: error: invalid instruction mnemonic 'abc'
// CRASH-REPORT-NOT: note: diagnostic msg:

int test1(int X) {
// CHECK: error: invalid instruction mnemonic 'abc'
  __asm__ ("abc incl    %0" : "+r" (X));
  return X;
}
