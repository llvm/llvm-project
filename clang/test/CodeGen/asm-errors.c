// REQUIRES: x86-registered-target

// RUN: not %clang_cc1 -triple i386-apple-darwin10 -emit-obj %s -o /dev/null > %t 2>&1
// RUN: FileCheck %s < %t
// RUN: not %clang -target i386-apple-darwin10 -fembed-bitcode -c %s -o /dev/null 2>&1 | \
// RUN:   FileCheck --check-prefix=CRASH-REPORT %s
// CRASH-REPORT: <inline asm>:
// CRASH-REPORT: error: invalid instruction mnemonic 'abc'
// CRASH-REPORT: error: cannot compile inline asm
// CRASH-REPORT-NOT: note: diagnostic msg:

int test1(int X) {
// CHECK: error: invalid instruction mnemonic 'abc'
  __asm__ ("abc incl    %0" : "+r" (X));
  return X;
}
