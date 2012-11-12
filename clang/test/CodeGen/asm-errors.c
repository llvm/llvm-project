// REQUIRES: x86-registered-target

// RUN: true
// UN: not %clang_cc1 -triple i386-apple-darwin10 -emit-obj %s -o /dev/null > %t 2>&1
// UN: FileCheck %s < %t

int test1(int X) {
// CHECK: error: invalid instruction mnemonic 'abc'
  __asm__ ("abc incl    %0" : "+r" (X));
  return X;
}
