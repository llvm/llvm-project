// RUN: %clang_cc1 %s -triple=i686-pc-linux-gnu -emit-llvm -o - > %t1
// RUN: grep "ax" %t1
// RUN: grep "bx" %t1
// RUN: grep "cx" %t1
// RUN: grep "dx" %t1
// RUN: grep "di" %t1
// RUN: grep "si" %t1
// RUN: grep "st" %t1
// RUN: grep "st(1)" %t1

void test1(void) {
  int d1, d2;
  asm ("" : "=a" (d1), "=b" (d2) :
       "c" (0), "d" (0), "S" (0), "D" (0), "t" (0), "u" (0));
}
