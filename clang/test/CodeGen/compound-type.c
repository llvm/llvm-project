// RUN: %clang_cc1 < %s -emit-llvm -triple i686-pc-linux-gnu > %t
// RUN: grep "div i32" %t
// RUN: grep "shl i32" %t

unsigned char a,b;
void c(void) {a <<= b;}
void d(void) {a /= b;}
