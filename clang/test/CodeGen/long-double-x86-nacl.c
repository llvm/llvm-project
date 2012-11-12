// RUN: %clang_cc1 %s -emit-llvm -o - -triple=i686-unknown-nacl | FileCheck %s

long double x = 0;
int checksize[sizeof(x) == 8 ? 1 : -1];

// CHECK: define void @s1(double %a)
void s1(long double a) {}
