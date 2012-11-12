// RUN: %clang_cc1 -emit-llvm -o %t %s
// RUN: grep '@g0 = external hidden constant i32' %t
// RUN: grep '@g1 = hidden constant i32 1' %t

__private_extern__ const int g0;
__private_extern__ const int g1 = 1;

int f0(void) {
  return g0;
}
