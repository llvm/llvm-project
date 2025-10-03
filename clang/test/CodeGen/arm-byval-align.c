// RUN: %clang_cc1 -triple=armv7-none-eabi < %s -emit-llvm | FileCheck %s

struct foo {
  long long a;
  char b;
  int c:16;
  int d[16];
};

// CHECK: ptr noundef byval(%struct.foo) align 8 %z
long long bar(int a, int b, int c, int d, int e,
              struct foo z) {
  return z.a;
}
