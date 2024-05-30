// RUN: %clang_cc1 -ast-print %s -o - | FileCheck %s

// CHECK: extern "C" int printf(const char *, ...);
extern "C" int printf(const char *...);

// CHECK: extern "C++" int f(int);
// CHECK-NEXT: extern "C++" int g(int);
extern "C++" int f(int), g(int);

// CHECK: extern "C" char a;
// CHECK-NEXT: extern "C" char b;
extern "C" char a, b;

// CHECK: extern "C" {
// CHECK-NEXT:  void foo();
// CHECK-NEXT:  int x;
// CHECK-NEXT:  int y;
// CHECK-NEXT:  extern short z;
// CHECK-NEXT: }
extern "C" {
  void foo(void);
  int x, y;
  extern short z;
}

// CHECK: extern "C" {
// CHECK-NEXT: }
extern "C" {}

// CHECK: extern "C++";
extern "C++";
