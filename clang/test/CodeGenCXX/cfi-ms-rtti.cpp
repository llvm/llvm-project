// RUN: %clang_cc1 -flto -emit-llvm -o - -triple=x86_64-pc-win32 %s -fsanitize=cfi-vcall | FileCheck --check-prefix=RTTI %s
// RUN: %clang_cc1 -flto -emit-llvm -o - -triple=x86_64-pc-win32 %s -fsanitize=cfi-vcall -fno-rtti-data | FileCheck --check-prefix=NO-RTTI %s

struct A {
  A();
  virtual void f() {}
};

A::A() {}

// RTTI: !{!"?AUA@@", [2 x i8*]* {{.*}}, i64 8}
// NO-RTTI: !{!"?AUA@@", [1 x i8*]* {{.*}}, i64 0}
