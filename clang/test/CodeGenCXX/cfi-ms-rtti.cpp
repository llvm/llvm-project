// RUN: %clang_cc1 -flto -emit-llvm -o - -triple=x86_64-pc-win32 %s -fsanitize=cfi-vcall | FileCheck --check-prefix=RTTI %s
// RUN: %clang_cc1 -flto -emit-llvm -o - -triple=x86_64-pc-win32 %s -fsanitize=cfi-vcall -fno-rtti-data | FileCheck --check-prefix=NO-RTTI %s

struct A {
  A();
  virtual void f() {}
};

A::A() {}

// RTTI: !{i64 8, !"?AUA@@"}
// NO-RTTI: !{i64 0, !"?AUA@@"}
