// RUN: %clang_cc1 %s -g -S -emit-llvm -o - | FileCheck %s

struct A {
  virtual void f();
};

struct B {
  virtual void f();
};

struct C : A, B {
  virtual void f();
};

void C::f() { }

// CHECK: [ DW_TAG_subprogram ] [line 15] [def] [_ZThn{{4|8}}_N1C1fEv]
