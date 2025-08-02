// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

struct A {
  virtual void f(char);
};

// This is just here to force the class definition to be emitted without
// requiring any other support. It will be removed when more complete
// vtable support is implemented.
A *a;

// CIR: !rec_A = !cir.record<struct "A" {!cir.vptr}>
