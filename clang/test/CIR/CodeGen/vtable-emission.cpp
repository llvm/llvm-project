// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct S {
  virtual void key();
  virtual void nonKey() {}
};

void S::key() {}

// The definition of the key function should result in the vtable being emitted.
// CHECK: cir.global external @_ZTV1S = #cir.vtable

// The reference from the vtable should result in nonKey being emitted.
// CHECK: cir.func linkonce_odr @_ZN1S6nonKeyEv({{.*}} {
