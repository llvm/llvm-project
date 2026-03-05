// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu \
// RUN:   -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir \
// RUN:   -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir

namespace {
struct A {
  ~A() {}
};

struct B : public A {};
}

B x;

// CHECK: cir.call @_ZN12_GLOBAL__N_11AD2Ev({{.*}}) : (!cir.ptr<!rec_28anonymous_namespace293A3AA>) -> ()
