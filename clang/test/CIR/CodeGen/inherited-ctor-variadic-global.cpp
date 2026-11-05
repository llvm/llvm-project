// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s

struct A { A(...); };
struct B : A { using A::A; };

B b(4);

// CHECK: cir.global external @b
// CHECK-LABEL: cir.func internal private @__cxx_global_var_init()
// CHECK: cir.get_global @b : !cir.ptr<!rec_B>
// CHECK: cir.const #cir.int<4> : !s32i
// CHECK: cir.call @_ZN1AC2Ez
