// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -I%S/../Inputs -clangir-disable-emit-cxx-default -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

#include "std-cxx.h"


struct S {
  S(int, int);
};

void m(int a, int b) {
  std::shared_ptr<S> l = std::make_shared<S>(a, b);
}

// CHECK: cir.func linkonce_odr @_ZSt11make_sharedI1SJRiS1_EESt10shared_ptrIT_EDpOT0_(
// CHECK:   %0 = cir.alloca !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>, ["args", init] {alignment = 8 : i64}
// CHECK:   %1 = cir.alloca !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>, ["args", init] {alignment = 8 : i64}
// CHECK:   %2 = cir.alloca !ty_22class2Estd3A3Ashared_ptr22, cir.ptr <!ty_22class2Estd3A3Ashared_ptr22>, ["__retval"] {alignment = 1 : i64}
// CHECK:   cir.store %arg0, %0 : !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>
// CHECK:   cir.store %arg1, %1 : !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>
// CHECK:   cir.scope {
// CHECK:     %4 = cir.const(#cir.int<1> : !u64i) : !u64i
// CHECK:     %5 = cir.call @_Znwm(%4) : (!u64i) -> !cir.ptr<!void>
// CHECK:     %6 = cir.cast(bitcast, %5 : !cir.ptr<!void>), !cir.ptr<!ty_22struct2ES22>
// CHECK:     %7 = cir.load %0 : cir.ptr <!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK:     %8 = cir.load %7 : cir.ptr <!s32i>, !s32i
// CHECK:     %9 = cir.load %1 : cir.ptr <!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK:     %10 = cir.load %9 : cir.ptr <!s32i>, !s32i
// CHECK:     cir.call @_ZN1SC1Eii(%6, %8, %10) : (!cir.ptr<!ty_22struct2ES22>, !s32i, !s32i) -> ()
// CHECK:     cir.call @_ZNSt10shared_ptrI1SEC1EPS0_(%2, %6) : (!cir.ptr<!ty_22class2Estd3A3Ashared_ptr22>, !cir.ptr<!ty_22struct2ES22>) -> ()
// CHECK:   }