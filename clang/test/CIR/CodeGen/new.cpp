// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -I%S/../Inputs -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

#include "std-cxx.h"

struct S {
  S(int, int);
};

void m(int a, int b) {
  std::shared_ptr<S> l = std::make_shared<S>(a, b);
}

// CHECK: cir.func linkonce_odr @_ZSt11make_sharedI1SJRiS1_EESt10shared_ptrIT_EDpOT0_(
// CHECK:   %0 = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["args", init, const] {alignment = 8 : i64}
// CHECK:   %1 = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["args", init, const] {alignment = 8 : i64}
// CHECK:   %2 = cir.alloca !ty_std3A3Ashared_ptr3CS3E, !cir.ptr<!ty_std3A3Ashared_ptr3CS3E>, ["__retval"] {alignment = 1 : i64}
// CHECK:   cir.store %arg0, %0 : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK:   cir.store %arg1, %1 : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK:   cir.scope {
// CHECK:     %4 = cir.const #cir.int<1> : !u64i
// CHECK:     %5 = cir.call @_Znwm(%4) : (!u64i) -> !cir.ptr<!void>
// CHECK:     %6 = cir.cast(bitcast, %5 : !cir.ptr<!void>), !cir.ptr<!ty_S>
// CHECK:     %7 = cir.load %0 : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK:     %8 = cir.load %7 : !cir.ptr<!s32i>, !s32i
// CHECK:     %9 = cir.load %1 : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK:     %10 = cir.load %9 : !cir.ptr<!s32i>, !s32i
// CHECK:     cir.call @_ZN1SC1Eii(%6, %8, %10) : (!cir.ptr<!ty_S>, !s32i, !s32i) -> ()
// CHECK:     cir.call @_ZNSt10shared_ptrI1SEC1EPS0_(%2, %6) : (!cir.ptr<!ty_std3A3Ashared_ptr3CS3E>, !cir.ptr<!ty_S>) -> ()
// CHECK:   }

class B {
public:
  void construct(B* __p) {
      ::new ((void*)__p) B;
  }
};

// CHECK: cir.func linkonce_odr @_ZN1B9constructEPS_(%arg0: !cir.ptr<!ty_B>
// CHECK:   %0 = cir.alloca !cir.ptr<!ty_B>, !cir.ptr<!cir.ptr<!ty_B>>, ["this", init] {alignment = 8 : i64}
// CHECK:   %1 = cir.alloca !cir.ptr<!ty_B>, !cir.ptr<!cir.ptr<!ty_B>>, ["__p", init] {alignment = 8 : i64}
// CHECK:   cir.store %arg0, %0 : !cir.ptr<!ty_B>, !cir.ptr<!cir.ptr<!ty_B>>
// CHECK:   cir.store %arg1, %1 : !cir.ptr<!ty_B>, !cir.ptr<!cir.ptr<!ty_B>>
// CHECK:   %2 = cir.load %0 : !cir.ptr<!cir.ptr<!ty_B>>, !cir.ptr<!ty_B>
// CHECK:   %3 = cir.const #cir.int<1> : !u64i
// CHECK:   %4 = cir.load %1 : !cir.ptr<!cir.ptr<!ty_B>>, !cir.ptr<!ty_B>
// CHECK:   %5 = cir.cast(bitcast, %4 : !cir.ptr<!ty_B>), !cir.ptr<!void>
// CHECK:   %6 = cir.cast(bitcast, %5 : !cir.ptr<!void>), !cir.ptr<!ty_B>

// cir.call @B::B()(%new_placament_ptr)
// CHECK:   cir.call @_ZN1BC1Ev(%6) : (!cir.ptr<!ty_B>) -> ()
// CHECK:   cir.return
// CHECK: }

void t() {
  B b;
  b.construct(&b);
}


void t_new_constant_size() {
  auto p = new double[16];
}

// In this test, NUM_ELEMENTS isn't used because no cookie is needed and there
//   are no constructor calls needed.

// CHECK:   cir.func @_Z19t_new_constant_sizev()
// CHECK:    %0 = cir.alloca !cir.ptr<!cir.double>, !cir.ptr<!cir.ptr<!cir.double>>, ["p", init] {alignment = 8 : i64}
// CHECK:    %[[#NUM_ELEMENTS:]] = cir.const #cir.int<16> : !u64i
// CHECK:    %[[#ALLOCATION_SIZE:]] = cir.const #cir.int<128> : !u64i
// CHECK:    %3 = cir.call @_Znam(%[[#ALLOCATION_SIZE]]) : (!u64i) -> !cir.ptr<!void>
// CHECK:    %4 = cir.cast(bitcast, %3 : !cir.ptr<!void>), !cir.ptr<!cir.double>
// CHECK:    cir.store %4, %0 : !cir.ptr<!cir.double>, !cir.ptr<!cir.ptr<!cir.double>>
// CHECK:    cir.return
// CHECK:  }

void t_new_multidim_constant_size() {
  auto p = new double[2][3][4];
}

// As above, NUM_ELEMENTS isn't used.

// CHECK:   cir.func @_Z28t_new_multidim_constant_sizev()
// CHECK:    %0 = cir.alloca !cir.ptr<!cir.array<!cir.array<!cir.double x 4> x 3>>, !cir.ptr<!cir.ptr<!cir.array<!cir.array<!cir.double x 4> x 3>>>, ["p", init] {alignment = 8 : i64}
// CHECK:    %[[#NUM_ELEMENTS:]] = cir.const #cir.int<24> : !u64i
// CHECK:    %[[#ALLOCATION_SIZE:]] = cir.const #cir.int<192> : !u64i
// CHECK:    %3 = cir.call @_Znam(%[[#ALLOCATION_SIZE]]) : (!u64i) -> !cir.ptr<!void>
// CHECK:    %4 = cir.cast(bitcast, %3 : !cir.ptr<!void>), !cir.ptr<!cir.double>
// CHECK:    %5 = cir.cast(bitcast, %0 : !cir.ptr<!cir.ptr<!cir.array<!cir.array<!cir.double x 4> x 3>>>), !cir.ptr<!cir.ptr<!cir.double>>
// CHECK:    cir.store %4, %5 : !cir.ptr<!cir.double>, !cir.ptr<!cir.ptr<!cir.double>>
// CHECK:  }

class C {
  public:
    ~C();
};

void t_constant_size_nontrivial() {
  auto p = new C[3];
}

// CHECK:  cir.func @_Z26t_constant_size_nontrivialv()
// CHECK:    %0 = cir.alloca !cir.ptr<!ty_C>, !cir.ptr<!cir.ptr<!ty_C>>, ["p", init] {alignment = 8 : i64}
// CHECK:    %[[#NUM_ELEMENTS:]] = cir.const #cir.int<3> : !u64i
// CHECK:    %[[#SIZE_WITHOUT_COOKIE:]] = cir.const #cir.int<3> : !u64i
// CHECK:    %[[#ALLOCATION_SIZE:]] = cir.const #cir.int<11> : !u64i
// CHECK:    %4 = cir.call @_Znam(%[[#ALLOCATION_SIZE]]) : (!u64i) -> !cir.ptr<!void>
// CHECK:    %5 = cir.cast(bitcast, %4 : !cir.ptr<!void>), !cir.ptr<!u64i>
// CHECK:    cir.store %[[#NUM_ELEMENTS]], %5 : !u64i, !cir.ptr<!u64i>
// CHECK:    %6 = cir.cast(bitcast, %4 : !cir.ptr<!void>), !cir.ptr<!u8i>
// CHECK:    %[[#COOKIE_SIZE:]] = cir.const #cir.int<8> : !s32i
// CHECK:    %8 = cir.ptr_stride(%6 : !cir.ptr<!u8i>, %[[#COOKIE_SIZE]] : !s32i), !cir.ptr<!u8i>
// CHECK:    %9 = cir.cast(bitcast, %8 : !cir.ptr<!u8i>), !cir.ptr<!ty_C>
// CHECK:    cir.store %9, %0 : !cir.ptr<!ty_C>, !cir.ptr<!cir.ptr<!ty_C>>
// CHECK:    cir.return
// CHECK:  }

class D {
  public:
    int x;
    ~D();
};

void t_constant_size_nontrivial2() {
  auto p = new D[3];
}

// In this test SIZE_WITHOUT_COOKIE isn't used, but it would be if there were
// an initializer.

// CHECK:  cir.func @_Z27t_constant_size_nontrivial2v()
// CHECK:    %0 = cir.alloca !cir.ptr<!ty_D>, !cir.ptr<!cir.ptr<!ty_D>>, ["p", init] {alignment = 8 : i64}
// CHECK:    %[[#NUM_ELEMENTS:]] = cir.const #cir.int<3> : !u64i
// CHECK:    %[[#SIZE_WITHOUT_COOKIE:]] = cir.const #cir.int<12> : !u64i
// CHECK:    %[[#ALLOCATION_SIZE:]] = cir.const #cir.int<20> : !u64i
// CHECK:    %4 = cir.call @_Znam(%[[#ALLOCATION_SIZE]]) : (!u64i) -> !cir.ptr<!void>
// CHECK:    %5 = cir.cast(bitcast, %4 : !cir.ptr<!void>), !cir.ptr<!u64i>
// CHECK:    cir.store %[[#NUM_ELEMENTS]], %5 : !u64i, !cir.ptr<!u64i>
// CHECK:    %6 = cir.cast(bitcast, %4 : !cir.ptr<!void>), !cir.ptr<!u8i>
// CHECK:    %[[#COOKIE_SIZE:]] = cir.const #cir.int<8> : !s32i
// CHECK:    %8 = cir.ptr_stride(%6 : !cir.ptr<!u8i>, %[[#COOKIE_SIZE]] : !s32i), !cir.ptr<!u8i>
// CHECK:    %9 = cir.cast(bitcast, %8 : !cir.ptr<!u8i>), !cir.ptr<!ty_D>
// CHECK:    cir.store %9, %0 : !cir.ptr<!ty_D>, !cir.ptr<!cir.ptr<!ty_D>>
// CHECK:    cir.return
// CHECK:  }