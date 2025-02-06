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
// CHECK:    %[[NUM_ELEMENTS:.*]] = cir.const #cir.int<16> : !u64i
// CHECK:    %[[ALLOCATION_SIZE:.*]] = cir.const #cir.int<128> : !u64i
// CHECK:    %3 = cir.call @_Znam(%[[ALLOCATION_SIZE]]) : (!u64i) -> !cir.ptr<!void>
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
// CHECK:    %[[NUM_ELEMENTS:.*]] = cir.const #cir.int<24> : !u64i
// CHECK:    %[[ALLOCATION_SIZE:.*]] = cir.const #cir.int<192> : !u64i
// CHECK:    %3 = cir.call @_Znam(%[[ALLOCATION_SIZE]]) : (!u64i) -> !cir.ptr<!void>
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
// CHECK:    %[[NUM_ELEMENTS:.*]] = cir.const #cir.int<3> : !u64i
// CHECK:    %[[SIZE_WITHOUT_COOKIE:.*]] = cir.const #cir.int<3> : !u64i
// CHECK:    %[[ALLOCATION_SIZE:.*]] = cir.const #cir.int<11> : !u64i
// CHECK:    %4 = cir.call @_Znam(%[[ALLOCATION_SIZE]]) : (!u64i) -> !cir.ptr<!void>
// CHECK:    %5 = cir.cast(bitcast, %4 : !cir.ptr<!void>), !cir.ptr<!u64i>
// CHECK:    cir.store %[[NUM_ELEMENTS]], %5 : !u64i, !cir.ptr<!u64i>
// CHECK:    %6 = cir.cast(bitcast, %4 : !cir.ptr<!void>), !cir.ptr<!u8i>
// CHECK:    %[[COOKIE_SIZE:.*]] = cir.const #cir.int<8> : !s32i
// CHECK:    %8 = cir.ptr_stride(%6 : !cir.ptr<!u8i>, %[[COOKIE_SIZE]] : !s32i), !cir.ptr<!u8i>
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
// CHECK:    %[[NUM_ELEMENTS:.*]] = cir.const #cir.int<3> : !u64i
// CHECK:    %[[SIZE_WITHOUT_COOKIE:.*]] = cir.const #cir.int<12> : !u64i
// CHECK:    %[[ALLOCATION_SIZE:.*]] = cir.const #cir.int<20> : !u64i
// CHECK:    %4 = cir.call @_Znam(%[[ALLOCATION_SIZE]]) : (!u64i) -> !cir.ptr<!void>
// CHECK:    %5 = cir.cast(bitcast, %4 : !cir.ptr<!void>), !cir.ptr<!u64i>
// CHECK:    cir.store %[[NUM_ELEMENTS]], %5 : !u64i, !cir.ptr<!u64i>
// CHECK:    %6 = cir.cast(bitcast, %4 : !cir.ptr<!void>), !cir.ptr<!u8i>
// CHECK:    %[[COOKIE_SIZE:.*]] = cir.const #cir.int<8> : !s32i
// CHECK:    %8 = cir.ptr_stride(%6 : !cir.ptr<!u8i>, %[[COOKIE_SIZE]] : !s32i), !cir.ptr<!u8i>
// CHECK:    %9 = cir.cast(bitcast, %8 : !cir.ptr<!u8i>), !cir.ptr<!ty_D>
// CHECK:    cir.store %9, %0 : !cir.ptr<!ty_D>, !cir.ptr<!cir.ptr<!ty_D>>
// CHECK:    cir.return
// CHECK:  }

void t_constant_size_memset_init() {
  auto p = new int[16] {};
}

// In this test, NUM_ELEMENTS isn't used because no cookie is needed and there
//   are no constructor calls needed.

// CHECK:  cir.func @_Z27t_constant_size_memset_initv()
// CHECK:    %[[NUM_ELEMENTS:.*]] = cir.const #cir.int<16> : !u64i
// CHECK:    %[[ALLOCATION_SIZE:.*]] = cir.const #cir.int<64> : !u64i
// CHECK:    %[[ALLOC_PTR:.*]] = cir.call @_Znam(%[[ALLOCATION_SIZE]]) : (!u64i) -> !cir.ptr<!void>
// CHECK:    %[[ELEM_PTR:.*]] = cir.cast(bitcast, %[[ALLOC_PTR]] : !cir.ptr<!void>), !cir.ptr<!s32i>
// CHECK:    %[[VOID_PTR:.*]] = cir.cast(bitcast, %[[ELEM_PTR]] : !cir.ptr<!s32i>), !cir.ptr<!void>
// CHECK:    %[[ZERO:.*]] = cir.const #cir.int<0> : !u8i
// CHECK:    %[[ZERO_I32:.*]] = cir.cast(integral, %[[ZERO]] : !u8i), !s32i
// CHECK:    cir.libc.memset %[[ALLOCATION_SIZE]] bytes from %[[VOID_PTR]] set to %[[ZERO_I32]] : !cir.ptr<!void>, !s32i, !u64i

void t_constant_size_partial_init() {
  auto p = new int[16] { 1, 2, 3 };
}

// CHECK:  cir.func @_Z28t_constant_size_partial_initv()
// CHECK:    %[[NUM_ELEMENTS:.*]] = cir.const #cir.int<16> : !u64i
// CHECK:    %[[ALLOCATION_SIZE:.*]] = cir.const #cir.int<64> : !u64i
// CHECK:    %[[ALLOC_PTR:.*]] = cir.call @_Znam(%[[ALLOCATION_SIZE]]) : (!u64i) -> !cir.ptr<!void>
// CHECK:    %[[ELEM_0_PTR:.*]] = cir.cast(bitcast, %[[ALLOC_PTR]] : !cir.ptr<!void>), !cir.ptr<!s32i>
// CHECK:    %[[CONST_ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:    cir.store %[[CONST_ONE]], %[[ELEM_0_PTR]] : !s32i, !cir.ptr<!s32i>
// CHECK:    %[[OFFSET:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:    %[[ELEM_1_PTR:.*]] = cir.ptr_stride(%[[ELEM_0_PTR]] : !cir.ptr<!s32i>, %[[OFFSET]] : !s32i), !cir.ptr<!s32i>
// CHECK:    %[[CONST_TWO:.*]] = cir.const #cir.int<2> : !s32i
// CHECK:    cir.store %[[CONST_TWO]], %[[ELEM_1_PTR]] : !s32i, !cir.ptr<!s32i>
// CHECK:    %[[OFFSET1:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:    %[[ELEM_2_PTR:.*]] = cir.ptr_stride(%[[ELEM_1_PTR]] : !cir.ptr<!s32i>, %[[OFFSET1]] : !s32i), !cir.ptr<!s32i>
// CHECK:    %[[CONST_THREE:.*]] = cir.const #cir.int<3> : !s32i
// CHECK:    cir.store %[[CONST_THREE]], %[[ELEM_2_PTR]] : !s32i, !cir.ptr<!s32i>
// CHECK:    %[[OFFSET2:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:    %[[ELEM_3_PTR:.*]] = cir.ptr_stride(%[[ELEM_2_PTR]] : !cir.ptr<!s32i>, %[[OFFSET2]] : !s32i), !cir.ptr<!s32i>
// CHECK:    %[[INIT_SIZE:.*]] = cir.const #cir.int<12> : !u64i
// CHECK:    %[[REMAINING_SIZE:.*]] = cir.binop(sub, %[[ALLOCATION_SIZE]], %[[INIT_SIZE]]) : !u64i
// CHECK:    %[[VOID_PTR:.*]] = cir.cast(bitcast, %[[ELEM_3_PTR]] : !cir.ptr<!s32i>), !cir.ptr<!void>
// CHECK:    %[[ZERO:.*]] = cir.const #cir.int<0> : !u8i
// CHECK:    %[[ZERO_I32:.*]] = cir.cast(integral, %[[ZERO]] : !u8i), !s32i
// CHECK:    cir.libc.memset %[[REMAINING_SIZE]] bytes from %[[VOID_PTR]] set to %[[ZERO_I32]] : !cir.ptr<!void>, !s32i, !u64i

void t_new_var_size(size_t n) {
  auto p = new char[n];
}

// CHECK:  cir.func @_Z14t_new_var_sizem
// CHECK:    %[[N:.*]] = cir.load %[[ARG_ALLOCA:.*]]
// CHECK:    %[[PTR:.*]] = cir.call @_Znam(%[[N]]) : (!u64i)

void t_new_var_size2(int n) {
  auto p = new char[n];
}

// CHECK:  cir.func @_Z15t_new_var_size2i
// CHECK:    %[[N:.*]] = cir.load %[[ARG_ALLOCA:.*]]
// CHECK:    %[[N_SIZE_T:.*]] = cir.cast(integral, %[[N]] : !s32i), !u64i
// CHECK:    %[[PTR:.*]] = cir.call @_Znam(%[[N_SIZE_T]]) : (!u64i)

void t_new_var_size3(size_t n) {
  auto p = new double[n];
}

// CHECK:  cir.func @_Z15t_new_var_size3m
// CHECK:    %[[N:.*]] = cir.load %[[ARG_ALLOCA:.*]]
// CHECK:    %[[ELEMENT_SIZE:.*]] = cir.const #cir.int<8> : !u64i
// CHECK:    %[[RESULT:.*]], %[[OVERFLOW:.*]] = cir.binop.overflow(mul, %[[N]], %[[ELEMENT_SIZE]]) : !u64i, (!u64i, !cir.bool)
// CHECK:    %[[ALL_ONES:.*]] = cir.const #cir.int<18446744073709551615> : !u64i
// CHECK:    %[[ALLOC_SIZE:.*]] = cir.select if %[[OVERFLOW]] then %[[ALL_ONES]] else %[[RESULT]] : (!cir.bool, !u64i, !u64i)
// CHECK:    %[[PTR:.*]] = cir.call @_Znam(%[[ALLOC_SIZE]]) : (!u64i)

void t_new_var_size4(int n) {
  auto p = new double[n];
}

// CHECK:  cir.func @_Z15t_new_var_size4i
// CHECK:    %[[N:.*]] = cir.load %[[ARG_ALLOCA:.*]]
// CHECK:    %[[N_SIZE_T:.*]] = cir.cast(integral, %[[N]] : !s32i), !u64i
// CHECK:    %[[ELEMENT_SIZE:.*]] = cir.const #cir.int<8> : !u64i
// CHECK:    %[[RESULT:.*]], %[[OVERFLOW:.*]] = cir.binop.overflow(mul, %[[N_SIZE_T]], %[[ELEMENT_SIZE]]) : !u64i, (!u64i, !cir.bool)
// CHECK:    %[[ALL_ONES:.*]] = cir.const #cir.int<18446744073709551615> : !u64i
// CHECK:    %[[ALLOC_SIZE:.*]] = cir.select if %[[OVERFLOW]] then %[[ALL_ONES]] else %[[RESULT]] : (!cir.bool, !u64i, !u64i)
// CHECK:    %[[PTR:.*]] = cir.call @_Znam(%[[ALLOC_SIZE]]) : (!u64i)

void t_new_var_size5(int n) {
  auto p = new double[n][2][3];
}

// NUM_ELEMENTS isn't used in this case because there is no cookie.

// CHECK:  cir.func @_Z15t_new_var_size5i
// CHECK:    %[[N:.*]] = cir.load %[[ARG_ALLOCA:.*]]
// CHECK:    %[[N_SIZE_T:.*]] = cir.cast(integral, %[[N]] : !s32i), !u64i
// CHECK:    %[[ELEMENT_SIZE:.*]] = cir.const #cir.int<48> : !u64i
// CHECK:    %[[RESULT:.*]], %[[OVERFLOW:.*]] = cir.binop.overflow(mul, %[[N_SIZE_T]], %[[ELEMENT_SIZE]]) : !u64i, (!u64i, !cir.bool)
// CHECK:    %[[NUM_ELEMENTS_MULTIPLIER:.*]] = cir.const #cir.int<6>
// CHECK:    %[[NUM_ELEMENTS:.*]] = cir.binop(mul, %[[N_SIZE_T]], %[[NUM_ELEMENTS_MULTIPLIER]]) : !u64i
// CHECK:    %[[ALL_ONES:.*]] = cir.const #cir.int<18446744073709551615> : !u64i
// CHECK:    %[[ALLOC_SIZE:.*]] = cir.select if %[[OVERFLOW]] then %[[ALL_ONES]] else %[[RESULT]] : (!cir.bool, !u64i, !u64i)
// CHECK:    %[[PTR:.*]] = cir.call @_Znam(%[[ALLOC_SIZE]]) : (!u64i)

void t_new_var_size6(int n) {
  auto p = new double[n] { 1, 2, 3 };
}

// CHECK:  cir.func @_Z15t_new_var_size6i
// CHECK:    %[[N:.*]] = cir.load %[[ARG_ALLOCA:.*]]
// CHECK:    %[[N_SIZE_T:.*]] = cir.cast(integral, %[[N]] : !s32i), !u64i
// CHECK:    %[[MIN_SIZE:.*]] = cir.const #cir.int<3> : !u64i
// CHECK:    %[[LT_MIN_SIZE:.*]] = cir.cmp(lt, %[[N_SIZE_T]], %[[MIN_SIZE]]) : !u64i, !cir.bool
// CHECK:    %[[ELEMENT_SIZE:.*]] = cir.const #cir.int<8> : !u64i
// CHECK:    %[[RESULT:.*]], %[[OVERFLOW:.*]] = cir.binop.overflow(mul, %[[N_SIZE_T]], %[[ELEMENT_SIZE]]) : !u64i, (!u64i, !cir.bool)
// CHECK:    %[[ANY_OVERFLOW:.*]] = cir.binop(or, %[[LT_MIN_SIZE]], %[[OVERFLOW]]) : !cir.bool
// CHECK:    %[[ALL_ONES:.*]] = cir.const #cir.int<18446744073709551615> : !u64i
// CHECK:    %[[ALLOC_SIZE:.*]] = cir.select if %[[ANY_OVERFLOW]] then %[[ALL_ONES]] else %[[RESULT]] : (!cir.bool, !u64i, !u64i)
// CHECK:    %[[PTR:.*]] = cir.call @_Znam(%[[ALLOC_SIZE]]) : (!u64i)

void t_new_var_size7(__int128 n) {
  auto p = new double[n];
}

// CHECK:  cir.func @_Z15t_new_var_size7n
// CHECK:    %[[N:.*]] = cir.load %[[ARG_ALLOCA:.*]]
// CHECK:    %[[N_SIZE_T:.*]] = cir.cast(integral, %[[N]] : !s128i), !u64i
// CHECK:    %[[ELEMENT_SIZE:.*]] = cir.const #cir.int<8> : !u64i
// CHECK:    %[[RESULT:.*]], %[[OVERFLOW:.*]] = cir.binop.overflow(mul, %[[N_SIZE_T]], %[[ELEMENT_SIZE]]) : !u64i, (!u64i, !cir.bool)
// CHECK:    %[[ALL_ONES:.*]] = cir.const #cir.int<18446744073709551615> : !u64i
// CHECK:    %[[ALLOC_SIZE:.*]] = cir.select if %[[OVERFLOW]] then %[[ALL_ONES]] else %[[RESULT]] : (!cir.bool, !u64i, !u64i)
// CHECK:    %[[PTR:.*]] = cir.call @_Znam(%[[ALLOC_SIZE]]) : (!u64i)

void t_new_var_size_nontrivial(size_t n) {
  auto p = new D[n];
}

// CHECK:  cir.func @_Z25t_new_var_size_nontrivialm
// CHECK:    %[[N:.*]] = cir.load %[[ARG_ALLOCA:.*]]
// CHECK:    %[[ELEMENT_SIZE:.*]] = cir.const #cir.int<4> : !u64i
// CHECK:    %[[SIZE_WITHOUT_COOKIE:.*]], %[[OVERFLOW:.*]] = cir.binop.overflow(mul, %[[N]], %[[ELEMENT_SIZE]]) : !u64i, (!u64i, !cir.bool)
// CHECK:    %[[COOKIE_SIZE:.*]] = cir.const #cir.int<8> : !u64i
// CHECK:    %[[SIZE:.*]], %[[OVERFLOW2:.*]] = cir.binop.overflow(add, %[[SIZE_WITHOUT_COOKIE]], %[[COOKIE_SIZE]]) : !u64i, (!u64i, !cir.bool)
// CHECK:    %[[ANY_OVERFLOW:.*]] = cir.binop(or, %[[OVERFLOW]], %[[OVERFLOW2]]) : !cir.bool
// CHECK:    %[[ALL_ONES:.*]] = cir.const #cir.int<18446744073709551615> : !u64i
// CHECK:    %[[ALLOC_SIZE:.*]] = cir.select if %[[ANY_OVERFLOW]] then %[[ALL_ONES]] else %[[SIZE]] : (!cir.bool, !u64i, !u64i)
// CHECK:    %[[PTR:.*]] = cir.call @_Znam(%[[ALLOC_SIZE]]) : (!u64i)
