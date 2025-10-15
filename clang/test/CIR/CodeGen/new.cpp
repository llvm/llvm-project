// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct S {
  int a;
  int b;
};

void test_basic_new() {
  S *ps = new S;
  int *pn = new int;
  double *pd = new double;
}

// CHECK: cir.func{{.*}} @_Z14test_basic_newv
// CHECK:   %[[PS_ADDR:.*]] = cir.alloca !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>, ["ps", init]
// CHECK:   %[[PN_ADDR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["pn", init]
// CHECK:   %[[PD_ADDR:.*]] = cir.alloca !cir.ptr<!cir.double>, !cir.ptr<!cir.ptr<!cir.double>>, ["pd", init]
// CHECK:   %[[EIGHT:.*]] = cir.const #cir.int<8>
// CHECK:   %[[NEW_S:.*]] = cir.call @_Znwm(%[[EIGHT]])
// CHECK:   %[[NEW_S_PTR:.*]] = cir.cast bitcast %[[NEW_S]]
// CHECK:   cir.store{{.*}} %[[NEW_S_PTR]], %[[PS_ADDR]]
// CHECK:   %[[FOUR:.*]] = cir.const #cir.int<4>
// CHECK:   %[[NEW_INT:.*]] = cir.call @_Znwm(%[[FOUR]])
// CHECK:   %[[NEW_INT_PTR:.*]] = cir.cast bitcast %[[NEW_INT]]
// CHECK:   cir.store{{.*}} %[[NEW_INT_PTR]], %[[PN_ADDR]]
// CHECK:   %[[EIGHT:.*]] = cir.const #cir.int<8>
// CHECK:   %[[NEW_DOUBLE:.*]] = cir.call @_Znwm(%[[EIGHT]])
// CHECK:   %[[NEW_DOUBLE_PTR:.*]] = cir.cast bitcast %[[NEW_DOUBLE]]
// CHECK:   cir.store{{.*}} %[[NEW_DOUBLE_PTR]], %[[PD_ADDR]]
// CHECK:   cir.return

// LLVM: define{{.*}} void @_Z14test_basic_newv
// LLVM:   %[[PS_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[PN_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[PD_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[NEW_S:.*]] = call{{.*}} ptr @_Znwm(i64 8)
// LLVM:   store ptr %[[NEW_S]], ptr %[[PS_ADDR]], align 8
// LLVM:   %[[NEW_INT:.*]] = call{{.*}} ptr @_Znwm(i64 4)
// LLVM:   store ptr %[[NEW_INT]], ptr %[[PN_ADDR]], align 8
// LLVM:   %[[NEW_DOUBLE:.*]] = call{{.*}} ptr @_Znwm(i64 8)
// LLVM:   store ptr %[[NEW_DOUBLE]], ptr %[[PD_ADDR]], align 8
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z14test_basic_newv
// OGCG:   %[[PS_ADDR:.*]] = alloca ptr, align 8
// OGCG:   %[[PN_ADDR:.*]] = alloca ptr, align 8
// OGCG:   %[[PD_ADDR:.*]] = alloca ptr, align 8
// OGCG:   %[[NEW_S:.*]] = call{{.*}} ptr @_Znwm(i64 noundef 8)
// OGCG:   store ptr %[[NEW_S]], ptr %[[PS_ADDR]], align 8
// OGCG:   %[[NEW_INT:.*]] = call{{.*}} ptr @_Znwm(i64 noundef 4)
// OGCG:   store ptr %[[NEW_INT]], ptr %[[PN_ADDR]], align 8
// OGCG:   %[[NEW_DOUBLE:.*]] = call{{.*}} ptr @_Znwm(i64 noundef 8)
// OGCG:   store ptr %[[NEW_DOUBLE]], ptr %[[PD_ADDR]], align 8
// OGCG:   ret void

void test_new_with_init() {
  int *pn = new int{2};
  double *pd = new double{3.0};
}

// CHECK: cir.func{{.*}} @_Z18test_new_with_initv
// CHECK:   %[[PN_ADDR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["pn", init]
// CHECK:   %[[PD_ADDR:.*]] = cir.alloca !cir.ptr<!cir.double>, !cir.ptr<!cir.ptr<!cir.double>>, ["pd", init]
// CHECK:   %[[FOUR:.*]] = cir.const #cir.int<4>
// CHECK:   %[[NEW_INT:.*]] = cir.call @_Znwm(%[[FOUR]])
// CHECK:   %[[NEW_INT_PTR:.*]] = cir.cast bitcast %[[NEW_INT]]
// CHECK:   %[[TWO:.*]] = cir.const #cir.int<2>
// CHECK:   cir.store{{.*}} %[[TWO]], %[[NEW_INT_PTR]]
// CHECK:   cir.store{{.*}} %[[NEW_INT_PTR]], %[[PN_ADDR]]
// CHECK:   %[[EIGHT:.*]] = cir.const #cir.int<8>
// CHECK:   %[[NEW_DOUBLE:.*]] = cir.call @_Znwm(%[[EIGHT]])
// CHECK:   %[[NEW_DOUBLE_PTR:.*]] = cir.cast bitcast %[[NEW_DOUBLE]]
// CHECK:   %[[THREE:.*]] = cir.const #cir.fp<3.000000e+00>
// CHECK:   cir.store{{.*}} %[[THREE]], %[[NEW_DOUBLE_PTR]]
// CHECK:   cir.store{{.*}} %[[NEW_DOUBLE_PTR]], %[[PD_ADDR]]
// CHECK:   cir.return

// LLVM: define{{.*}} void @_Z18test_new_with_initv
// LLVM:   %[[PN_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[PD_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[NEW_INT:.*]] = call{{.*}} ptr @_Znwm(i64 4)
// LLVM:   store i32 2, ptr %[[NEW_INT]], align 4
// LLVM:   store ptr %[[NEW_INT]], ptr %[[PN_ADDR]], align 8
// LLVM:   %[[NEW_DOUBLE:.*]] = call{{.*}} ptr @_Znwm(i64 8)
// LLVM:   store double 3.000000e+00, ptr %[[NEW_DOUBLE]], align 8
// LLVM:   store ptr %[[NEW_DOUBLE]], ptr %[[PD_ADDR]], align 8
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z18test_new_with_initv
// OGCG:   %[[PN_ADDR:.*]] = alloca ptr, align 8
// OGCG:   %[[PD_ADDR:.*]] = alloca ptr, align 8
// OGCG:   %[[NEW_INT:.*]] = call{{.*}} ptr @_Znwm(i64 noundef 4)
// OGCG:   store i32 2, ptr %[[NEW_INT]], align 4
// OGCG:   store ptr %[[NEW_INT]], ptr %[[PN_ADDR]], align 8
// OGCG:   %[[NEW_DOUBLE:.*]] = call{{.*}} ptr @_Znwm(i64 noundef 8)
// OGCG:   store double 3.000000e+00, ptr %[[NEW_DOUBLE]], align 8
// OGCG:   store ptr %[[NEW_DOUBLE]], ptr %[[PD_ADDR]], align 8
// OGCG:   ret void

struct S2 {
  S2();
  S2(int, int);
  int a;
  int b;
};

void test_new_with_ctor() {
  S2 *ps2 = new S2;
  S2 *ps2_2 = new S2(1, 2);
}

// CHECK: cir.func{{.*}} @_Z18test_new_with_ctorv
// CHECK:   %[[PS2_ADDR:.*]] = cir.alloca !cir.ptr<!rec_S2>, !cir.ptr<!cir.ptr<!rec_S2>>, ["ps2", init]
// CHECK:   %[[PS2_2_ADDR:.*]] = cir.alloca !cir.ptr<!rec_S2>, !cir.ptr<!cir.ptr<!rec_S2>>, ["ps2_2", init]
// CHECK:   %[[EIGHT:.*]] = cir.const #cir.int<8>
// CHECK:   %[[NEW_S2:.*]] = cir.call @_Znwm(%[[EIGHT]])
// CHECK:   %[[NEW_S2_PTR:.*]] = cir.cast bitcast %[[NEW_S2]]
// CHECK:   cir.call @_ZN2S2C1Ev(%[[NEW_S2_PTR]])
// CHECK:   cir.store{{.*}} %[[NEW_S2_PTR]], %[[PS2_ADDR]]
// CHECK:   %[[EIGHT:.*]] = cir.const #cir.int<8>
// CHECK:   %[[NEW_S2_2:.*]] = cir.call @_Znwm(%[[EIGHT]])
// CHECK:   %[[NEW_S2_2_PTR:.*]] = cir.cast bitcast %[[NEW_S2_2]]
// CHECK:   %[[ONE:.*]] = cir.const #cir.int<1>
// CHECK:   %[[TWO:.*]] = cir.const #cir.int<2>
// CHECK:   cir.call @_ZN2S2C1Eii(%[[NEW_S2_2_PTR]], %[[ONE]], %[[TWO]])
// CHECK:   cir.store{{.*}} %[[NEW_S2_2_PTR]], %[[PS2_2_ADDR]]
// CHECK:   cir.return

// LLVM: define{{.*}} void @_Z18test_new_with_ctorv
// LLVM:   %[[PS2_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[PS2_2_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[NEW_S2:.*]] = call{{.*}} ptr @_Znwm(i64 8)
// LLVM:   call{{.*}} void @_ZN2S2C1Ev(ptr %[[NEW_S2]])
// LLVM:   store ptr %[[NEW_S2]], ptr %[[PS2_ADDR]], align 8
// LLVM:   %[[NEW_S2_2:.*]] = call{{.*}} ptr @_Znwm(i64 8)
// LLVM:   call{{.*}} void @_ZN2S2C1Eii(ptr %[[NEW_S2_2]], i32 1, i32 2)
// LLVM:   store ptr %[[NEW_S2_2]], ptr %[[PS2_2_ADDR]], align 8
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z18test_new_with_ctorv
// OGCG:   %[[PS2_ADDR:.*]] = alloca ptr, align 8
// OGCG:   %[[PS2_2_ADDR:.*]] = alloca ptr, align 8
// OGCG:   %[[NEW_S2:.*]] = call{{.*}} ptr @_Znwm(i64 noundef 8)
// OGCG:   call{{.*}} void @_ZN2S2C1Ev(ptr {{.*}} %[[NEW_S2]])
// OGCG:   store ptr %[[NEW_S2]], ptr %[[PS2_ADDR]], align 8
// OGCG:   %[[NEW_S2_2:.*]] = call{{.*}} ptr @_Znwm(i64 noundef 8)
// OGCG:   call{{.*}} void @_ZN2S2C1Eii(ptr {{.*}} %[[NEW_S2_2]], i32 noundef 1, i32 noundef 2)
// OGCG:   store ptr %[[NEW_S2_2]], ptr %[[PS2_2_ADDR]], align 8
// OGCG:   ret void

void test_new_with_complex_type() {
  _Complex float *a = new _Complex float{1.0f, 2.0f};
}

// CHECK: cir.func{{.*}} @_Z26test_new_with_complex_typev
// CHECK:   %[[A_ADDR:.*]] = cir.alloca !cir.ptr<!cir.complex<!cir.float>>, !cir.ptr<!cir.ptr<!cir.complex<!cir.float>>>, ["a", init]
// CHECK:   %[[COMPLEX_SIZE:.*]] = cir.const #cir.int<8> : !u64i
// CHECK:   %[[NEW_COMPLEX:.*]] = cir.call @_Znwm(%[[COMPLEX_SIZE]]) : (!u64i) -> !cir.ptr<!void>
// CHECK:   %[[COMPLEX_PTR:.*]] = cir.cast bitcast %[[NEW_COMPLEX]] : !cir.ptr<!void> -> !cir.ptr<!cir.complex<!cir.float>>
// CHECK:   %[[COMPLEX_VAL:.*]] = cir.const #cir.const_complex<#cir.fp<1.000000e+00> : !cir.float, #cir.fp<2.000000e+00> : !cir.float> : !cir.complex<!cir.float>
// CHECK:   cir.store{{.*}} %[[COMPLEX_VAL]], %[[COMPLEX_PTR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CHECK:   cir.store{{.*}} %[[COMPLEX_PTR]], %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.ptr<!cir.ptr<!cir.complex<!cir.float>>>

// LLVM: define{{.*}} void @_Z26test_new_with_complex_typev
// LLVM:   %[[A_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[NEW_COMPLEX:.*]] = call ptr @_Znwm(i64 8)
// LLVM:   store { float, float } { float 1.000000e+00, float 2.000000e+00 }, ptr %[[NEW_COMPLEX]], align 8
// LLVM:   store ptr %[[NEW_COMPLEX]], ptr %[[A_ADDR]], align 8

// OGCG: define{{.*}} void @_Z26test_new_with_complex_typev
// OGCG:   %[[A_ADDR:.*]] = alloca ptr, align 8
// OGCG:   %[[NEW_COMPLEX:.*]] = call noalias noundef nonnull ptr @_Znwm(i64 noundef 8)
// OGCG:   %[[COMPLEX_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[NEW_COMPLEX]], i32 0, i32 0
// OGCG:   %[[COMPLEX_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[NEW_COMPLEX]], i32 0, i32 1
// OGCG:   store float 1.000000e+00, ptr %[[COMPLEX_REAL_PTR]], align 8
// OGCG:   store float 2.000000e+00, ptr %[[COMPLEX_IMAG_PTR]], align 4
// OGCG:   store ptr %[[NEW_COMPLEX]], ptr %[[A_ADDR]], align 8

void t_new_constant_size() {
  auto p = new double[16];
}

// In this test, NUM_ELEMENTS isn't used because no cookie is needed and there
//   are no constructor calls needed.

// CHECK:   cir.func{{.*}} @_Z19t_new_constant_sizev()
// CHECK:    %[[P_ADDR:.*]] = cir.alloca !cir.ptr<!cir.double>, !cir.ptr<!cir.ptr<!cir.double>>, ["p", init] {alignment = 8 : i64}
// CHECK:    %[[#NUM_ELEMENTS:]] = cir.const #cir.int<16> : !u64i
// CHECK:    %[[#ALLOCATION_SIZE:]] = cir.const #cir.int<128> : !u64i
// CHECK:    %[[RAW_PTR:.*]] = cir.call @_Znam(%[[#ALLOCATION_SIZE]]) : (!u64i) -> !cir.ptr<!void>
// CHECK:    %[[TYPED_PTR:.*]] = cir.cast bitcast %[[RAW_PTR]] : !cir.ptr<!void> -> !cir.ptr<!cir.double>
// CHECK:    cir.store align(8) %[[TYPED_PTR]], %[[P_ADDR]] : !cir.ptr<!cir.double>, !cir.ptr<!cir.ptr<!cir.double>>
// CHECK:    cir.return
// CHECK:  }

// LLVM: define{{.*}} void @_Z19t_new_constant_sizev
// LLVM:   %[[P_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[CALL:.*]] = call ptr @_Znam(i64 128)
// LLVM:   store ptr %[[CALL]], ptr %[[P_ADDR]], align 8

// OGCG: define{{.*}} void @_Z19t_new_constant_sizev
// OGCG:   %[[P_ADDR:.*]] = alloca ptr, align 8
// OGCG:   %[[CALL:.*]] = call noalias noundef nonnull ptr @_Znam(i64 noundef 128)
// OGCG:   store ptr %[[CALL]], ptr %[[P_ADDR]], align 8


void t_new_multidim_constant_size() {
  auto p = new double[2][3][4];
}

// As above, NUM_ELEMENTS isn't used.

// CHECK:   cir.func{{.*}} @_Z28t_new_multidim_constant_sizev()
// CHECK:    %[[P_ADDR:.*]] = cir.alloca !cir.ptr<!cir.array<!cir.array<!cir.double x 4> x 3>>, !cir.ptr<!cir.ptr<!cir.array<!cir.array<!cir.double x 4> x 3>>>, ["p", init] {alignment = 8 : i64}
// CHECK:    %[[#NUM_ELEMENTS:]] = cir.const #cir.int<24> : !u64i
// CHECK:    %[[#ALLOCATION_SIZE:]] = cir.const #cir.int<192> : !u64i
// CHECK:    %[[RAW_PTR:.*]] = cir.call @_Znam(%[[#ALLOCATION_SIZE]]) : (!u64i) -> !cir.ptr<!void>
// CHECK:    %[[TYPED_PTR:.*]] = cir.cast bitcast %[[RAW_PTR]] : !cir.ptr<!void> -> !cir.ptr<!cir.array<!cir.array<!cir.double x 4> x 3>>
// CHECK:    cir.store align(8) %[[TYPED_PTR]], %[[P_ADDR]] : !cir.ptr<!cir.array<!cir.array<!cir.double x 4> x 3>>, !cir.ptr<!cir.ptr<!cir.array<!cir.array<!cir.double x 4> x 3>>>
// CHECK:  }

// LLVM: define{{.*}} void @_Z28t_new_multidim_constant_sizev
// LLVM:   %[[P_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[CALL:.*]] = call ptr @_Znam(i64 192)
// LLVM:   store ptr %[[CALL]], ptr %[[P_ADDR]], align 8

// OGCG: define{{.*}} void @_Z28t_new_multidim_constant_sizev
// OGCG:   %[[P_ADDR:.*]] = alloca ptr, align 8
// OGCG:   %[[CALL:.*]] = call noalias noundef nonnull ptr @_Znam(i64 noundef 192)
// OGCG:   store ptr %[[CALL]], ptr %[[P_ADDR]], align 8
