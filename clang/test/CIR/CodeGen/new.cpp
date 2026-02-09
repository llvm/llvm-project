// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

typedef __typeof__(sizeof(int)) size_t;

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
// CHECK:   %[[NEW_COMPLEX:.*]] = cir.call @_Znwm(%[[COMPLEX_SIZE]]) {allocsize = array<i32: 0>} : (!u64i) -> (!cir.ptr<!void> {llvm.noundef})
// CHECK:   %[[COMPLEX_PTR:.*]] = cir.cast bitcast %[[NEW_COMPLEX]] : !cir.ptr<!void> -> !cir.ptr<!cir.complex<!cir.float>>
// CHECK:   %[[COMPLEX_VAL:.*]] = cir.const #cir.const_complex<#cir.fp<1.000000e+00> : !cir.float, #cir.fp<2.000000e+00> : !cir.float> : !cir.complex<!cir.float>
// CHECK:   cir.store{{.*}} %[[COMPLEX_VAL]], %[[COMPLEX_PTR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CHECK:   cir.store{{.*}} %[[COMPLEX_PTR]], %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.ptr<!cir.ptr<!cir.complex<!cir.float>>>

// LLVM: define{{.*}} void @_Z26test_new_with_complex_typev
// LLVM:   %[[A_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[NEW_COMPLEX:.*]] = call noundef ptr @_Znwm(i64 8)
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

// CHECK:   cir.func{{.*}} @_Z19t_new_constant_sizev()
// CHECK:    %[[P_ADDR:.*]] = cir.alloca !cir.ptr<!cir.double>, !cir.ptr<!cir.ptr<!cir.double>>, ["p", init] {alignment = 8 : i64}
// CHECK:    %[[ALLOCATION_SIZE:.*]] = cir.const #cir.int<128> : !u64i
// CHECK:    %[[RAW_PTR:.*]] = cir.call @_Znam(%[[ALLOCATION_SIZE]]) {allocsize = array<i32: 0>} : (!u64i) -> (!cir.ptr<!void> {llvm.noundef})
// CHECK:    %[[TYPED_PTR:.*]] = cir.cast bitcast %[[RAW_PTR]] : !cir.ptr<!void> -> !cir.ptr<!cir.double>
// CHECK:    cir.store align(8) %[[TYPED_PTR]], %[[P_ADDR]] : !cir.ptr<!cir.double>, !cir.ptr<!cir.ptr<!cir.double>>
// CHECK:    cir.return
// CHECK:  }

// LLVM: define{{.*}} void @_Z19t_new_constant_sizev
// LLVM:   %[[P_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[CALL:.*]] = call noundef ptr @_Znam(i64 128)
// LLVM:   store ptr %[[CALL]], ptr %[[P_ADDR]], align 8

// OGCG: define{{.*}} void @_Z19t_new_constant_sizev
// OGCG:   %[[P_ADDR:.*]] = alloca ptr, align 8
// OGCG:   %[[CALL:.*]] = call noalias noundef nonnull ptr @_Znam(i64 noundef 128)
// OGCG:   store ptr %[[CALL]], ptr %[[P_ADDR]], align 8

class C {
  public:
    ~C();
};

void t_constant_size_nontrivial() {
  auto p = new C[3];
}

// CHECK:  cir.func{{.*}} @_Z26t_constant_size_nontrivialv()
// CHECK:    %[[P_ADDR:.*]] = cir.alloca !cir.ptr<!rec_C>, !cir.ptr<!cir.ptr<!rec_C>>, ["p", init] {alignment = 8 : i64}
// CHECK:    %[[NUM_ELEMENTS:.*]] = cir.const #cir.int<3> : !u64i
// CHECK:    %[[ALLOCATION_SIZE:.*]] = cir.const #cir.int<11> : !u64i
// CHECK:    %[[RAW_PTR:.*]] = cir.call @_Znam(%[[ALLOCATION_SIZE]]) {allocsize = array<i32: 0>} : (!u64i) -> (!cir.ptr<!void> {llvm.noundef})
// CHECK:    %[[COOKIE_PTR_BASE:.*]] = cir.cast bitcast %[[RAW_PTR]] : !cir.ptr<!void> -> !cir.ptr<!cir.ptr<!u8i>>
// CHECK:    %[[COOKIE_PTR:.*]] = cir.cast bitcast %[[COOKIE_PTR_BASE]] : !cir.ptr<!cir.ptr<!u8i>> -> !cir.ptr<!u64i>
// CHECK:    cir.store align(8) %[[NUM_ELEMENTS]], %[[COOKIE_PTR]] : !u64i, !cir.ptr<!u64i>
// CHECK:    %[[COOKIE_SIZE:.*]] = cir.const #cir.int<8> : !s32i
// CHECK:    %[[DATA_PTR_RAW:.*]] = cir.ptr_stride %[[COOKIE_PTR_BASE]], %[[COOKIE_SIZE]] : (!cir.ptr<!cir.ptr<!u8i>>, !s32i) -> !cir.ptr<!cir.ptr<!u8i>>
// CHECK:    %[[DATA_PTR_VOID:.*]] = cir.cast bitcast %[[DATA_PTR_RAW]] : !cir.ptr<!cir.ptr<!u8i>> -> !cir.ptr<!void>
// CHECK:    %[[DATA_PTR:.*]] = cir.cast bitcast %[[DATA_PTR_VOID]] : !cir.ptr<!void> -> !cir.ptr<!rec_C>
// CHECK:    cir.store align(8) %[[DATA_PTR]], %[[P_ADDR]] : !cir.ptr<!rec_C>, !cir.ptr<!cir.ptr<!rec_C>>
// CHECK:    cir.return
// CHECK:  }

// LLVM: @_Z26t_constant_size_nontrivialv()
// LLVM:   %[[ALLOCA:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[COOKIE_PTR:.*]] = call noundef ptr @_Znam(i64 11)
// LLVM:   store i64 3, ptr %[[COOKIE_PTR]], align 8
// LLVM:   %[[ALLOCATED_PTR:.*]] = getelementptr ptr, ptr %[[COOKIE_PTR]], i64 8
// LLVM:   store ptr %[[ALLOCATED_PTR]], ptr %[[ALLOCA]], align 8

// OGCG: @_Z26t_constant_size_nontrivialv()
// OGCG:   %[[ALLOCA:.*]] = alloca ptr, align 8
// OGCG:   %[[COOKIE_PTR:.*]] = call noalias noundef nonnull ptr @_Znam(i64 noundef 11)
// OGCG:   store i64 3, ptr %[[COOKIE_PTR]], align 8
// OGCG:   %[[ALLOCATED_PTR:.*]] = getelementptr inbounds i8, ptr %[[COOKIE_PTR]], i64 8
// OGCG:   store ptr %[[ALLOCATED_PTR]], ptr %[[ALLOCA]], align 8

class D {
  public:
    int x;
    ~D();
};

void t_constant_size_nontrivial2() {
  auto p = new D[3];
}

// CHECK:  cir.func{{.*}} @_Z27t_constant_size_nontrivial2v()
// CHECK:    %[[P_ADDR:.*]] = cir.alloca !cir.ptr<!rec_D>, !cir.ptr<!cir.ptr<!rec_D>>, ["p", init] {alignment = 8 : i64}
// CHECK:    %[[NUM_ELEMENTS:.*]] = cir.const #cir.int<3> : !u64i
// CHECK:    %[[ALLOCATION_SIZE:.*]] = cir.const #cir.int<20> : !u64i
// CHECK:    %[[RAW_PTR:.*]] = cir.call @_Znam(%[[ALLOCATION_SIZE]]) {allocsize = array<i32: 0>} : (!u64i) -> (!cir.ptr<!void> {llvm.noundef})
// CHECK:    %[[COOKIE_PTR_BASE:.*]] = cir.cast bitcast %[[RAW_PTR]] : !cir.ptr<!void> -> !cir.ptr<!cir.ptr<!u8i>>
// CHECK:    %[[COOKIE_PTR:.*]] = cir.cast bitcast %[[COOKIE_PTR_BASE]] : !cir.ptr<!cir.ptr<!u8i>> -> !cir.ptr<!u64i>
// CHECK:    cir.store align(8) %[[NUM_ELEMENTS]], %[[COOKIE_PTR]] : !u64i, !cir.ptr<!u64i>
// CHECK:    %[[COOKIE_SIZE:.*]] = cir.const #cir.int<8> : !s32i
// CHECK:    %[[DATA_PTR_RAW:.*]] = cir.ptr_stride %[[COOKIE_PTR_BASE]], %[[COOKIE_SIZE]] : (!cir.ptr<!cir.ptr<!u8i>>, !s32i) -> !cir.ptr<!cir.ptr<!u8i>>
// CHECK:    %[[DATA_PTR_VOID:.*]] = cir.cast bitcast %[[DATA_PTR_RAW]] : !cir.ptr<!cir.ptr<!u8i>> -> !cir.ptr<!void>
// CHECK:    %[[DATA_PTR:.*]] = cir.cast bitcast %[[DATA_PTR_VOID]] : !cir.ptr<!void> -> !cir.ptr<!rec_D>
// CHECK:    cir.store align(8) %[[DATA_PTR]], %[[P_ADDR]] : !cir.ptr<!rec_D>, !cir.ptr<!cir.ptr<!rec_D>>
// CHECK:    cir.return
// CHECK:  }

// LLVM: @_Z27t_constant_size_nontrivial2v()
// LLVM:   %[[ALLOCA:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[COOKIE_PTR:.*]] = call noundef ptr @_Znam(i64 20)
// LLVM:   store i64 3, ptr %[[COOKIE_PTR]], align 8
// LLVM:   %[[ALLOCATED_PTR:.*]] = getelementptr ptr, ptr %[[COOKIE_PTR]], i64 8
// LLVM:   store ptr %[[ALLOCATED_PTR]], ptr %[[ALLOCA]], align 8

struct alignas(16) E {
  int x;
  ~E();
};

void t_align16_nontrivial() {
  auto p = new E[2];
}

// CHECK:  cir.func{{.*}} @_Z20t_align16_nontrivialv()
// CHECK:    %[[P_ADDR:.*]] = cir.alloca !cir.ptr<!rec_E>, !cir.ptr<!cir.ptr<!rec_E>>, ["p", init] {alignment = 8 : i64}
// CHECK:    %[[NUM_ELEMENTS:.*]] = cir.const #cir.int<2> : !u64i
// CHECK:    %[[ALLOCATION_SIZE:.*]] = cir.const #cir.int<48> : !u64i
// CHECK:    %[[RAW_PTR:.*]] = cir.call @_Znam(%[[ALLOCATION_SIZE]]) {allocsize = array<i32: 0>} : (!u64i) -> (!cir.ptr<!void> {llvm.noundef})
// CHECK:    %[[COOKIE_PTR_BASE:.*]] = cir.cast bitcast %[[RAW_PTR]] : !cir.ptr<!void> -> !cir.ptr<!cir.ptr<!u8i>>
// CHECK:    %[[COOKIE_OFFSET:.*]] = cir.const #cir.int<8> : !s32i
// CHECK:    %[[COOKIE_PTR_RAW:.*]] = cir.ptr_stride %[[COOKIE_PTR_BASE]], %[[COOKIE_OFFSET]] : (!cir.ptr<!cir.ptr<!u8i>>, !s32i) -> !cir.ptr<!cir.ptr<!u8i>>
// CHECK:    %[[COOKIE_PTR:.*]] = cir.cast bitcast %[[COOKIE_PTR_RAW]] : !cir.ptr<!cir.ptr<!u8i>> -> !cir.ptr<!u64i>
// CHECK:    cir.store align(8) %[[NUM_ELEMENTS]], %[[COOKIE_PTR]] : !u64i, !cir.ptr<!u64i>
// CHECK:    %[[COOKIE_SIZE:.*]] = cir.const #cir.int<16> : !s32i
// CHECK:    %[[DATA_PTR_RAW:.*]] = cir.ptr_stride %[[COOKIE_PTR_BASE]], %[[COOKIE_SIZE]] : (!cir.ptr<!cir.ptr<!u8i>>, !s32i) -> !cir.ptr<!cir.ptr<!u8i>>
// CHECK:    %[[DATA_PTR_VOID:.*]] = cir.cast bitcast %[[DATA_PTR_RAW]] : !cir.ptr<!cir.ptr<!u8i>> -> !cir.ptr<!void>
// CHECK:    %[[DATA_PTR:.*]] = cir.cast bitcast %[[DATA_PTR_VOID]] : !cir.ptr<!void> -> !cir.ptr<!rec_E>
// CHECK:    cir.store align(8) %[[DATA_PTR]], %[[P_ADDR]] : !cir.ptr<!rec_E>, !cir.ptr<!cir.ptr<!rec_E>>
// CHECK:    cir.return
// CHECK:  }

// LLVM: @_Z20t_align16_nontrivialv()
// LLVM:   %[[ALLOCA:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[RAW_PTR:.*]] = call noundef ptr @_Znam(i64 48)
// LLVM:   %[[COOKIE_PTR:.*]] = getelementptr ptr, ptr %[[RAW_PTR]], i64 8
// LLVM:   store i64 2, ptr %[[COOKIE_PTR]], align 8
// LLVM:   %[[ALLOCATED_PTR:.*]] = getelementptr ptr, ptr %[[RAW_PTR]], i64 16
// LLVM:   store ptr %[[ALLOCATED_PTR]], ptr %[[ALLOCA]], align 8

// OGCG: define{{.*}} void @_Z20t_align16_nontrivialv
// OGCG:   %[[ALLOCA:.*]] = alloca ptr, align 8
// OGCG:   %[[RAW_PTR:.*]] = call noalias noundef nonnull ptr @_Znam(i64 noundef 48)
// OGCG:   %[[COOKIE_PTR:.*]] = getelementptr inbounds i8, ptr %[[RAW_PTR]], i64 8
// OGCG:   store i64 2, ptr %[[COOKIE_PTR]], align 8
// OGCG:   %[[ALLOCATED_PTR:.*]] = getelementptr inbounds i8, ptr %[[RAW_PTR]], i64 16
// OGCG:   store ptr %[[ALLOCATED_PTR]], ptr %[[ALLOCA]], align 8
// OGCG:   ret void

void t_new_multidim_constant_size() {
  auto p = new double[2][3][4];
}

// CHECK:  cir.func{{.*}} @_Z28t_new_multidim_constant_sizev()
// CHECK:    %[[P_ADDR:.*]] = cir.alloca !cir.ptr<!cir.array<!cir.array<!cir.double x 4> x 3>>, !cir.ptr<!cir.ptr<!cir.array<!cir.array<!cir.double x 4> x 3>>>, ["p", init] {alignment = 8 : i64}
// CHECK:    %[[ALLOCATION_SIZE:.*]] = cir.const #cir.int<192> : !u64i
// CHECK:    %[[RAW_PTR:.*]] = cir.call @_Znam(%[[ALLOCATION_SIZE]]) {allocsize = array<i32: 0>} : (!u64i) -> (!cir.ptr<!void> {llvm.noundef})
// CHECK:    %[[TYPED_PTR:.*]] = cir.cast bitcast %[[RAW_PTR]] : !cir.ptr<!void> -> !cir.ptr<!cir.array<!cir.array<!cir.double x 4> x 3>>
// CHECK:    cir.store align(8) %[[TYPED_PTR]], %[[P_ADDR]] : !cir.ptr<!cir.array<!cir.array<!cir.double x 4> x 3>>, !cir.ptr<!cir.ptr<!cir.array<!cir.array<!cir.double x 4> x 3>>>
// CHECK:  }

// LLVM: define{{.*}} void @_Z28t_new_multidim_constant_sizev
// LLVM:   %[[P_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[CALL:.*]] = call noundef ptr @_Znam(i64 192)
// LLVM:   store ptr %[[CALL]], ptr %[[P_ADDR]], align 8

// OGCG: define{{.*}} void @_Z28t_new_multidim_constant_sizev
// OGCG:   %[[P_ADDR:.*]] = alloca ptr, align 8
// OGCG:   %[[CALL:.*]] = call noalias noundef nonnull ptr @_Znam(i64 noundef 192)
// OGCG:   store ptr %[[CALL]], ptr %[[P_ADDR]], align 8

void t_constant_size_memset_init() {
  auto p = new int[16] {};
}

// CHECK:  cir.func {{.*}} @_Z27t_constant_size_memset_initv()
// CHECK:    %[[ALLOCATION_SIZE:.*]] = cir.const #cir.int<64> : !u64i
// CHECK:    %[[ALLOC_PTR:.*]] = cir.call @_Znam(%[[ALLOCATION_SIZE]]) {allocsize = array<i32: 0>} : (!u64i) -> (!cir.ptr<!void> {llvm.noundef})
// CHECK:    %[[ELEM_PTR:.*]] = cir.cast bitcast %[[ALLOC_PTR]] : !cir.ptr<!void> -> !cir.ptr<!s32i>
// CHECK:    %[[VOID_PTR:.*]] = cir.cast bitcast %[[ELEM_PTR]] : !cir.ptr<!s32i> -> !cir.ptr<!void>
// CHECK:    %[[ZERO:.*]] = cir.const #cir.int<0> : !u8i
// CHECK:    cir.libc.memset %[[ALLOCATION_SIZE]] bytes at %[[VOID_PTR]] to %[[ZERO]] : !cir.ptr<!void>, !u8i, !u64i

// LLVM: define {{.*}} void @_Z27t_constant_size_memset_initv()
// LLVM:   %[[P:.*]] = call noundef ptr @_Znam(i64 64)
// LLVM:   call void @llvm.memset.p0.i64(ptr %[[P]], i8 0, i64 64, i1 false)

// OGCG: define {{.*}} void @_Z27t_constant_size_memset_initv()
// OGCG:   %[[P:.*]] = call{{.*}} ptr @_Znam(i64{{.*}} 64)
// OGCG:   call void @llvm.memset.p0.i64(ptr{{.*}} %[[P]], i8 0, i64 64, i1 false)

void t_constant_size_full_init() {
  auto p = new int[4] { 1, 2, 3, 4 };
}

// CHECK:  cir.func {{.*}} @_Z25t_constant_size_full_initv()
// CHECK:    %[[ALLOCATION_SIZE:.*]] = cir.const #cir.int<16> : !u64i
// CHECK:    %[[ALLOC_PTR:.*]] = cir.call @_Znam(%[[ALLOCATION_SIZE]]) {allocsize = array<i32: 0>} : (!u64i) -> (!cir.ptr<!void> {llvm.noundef})
// CHECK:    %[[ELEM_0_PTR:.*]] = cir.cast bitcast %[[ALLOC_PTR]] : !cir.ptr<!void> -> !cir.ptr<!s32i>
// CHECK:    %[[CONST_ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:    cir.store{{.*}} %[[CONST_ONE]], %[[ELEM_0_PTR]] : !s32i, !cir.ptr<!s32i>
// CHECK:    %[[OFFSET:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:    %[[ELEM_1_PTR:.*]] = cir.ptr_stride %[[ELEM_0_PTR]], %[[OFFSET]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
// CHECK:    %[[CONST_TWO:.*]] = cir.const #cir.int<2> : !s32i
// CHECK:    cir.store{{.*}} %[[CONST_TWO]], %[[ELEM_1_PTR]] : !s32i, !cir.ptr<!s32i>
// CHECK:    %[[OFFSET1:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:    %[[ELEM_2_PTR:.*]] = cir.ptr_stride %[[ELEM_1_PTR]], %[[OFFSET1]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
// CHECK:    %[[CONST_THREE:.*]] = cir.const #cir.int<3> : !s32i
// CHECK:    cir.store{{.*}} %[[CONST_THREE]], %[[ELEM_2_PTR]] : !s32i, !cir.ptr<!s32i>
// CHECK:    %[[OFFSET2:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:    %[[ELEM_3_PTR:.*]] = cir.ptr_stride %[[ELEM_2_PTR]], %[[OFFSET2]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
// CHECK:    %[[CONST_FOUR:.*]] = cir.const #cir.int<4> : !s32i
// CHECK:    cir.store{{.*}} %[[CONST_FOUR]], %[[ELEM_3_PTR]] : !s32i, !cir.ptr<!s32i>

// LLVM: define {{.*}} void @_Z25t_constant_size_full_initv()
// LLVM:   %[[P:.*]] = call noundef ptr @_Znam(i64 16)
// LLVM:   store i32 1, ptr %[[CALL]]
// LLVM:   %[[ELEM_1:.*]] = getelementptr i32, ptr %[[P]], i64 1
// LLVM:   store i32 2, ptr %[[ELEM_1]]
// LLVM:   %[[ELEM_2:.*]] = getelementptr i32, ptr %[[ELEM_1]], i64 1
// LLVM:   store i32 3, ptr %[[ELEM_2]]
// LLVM:   %[[ELEM_3:.*]] = getelementptr i32, ptr %[[ELEM_2]], i64 1
// LLVM:   store i32 4, ptr %[[ELEM_3]]

// OGCG: define {{.*}} void @_Z25t_constant_size_full_initv()
// OGCG:   %[[P:.*]] = call{{.*}} ptr @_Znam(i64{{.*}} 16)
// OGCG:   store i32 1, ptr %[[P]]
// OGCG:   %[[ELEM_1:.*]] = getelementptr inbounds i32, ptr %[[P]], i64 1
// OGCG:   store i32 2, ptr %[[ELEM_1]]
// OGCG:   %[[ELEM_2:.*]] = getelementptr inbounds i32, ptr %[[ELEM_1]], i64 1
// OGCG:   store i32 3, ptr %[[ELEM_2]]
// OGCG:   %[[ELEM_3:.*]] = getelementptr inbounds i32, ptr %[[ELEM_2]], i64 1
// OGCG:   store i32 4, ptr %[[ELEM_3]]

void t_constant_size_partial_init() {
  auto p = new int[16] { 1, 2, 3 };
}

// CHECK:  cir.func {{.*}} @_Z28t_constant_size_partial_initv()
// CHECK:    %[[ALLOCATION_SIZE:.*]] = cir.const #cir.int<64> : !u64i
// CHECK:    %[[ALLOC_PTR:.*]] = cir.call @_Znam(%[[ALLOCATION_SIZE]]) {allocsize = array<i32: 0>} : (!u64i) -> (!cir.ptr<!void> {llvm.noundef})
// CHECK:    %[[ELEM_0_PTR:.*]] = cir.cast bitcast %[[ALLOC_PTR]] : !cir.ptr<!void> -> !cir.ptr<!s32i>
// CHECK:    %[[CONST_ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:    cir.store{{.*}} %[[CONST_ONE]], %[[ELEM_0_PTR]] : !s32i, !cir.ptr<!s32i>
// CHECK:    %[[OFFSET:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:    %[[ELEM_1_PTR:.*]] = cir.ptr_stride %[[ELEM_0_PTR]], %[[OFFSET]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
// CHECK:    %[[CONST_TWO:.*]] = cir.const #cir.int<2> : !s32i
// CHECK:    cir.store{{.*}} %[[CONST_TWO]], %[[ELEM_1_PTR]] : !s32i, !cir.ptr<!s32i>
// CHECK:    %[[OFFSET1:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:    %[[ELEM_2_PTR:.*]] = cir.ptr_stride %[[ELEM_1_PTR]], %[[OFFSET1]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
// CHECK:    %[[CONST_THREE:.*]] = cir.const #cir.int<3> : !s32i
// CHECK:    cir.store{{.*}} %[[CONST_THREE]], %[[ELEM_2_PTR]] : !s32i, !cir.ptr<!s32i>
// CHECK:    %[[OFFSET2:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:    %[[ELEM_3_PTR:.*]] = cir.ptr_stride %[[ELEM_2_PTR]], %[[OFFSET2]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
// CHECK:    %[[INIT_SIZE:.*]] = cir.const #cir.int<12> : !u64i
// CHECK:    %[[REMAINING_SIZE:.*]] = cir.binop(sub, %[[ALLOCATION_SIZE]], %[[INIT_SIZE]]) : !u64i
// CHECK:    %[[VOID_PTR:.*]] = cir.cast bitcast %[[ELEM_3_PTR]] : !cir.ptr<!s32i> -> !cir.ptr<!void>
// CHECK:    %[[ZERO:.*]] = cir.const #cir.int<0> : !u8i
// CHECK:    cir.libc.memset %[[REMAINING_SIZE]] bytes at %[[VOID_PTR]] to %[[ZERO]] : !cir.ptr<!void>, !u8i, !u64i

// LLVM: define {{.*}} void @_Z28t_constant_size_partial_initv()
// LLVM:   %[[P:.*]] = call noundef ptr @_Znam(i64 64)
// LLVM:   store i32 1, ptr %[[P]]
// LLVM:   %[[ELEM_1:.*]] = getelementptr i32, ptr %[[P]], i64 1
// LLVM:   store i32 2, ptr %[[ELEM_1]]
// LLVM:   %[[ELEM_2:.*]] = getelementptr i32, ptr %[[ELEM_1]], i64 1
// LLVM:   store i32 3, ptr %[[ELEM_2]]
// LLVM:   %[[ELEM_3:.*]] = getelementptr i32, ptr %[[ELEM_2]], i64 1
// LLVM:   call void @llvm.memset.p0.i64(ptr %[[ELEM_3]], i8 0, i64 52, i1 false)

// OGCG: define {{.*}} void @_Z28t_constant_size_partial_initv()
// OGCG:   %[[P:.*]] = call{{.*}} ptr @_Znam(i64{{.*}} 64)
// OGCG:   store i32 1, ptr %[[P]]
// OGCG:   %[[ELEM_1:.*]] = getelementptr inbounds i32, ptr %[[P]], i64 1
// OGCG:   store i32 2, ptr %[[ELEM_1]]
// OGCG:   %[[ELEM_2:.*]] = getelementptr inbounds i32, ptr %[[ELEM_1]], i64 1
// OGCG:   store i32 3, ptr %[[ELEM_2]]
// OGCG:   %[[ELEM_3:.*]] = getelementptr inbounds i32, ptr %[[ELEM_2]], i64 1
// OGCG:   call void @llvm.memset.p0.i64(ptr{{.*}} %[[ELEM_3]], i8 0, i64 52, i1 false)

void t_new_var_size(size_t n) {
  auto p = new char[n];
}

// CHECK:  cir.func {{.*}} @_Z14t_new_var_sizem
// CHECK:    %[[N:.*]] = cir.load{{.*}} %[[ARG_ALLOCA:.*]]
// CHECK:    %[[PTR:.*]] = cir.call @_Znam(%[[N]]) {allocsize = array<i32: 0>} : (!u64i)

// LLVM: define{{.*}} void @_Z14t_new_var_sizem
// LLVM:   %[[N:.*]] = load i64, ptr %{{.+}}
// LLVM:   %[[PTR:.*]] = call noundef ptr @_Znam(i64 %[[N]])

// OGCG: define{{.*}} void @_Z14t_new_var_sizem
// OGCG:   %[[N:.*]] = load i64, ptr %{{.+}}
// OGCG:   %[[PTR:.*]] = call {{.*}} ptr @_Znam(i64 {{.*}} %[[N]])

void t_new_var_size2(int n) {
  auto p = new char[n];
}

// CHECK:  cir.func {{.*}} @_Z15t_new_var_size2i
// CHECK:    %[[N:.*]] = cir.load{{.*}} %[[ARG_ALLOCA:.*]]
// CHECK:    %[[N_SIZE_T:.*]] = cir.cast integral %[[N]] : !s32i -> !u64i
// CHECK:    %[[PTR:.*]] = cir.call @_Znam(%[[N_SIZE_T]]) {allocsize = array<i32: 0>} : (!u64i)

// LLVM: define{{.*}} void @_Z15t_new_var_size2i
// LLVM:   %[[N:.*]] = load i32, ptr %{{.+}}
// LLVM:   %[[N_SIZE_T:.*]] = sext i32 %[[N]] to i64
// LLVM:   %[[PTR:.*]] = call noundef ptr @_Znam(i64 %[[N_SIZE_T]])

// OGCG: define{{.*}} void @_Z15t_new_var_size2i
// OGCG:   %[[N:.*]] = load i32, ptr %{{.+}}
// OGCG:   %[[N_SIZE_T:.*]] = sext i32 %[[N]] to i64
// OGCG:   %[[PTR:.*]] = call {{.*}} ptr @_Znam(i64 {{.*}} %[[N_SIZE_T]])

void t_new_var_size3(size_t n) {
  auto p = new double[n];
}

// CHECK:  cir.func {{.*}} @_Z15t_new_var_size3m
// CHECK:    %[[N:.*]] = cir.load{{.*}} %[[ARG_ALLOCA:.*]]
// CHECK:    %[[ELEMENT_SIZE:.*]] = cir.const #cir.int<8> : !u64i
// CHECK:    %[[RESULT:.*]], %[[OVERFLOW:.*]] = cir.binop.overflow(mul, %[[N]], %[[ELEMENT_SIZE]]) : !u64i, (!u64i, !cir.bool)
// CHECK:    %[[ALL_ONES:.*]] = cir.const #cir.int<18446744073709551615> : !u64i
// CHECK:    %[[ALLOC_SIZE:.*]] = cir.select if %[[OVERFLOW]] then %[[ALL_ONES]] else %[[RESULT]] : (!cir.bool, !u64i, !u64i)
// CHECK:    %[[PTR:.*]] = cir.call @_Znam(%[[ALLOC_SIZE]]) {allocsize = array<i32: 0>} : (!u64i)

// LLVM: define{{.*}} void @_Z15t_new_var_size3m
// LLVM:   %[[N:.*]] = load i64, ptr %{{.+}}
// LLVM:   %[[MUL_OVERFLOW:.*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %[[N]], i64 8)
// LLVM:   %[[ELEMENT_SIZE:.*]] = extractvalue { i64, i1 } %[[MUL_OVERFLOW]], 0
// LLVM:   %[[OVERFLOW:.*]] = extractvalue { i64, i1 } %[[MUL_OVERFLOW]], 1
// LLVM:   %[[ALLOC_SIZE:.*]] = select i1 %[[OVERFLOW]], i64 -1, i64 %[[ELEMENT_SIZE]]
// LLVM:   %[[RESULT:.*]] = call noundef ptr @_Znam(i64 %[[ALLOC_SIZE]])

// OGCG: define{{.*}} void @_Z15t_new_var_size3m
// OGCG:   %[[N:.*]] = load i64, ptr %{{.+}}
// OGCG:   %[[MUL_OVERFLOW:.*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %[[N]], i64 8)
// OGCG:   %[[OVERFLOW:.*]] = extractvalue { i64, i1 } %[[MUL_OVERFLOW]], 1
// OGCG:   %[[ELEMENT_SIZE:.*]] = extractvalue { i64, i1 } %[[MUL_OVERFLOW]], 0
// OGCG:   %[[ALLOC_SIZE:.*]] = select i1 %[[OVERFLOW]], i64 -1, i64 %[[ELEMENT_SIZE]]
// OGCG:   %[[PTR:.*]] = call {{.*}} ptr @_Znam(i64 {{.*}} %[[ALLOC_SIZE]])

void t_new_var_size4(int n) {
  auto p = new double[n];
}

// CHECK:  cir.func {{.*}} @_Z15t_new_var_size4i
// CHECK:    %[[N:.*]] = cir.load{{.*}} %[[ARG_ALLOCA:.*]]
// CHECK:    %[[N_SIZE_T:.*]] = cir.cast integral %[[N]] : !s32i -> !u64i
// CHECK:    %[[ELEMENT_SIZE:.*]] = cir.const #cir.int<8> : !u64i
// CHECK:    %[[RESULT:.*]], %[[OVERFLOW:.*]] = cir.binop.overflow(mul, %[[N_SIZE_T]], %[[ELEMENT_SIZE]]) : !u64i, (!u64i, !cir.bool)
// CHECK:    %[[ALL_ONES:.*]] = cir.const #cir.int<18446744073709551615> : !u64i
// CHECK:    %[[ALLOC_SIZE:.*]] = cir.select if %[[OVERFLOW]] then %[[ALL_ONES]] else %[[RESULT]] : (!cir.bool, !u64i, !u64i)
// CHECK:    %[[PTR:.*]] = cir.call @_Znam(%[[ALLOC_SIZE]]) {allocsize = array<i32: 0>} : (!u64i)

// LLVM: define{{.*}} void @_Z15t_new_var_size4i
// LLVM:   %[[N:.*]] = load i32, ptr %{{.+}}
// LLVM:   %[[N_SIZE_T:.*]] = sext i32 %[[N]] to i64
// LLVM:   %[[MUL_OVERFLOW:.*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %[[N_SIZE_T]], i64 8)
// LLVM:   %[[ELEMENT_SIZE:.*]] = extractvalue { i64, i1 } %[[MUL_OVERFLOW]], 0
// LLVM:   %[[OVERFLOW:.*]] = extractvalue { i64, i1 } %[[MUL_OVERFLOW]], 1
// LLVM:   %[[ALLOC_SIZE:.*]] = select i1 %[[OVERFLOW]], i64 -1, i64 %[[ELEMENT_SIZE]]
// LLVM:   %[[PTR:.*]] = call noundef ptr @_Znam(i64 %[[ALLOC_SIZE]])

// OGCG: define{{.*}} void @_Z15t_new_var_size4i
// OGCG:   %[[N:.*]] = load i32, ptr %{{.+}}
// OGCG:   %[[N_SIZE_T:.*]] = sext i32 %[[N]] to i64
// OGCG:   %[[MUL_OVERFLOW:.*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %[[N_SIZE_T]], i64 8)
// OGCG:   %[[OVERFLOW:.*]] = extractvalue { i64, i1 } %[[MUL_OVERFLOW]], 1
// OGCG:   %[[ELEMENT_SIZE:.*]] = extractvalue { i64, i1 } %[[MUL_OVERFLOW]], 0
// OGCG:   %[[ALLOC_SIZE:.*]] = select i1 %[[OVERFLOW]], i64 -1, i64 %[[ELEMENT_SIZE]]
// OGCG:   %[[PTR:.*]] = call {{.*}} ptr @_Znam(i64 {{.*}} %[[ALLOC_SIZE]])

void t_new_var_size5(int n) {
  auto p = new double[n][2][3];
}

// NUM_ELEMENTS isn't used in this case because there is no cookie. It isn't
// used in the allocation size because the allocation size is calculated with
// the element size and the fixed size dimensions already combined (6 * 8 = 48).
// We don't DCE NUM_ELEMENTS because it's not a constant, but later
// optimizations will eliminate it.

// CHECK:  cir.func {{.*}} @_Z15t_new_var_size5i
// CHECK:    %[[N:.*]] = cir.load{{.*}} %[[ARG_ALLOCA:.*]]
// CHECK:    %[[N_SIZE_T:.*]] = cir.cast integral %[[N]] : !s32i -> !u64i
// CHECK:    %[[ELEMENT_SIZE:.*]] = cir.const #cir.int<48> : !u64i
// CHECK:    %[[RESULT:.*]], %[[OVERFLOW:.*]] = cir.binop.overflow(mul, %[[N_SIZE_T]], %[[ELEMENT_SIZE]]) : !u64i, (!u64i, !cir.bool)
// CHECK:    %[[NUM_ELEMENTS_MULTIPLIER:.*]] = cir.const #cir.int<6>
// CHECK:    %[[NUM_ELEMENTS:.*]] = cir.binop(mul, %[[N_SIZE_T]], %[[NUM_ELEMENTS_MULTIPLIER]]) : !u64i
// CHECK:    %[[ALL_ONES:.*]] = cir.const #cir.int<18446744073709551615> : !u64i
// CHECK:    %[[ALLOC_SIZE:.*]] = cir.select if %[[OVERFLOW]] then %[[ALL_ONES]] else %[[RESULT]] : (!cir.bool, !u64i, !u64i)
// CHECK:    %[[PTR:.*]] = cir.call @_Znam(%[[ALLOC_SIZE]]) {allocsize = array<i32: 0>} : (!u64i)

// LLVM: define{{.*}} void @_Z15t_new_var_size5i
// LLVM:   %[[N:.*]] = load i32, ptr %{{.+}}
// LLVM:   %[[N_SIZE_T:.*]] = sext i32 %[[N]] to i64
// LLVM:   %[[MUL_OVERFLOW:.*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %[[N_SIZE_T]], i64 48)
// LLVM:   %[[ELEMENT_SIZE:.*]] = extractvalue { i64, i1 } %[[MUL_OVERFLOW]], 0
// LLVM:   %[[OVERFLOW:.*]] = extractvalue { i64, i1 } %[[MUL_OVERFLOW]], 1
// LLVM:   %[[ALLOC_SIZE:.*]] = select i1 %[[OVERFLOW]], i64 -1, i64 %[[ELEMENT_SIZE]]
// LLVM:   %[[PTR:.*]] = call noundef ptr @_Znam(i64 %[[ALLOC_SIZE]])

// OGCG: define{{.*}} void @_Z15t_new_var_size5i
// OGCG:   %[[N:.*]] = load i32, ptr %{{.+}}
// OGCG:   %[[N_SIZE_T:.*]] = sext i32 %[[N]] to i64
// OGCG:   %[[MUL_OVERFLOW:.*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %[[N_SIZE_T]], i64 48)
// OGCG:   %[[OVERFLOW:.*]] = extractvalue { i64, i1 } %[[MUL_OVERFLOW]], 1
// OGCG:   %[[ELEMENT_SIZE:.*]] = extractvalue { i64, i1 } %[[MUL_OVERFLOW]], 0
// OGCG:   %[[ALLOC_SIZE:.*]] = select i1 %[[OVERFLOW]], i64 -1, i64 %[[ELEMENT_SIZE]]
// OGCG:   %[[PTR:.*]] = call {{.*}} ptr @_Znam(i64 {{.*}} %[[ALLOC_SIZE]])

void t_new_var_size6(int n) {
  auto p = new double[n] { 1.0, 2.0, 3.0 };
}

// CHECK:  cir.func {{.*}} @_Z15t_new_var_size6i
// CHECK:    %[[N:.*]] = cir.load{{.*}} %[[ARG_ALLOCA:.*]]
// CHECK:    %[[N_SIZE_T:.*]] = cir.cast integral %[[N]] : !s32i -> !u64i
// CHECK:    %[[MIN_SIZE:.*]] = cir.const #cir.int<3> : !u64i
// CHECK:    %[[LT_MIN_SIZE:.*]] = cir.cmp(lt, %[[N_SIZE_T]], %[[MIN_SIZE]]) : !u64i, !cir.bool
// CHECK:    %[[ELEMENT_SIZE:.*]] = cir.const #cir.int<8> : !u64i
// CHECK:    %[[RESULT:.*]], %[[OVERFLOW:.*]] = cir.binop.overflow(mul, %[[N_SIZE_T]], %[[ELEMENT_SIZE]]) : !u64i, (!u64i, !cir.bool)
// CHECK:    %[[ANY_OVERFLOW:.*]] = cir.binop(or, %[[LT_MIN_SIZE]], %[[OVERFLOW]]) : !cir.bool
// CHECK:    %[[ALL_ONES:.*]] = cir.const #cir.int<18446744073709551615> : !u64i
// CHECK:    %[[ALLOC_SIZE:.*]] = cir.select if %[[ANY_OVERFLOW]] then %[[ALL_ONES]] else %[[RESULT]] : (!cir.bool, !u64i, !u64i)
// CHECK:    %[[PTR:.*]] = cir.call @_Znam(%[[ALLOC_SIZE]]) {allocsize = array<i32: 0>} : (!u64i)
// CHECK:    %[[PTR_DOUBLE:.*]] = cir.cast bitcast %[[PTR]] : !cir.ptr<!void> -> !cir.ptr<!cir.double>
// CHECK:    %[[ELEM_0:.*]] = cir.const #cir.fp<1.000000e+00> : !cir.double
// CHECK:    cir.store{{.*}} %[[ELEM_0]], %[[PTR_DOUBLE]]
// CHECK:    %[[ONE:.*]] = cir.const #cir.int<1>
// CHECK:    %[[PTR_DOUBLE_1:.*]] = cir.ptr_stride %[[PTR_DOUBLE]], %[[ONE]]
// CHECK:    %[[ELEM_1:.*]] = cir.const #cir.fp<2.000000e+00> : !cir.double
// CHECK:    cir.store{{.*}} %[[ELEM_1]], %[[PTR_DOUBLE_1]]
// CHECK:    %[[ONE:.*]] = cir.const #cir.int<1>
// CHECK:    %[[PTR_DOUBLE_2:.*]] = cir.ptr_stride %[[PTR_DOUBLE_1]], %[[ONE]]
// CHECK:    %[[ELEM_2:.*]] = cir.const #cir.fp<3.000000e+00> : !cir.double
// CHECK:    cir.store{{.*}} %[[ELEM_2]], %[[PTR_DOUBLE_2]]
// CHECK:    %[[ONE:.*]] = cir.const #cir.int<1>
// CHECK:    %[[PTR_DOUBLE_3:.*]] = cir.ptr_stride %[[PTR_DOUBLE_2]], %[[ONE]]
// CHECK:    %[[INIT_SIZE:.*]] = cir.const #cir.int<24> : !u64i
// CHECK:    %[[REMAINING_SIZE:.*]] = cir.binop(sub, %[[ALLOC_SIZE]], %[[INIT_SIZE]]) : !u64i
// CHECK:    %[[PTR_DOUBLE_3_VOID:.*]] = cir.cast bitcast %[[PTR_DOUBLE_3]] : !cir.ptr<!cir.double> -> !cir.ptr<!void>
// CHECK:    %[[ZERO:.*]] = cir.const #cir.int<0> : !u8i
// CHECK:    cir.libc.memset{{.*}} bytes at %[[PTR_DOUBLE_3_VOID]] to %[[ZERO]]

// LLVM: define{{.*}} void @_Z15t_new_var_size6i
// LLVM:   %[[N:.*]] = load i32, ptr %{{.+}}
// LLVM:   %[[N_SIZE_T:.*]] = sext i32 %[[N]] to i64
// LLVM:   %[[LT_MIN_SIZE:.*]] = icmp ult i64 %[[N_SIZE_T]], 3
// LLVM:   %[[MUL_OVERFLOW:.*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %[[N_SIZE_T]], i64 8)
// LLVM:   %[[ELEMENT_SIZE:.*]] = extractvalue { i64, i1 } %[[MUL_OVERFLOW]], 0
// LLVM:   %[[OVERFLOW:.*]] = extractvalue { i64, i1 } %[[MUL_OVERFLOW]], 1
// LLVM:   %[[ANY_OVERFLOW:.*]] = or i1 %[[LT_MIN_SIZE]], %[[OVERFLOW]]
// LLVM:   %[[ALLOC_SIZE:.*]] = select i1 %[[ANY_OVERFLOW]], i64 -1, i64 %[[ELEMENT_SIZE]]
// LLVM:   %[[PTR:.*]] = call noundef ptr @_Znam(i64 %[[ALLOC_SIZE]])
// LLVM:   store double 1.000000e+00, ptr %[[PTR]], align 8
// LLVM:   %[[ELEM_1:.*]] = getelementptr double, ptr %[[PTR]], i64 1
// LLVM:   store double 2.000000e+00, ptr %[[ELEM_1]], align 8
// LLVM:   %[[ELEM_2:.*]] = getelementptr double, ptr %[[ELEM_1]], i64 1
// LLVM:   store double 3.000000e+00, ptr %[[ELEM_2]], align 8
// LLVM:   %[[ELEM_3:.*]] = getelementptr double, ptr %[[ELEM_2]], i64 1
// LLVM:   %[[REMAINING_SIZE:.*]] = sub i64 %[[ALLOC_SIZE]], 24
// LLVM:   call void @llvm.memset.p0.i64(ptr %[[ELEM_3]], i8 0, i64 %[[REMAINING_SIZE]], i1 false)

// OGCG: define{{.*}} void @_Z15t_new_var_size6i
// OGCG:   %[[N:.*]] = load i32, ptr %{{.+}}
// OGCG:   %[[N_SIZE_T:.*]] = sext i32 %[[N]] to i64
// OGCG:   %[[LT_MIN_SIZE:.*]] = icmp ult i64 %[[N_SIZE_T]], 3
// OGCG:   %[[MUL_OVERFLOW:.*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %[[N_SIZE_T]], i64 8)
// OGCG:   %[[OVERFLOW:.*]] = extractvalue { i64, i1 } %[[MUL_OVERFLOW]], 1
// OGCG:   %[[ANY_OVERFLOW:.*]] = or i1 %[[LT_MIN_SIZE]], %[[OVERFLOW]]
// OGCG:   %[[ELEMENT_SIZE:.*]] = extractvalue { i64, i1 } %[[MUL_OVERFLOW]], 0
// OGCG:   %[[ALLOC_SIZE:.*]] = select i1 %[[ANY_OVERFLOW]], i64 -1, i64 %[[ELEMENT_SIZE]]
// OGCG:   %[[PTR:.*]] = call {{.*}} ptr @_Znam(i64 {{.*}} %[[ALLOC_SIZE]])
// OGCG:   store double 1.000000e+00, ptr %[[PTR]], align 8
// OGCG:   %[[ELEM_1:.*]] = getelementptr inbounds double, ptr %[[PTR]], i64 1
// OGCG:   store double 2.000000e+00, ptr %[[ELEM_1]], align 8
// OGCG:   %[[ELEM_2:.*]] = getelementptr inbounds double, ptr %[[ELEM_1]], i64 1
// OGCG:   store double 3.000000e+00, ptr %[[ELEM_2]], align 8
// OGCG:   %[[ELEM_3:.*]] = getelementptr inbounds double, ptr %[[ELEM_2]], i64 1
// OGCG:   %[[REMAINING_SIZE:.*]] = sub i64 %[[ALLOC_SIZE]], 24
// OGCG:   call void @llvm.memset.p0.i64(ptr{{.*}} %[[ELEM_3]], i8 0, i64 %[[REMAINING_SIZE]], i1 false)

void t_new_var_size7(__int128 n) {
  auto p = new double[n];
}

// CHECK:  cir.func {{.*}} @_Z15t_new_var_size7n
// CHECK:    %[[N:.*]] = cir.load{{.*}} %[[ARG_ALLOCA:.*]]
// CHECK:    %[[N_SIZE_T:.*]] = cir.cast integral %[[N]] : !s128i -> !u64i
// CHECK:    %[[ELEMENT_SIZE:.*]] = cir.const #cir.int<8> : !u64i
// CHECK:    %[[RESULT:.*]], %[[OVERFLOW:.*]] = cir.binop.overflow(mul, %[[N_SIZE_T]], %[[ELEMENT_SIZE]]) : !u64i, (!u64i, !cir.bool)
// CHECK:    %[[ALL_ONES:.*]] = cir.const #cir.int<18446744073709551615> : !u64i
// CHECK:    %[[ALLOC_SIZE:.*]] = cir.select if %[[OVERFLOW]] then %[[ALL_ONES]] else %[[RESULT]] : (!cir.bool, !u64i, !u64i)
// CHECK:    %[[PTR:.*]] = cir.call @_Znam(%[[ALLOC_SIZE]]) {allocsize = array<i32: 0>} : (!u64i)

// LLVM: define{{.*}} void @_Z15t_new_var_size7n
// LLVM:   %[[N:.*]] = load i128, ptr %{{.+}}
// LLVM:   %[[N_SIZE_T:.*]] = trunc i128 %[[N]] to i64
// LLVM:   %[[MUL_OVERFLOW:.*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %[[N_SIZE_T]], i64 8)
// LLVM:   %[[ELEMENT_SIZE:.*]] = extractvalue { i64, i1 } %[[MUL_OVERFLOW]], 0
// LLVM:   %[[OVERFLOW:.*]] = extractvalue { i64, i1 } %[[MUL_OVERFLOW]], 1
// LLVM:   %[[ALLOC_SIZE:.*]] = select i1 %[[OVERFLOW]], i64 -1, i64 %[[ELEMENT_SIZE]]
// LLVM:   %[[PTR:.*]] = call noundef ptr @_Znam(i64 %[[ALLOC_SIZE]])

// OGCG: define{{.*}} void @_Z15t_new_var_size7n
// OGCG:   %[[N:.*]] = load i128, ptr %{{.+}}
// OGCG:   %[[N_SIZE_T:.*]] = trunc i128 %[[N]] to i64
// OGCG:   %[[MUL_OVERFLOW:.*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %[[N_SIZE_T]], i64 8)
// OGCG:   %[[OVERFLOW:.*]] = extractvalue { i64, i1 } %[[MUL_OVERFLOW]], 1
// OGCG:   %[[ELEMENT_SIZE:.*]] = extractvalue { i64, i1 } %[[MUL_OVERFLOW]], 0
// OGCG:   %[[ALLOC_SIZE:.*]] = select i1 %[[OVERFLOW]], i64 -1, i64 %[[ELEMENT_SIZE]]
// OGCG:   %[[PTR:.*]] = call {{.*}} ptr @_Znam(i64 {{.*}} %[[ALLOC_SIZE]])

void t_new_var_size_nontrivial(size_t n) {
  auto p = new D[n];
}

// CHECK:  cir.func {{.*}} @_Z25t_new_var_size_nontrivialm
// CHECK:    %[[N:.*]] = cir.load{{.*}} %[[ARG_ALLOCA:.*]]
// CHECK:    %[[ELEMENT_SIZE:.*]] = cir.const #cir.int<4> : !u64i
// CHECK:    %[[SIZE_WITHOUT_COOKIE:.*]], %[[OVERFLOW:.*]] = cir.binop.overflow(mul, %[[N]], %[[ELEMENT_SIZE]]) : !u64i, (!u64i, !cir.bool)
// CHECK:    %[[COOKIE_SIZE:.*]] = cir.const #cir.int<8> : !u64i
// CHECK:    %[[SIZE:.*]], %[[OVERFLOW2:.*]] = cir.binop.overflow(add, %[[SIZE_WITHOUT_COOKIE]], %[[COOKIE_SIZE]]) : !u64i, (!u64i, !cir.bool)
// CHECK:    %[[ANY_OVERFLOW:.*]] = cir.binop(or, %[[OVERFLOW]], %[[OVERFLOW2]]) : !cir.bool
// CHECK:    %[[ALL_ONES:.*]] = cir.const #cir.int<18446744073709551615> : !u64i
// CHECK:    %[[ALLOC_SIZE:.*]] = cir.select if %[[ANY_OVERFLOW]] then %[[ALL_ONES]] else %[[SIZE]] : (!cir.bool, !u64i, !u64i)
// CHECK:    %[[PTR:.*]] = cir.call @_Znam(%[[ALLOC_SIZE]]) {allocsize = array<i32: 0>} : (!u64i)

// LLVM: define{{.*}} void @_Z25t_new_var_size_nontrivialm
// LLVM:   %[[N:.*]] = load i64, ptr %{{.+}}
// LLVM:   %[[MUL_OVERFLOW:.*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %[[N]], i64 4)
// LLVM:   %[[MUL_SIZE:.*]] = extractvalue { i64, i1 } %[[MUL_OVERFLOW]], 0
// LLVM:   %[[OVERFLOW:.*]] = extractvalue { i64, i1 } %[[MUL_OVERFLOW]], 1
// LLVM:   %[[ADD_OVERFLOW:.*]] = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %[[MUL_SIZE]], i64 8)
// LLVM:   %[[ELEMENT_SIZE:.*]] = extractvalue { i64, i1 } %[[ADD_OVERFLOW]], 0
// LLVM:   %[[OVERFLOW_ADD:.*]] = extractvalue { i64, i1 } %[[ADD_OVERFLOW]], 1
// LLVM:   %[[ANY_OVERFLOW:.*]] = or i1 %[[OVERFLOW]], %[[OVERFLOW_ADD]]
// LLVM:   %[[ALLOC_SIZE:.*]] = select i1 %[[ANY_OVERFLOW]], i64 -1, i64 %[[ELEMENT_SIZE]]
// LLVM:   %[[PTR:.*]] = call noundef ptr @_Znam(i64 %[[ALLOC_SIZE]])

// OGCG: define{{.*}} void @_Z25t_new_var_size_nontrivialm
// OGCG:   %[[N:.*]] = load i64, ptr %{{.+}}
// OGCG:   %[[MUL_OVERFLOW:.*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %[[N]], i64 4)
// OGCG:   %[[OVERFLOW:.*]] = extractvalue { i64, i1 } %[[MUL_OVERFLOW]], 1
// OGCG:   %[[MUL_SIZE:.*]] = extractvalue { i64, i1 } %[[MUL_OVERFLOW]], 0
// OGCG:   %[[ADD_OVERFLOW:.*]] = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %[[MUL_SIZE]], i64 8)
// OGCG:   %[[OVERFLOW_ADD:.*]] = extractvalue { i64, i1 } %[[ADD_OVERFLOW]], 1
// OGCG:   %[[ANY_OVERFLOW:.*]] = or i1 %[[OVERFLOW]], %[[OVERFLOW_ADD]]
// OGCG:   %[[ELEMENT_SIZE:.*]] = extractvalue { i64, i1 } %[[ADD_OVERFLOW]], 0
// OGCG:   %[[ALLOC_SIZE:.*]] = select i1 %[[ANY_OVERFLOW]], i64 -1, i64 %[[ELEMENT_SIZE]]
// OGCG:   %[[PTR:.*]] = call {{.*}} ptr @_Znam(i64 {{.*}} %[[ALLOC_SIZE]])
