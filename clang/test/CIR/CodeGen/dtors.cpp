// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -mconstructor-aliases -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -mconstructor-aliases -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

struct A {
  ~A();
};

void test_temporary_dtor() {
  A();
}

// CIR: cir.func dso_local @_Z19test_temporary_dtorv()
// CIR:   %[[ALLOCA:.*]] = cir.alloca !rec_A, !cir.ptr<!rec_A>, ["agg.tmp0"]
// CIR:   cir.call @_ZN1AD1Ev(%[[ALLOCA]]) nothrow : (!cir.ptr<!rec_A>) -> ()

// LLVM: define dso_local void @_Z19test_temporary_dtorv()
// LLVM:   %[[ALLOCA:.*]] = alloca %struct.A, i64 1, align 1
// LLVM:   call void @_ZN1AD1Ev(ptr %[[ALLOCA]])

// OGCG: define dso_local void @_Z19test_temporary_dtorv()
// OGCG:   %[[ALLOCA:.*]] = alloca %struct.A, align 1
// OGCG:   call void @_ZN1AD1Ev(ptr {{.*}} %[[ALLOCA]])

struct B {
  int n;
  B(int n) : n(n) {}
  ~B() {}
};

bool make_temp(const B &) { return false; }
bool test_temp_or() { return make_temp(1) || make_temp(2); }

// CIR: cir.func{{.*}} @_Z12test_temp_orv()
// CIR:   %[[SCOPE:.*]] = cir.scope {
// CIR:     %[[REF_TMP0:.*]] = cir.alloca !rec_B, !cir.ptr<!rec_B>, ["ref.tmp0"]
// CIR:     %[[ONE:.*]] = cir.const #cir.int<1>
// CIR:     cir.call @_ZN1BC2Ei(%[[REF_TMP0]], %[[ONE]])
// CIR:     %[[MAKE_TEMP0:.*]] = cir.call @_Z9make_tempRK1B(%[[REF_TMP0]])
// CIR:     %[[TERNARY:.*]] = cir.ternary(%[[MAKE_TEMP0]], true {
// CIR:       %[[TRUE:.*]] = cir.const #true
// CIR:       cir.yield %[[TRUE]] : !cir.bool
// CIR:     }, false {
// CIR:       %[[REF_TMP1:.*]] = cir.alloca !rec_B, !cir.ptr<!rec_B>, ["ref.tmp1"]
// CIR:       %[[TWO:.*]] = cir.const #cir.int<2>
// CIR:       cir.call @_ZN1BC2Ei(%[[REF_TMP1]], %[[TWO]])
// CIR:       %[[MAKE_TEMP1:.*]] = cir.call @_Z9make_tempRK1B(%[[REF_TMP1]])
// CIR:       cir.call @_ZN1BD2Ev(%[[REF_TMP1]])
// CIR:       cir.yield %[[MAKE_TEMP1]] : !cir.bool
// CIR:     })
// CIR:     cir.call @_ZN1BD2Ev(%[[REF_TMP0]])
// CIR:     cir.yield %[[TERNARY]] : !cir.bool
// CIR:   } : !cir.bool

// LLVM: define{{.*}} i1 @_Z12test_temp_orv() {
// LLVM:   %[[REF_TMP0:.*]] = alloca %struct.B
// LLVM:   %[[REF_TMP1:.*]] = alloca %struct.B
// LLVM:   br label %[[LOR_BEGIN:.*]]
// LLVM: [[LOR_BEGIN]]:
// LLVM:   call void @_ZN1BC2Ei(ptr %[[REF_TMP0]], i32 1)
// LLVM:   %[[MAKE_TEMP0:.*]] = call i1 @_Z9make_tempRK1B(ptr %[[REF_TMP0]])
// LLVM:   br i1 %[[MAKE_TEMP0]], label %[[LHS_TRUE_BLOCK:.*]], label %[[LHS_FALSE_BLOCK:.*]]
// LLVM: [[LHS_TRUE_BLOCK]]:
// LLVM:   br label %[[RESULT_BLOCK:.*]]
// LLVM: [[LHS_FALSE_BLOCK]]:
// LLVM:   call void @_ZN1BC2Ei(ptr %[[REF_TMP1]], i32 2)
// LLVM:   %[[MAKE_TEMP1:.*]] = call i1 @_Z9make_tempRK1B(ptr %[[REF_TMP1]])
// LLVM:   call void @_ZN1BD2Ev(ptr %[[REF_TMP1]])
// LLVM:   br label %[[RESULT_BLOCK]]
// LLVM: [[RESULT_BLOCK]]:
// LLVM:   %[[RESULT:.*]] = phi i1 [ %[[MAKE_TEMP1]], %[[LHS_FALSE_BLOCK]] ], [ true, %[[LHS_TRUE_BLOCK]] ]
// LLVM:   br label %[[LOR_END:.*]]
// LLVM: [[LOR_END]]:
// LLVM:   call void @_ZN1BD2Ev(ptr %[[REF_TMP0]])

// OGCG: define {{.*}} i1 @_Z12test_temp_orv()
// OGCG: [[ENTRY:.*]]:
// OGCG:   %[[RETVAL:.*]] = alloca i1
// OGCG:   %[[REF_TMP0:.*]] = alloca %struct.B
// OGCG:   %[[REF_TMP1:.*]] = alloca %struct.B
// OGCG:   %[[CLEANUP_COND:.*]] = alloca i1
// OGCG:   call void @_ZN1BC2Ei(ptr {{.*}} %[[REF_TMP0]], i32 {{.*}} 1)
// OGCG:   %[[MAKE_TEMP0:.*]] = call {{.*}} i1 @_Z9make_tempRK1B(ptr {{.*}} %[[REF_TMP0]])
// OGCG:   store i1 false, ptr %cleanup.cond
// OGCG:   br i1 %[[MAKE_TEMP0]], label %[[LOR_END:.*]], label %[[LOR_RHS:.*]]
// OGCG: [[LOR_RHS]]:
// OGCG:   call void @_ZN1BC2Ei(ptr {{.*}} %[[REF_TMP1]], i32 {{.*}} 2)
// OGCG:   store i1 true, ptr %[[CLEANUP_COND]]
// OGCG:   %[[MAKE_TEMP1:.*]] = call {{.*}} i1 @_Z9make_tempRK1B(ptr {{.*}} %[[REF_TMP1]])
// OGCG:   br label %[[LOR_END]]
// OGCG: [[LOR_END]]:
// OGCG:    %[[PHI:.*]] = phi i1 [ true, %[[ENTRY]] ], [ %[[MAKE_TEMP1]], %[[LOR_RHS]] ]
// OGCG:   store i1 %[[PHI]], ptr %[[RETVAL]]
// OGCG:   %[[CLEANUP_IS_ACTIVE:.*]] = load i1, ptr %[[CLEANUP_COND]]
// OGCG:   br i1 %[[CLEANUP_IS_ACTIVE]], label %[[CLEANUP_ACTION:.*]], label %[[CLEANUP_DONE:.*]]
// OGCG: [[CLEANUP_ACTION]]:
// OGCG:   call void @_ZN1BD2Ev(ptr {{.*}} %[[REF_TMP1]])
// OGCG:   br label %[[CLEANUP_DONE]]
// OGCG: [[CLEANUP_DONE]]:
// OGCG:   call void @_ZN1BD2Ev(ptr {{.*}} %[[REF_TMP0]])

bool test_temp_and() { return make_temp(1) && make_temp(2); }

// CIR: cir.func{{.*}} @_Z13test_temp_andv()
// CIR:   %[[SCOPE:.*]] = cir.scope {
// CIR:     %[[REF_TMP0:.*]] = cir.alloca !rec_B, !cir.ptr<!rec_B>, ["ref.tmp0"]
// CIR:     %[[ONE:.*]] = cir.const #cir.int<1>
// CIR:     cir.call @_ZN1BC2Ei(%[[REF_TMP0]], %[[ONE]])
// CIR:     %[[MAKE_TEMP0:.*]] = cir.call @_Z9make_tempRK1B(%[[REF_TMP0]])
// CIR:     %[[TERNARY:.*]] = cir.ternary(%[[MAKE_TEMP0]], true {
// CIR:       %[[REF_TMP1:.*]] = cir.alloca !rec_B, !cir.ptr<!rec_B>, ["ref.tmp1"]
// CIR:       %[[TWO:.*]] = cir.const #cir.int<2>
// CIR:       cir.call @_ZN1BC2Ei(%[[REF_TMP1]], %[[TWO]])
// CIR:       %[[MAKE_TEMP1:.*]] = cir.call @_Z9make_tempRK1B(%[[REF_TMP1]])
// CIR:       cir.call @_ZN1BD2Ev(%[[REF_TMP1]])
// CIR:       cir.yield %[[MAKE_TEMP1]] : !cir.bool
// CIR:     }, false {
// CIR:       %[[FALSE:.*]] = cir.const #false
// CIR:       cir.yield %[[FALSE]] : !cir.bool
// CIR:     })
// CIR:     cir.call @_ZN1BD2Ev(%[[REF_TMP0]])
// CIR:     cir.yield %[[TERNARY]] : !cir.bool
// CIR:   } : !cir.bool

// LLVM: define{{.*}} i1 @_Z13test_temp_andv() {
// LLVM:   %[[REF_TMP0:.*]] = alloca %struct.B
// LLVM:   %[[REF_TMP1:.*]] = alloca %struct.B
// LLVM:   br label %[[LAND_BEGIN:.*]]
// LLVM: [[LAND_BEGIN]]:
// LLVM:   call void @_ZN1BC2Ei(ptr %[[REF_TMP0]], i32 1)
// LLVM:   %[[MAKE_TEMP0:.*]] = call i1 @_Z9make_tempRK1B(ptr %[[REF_TMP0]])
// LLVM:   br i1 %[[MAKE_TEMP0]], label %[[LHS_TRUE_BLOCK:.*]], label %[[LHS_FALSE_BLOCK:.*]]
// LLVM: [[LHS_TRUE_BLOCK]]:
// LLVM:   call void @_ZN1BC2Ei(ptr %[[REF_TMP1]], i32 2)
// LLVM:   %[[MAKE_TEMP1:.*]] = call i1 @_Z9make_tempRK1B(ptr %[[REF_TMP1]])
// LLVM:   call void @_ZN1BD2Ev(ptr %[[REF_TMP1]])
// LLVM:   br label %[[RESULT_BLOCK:.*]]
// LLVM: [[LHS_FALSE_BLOCK]]:
// LLVM:   br label %[[RESULT_BLOCK]]
// LLVM: [[RESULT_BLOCK]]:
// LLVM:   %[[RESULT:.*]] = phi i1 [ false, %[[LHS_FALSE_BLOCK]] ], [ %[[MAKE_TEMP1]], %[[LHS_TRUE_BLOCK]] ]
// LLVM:   br label %[[LAND_END:.*]]
// LLVM: [[LAND_END]]:
// LLVM:   call void @_ZN1BD2Ev(ptr %[[REF_TMP0]])

// OGCG: define {{.*}} i1 @_Z13test_temp_andv()
// OGCG: [[ENTRY:.*]]:
// OGCG:   %[[RETVAL:.*]] = alloca i1
// OGCG:   %[[REF_TMP0:.*]] = alloca %struct.B
// OGCG:   %[[REF_TMP1:.*]] = alloca %struct.B
// OGCG:   %[[CLEANUP_COND:.*]] = alloca i1
// OGCG:   call void @_ZN1BC2Ei(ptr {{.*}} %[[REF_TMP0]], i32 {{.*}} 1)
// OGCG:   %[[MAKE_TEMP0:.*]] = call {{.*}} i1 @_Z9make_tempRK1B(ptr {{.*}} %[[REF_TMP0]])
// OGCG:   store i1 false, ptr %cleanup.cond
// OGCG:   br i1 %[[MAKE_TEMP0]], label %[[LAND_RHS:.*]], label %[[LAND_END:.*]]
// OGCG: [[LAND_RHS]]:
// OGCG:   call void @_ZN1BC2Ei(ptr {{.*}} %[[REF_TMP1]], i32 {{.*}} 2)
// OGCG:   store i1 true, ptr %[[CLEANUP_COND]]
// OGCG:   %[[MAKE_TEMP1:.*]] = call {{.*}} i1 @_Z9make_tempRK1B(ptr {{.*}} %[[REF_TMP1]])
// OGCG:   br label %[[LAND_END]]
// OGCG: [[LAND_END]]:
// OGCG:   %[[PHI:.*]] = phi i1 [ false, %[[ENTRY]] ], [ %[[MAKE_TEMP1]], %[[LAND_RHS]] ]
// OGCG:   store i1 %[[PHI]], ptr %[[RETVAL]]
// OGCG:   %[[CLEANUP_IS_ACTIVE:.*]] = load i1, ptr %[[CLEANUP_COND]]
// OGCG:   br i1 %[[CLEANUP_IS_ACTIVE]], label %[[CLEANUP_ACTION:.*]], label %[[CLEANUP_DONE:.*]]
// OGCG: [[CLEANUP_ACTION]]:
// OGCG:   call void @_ZN1BD2Ev(ptr {{.*}} %[[REF_TMP1]])
// OGCG:   br label %[[CLEANUP_DONE]]
// OGCG: [[CLEANUP_DONE]]:
// OGCG:   call void @_ZN1BD2Ev(ptr {{.*}} %[[REF_TMP0]])
