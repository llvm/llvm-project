// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

struct S {
  S();
  ~S();
  int get();
};

void test_ternary_temporary(bool c, int x) {
  int result = c ? S().get() : x;
}
// CIR-LABEL: @_Z22test_ternary_temporarybi
// CIR:   %[[TMP:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["ref.tmp0"]
// CIR:   %[[ACTIVE:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.cond"]
// The cleanup scope wraps the full expression so cleanups run on all exits.
// CIR:   cir.cleanup.scope {
// Load condition, then active flag false before the ternary (destructor guard).
// CIR:     %[[COND:.*]] = cir.load {{.*}} : !cir.ptr<!cir.bool>, !cir.bool
// CIR:     %[[FALSE:.*]] = cir.const #false
// CIR:     cir.store %[[FALSE]], %[[ACTIVE]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:     %{{.*}} = cir.ternary(%[[COND]], true {
// True branch: mark active before calling get() so cleanup runs.
// CIR:       cir.call @_ZN1SC1Ev(%[[TMP]])
// CIR:       %[[SET_TRUE:.*]] = cir.const #true
// CIR:       cir.store %[[SET_TRUE]], %[[ACTIVE]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:       %[[GET_RESULT:.*]] = cir.call @_ZN1S3getEv(%[[TMP]])
// CIR:       cir.yield %[[GET_RESULT]] : !s32i
// CIR:     }, false {
// CIR:       cir.yield
// CIR:     cir.yield
// CIR:   } cleanup normal {
// CIR:     %[[IS_ACTIVE:.*]] = cir.load {{.*}} %[[ACTIVE]]
// CIR:     cir.if %[[IS_ACTIVE]] {
// CIR:       cir.call @_ZN1SD1Ev(%[[TMP]])
// CIR:     }
// CIR:     cir.yield
// CIR:   }

// LLVM-LABEL: define dso_local void @_Z22test_ternary_temporarybi(
// LLVM:         %[[TMP:.*]] = alloca %struct.S
// LLVM:         %[[ACTIVE:.*]] = alloca i8
// LLVM:         %[[RESULT_TMP:.*]] = alloca i32
// LLVM:         br label %[[INIT:.*]]
// LLVM:       [[INIT]]:
// LLVM:         %[[COND_BYTE:.*]] = load i8, ptr %{{.*}}
// LLVM:         %[[COND_BOOL:.*]] = trunc i8 %[[COND_BYTE]] to i1
// LLVM:         store i8 0, ptr %[[ACTIVE]]
// LLVM:         br i1 %[[COND_BOOL]], label %[[TRUE_BR:.*]], label %[[FALSE_BR:.*]]
// LLVM:       [[TRUE_BR]]:
// LLVM:         call void @_ZN1SC1Ev(ptr {{.*}} %[[TMP]])
// LLVM:         store i8 1, ptr %[[ACTIVE]]
// LLVM:         %[[GET_RESULT:.*]] = call {{.*}} i32 @_ZN1S3getEv(ptr {{.*}} %[[TMP]])
// LLVM:         br label %[[MERGE:.*]]
// LLVM:       [[FALSE_BR]]:
// LLVM:         %[[XVAL:.*]] = load i32, ptr %{{.*}}
// LLVM:         br label %[[MERGE]]
// LLVM:       [[MERGE]]:
// LLVM:         %[[PHI:.*]] = phi i32 [ %[[XVAL]], %[[FALSE_BR]] ], [ %[[GET_RESULT]], %[[TRUE_BR]] ]
// LLVM:         br label %[[STORE:.*]]
// LLVM:       [[STORE]]:
// LLVM:         store i32 %[[PHI]], ptr %[[RESULT_TMP]]
// LLVM:         br label %[[CLEANUP:.*]]
// LLVM:       [[CLEANUP]]:
// LLVM:         %[[ACTIVE_BYTE:.*]] = load i8, ptr %[[ACTIVE]]
// LLVM:         %[[ACTIVE_BOOL:.*]] = trunc i8 %[[ACTIVE_BYTE]] to i1
// LLVM:         br i1 %[[ACTIVE_BOOL]], label %[[DTOR:.*]], label %[[SKIP_DTOR:.*]]
// LLVM:       [[DTOR]]:
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[TMP]])
// LLVM:         br label %[[SKIP_DTOR]]
// LLVM:       [[SKIP_DTOR]]:
// LLVM:         br label %[[EXIT:.*]]
// LLVM:       [[EXIT]]:
// LLVM:         %[[RESULT:.*]] = load i32, ptr %[[RESULT_TMP]]
// LLVM:         store i32 %[[RESULT]], ptr %{{.*}}

// OGCG-LABEL: define dso_local void @_Z22test_ternary_temporarybi(
// OGCG:       entry:
// OGCG:         store i1 false, ptr %[[ACTIVE:.*]]
// OGCG:         br i1 %[[COND_BOOL:.*]], label %[[TRUE_BR:.*]], label %[[FALSE_BR:.*]]
// OGCG:       [[TRUE_BR]]:
// OGCG:         call void @_ZN1SC1Ev(ptr {{.*}} %[[TMP:.*]])
// OGCG:         store i1 true, ptr %[[ACTIVE]]
// OGCG:         %[[GET_RESULT:.*]] = call {{.*}} i32 @_ZN1S3getEv(ptr {{.*}} %[[TMP]])
// OGCG:         br label %[[MERGE:.*]]
// OGCG:       [[FALSE_BR]]:
// OGCG:         %[[XVAL:.*]] = load i32, ptr %{{.*}}
// OGCG:         br label %[[MERGE]]
// OGCG:       [[MERGE]]:
// OGCG:         %[[COND:.*]] = phi i32 [ %[[GET_RESULT]], %[[TRUE_BR]] ], [ %[[XVAL]], %[[FALSE_BR]] ]
// OGCG:         br i1 %[[NEED_DTOR:.*]], label %[[CLEANUP_ACT:.*]], label %[[CLEANUP_DONE:.*]]
// OGCG:       [[CLEANUP_ACT]]:
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[TMP]])
// OGCG:         br label %[[CLEANUP_DONE]]
// OGCG:       [[CLEANUP_DONE]]:
// OGCG:         store i32 %[[COND]], ptr %{{.*}}

struct A {
  A();
  ~A();
  int get();
};

struct B {
  B();
  ~B();
  int get();
};

// Both branches of the ternary create different temporaries (A vs B).
// Each gets its own active flag; both are checked in the cleanup region.
void test_ternary_both_branches(bool c) {
  int result = c ? A().get() : B().get();
}
// CIR-LABEL: @_Z26test_ternary_both_branchesb
// CIR:   %[[TMPA:.*]] = cir.alloca !rec_A, !cir.ptr<!rec_A>, ["ref.tmp0"]
// CIR:   %[[ACTA:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.cond"]
// CIR:   %[[TMPB:.*]] = cir.alloca !rec_B, !cir.ptr<!rec_B>, ["ref.tmp1"]
// CIR:   %[[ACTB:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.cond"]
// CIR:   cir.cleanup.scope {
// Both active flags start false; each branch sets its own to true when it runs.
// CIR:     %[[COND:.*]] = cir.load {{.*}} : !cir.ptr<!cir.bool>, !cir.bool
// CIR:     %[[FALSE_A:.*]] = cir.const #false
// CIR:     cir.store %[[FALSE_A]], %[[ACTA]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:     %[[FALSE_B:.*]] = cir.const #false
// CIR:     cir.store %[[FALSE_B]], %[[ACTB]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:     %{{.*}} = cir.ternary(%[[COND]], true {
// CIR:       cir.call @_ZN1AC1Ev(%[[TMPA]])
// CIR:       %[[TRUE_A:.*]] = cir.const #true
// CIR:       cir.store %[[TRUE_A]], %[[ACTA]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:       %[[GET_A:.*]] = cir.call @_ZN1A3getEv(%[[TMPA]])
// CIR:       cir.yield %[[GET_A]] : !s32i
// CIR:     }, false {
// CIR:       cir.call @_ZN1BC1Ev(%[[TMPB]])
// CIR:       %[[TRUE_B:.*]] = cir.const #true
// CIR:       cir.store %[[TRUE_B]], %[[ACTB]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:       %[[GET_B:.*]] = cir.call @_ZN1B3getEv(%[[TMPB]])
// CIR:       cir.yield %[[GET_B]] : !s32i
// CIR:     cir.yield
// CIR:   } cleanup normal {
// CIR:     %[[FLAG_B:.*]] = cir.load {{.*}} %[[ACTB]]
// CIR:     cir.if %[[FLAG_B]] {
// CIR:       cir.call @_ZN1BD1Ev(%[[TMPB]])
// CIR:     }
// CIR:     %[[FLAG_A:.*]] = cir.load {{.*}} %[[ACTA]]
// CIR:     cir.if %[[FLAG_A]] {
// CIR:       cir.call @_ZN1AD1Ev(%[[TMPA]])
// CIR:     }
// CIR:     cir.yield
// CIR:   }

// LLVM-LABEL: define dso_local void @_Z26test_ternary_both_branchesb(
// LLVM:         %{{.*}} = alloca i8
// LLVM:         %{{.*}} = alloca i32
// LLVM:         %[[TMPA:.*]] = alloca %struct.A
// LLVM:         %[[ACTA:.*]] = alloca i8
// LLVM:         %[[TMPB:.*]] = alloca %struct.B
// LLVM:         %[[ACTB:.*]] = alloca i8
// LLVM:         %[[RESULT_TMP:.*]] = alloca i32
// LLVM:         br label %[[INIT:.*]]
// LLVM:       [[INIT]]:
// LLVM:         %[[COND_BYTE:.*]] = load i8, ptr %{{.*}}
// LLVM:         %[[COND_BOOL:.*]] = trunc i8 %[[COND_BYTE]] to i1
// LLVM:         store i8 0, ptr %[[ACTA]]
// LLVM:         store i8 0, ptr %[[ACTB]]
// LLVM:         br i1 %[[COND_BOOL]], label %[[CONSTRUCT_A:.*]], label %[[CONSTRUCT_B:.*]]
// LLVM:       [[CONSTRUCT_A]]:
// LLVM:         call void @_ZN1AC1Ev({{.*}} %[[TMPA]])
// LLVM:         store i8 1, ptr %[[ACTA]]
// LLVM:         %[[CALLA:.*]] = call noundef i32 @_ZN1A3getEv({{.*}} %[[TMPA]])
// LLVM:         br label %[[MERGE:.*]]
// LLVM:       [[CONSTRUCT_B]]:
// LLVM:         call void @_ZN1BC1Ev({{.*}} %[[TMPB]])
// LLVM:         store i8 1, ptr %[[ACTB]]
// LLVM:         %[[CALLB:.*]] = call {{.*}} i32 @_ZN1B3getEv({{.*}} %[[TMPB]])
// LLVM:         br label %[[MERGE]]
// LLVM:       [[MERGE]]:
// LLVM:         %[[PHI:.*]] = phi i32 [ %[[CALLB]], %[[CONSTRUCT_B]] ], [ %[[CALLA]], %[[CONSTRUCT_A]] ]
// LLVM:         br label %[[STORE:.*]]
// LLVM:       [[STORE]]:
// LLVM:         store i32 %[[PHI]], ptr %[[RESULT_TMP]]
// LLVM:         br label %[[CLEANUP_B:.*]]
// LLVM:       [[CLEANUP_B]]:
// LLVM:         %[[ACTIVE_BYTE_B:.*]] = load i8, ptr %[[ACTB]]
// LLVM:         %[[ACTIVE_BOOL_B:.*]] = trunc i8 %[[ACTIVE_BYTE_B]] to i1
// LLVM:         br i1 %[[ACTIVE_BOOL_B]], label %[[DTOR_B:.*]], label %[[SKIP_DTOR_B:.*]]
// LLVM:       [[DTOR_B]]:
// LLVM:         call void @_ZN1BD1Ev({{.*}} %[[TMPB]])
// LLVM:         br label %[[SKIP_DTOR_B]]
// LLVM:       [[SKIP_DTOR_B]]:
// LLVM:         %[[ACTIVE_BYTE_A:.*]] = load i8, ptr %[[ACTA]]
// LLVM:         %[[ACTIVE_BOOL_A:.*]] = trunc i8 %[[ACTIVE_BYTE_A]] to i1
// LLVM:         br i1 %[[ACTIVE_BOOL_A]], label %[[DTOR_A:.*]], label %[[SKIP_DTOR_A:.*]]
// LLVM:       [[DTOR_A]]:
// LLVM:         call void @_ZN1AD1Ev({{.*}} %[[TMPA]])
// LLVM:         br label %[[SKIP_DTOR_A]]
// LLVM:       [[SKIP_DTOR_A]]:
// LLVM:         br label %{{.*}}

// OGCG-LABEL: define dso_local void @_Z26test_ternary_both_branchesb(
// OGCG:       entry:
// OGCG:         store i1 false, ptr %[[ACTA:.*]]
// OGCG:         store i1 false, ptr %[[ACTB:.*]]
// OGCG:         br i1 %[[COND_BOOL:.*]], label %[[TRUE_BR:.*]], label %[[FALSE_BR:.*]]
// OGCG:       [[TRUE_BR]]:
// OGCG:         call void @_ZN1AC1Ev({{.*}} %[[TMPA:.*]])
// OGCG:         store i1 true, ptr %[[ACTA]]
// OGCG:         br label %[[MERGE:.*]]
// OGCG:       [[FALSE_BR]]:
// OGCG:         call void @_ZN1BC1Ev({{.*}} %[[TMPB:.*]])
// OGCG:         store i1 true, ptr %[[ACTB]]
// OGCG:         br label %[[MERGE]]
// OGCG:       [[MERGE]]:
// OGCG:         %[[COND:.*]] = phi i32 [ %{{.*}}, %[[TRUE_BR]] ], [ %{{.*}}, %[[FALSE_BR]] ]
// OGCG:         br i1 %[[ACTB:.*]], label %[[DTOR_B:.*]], label %[[AFTER_DTOR_B:.*]]
// OGCG:       [[DTOR_B]]:
// OGCG:         call void @_ZN1BD1Ev({{.*}} %[[TMPB]])
// OGCG:         br label %[[AFTER_DTOR_B]]
// OGCG:       [[AFTER_DTOR_B]]:
// OGCG:         br i1 %[[ACTA:.*]], label %[[DTOR_A:.*]], label %[[AFTER_DTOR_A:.*]]
// OGCG:       [[DTOR_A]]:
// OGCG:         call void @_ZN1AD1Ev({{.*}} %[[TMPA]])
// OGCG:         br label %[[AFTER_DTOR_A]]
// OGCG:       [[AFTER_DTOR_A]]:
// OGCG:         store i32 %[[COND]], ptr %{{.*}}

// Return expression with ternary: emitReturnStmt strips ExprWithCleanups but
// must still enter a full-expression cleanup scope for the conditional.
int test_return_ternary(bool c) {
  return c ? A().get() : B().get();
}
// CIR-LABEL: @_Z19test_return_ternaryb
// CIR:   %[[TMPA:.*]] = cir.alloca !rec_A, !cir.ptr<!rec_A>, ["ref.tmp0"]
// CIR:   %[[ACTA:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.cond"]
// CIR:   %[[TMPB:.*]] = cir.alloca !rec_B, !cir.ptr<!rec_B>, ["ref.tmp1"]
// CIR:   %[[ACTB:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.cond"]
// CIR:   cir.cleanup.scope {
// CIR:     %[[COND:.*]] = cir.load {{.*}} : !cir.ptr<!cir.bool>, !cir.bool
// CIR:     %[[FALSE_A:.*]] = cir.const #false
// CIR:     cir.store %[[FALSE_A]], %[[ACTA]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:     %[[FALSE_B:.*]] = cir.const #false
// CIR:     cir.store %[[FALSE_B]], %[[ACTB]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:     %{{.*}} = cir.ternary(%[[COND]], true {
// CIR:       cir.call @_ZN1AC1Ev(%[[TMPA]])
// CIR:       %[[TRUE_A:.*]] = cir.const #true
// CIR:       cir.store %[[TRUE_A]], %[[ACTA]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:       %[[GET_A:.*]] = cir.call @_ZN1A3getEv(%[[TMPA]])
// CIR:       cir.yield %[[GET_A]] : !s32i
// CIR:     }, false {
// CIR:       cir.call @_ZN1BC1Ev(%[[TMPB]])
// CIR:       %[[TRUE_B:.*]] = cir.const #true
// CIR:       cir.store %[[TRUE_B]], %[[ACTB]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:       %[[GET_B:.*]] = cir.call @_ZN1B3getEv(%[[TMPB]])
// CIR:       cir.yield %[[GET_B]] : !s32i
// CIR:     })
// The result is stored to __retval inside the cleanup scope body.
// CIR:     cir.store %{{.*}}, %{{.*}} : !s32i, !cir.ptr<!s32i>
// CIR:     cir.yield
// CIR:   } cleanup normal {
// CIR:     %[[FLAG_B:.*]] = cir.load {{.*}} %[[ACTB]]
// CIR:     cir.if %[[FLAG_B]] {
// CIR:       cir.call @_ZN1BD1Ev(%[[TMPB]])
// CIR:     }
// CIR:     %[[FLAG_A:.*]] = cir.load {{.*}} %[[ACTA]]
// CIR:     cir.if %[[FLAG_A]] {
// CIR:       cir.call @_ZN1AD1Ev(%[[TMPA]])
// CIR:     }
// CIR:     cir.yield
// CIR:   }
// Value loaded from __retval and returned.
// CIR:   %[[RET:.*]] = cir.load %{{.*}} : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[RET]] : !s32i

// LLVM-LABEL: define dso_local noundef i32 @_Z19test_return_ternaryb(
// LLVM:         %{{.*}} = alloca i8
// LLVM:         %[[RETVAL:.*]] = alloca i32
// LLVM:         %[[TMPA:.*]] = alloca %struct.A
// LLVM:         %[[ACTA:.*]] = alloca i8
// LLVM:         %[[TMPB:.*]] = alloca %struct.B
// LLVM:         %[[ACTB:.*]] = alloca i8
// LLVM:         br label %[[INIT:.*]]
// LLVM:       [[INIT]]:
// LLVM:         %[[COND_BYTE:.*]] = load i8, ptr %{{.*}}
// LLVM:         %[[COND_BOOL:.*]] = trunc i8 %[[COND_BYTE]] to i1
// LLVM:         store i8 0, ptr %[[ACTA]]
// LLVM:         store i8 0, ptr %[[ACTB]]
// LLVM:         br i1 %[[COND_BOOL]], label %[[CONSTRUCT_A:.*]], label %[[CONSTRUCT_B:.*]]
// LLVM:       [[CONSTRUCT_A]]:
// LLVM:         call void @_ZN1AC1Ev({{.*}} %[[TMPA]])
// LLVM:         store i8 1, ptr %[[ACTA]]
// LLVM:         %[[CALLA:.*]] = call noundef i32 @_ZN1A3getEv({{.*}} %[[TMPA]])
// LLVM:         br label %[[MERGE:.*]]
// LLVM:       [[CONSTRUCT_B]]:
// LLVM:         call void @_ZN1BC1Ev({{.*}} %[[TMPB]])
// LLVM:         store i8 1, ptr %[[ACTB]]
// LLVM:         %[[CALLB:.*]] = call noundef i32 @_ZN1B3getEv({{.*}} %[[TMPB]])
// LLVM:         br label %[[MERGE]]
// LLVM:       [[MERGE]]:
// LLVM:         %[[PHI:.*]] = phi i32 [ %[[CALLB]], %[[CONSTRUCT_B]] ], [ %[[CALLA]], %[[CONSTRUCT_A]] ]
// LLVM:         br label %[[STORE_RET:.*]]
// LLVM:       [[STORE_RET]]:
// LLVM:         store i32 %[[PHI]], ptr %[[RETVAL]]
// LLVM:         br label %[[CLEANUP_B:.*]]
// LLVM:       [[CLEANUP_B]]:
// LLVM:         %[[ACTIVE_BYTE_B:.*]] = load i8, ptr %[[ACTB]]
// LLVM:         %[[ACTIVE_BOOL_B:.*]] = trunc i8 %[[ACTIVE_BYTE_B]] to i1
// LLVM:         br i1 %[[ACTIVE_BOOL_B]], label %[[DTOR_B:.*]], label %[[SKIP_DTOR_B:.*]]
// LLVM:       [[DTOR_B]]:
// LLVM:         call void @_ZN1BD1Ev({{.*}} %[[TMPB]])
// LLVM:         br label %[[SKIP_DTOR_B]]
// LLVM:       [[SKIP_DTOR_B]]:
// LLVM:         %[[ACTIVE_BYTE_A:.*]] = load i8, ptr %[[ACTA]]
// LLVM:         %[[ACTIVE_BOOL_A:.*]] = trunc i8 %[[ACTIVE_BYTE_A]] to i1
// LLVM:         br i1 %[[ACTIVE_BOOL_A]], label %[[DTOR_A:.*]], label %[[SKIP_DTOR_A:.*]]
// LLVM:       [[DTOR_A]]:
// LLVM:         call void @_ZN1AD1Ev({{.*}} %[[TMPA]])
// LLVM:         br label %[[SKIP_DTOR_A]]
// LLVM:       [[SKIP_DTOR_A]]:
// LLVM:         br label %[[EXIT:.*]]
// LLVM:       [[EXIT]]:
// LLVM:         %[[RET:.*]] = load i32, ptr %[[RETVAL]]
// LLVM:         ret i32 %[[RET]]

// OGCG-LABEL: define dso_local noundef i32 @_Z19test_return_ternaryb(
// OGCG:       entry:
// OGCG:         store i1 false, ptr %[[ACTA:.*]]
// OGCG:         store i1 false, ptr %[[ACTB:.*]]
// OGCG:         br i1 %[[COND_BOOL:.*]], label %[[TRUE_BR:.*]], label %[[FALSE_BR:.*]]
// OGCG:       [[TRUE_BR]]:
// OGCG:         call void @_ZN1AC1Ev({{.*}} %[[TMPA:.*]])
// OGCG:         store i1 true, ptr %[[ACTA]]
// OGCG:         %[[CALLA:.*]] = call noundef i32 @_ZN1A3getEv({{.*}} %[[TMPA]])
// OGCG:         br label %[[MERGE:.*]]
// OGCG:       [[FALSE_BR]]:
// OGCG:         call void @_ZN1BC1Ev({{.*}} %[[TMPB:.*]])
// OGCG:         store i1 true, ptr %[[ACTB]]
// OGCG:         %[[CALLB:.*]] = call noundef i32 @_ZN1B3getEv({{.*}} %[[TMPB]])
// OGCG:         br label %[[MERGE]]
// OGCG:       [[MERGE]]:
// OGCG:         %[[COND:.*]] = phi i32 [ %[[CALLA]], %[[TRUE_BR]] ], [ %[[CALLB]], %[[FALSE_BR]] ]
// OGCG:         store i32 %[[COND]], ptr %{{.*}}
// OGCG:         br i1 %[[ACTB:.*]], label %[[DTOR_B:.*]], label %[[AFTER_DTOR_B:.*]]
// OGCG:       [[DTOR_B]]:
// OGCG:         call void @_ZN1BD1Ev({{.*}} %[[TMPB]])
// OGCG:         br label %[[AFTER_DTOR_B]]
// OGCG:       [[AFTER_DTOR_B]]:
// OGCG:         br i1 %[[ACTA:.*]], label %[[DTOR_A:.*]], label %[[AFTER_DTOR_A:.*]]
// OGCG:       [[DTOR_A]]:
// OGCG:         call void @_ZN1AD1Ev({{.*}} %[[TMPA]])
// OGCG:         br label %[[AFTER_DTOR_A]]
// OGCG:       [[AFTER_DTOR_A]]:
// OGCG:         %{{.*}} = load i32, ptr %{{.*}}
// OGCG:         ret i32 %{{.*}}

// False positive: ExprWithCleanups wraps a ternary, but S() is constructed
// outside the conditional so no cleanup is deferred. The eagerly-created
// full-expression cir.cleanup.scope is inlined and erased, leaving only
// the LexicalScope cleanup for S()'s destructor.
// CIR-LABEL: @_Z31test_false_positive_conditionalb
int test_false_positive_conditional(bool c) {
  return S().get() ? 1 : 2;
}
// No cleanup.cond alloca — the destructor is unconditional.
// CIR-NOT:   cir.alloca {{.*}} ["cleanup.cond"]
// CIR:   %[[TMP:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["ref.tmp0"]
// CIR:   cir.call @_ZN1SC1Ev(%[[TMP]])
// The cleanup scope wraps the get() + select + store.
// CIR:   cir.cleanup.scope {
// CIR:     %[[VAL:.*]] = cir.call @_ZN1S3getEv(%[[TMP]])
// CIR:     %[[BOOL:.*]] = cir.cast int_to_bool %[[VAL]]
// No cir.ternary — both arms are constants, so this lowers to cir.select.
// CIR:     %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR:     %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CIR:     %[[SEL:.*]] = cir.select if %[[BOOL]] then %[[ONE]] else %[[TWO]]
// CIR:     cir.store %[[SEL]], %{{.*}} : !s32i, !cir.ptr<!s32i>
// CIR:     cir.yield
// S destructor runs unconditionally — no active-flag guard.
// CIR:   } cleanup normal {
// CIR:     cir.call @_ZN1SD1Ev(%[[TMP]])
// CIR:     cir.yield
// CIR:   }

// LLVM-LABEL: define dso_local noundef i32 @_Z31test_false_positive_conditionalb(
// LLVM:         %[[RETVAL:.*]] = alloca i32
// LLVM:         %[[TMP:.*]] = alloca %struct.S
// LLVM:         call void @_ZN1SC1Ev({{.*}} %[[TMP]])
// LLVM:         br label %[[BODY:.*]]
// LLVM:       [[BODY]]:
// LLVM:         %[[VAL:.*]] = call {{.*}} i32 @_ZN1S3getEv({{.*}} %[[TMP]])
// LLVM:         %[[CMP:.*]] = icmp ne i32 %[[VAL]], 0
// LLVM:         %[[SEL:.*]] = select i1 %[[CMP]], i32 1, i32 2
// LLVM:         store i32 %[[SEL]], ptr %[[RETVAL]]
// LLVM:         br label %[[DTOR:.*]]
// LLVM:       [[DTOR]]:
// LLVM:         call void @_ZN1SD1Ev({{.*}} %[[TMP]])
// LLVM:         br label %[[EXIT:.*]]
// LLVM:       [[EXIT]]:
// LLVM:         %[[RET:.*]] = load i32, ptr %[[RETVAL]]
// LLVM:         ret i32 %[[RET]]

// OGCG-LABEL: define dso_local noundef i32 @_Z31test_false_positive_conditionalb(
// OGCG:         call void @_ZN1SC1Ev({{.*}} %[[TMP:.*]])
// OGCG:         %[[VAL:.*]] = call {{.*}} i32 @_ZN1S3getEv({{.*}} %[[TMP]])
// OGCG:         %[[CMP:.*]] = icmp ne i32 %[[VAL]], 0
// OGCG:         %[[SEL:.*]] = select i1 %[[CMP]], i32 1, i32 2
// OGCG:         call void @_ZN1SD1Ev({{.*}} %[[TMP]])
// OGCG:         ret i32 %[[SEL]]

// Test nested ExprWithCleanups nodes, each containing a ternary operator.
//
// The outer ExprWithCleanups wraps the full-expression
//   `S result = ({...}) ? (...) : S(5);`
// The inner ExprWithCleanups wraps the variable initializer
//   `S s = c1 ? S(1) : S(2);`
// inside the statement expression, which is its own full-expression context.
//
// Both contain ConditionalOperators — exercising the save/restore of
// fullExprCleanupScope state.

struct T {
  T();
  T(int);
  T(const T &);
  ~T();
  operator bool();
};

void test_nested_ewc(bool c1, bool c2) {
  T result = ({ T s = c1 ? T(1) : T(2); s; }) ? (c2 ? T(3) : T(4))
                                                : T(5);
}

// CIR-LABEL: @_Z15test_nested_ewcbb
// CIR:   %[[RESULT:.*]] = cir.alloca !rec_T, !cir.ptr<!rec_T>, ["result", init]
// CIR:   %[[REF_TMP:.*]] = cir.alloca !rec_T, !cir.ptr<!rec_T>, ["ref.tmp0"]
// cir.scope for the statement expression.
// CIR:   cir.scope {
// CIR:     %[[S:.*]] = cir.alloca !rec_T, !cir.ptr<!rec_T>, ["s", init]
// Inner ternary: c1 ? T(1) : T(2) — no cleanup scope needed (no deferred dtors).
// CIR:     %[[C1:.*]] = cir.load {{.*}} : !cir.ptr<!cir.bool>, !cir.bool
// CIR:     cir.if %[[C1]] {
// CIR:       %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR:       cir.call @_ZN1TC1Ei(%[[S]], %[[ONE]])
// CIR:     } else {
// CIR:       %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CIR:       cir.call @_ZN1TC1Ei(%[[S]], %[[TWO]])
// CIR:     }
// Statement expression result: copy s into ref.tmp, then destroy s.
// CIR:     cir.cleanup.scope {
// CIR:       cir.call @_ZN1TC1ERKS_(%[[REF_TMP]], %[[S]])
// CIR:       cir.yield
// CIR:     } cleanup normal {
// CIR:       cir.call @_ZN1TD1Ev(%[[S]])
// CIR:       cir.yield
// CIR:     }
// CIR:   }
// Cleanup scope: wraps operator bool() + outer ternary + destroys ref.tmp.
// CIR:   cir.cleanup.scope {
// CIR:     %[[BOOL:.*]] = cir.call @_ZN1TcvbEv(%[[REF_TMP]])
// CIR:     cir.if %[[BOOL]] {
// CIR:       %[[C2:.*]] = cir.load {{.*}} : !cir.ptr<!cir.bool>, !cir.bool
// CIR:       cir.if %[[C2]] {
// CIR:         %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
// CIR:         cir.call @_ZN1TC1Ei(%[[RESULT]], %[[THREE]])
// CIR:       } else {
// CIR:         %[[FOUR:.*]] = cir.const #cir.int<4> : !s32i
// CIR:         cir.call @_ZN1TC1Ei(%[[RESULT]], %[[FOUR]])
// CIR:       }
// CIR:     } else {
// CIR:       %[[FIVE:.*]] = cir.const #cir.int<5> : !s32i
// CIR:       cir.call @_ZN1TC1Ei(%[[RESULT]], %[[FIVE]])
// CIR:     }
// CIR:     cir.yield
// CIR:   } cleanup normal {
// CIR:     cir.call @_ZN1TD1Ev(%[[REF_TMP]])
// CIR:     cir.yield
// CIR:   }
// result destructor runs unconditionally after the outer scope.
// CIR:   cir.cleanup.scope {
// CIR:     cir.yield
// CIR:   } cleanup normal {
// CIR:     cir.call @_ZN1TD1Ev(%[[RESULT]])
// CIR:     cir.yield
// CIR:   }

// LLVM-LABEL: define dso_local void @_Z15test_nested_ewcbb(
// Inner ternary: c1 ? T(1) : T(2).
// LLVM:         br i1 %{{.*}}, label %[[T1:.*]], label %[[T2:.*]]
// LLVM:       [[T1]]:
// LLVM:         call void @_ZN1TC1Ei({{.*}} %[[S:.*]], i32 {{.*}} 1)
// LLVM:         br label %[[INNER_MERGE:.*]]
// LLVM:       [[T2]]:
// LLVM:         call void @_ZN1TC1Ei({{.*}} %[[S]], i32 {{.*}} 2)
// LLVM:         br label %[[INNER_MERGE]]
// Copy construct ref.tmp from s, then destroy s.
// LLVM:       [[INNER_MERGE]]:
// LLVM:         call void @_ZN1TC1ERKS_({{.*}} %[[REF_TMP:.*]], {{.*}} %[[S]])
// LLVM:         call void @_ZN1TD1Ev({{.*}} %[[S]])
// Outer ternary: operator bool() on ref.tmp.
// LLVM:         %[[BOOL:.*]] = call {{.*}} i1 @_ZN1TcvbEv({{.*}} %[[REF_TMP]])
// LLVM:         br i1 %[[BOOL]], label %[[TRUE:.*]], label %[[FALSE:.*]]
// LLVM:       [[TRUE]]:
// LLVM:         br i1 %{{.*}}, label %[[T3:.*]], label %[[T4:.*]]
// LLVM:       [[T3]]:
// LLVM:         call void @_ZN1TC1Ei({{.*}} %[[RESULT:.*]], i32 {{.*}} 3)
// LLVM:         br label %[[OUTER_MERGE1:.*]]
// LLVM:       [[T4]]:
// LLVM:         call void @_ZN1TC1Ei({{.*}} %[[RESULT]], i32 {{.*}} 4)
// LLVM:         br label %[[OUTER_MERGE1]]
// LLVM:       [[OUTER_MERGE1]]:
// LLVM:         br label %[[OUTER_MERGE2:.*]]
// LLVM:       [[FALSE]]:
// LLVM:         call void @_ZN1TC1Ei({{.*}} %[[RESULT]], i32 {{.*}} 5)
// LLVM:         br label %[[OUTER_MERGE2]]
// Cleanup: destroy ref.tmp, then result.
// LLVM:       [[OUTER_MERGE2]]:
// LLVM:         call void @_ZN1TD1Ev({{.*}} %[[REF_TMP]])
// LLVM:         call void @_ZN1TD1Ev({{.*}} %[[RESULT]])

// OGCG-LABEL: define dso_local void @_Z15test_nested_ewcbb(
// Inner ternary: c1 ? T(1) : T(2).
// OGCG:         br i1 %{{.*}}, label %[[T1:.*]], label %[[T2:.*]]
// OGCG:       [[T1]]:
// OGCG:         call void @_ZN1TC1Ei({{.*}} %[[S:.*]], i32 {{.*}} 1)
// OGCG:         br label %[[INNER_MERGE:.*]]
// OGCG:       [[T2]]:
// OGCG:         call void @_ZN1TC1Ei({{.*}} %[[S]], i32 {{.*}} 2)
// OGCG:         br label %[[INNER_MERGE]]
// Copy construct ref.tmp from s, then destroy s.
// OGCG:       [[INNER_MERGE]]:
// OGCG:         call void @_ZN1TC1ERKS_({{.*}} %[[REF_TMP:.*]], {{.*}} %[[S]])
// OGCG:         call void @_ZN1TD1Ev({{.*}} %[[S]])
// Outer ternary: operator bool() + conditional construction of result.
// OGCG:         %[[BOOL:.*]] = call {{.*}} i1 @_ZN1TcvbEv({{.*}} %[[REF_TMP]])
// OGCG:         br i1 %[[BOOL]], label %[[TRUE:.*]], label %[[FALSE:.*]]
// OGCG:       [[TRUE]]:
// OGCG:         br i1 %{{.*}}, label %[[T3:.*]], label %[[T4:.*]]
// OGCG:       [[T3]]:
// OGCG:         call void @_ZN1TC1Ei({{.*}} %[[RESULT:.*]], i32 {{.*}} 3)
// OGCG:         br label %[[OUTER_MERGE1:.*]]
// OGCG:       [[T4]]:
// OGCG:         call void @_ZN1TC1Ei({{.*}} %[[RESULT]], i32 {{.*}} 4)
// OGCG:         br label %[[OUTER_MERGE1]]
// OGCG:       [[OUTER_MERGE1]]:
// OGCG:         br label %[[OUTER_MERGE2:.*]]
// OGCG:       [[FALSE]]:
// OGCG:         call void @_ZN1TC1Ei({{.*}} %[[RESULT]], i32 {{.*}} 5)
// OGCG:         br label %[[OUTER_MERGE2]]
// Cleanup: destroy ref.tmp, then result.
// OGCG:       [[OUTER_MERGE2]]:
// OGCG:         call void @_ZN1TD1Ev({{.*}} %[[REF_TMP]])
// OGCG:         call void @_ZN1TD1Ev({{.*}} %[[RESULT]])

// The result of the ternary is bound to an lvalue (the parameter of
// operator=), so the enclosing ExprWithCleanups is lowered through the
// LValue emission path.  The lvalue path must still open a
// FullExprCleanupScope so that conditional cleanups deferred by the
// ternary (here, the D temporary created by the default argument inside
// each branch) are consumed before the full-expression boundary.
struct U {
  U();
  ~U();
};

struct V {
  V(int, const U & = U());
  ~V();
  V &operator=(const V &);
};

void test_lvalue_ternary_cleanup(bool c, V &y) {
  y = c ? V(1) : V(2);
}
// CIR-LABEL: @_Z27test_lvalue_ternary_cleanupbR1V
// CIR:   %[[REFTMP:.*]] = cir.alloca !rec_V, !cir.ptr<!rec_V>, ["ref.tmp0"]
// CIR:   %[[UTRUE:.*]] = cir.alloca !rec_U, !cir.ptr<!rec_U>, ["ref.tmp1"]
// CIR:   %[[ACTTRUE:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.cond"]
// CIR:   %[[UFALSE:.*]] = cir.alloca !rec_U, !cir.ptr<!rec_U>, ["ref.tmp2"]
// CIR:   %[[ACTFALSE:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.cond"]
// The outer cleanup scope wraps the full expression containing the ternary
// and the operator= call.
// CIR:   cir.cleanup.scope {
// Both cleanup flags initialized to false before the ternary.
// CIR:     cir.store {{.*}}, %[[ACTTRUE]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:     cir.store {{.*}}, %[[ACTFALSE]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:     cir.if %{{.*}} {
// CIR:       cir.call @_ZN1UC1Ev(%[[UTRUE]])
// CIR:       cir.store {{.*}}, %[[ACTTRUE]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:       cir.call @_ZN1VC1EiRK1U(%[[REFTMP]], %{{.*}}, %[[UTRUE]])
// CIR:     } else {
// CIR:       cir.call @_ZN1UC1Ev(%[[UFALSE]])
// CIR:       cir.store {{.*}}, %[[ACTFALSE]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:       cir.call @_ZN1VC1EiRK1U(%[[REFTMP]], %{{.*}}, %[[UFALSE]])
// CIR:     }
// Inner cleanup scope for the operator= call destroys the ref.tmp.
// CIR:     cir.cleanup.scope {
// CIR:       cir.call @_ZN1VaSERKS_(%{{.*}}, %[[REFTMP]])
// CIR:       cir.yield
// CIR:     } cleanup normal {
// CIR:       cir.call @_ZN1VD1Ev(%[[REFTMP]])
// CIR:       cir.yield
// CIR:     }
// CIR:     cir.yield
// Outer cleanup region: conditionally destroy U temporaries by active flag.
// CIR:   } cleanup normal {
// CIR:     %[[F2:.*]] = cir.load {{.*}} %[[ACTFALSE]]
// CIR:     cir.if %[[F2]] {
// CIR:       cir.call @_ZN1UD1Ev(%[[UFALSE]])
// CIR:     }
// CIR:     %[[F1:.*]] = cir.load {{.*}} %[[ACTTRUE]]
// CIR:     cir.if %[[F1]] {
// CIR:       cir.call @_ZN1UD1Ev(%[[UTRUE]])
// CIR:     }
// CIR:     cir.yield
// CIR:   }

// LLVM-LABEL: define dso_local void @_Z27test_lvalue_ternary_cleanupbR1V(
// LLVM:         %[[REFTMP:.*]] = alloca %struct.V
// LLVM:         %[[UTRUE:.*]] = alloca %struct.U
// LLVM:         %[[ACTTRUE:.*]] = alloca i8
// LLVM:         %[[UFALSE:.*]] = alloca %struct.U
// LLVM:         %[[ACTFALSE:.*]] = alloca i8
// LLVM:         store i8 0, ptr %[[ACTTRUE]]
// LLVM:         store i8 0, ptr %[[ACTFALSE]]
// LLVM:         br i1 %{{.*}}, label %[[CONS_TRUE:.*]], label %[[CONS_FALSE:.*]]
// LLVM:       [[CONS_TRUE]]:
// LLVM:         call void @_ZN1UC1Ev({{.*}} %[[UTRUE]])
// LLVM:         store i8 1, ptr %[[ACTTRUE]]
// LLVM:         call void @_ZN1VC1EiRK1U({{.*}} %[[REFTMP]], i32 {{.*}} 1, {{.*}} %[[UTRUE]])
// LLVM:         br label %[[MERGE:.*]]
// LLVM:       [[CONS_FALSE]]:
// LLVM:         call void @_ZN1UC1Ev({{.*}} %[[UFALSE]])
// LLVM:         store i8 1, ptr %[[ACTFALSE]]
// LLVM:         call void @_ZN1VC1EiRK1U({{.*}} %[[REFTMP]], i32 {{.*}} 2, {{.*}} %[[UFALSE]])
// LLVM:         br label %[[MERGE]]
// LLVM:       [[MERGE]]:
// LLVM:         call {{.*}} ptr @_ZN1VaSERKS_({{.*}}, {{.*}} %[[REFTMP]])
// LLVM:         call void @_ZN1VD1Ev({{.*}} %[[REFTMP]])
// LLVM:         %[[F2_BYTE:.*]] = load i8, ptr %[[ACTFALSE]]
// LLVM:         %[[F2:.*]] = trunc i8 %[[F2_BYTE]] to i1
// LLVM:         br i1 %[[F2]], label %[[DTOR_F:.*]], label %[[SKIP_F:.*]]
// LLVM:       [[DTOR_F]]:
// LLVM:         call void @_ZN1UD1Ev({{.*}} %[[UFALSE]])
// LLVM:         br label %[[SKIP_F]]
// LLVM:       [[SKIP_F]]:
// LLVM:         %[[F1_BYTE:.*]] = load i8, ptr %[[ACTTRUE]]
// LLVM:         %[[F1:.*]] = trunc i8 %[[F1_BYTE]] to i1
// LLVM:         br i1 %[[F1]], label %[[DTOR_T:.*]], label %[[SKIP_T:.*]]
// LLVM:       [[DTOR_T]]:
// LLVM:         call void @_ZN1UD1Ev({{.*}} %[[UTRUE]])
// LLVM:         br label %[[SKIP_T]]

// OGCG-LABEL: define dso_local void @_Z27test_lvalue_ternary_cleanupbR1V(
// OGCG:         store i1 false, ptr %[[ACTTRUE:.*]]
// OGCG:         store i1 false, ptr %[[ACTFALSE:.*]]
// OGCG:         br i1 %{{.*}}, label %[[CTRUE:.*]], label %[[CFALSE:.*]]
// OGCG:       [[CTRUE]]:
// OGCG:         call void @_ZN1UC1Ev({{.*}} %[[UTRUE:.*]])
// OGCG:         store i1 true, ptr %[[ACTTRUE]]
// OGCG:         call void @_ZN1VC1EiRK1U({{.*}} %[[REFTMP:.*]], i32 {{.*}} 1, {{.*}} %[[UTRUE]])
// OGCG:         br label %[[MERGE:.*]]
// OGCG:       [[CFALSE]]:
// OGCG:         call void @_ZN1UC1Ev({{.*}} %[[UFALSE:.*]])
// OGCG:         store i1 true, ptr %[[ACTFALSE]]
// OGCG:         call void @_ZN1VC1EiRK1U({{.*}} %[[REFTMP]], i32 {{.*}} 2, {{.*}} %[[UFALSE]])
// OGCG:         br label %[[MERGE]]
// OGCG:       [[MERGE]]:
// OGCG:         call {{.*}} ptr @_ZN1VaSERKS_({{.*}}, {{.*}} %[[REFTMP]])
// OGCG:         call void @_ZN1VD1Ev({{.*}} %[[REFTMP]])
// OGCG:         br i1 %{{.*}}, label %[[DTOR_F:.*]], label %[[AFTER_F:.*]]
// OGCG:       [[DTOR_F]]:
// OGCG:         call void @_ZN1UD1Ev({{.*}} %[[UFALSE]])
// OGCG:         br label %[[AFTER_F]]
// OGCG:       [[AFTER_F]]:
// OGCG:         br i1 %{{.*}}, label %[[DTOR_T:.*]], label %[[AFTER_T:.*]]
// OGCG:       [[DTOR_T]]:
// OGCG:         call void @_ZN1UD1Ev({{.*}} %[[UTRUE]])
// OGCG:         br label %[[AFTER_T]]

// When an ExprWithCleanups produces an lvalue whose base pointer is computed
// *inside* the FullExprCleanupScope (here, via the `.field` GEP on the
// conditional's lvalue result), the lvalue path must spill the base pointer
// before the scope closes and reload it afterward so that uses outside the
// scope (the reference binding to `r`) see a dominating SSA value.
struct R {
  R(int);
  R(const R &);
  ~R();
  int field;
};

R &pickR(const R &x);

int *sink;

void test_lvalue_reload(bool c) {
  int &r = (c ? pickR(R(1)) : pickR(R(2))).field;
  sink = &r;
}
// CIR-LABEL: @_Z18test_lvalue_reloadb
// CIR:   %[[R_REF:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["r", init, const]
// CIR:   %[[TMP0:.*]] = cir.alloca !rec_R, !cir.ptr<!rec_R>, ["ref.tmp0"]
// CIR:   %[[ACT0:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.cond"]
// CIR:   %[[TMP1:.*]] = cir.alloca !rec_R, !cir.ptr<!rec_R>, ["ref.tmp1"]
// CIR:   %[[ACT1:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.cond"]
// The spill slot for the lvalue's base pointer.
// CIR:   %[[SPILL:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["tmp.exprcleanup"]
// CIR:   cir.cleanup.scope {
// CIR:     cir.store {{.*}}, %[[ACT0]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:     cir.store {{.*}}, %[[ACT1]] : !cir.bool, !cir.ptr<!cir.bool>
// The ternary returns an !cir.ptr<!rec_R> lvalue from pickR().
// CIR:     %[[TERN:.*]] = cir.ternary({{.*}}, true {
// CIR:       cir.call @_ZN1RC1Ei(%[[TMP0]], %{{.*}})
// CIR:       cir.store {{.*}}, %[[ACT0]]
// CIR:       %[[CALL_T:.*]] = cir.call @_Z5pickRRK1R(%[[TMP0]])
// CIR:       cir.yield %[[CALL_T]] : !cir.ptr<!rec_R>
// CIR:     }, false {
// CIR:       cir.call @_ZN1RC1Ei(%[[TMP1]], %{{.*}})
// CIR:       cir.store {{.*}}, %[[ACT1]]
// CIR:       %[[CALL_F:.*]] = cir.call @_Z5pickRRK1R(%[[TMP1]])
// CIR:       cir.yield %[[CALL_F]] : !cir.ptr<!rec_R>
// CIR:     }) : (!cir.bool) -> !cir.ptr<!rec_R>
// `.field` GEP and its spill happen inside the cleanup scope body.
// CIR:     %[[GEP:.*]] = cir.get_member %[[TERN]][0] {name = "field"} : !cir.ptr<!rec_R> -> !cir.ptr<!s32i>
// CIR:     cir.store {{.*}} %[[GEP]], %[[SPILL]]
// CIR:     cir.yield
// CIR:   } cleanup normal {
// CIR:     %[[F1:.*]] = cir.load {{.*}} %[[ACT1]]
// CIR:     cir.if %[[F1]] {
// CIR:       cir.call @_ZN1RD1Ev(%[[TMP1]])
// CIR:     }
// CIR:     %[[F0:.*]] = cir.load {{.*}} %[[ACT0]]
// CIR:     cir.if %[[F0]] {
// CIR:       cir.call @_ZN1RD1Ev(%[[TMP0]])
// CIR:     }
// CIR:     cir.yield
// CIR:   }
// Reload happens after the cleanup scope; the reloaded pointer initializes r.
// CIR:   %[[RELOAD:.*]] = cir.load {{.*}} %[[SPILL]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR:   cir.store {{.*}} %[[RELOAD]], %[[R_REF]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>

// LLVM-LABEL: define dso_local void @_Z18test_lvalue_reloadb(
// LLVM:         %[[R_REF:.*]] = alloca ptr
// LLVM:         %[[TMP0:.*]] = alloca %struct.R
// LLVM:         %[[ACT0:.*]] = alloca i8
// LLVM:         %[[TMP1:.*]] = alloca %struct.R
// LLVM:         %[[ACT1:.*]] = alloca i8
// LLVM:         %[[SPILL:.*]] = alloca ptr
// LLVM:         br i1 %{{.*}}, label %[[BR_T:.*]], label %[[BR_F:.*]]
// LLVM:       [[BR_T]]:
// LLVM:         call void @_ZN1RC1Ei({{.*}} %[[TMP0]], i32 {{.*}} 1)
// LLVM:         store i8 1, ptr %[[ACT0]]
// LLVM:         %[[CALL_T:.*]] = call {{.*}} ptr @_Z5pickRRK1R({{.*}} %[[TMP0]])
// LLVM:         br label %[[MERGE:.*]]
// LLVM:       [[BR_F]]:
// LLVM:         call void @_ZN1RC1Ei({{.*}} %[[TMP1]], i32 {{.*}} 2)
// LLVM:         store i8 1, ptr %[[ACT1]]
// LLVM:         %[[CALL_F:.*]] = call {{.*}} ptr @_Z5pickRRK1R({{.*}} %[[TMP1]])
// LLVM:         br label %[[MERGE]]
// LLVM:       [[MERGE]]:
// LLVM:         %[[PHI:.*]] = phi ptr [ %[[CALL_F]], %[[BR_F]] ], [ %[[CALL_T]], %[[BR_T]] ]
// LLVM:         %[[GEP:.*]] = getelementptr {{.*}} %struct.R, ptr %[[PHI]]
// LLVM:         store ptr %[[GEP]], ptr %[[SPILL]]
// Cleanup checks happen between the spill and the reload.
// LLVM:         load i8, ptr %[[ACT1]]
// LLVM:         load i8, ptr %[[ACT0]]
// LLVM:         %[[RELOAD:.*]] = load ptr, ptr %[[SPILL]]
// LLVM:         store ptr %[[RELOAD]], ptr %[[R_REF]]

// OGCG-LABEL: define dso_local void @_Z18test_lvalue_reloadb(
// OGCG:         %[[R_REF:.*]] = alloca ptr
// OGCG:         br i1 %{{.*}}, label %[[BR_T:.*]], label %[[BR_F:.*]]
// OGCG:       [[BR_T]]:
// OGCG:         call void @_ZN1RC1Ei({{.*}} %[[TMP0:.*]], i32 {{.*}} 1)
// OGCG:         %[[CALL_T:.*]] = call {{.*}} ptr @_Z5pickRRK1R({{.*}} %[[TMP0]])
// OGCG:         br label %[[MERGE:.*]]
// OGCG:       [[BR_F]]:
// OGCG:         call void @_ZN1RC1Ei({{.*}} %[[TMP1:.*]], i32 {{.*}} 2)
// OGCG:         %[[CALL_F:.*]] = call {{.*}} ptr @_Z5pickRRK1R({{.*}} %[[TMP1]])
// OGCG:         br label %[[MERGE]]
// OGCG:       [[MERGE]]:
// OGCG:         %[[PHI:.*]] = phi ptr [ %[[CALL_T]], %[[BR_T]] ], [ %[[CALL_F]], %[[BR_F]] ]
// OGCG:         %[[GEP:.*]] = getelementptr {{.*}} %struct.R, ptr %[[PHI]]
// Classic codegen uses the phi-merged pointer directly; the cleanups run, and
// then the lvalue address is stored into r.
// OGCG:         call void @_ZN1RD1Ev({{.*}} %[[TMP1]])
// OGCG:         call void @_ZN1RD1Ev({{.*}} %[[TMP0]])
// OGCG:         store ptr %[[GEP]], ptr %[[R_REF]]

// When the result of an ExprWithCleanups is a _Complex value, the complex
// emitter must use FullExprCleanupScope so that conditional cleanups deferred
// by the inner conditional operator are consumed at the full-expression
// boundary.  Without this, the destructor cleanup for the temporary `D` in the
// true branch would remain on the deferredConditionalCleanupStack and trip
// the assertion in finishFunction.
struct CplxD {
  CplxD();
  ~CplxD();
  _Complex float get();
};

_Complex float test_complex_cond_cleanup(bool b, _Complex float x) {
  return b ? CplxD().get() : x;
}
// CIR-LABEL: @_Z25test_complex_cond_cleanupbCf
// CIR:   %[[TMP:.*]] = cir.alloca !rec_CplxD, !cir.ptr<!rec_CplxD>, ["ref.tmp0"]
// CIR:   %[[ACTIVE:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.cond"]
// The full expression is wrapped in a single cleanup scope.
// CIR:   cir.cleanup.scope {
// CIR:     %[[COND:.*]] = cir.load {{.*}} : !cir.ptr<!cir.bool>, !cir.bool
// Active flag is initialized to false before the ternary so the dtor only runs
// when the true branch was actually taken.
// CIR:     %[[FALSE:.*]] = cir.const #false
// CIR:     cir.store %[[FALSE]], %[[ACTIVE]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:     %{{.*}} = cir.ternary(%[[COND]], true {
// CIR:       cir.call @_ZN5CplxDC1Ev(%[[TMP]])
// CIR:       %[[SET_TRUE:.*]] = cir.const #true
// CIR:       cir.store %[[SET_TRUE]], %[[ACTIVE]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:       %[[CALL:.*]] = cir.call @_ZN5CplxD3getEv(%[[TMP]])
// CIR:       cir.yield %[[CALL]] : !cir.complex<!cir.float>
// CIR:     }, false {
// CIR:       %[[XV:.*]] = cir.load {{.*}} : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR:       cir.yield %[[XV]] : !cir.complex<!cir.float>
// CIR:     }) : (!cir.bool) -> !cir.complex<!cir.float>
// CIR:     cir.yield
// CIR:   } cleanup normal {
// CIR:     %[[IS_ACTIVE:.*]] = cir.load {{.*}} %[[ACTIVE]]
// CIR:     cir.if %[[IS_ACTIVE]] {
// CIR:       cir.call @_ZN5CplxDD1Ev(%[[TMP]])
// CIR:     }
// CIR:     cir.yield
// CIR:   }

// LLVM-LABEL: define dso_local {{.*}} { float, float } @_Z25test_complex_cond_cleanupbCf(
// LLVM:         %[[TMP:.*]] = alloca %struct.CplxD
// LLVM:         %[[ACTIVE:.*]] = alloca i8
// LLVM:         br i1 %{{.*}}, label %[[TRUE_BR:.*]], label %[[FALSE_BR:.*]]
// LLVM:       [[TRUE_BR]]:
// LLVM:         call void @_ZN5CplxDC1Ev(ptr {{.*}} %[[TMP]])
// LLVM:         store i8 1, ptr %[[ACTIVE]]
// LLVM:         %[[CALL:.*]] = call {{.*}} { float, float } @_ZN5CplxD3getEv(ptr {{.*}} %[[TMP]])
// LLVM:         br label %[[MERGE:.*]]
// LLVM:       [[FALSE_BR]]:
// LLVM:         %[[XV:.*]] = load { float, float }, ptr %{{.*}}
// LLVM:         br label %[[MERGE]]
// LLVM:       [[MERGE]]:
// LLVM:         %{{.*}} = phi { float, float } [ %[[XV]], %[[FALSE_BR]] ], [ %[[CALL]], %[[TRUE_BR]] ]
// LLVM:         %[[ACT:.*]] = load i8, ptr %[[ACTIVE]]
// LLVM:         %[[ACT_B:.*]] = trunc i8 %[[ACT]] to i1
// LLVM:         br i1 %[[ACT_B]], label %[[DTOR:.*]], label %[[SKIP:.*]]
// LLVM:       [[DTOR]]:
// LLVM:         call void @_ZN5CplxDD1Ev(ptr {{.*}} %[[TMP]])
// LLVM:         br label %[[SKIP]]

// OGCG-LABEL: define dso_local {{.*}} <2 x float> @_Z25test_complex_cond_cleanupbCf(
// OGCG:         %[[TMP:.*]] = alloca %struct.CplxD
// OGCG:         %[[ACTIVE:.*]] = alloca i1
// OGCG:         store i1 false, ptr %[[ACTIVE]]
// OGCG:         br i1 %{{.*}}, label %[[CTRUE:.*]], label %[[CFALSE:.*]]
// OGCG:       [[CTRUE]]:
// OGCG:         call void @_ZN5CplxDC1Ev(ptr {{.*}} %[[TMP]])
// OGCG:         store i1 true, ptr %[[ACTIVE]]
// OGCG:         %[[CALL:.*]] = call {{.*}} <2 x float> @_ZN5CplxD3getEv(ptr {{.*}} %[[TMP]])
// OGCG:         br label %[[MERGE:.*]]
// OGCG:       [[CFALSE]]:
// OGCG:         br label %[[MERGE]]
// OGCG:       [[MERGE]]:
// OGCG:         %[[ACT:.*]] = load i1, ptr %[[ACTIVE]]
// OGCG:         br i1 %[[ACT]], label %[[DTOR:.*]], label %[[DONE:.*]]
// OGCG:       [[DTOR]]:
// OGCG:         call void @_ZN5CplxDD1Ev(ptr {{.*}} %[[TMP]])
// OGCG:         br label %[[DONE]]
