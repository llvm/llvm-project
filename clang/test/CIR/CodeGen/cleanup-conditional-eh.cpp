// Exceptions-enabled variant of cleanup-conditional.cpp.
// When -fcxx-exceptions is active, cleanup scopes use "cleanup all" (both
// normal and exception paths) and the LLVM lowering emits invoke/landingpad
// instead of plain calls for operations that can throw.
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -fcxx-exceptions -fexceptions %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -fcxx-exceptions -fexceptions %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fcxx-exceptions -fexceptions %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

struct S {
  S();
  ~S();
  int get();
};

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

void test_ternary_temporary(bool c, int x) {
  int result = c ? S().get() : x;
}
// CIR-LABEL: @_Z22test_ternary_temporarybi
// CIR:   %[[TMP:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["ref.tmp0"]
// CIR:   %[[ACTIVE:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.cond"]
// CIR:   cir.cleanup.scope {
// CIR:     %[[COND:.*]] = cir.load {{.*}} : !cir.ptr<!cir.bool>, !cir.bool
// CIR:     %[[FALSE:.*]] = cir.const #false
// CIR:     cir.store %[[FALSE]], %[[ACTIVE]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:     %{{.*}} = cir.ternary(%[[COND]], true {
// CIR:       cir.call @_ZN1SC1Ev(%[[TMP]])
// CIR:       %[[TRUE:.*]] = cir.const #true
// CIR:       cir.store %[[TRUE]], %[[ACTIVE]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:       %[[GET_RESULT:.*]] = cir.call @_ZN1S3getEv(%[[TMP]])
// CIR:       cir.yield %[[GET_RESULT]] : !s32i
// CIR:     }, false {
// CIR:       cir.yield
// CIR:     cir.yield
// With exceptions, the cleanup runs on both normal and EH paths.
// CIR:   } cleanup all {
// CIR:     %[[IS_ACTIVE:.*]] = cir.load {{.*}} %[[ACTIVE]]
// CIR:     cir.if %[[IS_ACTIVE]] {
// CIR:       cir.call @_ZN1SD1Ev(%[[TMP]])
// CIR:     }
// CIR:     cir.yield
// CIR:   }

// LLVM-LABEL: define dso_local void @_Z22test_ternary_temporarybi(
// LLVM-SAME: personality ptr @__gxx_personality_v0
// LLVM:         %[[TMP:.*]] = alloca %struct.S
// LLVM:         %[[ACTIVE:.*]] = alloca i8
// LLVM:         store i8 0, ptr %[[ACTIVE]]
// LLVM:         br i1 %{{.*}}, label %[[TRUE_BR:.*]], label %[[FALSE_BR:.*]]
// Constructor and get() become invoke, unwinding to the landing pad.
// LLVM:       [[TRUE_BR]]:
// LLVM:         invoke void @_ZN1SC1Ev({{.*}} %[[TMP]])
// LLVM-NEXT:            to label %[[CTOR_CONT:.*]] unwind label %[[PAD:.*]]
// LLVM:       [[CTOR_CONT]]:
// LLVM:         store i8 1, ptr %[[ACTIVE]]
// LLVM:         %[[GET_RESULT:.*]] = invoke noundef i32 @_ZN1S3getEv({{.*}} %[[TMP]])
// LLVM-NEXT:            to label %[[GET_CONT:.*]] unwind label %[[PAD]]
// LLVM:       [[GET_CONT]]:
// LLVM:         br label %[[MERGE:.*]]
// LLVM:       [[FALSE_BR]]:
// LLVM:         %[[XVAL:.*]] = load i32, ptr %{{.*}}
// LLVM:         br label %[[MERGE]]
// LLVM:       [[MERGE]]:
// LLVM:         %[[PHI:.*]] = phi i32 [ %[[XVAL]], %[[FALSE_BR]] ], [ %[[GET_RESULT]], %[[GET_CONT]] ]
// Normal cleanup: check active flag, conditionally run destructor.
// LLVM:         %{{.*}} = load i8, ptr %[[ACTIVE]]
// LLVM:         br i1 %{{.*}}, label %[[DTOR:.*]], label %[[SKIP_DTOR:.*]]
// LLVM:       [[DTOR]]:
// LLVM:         call void @_ZN1SD1Ev({{.*}} %[[TMP]])
// LLVM:         br label %[[SKIP_DTOR]]
// EH cleanup: landingpad runs the same active-flag-guarded destructor.
// LLVM:       [[PAD]]:
// LLVM:         landingpad { ptr, i32 }
// LLVM-NEXT:            cleanup
// LLVM:         %{{.*}} = load i8, ptr %[[ACTIVE]]
// LLVM:         br i1 %{{.*}}, label %[[EH_DTOR:.*]], label %[[EH_SKIP_DTOR:.*]]
// LLVM:       [[EH_DTOR]]:
// LLVM:         call void @_ZN1SD1Ev({{.*}} %[[TMP]])
// LLVM:         br label %[[EH_SKIP_DTOR]]
// LLVM:       [[EH_SKIP_DTOR]]:
// LLVM:         resume { ptr, i32 }

// OGCG-LABEL: define dso_local void @_Z22test_ternary_temporarybi(
// OGCG-SAME: personality ptr @__gxx_personality_v0
// OGCG:       entry:
// OGCG:         store i1 false, ptr %[[ACTIVE:.*]]
// OGCG:         br i1 %{{.*}}, label %[[TRUE_BR:.*]], label %[[FALSE_BR:.*]]
// The constructor is a plain call — no active cleanup exists yet.
// OGCG:       [[TRUE_BR]]:
// OGCG:         call void @_ZN1SC1Ev({{.*}} %[[TMP:.*]])
// OGCG:         store i1 true, ptr %[[ACTIVE]]
// With the destructor now active, get() becomes invoke.
// OGCG:         %[[GET_RESULT:.*]] = invoke {{.*}} i32 @_ZN1S3getEv({{.*}} %[[TMP]])
// OGCG-NEXT:            to label %[[INVOKE_CONT:.*]] unwind label %[[LPAD:.*]]
// OGCG:       [[INVOKE_CONT]]:
// OGCG:         br label %[[MERGE:.*]]
// OGCG:       [[FALSE_BR]]:
// OGCG:         br label %[[MERGE]]
// OGCG:       [[MERGE]]:
// OGCG:         %[[COND:.*]] = phi i32 [ %[[GET_RESULT]], %[[INVOKE_CONT]] ], [ %{{.*}}, %[[FALSE_BR]] ]
// Normal cleanup.
// OGCG:         br i1 %{{.*}}, label %[[CLEANUP_ACT:.*]], label %[[CLEANUP_DONE:.*]]
// OGCG:       [[CLEANUP_ACT]]:
// OGCG:         call void @_ZN1SD1Ev({{.*}} %[[TMP]])
// OGCG:         br label %[[CLEANUP_DONE]]
// OGCG:       [[CLEANUP_DONE]]:
// OGCG:         store i32 %[[COND]], ptr %{{.*}}
// EH cleanup: landing pad + same destructor check + resume.
// OGCG:       [[LPAD]]:
// OGCG:         landingpad { ptr, i32 }
// OGCG-NEXT:            cleanup
// OGCG:         br i1 %{{.*}}, label %[[EH_DTOR:.*]], label %[[EH_AFTER_DTOR:.*]]
// OGCG:       [[EH_DTOR]]:
// OGCG:         call void @_ZN1SD1Ev({{.*}} %[[TMP]])
// OGCG:         br label %[[EH_AFTER_DTOR]]
// OGCG:       [[EH_AFTER_DTOR]]:
// OGCG:         resume { ptr, i32 }

void test_ternary_both_branches(bool c) {
  int result = c ? A().get() : B().get();
}
// CIR-LABEL: @_Z26test_ternary_both_branchesb
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
// CIR:     cir.yield
// CIR:   } cleanup all {
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
// LLVM-SAME: personality ptr @__gxx_personality_v0
// LLVM:         %[[TMPA:.*]] = alloca %struct.A
// LLVM:         %[[ACTA:.*]] = alloca i8
// LLVM:         %[[TMPB:.*]] = alloca %struct.B
// LLVM:         %[[ACTB:.*]] = alloca i8
// LLVM:         store i8 0, ptr %[[ACTA]]
// LLVM:         store i8 0, ptr %[[ACTB]]
// LLVM:         br i1 %{{.*}}, label %[[CONSTRUCT_A:.*]], label %[[CONSTRUCT_B:.*]]
// LLVM:       [[CONSTRUCT_A]]:
// LLVM:         invoke void @_ZN1AC1Ev({{.*}} %[[TMPA]])
// LLVM-NEXT:            to label %[[A_CTOR_CONT:.*]] unwind label %[[PAD:.*]]
// LLVM:       [[A_CTOR_CONT]]:
// LLVM:         store i8 1, ptr %[[ACTA]], align 1
// LLVM:         %[[CALLA:.*]] = invoke {{.*}} i32 @_ZN1A3getEv({{.*}} %[[TMPA]])
// LLVM-NEXT:            to label %[[A_GET_CONT:.*]] unwind label %[[LPAD:.*]]
// LLVM:       [[CONSTRUCT_B]]:
// LLVM:         invoke void @_ZN1BC1Ev({{.*}} %[[TMPB]])
// LLVM-NEXT:            to label %[[B_CTOR_CONT:.*]] unwind label %[[LPAD]]
// LLVM:       [[B_CTOR_CONT]]:
// LLVM:         store i8 1, ptr %[[ACTB]]
// LLVM:         %[[CALLB:.*]] = invoke {{.*}} i32 @_ZN1B3getEv({{.*}} %[[TMPB]])
// LLVM-NEXT:            to label %{{.*}} unwind label %[[PAD]]
// Normal cleanup: check both active flags.
// LLVM:         %{{.*}} = load i8, ptr %[[ACTB]]
// LLVM:         br i1 %{{.*}}, label %[[DTOR_B:.*]], label %[[SKIP_DTOR_B:.*]]
// LLVM:       [[DTOR_B]]:
// LLVM:         call void @_ZN1BD1Ev({{.*}} %[[TMPB]])
// LLVM:       [[SKIP_DTOR_B]]:
// LLVM:         %{{.*}} = load i8, ptr %[[ACTA]]
// LLVM:         br i1 %{{.*}}, label %[[DTOR_A:.*]], label %[[SKIP_DTOR_A:.*]]
// LLVM:       [[DTOR_A]]:
// LLVM:         call void @_ZN1AD1Ev({{.*}} %[[TMPA]])
// EH cleanup: single landingpad, then same active-flag checks for both.
// LLVM:       [[LPAD]]:
// LLVM:         landingpad { ptr, i32 }
// LLVM-NEXT:            cleanup
// LLVM:         %{{.*}} = load i8, ptr %[[ACTB]]
// LLVM:         br i1 %{{.*}}, label %[[EH_DTOR_B:.*]], label %[[EH_SKIP_DTOR_B:.*]]
// LLVM:       [[EH_DTOR_B]]:
// LLVM:         call void @_ZN1BD1Ev({{.*}} %[[TMPB]])
// LLVM:       [[EH_SKIP_DTOR_B]]:
// LLVM:         %{{.*}} = load i8, ptr %[[ACTA]]
// LLVM:         br i1 %{{.*}}, label %[[EH_DTOR_A:.*]], label %[[EH_SKIP_DTOR_A:.*]]
// LLVM:       [[EH_DTOR_A]]:
// LLVM:         call void @_ZN1AD1Ev({{.*}} %[[TMPA]])
// LLVM:       [[EH_SKIP_DTOR_A]]:
// LLVM:         resume { ptr, i32 }

// OGCG-LABEL: define dso_local void @_Z26test_ternary_both_branchesb(
// OGCG-SAME: personality ptr @__gxx_personality_v0
// OGCG:       entry:
// OGCG:         store i1 false, ptr %[[ACTA:.*]]
// OGCG:         store i1 false, ptr %[[ACTB:.*]]
// OGCG:         br i1 %{{.*}}, label %[[TRUE_BR:.*]], label %[[FALSE_BR:.*]]
// A constructor is call (no active cleanup yet); A get() is invoke.
// OGCG:       [[TRUE_BR]]:
// OGCG:         call void @_ZN1AC1Ev({{.*}} %[[TMPA:.*]])
// OGCG:         store i1 true, ptr %[[ACTA]]
// OGCG:         %[[CALLA:.*]] = invoke {{.*}} i32 @_ZN1A3getEv({{.*}} %[[TMPA]])
// OGCG-NEXT:            to label %{{.*}} unwind label %[[LPAD1:.*]]
// B constructor is invoke (A cleanup is active); B get() invokes to a second pad.
// OGCG:       [[FALSE_BR]]:
// OGCG:         invoke void @_ZN1BC1Ev({{.*}} %[[TMPB:.*]])
// OGCG-NEXT:            to label %{{.*}} unwind label %[[LPAD1]]
// OGCG:         store i1 true, ptr %[[ACTB]]
// OGCG:         invoke {{.*}} i32 @_ZN1B3getEv({{.*}} %[[TMPB]])
// OGCG-NEXT:            to label %{{.*}} unwind label %[[LPAD2:.*]]
// Normal cleanup: B first, then A (reverse construction order).
// OGCG:       [[MERGE:.*]]:
// OGCG:         br i1 %{{.*}}, label %[[DTOR_B:.*]], label %[[AFTER_DTOR_B:.*]]
// OGCG:       [[DTOR_B]]:
// OGCG:         call void @_ZN1BD1Ev({{.*}} %[[TMPB]])
// OGCG:         br label %[[AFTER_DTOR_B]]
// OGCG:       [[AFTER_DTOR_B]]:
// OGCG:         br i1 %{{.*}}, label %[[DTOR_A:.*]], label %[[AFTER_DTOR_A:.*]]
// OGCG:       [[DTOR_A]]:
// OGCG:         call void @_ZN1AD1Ev({{.*}} %[[TMPA]])
// OGCG:         br label %[[AFTER_DTOR_A]]
// First landing pad: from A.get() or B ctor — only A cleanup needed.
// OGCG:       [[LPAD1]]:
// OGCG:         landingpad { ptr, i32 }
// OGCG-NEXT:            cleanup
// OGCG:         br label %[[EH_CLEANUP:.*]]
// Second landing pad: from B.get() — B cleanup, then A cleanup.
// OGCG:       [[LPAD2]]:
// OGCG:         landingpad { ptr, i32 }
// OGCG-NEXT:            cleanup
// OGCG:         br i1 %{{.*}}, label %[[EH_DTOR_B:.*]], label %[[EH_AFTER_DTOR_B:.*]]
// OGCG:       [[EH_DTOR_B]]:
// OGCG:         call void @_ZN1BD1Ev({{.*}} %[[TMPB]])
// OGCG:         br label %[[EH_AFTER_DTOR_B]]
// OGCG:       [[EH_AFTER_DTOR_B]]:
// OGCG:         br label %[[EH_CLEANUP]]
// Shared EH cleanup for A's destructor, then resume.
// OGCG:       [[EH_CLEANUP]]:
// OGCG:         br i1 %{{.*}}, label %[[EH_DTOR_A:.*]], label %[[EH_AFTER_DTOR_A:.*]]
// OGCG:       [[EH_DTOR_A]]:
// OGCG:         call void @_ZN1AD1Ev({{.*}} %[[TMPA]])
// OGCG:         br label %[[EH_AFTER_DTOR_A]]
// OGCG:       [[EH_AFTER_DTOR_A]]:
// OGCG:         resume { ptr, i32 }

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
// CIR:     cir.store %{{.*}}, %{{.*}} : !s32i, !cir.ptr<!s32i>
// CIR:     cir.yield
// CIR:   } cleanup all {
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
// CIR:   %[[RET:.*]] = cir.load %{{.*}} : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[RET]] : !s32i

// LLVM-LABEL: define dso_local noundef i32 @_Z19test_return_ternaryb(
// LLVM-SAME: personality ptr @__gxx_personality_v0
// LLVM:         %[[RETVAL:.*]] = alloca i32
// LLVM:         %[[TMPA:.*]] = alloca %struct.A
// LLVM:         %[[ACTA:.*]] = alloca i8
// LLVM:         %[[TMPB:.*]] = alloca %struct.B
// LLVM:         %[[ACTB:.*]] = alloca i8
// LLVM:         store i8 0, ptr %[[ACTA]]
// LLVM:         store i8 0, ptr %[[ACTB]]
// LLVM:         br i1 %{{.*}}, label %[[CONSTRUCT_A:.*]], label %[[CONSTRUCT_B:.*]]
// LLVM:       [[CONSTRUCT_A]]:
// LLVM:         invoke void @_ZN1AC1Ev({{.*}} %[[TMPA]])
// LLVM-NEXT:            to label %[[A_CTOR_CONT:.*]] unwind label %[[LPAD:.*]]
// LLVM:       [[A_CTOR_CONT]]:
// LLVM:         store i8 1, ptr %[[ACTA]], align 1
// LLVM:         %[[CALLA:.*]] = invoke {{.*}} i32 @_ZN1A3getEv({{.*}} %[[TMPA]])
// LLVM-NEXT:            to label %{{.*}} unwind label %[[LPAD]]
// LLVM:       [[CONSTRUCT_B]]:
// LLVM:         invoke void @_ZN1BC1Ev({{.*}} %[[TMPB]])
// LLVM-NEXT:            to label %[[B_CTOR_CONT:.*]] unwind label %[[LPAD]]
// LLVM:       [[B_CTOR_CONT]]:
// LLVM:         store i8 1, ptr %[[ACTB]], align 1
// LLVM:         %[[CALLB:.*]] = invoke {{.*}} i32 @_ZN1B3getEv({{.*}} %[[TMPB]])
// LLVM-NEXT:            to label %{{.*}} unwind label %[[LPAD]]
// LLVM:         store i32 %{{.*}}, ptr %[[RETVAL]]
// Normal cleanup: check both active flags.
// LLVM:         %{{.*}} = load i8, ptr %[[ACTB]]
// LLVM:         br i1 %{{.*}}, label %[[DTOR_B:.*]], label %[[SKIP_DTOR_B:.*]]
// LLVM:       [[DTOR_B]]:
// LLVM:         call void @_ZN1BD1Ev({{.*}} %[[TMPB]])
// LLVM:       [[SKIP_DTOR_B]]:
// LLVM:         %{{.*}} = load i8, ptr %[[ACTA]]
// LLVM:         br i1 %{{.*}}, label %[[DTOR_A:.*]], label %[[SKIP_DTOR_A:.*]]
// LLVM:       [[DTOR_A]]:
// LLVM:         call void @_ZN1AD1Ev({{.*}} %[[TMPA]])
// EH cleanup: same active-flag checks, then resume.
// LLVM:       [[LPAD]]:
// LLVM:         landingpad { ptr, i32 }
// LLVM-NEXT:            cleanup
// LLVM:         %{{.*}} = load i8, ptr %[[ACTB]]
// LLVM:         br i1 %{{.*}}, label %[[EH_DTOR_B:.*]], label %[[EH_SKIP_DTOR_B:.*]]
// LLVM:       [[EH_DTOR_B]]:
// LLVM:         call void @_ZN1BD1Ev({{.*}} %[[TMPB]])
// LLVM:       [[EH_SKIP_DTOR_B]]:
// LLVM:         %{{.*}} = load i8, ptr %[[ACTA]]
// LLVM:         br i1 %{{.*}}, label %[[EH_DTOR_A:.*]], label %[[EH_SKIP_DTOR_A:.*]]
// LLVM:       [[EH_DTOR_A]]:
// LLVM:         call void @_ZN1AD1Ev({{.*}} %[[TMPA]])
// LLVM:       [[EH_SKIP_DTOR_A]]:
// LLVM:         resume { ptr, i32 }
// LLVM:         %[[RET:.*]] = load i32, ptr %[[RETVAL]]
// LLVM:         ret i32 %[[RET]]

// OGCG-LABEL: define dso_local noundef i32 @_Z19test_return_ternaryb(
// OGCG-SAME: personality ptr @__gxx_personality_v0
// OGCG:       entry:
// OGCG:         store i1 false, ptr %[[ACTA:.*]]
// OGCG:         store i1 false, ptr %[[ACTB:.*]]
// OGCG:         br i1 %{{.*}}, label %[[TRUE_BR:.*]], label %[[FALSE_BR:.*]]
// OGCG:       [[TRUE_BR]]:
// OGCG:         call void @_ZN1AC1Ev({{.*}} %[[TMPA:.*]])
// OGCG:         store i1 true, ptr %[[ACTA]]
// OGCG:         %[[CALLA:.*]] = invoke {{.*}} i32 @_ZN1A3getEv({{.*}} %[[TMPA]])
// OGCG-NEXT:            to label %{{.*}} unwind label %[[LPAD1:.*]]
// OGCG:       [[FALSE_BR]]:
// OGCG:         invoke void @_ZN1BC1Ev({{.*}} %[[TMPB:.*]])
// OGCG-NEXT:            to label %{{.*}} unwind label %[[LPAD1]]
// OGCG:         store i1 true, ptr %[[ACTB]]
// OGCG:         invoke {{.*}} i32 @_ZN1B3getEv({{.*}} %[[TMPB]])
// OGCG-NEXT:            to label %{{.*}} unwind label %[[LPAD2:.*]]
// Normal cleanup: B first, then A.
// OGCG:       [[MERGE:.*]]:
// OGCG:         br i1 %{{.*}}, label %[[DTOR_B:.*]], label %[[AFTER_DTOR_B:.*]]
// OGCG:       [[DTOR_B]]:
// OGCG:         call void @_ZN1BD1Ev({{.*}} %[[TMPB]])
// OGCG:         br label %[[AFTER_DTOR_B]]
// OGCG:       [[AFTER_DTOR_B]]:
// OGCG:         br i1 %{{.*}}, label %[[DTOR_A:.*]], label %[[AFTER_DTOR_A:.*]]
// OGCG:       [[DTOR_A]]:
// OGCG:         call void @_ZN1AD1Ev({{.*}} %[[TMPA]])
// OGCG:         br label %[[AFTER_DTOR_A]]
// OGCG:       [[AFTER_DTOR_A]]:
// OGCG:         ret i32 %{{.*}}
// First landing pad: from A.get() or B ctor.
// OGCG:       [[LPAD1]]:
// OGCG:         landingpad { ptr, i32 }
// OGCG-NEXT:            cleanup
// OGCG:         br label %[[EH_CLEANUP:.*]]
// Second landing pad: from B.get() — B cleanup, then A cleanup.
// OGCG:       [[LPAD2]]:
// OGCG:         landingpad { ptr, i32 }
// OGCG-NEXT:            cleanup
// OGCG:         br i1 %{{.*}}, label %[[EH_DTOR_B:.*]], label %[[EH_AFTER_DTOR_B:.*]]
// OGCG:       [[EH_DTOR_B]]:
// OGCG:         call void @_ZN1BD1Ev({{.*}} %[[TMPB]])
// OGCG:         br label %[[EH_AFTER_DTOR_B]]
// OGCG:       [[EH_AFTER_DTOR_B]]:
// OGCG:         br label %[[EH_CLEANUP]]
// OGCG:       [[EH_CLEANUP]]:
// OGCG:         br i1 %{{.*}}, label %[[EH_DTOR_A:.*]], label %[[EH_AFTER_DTOR_A:.*]]
// OGCG:       [[EH_DTOR_A]]:
// OGCG:         call void @_ZN1AD1Ev({{.*}} %[[TMPA]])
// OGCG:         br label %[[EH_AFTER_DTOR_A]]
// OGCG:       [[EH_AFTER_DTOR_A]]:
// OGCG:         resume { ptr, i32 }

// False positive: ExprWithCleanups wraps a ternary, but S() is constructed
// outside the conditional so no cleanup is deferred. The cleanup.scope still
// uses "cleanup all" for the unconditional destructor.
int test_false_positive_conditional(bool c) {
  return S().get() ? 1 : 2;
}
// CIR-LABEL: @_Z31test_false_positive_conditionalb
// CIR-NOT:   cir.alloca {{.*}} ["cleanup.cond"]
// CIR:   %[[TMP:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["ref.tmp0"]
// CIR:   cir.call @_ZN1SC1Ev(%[[TMP]])
// CIR:   cir.cleanup.scope {
// CIR:     %[[VAL:.*]] = cir.call @_ZN1S3getEv(%[[TMP]])
// CIR:     %[[BOOL:.*]] = cir.cast int_to_bool %[[VAL]]
// CIR:     %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR:     %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CIR:     %[[SEL:.*]] = cir.select if %[[BOOL]] then %[[ONE]] else %[[TWO]]
// CIR:     cir.store %[[SEL]], %{{.*}} : !s32i, !cir.ptr<!s32i>
// CIR:     cir.yield
// CIR:   } cleanup all {
// CIR:     cir.call @_ZN1SD1Ev(%[[TMP]])
// CIR:     cir.yield
// CIR:   }

// LLVM-LABEL: define dso_local noundef i32 @_Z31test_false_positive_conditionalb(
// LLVM-SAME: personality ptr @__gxx_personality_v0
// LLVM:         %[[RETVAL:.*]] = alloca i32
// LLVM:         %[[TMP:.*]] = alloca %struct.S
// The constructor is call — no active EH cleanup yet.
// LLVM:         call void @_ZN1SC1Ev({{.*}} %[[TMP]])
// get() becomes invoke because the destructor cleanup is active.
// LLVM:         %[[VAL:.*]] = invoke {{.*}} i32 @_ZN1S3getEv({{.*}} %[[TMP]])
// LLVM-NEXT:            to label %[[GET_CONT:.*]] unwind label %[[LPAD:.*]]
// LLVM:       [[GET_CONT]]:
// LLVM:         %[[CMP:.*]] = icmp ne i32 %[[VAL]], 0
// LLVM:         %[[SEL:.*]] = select i1 %[[CMP]], i32 1, i32 2
// LLVM:         store i32 %[[SEL]], ptr %[[RETVAL]]
// Normal path: unconditional destructor.
// LLVM:         call void @_ZN1SD1Ev({{.*}} %[[TMP]])
// EH path: landingpad + unconditional destructor + resume.
// LLVM:       [[LPAD]]:
// LLVM:         landingpad { ptr, i32 }
// LLVM-NEXT:            cleanup
// LLVM:         call void @_ZN1SD1Ev({{.*}} %[[TMP]])
// LLVM:         resume { ptr, i32 }
// LLVM:         %[[RET:.*]] = load i32, ptr %[[RETVAL]]
// LLVM:         ret i32 %[[RET]]

// OGCG-LABEL: define dso_local noundef i32 @_Z31test_false_positive_conditionalb(
// OGCG-SAME: personality ptr @__gxx_personality_v0
// OGCG:       entry:
// The constructor is call; get() is invoke.
// OGCG:         call void @_ZN1SC1Ev({{.*}} %[[TMP:.*]])
// OGCG:         %[[VAL:.*]] = invoke {{.*}} i32 @_ZN1S3getEv({{.*}} %[[TMP]])
// OGCG-NEXT:            to label %[[INVOKE_CONT:.*]] unwind label %[[LPAD:.*]]
// OGCG:       [[INVOKE_CONT]]:
// OGCG:         %[[CMP:.*]] = icmp ne i32 %[[VAL]], 0
// OGCG:         %[[SEL:.*]] = select i1 %[[CMP]], i32 1, i32 2
// Normal path: unconditional destructor + return.
// OGCG:         call void @_ZN1SD1Ev({{.*}} %[[TMP]])
// OGCG:         ret i32 %[[SEL]]
// EH path: unconditional destructor + resume.
// OGCG:       [[LPAD]]:
// OGCG:         landingpad { ptr, i32 }
// OGCG-NEXT:            cleanup
// OGCG:         call void @_ZN1SD1Ev({{.*}} %[[TMP]])
// OGCG:         resume { ptr, i32 }

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
// Inner cir.scope for the statement expression.
// CIR:   cir.scope {
// CIR:     %[[S:.*]] = cir.alloca !rec_T, !cir.ptr<!rec_T>, ["s", init]
// Inner ternary: c1 ? T(1) : T(2) — no cleanup scope needed.
// CIR:     %[[C1:.*]] = cir.load {{.*}} : !cir.ptr<!cir.bool>, !cir.bool
// CIR:     cir.if %[[C1]] {
// CIR:       %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR:       cir.call @_ZN1TC1Ei(%[[S]], %[[ONE]])
// CIR:     } else {
// CIR:       %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CIR:       cir.call @_ZN1TC1Ei(%[[S]], %[[TWO]])
// CIR:     }
// Copy s into ref.tmp; destroy s on all paths (normal + EH).
// CIR:     cir.cleanup.scope {
// CIR:       cir.call @_ZN1TC1ERKS_(%[[REF_TMP]], %[[S]])
// CIR:       cir.yield
// CIR:     } cleanup all {
// CIR:       cir.call @_ZN1TD1Ev(%[[S]])
// CIR:       cir.yield
// CIR:     }
// CIR:   }
// Outer cleanup scope: operator bool() + outer ternary + destroys ref.tmp.
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
// CIR:   } cleanup all {
// CIR:     cir.call @_ZN1TD1Ev(%[[REF_TMP]])
// CIR:     cir.yield
// CIR:   }
// result destructor.
// CIR:   cir.cleanup.scope {
// CIR:     cir.yield
// CIR:   } cleanup all {
// CIR:     cir.call @_ZN1TD1Ev(%[[RESULT]])
// CIR:     cir.yield
// CIR:   }

// LLVM-LABEL: define dso_local void @_Z15test_nested_ewcbb(
// LLVM-SAME: personality ptr @__gxx_personality_v0
// Inner ternary: c1 ? T(1) : T(2) — plain calls (no active cleanup yet).
// LLVM:         br i1 %{{.*}}, label %[[T1:.*]], label %[[T2:.*]]
// LLVM:       [[T1]]:
// LLVM:         call void @_ZN1TC1Ei({{.*}} %[[S:.*]], i32 {{.*}} 1)
// LLVM:         br label %[[INNER_MERGE:.*]]
// LLVM:       [[T2]]:
// LLVM:         call void @_ZN1TC1Ei({{.*}} %[[S]], i32 {{.*}} 2)
// LLVM:         br label %[[INNER_MERGE]]
// Copy construct: invoke (s destructor is active EH cleanup).
// LLVM:       [[INNER_MERGE]]:
// LLVM:         invoke void @_ZN1TC1ERKS_({{.*}} %[[REF_TMP:.*]], {{.*}} %[[S]])
// LLVM-NEXT:            to label %[[COPY_CONT:.*]] unwind label %[[LPAD_S:.*]]
// LLVM:       [[COPY_CONT]]:
// LLVM:         call void @_ZN1TD1Ev({{.*}} %[[S]])
// EH cleanup for s: appears here in block order, before outer ternary.
// LLVM:       [[LPAD_S]]:
// LLVM:         landingpad { ptr, i32 }
// LLVM-NEXT:            cleanup
// LLVM:         call void @_ZN1TD1Ev({{.*}} %[[S]])
// LLVM:         resume { ptr, i32 }
// Outer ternary: operator bool() is invoke (ref.tmp cleanup is active).
// LLVM:         %[[BOOL:.*]] = invoke {{.*}} i1 @_ZN1TcvbEv({{.*}} %[[REF_TMP]])
// LLVM-NEXT:            to label %[[BOOL_CONT:.*]] unwind label %[[LPAD_REF:.*]]
// LLVM:       [[BOOL_CONT]]:
// LLVM:         br i1 %[[BOOL]], label %[[TRUE:.*]], label %[[FALSE:.*]]
// LLVM:       [[TRUE]]:
// LLVM:         br i1 %{{.*}}, label %[[T3:.*]], label %[[T4:.*]]
// LLVM:       [[T3]]:
// LLVM:         invoke void @_ZN1TC1Ei({{.*}} %[[RESULT:.*]], i32 {{.*}} 3)
// LLVM-NEXT:            to label %{{.*}} unwind label %[[LPAD_REF]]
// LLVM:       [[T4]]:
// LLVM:         invoke void @_ZN1TC1Ei({{.*}} %[[RESULT]], i32 {{.*}} 4)
// LLVM-NEXT:            to label %{{.*}} unwind label %[[LPAD_REF]]
// LLVM:       [[FALSE]]:
// LLVM:         invoke void @_ZN1TC1Ei({{.*}} %[[RESULT]], i32 {{.*}} 5)
// LLVM-NEXT:            to label %{{.*}} unwind label %[[LPAD_REF]]
// Normal cleanup: destroy ref.tmp.
// LLVM:         call void @_ZN1TD1Ev({{.*}} %[[REF_TMP]])
// EH cleanup for ref.tmp: landingpad + destroy ref.tmp + resume.
// LLVM:       [[LPAD_REF]]:
// LLVM:         landingpad { ptr, i32 }
// LLVM-NEXT:            cleanup
// LLVM:         call void @_ZN1TD1Ev({{.*}} %[[REF_TMP]])
// LLVM:         resume { ptr, i32 }
// Result destructor on normal path.
// LLVM:         call void @_ZN1TD1Ev({{.*}} %[[RESULT]])

// OGCG-LABEL: define dso_local void @_Z15test_nested_ewcbb(
// OGCG-SAME: personality ptr @__gxx_personality_v0
// Inner ternary: c1 ? T(1) : T(2).
// OGCG:         br i1 %{{.*}}, label %[[T1:.*]], label %[[T2:.*]]
// OGCG:       [[T1]]:
// OGCG:         call void @_ZN1TC1Ei({{.*}} %[[S:.*]], i32 {{.*}} 1)
// OGCG:         br label %[[INNER_MERGE:.*]]
// OGCG:       [[T2]]:
// OGCG:         call void @_ZN1TC1Ei({{.*}} %[[S]], i32 {{.*}} 2)
// OGCG:         br label %[[INNER_MERGE]]
// Copy construct ref.tmp: invoke (s destructor is EH cleanup).
// OGCG:       [[INNER_MERGE]]:
// OGCG:         invoke void @_ZN1TC1ERKS_({{.*}} %[[REF_TMP:.*]], {{.*}} %[[S]])
// OGCG-NEXT:            to label %[[COPY_CONT:.*]] unwind label %[[LPAD_S:.*]]
// OGCG:       [[COPY_CONT]]:
// OGCG:         call void @_ZN1TD1Ev({{.*}} %[[S]])
// Outer: operator bool() is invoke, then outer ternary branches.
// OGCG:         %[[BOOL:.*]] = invoke {{.*}} i1 @_ZN1TcvbEv({{.*}} %[[REF_TMP]])
// OGCG-NEXT:            to label %[[BOOL_CONT:.*]] unwind label %[[LPAD_REF:.*]]
// OGCG:       [[BOOL_CONT]]:
// OGCG:         br i1 %[[BOOL]], label %[[TRUE:.*]], label %[[FALSE:.*]]
// OGCG:       [[TRUE]]:
// OGCG:         br i1 %{{.*}}, label %[[T3:.*]], label %[[T4:.*]]
// OGCG:       [[T3]]:
// OGCG:         invoke void @_ZN1TC1Ei({{.*}} %[[RESULT:.*]], i32 {{.*}} 3)
// OGCG-NEXT:            to label %{{.*}} unwind label %[[LPAD_REF]]
// OGCG:       [[T4]]:
// OGCG:         invoke void @_ZN1TC1Ei({{.*}} %[[RESULT]], i32 {{.*}} 4)
// OGCG-NEXT:            to label %{{.*}} unwind label %[[LPAD_REF]]
// OGCG:       [[FALSE]]:
// OGCG:         invoke void @_ZN1TC1Ei({{.*}} %[[RESULT]], i32 {{.*}} 5)
// OGCG-NEXT:            to label %{{.*}} unwind label %[[LPAD_REF]]
// Normal cleanup: destroy ref.tmp, then result.
// OGCG:         call void @_ZN1TD1Ev({{.*}} %[[REF_TMP]])
// OGCG:         call void @_ZN1TD1Ev({{.*}} %[[RESULT]])
// EH cleanup for s: landingpad + destroy s.
// OGCG:       [[LPAD_S]]:
// OGCG:         landingpad { ptr, i32 }
// OGCG-NEXT:            cleanup
// OGCG:         call void @_ZN1TD1Ev({{.*}} %[[S]])
// EH cleanup for ref.tmp: landingpad + destroy ref.tmp.
// OGCG:       [[LPAD_REF]]:
// OGCG:         landingpad { ptr, i32 }
// OGCG-NEXT:            cleanup
// OGCG:         call void @_ZN1TD1Ev({{.*}} %[[REF_TMP]])
// OGCG:         resume { ptr, i32 }
