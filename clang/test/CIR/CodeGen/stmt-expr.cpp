// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

class A {
public:
  A(): x(0) {}
  A(A &a) : x(a.x) {}
  int x;
  void Foo() {}
};

void test1() {
  ({
    A a;
    a;
  }).Foo();
}

// CIR: cir.func {{.*}} @_Z5test1v()
// CIR:   cir.scope {
// CIR:     %[[REF_TMP0:.+]] = cir.alloca !rec_A, !cir.ptr<!rec_A>, ["ref.tmp0"]
// CIR:     %[[TMP:.+]]      = cir.alloca !rec_A, !cir.ptr<!rec_A>, ["tmp"]
// CIR:     cir.scope {
// CIR:       %[[A:.+]] = cir.alloca !rec_A, !cir.ptr<!rec_A>, ["a", init]
// CIR:       cir.call @_ZN1AC2Ev(%[[A]]) : (!cir.ptr<!rec_A>) -> ()
// CIR:       cir.call @_ZN1AC2ERS_(%[[REF_TMP0]], %[[A]]) : (!cir.ptr<!rec_A>, !cir.ptr<!rec_A>) -> ()
// CIR:     }
// CIR:     cir.call @_ZN1A3FooEv(%[[REF_TMP0]]) : (!cir.ptr<!rec_A>) -> ()
// CIR:   }
// CIR:   cir.return

// LLVM: define dso_local void @_Z5test1v()
// LLVM:   %[[VAR1:.+]] = alloca %class.A, i64 1
// LLVM:   %[[VAR2:.+]] = alloca %class.A, i64 1
// LLVM:   %[[VAR3:.+]] = alloca %class.A, i64 1
// LLVM:   br label %[[LBL4:.+]]
// LLVM: [[LBL4]]:
// LLVM:     br label %[[LBL5:.+]]
// LLVM: [[LBL5]]:
// LLVM:     call void @_ZN1AC2Ev(ptr %[[VAR3]])
// LLVM:     call void @_ZN1AC2ERS_(ptr %[[VAR1]], ptr %[[VAR3]])
// LLVM:     br label %[[LBL6:.+]]
// LLVM: [[LBL6]]:
// LLVM:     call void @_ZN1A3FooEv(ptr %[[VAR1]])
// LLVM:     br label %[[LBL7:.+]]
// LLVM: [[LBL7]]:
// LLVM:     ret void

// OGCG: define dso_local void @_Z5test1v()
// OGCG: entry:
// OGCG:   %[[REF_TMP:.+]] = alloca %class.A
// OGCG:   %[[A:.+]]       = alloca %class.A
// OGCG:   call void @_ZN1AC2Ev(ptr {{.*}} %[[A]])
// OGCG:   call void @_ZN1AC2ERS_(ptr {{.*}} %[[REF_TMP]], ptr {{.*}} %[[A]])
// OGCG:   call void @_ZN1A3FooEv(ptr {{.*}} %[[REF_TMP]])
// OGCG:   ret void

struct with_dtor {
  ~with_dtor();
};

void cleanup() {
  ({ with_dtor wd; });
}

// CIR: cir.func {{.*}} @_Z7cleanupv()
// CIR:   cir.scope {
// CIR:     %[[WD:.+]] = cir.alloca !rec_with_dtor, !cir.ptr<!rec_with_dtor>, ["wd"]
// CIR:     cir.call @_ZN9with_dtorD1Ev(%[[WD]]) nothrow : (!cir.ptr<!rec_with_dtor>) -> ()
// CIR:   }
// CIR:   cir.return

// LLVM: define dso_local void @_Z7cleanupv()
// LLVM:   %[[WD:.+]] = alloca %struct.with_dtor, i64 1
// LLVM:   br label %[[LBL2:.+]]
// LLVM: [[LBL2]]:
// LLVM:     call void @_ZN9with_dtorD1Ev(ptr %[[WD]])
// LLVM:     br label %[[LBL3:.+]]
// LLVM: [[LBL3]]:
// LLVM:     ret void

// OGCG: define dso_local void @_Z7cleanupv()
// OGCG: entry:
// OGCG:   %[[WD:.+]] = alloca %struct.with_dtor
// OGCG:   call void @_ZN9with_dtorD1Ev(ptr {{.*}} %[[WD]])
// OGCG:   ret void

void gnu_statement_extension() {
  float b = __real__ ({float _Complex a; a;});
}

// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["b", init]
// CIR: %[[TMP_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["tmp"]
// CIR: cir.scope {
// CIR:   %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR:   %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR:   cir.store {{.*}} %[[TMP_A]], %[[TMP_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR: }
// CIR: %[[TMP:.*]] = cir.load {{.*}} %[[TMP_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[REAL:.*]] = cir.complex.real %[[TMP]] : !cir.complex<!cir.float> -> !cir.float
// CIR: cir.store {{.*}} %[[REAL]], %[[B_ADDR]] : !cir.float, !cir.ptr<!cir.float>

// LLVM:   %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM:   %[[B_ADDR:.*]] = alloca float, i64 1, align 4
// LLVM:   %[[TMP_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM:   br label %[[LABEL_1:.*]]
// LLVM: [[LABEL_1]]:
// LLVM:   %[[TMP_A:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM:   store { float, float } %[[TMP_A]], ptr %[[TMP_ADDR]], align 4
// LLVM:   br label %[[LABEL_2:.*]]
// LLVM: [[LABEL_2]]:
// LLVM:   %[[TMP:.*]] = load { float, float }, ptr %[[TMP_ADDR]], align 4
// LLVM:   %[[REAL:.*]] = extractvalue { float, float } %[[TMP]], 0
// LLVM:   store float %[[REAL]], ptr %[[B_ADDR]], align 4

// OGCG: %[[B_ADDR:.*]] = alloca float, align 4
// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[TMP_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[TMP_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[TMP_ADDR]], i32 0, i32 0
// OGCG: %[[TMP_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[TMP_ADDR]], i32 0, i32 1
// OGCG: store float %[[A_REAL]], ptr %[[TMP_REAL_PTR]], align 4
// OGCG: store float %[[A_IMAG]], ptr %[[TMP_IMAG_PTR]], align 4
// OGCG: %[[TMP_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[TMP_ADDR]], i32 0, i32 0
// OGCG: %[[TMP_REAL:.*]] = load float, ptr %[[TMP_REAL_PTR]], align 4
// OGCG: store float %[[TMP_REAL]], ptr %[[B_ADDR]], align 4
