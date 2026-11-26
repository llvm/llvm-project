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
// CIR:   %[[REF_TMP0:.+]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<!rec_A>
// CIR:   %[[TMP:.+]]      = cir.alloca "tmp" {{.*}} : !cir.ptr<!rec_A>
// CIR:   cir.scope {
// CIR:     %[[A:.+]] = cir.alloca "a" {{.*}} init : !cir.ptr<!rec_A>
// CIR:     cir.call @_ZN1AC2Ev(%[[A]]) : (!cir.ptr<!rec_A> {{.*}}) -> ()
// CIR:     cir.call @_ZN1AC2ERS_(%[[REF_TMP0]], %[[A]]) : (!cir.ptr<!rec_A> {{.*}}, !cir.ptr<!rec_A> {{.*}}) -> ()
// CIR:   }
// CIR:   cir.call @_ZN1A3FooEv(%[[REF_TMP0]]) : (!cir.ptr<!rec_A> {{.*}}) -> ()
// CIR:   cir.return

// LLVM: define dso_local void @_Z5test1v()
// LLVM:   %[[REF_TMP:.+]] = alloca %class.A, 
// LLVM:   %[[TMP:.+]] = alloca %class.A, 
// LLVM:   %[[A:.+]] = alloca %class.A, 
// LLVM:   br label %[[LBL4:.+]]
// LLVM: [[LBL4]]:
// LLVM:     call void @_ZN1AC2Ev(ptr {{.*}} %[[A]])
// LLVM:     call void @_ZN1AC2ERS_(ptr {{.*}} %[[REF_TMP]], ptr {{.*}} %[[A]])
// LLVM:     br label %[[LBL5:.+]]
// LLVM: [[LBL5]]:
// LLVM:     call void @_ZN1A3FooEv(ptr {{.*}} %[[REF_TMP]])
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
// CIR:     %[[WD:.+]] = cir.alloca "wd" {{.*}} : !cir.ptr<!rec_with_dtor>
// CIR:     cir.cleanup.scope {
// CIR:       cir.yield
// CIR:     } cleanup normal {
// CIR:       cir.call @_ZN9with_dtorD1Ev(%[[WD]]) nothrow : (!cir.ptr<!rec_with_dtor> {{.*}}) -> ()
// CIR:       cir.yield
// CIR:     }
// CIR:   }
// CIR:   cir.return

// LLVM: define dso_local void @_Z7cleanupv()
// LLVM:   %[[WD:.+]] = alloca %struct.with_dtor, 
// LLVM:   br label %[[LBL2:.+]]
// LLVM: [[LBL2]]:
// LLVM:     call void @_ZN9with_dtorD1Ev(ptr {{.*}} %[[WD]])
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

// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} init : !cir.ptr<!cir.float>
// CIR: %[[TMP_ADDR:.*]] = cir.alloca "tmp" {{.*}} : !cir.ptr<!cir.complex<!cir.float>>
// CIR: cir.scope {
// CIR:   %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.complex<!cir.float>>
// CIR:   %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR:   cir.store {{.*}} %[[TMP_A]], %[[TMP_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR: }
// CIR: %[[TMP:.*]] = cir.load {{.*}} %[[TMP_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[REAL:.*]] = cir.complex.real %[[TMP]] : !cir.complex<!cir.float> -> !cir.float
// CIR: cir.store {{.*}} %[[REAL]], %[[B_ADDR]] : !cir.float, !cir.ptr<!cir.float>

// LLVM:   %[[B_ADDR:.*]] = alloca float, align 4
// LLVM:   %[[TMP_ADDR:.*]] = alloca { float, float }, align 4
// LLVM:   %[[A_ADDR:.*]] = alloca { float, float }, align 4
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

struct S {
  int x;
};

int test_stmt_expr_lvalue() {
  return ({
    struct S s = {42};
    s;
  }).x;
}

// CIR-LABEL: cir.func {{.*}} @_Z21test_stmt_expr_lvaluev()
// CIR:         %[[REF_TMP:.*]] = cir.alloca "ref.tmp0" align(4) : !cir.ptr<!rec_S>
// CIR:         %[[TMP:.*]] = cir.alloca "tmp" align(4) : !cir.ptr<!rec_S>
// CIR:         cir.scope {
// CIR:           %[[S:.*]] = cir.alloca "s" align(4) init : !cir.ptr<!rec_S>
// CIR:           %[[GLOB:.*]] = cir.get_global @__const._Z21test_stmt_expr_lvaluev.s : !cir.ptr<!rec_S>
// CIR:           cir.copy %[[GLOB]] to %[[S]] : !cir.ptr<!rec_S>
// CIR:           cir.copy %[[S]] align(4) to %[[REF_TMP]] align(4) : !cir.ptr<!rec_S>
// CIR:         }
// CIR:         %[[MEM:.*]] = cir.get_member %[[REF_TMP]][0] {name = "x"} : !cir.ptr<!rec_S> -> !cir.ptr<!s32i>
// CIR:         %{{.+}} = cir.load align(4) %[[MEM]] : !cir.ptr<!s32i>, !s32i

// LLVM-LABEL: define dso_local {{.*}}i32 @_Z21test_stmt_expr_lvaluev()
// LLVM:         %[[REF_TMP:.*]] = alloca %struct.S
// LLVM:         %[[TMP:.*]] = alloca %struct.S
// LLVM:         %[[S:.*]] = alloca %struct.S
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[S]], ptr align 4 @__const._Z21test_stmt_expr_lvaluev.s, i64 4, i1 false)
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[REF_TMP]], ptr align 4 %[[S]], i64 4, i1 false)
// LLVM:         %[[MEM:.*]] = getelementptr inbounds nuw %struct.S, ptr %[[REF_TMP]], i32 0, i32 0
// LLVM:         %{{.+}} = load i32, ptr %[[MEM]], align 4

// OGCG-LABEL: define dso_local {{.*}}i32 @_Z21test_stmt_expr_lvaluev()
// OGCG:         %[[REF_TMP:.*]] = alloca %struct.S
// OGCG:         %[[S:.*]] = alloca %struct.S
// OGCG:         call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[S]], ptr align 4 @__const._Z21test_stmt_expr_lvaluev.s, i64 4, i1 false)
// OGCG:         call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[REF_TMP]], ptr align 4 %[[S]], i64 4, i1 false)
// OGCG:         %[[MEM:.*]] = getelementptr inbounds nuw %struct.S, ptr %[[REF_TMP]], i32 0, i32 0
// OGCG:         %{{.+}} = load i32, ptr %[[MEM]], align 4
