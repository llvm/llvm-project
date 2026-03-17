// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// ?: in "lvalue"
struct s6 { int f0; };
int test_agg_cond(int a0, struct s6 a1, struct s6 a2) {
  return (a0 ? a1 : a2).f0;
}

// CIR: cir.func {{.*}} @test_agg_cond
// CIR:  %[[A0:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a0"
// CIR:  %[[A1:.*]] = cir.alloca !rec_s6, !cir.ptr<!rec_s6>, ["a1"
// CIR:  %[[A2:.*]] = cir.alloca !rec_s6, !cir.ptr<!rec_s6>, ["a2"
// CIR:  cir.scope {
// CIR:    %[[TMP:.*]] = cir.alloca !rec_s6, !cir.ptr<!rec_s6>, ["ref.tmp0"]
// CIR:    %[[LOAD_A0:.*]] = cir.load{{.*}} %[[A0]] : !cir.ptr<!s32i>, !s32i
// CIR:    %[[COND:.*]] = cir.cast int_to_bool %[[LOAD_A0]] : !s32i -> !cir.bool
// CIR:    cir.if %[[COND]] {
// CIR:      cir.copy %[[A1]] to %[[TMP]] : !cir.ptr<!rec_s6>
// CIR:    } else {
// CIR:      cir.copy %[[A2]] to %[[TMP]] : !cir.ptr<!rec_s6>
// CIR:    }
// CIR:    cir.get_member %[[TMP]][0] {name = "f0"} : !cir.ptr<!rec_s6> -> !cir.ptr<!s32i>

// LLVM: define {{.*}} i32 @test_agg_cond
// LLVM:    %[[LOAD_A0:.*]] = load i32, ptr {{.*}}
// LLVM:    %[[COND:.*]] = icmp ne i32 %[[LOAD_A0]], 0
// LLVM:    br i1 %[[COND]], label %[[A1_PATH:.*]], label %[[A2_PATH:.*]]
// LLVM:  [[A1_PATH]]:
// LLVM:    call void @llvm.memcpy.p0.p0.i64(ptr %[[TMP:.*]], ptr {{.*}}, i64 4, i1 false)
// LLVM:    br label %[[EXIT:.*]]
// LLVM:  [[A2_PATH]]:
// LLVM:    call void @llvm.memcpy.p0.p0.i64(ptr %[[TMP]], ptr {{.*}}, i64 4, i1 false)
// LLVM:    br label %[[EXIT]]
// LLVM:  [[EXIT]]:
// LLVM:    getelementptr {{.*}}, ptr %[[TMP]], i32 0, i32 0

// OGCG: define {{.*}} i32 @test_agg_cond
// OGCG:    %[[LOAD_A0:.*]] = load i32, ptr {{.*}}
// OGCG:    %[[COND:.*]] = icmp ne i32 %[[LOAD_A0]], 0
// OGCG:    br i1 %[[COND]], label %[[A1_PATH:.*]], label %[[A2_PATH:.*]]
// OGCG:  [[A1_PATH]]:
// OGCG:    call void @llvm.memcpy.p0.p0.i64(ptr {{.*}} %[[TMP:.*]], ptr {{.*}}, i64 4, i1 false)
// OGCG:    br label %[[EXIT:.*]]
// OGCG:  [[A2_PATH]]:
// OGCG:    call void @llvm.memcpy.p0.p0.i64(ptr {{.*}} %[[TMP]], ptr {{.*}}, i64 4, i1 false)
// OGCG:    br label %[[EXIT]]
// OGCG:  [[EXIT]]:
// OGCG:    getelementptr {{.*}}, ptr %[[TMP]], i32 0, i32 0

void foo(void);
int test_stmt_expr(int flag, struct s6 a1, struct s6 a2) {
  return (flag
            ? ({ struct s6 t = a1; foo(); t; })   // statement expression arm
            : a2)
      .f0;
}

// CIR: cir.func {{.*}} @test_stmt_expr
// CIR:  %[[FLAG:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["flag"
// CIR:  %[[A1:.*]] = cir.alloca !rec_s6, !cir.ptr<!rec_s6>, ["a1"
// CIR:  %[[A2:.*]] = cir.alloca !rec_s6, !cir.ptr<!rec_s6>, ["a2"
// CIR:  cir.scope {
// CIR:    %[[TMP:.*]] = cir.alloca !rec_s6, !cir.ptr<!rec_s6>, ["ref.tmp0"]
// CIR:    %[[LOAD_FLAG:.*]] = cir.load{{.*}} %[[FLAG]] : !cir.ptr<!s32i>, !s32i
// CIR:    %[[COND:.*]] = cir.cast int_to_bool %[[LOAD_FLAG]] : !s32i -> !cir.bool
// CIR:    cir.if %[[COND]] {
// CIR:      %[[STMT_TMP:.*]] = cir.alloca !rec_s6, !cir.ptr<!rec_s6>, ["tmp"]
// CIR:      cir.scope {
// CIR:        %[[T:.*]] = cir.alloca !rec_s6, !cir.ptr<!rec_s6>, ["t", init]
// CIR:        cir.copy %[[A1]] to %[[T]] : !cir.ptr<!rec_s6>
// CIR:        cir.call @foo() : () -> ()
// CIR:        cir.copy %[[T]] to %[[TMP]] : !cir.ptr<!rec_s6>
// CIR:      }
// CIR:    } else {
// CIR:      cir.copy %[[A2]] to %[[TMP]] : !cir.ptr<!rec_s6>
// CIR:    }
// CIR:    cir.get_member %[[TMP]][0] {name = "f0"} : !cir.ptr<!rec_s6> -> !cir.ptr<!s32i>

// LLVM: define {{.*}} i32 @test_stmt_expr
// LLVM:    %[[LOAD_FLAG:.*]] = load i32, ptr {{.*}}
// LLVM:    %[[COND:.*]] = icmp ne i32 %[[LOAD_FLAG]], 0
// LLVM:    br i1 %[[COND]], label %[[TRUE:.*]], label %[[FALSE:.*]]
// LLVM:  [[TRUE]]:
// LLVM:    br label %[[STMT_BODY:.*]]
// LLVM:  [[STMT_BODY]]:
// LLVM:    call void @llvm.memcpy.p0.p0.i64(ptr %[[T:.*]], ptr {{.*}}, i64 4, i1 false)
// LLVM:    call void @foo()
// LLVM:    call void @llvm.memcpy.p0.p0.i64(ptr %[[TMP:.*]], ptr %[[T]], i64 4, i1 false)
// LLVM:    br label %[[JOIN_FROM_TRUE:.*]]
// LLVM:  [[JOIN_FROM_TRUE]]:
// LLVM:    br label %[[EXIT:.*]]
// LLVM:  [[FALSE]]:
// LLVM:    call void @llvm.memcpy.p0.p0.i64(ptr %[[TMP]], ptr {{.*}}, i64 4, i1 false)
// LLVM:    br label %[[EXIT]]
// LLVM:  [[EXIT]]:
// LLVM:    getelementptr %struct.s6, ptr %[[TMP]], i32 0, i32 0

// OGCG: define {{.*}} i32 @test_stmt_expr
// OGCG:    %[[LOAD_FLAG:.*]] = load i32, ptr {{.*}}
// OGCG:    %[[COND:.*]] = icmp ne i32 %[[LOAD_FLAG]], 0
// OGCG:    br i1 %[[COND]], label %[[TRUE:.*]], label %[[FALSE:.*]]
// OGCG:  [[TRUE]]:
// OGCG:    call void @llvm.memcpy.p0.p0.i64(ptr {{.*}} %[[T:.*]], ptr {{.*}}, i64 4, i1 false)
// OGCG:    call void @foo()
// OGCG:    call void @llvm.memcpy.p0.p0.i64(ptr {{.*}} %[[TMP:.*]], ptr {{.*}} %[[T]], i64 4, i1 false)
// OGCG:    br label %[[EXIT:.*]]
// OGCG:  [[FALSE]]:
// OGCG:    call void @llvm.memcpy.p0.p0.i64(ptr {{.*}} %[[TMP]], ptr {{.*}}, i64 4, i1 false)
// OGCG:    br label %[[EXIT]]
// OGCG:  [[EXIT]]:
// OGCG:    getelementptr {{.*}}, ptr %[[TMP]], i32 0, i32 0
