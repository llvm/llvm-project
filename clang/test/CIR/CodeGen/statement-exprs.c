// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

int f19(void) {
  return ({ 3;;4;; });
}

// CIR: cir.func dso_local @f19() -> !s32i
// CIR:   %[[RETVAL:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:   %[[TMP:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["tmp"]
// CIR:   cir.scope {
// CIR:     %[[C3:.+]] = cir.const #cir.int<3> : !s32i
// CIR:     %[[C4:.+]] = cir.const #cir.int<4> : !s32i
// CIR:     cir.store {{.*}} %[[C4]], %[[TMP]] : !s32i, !cir.ptr<!s32i>
// CIR:   }
// CIR:   %[[TMP_VAL:.+]] = cir.load {{.*}} %[[TMP]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.store %[[TMP_VAL]], %[[RETVAL]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[RES:.+]] = cir.load %[[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[RES]] : !s32i

// LLVM: define dso_local i32 @f19()
// LLVM:   %[[VAR1:.+]] = alloca i32, i64 1
// LLVM:   %[[VAR2:.+]] = alloca i32, i64 1
// LLVM:   br label %[[LBL3:.+]]
// LLVM: [[LBL3]]:
// LLVM:     store i32 4, ptr %[[VAR2]]
// LLVM:     br label %[[LBL4:.+]]
// LLVM: [[LBL4]]:
// LLVM:     %[[V1:.+]] = load i32, ptr %[[VAR2]]
// LLVM:     store i32 %[[V1]], ptr %[[VAR1]]
// LLVM:     %[[RES:.+]] = load i32, ptr %[[VAR1]]
// LLVM:     ret i32 %[[RES]]

// OGCG: define dso_local i32 @f19()
// OGCG: entry:
// OGCG:   %[[TMP:.+]] = alloca i32
// OGCG:   store i32 4, ptr %[[TMP]]
// OGCG:   %[[TMP_VAL:.+]] = load i32, ptr %[[TMP]]
// OGCG:   ret i32 %[[TMP_VAL]]


int nested(void) {
  ({123;});
  {
    int bar = 987;
    return ({ ({ int asdf = 123; asdf; }); ({9999;}); });
  }
}

// CIR: cir.func dso_local @nested() -> !s32i
// CIR:   %[[RETVAL:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:   %[[TMP_OUTER:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["tmp"]
// CIR:   cir.scope {
// CIR:     %[[C123_OUTER:.+]] = cir.const #cir.int<123> : !s32i
// CIR:     cir.store {{.*}} %[[C123_OUTER]], %[[TMP_OUTER]] : !s32i, !cir.ptr<!s32i>
// CIR:   }
// CIR:   %[[LOAD_TMP_OUTER:.+]] = cir.load {{.*}} %[[TMP_OUTER]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.scope {
// CIR:     %[[BAR:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["bar", init]
// CIR:     %[[TMP_BARRET:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["tmp"]
// CIR:     %[[C987:.+]] = cir.const #cir.int<987> : !s32i
// CIR:     cir.store {{.*}} %[[C987]], %[[BAR]] : !s32i, !cir.ptr<!s32i>
// CIR:     cir.scope {
// CIR:       %[[TMP1:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["tmp"]
// CIR:       %[[TMP2:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["tmp"]
// CIR:       cir.scope {
// CIR:         %[[ASDF:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["asdf", init]
// CIR:         %[[C123_INNER:.+]] = cir.const #cir.int<123> : !s32i
// CIR:         cir.store {{.*}} %[[C123_INNER]], %[[ASDF]] : !s32i, !cir.ptr<!s32i>
// CIR:         %[[LOAD_ASDF:.+]] = cir.load {{.*}} %[[ASDF]] : !cir.ptr<!s32i>, !s32i
// CIR:         cir.store {{.*}} %[[LOAD_ASDF]], %[[TMP1]] : !s32i, !cir.ptr<!s32i>
// CIR:       }
// CIR:       %[[V1:.+]] = cir.load {{.*}} %[[TMP1]] : !cir.ptr<!s32i>, !s32i
// CIR:       cir.scope {
// CIR:         %[[C9999:.+]] = cir.const #cir.int<9999> : !s32i
// CIR:         cir.store {{.*}} %[[C9999]], %[[TMP2]] : !s32i, !cir.ptr<!s32i>
// CIR:       }
// CIR:       %[[V2:.+]] = cir.load {{.*}} %[[TMP2]] : !cir.ptr<!s32i>, !s32i
// CIR:       cir.store {{.*}} %[[V2]], %[[TMP_BARRET]] : !s32i, !cir.ptr<!s32i>
// CIR:     }
// CIR:     %[[BARRET_VAL:.+]] = cir.load {{.*}} %[[TMP_BARRET]] : !cir.ptr<!s32i>, !s32i
// CIR:     cir.store %[[BARRET_VAL]], %[[RETVAL]] : !s32i, !cir.ptr<!s32i>
// CIR:     %[[RES:.+]] = cir.load %[[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR:     cir.return %[[RES]] : !s32i
// CIR:   }
// CIR:   %[[FINAL_RES:.+]] = cir.load %[[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[FINAL_RES]] : !s32i

// LLVM: define dso_local i32 @nested()
// LLVM:   %[[VAR1:.+]] = alloca i32, i64 1
// LLVM:   %[[VAR2:.+]] = alloca i32, i64 1
// LLVM:   %[[VAR3:.+]] = alloca i32, i64 1
// LLVM:   %[[VAR4:.+]] = alloca i32, i64 1
// LLVM:   %[[VAR5:.+]] = alloca i32, i64 1
// LLVM:   %[[VAR6:.+]] = alloca i32, i64 1
// LLVM:   %[[VAR7:.+]] = alloca i32, i64 1
// LLVM:   br label %[[LBL8:.+]]
// LLVM: [[LBL8]]:
// LLVM:     store i32 123, ptr %[[VAR7]]
// LLVM:     br label %[[LBL9:.+]]
// LLVM: [[LBL9]]:
// LLVM:     br label %[[LBL10:.+]]
// LLVM: [[LBL10]]:
// LLVM:     store i32 987, ptr %[[VAR1]]
// LLVM:     br label %[[LBL11:.+]]
// LLVM: [[LBL11]]:
// LLVM:     br label %[[LBL12:.+]]
// LLVM: [[LBL12]]:
// LLVM:     store i32 123, ptr %[[VAR5]]
// LLVM:     %[[V1:.+]] = load i32, ptr %[[VAR5]]
// LLVM:     store i32 %[[V1]], ptr %[[VAR3]]
// LLVM:     br label %[[LBL14:.+]]
// LLVM: [[LBL14]]:
// LLVM:     br label %[[LBL15:.+]]
// LLVM: [[LBL15]]:
// LLVM:     store i32 9999, ptr %[[VAR4]]
// LLVM:     br label %[[LBL16:.+]]
// LLVM: [[LBL16]]:
// LLVM:     %[[V2:.+]] = load i32, ptr %[[VAR4]]
// LLVM:     store i32 %[[V2]], ptr %[[VAR2]]
// LLVM:     br label %[[LBL18:.+]]
// LLVM: [[LBL18]]:
// LLVM:     %[[V3:.+]] = load i32, ptr %[[VAR2]]
// LLVM:     store i32 %[[V3]], ptr %[[VAR6]]
// LLVM:     %[[RES:.+]] = load i32, ptr %[[VAR6]]
// LLVM:     ret i32 %[[RES]]

// OGCG: define dso_local i32 @nested()
// OGCG: entry:
// OGCG:   %[[TMP_OUTER:.+]] = alloca i32
// OGCG:   %[[BAR:.+]] = alloca i32
// OGCG:   %[[ASDF:.+]] = alloca i32
// OGCG:   %[[TMP1:.+]] = alloca i32
// OGCG:   %[[TMP2:.+]] = alloca i32
// OGCG:   %[[TMP3:.+]] = alloca i32
// OGCG:   store i32 123, ptr %[[TMP_OUTER]]
// OGCG:   %[[OUTER_VAL:.+]] = load i32, ptr %[[TMP_OUTER]]
// OGCG:   store i32 987, ptr %[[BAR]]
// OGCG:   store i32 123, ptr %[[ASDF]]
// OGCG:   %[[ASDF_VAL:.+]] = load i32, ptr %[[ASDF]]
// OGCG:   store i32 %[[ASDF_VAL]], ptr %[[TMP1]]
// OGCG:   %[[TMP1_VAL:.+]] = load i32, ptr %[[TMP1]]
// OGCG:   store i32 9999, ptr %[[TMP3]]
// OGCG:   %[[TMP3_VAL:.+]] = load i32, ptr %[[TMP3]]
// OGCG:   store i32 %[[TMP3_VAL]], ptr %[[TMP2]]
// OGCG:   %[[RES:.+]] = load i32, ptr %[[TMP2]]
// OGCG:   ret i32 %[[RES]]

void empty() {
  return ({;;;;});
}

// CIR: cir.func no_proto dso_local @empty()
// CIR-NEXT:   cir.return

// LLVM: define dso_local void @empty()
// LLVM:   ret void
// LLVM: }

// OGCG: define dso_local void @empty()
// OGCG:   ret void
// OGCG: }

void empty2() { ({ }); }

// CIR: @empty2
// CIR-NEXT: cir.return

// LLVM: @empty2()
// LLVM:   ret void
// LLVM: }

// OGCG: @empty2()
// OGCG:   ret void
// OGCG: }


// Yields an out-of-scope scalar.
void test2() { ({int x = 3; x; }); }
// CIR: @test2
// CIR: %[[RETVAL:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>
// CIR: cir.scope {
// CIR:   %[[VAR:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
//          [...]
// CIR:   %[[TMP:.+]] = cir.load{{.*}} %[[VAR]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.store{{.*}} %[[TMP]], %[[RETVAL]] : !s32i, !cir.ptr<!s32i>
// CIR: }
// CIR: %{{.+}} = cir.load{{.*}} %[[RETVAL]] : !cir.ptr<!s32i>, !s32i

// LLVM: define dso_local void @test2()
// LLVM:   %[[X:.+]] = alloca i32, i64 1
// LLVM:   %[[TMP:.+]] = alloca i32, i64 1
// LLVM:   br label %[[LBL3:.+]]
// LLVM: [[LBL3]]:
// LLVM:     store i32 3, ptr %[[X]]
// LLVM:     %[[X_VAL:.+]] = load i32, ptr %[[X]]
// LLVM:     store i32 %[[X_VAL]], ptr %[[TMP]]
// LLVM:     br label %[[LBL5:.+]]
// LLVM: [[LBL5]]:
// LLVM:     ret void

// OGCG: define dso_local void @test2()
// OGCG: entry:
// OGCG:   %[[X:.+]] = alloca i32
// OGCG:   %[[TMP:.+]] = alloca i32
// OGCG:   store i32 3, ptr %[[X]]
// OGCG:   %[[X_VAL:.+]] = load i32, ptr %[[X]]
// OGCG:   store i32 %[[X_VAL]], ptr %[[TMP]]
// OGCG:   %[[TMP_VAL:.+]] = load i32, ptr %[[TMP]]
// OGCG:   ret void

// Yields an aggregate.
struct S { int x; };
int test3() { return ({ struct S s = {1}; s; }).x; }
// CIR: cir.func no_proto dso_local @test3() -> !s32i
// CIR:   %[[RETVAL:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:   %[[YIELDVAL:.+]] = cir.scope {
// CIR:     %[[REF_TMP0:.+]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["ref.tmp0"]
// CIR:     %[[TMP:.+]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["tmp"]
// CIR:     cir.scope {
// CIR:       %[[S:.+]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["s", init]
// CIR:       %[[GEP_X_S:.+]] = cir.get_member %[[S]][0] {name = "x"} : !cir.ptr<!rec_S> -> !cir.ptr<!s32i>
// CIR:       %[[C1:.+]] = cir.const #cir.int<1> : !s32i
// CIR:       cir.store {{.*}} %[[C1]], %[[GEP_X_S]] : !s32i, !cir.ptr<!s32i>
// CIR:     }
// CIR:     %[[GEP_X_TMP:.+]] = cir.get_member %[[REF_TMP0]][0] {name = "x"} : !cir.ptr<!rec_S> -> !cir.ptr<!s32i>
// CIR:     %[[XVAL:.+]] = cir.load {{.*}} %[[GEP_X_TMP]] : !cir.ptr<!s32i>, !s32i
// CIR:     cir.yield %[[XVAL]] : !s32i
// CIR:   } : !s32i
// CIR:   cir.store %[[YIELDVAL]], %[[RETVAL]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[RES:.+]] = cir.load %[[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[RES]] : !s32i

// LLVM: define dso_local i32 @test3()
// LLVM:   %[[VAR1:.+]] = alloca %struct.S, i64 1
// LLVM:   %[[VAR2:.+]] = alloca %struct.S, i64 1
// LLVM:   %[[VAR3:.+]] = alloca %struct.S, i64 1
// LLVM:   %[[VAR4:.+]] = alloca i32, i64 1
// LLVM:   br label %[[LBL5:.+]]
// LLVM: [[LBL5]]:
// LLVM:     br label %[[LBL6:.+]]
// LLVM: [[LBL6]]:
// LLVM:     %[[GEP_S:.+]] = getelementptr %struct.S, ptr %[[VAR3]], i32 0, i32 0
// LLVM:     store i32 1, ptr %[[GEP_S]]
// LLVM:     br label %[[LBL8:.+]]
// LLVM: [[LBL8]]:
// LLVM:     %[[GEP_VAR1:.+]] = getelementptr %struct.S, ptr %[[VAR1]], i32 0, i32 0
// LLVM:     %[[LOAD_X:.+]] = load i32, ptr %[[GEP_VAR1]]
// LLVM:     br label %[[LBL11:.+]]
// LLVM: [[LBL11]]:
// LLVM:     %[[PHI:.+]] = phi i32 [ %[[LOAD_X]], %[[LBL8]] ]
// LLVM:     store i32 %[[PHI]], ptr %[[VAR4]]
// LLVM:     %[[RES:.+]] = load i32, ptr %[[VAR4]]
// LLVM:     ret i32 %[[RES]]

// OGCG: define dso_local i32 @test3()
// OGCG: entry:
// OGCG:   %[[REF_TMP:.+]] = alloca %struct.S
// OGCG:   %[[S:.+]] = alloca %struct.S
// OGCG:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[S]], ptr align 4 @__const.test3.s, i64 4, i1 false)
// OGCG:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[REF_TMP]], ptr align 4 %[[S]], i64 4, i1 false)
// OGCG:   %[[GEP:.+]] = getelementptr inbounds nuw %struct.S, ptr %[[REF_TMP]], i32 0, i32 0
// OGCG:   %[[XVAL:.+]] = load i32, ptr %[[GEP]]
// OGCG:   ret i32 %[[XVAL]]

// Expression is wrapped in an expression attribute (just ensure it does not crash).
void test4(int x) { ({[[gsl::suppress("foo")]] x;}); }
// CIR: @test4
// LLVM: @test4
// OGCG: @test4
