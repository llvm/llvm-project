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
