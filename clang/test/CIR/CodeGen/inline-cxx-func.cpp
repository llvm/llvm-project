// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct S {
  int Member;

  int InlineFunc() {
    return Member;
  }
};

// CIR: !rec_S = !cir.record<struct "S" {!s32i}>
// LLVM: %struct.S = type { i32 }
// OGCG: %struct.S = type { i32 }

// CIR: cir.func{{.*}} @_ZN1S10InlineFuncEv(%arg0: !cir.ptr<!rec_S> {{.*}}) -> !s32i
// CIR:   %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>, ["this", init]
// CIR:   %[[RET_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:   cir.store %arg0, %[[THIS_ADDR]] : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>
// CIR:   %[[THIS:.*]] = cir.load %[[THIS_ADDR]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:   %[[MEMBER_ADDR:.*]] = cir.get_member %[[THIS]][0] {name = "Member"} : !cir.ptr<!rec_S> -> !cir.ptr<!s32i>
// CIR:   %[[MEMBER:.*]] = cir.load{{.*}} %[[MEMBER_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.store %[[MEMBER]], %[[RET_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[RET_VAL:.*]] = cir.load %[[RET_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[RET_VAL]] : !s32i

// LLVM: define{{.*}} i32 @_ZN1S10InlineFuncEv(ptr %[[ARG0:.*]])
// LLVM:   %[[THIS_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[RET_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM:   store ptr %[[ARG0]], ptr %[[THIS_ADDR]], align 8
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]], align 8
// LLVM:   %[[MEMBER_ADDR:.*]] = getelementptr %struct.S, ptr %[[THIS]], i32 0, i32 0
// LLVM:   %[[MEMBER:.*]] = load i32, ptr %[[MEMBER_ADDR]], align 4
// LLVM:   store i32 %[[MEMBER]], ptr %[[RET_ADDR]], align 4
// LLVM:   %[[RET_VAL:.*]] = load i32, ptr %[[RET_ADDR]], align 4
// LLVM:   ret i32 %[[RET_VAL]]

// The inlined function is defined after use() in OGCG

void use() {
  S s;
  s.InlineFunc();
}

// CIR: cir.func{{.*}} @_Z3usev()
// CIR:   %[[S_ADDR:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["s"]
// CIR:   %[[RET_VAL:.*]] = cir.call @_ZN1S10InlineFuncEv(%[[S_ADDR]]) : (!cir.ptr<!rec_S>) -> !s32i
// CIR:   cir.return

// LLVM: define{{.*}} void @_Z3usev()
// LLVM:   %[[S_ADDR:.*]] = alloca %struct.S
// LLVM:   %[[RET_VAL:.*]] = call i32 @_ZN1S10InlineFuncEv(ptr %[[S_ADDR]])
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z3usev()
// OGCG:   %[[S_ADDR:.*]] = alloca %struct.S
// OGCG:   %[[RET_VAL:.*]] = call{{.*}} i32 @_ZN1S10InlineFuncEv(ptr{{.*}} %[[S_ADDR]])
// OGCG:   ret void

// OGCG: define{{.*}} i32 @_ZN1S10InlineFuncEv(ptr{{.*}} %[[ARG0:.*]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr, align 8
// OGCG:   store ptr %[[ARG0]], ptr %[[THIS_ADDR]], align 8
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]], align 8
// OGCG:   %[[MEMBER_ADDR:.*]] = getelementptr inbounds nuw %struct.S, ptr %[[THIS]], i32 0, i32 0
// OGCG:   %[[MEMBER:.*]] = load i32, ptr %[[MEMBER_ADDR]], align 4
// OGCG:   ret i32 %[[MEMBER]]
