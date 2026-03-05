// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm -fno-clangir-call-conv-lowering %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// ?: in "lvalue"
struct s6 { int f0; };
int f6(int a0, struct s6 a1, struct s6 a2) {
  return (a0 ? a1 : a2).f0;
}

// CIR-LABEL: @f6
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

// LLVM-LABEL: @f6
// LLVM:    %[[LOAD_A0:.*]] = load i32, ptr {{.*}}
// LLVM:    %[[COND:.*]] = icmp ne i32 %[[LOAD_A0]], 0
// LLVM:    br i1 %[[COND]], label %[[A1_PATH:.*]], label %[[A2_PATH:.*]]
// LLVM:  [[A1_PATH]]:
// LLVM:    call void @llvm.memcpy.p0.p0.i32(ptr %[[TMP:.*]], ptr {{.*}}, i32 4, i1 false)
// LLVM:    br label %[[EXIT:[a-z0-9]+]]
// LLVM:  [[A2_PATH]]:
// LLVM:    call void @llvm.memcpy.p0.p0.i32(ptr %[[TMP]], ptr {{.*}}, i32 4, i1 false)
// LLVM:    br label %[[EXIT]]
// LLVM:  [[EXIT]]:
// LLVM:    getelementptr {{.*}}, ptr %[[TMP]], i32 0, i32 0
