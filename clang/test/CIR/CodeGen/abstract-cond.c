// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// ?: in "lvalue"
struct s6 { int f0; };
int f6(int a0, struct s6 a1, struct s6 a2) {
  return (a0 ? a1 : a2).f0;
}

// CIR-LABEL: @f6
// CIR:  %[[A0:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a0"
// CIR:  %[[A1:.*]] = cir.alloca !ty_s6_, !cir.ptr<!ty_s6_>, ["a1"
// CIR:  %[[A2:.*]] = cir.alloca !ty_s6_, !cir.ptr<!ty_s6_>, ["a2"
// CIR:  %[[TMP:.*]] = cir.alloca !ty_s6_, !cir.ptr<!ty_s6_>, ["tmp"] {alignment = 4 : i64}
// CIR:  %[[LOAD_A0:.*]] = cir.load %[[A0]] : !cir.ptr<!s32i>, !s32i
// CIR:  %[[COND:.*]] = cir.cast(int_to_bool, %[[LOAD_A0]] : !s32i), !cir.bool
// CIR:  cir.if %[[COND]] {
// CIR:    cir.copy %[[A1]] to %[[TMP]] : !cir.ptr<!ty_s6_>
// CIR:  } else {
// CIR:    cir.copy %[[A2]] to %[[TMP]] : !cir.ptr<!ty_s6_>
// CIR:  }
// CIR:  cir.get_member %[[TMP]][0] {name = "f0"} : !cir.ptr<!ty_s6_> -> !cir.ptr<!s32i>

// LLVM-LABEL: @f6
// LLVM:    %[[LOAD_A0:.*]] = load i32, ptr {{.*}}
// LLVM:    %[[COND:.*]] = icmp ne i32 %[[LOAD_A0]], 0
// LLVM:    br i1 %[[COND]], label %[[A1_PATH:.*]], label %[[A2_PATH:.*]],
// LLVM:  [[A2_PATH]]:
// LLVM:    call void @llvm.memcpy.p0.p0.i32(ptr %[[TMP:.*]], ptr {{.*}}, i32 4, i1 false)
// LLVM:    br label %[[EXIT:[a-z0-9]+]]
// LLVM:  [[A1_PATH]]:
// LLVM:    call void @llvm.memcpy.p0.p0.i32(ptr %[[TMP]], ptr {{.*}}, i32 4, i1 false)
// LLVM:    br label %[[EXIT]]
// LLVM:  [[EXIT]]:
// LLVM:    getelementptr {{.*}}, ptr %[[TMP]], i32 0, i32 0