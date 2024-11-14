// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu  -fclangir -fclangir-call-conv-lowering -emit-cir-flat -mmlir --mlir-print-ir-after=cir-call-conv-lowering %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu  -fclangir -emit-llvm %s -o %t.ll -fclangir-call-conv-lowering
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// CIR: !ty_U = !cir.struct<union "U" {!s32i, !s32i, !s32i}>
// LLVM: %union.U = type { i32 }
typedef union {
  int a, b, c;
} U;

// CIR: cir.func @foo(%arg0: !u64i
// CIR: %[[#V0:]] = cir.alloca !ty_U, !cir.ptr<!ty_U>, [""] {alignment = 4 : i64}
// CIR: %[[#V1:]] = cir.cast(integral, %arg0 : !u64i), !u32i
// CIR: %[[#V2:]] = cir.cast(bitcast, %[[#V0]] : !cir.ptr<!ty_U>), !cir.ptr<!u32i>
// CIR: cir.store %[[#V1]], %[[#V2]] : !u32i, !cir.ptr<!u32i>
// CIR: cir.return

// LLVM: void @foo(i64 %[[#V0:]]
// LLVM: %[[#V2:]] = alloca %union.U, i64 1, align 4
// LLVM: %[[#V3:]] = trunc i64 %[[#V0]] to i32
// LLVM: store i32 %[[#V3]], ptr %[[#V2]], align 4
// LLVM: ret void
void foo(U u) {}

// CIR: cir.func no_proto @init() -> !u32i
// CIR: %[[#V0:]] = cir.alloca !ty_U, !cir.ptr<!ty_U>, ["__retval"] {alignment = 4 : i64}
// CIR: %[[#V1:]] = cir.load %[[#V0]] : !cir.ptr<!ty_U>, !ty_U
// CIR: %[[#V2:]] = cir.cast(bitcast, %[[#V0]] : !cir.ptr<!ty_U>), !cir.ptr<!u32i>
// CIR: %[[#V3:]] = cir.load %[[#V2]] : !cir.ptr<!u32i>, !u32i
// CIR: cir.return %[[#V3]] : !u32i

// LLVM: i32 @init()
// LLVM: %[[#V1:]] = alloca %union.U, i64 1, align 4
// LLVM: %[[#V2:]] = load %union.U, ptr %[[#V1]], align 4
// LLVM: %[[#V3:]] = load i32, ptr %[[#V1]], align 4
// LLVM: ret i32 %[[#V3]]
U init() {
  U u;
  return u;
}

typedef union {

  struct {
    short a;
    char b;
    char c;
  };

  int x;
} A;

void passA(A x) {}

// CIR: cir.func {{.*@callA}}()
// CIR:   %[[#V0:]] = cir.alloca !ty_A, !cir.ptr<!ty_A>, ["x"] {alignment = 4 : i64}
// CIR:   %[[#V1:]] = cir.cast(bitcast, %[[#V0:]] : !cir.ptr<!ty_A>), !cir.ptr<!s32i>
// CIR:   %[[#V2:]] = cir.load %[[#V1]] : !cir.ptr<!s32i>, !s32i
// CIR:   %[[#V3:]] = cir.cast(integral, %[[#V2]] : !s32i), !u64i
// CIR:   cir.call @passA(%[[#V3]]) : (!u64i) -> ()

// LLVM: void @callA()
// LLVM:   %[[#V0:]] = alloca %union.A, i64 1, align 4
// LLVM:   %[[#V1:]] = load i32, ptr %[[#V0]], align 4
// LLVM:   %[[#V2:]] = sext i32 %[[#V1]] to i64
// LLVM:   call void @passA(i64 %[[#V2]])
void callA() {
  A x;
  passA(x);
}
