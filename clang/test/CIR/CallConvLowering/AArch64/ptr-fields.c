// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu  -fclangir -fclangir-call-conv-lowering -emit-cir-flat -mmlir --mlir-print-ir-after=cir-call-conv-lowering %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu  -fclangir -emit-llvm %s -o %t.ll -fclangir-call-conv-lowering
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

typedef int (*myfptr)(int);

typedef struct {
  myfptr f;
} A;

int foo(int x) { return x; }

// CIR: cir.func @passA(%arg0: !u64i
// CIR: %[[#V0:]] = cir.alloca !ty_A, !cir.ptr<!ty_A>, [""] {alignment = 4 : i64}
// CIR: %[[#V1:]] = cir.cast(bitcast, %[[#V0]] : !cir.ptr<!ty_A>), !cir.ptr<!u64i>
// CIR: cir.store %arg0, %[[#V1]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[#V2:]] = cir.get_global @foo : !cir.ptr<!cir.func<(!s32i) -> !s32i>>
// CIR: %[[#V3:]] = cir.get_member %[[#V0]][0] {name = "f"} : !cir.ptr<!ty_A> -> !cir.ptr<!cir.ptr<!cir.func<(!s32i) -> !s32i>>>
// CIR: cir.store %[[#V2]], %[[#V3]] : !cir.ptr<!cir.func<(!s32i) -> !s32i>>, !cir.ptr<!cir.ptr<!cir.func<(!s32i) -> !s32i>>>
// CIR: cir.return

// LLVM: void @passA(i64 %[[#V0:]])
// LLVM: %[[#V2:]] = alloca %struct.A, i64 1, align 4
// LLVM: store i64 %[[#V0]], ptr %[[#V2]], align 8
// LLVM: %[[#V3:]] = getelementptr %struct.A, ptr %[[#V2]], i32 0, i32 0
// LLVM: store ptr @foo, ptr %[[#V3]], align 8
// LLVM: ret void
void passA(A a) { a.f = foo; }

typedef struct {
  int a;
} S_1;

typedef struct {
  S_1* s;
} S_2;

// CIR: cir.func @passB(%arg0: !u64i
// CIR: %[[#V0:]]  = cir.alloca !ty_S_2_, !cir.ptr<!ty_S_2_>, [""] {alignment = 4 : i64}
// CIR: %[[#V1:]]  = cir.cast(bitcast, %[[#V0]]  : !cir.ptr<!ty_S_2_>), !cir.ptr<!u64i>
// CIR: cir.store %arg0, %[[#V1]]  : !u64i, !cir.ptr<!u64i>
// CIR: cir.return

// LLVM: void @passB(i64 %[[#V0:]])
// LLVM: %[[#V2:]]  = alloca %struct.S_2, i64 1, align 4
// LLVM: store i64 %[[#V0]], ptr %[[#V2]], align 8
// LLVM: ret void
void passB(S_2 s) {}
