// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void func(int len) {
  int arr[len];
  int e = arr[0];
}

// CIR: %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["len", init]
// CIR: cir.store %arg0, %0 : !s32i, !cir.ptr<!s32i>
// CIR: %1 = cir.load %0 : !cir.ptr<!s32i>, !s32i
// CIR: %2 = cir.cast(integral, %1 : !s32i), !u64i
// CIR: %3 = cir.alloca !s32i, !cir.ptr<!s32i>, %2 : !u64i, ["vla"]
// CIR: %4 = cir.alloca !s32i, !cir.ptr<!s32i>, ["e", init]
// CIR: %5 = cir.const #cir.int<0> : !s32i
// CIR: %6 = cir.ptr_stride(%3 : !cir.ptr<!s32i>, %5 : !s32i), !cir.ptr<!s32i>
// CIR: %7 = cir.load %6 : !cir.ptr<!s32i>, !s32i
// CIR: cir.store %7, %4 : !s32i, !cir.ptr<!s32i>

// LLVM: %2 = alloca i32, i64 1
// LLVM: store i32 %0, ptr %2
// LLVM: %3 = load i32, ptr %2
// LLVM: %4 = sext i32 %3 to i64
// LLVM: %5 = alloca i32, i64 %4
// LLVM: %6 = alloca ptr, i64 1
// LLVM: %7 = call ptr @llvm.stacksave.p0()
// LLVM: store ptr %7, ptr %6
// LLVM: %8 = alloca i32, i64 1
// LLVM: %9 = getelementptr i32, ptr %5, i64 0
// LLVM: %10 = load i32, ptr %9
// LLVM: store i32 %10, ptr %8
// LLVM: %11 = load ptr, ptr %6
// LLVM: call void @llvm.stackrestore.p0(ptr %11)

// OGCG: %len.addr = alloca i32
// OGCG: %saved_stack = alloca ptr
// OGCG: %__vla_expr0 = alloca i64
// OGCG: %e = alloca i32
// OGCG: store i32 %len, ptr %len.addr
// OGCG: %0 = load i32, ptr %len.addr
// OGCG: %1 = zext i32 %0 to i64
// OGCG: %2 = call ptr @llvm.stacksave.p0()
// OGCG: store ptr %2, ptr %saved_stack
// OGCG: %vla = alloca i32, i64 %1
// OGCG: store i64 %1, ptr %__vla_expr0
// OGCG: %arrayidx = getelementptr inbounds i32, ptr %vla, i64 0
// OGCG: %3 = load i32, ptr %arrayidx
// OGCG: store i32 %3, ptr %e
// OGCG: %4 = load ptr, ptr %saved_stack
// OGCG: call void @llvm.stackrestore.p0(ptr %4)

void func2(short width, int data[][width]) {}

// CIR: %0 = cir.alloca !s16i, !cir.ptr<!s16i>, ["width", init]
// CIR: %1 = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["data", init]
// CIR: cir.store %arg0, %0 : !s16i, !cir.ptr<!s16i>
// CIR: cir.store %arg1, %1 : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>

// LLVM: %3 = alloca i16, i64 1
// LLVM: %4 = alloca ptr, i64 1
// LLVM: store i16 %0, ptr %3
// LLVM: store ptr %1, ptr %4

// OGCG: %width.addr = alloca i16
// OGCG: %data.addr = alloca ptr
// OGCG: store i16 %width, ptr %width.addr
// OGCG: store ptr %data, ptr %data.addr
