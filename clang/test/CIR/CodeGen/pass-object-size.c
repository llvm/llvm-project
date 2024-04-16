// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
void b(void *__attribute__((pass_object_size(0))));
void e(void *__attribute__((pass_object_size(2))));
void c() {
  int a;
  int d[a];
  b(d);
  e(d);
}

// CIR: cir.func no_proto @c()
// CIR: [[TMP0:%.*]] = cir.alloca !s32i, cir.ptr <!s32i>, %{{[0-9]+}} : !u64i, ["vla"] {alignment = 16 : i64}
// CIR: [[TMP1:%.*]] = cir.cast(bitcast, [[TMP0]] : !cir.ptr<!s32i>), !cir.ptr<!void>
// CIR-NEXT: [[TMP2:%.*]] = cir.objsize([[TMP1]] : <!void>, max) -> !u64i
// CIR-NEXT: cir.call @b([[TMP1]], [[TMP2]]) : (!cir.ptr<!void>, !u64i) -> ()
// CIR: [[TMP3:%.*]] = cir.cast(bitcast, [[TMP0]] : !cir.ptr<!s32i>), !cir.ptr<!void>
// CIR: [[TMP4:%.*]] = cir.objsize([[TMP3]] : <!void>, min) -> !u64i
// CIR-NEXT: cir.call @e([[TMP3]], [[TMP4]]) : (!cir.ptr<!void>, !u64i) -> ()

// LLVM: define void @c()
// LLVM: [[TMP0:%.*]] = alloca i32, i64 %{{[0-9]+}},
// LLVM: [[TMP1:%.*]] = call i64 @llvm.objectsize.i64.p0(ptr [[TMP0]], i1 false, i1 true, i1 false),
// LLVM-NEXT: call void @b(ptr [[TMP0]], i64 [[TMP1]])
// LLVM: [[TMP2:%.*]] = call i64 @llvm.objectsize.i64.p0(ptr [[TMP0]], i1 true, i1 true, i1 false),
// LLVM-NEXT: call void @e(ptr [[TMP0]], i64 [[TMP2]])
