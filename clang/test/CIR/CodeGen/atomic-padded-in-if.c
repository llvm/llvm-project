// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

struct T {
  char a, b, c;
}; // size 3 => padded atomic representation

void to_atomic(int c, struct T v) {
  if (c) {
    _Atomic(struct T) a = v;
  }
}

// CIR-LABEL: cir.func{{.*}} @to_atomic(
// CIR:   %[[V:.*]] = cir.alloca "v" {{.*}} : !cir.ptr<!rec_T>
// CIR:   cir.if %{{.*}} {
// CIR:     %[[A:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!rec_anon_struct>
// CIR:     cir.libc.memset {{.*}} : !cir.ptr<!void>, !u8i, !u64i
// CIR:     %[[VALUE:.*]] = cir.get_member %[[A]][0] {name = "value_addr"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!rec_T>
// CIR:     cir.copy %[[V]] align(1) to %[[VALUE]] align(4) : !cir.ptr<!rec_T>

// LLVM-LABEL: define dso_local void @to_atomic(
// LLVM:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
// LLVM:   call void @llvm.memset.p0.i64(ptr align 4 %[[A:.*]], i8 0, i64 4, i1 false)
// LLVM:   %[[VALUE:.*]] = getelementptr inbounds nuw { %struct.T, [1 x i8] }, ptr %[[A]], i32 0, i32 0
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[VALUE]], ptr align 1 %{{.*}}, i64 3, i1 false)

void from_atomic(int c, _Atomic(struct T) *p, struct T *r) {
  if (c) {
    *r = *p;
  }
}

// CIR-LABEL: cir.func{{.*}} @from_atomic(
// CIR:   cir.if %{{.*}} {
// CIR:     %[[TMP:.*]] = cir.alloca "tmp" {{.*}} : !cir.ptr<!rec_anon_struct>
// CIR:     %[[R:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!rec_T>>, !cir.ptr<!rec_T>
// CIR:     %[[ATOMIC:.*]] = cir.load {{.*}} atomic(seq_cst) %{{.*}} : !cir.ptr<!u32i>, !u32i
// CIR:     %[[TMP_U32:.*]] = cir.cast bitcast %[[TMP]] : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR:     cir.store {{.*}} %[[ATOMIC]], %[[TMP_U32]] : !u32i, !cir.ptr<!u32i>
// CIR:     %[[VALUE:.*]] = cir.get_member %[[TMP]][0] {name = "value_addr"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!rec_T>
// CIR:     cir.copy %[[VALUE]] align(4) to %[[R]] align(1) : !cir.ptr<!rec_T>

// LLVM-LABEL: define dso_local void @from_atomic(
// LLVM:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
// LLVM:   %[[ATOMIC:.*]] = load atomic i32, ptr %{{.*}} seq_cst, align 4
// LLVM:   store i32 %[[ATOMIC]], ptr %[[TMP:.*]], align 4
// LLVM:   %[[VALUE:.*]] = getelementptr inbounds nuw { %struct.T, [1 x i8] }, ptr %[[TMP]], i32 0, i32 0
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %{{.*}}, ptr align 4 %[[VALUE]], i64 3, i1 false)

void atomic_init(int c, _Atomic(struct T) *p, struct T v) {
  if (c) {
    __c11_atomic_init(p, v);
  }
}

// CIR-LABEL: cir.func{{.*}} @atomic_init(
// CIR:   %[[V:.*]] = cir.alloca "v" {{.*}} : !cir.ptr<!rec_T>
// CIR:   cir.if %{{.*}} {
// CIR:     %[[P:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!rec_anon_struct>>, !cir.ptr<!rec_anon_struct>
// CIR:     cir.libc.memset {{.*}} : !cir.ptr<!void>, !u8i, !u64i
// CIR:     %[[VALUE:.*]] = cir.get_member %[[P]][0] {name = "value"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!rec_T>
// CIR:     cir.copy %[[V]] align(1) to %[[VALUE]] align(4) : !cir.ptr<!rec_T>

// LLVM-LABEL: define dso_local void @atomic_init(
// LLVM:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
// LLVM:   call void @llvm.memset.p0.i64(ptr align 4 %[[P:.*]], i8 0, i64 4, i1 false)
// LLVM:   %[[VALUE:.*]] = getelementptr inbounds nuw { %struct.T, [1 x i8] }, ptr %[[P]], i32 0, i32 0
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[VALUE]], ptr align 1 %{{.*}}, i64 3, i1 false)
