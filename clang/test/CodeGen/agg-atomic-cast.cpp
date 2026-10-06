// RUN: %clang_cc1 %s -triple=x86_64-linux -emit-llvm -o - | FileCheck %s

struct S {
    char a;
};

void static_cast_non_atomic_to_atomic() {
  _Atomic struct S sa;
  S s;
  sa = static_cast<_Atomic(struct S)>(s);
}

// CHECK: %[[SA_ADDR:.*]] = alloca %struct.S, align 1
// CHECK: %[[S_ADDR:.*]] = alloca %struct.S, align 1
// CHECK: %[[TMP_ADDR:.*]] = alloca %struct.S, align 1
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[TMP_ADDR]], ptr align 1 %[[S_ADDR]], i64 1, i1 false)
// CHECK: %[[TMP:.*]] = load i8, ptr %[[TMP_ADDR]], align 1
// CHECK: store atomic i8 %[[TMP]], ptr %[[SA_ADDR]] seq_cst, align 1

void non_atomic_to_atomic_cast() {
  S s;
  _Atomic(S) sa = (_Atomic(S)) s;
}

// CHECK: %[[S_ADDR:.*]] = alloca %struct.S, align 1
// CHECK: %[[SA_ADDR:.*]] = alloca %struct.S, align 1
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[SA_ADDR]], ptr align 1 %[[S_ADDR]], i64 1, i1 false)
