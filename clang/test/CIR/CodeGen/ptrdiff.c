// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

int addrcmp(const void* a, const void* b) {
  // CIR-LABEL: addrcmp
  // CIR: %[[R:.*]] = cir.ptr_diff
  // CIR: cir.cast integral %[[R]] : !s64i -> !s32i

  // LLVM-LABEL: define dso_local i32 @addrcmp(
  // LLVM: %[[PTR_A:.*]] = ptrtoint ptr {{.*}} to i64
  // LLVM: %[[PTR_B:.*]] = ptrtoint ptr {{.*}} to i64
  // LLVM: %[[SUB:.*]] = sub i64 %[[PTR_A]], %[[PTR_B]]
  // LLVM-NOT: sdiv
  // LLVM: trunc i64 %[[SUB]] to i32

  // OGCG-LABEL: define dso_local i32 @addrcmp(
  // OGCG: %[[PTR_A:.*]] = ptrtoint ptr {{.*}} to i64
  // OGCG: %[[PTR_B:.*]] = ptrtoint ptr {{.*}} to i64
  // OGCG: %[[SUB:.*]] = sub i64 %[[PTR_A]], %[[PTR_B]]
  // OGCG-NOT: sdiv
  // OGCG: trunc i64 %[[SUB]] to i32
  return *(const void**)a - *(const void**)b;
}

unsigned long long test_ptr_diff(int *a, int* b) {
  // CIR-LABEL: test_ptr_diff
  // CIR: %[[D:.*]] = cir.ptr_diff {{.*}} : !cir.ptr<!s32i> -> !s64i
  // CIR: %[[U:.*]] = cir.cast integral %[[D]] : !s64i -> !u64i
  // CIR: cir.return {{.*}} : !u64i

  // LLVM-LABEL: define dso_local i64 @test_ptr_diff(
  // LLVM: %[[IA:.*]] = ptrtoint ptr %{{.*}} to i64
  // LLVM: %[[IB:.*]] = ptrtoint ptr %{{.*}} to i64
  // LLVM: %[[SUB:.*]] = sub i64 %[[IA]], %[[IB]]
  // LLVM: %[[Q:.*]] = sdiv exact i64 %[[SUB]], 4
  // LLVM: store i64 %[[Q]], ptr %[[RETADDR:.*]], align
  // LLVM: %[[RETLOAD:.*]] = load i64, ptr %[[RETADDR]], align
  // LLVM: ret i64 %[[RETLOAD]]

  // OGCG-LABEL: define dso_local i64 @test_ptr_diff(
  // OGCG: %[[IA:.*]] = ptrtoint ptr %{{.*}} to i64
  // OGCG: %[[IB:.*]] = ptrtoint ptr %{{.*}} to i64
  // OGCG: %[[SUB:.*]] = sub i64 %[[IA]], %[[IB]]
  // OGCG: %[[Q:.*]] = sdiv exact i64 %[[SUB]], 4
  // OGCG: ret i64 %[[Q]]
  return a - b;
}