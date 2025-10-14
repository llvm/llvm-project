// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

int addrcmp(const void* a, const void* b) {
  // CIR-LABEL: addrcmp
  // CIR: %[[R:.*]] = cir.ptr_diff
  // CIR: cir.cast integral  %[[R]] : !s64i -> !s32

  // LLVM-LABEL: addrcmp
  // LLVM: %[[PTR_A:.*]] = ptrtoint ptr {{.*}} to i64
  // LLVM: %[[PTR_B:.*]] = ptrtoint ptr {{.*}} to i64
  // LLVM: %[[SUB:.*]] = sub i64 %[[PTR_A]], %[[PTR_B]]
  // LLVM-NOT: sdiv
  // LLVM: trunc i64 %[[SUB]] to i32
  return *(const void**)a - *(const void**)b;
}

unsigned long long test_ptr_diff(int *a, int* b) {
  // CIR-LABEL: test_ptr_diff
  // CIR: %[[D:.*]] = cir.ptr_diff{{.*}} : !cir.ptr<!s32i> -> !u64i
  // CIR: cir.return %[[D]] : !u64i

  // LLVM-LABEL: @_Z13test_ptr_diffPiS_
  // LLVM: %[[IA:.*]] = ptrtoint ptr %{{.*}} to i64
  // LLVM: %[[IB:.*]] = ptrtoint ptr %{{.*}} to i64
  // LLVM: %[[SUB:.*]] = sub i64 %[[IA]], %[[IB]]
  // LLVM: %[[Q:.*]] = sdiv exact i64 %[[SUB]], 4
  // LLVM: ret i64 %[[Q]]
  return a - b;
}
