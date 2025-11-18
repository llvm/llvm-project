// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

typedef unsigned long size_type;

size_type size(unsigned long *_start, unsigned long *_finish) {
  // CIR-LABEL: cir.func dso_local @_Z4sizePmS_
  // CIR: %[[D:.*]] = cir.ptr_diff {{.*}} : !cir.ptr<!u64i> -> !s64i
  // CIR: %[[U:.*]] = cir.cast integral %[[D]] : !s64i -> !u64i
  // CIR: cir.return {{.*}} : !u64i

  // LLVM-LABEL: define dso_local {{.*}}i64 @_Z4sizePmS_(
  // LLVM: %[[IA:.*]] = ptrtoint ptr %{{.*}} to i64
  // LLVM: %[[IB:.*]] = ptrtoint ptr %{{.*}} to i64
  // LLVM: %[[SUB:.*]] = sub i64 %[[IA]], %[[IB]]
  // LLVM: %[[Q:.*]] = sdiv exact i64 %[[SUB]], 8
  // LLVM: store i64 %[[Q]], ptr %[[RETADDR:.*]], align
  // LLVM: %[[RET:.*]] = load i64, ptr %[[RETADDR]], align
  // LLVM: ret i64 %[[RET]]

  // OGCG-LABEL: define dso_local {{.*}}i64 @_Z4sizePmS_(
  // OGCG: %[[IA:.*]] = ptrtoint ptr %{{.*}} to i64
  // OGCG: %[[IB:.*]] = ptrtoint ptr %{{.*}} to i64
  // OGCG: %[[SUB:.*]] = sub i64 %[[IA]], %[[IB]]
  // OGCG: %[[Q:.*]] = sdiv exact i64 %[[SUB]], 8
  // OGCG: ret i64 %[[Q]]

  return static_cast<size_type>(_finish - _start);
}