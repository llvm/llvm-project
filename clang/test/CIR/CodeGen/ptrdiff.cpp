// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

typedef unsigned long size_type;

size_type size(unsigned long *_start, unsigned long *_finish) {
  // CIR-LABEL: cir.func dso_local @_Z4sizePmS_
  // CIR: %[[D:.*]] = cir.ptr_diff {{.*}} : !cir.ptr<!u64i> -> !s64i
  // CIR: %[[U:.*]] = cir.cast integral %[[D]] : !s64i -> !u64i
  // CIR: cir.return %[[U]] : !u64i

  // LLVM-LABEL: @_Z4sizePmS_
  // LLVM: ptrtoint ptr {{.*}} to i64
  // LLVM: ptrtoint ptr {{.*}} to i64
  // LLVM: %[[SUB:.*]] = sub i64 {{.*}}, {{.*}}
  // LLVM: %[[DIV:.*]] = sdiv exact i64 %[[SUB]], 8
  // LLVM: ret i64 %[[DIV]]
  return static_cast<size_type>(_finish - _start);
}

long add(char *a, char *b) {
  // CIR-LABEL: cir.func dso_local @_Z3addPcS_
  // CIR: %[[D:.*]] = cir.ptr_diff {{.*}} : !cir.ptr<!s8i> -> !s64i
  // CIR: %[[C1:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[C1W:.*]] = cir.cast integral %[[C1]] : !s32i -> !s64i
  // CIR: %[[S:.*]] = cir.binop(add, %[[D]], %[[C1W]]) : !s64i
  // CIR: cir.return %[[S]] : !s64i

  // LLVM-LABEL: @_Z3addPcS_
  // LLVM: ptrtoint ptr {{.*}} to i64
  // LLVM: ptrtoint ptr {{.*}} to i64
  // LLVM: %[[SUB:.*]] = sub i64 {{.*}}, {{.*}}
  // LLVM: %[[ADD1:.*]] = add i64 %[[SUB]], 1
  // LLVM: ret i64 %[[ADD1]]
  return a - b + 1;
}
