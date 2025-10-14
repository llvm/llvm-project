// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR

typedef unsigned long size_type;

size_type size(unsigned long *_start, unsigned long *_finish) {
  // CIR-LABEL: cir.func dso_local @_Z4sizePmS_
  // CIR: %[[D:.*]] = cir.ptr_diff {{.*}} : !cir.ptr<!u64i> -> !s64i
  // CIR: %[[U:.*]] = cir.cast integral %[[D]] : !s64i -> !u64i
  // CIR: cir.return {{.*}} : !u64i

  return static_cast<size_type>(_finish - _start);
}
