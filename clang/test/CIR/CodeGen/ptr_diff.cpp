// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

typedef unsigned long size_type;
size_type size(unsigned long *_start, unsigned long *_finish) {
  return static_cast<size_type>(_finish - _start);
}

// CHECK: cir.func @_Z4sizePmS_(%arg0: !cir.ptr<!u64i>
// CHECK:   %3 = cir.load %1 : cir.ptr <!cir.ptr<!u64i>>, !cir.ptr<!u64i>
// CHECK:   %4 = cir.load %0 : cir.ptr <!cir.ptr<!u64i>>, !cir.ptr<!u64i>
// CHECK:   %5 = cir.ptr_diff(%3, %4) : !cir.ptr<!u64i> -> !u64i
  