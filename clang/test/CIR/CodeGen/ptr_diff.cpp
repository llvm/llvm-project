// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

typedef unsigned long size_type;
size_type size(unsigned long *_start, unsigned long *_finish) {
  return static_cast<size_type>(_finish - _start);
}

// CHECK: cir.func @_Z4sizePmS_(%arg0: !cir.ptr<!u64i>
// CHECK:   %3 = cir.load %1 : cir.ptr <!cir.ptr<!u64i>>, !cir.ptr<!u64i>
// CHECK:   %4 = cir.load %0 : cir.ptr <!cir.ptr<!u64i>>, !cir.ptr<!u64i>
// CHECK:   %5 = cir.ptr_diff(%3, %4) : !cir.ptr<!u64i> -> !s64i
// CHECK:   %6 = cir.cast(integral, %5 : !s64i), !u64i

long add(char *a, char *b) {
  return a - b + 1;
}

// CHECK: cir.func @_Z3addPcS_(%arg0: !cir.ptr<!s8i>
//          %5 = cir.ptr_diff(%3, %4) : !cir.ptr<!s8i> -> !s64i
//          %6 = cir.const(#cir.int<1> : !s32i) : !s32i
//          %7 = cir.cast(integral, %6 : !s32i), !s64i
//          %8 = cir.binop(add, %5, %7) : !s64i

