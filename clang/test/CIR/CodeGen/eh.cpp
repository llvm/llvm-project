// RUN: %clang_cc1 -std=c++20 -triple aarch64-none-linux-android21 -fclangir -fcxx-exceptions -fexceptions -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple aarch64-none-linux-android21 -fclangir -fcxx-exceptions -fexceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

struct test1_D {
  double d;
} d1;

void test1() {
  throw d1;
}

// CIR-LABEL: @_Z5test1v
// FIXME: this is overaligned, should be 4.
// CIR:   %[[ALLOC:.*]] = cir.alloc.exception 8 -> !cir.ptr<!ty_test1_D>
// CIR:   %[[G:.*]] = cir.get_global @d1 : !cir.ptr<!ty_test1_D>
// CIR:   cir.call @_ZN7test1_DC1ERKS_(%[[ALLOC]], %[[G]]) : (!cir.ptr<!ty_test1_D>, !cir.ptr<!ty_test1_D>) -> ()
// CIR:   cir.throw %[[ALLOC]] : !cir.ptr<!ty_test1_D>, @_ZTI7test1_D
// CIR:   cir.unreachable
// CIR: }

// LLVM-LABEL: @_Z5test1v
// FIXME: this is overaligned, should be 4.
// LLVM:   %[[ALLOC:.*]] = call ptr @__cxa_allocate_exception(i64 8)

// FIXME: this is a simple store once we fix isTrivialCtorOrDtor().
// LLVM:   call void @_ZN7test1_DC1ERKS_(ptr %[[ALLOC]], ptr @d1)
// LLVM:   call void @__cxa_throw(ptr %[[ALLOC]], ptr @_ZTI7test1_D, ptr null)
// LLVM:   unreachable
// LLVM: }