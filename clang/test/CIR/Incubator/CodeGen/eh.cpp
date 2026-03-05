// XFAIL: *
// RUN: %clang_cc1 -std=c++20 -triple aarch64-none-linux-android21 -fclangir -fcxx-exceptions -fexceptions -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple aarch64-none-linux-android21 -fclangir -fcxx-exceptions -fexceptions -emit-llvm -fno-clangir-call-conv-lowering %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

struct test1_D {
  double d;
} d1;

void test1() {
  throw d1;
}

// CIR-LABEL: @_Z5test1v
// CIR:   %[[ALLOC:.*]] = cir.alloc.exception 8 -> !cir.ptr<!rec_test1_D>
// CIR:   %[[G:.*]] = cir.get_global @d1 : !cir.ptr<!rec_test1_D>
// CIR:   cir.copy %1 to %0 : !cir.ptr<!rec_test1_D>
// CIR:   cir.throw %[[ALLOC]] : !cir.ptr<!rec_test1_D>, @_ZTI7test1_D
// CIR:   cir.unreachable
// CIR: }

// LLVM-LABEL: @_Z5test1v
// LLVM:   %[[ALLOC:.*]] = call ptr @__cxa_allocate_exception(i64 8)

// FIXME: this is a llvm.memcpy.p0.p0.i64 once we fix isTrivialCtorOrDtor().
// LLVM:   call void @llvm.memcpy.p0.p0.i32(ptr %1, ptr @d1, i32 8, i1 false)
// LLVM:   call void @__cxa_throw(ptr %[[ALLOC]], ptr @_ZTI7test1_D, ptr null)
// LLVM:   unreachable
// LLVM: }

struct test2_D {
  test2_D(const test2_D&o);
  test2_D();
  virtual void bar() { }
  int i; int j;
} d2;

void test2() {
  throw d2;
}

// CIR-LABEL: @_Z5test2v
// CIR:   %[[ALLOC:.*]] = cir.alloc.exception 16 -> !cir.ptr<!rec_test2_D>
// CIR:   %[[G:.*]] = cir.get_global @d2 : !cir.ptr<!rec_test2_D>
// CIR:   cir.try synthetic cleanup {
// CIR:     cir.copy %[[G:.*]] to %[[ALLOC:.*]] : !cir.ptr<!rec_test2_D>
// CIR:     cir.yield
// CIR:   } catch [#cir.unwind {
// CIR:     cir.resume
// CIR:   }]
// CIR:   cir.throw %[[ALLOC]] : !cir.ptr<!rec_test2_D>, @_ZTI7test2_D
// CIR:   cir.unreachable

// LLVM-LABEL: @_Z5test2v

// LLVM: %[[ALLOC:.*]] = call ptr @__cxa_allocate_exception(i64 16)

// LLVM: landingpad { ptr, i32 }
// LLVM:         cleanup
// LLVM: extractvalue { ptr, i32 }
// LLVM: extractvalue { ptr, i32 }
// LLVM: call void @__cxa_free_exception(ptr %[[ALLOC]])
