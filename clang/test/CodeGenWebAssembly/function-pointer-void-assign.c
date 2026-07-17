// RUN: %clang_cc1 -triple wasm32-unknown-unknown -emit-llvm -O0 \
// RUN:   -fwasm-fix-function-bitcasts -o - %s | FileCheck %s

// Test that a function pointer stored via a bare assignment into a void*
// variable is correctly wrapped in a thunk when later cast to a different
// function pointer type with more parameters.

typedef int (*FP)(int, int);

static int fp_one(int a) { return a < 0; }

void test_void_ptr_bare_assign() {
  // Bare assignment (no initializer) into void* — not via VD->hasInit()
  void *p;
  p = (void *)fp_one;
  FP fp = (FP)p;
  fp(10, 20);
}

// CHECK-LABEL: define void @test_void_ptr_bare_assign
// CHECK:         store ptr @__fp_one_iii, ptr %fp
// CHECK:       define internal i32 @__fp_one_iii(i32 %0, i32 %1)
// CHECK-NEXT:  entry:
// CHECK-NEXT:    %2 = call i32 @fp_one(i32 %0)
