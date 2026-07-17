// REQUIRES: webassembly-registered-target
// RUN: %clang_cc1 -triple wasm32-unknown-unknown -emit-llvm -O0 -fwasm-fix-function-bitcasts -o - %s | FileCheck %s

// Test of function pointer bitcast stored in a local variable with different
// argument number in wasm32. The cast happens via a CK_BitCast in the scalar
// expression path (local variable assignment), which is intercepted in
// CGExprScalar.cpp to generate a thunk — the same mechanism used for
// function-argument and struct-field cases.

#define FUNCTION_POINTER(f) ((FunctionPointer)(f))
typedef int (*FunctionPointer)(int a, int b);

int fp_less(int a) {
  return a;
}

// CHECK-LABEL: @test
// CHECK: store ptr @__fp_less_iii, ptr %fp
// CHECK: %[[FP:.*]] = load ptr, ptr %fp
// CHECK: call i32 %[[FP]](i32 noundef 10, i32 noundef 20)
void test() {
  FunctionPointer fp = FUNCTION_POINTER(fp_less);
  fp(10, 20);
}

// CHECK: define internal i32 @__fp_less_iii(i32 %0, i32 %1)
// CHECK: %2 = call i32 @fp_less(i32 %0)
// CHECK: ret i32 %2
