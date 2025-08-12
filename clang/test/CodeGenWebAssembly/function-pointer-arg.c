// REQUIRES: webassembly-registered-target
// RUN: %clang_cc1 -triple wasm32-unknown-unknown -emit-llvm -O0 -o - %s | FileCheck %s

// Test of function pointer bitcast in a function argument with different argument number in wasm32

#define FUNCTION_POINTER(f) ((FunctionPointer)(f))
typedef int (*FunctionPointer)(int a, int b);

int fp_as_arg(FunctionPointer fp, int a, int b) {
  return fp(a, b);
}

int fp_less(int a) {
  return a;
}

// CHECK-LABEL: @test
// CHECK: call i32 @fp_as_arg(ptr noundef @__fp_less_iii, i32 noundef 10, i32 noundef 20)
void test() {
  fp_as_arg(FUNCTION_POINTER(fp_less), 10, 20);
}

// CHECK: define internal i32 @__fp_less_iii(i32 %0, i32 %1)
// CHECK: %2 = call i32 @fp_less(i32 %0)
// CHECK: ret i32 %2