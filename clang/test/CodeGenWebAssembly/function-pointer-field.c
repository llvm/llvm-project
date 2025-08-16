// REQUIRES: webassembly-registered-target
// RUN: %clang_cc1 -triple wasm32-unknown-unknown -emit-llvm -O0 -o - %s | FileCheck %s

// Test of function pointer bitcast in a struct field with different argument number in wasm32

#define FUNCTION_POINTER(f) ((FunctionPointer)(f))
typedef int (*FunctionPointer)(int a, int b); 

// CHECK: @__const.test.sfp = private unnamed_addr constant %struct._StructWithFunctionPointer { ptr @__fp_less_iii }, align 4

typedef struct _StructWithFunctionPointer {
  FunctionPointer fp;
} StructWithFunctionPointer;

int fp_less(int a) {
  return a;
}
                                                  
// CHECK-LABEL: @test
void test() {
  StructWithFunctionPointer sfp = {
    FUNCTION_POINTER(fp_less)
  };

  int a1 = sfp.fp(10, 20);
}

// CHECK: define internal i32 @__fp_less_iii(i32 %0, i32 %1)
// CHECK: %2 = call i32 @fp_less(i32 %0)
// CHECK: ret i32 %2
