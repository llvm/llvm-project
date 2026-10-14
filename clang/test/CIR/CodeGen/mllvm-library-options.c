// -mllvm options of per-context libraries reach the LLVMContext that CIR is
// lowered into.
// RUN: %clang_cc1 -triple x86_64 -O1 -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,SINK
// RUN: %clang_cc1 -triple x86_64 -O1 -fclangir -emit-llvm -mllvm -instcombine-code-sinking=0 %s -o - | FileCheck %s --check-prefixes=CHECK,NOSINK

void use(int);

// CHECK-LABEL: define {{.*}} @f(
// SINK-NOT:      sdiv
// NOSINK:        sdiv
// CHECK:         br i1
// SINK:          sdiv
void f(int c, int a, int b) {
  int d = a / b;
  if (c)
    use(d);
}
