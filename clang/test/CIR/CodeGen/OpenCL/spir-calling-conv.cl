// RUN: %clang_cc1 -fclangir %s -O0 -triple "spirv64-unknown-unknown" -emit-cir -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=CIR
// RUN: %clang_cc1 -fclangir %s -O0 -triple "spirv64-unknown-unknown" -emit-llvm -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=LLVM

// CIR: cir.func {{.*}}@get_dummy_id{{.*}} cc(spir_function)
// LLVM-DAG: declare{{.*}} spir_func i32 @get_dummy_id(
int get_dummy_id(int D);

// CIR: cir.func {{.*}}@bar{{.*}} cc(spir_kernel)
// LLVM-DAG: declare{{.*}} spir_kernel void @bar(
kernel void bar(global int *A);

// CIR: cir.func {{.*}}@foo{{.*}} cc(spir_kernel)
// LLVM-DAG: define{{.*}} spir_kernel void @foo(
kernel void foo(global int *A) {
  int id = get_dummy_id(0);
  A[id] = id;
  bar(A);
}
