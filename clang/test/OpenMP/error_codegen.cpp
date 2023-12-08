// RUN: %clang_cc1 -std=c++11 -fopenmp -fopenmp-version=51 -triple x86_64 \
// RUN:   -emit-llvm -o - %s | FileCheck %s

// RUN: %clang_cc1 -std=c++11 -fopenmp-simd -fopenmp-version=51 \
// RUN:  -debug-info-kind=limited -triple x86_64 -emit-llvm -o - %s |  \
// RUN:  FileCheck  --check-prefix SIMD %s

//CHECK: @.str = private unnamed_addr constant [23 x i8] c"GPU compiler required.\00", align 1
//CHECK: @0 = private unnamed_addr constant {{.*}}error_codegen.cpp;main;52;1;;\00", align 1
//CHECK: @1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 {{.*}}, ptr @0 }, align 8
//CHECK: @.str.1 = private unnamed_addr constant [27 x i8] c"Note this is functioncall.\00", align 1
//CHECK: @2 = private unnamed_addr constant {{.*}}error_codegen.cpp;main;54;1;;\00", align 1
//CHECK: @3 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 {{.*}}, ptr @2 }, align 8
//CHECK: @.str.2 = private unnamed_addr constant [23 x i8] c"GNU compiler required.\00", align 1
//CHECK: @4 = private unnamed_addr constant {{.*}}error_codegen.cpp;tmain;29;1;;\00", align 1
//CHECK: @5 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 {{.*}}, ptr @4 }, align 8
//CHECK: @.str.3 = private unnamed_addr constant [22 x i8] c"Notice: add for loop.\00", align 1
//CHECK: @6 = private unnamed_addr constant {{.*}}error_codegen.cpp;tmain;32;1;;\00", align 1
//CHECK: @7 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 {{.*}}, ptr @6 }, align 8
//CHECK: @8 = private unnamed_addr constant {{.*}}error_codegen.cpp;tmain;38;1;;\00", align 1
//CHECK: @9 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 {{.*}}, ptr @8 }, align 8

void foo() {}

template <typename T, int N>
int tmain(T argc, char **argv) {
  T b = argc, c, d, e, f, g;
  static int a;
#pragma omp error at(execution) severity(fatal) message("GNU compiler required.")
  a = argv[0][0];
  ++a;
#pragma omp error at(execution) severity(warning) message("Notice: add for loop.")
  {
    int b = 10;
    T c = 100;
    a = b + c;
  }
#pragma omp  error at(execution) severity(fatal) message("GPU compiler required.")
  foo();
return N;
}
// CHECK-LABEL: @main(
// SIMD-LABEL: @main(
// CHECK:    call void @__kmpc_error(ptr @1, i32 2, ptr @.str)
// SIMD-NOT:    call void @__kmpc_error(ptr @1, i32 2, ptr @.str)
// CHECK:    call void @__kmpc_error(ptr @3, i32 1, ptr @.str.1)
// SIMD-NOT:    call void @__kmpc_error(ptr @3, i32 1, ptr @.str.1)
//
int main (int argc, char **argv) {
  int b = argc, c, d, e, f, g;
  static int a;
#pragma omp error at(execution) severity(fatal) message("GPU compiler required.")
   a=2;
#pragma omp error at(execution) severity(warning) message("Note this is functioncall.")
  foo();
  tmain<int, 10>(argc, argv);
}

//CHECK-LABEL: @_Z5tmainIiLi10EEiT_PPc(
//SIMD-LABEL: @_Z5tmainIiLi10EEiT_PPc(
//CHECK: call void @__kmpc_error(ptr @5, i32 2, ptr @.str.2)
//CHECK: call void @__kmpc_error(ptr @7, i32 1, ptr @.str.3)
//CHECK: call void @__kmpc_error(ptr @9, i32 2, ptr @.str)
//SIMD-NOT: call void @__kmpc_error(ptr @5, i32 2, ptr @.str.2)
//SIMD-NOT: call void @__kmpc_error(ptr @7, i32 1, ptr @.str.3)
//SIMD-NOT: call void @__kmpc_error(ptr @9, i32 2, ptr @.str)
//CHECK: ret i32 10
