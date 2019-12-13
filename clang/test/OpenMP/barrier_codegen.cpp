// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,CLANGCG
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CLANGCG

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -fopenmp-enable-irbuilder -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,IRBUILDER
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fopenmp-enable-irbuilder -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-unknown -fopenmp-enable-irbuilder -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,IRBUILDER

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -fopenmp-enable-irbuilder -mllvm -openmp-ir-builder-optimistic-attributes -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,IRBUILDER_OPT
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fopenmp-enable-irbuilder -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-unknown -fopenmp-enable-irbuilder -mllvm -openmp-ir-builder-optimistic-attributes -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,IRBUILDER_OPT

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK: [[IDENT_T:%.+]] = type { i32, i32, i32, i32, i8* }
// CHECK-DAG: [[EXPLICIT_BARRIER_LOC:@.+]] = {{.+}} [[IDENT_T]] { i32 0, i32 34, i32 0, i32 0, i8* getelementptr inbounds ([{{[0-9]+}} x i8], [{{[0-9]+}} x i8]* @{{.+}}, i32 0, i32 0) }
// CHECK-DAG: [[LOC:@.+]] = {{.+}} [[IDENT_T]] { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([{{[0-9]+}} x i8], [{{[0-9]+}} x i8]* @{{.+}}, i32 0, i32 0) }

void foo() {}

template <class T>
T tmain(T argc) {
  static T a;
#pragma omp barrier
  return a + argc;
}

// CHECK-LABEL: @main
int main(int argc, char **argv) {
  static int a;
#pragma omp barrier
  // CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num([[IDENT_T]]* [[LOC]])
  // CHECK: call void @__kmpc_barrier([[IDENT_T]]* [[EXPLICIT_BARRIER_LOC]], i32 [[GTID]])
  // CHECK: call {{.+}} [[TMAIN_INT:@.+]](i{{[0-9][0-9]}}
  // CHECK: call {{.+}} [[TMAIN_CHAR:@.+]](i{{[0-9]}}
  return tmain(argc) + tmain(argv[0][0]) + a;
}

// CLANGCG:            declare i32 @__kmpc_global_thread_num(%struct.ident_t*)
// CLANGCG-NOT:        #
// IRBUILDER:          ; Function Attrs: nounwind
// IRBUILDER-NEXT:     declare i32 @__kmpc_global_thread_num(%struct.ident_t*) #
// IRBUILDER_OPT:      ; Function Attrs: nofree nosync nounwind readonly
// IRBUILDER_OPT-NEXT: declare i32 @__kmpc_global_thread_num(%struct.ident_t*) #

// CHECK: define {{.+}} [[TMAIN_INT]](
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num([[IDENT_T]]* [[LOC]])
// CHECK: call void @__kmpc_barrier([[IDENT_T]]* [[EXPLICIT_BARRIER_LOC]], i32 [[GTID]])

// CHECK: define {{.+}} [[TMAIN_CHAR]](
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num([[IDENT_T]]* [[LOC]])
// CHECK: call void @__kmpc_barrier([[IDENT_T]]* [[EXPLICIT_BARRIER_LOC]], i32 [[GTID]])

#endif
