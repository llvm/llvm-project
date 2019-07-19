// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp -x c -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp-simd -x c -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

void foo();

// CHECK-LABEL: @main
int main() {
  // CHECK: call i32 @__kmpc_global_thread_num(
  // CHECK: call i8* @__kmpc_omp_task_alloc(
  // CHECK: call i32 @__kmpc_omp_task(
#pragma omp task
  {
#pragma omp taskgroup
    {
#pragma omp task
      foo();
    }
  }
  // CHECK: ret i32 0
  return 0;
}
// CHECK: call void @__kmpc_taskgroup(
// CHECK: call i8* @__kmpc_omp_task_alloc(
// CHECK: call i32 @__kmpc_omp_task(
// CHECK: call void @__kmpc_end_taskgroup(

// CHECK-LINE: @bar
void bar() {
  // CHECK: call void @__kmpc_for_static_init_4(
#pragma omp for
for (int i = 0; i < 10; ++i)
  // CHECK: [[BUF:%.+]] = call i8* @__kmpc_omp_task_alloc(%struct.ident_t* @{{.+}}, i32 %{{.+}}, i32 1, i64 48,
  // CHECK: [[BC_BUF:%.+]] = bitcast i8* [[BUF]] to [[TT_WITH_PRIVS:%.+]]*
  // CHECK: [[PRIVS:%.+]] = getelementptr inbounds [[TT_WITH_PRIVS]], [[TT_WITH_PRIVS]]* [[BC_BUF]], i32 0, i32 1
  // CHECK: [[I_PRIV:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}} [[PRIVS]], i32 0, i32 0
  // CHECK: store i32 %{{.+}}, i32* [[I_PRIV]],
  // CHECK: = call i32 @__kmpc_omp_task(%struct.ident_t* @{{.+}}, i32 %{{.+}}, i8* [[BUF]])
#pragma omp task
++i;
}
#endif
