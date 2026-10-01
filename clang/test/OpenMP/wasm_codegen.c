// REQUIRES: webassembly-registered-target
// RUN: %clang_cc1 -triple wasm32-unknown-emscripten -fopenmp -fnoopenmp-use-tls -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,WASM32
// RUN: %clang_cc1 -triple wasm64-unknown-emscripten -fopenmp -fnoopenmp-use-tls -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,WASM64
// RUN: %clang_cc1 -triple wasm32-unknown-emscripten -fopenmp -fnoopenmp-use-tls -emit-obj -o %t.wasm32.o %s
// RUN: %clang_cc1 -triple wasm64-unknown-emscripten -fopenmp -fnoopenmp-use-tls -emit-obj -o %t.wasm64.o %s

// Exercise representative host OpenMP lowering for both WebAssembly pointer
// models. WebAssembly does not support common linkage, so compiler-generated
// locks and caches must have internal linkage.
// CHECK-DAG: @.gomp_critical_user_.var = internal global [8 x i32] zeroinitializer
// CHECK-DAG: @.gomp_critical_user_named.var = internal global [8 x i32] zeroinitializer
// CHECK-DAG: @.gomp_critical_user_.reduction.var = internal global [8 x i32] zeroinitializer
// CHECK-DAG: @threadprivate_var.cache. = internal global ptr null

// The dependency record and size_t parameters must follow the target pointer
// width rather than assuming wasm32.
// WASM32-DAG: %struct.kmp_depend_info = type { i32, i32, i8 }
// WASM64-DAG: %struct.kmp_depend_info = type { i64, i64, i8 }
// WASM32-DAG: call ptr @__kmpc_omp_task_alloc({{.*}}i32 {{[0-9]+}}, i32 {{[0-9]+}}, ptr
// WASM64-DAG: call ptr @__kmpc_omp_task_alloc({{.*}}i64 {{[0-9]+}}, i64 {{[0-9]+}}, ptr
// WASM32-DAG: call ptr @__kmpc_threadprivate_cached({{.*}}i32 4, ptr @threadprivate_var.cache.)
// WASM64-DAG: call ptr @__kmpc_threadprivate_cached({{.*}}i64 4, ptr @threadprivate_var.cache.)
// WASM32-DAG: call i32 @__kmpc_reduce_nowait({{.*}}i32 4, ptr
// WASM64-DAG: call i32 @__kmpc_reduce_nowait({{.*}}i64 8, ptr

// CHECK-DAG: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(
// CHECK-DAG: call void @__kmpc_for_static_init_4(
// CHECK-DAG: call i32 @__kmpc_omp_task_with_deps(
// CHECK-DAG: call i32 @__kmpc_omp_taskwait(
// CHECK-DAG: call void @__kmpc_critical(

void critical_regions(void) {
#pragma omp critical
  {}

#pragma omp critical(named)
  {}
}

int threadprivate_var;
#pragma omp threadprivate(threadprivate_var)

int openmp_constructs(int *values, int count) {
  int sum = 0;
#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < count; ++i)
    sum += values[i];

#pragma omp task shared(values) depend(inout : values[0 : count])
  values[0] = sum;

#pragma omp taskwait

#pragma omp critical(named)
  threadprivate_var += sum;

  return sum;
}
