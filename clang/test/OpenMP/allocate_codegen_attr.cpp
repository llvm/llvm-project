// RUN: %clang_cc1 -verify -fopenmp -triple x86_64-apple-darwin10.6.0 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -triple x86_64-apple-darwin10.6.0 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -triple x86_64-apple-darwin10.6.0 -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fnoopenmp-use-tls -triple x86_64-unknown-linux-gnu -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fnoopenmp-use-tls -triple x86_64-unknown-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -triple x86_64-apple-darwin10.6.0 -emit-llvm -o - %s | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -triple x86_64-apple-darwin10.6.0 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -triple x86_64-apple-darwin10.6.0 -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fnoopenmp-use-tls -triple x86_64-unknown-linux-gnu -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fnoopenmp-use-tls -triple x86_64-unknown-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

enum omp_allocator_handle_t {
  omp_null_allocator = 0,
  omp_default_mem_alloc = 1,
  omp_large_cap_mem_alloc = 2,
  omp_const_mem_alloc = 3,
  omp_high_bw_mem_alloc = 4,
  omp_low_lat_mem_alloc = 5,
  omp_cgroup_mem_alloc = 6,
  omp_pteam_mem_alloc = 7,
  omp_thread_mem_alloc = 8,
  KMP_ALLOCATOR_MAX_HANDLE = __UINTPTR_MAX__
};

struct St {
  int a;
};

struct St1 {
  int a;
  static int b;
  [[omp::directive(allocate(b) allocator(omp_default_mem_alloc))]];
} d;

int a, b, c;
[[ omp::directive(allocate(a) allocator(omp_large_cap_mem_alloc)),
   directive(allocate(b) allocator(omp_const_mem_alloc)),
   directive(allocate(d, c) allocator(omp_high_bw_mem_alloc)) ]];

template <class T>
struct ST {
  static T m;
  [[omp::directive(allocate(m) allocator(omp_low_lat_mem_alloc))]];
};

template <class T> T foo() {
  T v;
  [[omp::directive(allocate(v) allocator(omp_cgroup_mem_alloc))]];
  v = ST<T>::m;
  return v;
}

namespace ns {
int a;
}
[[omp::directive(allocate(ns::a) allocator(omp_pteam_mem_alloc))]];

// CHECK-NOT:  call {{.+}} {{__kmpc_alloc|__kmpc_free}}

int main() {
  static int a;
  [[omp::directive(allocate(a) allocator(omp_thread_mem_alloc))]];
  a = 2;
  double b = 3;
  [[omp::directive(allocate(b))]];
  return (foo<int>());
}

// CHECK-LABEL: define {{[^@]+}}@main
// CHECK-SAME: () #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RETVAL:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__kmpc_global_thread_num(ptr @[[GLOB1:[0-9]+]])
// CHECK-NEXT:    store i32 0, ptr [[RETVAL]], align 4
// CHECK-NEXT:    store i32 2, ptr @_ZZ4mainE1a, align 4
// CHECK-NEXT:    [[DOTB__VOID_ADDR:%.*]] = call ptr @__kmpc_alloc(i32 [[TMP0]], i64 8, ptr null)
// CHECK-NEXT:    store double 3.000000e+00, ptr [[DOTB__VOID_ADDR]], align 8
// CHECK-NEXT:    [[CALL:%.*]] = call noundef i32 @_Z3fooIiET_v()
// CHECK-NEXT:    store i32 [[CALL]], ptr [[RETVAL]], align 4
// CHECK-NEXT:    call void @__kmpc_free(i32 [[TMP0]], ptr [[DOTB__VOID_ADDR]], ptr null)
// CHECK-NEXT:    [[TMP2:%.*]] = load i32, ptr [[RETVAL]], align 4
// CHECK-NEXT:    ret i32 [[TMP2]]

// CHECK: define {{.*}}i32 @{{.+}}foo{{.+}}()
// CHECK:      [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(ptr @{{.+}})
// CHECK-NEXT: [[V_VOID_ADDR:%.+]] = call ptr @__kmpc_alloc(i32 [[GTID]], i64 4, ptr inttoptr (i64 6 to ptr))
// CHECK-NOT:  {{__kmpc_alloc|__kmpc_free}}
// CHECK:      store i32 %{{.+}}, ptr [[V_VOID_ADDR]],
// CHECK-NEXT: [[V_VAL:%.+]] = load i32, ptr [[V_VOID_ADDR]],
// CHECK-NEXT: call void @__kmpc_free(i32 [[GTID]], ptr [[V_VOID_ADDR]], ptr inttoptr (i64 6 to ptr))
// CHECK-NOT:  {{__kmpc_alloc|__kmpc_free}}
// CHECK:      ret i32 [[V_VAL]]

// CHECK-NOT:  call {{.+}} {{__kmpc_alloc|__kmpc_free}}
extern template int ST<int>::m;

// CHECK: define{{.*}} void @{{.+}}bar{{.+}}(i32 noundef %{{.+}}, ptr noundef {{.+}})
void bar(int a, float &z) {
  // CHECK: [[A_VOID_PTR:%.+]] = call ptr @__kmpc_alloc(i32 [[GTID:%.+]], i64 4, ptr inttoptr (i64 1 to ptr))
  // CHECK: store i32 %{{.+}}, ptr [[A_VOID_PTR]],
  // CHECK: [[Z_VOID_PTR:%.+]] = call ptr @__kmpc_alloc(i32 [[GTID]], i64 8, ptr inttoptr (i64 1 to ptr))
  // CHECK: store ptr %{{.+}}, ptr [[Z_VOID_PTR]],
  [[omp::directive(allocate(a, z) allocator(omp_default_mem_alloc))]];
  // CHECK: call void @__kmpc_free(i32 [[GTID]], ptr [[Z_VOID_PTR]], ptr inttoptr (i64 1 to ptr))
  // CHECK: call void @__kmpc_free(i32 [[GTID]], ptr [[A_VOID_PTR]], ptr inttoptr (i64 1 to ptr))
  // CHECK: ret void
}
#endif
