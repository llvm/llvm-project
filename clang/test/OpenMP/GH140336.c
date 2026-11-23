// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -verify -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fopenmp-enable-irbuilder -verify -emit-llvm -o - %s | FileCheck %s --check-prefix=IRB
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -x c++ -verify -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,CXX
// expected-no-diagnostics

typedef enum omp_allocator_handle_t {
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
} omp_allocator_handle_t;

#ifdef __cplusplus
extern "C" {
#endif

void stop(void) __attribute__((noreturn));

// CHECK-LABEL: define {{.*}}void @gh140336(
// CHECK-NOT: @__kmpc_
// CHECK: switch i32
// CHECK-NOT: @__kmpc_
// CHECK: ret void
// IRB-LABEL: define {{.*}}void @gh140336(
// IRB-NOT: @__kmpc_
// IRB: ret void
void gh140336(int i) {
  switch (i) {
    int j;
#pragma omp allocate(j)
  }
}

// CHECK-LABEL: define {{.*}}void @after_return(
// CHECK-NOT: @__kmpc_
// CHECK: ret void
// IRB-LABEL: define {{.*}}void @after_return(
// IRB-NOT: @__kmpc_
// IRB: ret void
void after_return(int i) {
  return;
  int j;
#pragma omp allocate(j) allocator(omp_large_cap_mem_alloc)
  j = i;
}

// CHECK-LABEL: define {{.*}}void @after_noreturn(
// CHECK-NOT: @__kmpc_
// CHECK: unreachable
// IRB-LABEL: define {{.*}}void @after_noreturn(
// IRB-NOT: @__kmpc_
// IRB: unreachable
void after_noreturn(void) {
  stop();
  int j;
#pragma omp allocate(j) align(32) allocator(omp_large_cap_mem_alloc)
}

// CHECK-LABEL: define {{.*}}void @reachable(
// CHECK: [[TID:%.+]] = call i32 @__kmpc_global_thread_num(ptr @{{.+}})
// CHECK: [[J:%.+]] = call ptr @__kmpc_alloc(i32 [[TID]], i64 4, ptr inttoptr (i64 2 to ptr))
// CHECK: [[K:%.+]] = call ptr @__kmpc_alloc(i32 [[TID]], i64 4, ptr null)
// CHECK: store i32 %{{.+}}, ptr [[J]],
// CHECK: store i32 %{{.+}}, ptr [[K]],
// CHECK: call void @__kmpc_free(i32 [[TID]], ptr [[K]], ptr null)
// CHECK: call void @__kmpc_free(i32 [[TID]], ptr [[J]], ptr inttoptr (i64 2 to ptr))
// CHECK: ret void
// IRB-LABEL: define {{.*}}void @reachable(
// IRB: [[J:%.+]] = call ptr @__kmpc_alloc(i32 %{{.+}}, i64 4, ptr inttoptr (i64 2 to ptr))
// IRB: store i32 %{{.+}}, ptr [[J]],
// IRB: call void @__kmpc_free(i32 %{{.+}}, ptr [[J]], ptr inttoptr (i64 2 to ptr))
// IRB: ret void
void reachable(int i) {
  int j;
#pragma omp allocate(j) allocator(omp_large_cap_mem_alloc)
  int k;
#pragma omp allocate(k)
  j = i;
  k = i;
}

#ifdef __cplusplus
}

struct S {
  ~S();
};

// CXX-LABEL: define {{.*}}void @_Z17after_return_dtorv(
// CXX-NOT: @__kmpc_
// CXX-NOT: @_ZN1SD
// CXX: ret void
void after_return_dtor() {
  return;
  S s;
#pragma omp allocate(s) allocator(omp_large_cap_mem_alloc)
}
#endif
