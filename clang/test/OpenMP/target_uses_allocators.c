// Test host codegen.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50  -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -verify -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -include-pch %t %s -emit-llvm -o - | FileCheck %s

#ifndef HEADER
#define HEADER

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

// CHECK: define {{.*}}[[FIE:@.+]]()
void fie(void) {
  int x;
  #pragma omp target uses_allocators(omp_null_allocator) allocate(omp_null_allocator: x) firstprivate(x)
  {}
  #pragma omp target uses_allocators(omp_default_mem_alloc) allocate(omp_default_mem_alloc: x) firstprivate(x)
  {}
  #pragma omp target uses_allocators(omp_large_cap_mem_alloc) allocate(omp_large_cap_mem_alloc: x) firstprivate(x)
  {}
  #pragma omp target uses_allocators(omp_const_mem_alloc) allocate(omp_const_mem_alloc: x) firstprivate(x)
  {}
  #pragma omp target uses_allocators(omp_high_bw_mem_alloc) allocate(omp_high_bw_mem_alloc: x) firstprivate(x)
  {}
  #pragma omp target uses_allocators(omp_low_lat_mem_alloc) allocate(omp_low_lat_mem_alloc: x) firstprivate(x)
  {}
  #pragma omp target uses_allocators(omp_cgroup_mem_alloc) allocate(omp_cgroup_mem_alloc: x) firstprivate(x)
  {}
  #pragma omp target uses_allocators(omp_pteam_mem_alloc) allocate(omp_pteam_mem_alloc: x) firstprivate(x)
  {}
  #pragma omp target uses_allocators(omp_thread_mem_alloc) allocate(omp_thread_mem_alloc: x) firstprivate(x) // expected-warning {{allocator with the 'thread' trait access has unspecified behavior on 'target' directive}}
  {}
}

#endif

// CHECK: %[[#R0:]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK-NEXT: store i64 %x, ptr %x.addr, align 8
// CHECK-NEXT: %.x..void.addr = call ptr @__kmpc_alloc(i32 %[[#R0]], i64 4, ptr null)
// CHECK-NEXT: %[[#R1:]] = load i32, ptr %x.addr, align 4
// CHECK-NEXT: store i32 %[[#R1]], ptr %.x..void.addr, align 4
// CHECK-NEXT: call void @__kmpc_free(i32 %[[#R0]], ptr %.x..void.addr, ptr null)

// CHECK: %[[#R0:]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK-NEXT: store i64 %x, ptr %x.addr, align 8
// CHECK-NEXT: %.x..void.addr = call ptr @__kmpc_alloc(i32 %[[#R0]], i64 4, ptr inttoptr (i64 1 to ptr))
// CHECK-NEXT: %[[#R1:]] = load i32, ptr %x.addr, align 4
// CHECK-NEXT: store i32 %[[#R1]], ptr %.x..void.addr, align 4
// CHECK-NEXT: call void @__kmpc_free(i32 %[[#R0]], ptr %.x..void.addr, ptr inttoptr (i64 1 to ptr))

// CHECK: %[[#R0:]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK-NEXT: store i64 %x, ptr %x.addr, align 8
// CHECK-NEXT: %.x..void.addr = call ptr @__kmpc_alloc(i32 %[[#R0]], i64 4, ptr inttoptr (i64 2 to ptr))
// CHECK-NEXT: %[[#R1:]] = load i32, ptr %x.addr, align 4
// CHECK-NEXT: store i32 %[[#R1]], ptr %.x..void.addr, align 4
// CHECK-NEXT: call void @__kmpc_free(i32 %[[#R0]], ptr %.x..void.addr, ptr inttoptr (i64 2 to ptr))

// CHECK: %[[#R0:]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK-NEXT: store i64 %x, ptr %x.addr, align 8
// CHECK-NEXT: %.x..void.addr = call ptr @__kmpc_alloc(i32 %[[#R0]], i64 4, ptr inttoptr (i64 3 to ptr))
// CHECK-NEXT: %[[#R1:]] = load i32, ptr %x.addr, align 4
// CHECK-NEXT: store i32 %[[#R1]], ptr %.x..void.addr, align 4
// CHECK-NEXT: call void @__kmpc_free(i32 %[[#R0]], ptr %.x..void.addr, ptr inttoptr (i64 3 to ptr))

// CHECK: %[[#R0:]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK-NEXT: store i64 %x, ptr %x.addr, align 8
// CHECK-NEXT: %.x..void.addr = call ptr @__kmpc_alloc(i32 %[[#R0]], i64 4, ptr inttoptr (i64 4 to ptr))
// CHECK-NEXT: %[[#R1:]] = load i32, ptr %x.addr, align 4
// CHECK-NEXT: store i32 %[[#R1]], ptr %.x..void.addr, align 4
// CHECK-NEXT: call void @__kmpc_free(i32 %[[#R0]], ptr %.x..void.addr, ptr inttoptr (i64 4 to ptr))

// CHECK: %[[#R0:]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK-NEXT: store i64 %x, ptr %x.addr, align 8
// CHECK-NEXT: %.x..void.addr = call ptr @__kmpc_alloc(i32 %[[#R0]], i64 4, ptr inttoptr (i64 5 to ptr))
// CHECK-NEXT: %[[#R1:]] = load i32, ptr %x.addr, align 4
// CHECK-NEXT: store i32 %[[#R1]], ptr %.x..void.addr, align 4
// CHECK-NEXT: call void @__kmpc_free(i32 %[[#R0]], ptr %.x..void.addr, ptr inttoptr (i64 5 to ptr))

// CHECK: %[[#R0:]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK-NEXT: store i64 %x, ptr %x.addr, align 8
// CHECK-NEXT: %.x..void.addr = call ptr @__kmpc_alloc(i32 %[[#R0]], i64 4, ptr inttoptr (i64 6 to ptr))
// CHECK-NEXT: %[[#R1:]] = load i32, ptr %x.addr, align 4
// CHECK-NEXT: store i32 %[[#R1]], ptr %.x..void.addr, align 4
// CHECK-NEXT: call void @__kmpc_free(i32 %[[#R0]], ptr %.x..void.addr, ptr inttoptr (i64 6 to ptr))

// CHECK: %[[#R0:]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK-NEXT: store i64 %x, ptr %x.addr, align 8
// CHECK-NEXT: %.x..void.addr = call ptr @__kmpc_alloc(i32 %[[#R0]], i64 4, ptr inttoptr (i64 7 to ptr))
// CHECK-NEXT: %[[#R1:]] = load i32, ptr %x.addr, align 4
// CHECK-NEXT: store i32 %[[#R1]], ptr %.x..void.addr, align 4
// CHECK-NEXT: call void @__kmpc_free(i32 %[[#R0]], ptr %.x..void.addr, ptr inttoptr (i64 7 to ptr))

// CHECK: %[[#R0:]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK-NEXT: store i64 %x, ptr %x.addr, align 8
// CHECK-NEXT: %.x..void.addr = call ptr @__kmpc_alloc(i32 %[[#R0]], i64 4, ptr inttoptr (i64 8 to ptr))
// CHECK-NEXT: %[[#R1:]] = load i32, ptr %x.addr, align 4
// CHECK-NEXT: store i32 %[[#R1]], ptr %.x..void.addr, align 4
// CHECK-NEXT: call void @__kmpc_free(i32 %[[#R0]], ptr %.x..void.addr, ptr inttoptr (i64 8 to ptr))
