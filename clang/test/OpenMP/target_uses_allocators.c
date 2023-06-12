// Test host codegen.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50  -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -verify -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -include-pch %t %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -verify -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -include-pch %t %s -emit-llvm -o - | FileCheck %s

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

typedef enum omp_alloctrait_key_t { omp_atk_sync_hint = 1,
                                    omp_atk_alignment = 2,
                                    omp_atk_access = 3,
                                    omp_atk_pool_size = 4,
                                    omp_atk_fallback = 5,
                                    omp_atk_fb_data = 6,
                                    omp_atk_pinned = 7,
                                    omp_atk_partition = 8
} omp_alloctrait_key_t;

typedef struct omp_alloctrait_t {
  omp_alloctrait_key_t key;
  __UINTPTR_TYPE__ value;
} omp_alloctrait_t;


// CHECK: define {{.*}}[[FIE:@.+]]()
void fie(void) {
  int x;
  omp_allocator_handle_t my_allocator;
  omp_alloctrait_t traits[10];
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
#pragma omp target uses_allocators(omp_null_allocator, omp_thread_mem_alloc, my_allocator(traits))
  {}
}

typedef enum omp_memspace_handle_t {
  omp_default_mem_space = 0,
  omp_large_cap_mem_space = 1,
  omp_const_mem_space = 2,
  omp_high_bw_mem_space = 3,
  omp_low_lat_mem_space = 4,
  llvm_omp_target_host_mem_space = 100,
  llvm_omp_target_shared_mem_space = 101,
  llvm_omp_target_device_mem_space = 102,
  KMP_MEMSPACE_MAX_HANDLE = __UINTPTR_MAX__
} omp_memspace_handle_t;

extern omp_allocator_handle_t
omp_init_allocator(omp_memspace_handle_t memspace, int ntraits,
                   const omp_alloctrait_t traits[]);

void *omp_aligned_alloc(unsigned long alignment, unsigned long size,
                        omp_allocator_handle_t allocator);
extern void * omp_alloc(int size, omp_allocator_handle_t a);
#define N 1024

void foo() {
  int errors = 0;
  omp_memspace_handle_t  memspace = omp_default_mem_space;
  omp_alloctrait_t       traits[1] = {{omp_atk_alignment, 64}};
  omp_allocator_handle_t alloc = omp_init_allocator(memspace,1,traits);
  #pragma omp target map(tofrom: errors) uses_allocators(alloc(traits))
  { }
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

// CHECK: [[TRAITS_ADDR_REF:%.+]] = alloca ptr,
// CHECK: [[MY_ALLOCATOR_ADDR:%.+]] = alloca i64,
// CHECK: [[TRAITS_ADDR:%.+]] = load ptr, ptr [[TRAITS_ADDR_REF]],
// CHECK: [[ALLOCATOR:%.+]] = call ptr @__kmpc_init_allocator(i32 %{{.+}}, ptr null, i32 10, ptr [[TRAITS_ADDR]])
// CHECK: [[CONV:%.+]] = ptrtoint ptr [[ALLOCATOR]] to i64
// CHECK: store i64 [[CONV]], ptr [[MY_ALLOCATOR_ADDR]],

// Destroy allocator upon exit from the region.
// CHECK: [[ALLOCATOR:%.+]] = load i64, ptr [[MY_ALLOCATOR_ADDR]],
// CHECK: [[CONV:%.+]] = inttoptr i64 [[ALLOCATOR]] to ptr
// CHECK: call void @__kmpc_destroy_allocator(i32 %{{.+}}, ptr [[CONV]])

// CHECK: [[TRAITS_ADDR_REF:%.+]] = alloca ptr,
// CHECK: [[MY_ALLOCATOR_ADDR:%alloc]] = alloca i64,
// CHECK: [[TRAITS_ADDR:%.+]] = load ptr, ptr [[TRAITS_ADDR_REF]],
// CHECK: [[ALLOCATOR:%.+]] = call ptr @__kmpc_init_allocator(i32 %{{.+}}, ptr null, i32 1, ptr [[TRAITS_ADDR]])
// CHECK: [[CONV:%.+]] = ptrtoint ptr [[ALLOCATOR]] to i64
// CHECK: store i64 [[CONV]], ptr [[MY_ALLOCATOR_ADDR]],

// Destroy allocator upon exit from the region.
// CHECK: [[ALLOCATOR:%.+]] = load i64, ptr [[MY_ALLOCATOR_ADDR]],
// CHECK: [[CONV1:%.+]] = inttoptr i64 [[ALLOCATOR]] to ptr
// CHECK: call void @__kmpc_destroy_allocator(i32 %{{.+}}, ptr [[CONV1]])
