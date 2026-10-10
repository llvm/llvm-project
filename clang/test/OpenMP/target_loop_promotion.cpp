// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux-gnu \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-host.bc

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple amdgcn-amd-amdhsa \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-host.bc \
// RUN:   -fopenmp-assume-teams-oversubscription \
// RUN:   -fopenmp-assume-threads-oversubscription \
// RUN:   -emit-llvm %s -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,NOLOOP

// RUN: %clang_cc1 -fopenmp -x c++ -triple amdgcn-amd-amdhsa \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-host.bc \
// RUN:   -fopenmp-assume-teams-oversubscription \
// RUN:   -fopenmp-assume-threads-oversubscription \
// RUN:   -emit-pch %s -o %t.pch

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple amdgcn-amd-amdhsa \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-host.bc \
// RUN:   -fopenmp-assume-teams-oversubscription \
// RUN:   -fopenmp-assume-threads-oversubscription \
// RUN:   -include-pch %t.pch -emit-llvm %s -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,NOLOOP

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple amdgcn-amd-amdhsa \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-host.bc \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,STRIDED

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple amdgcn-amd-amdhsa \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-host.bc \
// RUN:   -fopenmp-assume-teams-oversubscription \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,STRIDED

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple amdgcn-amd-amdhsa \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-host.bc \
// RUN:   -fopenmp-assume-threads-oversubscription \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,STRIDED

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

int foo(int i);

void promotable_no_loop(int *array, int n, int c) {
#pragma omp target teams distribute parallel for
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;

#pragma omp target teams distribute parallel for simd
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;

#pragma omp target teams distribute parallel for nowait
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;

#pragma omp target teams distribute parallel for
  for (int i = 0; i < n; ++i)
    array[i] = i + 1;

  {
    int tmp;
#pragma omp target teams distribute parallel for private(tmp)
    for (int i = 0; i < 1024; ++i) {
      tmp = i + 1;
      array[i] = tmp;
    }
  }

  {
    auto set = [&array](int i) { array[i] = i + 1; };
#pragma omp target teams distribute parallel for
    for (int i = 0; i < 1024; ++i)
      set(i);
  }

  {
    int i;
#pragma omp target teams distribute parallel for lastprivate(i)
    for (i = 0; i < 1024; ++i)
      array[i] = i + 1;
  }

  {
    int last = 0;
#pragma omp target teams distribute parallel for lastprivate(last)
    for (int i = 0; i < 1024; ++i) {
      array[i] = i + 1;
      last = i;
    }
  }

  {
    int last = 0;
#pragma omp target teams distribute parallel for lastprivate(last) nowait
    for (int i = 0; i < 1024; ++i) {
      array[i] = i + 1;
      last = i;
    }
  }

  {
    int i;
#pragma omp target teams distribute parallel for simd linear(i)
    for (i = 0; i < 1024; ++i)
      array[i] = i + 1;
  }

#pragma omp target teams loop
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;

#pragma omp target
  {
#pragma omp teams loop
    for (int i = 0; i < 1024; ++i)
      array[i] = i + 1;
  }

#pragma omp target teams
  {
#pragma omp distribute parallel for
    for (int i = 0; i < 1024; ++i)
      array[i] = i + 1;
  }

#pragma omp target
  {
#pragma omp teams
    {
#pragma omp distribute parallel for
      for (int i = 0; i < 1024; ++i)
        array[i] = i + 1;
    }
  }

#pragma omp target teams distribute parallel for schedule(static, 1)
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;

#pragma omp target teams distribute parallel for schedule(auto)
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;

#pragma omp target teams distribute parallel for if(parallel : 1)
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;

#pragma omp target teams distribute parallel for if(target : c)
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;
}

void promotable_strided(int *array) {
#pragma omp target teams distribute parallel for num_teams(3)
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;

#pragma omp target teams num_teams(3)
  {
#pragma omp distribute parallel for
    for (int i = 0; i < 1024; ++i)
      array[i] = i + 1;
  }
}

void non_promotable(int *array, int c) {
#pragma omp target teams distribute parallel for collapse(2)
  for (int i = 0; i < 32; ++i)
    for (int j = 0; j < 32; ++j)
      array[i * 32 + j] = i + j;

#pragma omp target teams distribute parallel for schedule(static)
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;

#pragma omp target teams distribute parallel for schedule(guided)
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;

#pragma omp target teams distribute parallel for dist_schedule(static)
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;

#pragma omp target teams distribute parallel for
  for (int i = 0; i < 1024; ++i) {
    array[i] = i + 1;
#pragma omp cancel for
  }

#pragma omp target
  {
#pragma omp teams loop
    for (int i = 0; i < 1024; ++i)
      array[i] = foo(i);
  }

#pragma omp target teams distribute parallel for if(parallel : c)
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;

#pragma omp target teams distribute parallel for if(c)
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;

#pragma omp target teams
  {
#pragma omp distribute parallel for if(parallel : c)
    for (int i = 0; i < 1024; ++i)
      array[i] = i + 1;
  }

#pragma omp target teams loop if(c)
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;
}

#endif

// NOLOOP-COUNT-18: promotable_no_loop{{.*}}_kernel_environment {{.*}} i8 0, i8 1, i8 6
// NOLOOP-COUNT-2: promotable_strided{{.*}}_kernel_environment {{.*}} i8 0, i8 1, i8 10
// NOLOOP-COUNT-10: non_promotable{{.*}}_kernel_environment {{.*}} i8 0, i8 1, i8 2

// STRIDED-COUNT-18: promotable_no_loop{{.*}}_kernel_environment {{.*}} i8 0, i8 1, i8 10
// STRIDED-COUNT-2: promotable_strided{{.*}}_kernel_environment {{.*}} i8 0, i8 1, i8 10
// STRIDED-COUNT-10: non_promotable{{.*}}_kernel_environment {{.*}} i8 0, i8 1, i8 2

// no clause
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l54_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l54_{{.*}}, i32 0, i32 0, i8 1)
// STRIDED: @__kmpc_distribute_for_static_loop_4u({{.*}}_l54_{{.*}}, i32 0, i32 0, i8 0)
// CHECK: omp_loop.after:
// CHECK-NEXT: ret void

// simd
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l58_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: store i32 1024, ptr %i
// CHECK-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l58_{{.*}}, i32 0, i32 0, i8 1)
// STRIDED: @__kmpc_distribute_for_static_loop_4u({{.*}}_l58_{{.*}}, i32 0, i32 0, i8 0)
// CHECK: omp_loop.after:
// CHECK-NEXT: ret void

// nowait
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l62_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l62_{{.*}}, i32 0, i32 0, i8 1)
// STRIDED: @__kmpc_distribute_for_static_loop_4u({{.*}}_l62_{{.*}}, i32 0, i32 0, i8 0)
// CHECK: omp_loop.exit:
// CHECK-NEXT: br label %omp_loop.after
// CHECK: omp_loop.after:
// CHECK-NEXT: ret void

// runtime trip count
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l66_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: br label %omp.precond.end
// CHECK: [[LAST:%.*]] = load i32, ptr %.capture_expr.1.ascast
// CHECK-NEXT: [[TC:%.*]] = add nsw i32 [[LAST]], 1
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l66_{{.*}}, i32 [[TC]], i32 %{{.*}}, i32 0, i32 0, i8 1)
// STRIDED: @__kmpc_distribute_for_static_loop_4u({{.*}}_l66_{{.*}}, i32 [[TC]], i32 %{{.*}}, i32 0, i32 0, i8 0)
// CHECK: omp_loop.after:
// CHECK-NEXT: ret void

// private
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l72_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: ret void
// CHECK: store ptr %tmp1.ascast, ptr addrspace(5) %gep_tmp1.ascast
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l72_{{.*}}, i32 0, i32 0, i8 1)
// STRIDED: @__kmpc_distribute_for_static_loop_4u({{.*}}_l72_{{.*}}, i32 0, i32 0, i8 0)
// CHECK: omp_loop.after:
// CHECK-NEXT: ret void
// CHECK: store i32 {{.*}}, ptr %loadgep_tmp1.ascast

// lambda
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l81_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: ret void
// CHECK: [[SET:%.*]] = load ptr, ptr %set.addr.ascast
// CHECK-NEXT: [[FIELD:%.*]] = getelementptr inbounds nuw %class.anon, ptr [[SET]], i32 0, i32 0
// CHECK-NEXT: store ptr %array.addr.ascast, ptr [[FIELD]]
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l81_{{.*}}, i32 0, i32 0, i8 1)
// STRIDED: @__kmpc_distribute_for_static_loop_4u({{.*}}_l81_{{.*}}, i32 0, i32 0, i8 0)
// CHECK: omp_loop.after:
// CHECK-NEXT: ret void

// lastprivate loop counter
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l88_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: store i32 1024, ptr %i
// CHECK-NEXT: @__kmpc_free_shared(ptr %i{{.*}})
// CHECK-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l88_{{.*}}, i32 0, i32 0, i8 1)
// STRIDED: @__kmpc_distribute_for_static_loop_4u({{.*}}_l88_{{.*}}, i32 0, i32 0, i8 0)
// CHECK: @__kmpc_barrier
// CHECK: omp_loop.after:
// CHECK-NEXT: ret void

// lastprivate scalar
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l95_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: @__kmpc_free_shared(ptr %last{{.*}})
// CHECK-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l95_{{.*}}, i32 0, i32 0, i8 1)
// STRIDED: @__kmpc_distribute_for_static_loop_4u({{.*}}_l95_{{.*}}, i32 0, i32 0, i8 0)
// CHECK: @__kmpc_barrier
// CHECK: store {{.*}}, ptr %last.
// CHECK-NEXT: %.omp.lastprivate.done

// lastprivate scalar, nowait
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l104_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: @__kmpc_free_shared(ptr %last{{.*}})
// CHECK-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l104_{{.*}}, i32 0, i32 0, i8 1)
// STRIDED: @__kmpc_distribute_for_static_loop_4u({{.*}}_l104_{{.*}}, i32 0, i32 0, i8 0)
// CHECK: @__kmpc_barrier
// CHECK: store {{.*}}, ptr %last.
// CHECK-NEXT: %.omp.lastprivate.done

// simd linear loop counter
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l113_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: store i32 1024, ptr %i
// CHECK-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l113_{{.*}}, i32 0, i32 0, i8 1)
// STRIDED: @__kmpc_distribute_for_static_loop_4u({{.*}}_l113_{{.*}}, i32 0, i32 0, i8 0)
// CHECK: omp_loop.after:
// CHECK-NEXT: ret void

// target teams loop
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l118_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l118_{{.*}}, i32 0, i32 0, i8 1)
// STRIDED: @__kmpc_distribute_for_static_loop_4u({{.*}}_l118_{{.*}}, i32 0, i32 0, i8 0)
// CHECK: omp_loop.after:
// CHECK-NEXT: ret void

// target + teams loop
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l122_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l122_{{.*}}, i32 0, i32 0, i8 1)
// STRIDED: @__kmpc_distribute_for_static_loop_4u({{.*}}_l122_{{.*}}, i32 0, i32 0, i8 0)
// CHECK: omp_loop.after:
// CHECK-NEXT: ret void

// target teams + distribute parallel for
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l129_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l129_{{.*}}, i32 0, i32 0, i8 1)
// STRIDED: @__kmpc_distribute_for_static_loop_4u({{.*}}_l129_{{.*}}, i32 0, i32 0, i8 0)
// CHECK: omp_loop.after:
// CHECK-NEXT: ret void

// target + teams + distribute parallel for
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l136_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l136_{{.*}}, i32 0, i32 0, i8 1)
// STRIDED: @__kmpc_distribute_for_static_loop_4u({{.*}}_l136_{{.*}}, i32 0, i32 0, i8 0)
// CHECK: omp_loop.after:
// CHECK-NEXT: ret void

// num_teams(3)
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_strided{{.*}}_l164_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: ret void
// CHECK: @__kmpc_distribute_for_static_loop_4u({{.*}}_l164_{{.*}}, i32 0, i32 0, i8 0)
// CHECK: omp_loop.after:
// CHECK-NEXT: ret void

// num_teams(3)
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_strided{{.*}}_l168_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: ret void
// CHECK: @__kmpc_distribute_for_static_loop_4u({{.*}}_l168_{{.*}}, i32 0, i32 0, i8 0)
// CHECK: omp_loop.after:
// CHECK-NEXT: ret void
