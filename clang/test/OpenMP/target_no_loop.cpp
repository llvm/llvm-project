// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux-gnu \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-host.bc

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux-gnu \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:   -fopenmp-assume-teams-oversubscription \
// RUN:   -fopenmp-assume-threads-oversubscription \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=HOST

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux-gnu \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -o - | FileCheck %s \
// RUN:   --check-prefix=HOST-NOFLAGS --implicit-check-not=omp.tripcount

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple amdgcn-amd-amdhsa \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-host.bc \
// RUN:   -fopenmp-assume-teams-oversubscription \
// RUN:   -fopenmp-assume-threads-oversubscription \
// RUN:   -emit-llvm %s -o - | FileCheck %s \
// RUN:   --check-prefixes=NOLOOP

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
// RUN:   --check-prefixes=NOLOOP

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple amdgcn-amd-amdhsa \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-host.bc \
// RUN:   -emit-llvm %s -o - | FileCheck %s \
// RUN:   --check-prefix=SPMD --implicit-check-not=__kmpc_distribute_for_static_loop_4u

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple amdgcn-amd-amdhsa \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-host.bc \
// RUN:   -fopenmp-assume-teams-oversubscription \
// RUN:   -emit-llvm %s -o - | FileCheck %s \
// RUN:   --check-prefix=SPMD --implicit-check-not=__kmpc_distribute_for_static_loop_4u

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple amdgcn-amd-amdhsa \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-host.bc \
// RUN:   -fopenmp-assume-threads-oversubscription \
// RUN:   -emit-llvm %s -o - | FileCheck %s \
// RUN:   --check-prefix=SPMD --implicit-check-not=__kmpc_distribute_for_static_loop_4u

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

int foo(int i);

void promotable_no_loop(int *array, int n) {
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

#pragma omp target teams distribute parallel for if(1)
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;

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
}

void non_promotable(int *array, int n) {
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

#pragma omp target teams distribute parallel for if(parallel: n > 0)
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;

#pragma omp target teams distribute parallel for if(n > 0)
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;

#pragma omp target
  {
#pragma omp teams loop
    for (int i = 0; i < 1024; ++i)
      array[i] = foo(i);
  }
}

#pragma omp declare target
static int offset = 1;
#pragma omp end declare target

void promotable_after_reduction(int *array) {
  int sum = 0;
#pragma omp target teams distribute parallel for reduction(+ : sum)
  for (int i = 0; i < 1024; ++i)
    sum += array[i];

#pragma omp target teams distribute parallel for
  for (int i = 0; i < 1024; ++i)
    array[i] = i + offset;
}

#endif

// NOLOOP-COUNT-17: promotable_no_loop{{.*}}_kernel_environment {{.*}} i8 0, i8 1, i8 6
// NOLOOP-COUNT-8: non_promotable{{.*}}_kernel_environment {{.*}} i8 0, i8 1, i8 2
// NOLOOP: promotable_after_reduction{{.*}}_kernel_environment {{.*}} i8 0, i8 1, i8 2
// NOLOOP: promotable_after_reduction{{.*}}_kernel_environment {{.*}} i8 0, i8 1, i8 6

// SPMD-COUNT-17: promotable_no_loop{{.*}}_kernel_environment {{.*}} i8 0, i8 1, i8 2
// SPMD-COUNT-8: non_promotable{{.*}}_kernel_environment {{.*}} i8 0, i8 1, i8 2
// SPMD-COUNT-2: promotable_after_reduction{{.*}}_kernel_environment {{.*}} i8 0, i8 1, i8 2

// reduction before a promoted kernel
// NOLOOP: @llvm.compiler.used = {{.*}}$ref
// NOLOOP-NOT: @llvm.compiler.used.

// empty range reports a zero trip count
// HOST-LABEL: define {{.*}}@_Z18promotable_no_loopPii(
// HOST: %.omp.tripcount = alloca i64
// HOST: store i64 0, ptr %.omp.tripcount
// HOST: br i1 {{.*}}, label %omp.tripcount.then, label %omp.tripcount.end
// HOST: omp.tripcount.then:
// HOST: store i64 {{.*}}, ptr %.omp.tripcount
// HOST: omp.tripcount.end:
// HOST: load i64, ptr %.omp.tripcount

// HOST-NOFLAGS-LABEL: define {{.*}}@_Z18promotable_no_loopPii(

// no clause
// NOLOOP-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l67_{{.*}})
// NOLOOP: omp.loop.exit:
// NOLOOP-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l67_{{.*}}, i32 0, i32 0, i8 1)
// NOLOOP: omp_loop.after:
// NOLOOP-NEXT: ret void

// simd
// NOLOOP-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l71_{{.*}})
// NOLOOP: omp.loop.exit:
// NOLOOP-NEXT: store i32 1024, ptr %i
// NOLOOP-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l71_{{.*}}, i32 0, i32 0, i8 1)
// NOLOOP: omp_loop.after:
// NOLOOP-NEXT: ret void

// nowait
// NOLOOP-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l75_{{.*}})
// NOLOOP: omp.loop.exit:
// NOLOOP-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l75_{{.*}}, i32 0, i32 0, i8 1)
// NOLOOP: omp_loop.exit:
// NOLOOP-NEXT: br label %omp_loop.after
// NOLOOP: omp_loop.after:
// NOLOOP-NEXT: ret void

// runtime trip count
// NOLOOP-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l79_{{.*}})
// NOLOOP: omp.loop.exit:
// NOLOOP-NEXT: br label %omp.precond.end
// NOLOOP: [[LAST:%.*]] = load i32, ptr %.capture_expr.1.ascast
// NOLOOP-NEXT: [[TC:%.*]] = add nsw i32 [[LAST]], 1
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l79_{{.*}}, i32 [[TC]], i32 %{{.*}}, i32 0, i32 0, i8 1)
// NOLOOP: omp_loop.after:
// NOLOOP-NEXT: ret void

// private
// NOLOOP-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l85_{{.*}})
// NOLOOP: omp.loop.exit:
// NOLOOP-NEXT: ret void
// NOLOOP: store ptr %tmp1.ascast, ptr addrspace(5) %gep_tmp1.ascast
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l85_{{.*}}, i32 0, i32 0, i8 1)
// NOLOOP: omp_loop.after:
// NOLOOP-NEXT: ret void
// NOLOOP: store i32 {{.*}}, ptr %loadgep_tmp1.ascast

// lambda
// NOLOOP-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l94_{{.*}})
// NOLOOP: omp.loop.exit:
// NOLOOP-NEXT: ret void
// NOLOOP: [[SET:%.*]] = load ptr, ptr %set.addr.ascast
// NOLOOP-NEXT: [[FIELD:%.*]] = getelementptr inbounds nuw %class.anon, ptr [[SET]], i32 0, i32 0
// NOLOOP-NEXT: store ptr %array.addr.ascast, ptr [[FIELD]]
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l94_{{.*}}, i32 0, i32 0, i8 1)
// NOLOOP: omp_loop.after:
// NOLOOP-NEXT: ret void

// lastprivate loop counter
// NOLOOP-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l105_{{.*}})
// NOLOOP: omp.loop.exit:
// NOLOOP-NEXT: store i32 1024, ptr %i
// NOLOOP-NEXT: @__kmpc_free_shared(ptr %i{{.*}})
// NOLOOP-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l105_{{.*}}, i32 0, i32 0, i8 1)
// NOLOOP: @__kmpc_barrier
// NOLOOP: omp_loop.after:
// NOLOOP-NEXT: ret void

// lastprivate scalar
// NOLOOP-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l112_{{.*}})
// NOLOOP: omp.loop.exit:
// NOLOOP-NEXT: @__kmpc_free_shared(ptr %last{{.*}})
// NOLOOP-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l112_{{.*}}, i32 0, i32 0, i8 1)
// NOLOOP: @__kmpc_barrier
// NOLOOP: store {{.*}}, ptr %last.
// NOLOOP-NEXT: %.omp.lastprivate.done

// lastprivate scalar, nowait
// NOLOOP-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l121_{{.*}})
// NOLOOP: omp.loop.exit:
// NOLOOP-NEXT: @__kmpc_free_shared(ptr %last{{.*}})
// NOLOOP-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l121_{{.*}}, i32 0, i32 0, i8 1)
// NOLOOP: @__kmpc_barrier
// NOLOOP: store {{.*}}, ptr %last.
// NOLOOP-NEXT: %.omp.lastprivate.done

// simd linear loop counter
// NOLOOP-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l130_{{.*}})
// NOLOOP: omp.loop.exit:
// NOLOOP-NEXT: store i32 1024, ptr %i
// NOLOOP-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l130_{{.*}}, i32 0, i32 0, i8 1)
// NOLOOP: omp_loop.after:
// NOLOOP-NEXT: ret void

// target teams loop
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}promotable_no_loop{{.*}}_l135_{{.*}})

// target + teams loop
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}promotable_no_loop{{.*}}_l139_{{.*}})
