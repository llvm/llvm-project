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
int omp_get_team_num();

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

#pragma omp target teams distribute parallel for if(1)
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;

#pragma omp target teams distribute parallel for if(target : c)
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;

  {
    int i;
#pragma omp target teams loop lastprivate(i)
    for (i = 0; i < 1024; ++i)
      array[i] = i + 1;
  }

  {
    int i;
#pragma omp target
    {
#pragma omp teams loop lastprivate(i)
      for (i = 0; i < 1024; ++i)
        array[i] = i + 1;
    }
  }
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

#pragma omp target
  {
#pragma omp teams loop num_teams(4)
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

#pragma omp target
  {
#pragma omp teams loop
    for (int i = 0; i < 1024; ++i) {
      foo(i);
      array[i] = omp_get_team_num();
    }
  }

#pragma omp target teams loop
  for (int i = 0; i < 1024; ++i) {
    foo(i);
    array[i] = omp_get_team_num();
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

// NOLOOP-COUNT-21: promotable_no_loop{{.*}}_kernel_environment {{.*}} i8 0, i8 1, i8 6
// NOLOOP-COUNT-3: promotable_strided{{.*}}_kernel_environment {{.*}} i8 0, i8 1, i8 10
// NOLOOP-COUNT-11: non_promotable{{.*}}_kernel_environment {{.*}} i8 0, i8 1, i8 2
// NOLOOP: non_promotable{{.*}}_kernel_environment {{.*}} i8 1, i8 1, i8 1
// NOLOOP: promotable_after_reduction{{.*}}_kernel_environment {{.*}} i8 0, i8 1, i8 2
// NOLOOP: promotable_after_reduction{{.*}}_kernel_environment {{.*}} i8 0, i8 1, i8 6

// STRIDED-COUNT-21: promotable_no_loop{{.*}}_kernel_environment {{.*}} i8 0, i8 1, i8 10
// STRIDED-COUNT-3: promotable_strided{{.*}}_kernel_environment {{.*}} i8 0, i8 1, i8 10
// STRIDED-COUNT-11: non_promotable{{.*}}_kernel_environment {{.*}} i8 0, i8 1, i8 2
// STRIDED: non_promotable{{.*}}_kernel_environment {{.*}} i8 1, i8 1, i8 1
// STRIDED: promotable_after_reduction{{.*}}_kernel_environment {{.*}} i8 0, i8 1, i8 2
// STRIDED: promotable_after_reduction{{.*}}_kernel_environment {{.*}} i8 0, i8 1, i8 10

// reduction before a promoted kernel
// CHECK: @llvm.compiler.used = {{.*}}$ref
// CHECK-NOT: @llvm.compiler.used.

// empty range reports a zero trip count
// HOST-LABEL: define {{.*}}@_Z18promotable_no_loopPiii(
// HOST: %.omp.tripcount = alloca i64
// HOST: store i64 0, ptr %.omp.tripcount
// HOST: br i1 {{.*}}, label %omp.tripcount.then, label %omp.tripcount.end
// HOST: omp.tripcount.then:
// HOST: store i64 {{.*}}, ptr %.omp.tripcount
// HOST: omp.tripcount.end:
// HOST: load i64, ptr %.omp.tripcount

// HOST-NOFLAGS-LABEL: define {{.*}}@_Z18promotable_no_loopPiii(

// no clause
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l65_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l65_{{.*}}, i32 0, i32 0, i8 1)
// STRIDED: @__kmpc_distribute_for_static_loop_4u({{.*}}_l65_{{.*}}, i32 0, i32 0, i8 0)
// CHECK: omp_loop.after:
// CHECK-NEXT: ret void

// simd
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l69_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: store i32 1024, ptr %i
// CHECK-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l69_{{.*}}, i32 0, i32 0, i8 1)
// STRIDED: @__kmpc_distribute_for_static_loop_4u({{.*}}_l69_{{.*}}, i32 0, i32 0, i8 0)
// CHECK: omp_loop.after:
// CHECK-NEXT: ret void

// nowait
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l73_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l73_{{.*}}, i32 0, i32 0, i8 1)
// STRIDED: @__kmpc_distribute_for_static_loop_4u({{.*}}_l73_{{.*}}, i32 0, i32 0, i8 0)
// CHECK: omp_loop.exit:
// CHECK-NEXT: br label %omp_loop.after
// CHECK: omp_loop.after:
// CHECK-NEXT: ret void

// runtime trip count
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l77_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: br label %omp.precond.end
// CHECK: [[LAST:%.*]] = load i32, ptr %.capture_expr.1.ascast
// CHECK-NEXT: [[TC:%.*]] = add nsw i32 [[LAST]], 1
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l77_{{.*}}, i32 [[TC]], i32 %{{.*}}, i32 0, i32 0, i8 1)
// STRIDED: @__kmpc_distribute_for_static_loop_4u({{.*}}_l77_{{.*}}, i32 [[TC]], i32 %{{.*}}, i32 0, i32 0, i8 0)
// CHECK: omp_loop.after:
// CHECK-NEXT: ret void

// private
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l83_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: ret void
// CHECK: store ptr %tmp1.ascast, ptr addrspace(5) %gep_tmp1.ascast
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l83_{{.*}}, i32 0, i32 0, i8 1)
// STRIDED: @__kmpc_distribute_for_static_loop_4u({{.*}}_l83_{{.*}}, i32 0, i32 0, i8 0)
// CHECK: omp_loop.after:
// CHECK-NEXT: ret void
// CHECK: store i32 {{.*}}, ptr %loadgep_tmp1.ascast

// lambda
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l92_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: ret void
// CHECK: [[SET:%.*]] = load ptr, ptr %set.addr.ascast
// CHECK-NEXT: [[FIELD:%.*]] = getelementptr inbounds nuw %class.anon, ptr [[SET]], i32 0, i32 0
// CHECK-NEXT: store ptr %array.addr.ascast, ptr [[FIELD]]
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l92_{{.*}}, i32 0, i32 0, i8 1)
// STRIDED: @__kmpc_distribute_for_static_loop_4u({{.*}}_l92_{{.*}}, i32 0, i32 0, i8 0)
// CHECK: omp_loop.after:
// CHECK-NEXT: ret void

// lastprivate loop counter
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l99_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: store i32 1024, ptr %i
// CHECK-NEXT: @__kmpc_free_shared(ptr %i{{.*}})
// CHECK-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l99_{{.*}}, i32 0, i32 0, i8 1)
// STRIDED: @__kmpc_distribute_for_static_loop_4u({{.*}}_l99_{{.*}}, i32 0, i32 0, i8 0)
// CHECK: @__kmpc_barrier
// CHECK: omp_loop.after:
// CHECK-NEXT: ret void

// lastprivate scalar
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l106_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: @__kmpc_free_shared(ptr %last{{.*}})
// CHECK-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l106_{{.*}}, i32 0, i32 0, i8 1)
// STRIDED: @__kmpc_distribute_for_static_loop_4u({{.*}}_l106_{{.*}}, i32 0, i32 0, i8 0)
// CHECK: @__kmpc_barrier
// CHECK: store {{.*}}, ptr %last.
// CHECK-NEXT: %.omp.lastprivate.done

// lastprivate scalar, nowait
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l115_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: @__kmpc_free_shared(ptr %last{{.*}})
// CHECK-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l115_{{.*}}, i32 0, i32 0, i8 1)
// STRIDED: @__kmpc_distribute_for_static_loop_4u({{.*}}_l115_{{.*}}, i32 0, i32 0, i8 0)
// CHECK-NOT: @__kmpc_barrier
// CHECK: store {{.*}}, ptr %last.
// CHECK-NEXT: %.omp.lastprivate.done

// simd linear loop counter
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l124_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: store i32 1024, ptr %i
// CHECK-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l124_{{.*}}, i32 0, i32 0, i8 1)
// STRIDED: @__kmpc_distribute_for_static_loop_4u({{.*}}_l124_{{.*}}, i32 0, i32 0, i8 0)
// CHECK: omp_loop.after:
// CHECK-NEXT: ret void

// target teams loop
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l129_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l129_{{.*}}, i32 0, i32 0, i8 1)
// STRIDED: @__kmpc_distribute_for_static_loop_4u({{.*}}_l129_{{.*}}, i32 0, i32 0, i8 0)
// CHECK: omp_loop.after:
// CHECK-NEXT: ret void

// target + teams loop
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l133_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l133_{{.*}}, i32 0, i32 0, i8 1)
// STRIDED: @__kmpc_distribute_for_static_loop_4u({{.*}}_l133_{{.*}}, i32 0, i32 0, i8 0)
// CHECK: omp_loop.after:
// CHECK-NEXT: ret void

// target teams + distribute parallel for
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l140_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l140_{{.*}}, i32 0, i32 0, i8 1)
// STRIDED: @__kmpc_distribute_for_static_loop_4u({{.*}}_l140_{{.*}}, i32 0, i32 0, i8 0)
// CHECK: omp_loop.after:
// CHECK-NEXT: ret void

// target + teams + distribute parallel for
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l147_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l147_{{.*}}, i32 0, i32 0, i8 1)
// STRIDED: @__kmpc_distribute_for_static_loop_4u({{.*}}_l147_{{.*}}, i32 0, i32 0, i8 0)
// CHECK: omp_loop.after:
// CHECK-NEXT: ret void

// target teams loop, lastprivate loop counter
// NOLOOP-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l179_{{.*}})
// NOLOOP: omp.loop.exit:
// NOLOOP-NEXT: store i32 1024, ptr %i
// NOLOOP-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l179_{{.*}}, i32 0, i32 0, i8 1)
// NOLOOP: @__kmpc_barrier
// NOLOOP: omp_loop.after:
// NOLOOP-NEXT: ret void

// target + teams loop, lastprivate loop counter
// NOLOOP-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l186_{{.*}})
// NOLOOP: omp.loop.exit:
// NOLOOP-NEXT: store i32 1024, ptr %{{[0-9]+}}
// NOLOOP-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l186_{{.*}}, i32 0, i32 0, i8 1)
// NOLOOP: @__kmpc_barrier
// NOLOOP: omp_loop.after:
// NOLOOP-NEXT: ret void

// num_teams(3)
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_strided{{.*}}_l196_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: ret void
// CHECK: @__kmpc_distribute_for_static_loop_4u({{.*}}_l196_{{.*}}, i32 0, i32 0, i8 0)
// CHECK: omp_loop.after:
// CHECK-NEXT: ret void

// num_teams(3)
// CHECK-LABEL: @__kmpc_parallel_60({{.*}}promotable_strided{{.*}}_l200_{{.*}})
// CHECK: omp.loop.exit:
// CHECK-NEXT: ret void
// CHECK: @__kmpc_distribute_for_static_loop_4u({{.*}}_l200_{{.*}}, i32 0, i32 0, i8 0)
// CHECK: omp_loop.after:
// CHECK-NEXT: ret void
