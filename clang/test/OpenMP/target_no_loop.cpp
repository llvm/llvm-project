// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux-gnu \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-host.bc

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple amdgcn-amd-amdhsa \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-host.bc \
// RUN:   -fopenmp-assume-teams-oversubscription \
// RUN:   -fopenmp-assume-threads-oversubscription \
// RUN:   -emit-llvm %s -o - | FileCheck %s \
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
}

void non_promotable(int *array) {
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
}

// NOLOOP-COUNT-10: promotable_no_loop{{.*}}_kernel_environment {{.*}} i8 0, i8 1, i8 6
// NOLOOP-COUNT-5: non_promotable{{.*}}_kernel_environment {{.*}} i8 0, i8 1, i8 2

// SPMD-COUNT-10: promotable_no_loop{{.*}}_kernel_environment {{.*}} i8 0, i8 1, i8 2
// SPMD-COUNT-5: non_promotable{{.*}}_kernel_environment {{.*}} i8 0, i8 1, i8 2

// no clause
// NOLOOP-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l37_{{.*}})
// NOLOOP: omp.loop.exit:
// NOLOOP-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l37_{{.*}}, i32 0, i32 0, i8 1)
// NOLOOP: omp_loop.after:
// NOLOOP-NEXT: ret void

// simd
// NOLOOP-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l41_{{.*}})
// NOLOOP: omp.loop.exit:
// NOLOOP-NEXT: store i32 1024, ptr %i
// NOLOOP-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l41_{{.*}}, i32 0, i32 0, i8 1)
// NOLOOP: omp_loop.after:
// NOLOOP-NEXT: ret void

// nowait
// NOLOOP-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l45_{{.*}})
// NOLOOP: omp.loop.exit:
// NOLOOP-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l45_{{.*}}, i32 0, i32 0, i8 1)
// NOLOOP: omp_loop.exit:
// NOLOOP-NEXT: br label %omp_loop.after
// NOLOOP: omp_loop.after:
// NOLOOP-NEXT: ret void

// runtime trip count
// NOLOOP-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l49_{{.*}})
// NOLOOP: omp.loop.exit:
// NOLOOP-NEXT: br label %omp.precond.end
// NOLOOP: [[LAST:%.*]] = load i32, ptr %.capture_expr.1.ascast
// NOLOOP-NEXT: [[TC:%.*]] = add nsw i32 [[LAST]], 1
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l49_{{.*}}, i32 [[TC]], i32 %{{.*}}, i32 0, i32 0, i8 1)
// NOLOOP: omp_loop.after:
// NOLOOP-NEXT: ret void

// private
// NOLOOP-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l55_{{.*}})
// NOLOOP: omp.loop.exit:
// NOLOOP-NEXT: ret void
// NOLOOP: store ptr %tmp1.ascast, ptr addrspace(5) %gep_tmp1.ascast
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l55_{{.*}}, i32 0, i32 0, i8 1)
// NOLOOP: omp_loop.after:
// NOLOOP-NEXT: ret void
// NOLOOP: store i32 {{.*}}, ptr %loadgep_tmp1.ascast

// lambda
// NOLOOP-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l64_{{.*}})
// NOLOOP: omp.loop.exit:
// NOLOOP-NEXT: ret void
// NOLOOP: [[SET:%.*]] = load ptr, ptr %set.addr.ascast
// NOLOOP-NEXT: [[FIELD:%.*]] = getelementptr inbounds nuw %class.anon, ptr [[SET]], i32 0, i32 0
// NOLOOP-NEXT: store ptr %array.addr.ascast, ptr [[FIELD]]
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l64_{{.*}}, i32 0, i32 0, i8 1)
// NOLOOP: omp_loop.after:
// NOLOOP-NEXT: ret void

// lastprivate loop counter
// NOLOOP-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l71_{{.*}})
// NOLOOP: omp.loop.exit:
// NOLOOP-NEXT: store i32 1024, ptr %i
// NOLOOP-NEXT: @__kmpc_free_shared(ptr %i{{.*}})
// NOLOOP-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l71_{{.*}}, i32 0, i32 0, i8 1)
// NOLOOP: @__kmpc_barrier
// NOLOOP: omp_loop.after:
// NOLOOP-NEXT: ret void

// lastprivate scalar
// NOLOOP-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l78_{{.*}})
// NOLOOP: omp.loop.exit:
// NOLOOP-NEXT: @__kmpc_free_shared(ptr %last{{.*}})
// NOLOOP-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l78_{{.*}}, i32 0, i32 0, i8 1)
// NOLOOP: @__kmpc_barrier
// NOLOOP: store {{.*}}, ptr %last.
// NOLOOP-NEXT: %.omp.lastprivate.done

// lastprivate scalar, nowait
// NOLOOP-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l87_{{.*}})
// NOLOOP: omp.loop.exit:
// NOLOOP-NEXT: @__kmpc_free_shared(ptr %last{{.*}})
// NOLOOP-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l87_{{.*}}, i32 0, i32 0, i8 1)
// NOLOOP: @__kmpc_barrier
// NOLOOP: store {{.*}}, ptr %last.
// NOLOOP-NEXT: %.omp.lastprivate.done

// simd linear loop counter
// NOLOOP-LABEL: @__kmpc_parallel_60({{.*}}promotable_no_loop{{.*}}_l96_{{.*}})
// NOLOOP: omp.loop.exit:
// NOLOOP-NEXT: store i32 1024, ptr %i
// NOLOOP-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}_l96_{{.*}}, i32 0, i32 0, i8 1)
// NOLOOP: omp_loop.after:
// NOLOOP-NEXT: ret void
