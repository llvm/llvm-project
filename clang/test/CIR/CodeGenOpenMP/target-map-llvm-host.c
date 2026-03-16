// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=OGCG

void use(int);

void target_map_to(int x) {
#pragma omp target map(to : x)
  {
    use(x);
  }
}

void target_map_from(int x) {
#pragma omp target map(from : x)
  {
    x = 42;
  }
}

void target_map_tofrom(int x) {
#pragma omp target map(tofrom : x)
  {
    x = x + 1;
  }
}

void target_map_multiple(int a, int b) {
#pragma omp target map(to : a) map(from : b)
  {
    b = a;
  }
}

// Host wrappers

// LLVM-LABEL: define {{.*}} void @target_map_to(i32
// LLVM:         call i32 @__tgt_target_kernel(
// LLVM:       omp_offload.failed:
// LLVM:         call void @__omp_offloading_{{.*}}_target_map_to_l

// LLVM-LABEL: define {{.*}} void @target_map_from(i32
// LLVM:         call i32 @__tgt_target_kernel(
// LLVM:       omp_offload.failed:
// LLVM:         call void @__omp_offloading_{{.*}}_target_map_from_l

// LLVM-LABEL: define {{.*}} void @target_map_tofrom(i32
// LLVM:         call i32 @__tgt_target_kernel(
// LLVM:       omp_offload.failed:
// LLVM:         call void @__omp_offloading_{{.*}}_target_map_tofrom_l

// LLVM-LABEL: define {{.*}} void @target_map_multiple(i32
// LLVM:         call i32 @__tgt_target_kernel(
// LLVM:       omp_offload.failed:
// LLVM:         call void @__omp_offloading_{{.*}}_target_map_multiple_l

// Outlined target functions

// LLVM-LABEL: define internal void @__omp_offloading_{{.*}}_target_map_to_l
// LLVM:         %[[V:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM:         call void @use(i32 {{.*}} %[[V]])
// LLVM:         ret void

// LLVM-LABEL: define internal void @__omp_offloading_{{.*}}_target_map_from_l
// LLVM:         store i32 42, ptr %{{.*}}, align 4
// LLVM:         ret void

// LLVM-LABEL: define internal void @__omp_offloading_{{.*}}_target_map_tofrom_l
// LLVM:         %[[LD:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM:         %[[ADD:.*]] = add nsw i32 %[[LD]], 1
// LLVM:         store i32 %[[ADD]], ptr %{{.*}}, align 4
// LLVM:         ret void

// LLVM-LABEL: define internal void @__omp_offloading_{{.*}}_target_map_multiple_l
// LLVM:         %[[A:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM:         store i32 %[[A]], ptr %{{.*}}, align 4
// LLVM:         ret void

// OGCG interleaves host wrapper and outlined function per target region.

// OGCG-LABEL: define {{.*}} void @target_map_to(i32
// OGCG:         call i32 @__tgt_target_kernel(
// OGCG:       omp_offload.failed:
// OGCG:         call void @__omp_offloading_{{.*}}_target_map_to_l

// OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_map_to_l
// OGCG:         %[[V:.*]] = load i32, ptr %{{.*}}, align 4
// OGCG:         call void @use(i32 {{.*}} %[[V]])
// OGCG:         ret void

// OGCG-LABEL: define {{.*}} void @target_map_from(i32
// OGCG:         call i32 @__tgt_target_kernel(
// OGCG:       omp_offload.failed:
// OGCG:         call void @__omp_offloading_{{.*}}_target_map_from_l

// OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_map_from_l
// OGCG:         store i32 42, ptr %{{.*}}, align 4
// OGCG:         ret void

// OGCG-LABEL: define {{.*}} void @target_map_tofrom(i32
// OGCG:         call i32 @__tgt_target_kernel(
// OGCG:       omp_offload.failed:
// OGCG:         call void @__omp_offloading_{{.*}}_target_map_tofrom_l

// OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_map_tofrom_l
// OGCG:         %[[LD:.*]] = load i32, ptr %{{.*}}, align 4
// OGCG:         %[[ADD:.*]] = add nsw i32 %[[LD]], 1
// OGCG:         store i32 %[[ADD]], ptr %{{.*}}, align 4
// OGCG:         ret void

// OGCG-LABEL: define {{.*}} void @target_map_multiple(i32
// OGCG:         call i32 @__tgt_target_kernel(
// OGCG:       omp_offload.failed:
// OGCG:         call void @__omp_offloading_{{.*}}_target_map_multiple_l

// OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_map_multiple_l
// OGCG:         %[[A:.*]] = load i32, ptr %{{.*}}, align 4
// OGCG:         store i32 %[[A]], ptr %{{.*}}, align 4
// OGCG:         ret void
