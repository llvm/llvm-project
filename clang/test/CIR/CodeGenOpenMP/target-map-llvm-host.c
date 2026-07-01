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
//
// LLVM-LABEL: define {{.*}} void @target_map_to(
// LLVM-SAME:  i32 noundef %[[ARG:[^,)]+]]
// LLVM:         %[[X_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM:         store i32 %[[ARG]], ptr %[[X_ADDR]], align 4
// LLVM:         %[[BP:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// LLVM:         store ptr %[[X_ADDR]], ptr %[[BP]], align 8
// LLVM:         %[[P:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// LLVM:         store ptr %[[X_ADDR]], ptr %[[P]], align 8
// LLVM:         call i32 @__tgt_target_kernel(
// LLVM:       omp_offload.failed:
// LLVM:         call void @__omp_offloading_{{.*}}_target_map_to_l{{.*}}(ptr %[[X_ADDR]], ptr null)

// LLVM-LABEL: define {{.*}} void @target_map_from(
// LLVM-SAME:  i32 noundef %[[ARG:[^,)]+]]
// LLVM:         %[[X_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM:         store i32 %[[ARG]], ptr %[[X_ADDR]], align 4
// LLVM:         %[[BP:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// LLVM:         store ptr %[[X_ADDR]], ptr %[[BP]], align 8
// LLVM:         %[[P:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// LLVM:         store ptr %[[X_ADDR]], ptr %[[P]], align 8
// LLVM:         call i32 @__tgt_target_kernel(
// LLVM:       omp_offload.failed:
// LLVM:         call void @__omp_offloading_{{.*}}_target_map_from_l{{.*}}(ptr %[[X_ADDR]], ptr null)

// LLVM-LABEL: define {{.*}} void @target_map_tofrom(
// LLVM-SAME:  i32 noundef %[[ARG:[^,)]+]]
// LLVM:         %[[X_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM:         store i32 %[[ARG]], ptr %[[X_ADDR]], align 4
// LLVM:         %[[BP:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// LLVM:         store ptr %[[X_ADDR]], ptr %[[BP]], align 8
// LLVM:         %[[P:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// LLVM:         store ptr %[[X_ADDR]], ptr %[[P]], align 8
// LLVM:         call i32 @__tgt_target_kernel(
// LLVM:       omp_offload.failed:
// LLVM:         call void @__omp_offloading_{{.*}}_target_map_tofrom_l{{.*}}(ptr %[[X_ADDR]], ptr null)

// LLVM-LABEL: define {{.*}} void @target_map_multiple(
// LLVM-SAME:  i32 noundef %[[ARG_A:[^,)]+]], i32 noundef %[[ARG_B:[^,)]+]]
// LLVM:         %[[A_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM:         %[[B_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM:         store i32 %[[ARG_A]], ptr %[[A_ADDR]], align 4
// LLVM:         store i32 %[[ARG_B]], ptr %[[B_ADDR]], align 4
// LLVM:         %[[BP_A:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// LLVM:         store ptr %[[A_ADDR]], ptr %[[BP_A]], align 8
// LLVM:         %[[P_A:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// LLVM:         store ptr %[[A_ADDR]], ptr %[[P_A]], align 8
// LLVM:         %[[BP_B:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
// LLVM:         store ptr %[[B_ADDR]], ptr %[[BP_B]], align 8
// LLVM:         %[[P_B:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_ptrs, i32 0, i32 1
// LLVM:         store ptr %[[B_ADDR]], ptr %[[P_B]], align 8
// LLVM:         call i32 @__tgt_target_kernel(
// LLVM:       omp_offload.failed:
// LLVM:         call void @__omp_offloading_{{.*}}_target_map_multiple_l{{.*}}(ptr %[[A_ADDR]], ptr %[[B_ADDR]], ptr null)

// Outlined target functions
//
// The mapped pointer arrives as the first function argument; load/store the
// user value directly through it.

// LLVM-LABEL: define internal void @__omp_offloading_{{.*}}_target_map_to_l
// LLVM-SAME:  (ptr %[[ARG:[^,]+]], ptr
// LLVM:         %[[V:.*]] = load i32, ptr %[[ARG]], align 4
// LLVM:         call void @use(i32 {{.*}} %[[V]])
// LLVM:         ret void

// LLVM-LABEL: define internal void @__omp_offloading_{{.*}}_target_map_from_l
// LLVM-SAME:  (ptr %[[ARG:[^,]+]], ptr
// LLVM:         store i32 42, ptr %[[ARG]], align 4
// LLVM:         ret void

// LLVM-LABEL: define internal void @__omp_offloading_{{.*}}_target_map_tofrom_l
// LLVM-SAME:  (ptr %[[ARG:[^,]+]], ptr
// LLVM:         %[[LD:.*]] = load i32, ptr %[[ARG]], align 4
// LLVM:         %[[ADD:.*]] = add nsw i32 %[[LD]], 1
// LLVM:         store i32 %[[ADD]], ptr %[[ARG]], align 4
// LLVM:         ret void

// LLVM-LABEL: define internal void @__omp_offloading_{{.*}}_target_map_multiple_l
// LLVM-SAME:  (ptr %[[ARG_A:[^,]+]], ptr %[[ARG_B:[^,]+]], ptr
// LLVM:         %[[A:.*]] = load i32, ptr %[[ARG_A]], align 4
// LLVM:         store i32 %[[A]], ptr %[[ARG_B]], align 4
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
