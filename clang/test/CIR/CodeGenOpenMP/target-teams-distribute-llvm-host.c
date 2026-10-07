// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=OGCG

void during(int);

void explicit_target_teams_distribute() {
#pragma omp target
  {
#pragma omp teams
    {
#pragma omp distribute
      for (int i = 0; i < 10; i++) {
        during(i);
      }
    }
  }
}

void target_teams_distribute() {
#pragma omp target teams distribute
  for (int i = 0; i < 10; i++) {
    during(i);
  }
}

// LLVM-LABEL: define {{.*}} void @explicit_target_teams_distribute(
// LLVM:         %[[TRIPCOUNT_GEP:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 8
// LLVM:         store i64 10, ptr %[[TRIPCOUNT_GEP]], align 8
// LLVM:         call i32 @__tgt_target_kernel(

// LLVM-LABEL: define {{.*}} void @target_teams_distribute(
// LLVM:         %[[TRIPCOUNT_GEP:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 8
// LLVM:         store i64 10, ptr %[[TRIPCOUNT_GEP]], align 8
// LLVM:         call i32 @__tgt_target_kernel(

// OGCG-LABEL: define {{.*}} void @explicit_target_teams_distribute(
// OGCG:         %[[TRIPCOUNT_GEP:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 8
// OGCG:         store i64 10, ptr %[[TRIPCOUNT_GEP]], align 8
// OGCG:         call i32 @__tgt_target_kernel(

// OGCG-LABEL: define {{.*}} void @target_teams_distribute(
// OGCG:         %[[TRIPCOUNT_GEP:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 8
// OGCG:         store i64 10, ptr %[[TRIPCOUNT_GEP]], align 8
// OGCG:         call i32 @__tgt_target_kernel(
