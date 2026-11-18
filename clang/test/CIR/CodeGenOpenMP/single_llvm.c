// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=LLVM,OGCG

// Check the LLVM IR of omp single from ClangIR and classic codegen. The
// prefixes are LLVM (checks shared by both paths), CIR (IR lowered from
// ClangIR) and OGCG (IR from the original, classic Clang codegen).

void work(void);
void other_work(void);

void single_only(void) {
#pragma omp single
  work();
}

// LLVM-LABEL: define {{.*}}void @single_only()
// LLVM:         call i32 @__kmpc_single(
// LLVM:         call void @work()
// LLVM:         call void @__kmpc_end_single(
// LLVM:         call void @__kmpc_barrier(

void nowait_single_only(void) {
#pragma omp single nowait
  work();
}

// LLVM-LABEL: define {{.*}}void @nowait_single_only()
// LLVM:         call i32 @__kmpc_single(
// LLVM:         call void @work()
// LLVM:         call void @__kmpc_end_single(
// LLVM-NOT:     @__kmpc_barrier

void single_inside_parallel(void) {
#pragma omp parallel
  {
#pragma omp single
    work();
  }
}

// CIR-LABEL: define {{.*}}void @single_inside_parallel()
// CIR:         call {{.*}}@__kmpc_fork_call(
// CIR-LABEL: define {{.*}}void @single_inside_parallel..omp_par(
// CIR:         call i32 @__kmpc_single(
// CIR-DAG:     call void @work()
// CIR-DAG:     call void @__kmpc_end_single(
// CIR-DAG:     call void @__kmpc_barrier(

// OGCG-LABEL: define {{.*}}void @single_inside_parallel()
// OGCG:         call {{.*}}@__kmpc_fork_call(
// OGCG-LABEL: define {{.*}}void @single_inside_parallel.omp_outlined(
// OGCG:         call i32 @__kmpc_single(
// OGCG:         call void @work()
// OGCG:         call void @__kmpc_end_single(
// OGCG:         call void @__kmpc_barrier(

void nowait_single_then_other_work_in_parallel(void) {
#pragma omp parallel
  {
#pragma omp single nowait
    work();
    other_work();
  }
}

// CIR-LABEL: define {{.*}}void @nowait_single_then_other_work_in_parallel()
// CIR:         call {{.*}}@__kmpc_fork_call(
// CIR-LABEL: define {{.*}}void @nowait_single_then_other_work_in_parallel..omp_par(
// CIR:         call i32 @__kmpc_single(
// CIR-DAG:     call void @work()
// CIR-DAG:     call void @__kmpc_end_single(
// CIR-DAG:     call void @other_work()
// CIR-NOT:     @__kmpc_barrier

// OGCG-LABEL: define {{.*}}void @nowait_single_then_other_work_in_parallel()
// OGCG:         call {{.*}}@__kmpc_fork_call(
// OGCG-LABEL: define {{.*}}void @nowait_single_then_other_work_in_parallel.omp_outlined(
// OGCG:         call i32 @__kmpc_single(
// OGCG:         call void @work()
// OGCG:         call void @__kmpc_end_single(
// OGCG:         call void @other_work()
// OGCG-NOT:     @__kmpc_barrier

void single_in_parallel_with_clauses(int num_threads_value, int condition) {
#pragma omp parallel num_threads(num_threads_value) if (condition)
  {
#pragma omp single
    work();
  }
}

// CIR-LABEL: define {{.*}}void @single_in_parallel_with_clauses(
// CIR:         call void @__kmpc_push_num_threads(
// CIR:         call {{.*}}@__kmpc_fork_call_if(
// CIR-LABEL: define {{.*}}void @single_in_parallel_with_clauses..omp_par(
// CIR:         call i32 @__kmpc_single(
// CIR-DAG:     call void @work()
// CIR-DAG:     call void @__kmpc_end_single(
// CIR-DAG:     call void @__kmpc_barrier(

// OGCG-LABEL: define {{.*}}void @single_in_parallel_with_clauses(
// OGCG:         call void @__kmpc_push_num_threads(
// OGCG:         call {{.*}}@__kmpc_fork_call(
// OGCG-LABEL: define {{.*}}void @single_in_parallel_with_clauses.omp_outlined(
// OGCG:         call i32 @__kmpc_single(
// OGCG:         call void @work()
// OGCG:         call void @__kmpc_end_single(
// OGCG:         call void @__kmpc_barrier(

void two_singles_around_barrier(void) {
#pragma omp parallel
  {
#pragma omp single
    work();
#pragma omp barrier
#pragma omp single nowait
    other_work();
  }
}

// CIR-LABEL: define {{.*}}void @two_singles_around_barrier()
// CIR:         call {{.*}}@__kmpc_fork_call(
// CIR-LABEL: define {{.*}}void @two_singles_around_barrier..omp_par(
// CIR:         call i32 @__kmpc_single(
// CIR-DAG:     call void @work()
// CIR-DAG:     call void @__kmpc_end_single(
// CIR-DAG:     call void @__kmpc_barrier(
// CIR-DAG:     call void @__kmpc_barrier(
// CIR-DAG:     call i32 @__kmpc_single(
// CIR-DAG:     call void @other_work()
// CIR-DAG:     call void @__kmpc_end_single(

// OGCG-LABEL: define {{.*}}void @two_singles_around_barrier()
// OGCG:         call {{.*}}@__kmpc_fork_call(
// OGCG-LABEL: define {{.*}}void @two_singles_around_barrier.omp_outlined(
// OGCG:         call i32 @__kmpc_single(
// OGCG:         call void @work()
// OGCG:         call void @__kmpc_end_single(
// OGCG:         call void @__kmpc_barrier(
// OGCG:         call void @__kmpc_barrier(
// OGCG:         call i32 @__kmpc_single(
// OGCG:         call void @other_work()
// OGCG:         call void @__kmpc_end_single(

void single_inside_target_parallel(void) {
#pragma omp target parallel
  {
#pragma omp single
    work();
  }
}

// CIR-LABEL: define {{.*}}void @single_inside_target_parallel()
// CIR-LABEL: define {{.*}}void @__omp_offloading_{{.*}}_single_inside_target_parallel_l{{[0-9]+}}(
// CIR:         call {{.*}}@__kmpc_fork_call(
// CIR-LABEL: define {{.*}}void @__omp_offloading_{{.*}}_single_inside_target_parallel_l{{[0-9]+}}..omp_par(
// CIR:         call i32 @__kmpc_single(
// CIR-DAG:     call void @work()
// CIR-DAG:     call void @__kmpc_end_single(
// CIR-DAG:     call void @__kmpc_barrier(

// OGCG-LABEL: define {{.*}}void @single_inside_target_parallel()
// OGCG-LABEL: define {{.*}}void @__omp_offloading_{{.*}}_single_inside_target_parallel_l{{[0-9]+}}(
// OGCG:         call {{.*}}@__kmpc_fork_call(
// OGCG-LABEL: define {{.*}}void @__omp_offloading_{{.*}}_single_inside_target_parallel_l{{[0-9]+}}.omp_outlined(
// OGCG:         call i32 @__kmpc_single(
// OGCG:         call void @work()
// OGCG:         call void @__kmpc_end_single(
// OGCG:         call void @__kmpc_barrier(
