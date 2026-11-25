// RUN: %clang_cc1 -fopenmp -triple x86_64-linux-gnu -emit-cir -fclangir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

// Check the CIR of omp single, and the LLVM IR from both ClangIR and classic
// codegen. The prefixes are CIR (ClangIR output) and LLVM (LLVM IR checks
// shared by both paths).

void before(void);
void during(void);
void after(void);

void emit_simple_single() {
  // CIR: cir.func{{.*}}@emit_simple_single
  before();
  // CIR-NEXT: cir.call @before
#pragma omp single
  { during(); }
  // CIR-NEXT: omp.single {
  // CIR-NEXT: cir.call @during
  // CIR-NEXT: omp.terminator
  // CIR-NEXT: }
  after();
  // CIR-NEXT: cir.call @after
}

void nowait_single() {
  // CIR: cir.func{{.*}}@nowait_single
#pragma omp single nowait
  {}
  // CIR-NEXT: omp.single nowait {
  // CIR-NEXT: omp.terminator
  // CIR-NEXT: }
}

void single_with_local_variable(int *ptr) {
  // CIR: cir.func{{.*}}@single_with_local_variable
#pragma omp single
  {
    int local = *ptr;
    *ptr = local + 1;
  }
  // CIR: omp.single {
  // CIR-NEXT: cir.alloca {{.*}}"local"
  // CIR: omp.terminator
  // CIR-NEXT: }
}

void single_inside_parallel(void) {
  // CIR: cir.func{{.*}}@single_inside_parallel
#pragma omp parallel
  {
#pragma omp single
    { during(); }
  }
  // CIR-NEXT: omp.parallel {
  // CIR-NEXT: omp.single {
  // CIR-NEXT: cir.call @during
  // CIR-NEXT: omp.terminator
  // CIR-NEXT: }
  // CIR-NEXT: omp.terminator
  // CIR-NEXT: }
}

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

void single_in_parallel_region(void) {
#pragma omp parallel
  {
#pragma omp single
    work();
  }
}

// LLVM-LABEL: define {{.*}}void @single_in_parallel_region()
// LLVM:         call {{.*}}@__kmpc_fork_call{{.*}}(
// The outlined function name differs between the two paths (ClangIR:
// "<fn>..omp_par", classic codegen: "<fn>.omp_outlined"), so the dot count
// and suffix are matched with a regex.
// LLVM-LABEL: define {{.*}}void @single_in_parallel_region{{\.+}}omp_{{(par|outlined)}}(
// LLVM:         call i32 @__kmpc_single(
// LLVM-DAG:     call void @work()
// LLVM-DAG:     call void @__kmpc_end_single(
// LLVM-DAG:     call void @__kmpc_barrier(

void nowait_single_then_other_work_in_parallel(void) {
#pragma omp parallel
  {
#pragma omp single nowait
    work();
    other_work();
  }
}

// LLVM-LABEL: define {{.*}}void @nowait_single_then_other_work_in_parallel()
// LLVM:         call {{.*}}@__kmpc_fork_call{{.*}}(
// LLVM-LABEL: define {{.*}}void @nowait_single_then_other_work_in_parallel{{\.+}}omp_{{(par|outlined)}}(
// LLVM:         call i32 @__kmpc_single(
// LLVM-DAG:     call void @work()
// LLVM-DAG:     call void @__kmpc_end_single(
// LLVM-DAG:     call void @other_work()
// LLVM-NOT:     @__kmpc_barrier

void single_in_parallel_with_clauses(int num_threads_value, int condition) {
#pragma omp parallel num_threads(num_threads_value) if (condition)
  {
#pragma omp single
    work();
  }
}

// LLVM-LABEL: define {{.*}}void @single_in_parallel_with_clauses(
// LLVM:         call void @__kmpc_push_num_threads(
// LLVM:         call {{.*}}@__kmpc_fork_call{{.*}}(
// LLVM-LABEL: define {{.*}}void @single_in_parallel_with_clauses{{\.+}}omp_{{(par|outlined)}}(
// LLVM:         call i32 @__kmpc_single(
// LLVM-DAG:     call void @work()
// LLVM-DAG:     call void @__kmpc_end_single(
// LLVM-DAG:     call void @__kmpc_barrier(

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

// LLVM-LABEL: define {{.*}}void @two_singles_around_barrier()
// LLVM:         call {{.*}}@__kmpc_fork_call{{.*}}(
// LLVM-LABEL: define {{.*}}void @two_singles_around_barrier{{\.+}}omp_{{(par|outlined)}}(
// LLVM:         call i32 @__kmpc_single(
// LLVM-DAG:     call void @work()
// LLVM-DAG:     call void @__kmpc_end_single(
// LLVM-DAG:     call void @__kmpc_barrier(
// LLVM-DAG:     call void @__kmpc_barrier(
// LLVM-DAG:     call i32 @__kmpc_single(
// LLVM-DAG:     call void @other_work()
// LLVM:         call void @__kmpc_end_single(

void single_inside_target_parallel(void) {
#pragma omp target parallel
  {
#pragma omp single
    work();
  }
}

// LLVM-LABEL: define {{.*}}void @single_inside_target_parallel()
// LLVM-LABEL: define {{.*}}void @__omp_offloading_{{.*}}_single_inside_target_parallel_l{{[0-9]+}}(
// LLVM:         call {{.*}}@__kmpc_fork_call{{.*}}(
// LLVM-LABEL: define {{.*}}void @__omp_offloading_{{.*}}_single_inside_target_parallel_l{{[0-9]+}}{{\.+}}omp_{{(par|outlined)}}(
// LLVM:         call i32 @__kmpc_single(
// LLVM-DAG:     call void @work()
// LLVM-DAG:     call void @__kmpc_end_single(
// LLVM-DAG:     call void @__kmpc_barrier(
