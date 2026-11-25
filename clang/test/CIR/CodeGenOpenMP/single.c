// RUN: %clang_cc1 -fopenmp -triple x86_64-linux-gnu -emit-cir -fclangir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,LLVM-CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=LLVM,LLVM-OGCG

// Prefixes:
//   CIR       - CIR dialect output.
//   LLVM      - LLVM IR checks shared by ClangIR and classic codegen.
//   LLVM-CIR  - LLVM IR checks for ClangIR only.
//   LLVM-OGCG - LLVM IR checks for classic codegen only.
//
// NOTE: LLVM-CIR and LLVM-OGCG are separate because the block order and the
// block labels differ. The control flow is the same: a body block and an end
// block, with the barrier in the end block.
//   - Order:  ClangIR prints the end block first, Classic the body first.
//   - Labels: ClangIR uses omp_region.body/end, Classic uses omp_if.then/end.
//             The checks capture them as [[BODY]] and [[END]].

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

// LLVM-LABEL:     define {{.*}}void @single_only()
// LLVM:           call i32 @__kmpc_single(
// LLVM:           call void @work()
// LLVM:           call void @__kmpc_end_single(
// LLVM:           call void @__kmpc_barrier(

void nowait_single_only(void) {
#pragma omp single nowait
  work();
}

// LLVM-LABEL:     define {{.*}}void @nowait_single_only()
// LLVM:           call i32 @__kmpc_single(
// LLVM:           call void @work()
// LLVM:           call void @__kmpc_end_single(
// LLVM-NOT:       @__kmpc_barrier

void single_in_parallel_region(void) {
#pragma omp parallel
  {
#pragma omp single
    work();
  }
}

// Flow of execution in the outlined function:
//   - The thread that wins __kmpc_single runs the body block (work, then
//     __kmpc_end_single) and then goes to the end block.
//   - The other threads go straight to the end block.
//   - The end block holds the barrier, so every thread waits there until the
//     body has finished.
// The remaining tests below follow the same flow, with their own extra calls.

// LLVM-LABEL:      define {{.*}}void @single_in_parallel_region()
// LLVM:            call {{.*}}@__kmpc_fork_call{{.*}}(

// LLVM-OGCG-LABEL: define {{.*}}void @single_in_parallel_region.omp_outlined(
// LLVM-OGCG:       call i32 @__kmpc_single(
// LLVM-OGCG:       br i1 %{{.*}}, label %[[BODY:[^,]+]], label %[[END:.+]]
// LLVM-OGCG:       {{^}}[[BODY]]:
// LLVM-OGCG-NEXT:    call void @work()
// LLVM-OGCG-NEXT:    call void @__kmpc_end_single(
// LLVM-OGCG-NEXT:    br label %[[END]]
// LLVM-OGCG:       {{^}}[[END]]:
// LLVM-OGCG-NEXT:    call void @__kmpc_barrier(

// LLVM-CIR-LABEL:  define {{.*}}void @single_in_parallel_region..omp_par(
// LLVM-CIR:        call i32 @__kmpc_single(
// LLVM-CIR:        br i1 %{{.*}}, label %[[BODY:[^,]+]], label %[[END:.+]]
// LLVM-CIR:        {{^}}[[END]]:
// LLVM-CIR:          call void @__kmpc_barrier(
// LLVM-CIR:        {{^}}[[BODY]]:
// LLVM-CIR:          call void @work()
// LLVM-CIR:          call void @__kmpc_end_single(
// LLVM-CIR:          br label %[[END]]

void nowait_single_then_other_work_in_parallel(void) {
#pragma omp parallel
  {
#pragma omp single nowait
    work();
    other_work();
  }
}

// LLVM-LABEL:      define {{.*}}void @nowait_single_then_other_work_in_parallel()
// LLVM:            call {{.*}}@__kmpc_fork_call{{.*}}(

// LLVM-OGCG-LABEL: define {{.*}}void @nowait_single_then_other_work_in_parallel.omp_outlined(
// LLVM-OGCG:       call i32 @__kmpc_single(
// LLVM-OGCG:       br i1 %{{.*}}, label %[[BODY:[^,]+]], label %[[END:.+]]
// LLVM-OGCG:       {{^}}[[BODY]]:
// LLVM-OGCG-NEXT:    call void @work()
// LLVM-OGCG-NEXT:    call void @__kmpc_end_single(
// LLVM-OGCG-NEXT:    br label %[[END]]
// LLVM-OGCG:       {{^}}[[END]]:
// LLVM-OGCG-NEXT:    call void @other_work()
// LLVM-OGCG-NOT:   @__kmpc_barrier

// LLVM-CIR-LABEL:  define {{.*}}void @nowait_single_then_other_work_in_parallel..omp_par(
// LLVM-CIR:        call i32 @__kmpc_single(
// LLVM-CIR:        br i1 %{{.*}}, label %[[BODY:[^,]+]], label %[[END:.+]]
// LLVM-CIR:        {{^}}[[END]]:
// LLVM-CIR:          call void @other_work()
// LLVM-CIR:        {{^}}[[BODY]]:
// LLVM-CIR:          call void @work()
// LLVM-CIR:          call void @__kmpc_end_single(
// LLVM-CIR:          br label %[[END]]
// LLVM-CIR-NOT:    @__kmpc_barrier

void single_in_parallel_with_clauses(int num_threads_value, int condition) {
#pragma omp parallel num_threads(num_threads_value) if (condition)
  {
#pragma omp single
    work();
  }
}

// LLVM-LABEL:      define {{.*}}void @single_in_parallel_with_clauses(
// LLVM:            call void @__kmpc_push_num_threads(
// LLVM:            call {{.*}}@__kmpc_fork_call{{.*}}(

// LLVM-OGCG-LABEL: define {{.*}}void @single_in_parallel_with_clauses.omp_outlined(
// LLVM-OGCG:       call i32 @__kmpc_single(
// LLVM-OGCG:       br i1 %{{.*}}, label %[[BODY:[^,]+]], label %[[END:.+]]
// LLVM-OGCG:       {{^}}[[BODY]]:
// LLVM-OGCG-NEXT:    call void @work()
// LLVM-OGCG-NEXT:    call void @__kmpc_end_single(
// LLVM-OGCG-NEXT:    br label %[[END]]
// LLVM-OGCG:       {{^}}[[END]]:
// LLVM-OGCG-NEXT:    call void @__kmpc_barrier(

// LLVM-CIR-LABEL:  define {{.*}}void @single_in_parallel_with_clauses..omp_par(
// LLVM-CIR:        call i32 @__kmpc_single(
// LLVM-CIR:        br i1 %{{.*}}, label %[[BODY:[^,]+]], label %[[END:.+]]
// LLVM-CIR:        {{^}}[[END]]:
// LLVM-CIR:          call void @__kmpc_barrier(
// LLVM-CIR:        {{^}}[[BODY]]:
// LLVM-CIR:          call void @work()
// LLVM-CIR:          call void @__kmpc_end_single(
// LLVM-CIR:          br label %[[END]]

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

// LLVM-LABEL:      define {{.*}}void @two_singles_around_barrier()
// LLVM:            call {{.*}}@__kmpc_fork_call{{.*}}(

// LLVM-OGCG-LABEL: define {{.*}}void @two_singles_around_barrier.omp_outlined(
// LLVM-OGCG:       call i32 @__kmpc_single(
// LLVM-OGCG:       br i1 %{{.*}}, label %[[BODY:[^,]+]], label %[[END:.+]]
// LLVM-OGCG:       {{^}}[[BODY]]:
// LLVM-OGCG-NEXT:    call void @work()
// LLVM-OGCG-NEXT:    call void @__kmpc_end_single(
// LLVM-OGCG-NEXT:    br label %[[END]]
// LLVM-OGCG:       {{^}}[[END]]:
// LLVM-OGCG-NEXT:    call void @__kmpc_barrier(
// LLVM-OGCG-NEXT:    call void @__kmpc_barrier(
// LLVM-OGCG-NEXT:    call i32 @__kmpc_single(
// LLVM-OGCG-NEXT:    icmp
// LLVM-OGCG-NEXT:    br i1 %{{.*}}, label %[[BODY2:[^,]+]], label %[[END2:.+]]
// LLVM-OGCG:       {{^}}[[BODY2]]:
// LLVM-OGCG-NEXT:    call void @other_work()
// LLVM-OGCG-NEXT:    call void @__kmpc_end_single(

// LLVM-CIR-LABEL:  define {{.*}}void @two_singles_around_barrier..omp_par(
// LLVM-CIR:        call i32 @__kmpc_single(
// LLVM-CIR:        br i1 %{{.*}}, label %[[BODY:[^,]+]], label %[[END:.+]]
// LLVM-CIR:        {{^}}[[END]]:
// LLVM-CIR:          call void @__kmpc_barrier(
// LLVM-CIR:          call void @__kmpc_barrier(
// LLVM-CIR:          call i32 @__kmpc_single(
// LLVM-CIR:          icmp
// LLVM-CIR:          br i1 %{{.*}}, label %[[BODY2:[^,]+]], label %[[END2:.+]]
// LLVM-CIR:        {{^}}[[BODY2]]:
// LLVM-CIR:          call void @other_work()
// LLVM-CIR:          call void @__kmpc_end_single(
// LLVM-CIR:        {{^}}[[BODY]]:
// LLVM-CIR:          call void @work()
// LLVM-CIR:          call void @__kmpc_end_single(
// LLVM-CIR:          br label %[[END]]

void single_inside_target_parallel(void) {
#pragma omp target parallel
  {
#pragma omp single
    work();
  }
}

// LLVM-LABEL:      define {{.*}}void @single_inside_target_parallel()

// LLVM-OGCG-LABEL: define {{.*}}void @__omp_offloading_{{.*}}_single_inside_target_parallel_l{{[0-9]+}}.omp_outlined(
// LLVM-OGCG:       call i32 @__kmpc_single(
// LLVM-OGCG:       br i1 %{{.*}}, label %[[BODY:[^,]+]], label %[[END:.+]]
// LLVM-OGCG:       {{^}}[[BODY]]:
// LLVM-OGCG-NEXT:    call void @work()
// LLVM-OGCG-NEXT:    call void @__kmpc_end_single(
// LLVM-OGCG-NEXT:    br label %[[END]]
// LLVM-OGCG:       {{^}}[[END]]:
// LLVM-OGCG-NEXT:    call void @__kmpc_barrier(

// LLVM-CIR-LABEL:  define {{.*}}void @__omp_offloading_{{.*}}_single_inside_target_parallel_l{{[0-9]+}}..omp_par(
// LLVM-CIR:        call i32 @__kmpc_single(
// LLVM-CIR:        br i1 %{{.*}}, label %[[BODY:[^,]+]], label %[[END:.+]]
// LLVM-CIR:        {{^}}[[END]]:
// LLVM-CIR:          call void @__kmpc_barrier(
// LLVM-CIR:        {{^}}[[BODY]]:
// LLVM-CIR:          call void @work()
// LLVM-CIR:          call void @__kmpc_end_single(
// LLVM-CIR:          br label %[[END]]
