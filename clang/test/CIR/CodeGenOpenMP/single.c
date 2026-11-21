// RUN: %clang_cc1 -fopenmp -triple x86_64-linux-gnu -emit-cir -fclangir %s -o - | FileCheck %s

void before(void);
void during(void);
void after(void);

void emit_simple_single() {
  // CHECK: cir.func{{.*}}@emit_simple_single
  before();
  // CHECK-NEXT: cir.call @before
#pragma omp single
  { during(); }
  // CHECK-NEXT: omp.single {
  // CHECK-NEXT: cir.call @during
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
  after();
  // CHECK-NEXT: cir.call @after
}

void nowait_single() {
  // CHECK: cir.func{{.*}}@nowait_single
#pragma omp single nowait
  {}
  // CHECK-NEXT: omp.single nowait {
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
}

void single_with_local_variable(int *ptr) {
  // CHECK: cir.func{{.*}}@single_with_local_variable
#pragma omp single
  {
    int local = *ptr;
    *ptr = local + 1;
  }
  // CHECK: omp.single {
  // CHECK-NEXT: cir.alloca {{.*}}"local"
  // CHECK: omp.terminator
  // CHECK-NEXT: }
}

void single_inside_parallel(void) {
  // CHECK: cir.func{{.*}}@single_inside_parallel
#pragma omp parallel
  {
#pragma omp single
    { during(); }
  }
  // CHECK-NEXT: omp.parallel {
  // CHECK-NEXT: omp.single {
  // CHECK-NEXT: cir.call @during
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
}
