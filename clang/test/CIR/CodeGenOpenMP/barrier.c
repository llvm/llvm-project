// RUN: %clang_cc1 -fopenmp -triple x86_64-linux-gnu -emit-cir -fclangir %s -o - | FileCheck %s

void before(void);
void after(void);

void emit_simple_barrier() {
  // CHECK: cir.func{{.*}}@emit_simple_barrier
  before();
  // CHECK-NEXT: cir.call @before
#pragma omp barrier
  // CHECK-NEXT: omp.barrier
  after();
  // CHECK-NEXT: cir.call @after
}
