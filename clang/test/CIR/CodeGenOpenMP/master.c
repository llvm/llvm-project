
// RUN: %clang_cc1 -fopenmp -triple x86_64-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR

void before(void);
void body(void);
void after(void);

// CIR-LABEL: cir.func{{.*}}@simple_master(
void simple_master(void) {
  // CIR: cir.call @before
  before();

  // CIR-NEXT: omp.master {
#pragma omp master
  body();
  // CIR-NEXT: cir.call @body
  // CIR-NEXT: omp.terminator
  // CIR-NEXT: }

  // CIR-NEXT: cir.call @after
  after();
}
