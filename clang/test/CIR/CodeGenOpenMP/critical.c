// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

void before(void);
void during(int);
void after(void);

// LLVM: @.gomp_critical_user_.var = {{.*}}[8 x i32] zeroinitializer

void emit_critical(int n) {
  // CIR-LABEL: cir.func{{.*}}@emit_critical
  // LLVM-LABEL: define{{.*}}@emit_critical
  before();
  // CIR: cir.call @before()
#pragma omp critical
  {
    int i = n;
    during(i);
  }
  // CIR-NEXT: omp.critical {
  // CIR-NEXT:   %[[I:.*]] = cir.alloca "i"
  // CIR:        cir.call @during(
  // CIR-NEXT:   omp.terminator
  // CIR-NEXT: }
  // CIR-NEXT: cir.call @after()
  // LLVM: call void @before()
  // LLVM: call void @__kmpc_critical(ptr @{{.*}}, i32 %{{.*}}, ptr @.gomp_critical_user_.var)
  // LLVM: call void @during(
  // LLVM: call void @__kmpc_end_critical(ptr @{{.*}}, i32 %{{.*}}, ptr @.gomp_critical_user_.var)
  // LLVM: call void @after()
  after();
}
