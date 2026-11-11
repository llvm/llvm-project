// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -O2 -emit-cir %s -o %t-O2.cir
// RUN: FileCheck --input-file=%t-O2.cir %s --check-prefix=CIR-O2
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -O2 -disable-llvm-passes -emit-llvm %s -o %t-cir-O2.ll
// RUN: FileCheck --input-file=%t-cir-O2.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -O2 -disable-llvm-passes -emit-llvm %s -o %t-O2.ll
// RUN: FileCheck --input-file=%t-O2.ll %s --check-prefix=LLVM

void before(void);
void during(int);
void after(void);

// LLVM: @.gomp_critical_user_.var = {{.*}}[8 x i32] zeroinitializer

void emit_critical(int n) {
  // CIR-LABEL: cir.func{{.*}}@emit_critical
  // CIR-O2-LABEL: cir.func{{.*}}@emit_critical
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
  // CIR-O2:      omp.critical {
  // CIR-O2-NEXT:   %[[I:.*]] = cir.alloca "i"
  // CIR-O2-NEXT:   cir.lifetime.start %[[I]]
  // CIR-O2-NEXT:   cir.cleanup.scope {
  // CIR-O2:          cir.call @during(
  // CIR-O2:        } cleanup normal {
  // CIR-O2-NEXT:     cir.lifetime.end %[[I]]
  // CIR-O2-NEXT:     cir.yield
  // CIR-O2-NEXT:   }
  // CIR-O2-NEXT:   omp.terminator
  // CIR-O2-NEXT: }
  // CIR-O2-NEXT: cir.call @after()
  // LLVM: call void @before()
  // LLVM: call void @__kmpc_critical(ptr @{{.*}}, i32 %{{.*}}, ptr @.gomp_critical_user_.var)
  // LLVM: call void @during(
  // LLVM: call void @__kmpc_end_critical(ptr @{{.*}}, i32 %{{.*}}, ptr @.gomp_critical_user_.var)
  // LLVM: call void @after()
  after();
}
