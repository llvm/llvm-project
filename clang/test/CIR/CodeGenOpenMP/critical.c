// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

void work(void);

void unnamed(void) {
#pragma omp critical
  work();
}

void local_var(int *p) {
#pragma omp critical
  {
    int tmp = *p;
    *p = tmp + 1;
  }
}

// CIR-LABEL: cir.func {{.*}}@unnamed()
// CIR:         omp.critical {
// CIR:           cir.call @work() : () -> ()
// CIR:           omp.terminator
// CIR:         }
// CIR:         cir.return

// CIR-LABEL: cir.func {{.*}}@local_var(
// CIR:         omp.critical {
// CIR:           cir.alloca "tmp"
// CIR:           omp.terminator
// CIR:         }

// LLVM-DAG: @[[UNNAMED:.gomp_critical_user_.var]] = {{.*}}global [8 x i32] zeroinitializer

// LLVM-LABEL: define {{.*}}void @unnamed()
// LLVM:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// LLVM:         call void @work()
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])

// LLVM-LABEL: define {{.*}}void @local_var(
// LLVM:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])

// OGCG-DAG: @[[UNNAMED:.gomp_critical_user_.var]] = {{.*}}global [8 x i32] zeroinitializer

// OGCG-LABEL: define {{.*}}void @unnamed()
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// OGCG:         call void @work()
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])

// OGCG-LABEL: define {{.*}}void @local_var(
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
