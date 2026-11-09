// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

void work(void);
void work2(void);

void unnamed(void) {
#pragma omp critical
  work();
}

void named(void) {
#pragma omp critical(mylock)
  work();
}

// The same name in another function reuses the one declaration.
void named_again(void) {
#pragma omp critical(mylock)
  work2();
}

void no_braces(int *p) {
#pragma omp critical(mylock)
  *p += 1;
}

void local_var(int *p) {
#pragma omp critical
  {
    int tmp = *p;
    *p = tmp + 1;
  }
}

void in_parallel(int *p) {
#pragma omp parallel
  {
#pragma omp critical(mylock)
    { *p += 1; }
  }
}

void nested(void) {
#pragma omp critical(outer)
  {
    work();
#pragma omp critical(inner)
    work2();
  }
}

// Each name is declared once, at the start of the module.
// CIR-DAG: omp.critical.declare @mylock
// CIR-DAG: omp.critical.declare @outer
// CIR-DAG: omp.critical.declare @inner

// CIR-LABEL: cir.func {{.*}}@unnamed()
// CIR:         omp.critical {
// CIR:           cir.call @work() : () -> ()
// CIR:           omp.terminator
// CIR:         }
// CIR:         cir.return

// CIR-LABEL: cir.func {{.*}}@named()
// CIR:         omp.critical(@mylock) {
// CIR:           cir.call @work() : () -> ()
// CIR:           omp.terminator
// CIR:         }

// CIR-LABEL: cir.func {{.*}}@named_again()
// CIR:         omp.critical(@mylock) {
// CIR:           cir.call @work2() : () -> ()
// CIR:           omp.terminator
// CIR:         }

// CIR-LABEL: cir.func {{.*}}@no_braces(
// CIR:         omp.critical(@mylock) {
// CIR:           cir.load
// CIR:           cir.add
// CIR:           cir.store
// CIR:           omp.terminator
// CIR:         }

// CIR-LABEL: cir.func {{.*}}@local_var(
// CIR:         omp.critical {
// CIR:           cir.alloca "tmp"
// CIR:           omp.terminator
// CIR:         }

// CIR-LABEL: cir.func {{.*}}@in_parallel(
// CIR:         omp.parallel {
// CIR:           omp.critical(@mylock) {
// CIR:             omp.terminator
// CIR:           }
// CIR:           omp.terminator
// CIR:         }

// CIR-LABEL: cir.func {{.*}}@nested()
// CIR:         omp.critical(@outer) {
// CIR:           cir.call @work() : () -> ()
// CIR:           omp.critical(@inner) {
// CIR:             cir.call @work2() : () -> ()
// CIR:             omp.terminator
// CIR:           }
// CIR:           omp.terminator
// CIR:         }

// The lock variables are shared by every critical section of the same name.
// Unlike classic codegen, a named critical section always goes through
// __kmpc_critical_with_hint, with a hint of 0 when none was given.
// LLVM-DAG: @[[UNNAMED:.gomp_critical_user_.var]] = {{.*}}global [8 x i32] zeroinitializer
// LLVM-DAG: @[[MYLOCK:.gomp_critical_user_mylock.var]] = {{.*}}global [8 x i32] zeroinitializer
// LLVM-DAG: @[[OUTER:.gomp_critical_user_outer.var]] = {{.*}}global [8 x i32] zeroinitializer
// LLVM-DAG: @[[INNER:.gomp_critical_user_inner.var]] = {{.*}}global [8 x i32] zeroinitializer

// LLVM-LABEL: define {{.*}}void @unnamed()
// LLVM:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// LLVM:         call void @work()
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])

// LLVM-LABEL: define {{.*}}void @named()
// LLVM:         call void @__kmpc_critical_with_hint({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]], i32 0)
// LLVM:         call void @work()
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])

// LLVM-LABEL: define {{.*}}void @named_again()
// LLVM:         call void @__kmpc_critical_with_hint({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]], i32 0)
// LLVM:         call void @work2()
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])

// LLVM-LABEL: define {{.*}}void @no_braces(
// LLVM:         call void @__kmpc_critical_with_hint({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]], i32 0)
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])

// LLVM-LABEL: define {{.*}}void @local_var(
// LLVM:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])

// LLVM-LABEL: define {{.*}}void @in_parallel(
// LLVM:         call {{.*}}@__kmpc_fork_call(
// LLVM-LABEL: define {{.*}}void @in_parallel..omp_par(
// LLVM:         call void @__kmpc_critical_with_hint({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]], i32 0)
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])

// LLVM-LABEL: define {{.*}}void @nested()
// LLVM:         call void @__kmpc_critical_with_hint({{.*}}, i32 {{.*}}, ptr @[[OUTER]], i32 0)
// LLVM:         call void @work()
// LLVM:         call void @__kmpc_critical_with_hint({{.*}}, i32 {{.*}}, ptr @[[INNER]], i32 0)
// LLVM:         call void @work2()
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[INNER]])
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[OUTER]])

// OGCG-DAG: @[[UNNAMED:.gomp_critical_user_.var]] = {{.*}}global [8 x i32] zeroinitializer
// OGCG-DAG: @[[MYLOCK:.gomp_critical_user_mylock.var]] = {{.*}}global [8 x i32] zeroinitializer
// OGCG-DAG: @[[OUTER:.gomp_critical_user_outer.var]] = {{.*}}global [8 x i32] zeroinitializer
// OGCG-DAG: @[[INNER:.gomp_critical_user_inner.var]] = {{.*}}global [8 x i32] zeroinitializer

// OGCG-LABEL: define {{.*}}void @unnamed()
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// OGCG:         call void @work()
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])

// OGCG-LABEL: define {{.*}}void @named()
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])
// OGCG:         call void @work()
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])

// OGCG-LABEL: define {{.*}}void @named_again()
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])
// OGCG:         call void @work2()
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])

// OGCG-LABEL: define {{.*}}void @no_braces(
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])

// OGCG-LABEL: define {{.*}}void @local_var(
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])

// OGCG-LABEL: define {{.*}}void @in_parallel(
// OGCG:         call {{.*}}@__kmpc_fork_call(
// OGCG-LABEL: define {{.*}}void @in_parallel.omp_outlined(
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[MYLOCK]])

// OGCG-LABEL: define {{.*}}void @nested()
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[OUTER]])
// OGCG:         call void @work()
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[INNER]])
// OGCG:         call void @work2()
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[INNER]])
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[OUTER]])
