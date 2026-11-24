// The memory ordering of an 'omp atomic' construct is carried by the atomic
// operation itself.  No flush is emitted for it, except for the implicit flush
// that OpenMP 4.5 and earlier ask for on a seq_cst atomic construct.

// RUN: %clang_cc1 -verify -triple x86_64-unknown-linux-gnu -fopenmp -fopenmp-version=45 -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,OMP45
// RUN: %clang_cc1 -verify -triple x86_64-unknown-linux-gnu -fopenmp -fopenmp-version=50 -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,OMP50
// RUN: %clang_cc1 -verify -triple x86_64-unknown-linux-gnu -fopenmp -fopenmp-version=51 -DOMP51 -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,OMP50,OMP51
// RUN: %clang_cc1 -verify -triple x86_64-unknown-linux-gnu -fopenmp -fopenmp-version=60 -DOMP51 -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,OMP50,OMP51
// expected-no-diagnostics

int x, v;
long double ld;

// CHECK-LABEL: define {{.*}}@seq_cst_ops(
void seq_cst_ops(int e) {
// CHECK: load atomic i32, ptr @x seq_cst
// OMP45: call void @__kmpc_flush(
// OMP50-NOT: call void @__kmpc_flush
#pragma omp atomic read seq_cst
  v = x;
// CHECK: store atomic i32 %{{.+}}, ptr @x seq_cst
// OMP45: call void @__kmpc_flush(
// OMP50-NOT: call void @__kmpc_flush
#pragma omp atomic write seq_cst
  x = e;
// CHECK: atomicrmw add ptr @x, i32 %{{.+}} seq_cst
// OMP45: call void @__kmpc_flush(
// OMP50-NOT: call void @__kmpc_flush
#pragma omp atomic update seq_cst
  x += e;
// CHECK: atomicrmw add ptr @x, i32 %{{.+}} seq_cst
// OMP45: call void @__kmpc_flush(
// OMP50-NOT: call void @__kmpc_flush
#pragma omp atomic capture seq_cst
  v = x += e;
// An operation that goes through a library call.
// CHECK: call void @__atomic_store(i64 noundef 16, ptr noundef @ld, ptr noundef %{{.+}}, i32 noundef 5)
// OMP45: call void @__kmpc_flush(
// OMP50-NOT: call void @__kmpc_flush
#pragma omp atomic write seq_cst
  ld = e;
// CHECK: ret void
}

// Without a memory-order clause no flush is emitted in any version.
// CHECK-LABEL: define {{.*}}@relaxed_ops(
// CHECK-NOT: call void @__kmpc_flush
// CHECK: ret void
void relaxed_ops(int e) {
#pragma omp atomic read
  v = x;
#pragma omp atomic write
  x = e;
#pragma omp atomic
  x += e;
#pragma omp atomic capture
  v = x += e;
}

#ifdef OMP51
// OMP51-LABEL: define {{.*}}@ordered_ops(
// OMP51-NOT: call void @__kmpc_flush
void ordered_ops(int e, int d) {
// OMP51: load atomic i32, ptr @x acquire
#pragma omp atomic read acquire
  v = x;
// OMP51: store atomic i32 %{{.+}}, ptr @x release
#pragma omp atomic write release
  x = e;
// OMP51: atomicrmw add ptr @x, i32 %{{.+}} release
#pragma omp atomic update release
  x += e;
// OMP51: atomicrmw add ptr @x, i32 %{{.+}} acq_rel
#pragma omp atomic capture acq_rel
  v = x += e;
// OMP51: cmpxchg ptr @x, i32 %{{.+}}, i32 %{{.+}} acq_rel acquire
#pragma omp atomic compare acq_rel
  if (x == e)
    x = d;
// OMP51: cmpxchg ptr @x, i32 %{{.+}}, i32 %{{.+}} seq_cst seq_cst
#pragma omp atomic compare seq_cst
  if (x == e)
    x = d;
// OMP51: ret void
}
#endif
