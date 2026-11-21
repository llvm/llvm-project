// Check that an empty worksharing loop assigns the for-init value to a
// lastprivate loop iteration variable, and changes nothing else.
//
// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fopenmp -emit-llvm %s -o - | FileCheck %s

// expected-no-diagnostics

// Runtime trip count: the empty path stores the init to the original i only.
// lastprivate j is not a loop iteration variable and is not assigned.
// CHECK-LABEL: define {{.*}}void @runtime_empty.omp_outlined(
// CHECK:         [[I:%.+]] = load ptr, ptr %i.addr
// CHECK:         br i1 {{.+}}, label %omp.precond.then, label %omp.precond.empty
// CHECK:       omp.precond.empty:
// CHECK-NEXT:    store i32 0, ptr [[I]]
// CHECK-NEXT:    br label %omp.precond.end
// CHECK:       omp.precond.then:
// CHECK:         call void @__kmpc_for_static_init
// CHECK:         br i1 {{.+}}, label %.omp.lastprivate.then, label %.omp.lastprivate.done
extern "C" void runtime_empty(int n) {
  int i = -1;
  int j = -1;
#pragma omp parallel for lastprivate(i, j)
  for (i = 0; i < n; ++i)
    for (j = 0; j < 3; ++j)
      ;
}

// Constant empty loop: no loop is emitted, only the init store.
// CHECK-LABEL: define {{.*}}void @constant_empty.omp_outlined(
// CHECK:         [[I:%.+]] = load ptr, ptr %i.addr
// CHECK-NEXT:    store i32 0, ptr [[I]]
// CHECK-NEXT:    ret void
extern "C" void constant_empty() {
  int i = -1;
#pragma omp parallel for lastprivate(i)
  for (i = 0; i < 0; ++i)
    ;
}

// The loop iteration variable is not lastprivate: no extra block.
// CHECK-LABEL: define {{.*}}void @counter_not_lastprivate.omp_outlined(
// CHECK:         br i1 {{.+}}, label %omp.precond.then, label %omp.precond.end
// CHECK-NOT:   omp.precond.empty
// CHECK:       omp.precond.end:
extern "C" void counter_not_lastprivate(int n) {
  int j = -1;
#pragma omp parallel for lastprivate(j)
  for (int i = 0; i < n; ++i)
    j = i;
}

// Collapsed nest: an outer variable may need its final value, not its init,
// so the empty path is not changed.
// CHECK-LABEL: define {{.*}}void @collapse_empty.omp_outlined(
// CHECK-NOT:   omp.precond.empty
// CHECK:       omp.precond.end:
extern "C" void collapse_empty(int n, int m) {
  int i = -1;
  int j = -1;
#pragma omp parallel for collapse(2) lastprivate(i, j)
  for (i = 0; i < n; ++i)
    for (j = 0; j < m; ++j)
      ;
}
