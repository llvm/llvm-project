// Check that 'target loop' is lowered like 'target simd'.

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c++ \
// RUN:   -triple x86_64-unknown-linux-gnu \
// RUN:   -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=HOST

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c++ \
// RUN:   -triple x86_64-unknown-linux-gnu \
// RUN:   -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c++ \
// RUN:   -triple nvptx64-nvidia-cuda -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-host.bc -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=DEV

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=60 -x c++ \
// RUN:   -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=SIMD \
// RUN:     --implicit-check-not="{{__kmpc|__tgt|parallel_accesses|vectorize}}"

// expected-no-diagnostics

// One team, one thread, SPMD, with a vectorizable loop.
// HOST-LABEL: define {{.*}}void @_Z2tlPii(
// HOST: call i32 @__tgt_target_kernel({{.*}}, i64 -1, i32 1, i32 1,
// HOST-SAME: ptr @.{{.*}}_Z2tlPii_l{{[0-9]+}}.region_id,
// HOST: define internal void @{{.*}}_Z2tlPii_l{{[0-9]+}}(
// HOST: br label %omp.inner.for.cond, !llvm.loop ![[HLOOP1:[0-9]+]]

// DEV: @{{.*}}_Z2tlPii_l{{[0-9]+}}_kernel_environment =
// DEV-SAME: { i8 0, i8 1, i8 2, i32 1, i32 1, i32 1, i32 1, i32 0 }
// DEV: @{{.*}}_Z7tl_bindPii_l{{[0-9]+}}_kernel_environment =
// DEV-SAME: { i8 0, i8 1, i8 2, i32 1, i32 1, i32 1, i32 1, i32 0 }
// DEV: define {{.*}}void @{{.*}}_Z2tlPii_l{{[0-9]+}}(
// DEV: br label %omp.inner.for.cond, !llvm.loop ![[DLOOP1:[0-9]+]]

// 'loop' is not a simd construct; -fopenmp-simd emits a plain loop.
// SIMD-LABEL: define {{.*}}void @_Z2tlPii(
// SIMD: br label %for.cond, !llvm.loop
void tl(int *a, int n) {
#pragma omp target loop map(tofrom: a[0:n])
  for (int i = 0; i < n; ++i)
    a[i] += 1;
}

// bind(parallel) binds to the implicit parallel region of 'target'.
// HOST-LABEL: define {{.*}}void @_Z7tl_bindPii(
// HOST: call i32 @__tgt_target_kernel({{.*}}, i64 -1, i32 1, i32 1,
// HOST-SAME: ptr @.{{.*}}_Z7tl_bindPii_l{{[0-9]+}}.region_id,
// HOST: define internal void @{{.*}}_Z7tl_bindPii_l{{[0-9]+}}(
// HOST: br label %omp.inner.for.cond, !llvm.loop ![[HLOOP2:[0-9]+]]

// DEV: define {{.*}}void @{{.*}}_Z7tl_bindPii_l{{[0-9]+}}(
// DEV: br label %omp.inner.for.cond, !llvm.loop ![[DLOOP2:[0-9]+]]

// SIMD-LABEL: define {{.*}}void @_Z7tl_bindPii(
// SIMD: br label %for.cond, !llvm.loop
void tl_bind(int *a, int n) {
#pragma omp target loop bind(parallel) map(tofrom: a[0:n])
  for (int i = 0; i < n; ++i)
    a[i] += 1;
}

// A nested loop transformation: the target body is a directive, not a loop.
// HOST-LABEL: define {{.*}}void @_Z7tl_tilePii(
// HOST: call i32 @__tgt_target_kernel({{.*}}, i64 -1, i32 1, i32 1,
// HOST-SAME: ptr @.{{.*}}_Z7tl_tilePii_l{{[0-9]+}}.region_id,

// DEV: define {{.*}}void @{{.*}}_Z7tl_tilePii_l{{[0-9]+}}(

// SIMD-LABEL: define {{.*}}void @_Z7tl_tilePii(
void tl_tile(int *a, int n) {
#pragma omp target loop map(tofrom: a[0:n])
#pragma omp tile sizes(4)
  for (int i = 0; i < n; ++i)
    a[i] += 1;
}

// 'if' applies only to 'target', not to the loop.
// HOST-LABEL: define {{.*}}void @_Z5tl_ifPiii(
// HOST: br i1 %{{.*}}, label %omp_if.then, label %omp_if.else
// HOST: define internal void @{{.*}}_Z5tl_ifPiii_l{{[0-9]+}}(
// HOST-NOT: omp_if
// HOST: br label %omp.inner.for.cond, !llvm.loop
// HOST-NOT: omp_if
// HOST: ret void

// DEV: define {{.*}}void @{{.*}}_Z5tl_ifPiii_l{{[0-9]+}}(
// DEV-NOT: omp_if
// DEV: br label %omp.inner.for.cond, !llvm.loop
// DEV-NOT: omp_if
// DEV: ret void

// SIMD-LABEL: define {{.*}}void @_Z5tl_ifPiii(
void tl_if(int *a, int m, int n) {
#pragma omp target loop if(n > 10) map(tofrom: a[0:m])
  for (int i = 0; i < m; ++i)
    a[i] += 1;
}

// -fopenmp-simd must still privatize 'x'.
// SIMD-LABEL: define {{.*}}void @_Z10tl_privatePii(
// SIMD: %x = alloca i32
// SIMD: %[[XP:x[0-9]+]] = alloca i32
// SIMD-NOT: store i32 {{.*}}, ptr %x,
// SIMD: store i32 {{.*}}, ptr %[[XP]],
void tl_private(int *a, int n) {
  int x;
#pragma omp target loop private(x) map(tofrom: a[0:n])
  for (int i = 0; i < n; ++i) {
    x = i;
    a[i] += x;
  }
}

// HOST: ![[HLOOP1]] = distinct !{![[HLOOP1]], ![[HPA1:[0-9]+]],
// HOST-SAME: ![[VEC:[0-9]+]]}
// HOST: ![[HPA1]] = !{!"llvm.loop.parallel_accesses", !{{[0-9]+}}}
// HOST: ![[VEC]] = !{!"llvm.loop.vectorize.enable"}
// HOST: ![[HLOOP2]] = distinct !{![[HLOOP2]], ![[HPA2:[0-9]+]], ![[VEC]]}
// HOST: ![[HPA2]] = !{!"llvm.loop.parallel_accesses", !{{[0-9]+}}}

// DEV: ![[DLOOP1]] = distinct !{![[DLOOP1]], ![[DPA1:[0-9]+]],
// DEV-SAME: ![[DVEC:[0-9]+]]}
// DEV: ![[DPA1]] = !{!"llvm.loop.parallel_accesses", !{{[0-9]+}}}
// DEV: ![[DVEC]] = !{!"llvm.loop.vectorize.enable"}
// DEV: ![[DLOOP2]] = distinct !{![[DLOOP2]], ![[DPA2:[0-9]+]], ![[DVEC]]}
// DEV: ![[DPA2]] = !{!"llvm.loop.parallel_accesses", !{{[0-9]+}}}
