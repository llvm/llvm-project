// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp \
// RUN:   -debug-info-kind=line-tables-only -emit-llvm -o - %s | FileCheck %s

// The !llvm.loop start/end locations of OpenMP loops must be the associated
// 'for' statement, not the directive.

// CHECK-LABEL: define internal void @static_sched.omp_outlined(
// CHECK: br label %omp.inner.for.cond, {{.*}}!llvm.loop ![[STATIC:[0-9]+]]

// CHECK-LABEL: define internal void @dynamic_sched.omp_outlined(
// CHECK: br label %omp.inner.for.cond, {{.*}}!llvm.loop ![[DYN_INNER:[0-9]+]]
// CHECK: br label %omp.dispatch.cond, {{.*}}!llvm.loop ![[DYN_OUTER:[0-9]+]]

// CHECK-LABEL: define dso_local void @unroll_partial(
// CHECK: br label %omp.inner.for.cond, {{.*}}!llvm.loop ![[UNROLL:[0-9]+]]

// CHECK-LABEL: define dso_local void @simd_loop(
// CHECK: br label %omp.inner.for.cond, {{.*}}!llvm.loop ![[SIMD:[0-9]+]]

// CHECK-LABEL: define dso_local void @unroll_tile(
// CHECK: br label %omp.inner.for.cond, {{.*}}!llvm.loop ![[STACKED:[0-9]+]]

// CHECK-DAG: ![[STATIC]] = distinct !{![[STATIC]], ![[L101:[0-9]+]], ![[L103:[0-9]+]]
// CHECK-DAG: ![[L101]] = !DILocation(line: 101, column: 3,
// CHECK-DAG: ![[L103]] = !DILocation(line: 103, column: 3,
// CHECK-DAG: ![[DYN_INNER]] = distinct !{![[DYN_INNER]], ![[L201:[0-9]+]], ![[L203:[0-9]+]]
// CHECK-DAG: ![[DYN_OUTER]] = distinct !{![[DYN_OUTER]], ![[L201]], ![[L203]]
// CHECK-DAG: ![[L201]] = !DILocation(line: 201, column: 3,
// CHECK-DAG: ![[L203]] = !DILocation(line: 203, column: 3,
// CHECK-DAG: ![[UNROLL]] = distinct !{![[UNROLL]], ![[L302:[0-9]+]], ![[L304:[0-9]+]]
// CHECK-DAG: ![[L302]] = !DILocation(line: 302, column: 3,
// CHECK-DAG: ![[L304]] = !DILocation(line: 304, column: 3,
// CHECK-DAG: ![[SIMD]] = distinct !{![[SIMD]], ![[L401:[0-9]+]], ![[L403:[0-9]+]]
// CHECK-DAG: ![[L401]] = !DILocation(line: 401, column: 3,
// CHECK-DAG: ![[L403]] = !DILocation(line: 403, column: 3,
// CHECK-DAG: ![[STACKED]] = distinct !{![[STACKED]], ![[L503:[0-9]+]], ![[L505:[0-9]+]]
// CHECK-DAG: ![[L503]] = !DILocation(line: 503, column: 3,
// CHECK-DAG: ![[L505]] = !DILocation(line: 505, column: 3,

void static_sched(int *a, int n) {
#line 100
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    a[i]++;
  }
}

void dynamic_sched(int *a, int n) {
#line 200
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < n; ++i) {
    a[i]++;
  }
}

void unroll_partial(int *a, int n) {
#line 300
#pragma omp for
#pragma omp unroll partial(2)
  for (int i = 0; i < n; ++i) {
    a[i]++;
  }
}

void simd_loop(int *a, int n) {
#line 400
#pragma omp simd
  for (int i = 0; i < n; ++i) {
    a[i]++;
  }
}

void unroll_tile(int *a, int n) {
#line 500
#pragma omp for
#pragma omp unroll partial(2)
#pragma omp tile sizes(4)
  for (int i = 0; i < n; ++i) {
    a[i]++;
  }
}
