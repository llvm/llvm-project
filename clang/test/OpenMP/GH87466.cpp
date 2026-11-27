// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics

// The temp arrays of an inscan reduction on a worksharing loop not combined
// with 'parallel' must be shared by the team, not allocated by every thread.

int omp_get_num_threads();
int thread_data[64];
int scan[64];
int gsum;

// CHECK-LABEL: define {{.*}}void @_Z21exclusive_in_parallelv(
// CHECK: call void {{.*}}@__kmpc_fork_call(ptr @{{.+}}, i32 {{[0-9]+}}, ptr [[OUTLINED:@[^,]+]],
void exclusive_in_parallel() {
  int curr_scan = 0;
#pragma omp parallel shared(thread_data, scan, curr_scan)
  {
    int n = omp_get_num_threads();
#pragma omp for reduction(inscan, + : curr_scan)
    for (int i = 0; i < n; ++i) {
      scan[i] = curr_scan;
#pragma omp scan exclusive(curr_scan)
      curr_scan += thread_data[i];
    }
  }
}

// CHECK: define internal void [[OUTLINED]](
// CHECK-NOT: alloca i32, i64
// CHECK: call i32 @__kmpc_single(
// CHECK: [[BUF:%.+]] = call ptr @__kmpc_aligned_alloc(i32 %{{.+}}, i64 16, i64 %{{.+}}, ptr null)
// CHECK: store ptr [[BUF]], ptr [[BUF_ADDR:%.+]],
// CHECK: call void @__kmpc_end_single(
// CHECK: call void @__kmpc_copyprivate(ptr @{{.+}}, i32 %{{.+}}, i64 8, ptr %{{.+}}, ptr @.omp.copyprivate.copy_func{{.*}}, i32 %{{.+}})
// CHECK: [[BUF_PTR:%.+]] = load ptr, ptr [[BUF_ADDR]],
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: getelementptr inbounds nuw i32, ptr [[BUF_PTR]], i64 %
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: call void @__kmpc_barrier(
// CHECK: call i32 @__kmpc_master(
// CHECK: call double @llvm.log2.f64(
// CHECK: [[LAST:%.+]] = sub nsw i64 %{{.+}}, 1
// CHECK: [[LAST_REF:%.+]] = getelementptr inbounds nuw i32, ptr [[BUF_PTR]], i64 [[LAST]]
// CHECK: [[LAST_VAL:%.+]] = load i32, ptr [[LAST_REF]],
// CHECK: store i32 [[LAST_VAL]], ptr %
// CHECK: call void @__kmpc_end_master(
// CHECK: call void @__kmpc_barrier(
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: getelementptr inbounds nuw i32, ptr [[BUF_PTR]], i64 %
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: call void @__kmpc_barrier(
// CHECK: call void @__kmpc_free(i32 %{{.+}}, ptr [[BUF_PTR]], ptr null)
// CHECK-NOT: call void @__kmpc_barrier(
// CHECK: ret void

// CHECK-LABEL: define {{.*}}void @_Z21inclusive_in_parallelv(
// CHECK: call void {{.*}}@__kmpc_fork_call(ptr @{{.+}}, i32 {{[0-9]+}}, ptr [[OUTLINED_INCL:@[^,]+]],
void inclusive_in_parallel() {
  int curr_scan = 0;
#pragma omp parallel
  {
    int n = omp_get_num_threads();
#pragma omp for reduction(inscan, + : curr_scan)
    for (int i = 0; i < n; ++i) {
      curr_scan += thread_data[i];
#pragma omp scan inclusive(curr_scan)
      scan[i] = curr_scan;
    }
  }
}

// CHECK: define internal void [[OUTLINED_INCL]](
// CHECK-NOT: alloca i32, i64
// CHECK: call i32 @__kmpc_single(
// CHECK: call ptr @__kmpc_aligned_alloc(
// CHECK: call void @__kmpc_end_single(
// CHECK: call void @__kmpc_copyprivate(
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: call void @__kmpc_barrier(
// CHECK: call i32 @__kmpc_master(
// CHECK: call void @__kmpc_end_master(
// CHECK: call void @__kmpc_barrier(
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: call void @__kmpc_barrier(
// CHECK: call void @__kmpc_free(
// CHECK: ret void

// CHECK-LABEL: define {{.*}}void @_Z8orphanedi(
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(
// CHECK-NOT: alloca i32, i64
// CHECK: call i32 @__kmpc_single(ptr @{{.+}}, i32 [[GTID]])
// CHECK: call ptr @__kmpc_aligned_alloc(i32 [[GTID]], i64 16, i64 %{{.+}}, ptr null)
// CHECK: call void @__kmpc_end_single(ptr @{{.+}}, i32 [[GTID]])
// CHECK: call void @__kmpc_copyprivate(ptr @{{.+}}, i32 [[GTID]], i64 8, ptr %{{.+}}, ptr @.omp.copyprivate.copy_func{{.*}}, i32 %{{.+}})
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call void @__kmpc_barrier(ptr @{{.+}}, i32 [[GTID]])
// CHECK: call i32 @__kmpc_master(ptr @{{.+}}, i32 [[GTID]])
// CHECK: store i32 %{{.+}}, ptr @gsum,
// CHECK: call void @__kmpc_end_master(ptr @{{.+}}, i32 [[GTID]])
// CHECK: call void @__kmpc_barrier(ptr @{{.+}}, i32 [[GTID]])
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call void @__kmpc_barrier(ptr @{{.+}}, i32 [[GTID]])
// CHECK: call void @__kmpc_free(i32 [[GTID]], ptr %{{.+}}, ptr null)
// CHECK: ret void
void orphaned(int n) {
#pragma omp for reduction(inscan, + : gsum)
  for (int i = 0; i < n; ++i) {
    gsum += thread_data[i];
#pragma omp scan inclusive(gsum)
    scan[i] = gsum;
  }
}

// CHECK-LABEL: define {{.*}}void @_Z20for_simd_in_parallelv(
// CHECK: call void {{.*}}@__kmpc_fork_call(ptr @{{.+}}, i32 {{[0-9]+}}, ptr [[OUTLINED_SIMD:@[^,]+]],
void for_simd_in_parallel() {
  int curr_scan = 0;
#pragma omp parallel
  {
    int n = omp_get_num_threads();
#pragma omp for simd reduction(inscan, + : curr_scan)
    for (int i = 0; i < n; ++i) {
      scan[i] = curr_scan;
#pragma omp scan exclusive(curr_scan)
      curr_scan += thread_data[i];
    }
  }
}

// CHECK: define internal void [[OUTLINED_SIMD]](
// CHECK-NOT: alloca i32, i64
// CHECK: call i32 @__kmpc_single(
// CHECK: call ptr @__kmpc_aligned_alloc(
// CHECK: call void @__kmpc_copyprivate(
// CHECK: call i32 @__kmpc_master(
// CHECK: call void @__kmpc_end_master(
// CHECK: call void @__kmpc_barrier(
// CHECK: call void @__kmpc_free(
// CHECK: ret void

// Combined directive: the buffer is allocated before the parallel region.
// CHECK-LABEL: define {{.*}}void @_Z8combinedi(
// CHECK: [[BUF:%.+]] = alloca i32, i64 %
// CHECK: call void {{.*}}@__kmpc_fork_call(ptr @{{.+}}, i32 {{[0-9]+}}, ptr [[OUTLINED_COMB:@[^,]+]],
// CHECK: getelementptr inbounds nuw i32, ptr [[BUF]], i64 %
// CHECK: store i32 %{{.+}}, ptr %curr_scan,
// CHECK: ret void
void combined(int n) {
  int curr_scan = 0;
#pragma omp parallel for reduction(inscan, + : curr_scan)
  for (int i = 0; i < n; ++i) {
    scan[i] = curr_scan;
#pragma omp scan exclusive(curr_scan)
    curr_scan += thread_data[i];
  }
}

// CHECK: define internal void [[OUTLINED_COMB]](
// CHECK-NOT: call i32 @__kmpc_single(
// CHECK-NOT: call ptr @__kmpc_aligned_alloc(
// CHECK-NOT: call void @__kmpc_copyprivate(
// CHECK: call i32 @__kmpc_master(
// CHECK-NOT: call void @__kmpc_free(
// CHECK: ret void

// A cancellation seen at the barrier after the input loop skips the prefix and
// scan phases; the buffer is released after a barrier every thread reaches.
// CHECK-LABEL: define {{.*}}void @_Z18cancel_in_paralleli(
// CHECK: call void {{.*}}@__kmpc_fork_call(ptr @{{.+}}, i32 {{[0-9]+}}, ptr [[OUTLINED_CANCEL:@[^,]+]],
void cancel_in_parallel(int n) {
  int curr_scan = 0;
#pragma omp parallel
  {
#pragma omp for reduction(inscan, + : curr_scan)
    for (int i = 0; i < n; ++i) {
      curr_scan += thread_data[i];
      if (thread_data[i] < 0) {
#pragma omp cancel for
      }
#pragma omp scan inclusive(curr_scan)
      scan[i] = curr_scan;
    }
  }
}

// CHECK: define internal void [[OUTLINED_CANCEL]](
// CHECK: call i32 @__kmpc_single(
// CHECK: call void @__kmpc_copyprivate(
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call i32 @__kmpc_cancel(ptr @{{.+}}, i32 %{{.+}}, i32 2)
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: [[RES:%.+]] = call i32 @__kmpc_cancel_barrier(
// CHECK: [[CANCELLED:%.+]] = icmp ne i32 [[RES]], 0
// CHECK: br i1 [[CANCELLED]], label %[[CANCEL_EXIT:[^,]+]], label %[[CONT:[^,]+]]
// CHECK: [[CANCEL_EXIT]]:
// CHECK: br label %[[SCAN_EXIT:[^,]+]]
// CHECK: [[CONT]]:
// CHECK: call i32 @__kmpc_master(
// CHECK: call void @__kmpc_end_master(
// CHECK: call void @__kmpc_barrier(
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: [[SCAN_EXIT]]:
// CHECK: call i32 @__kmpc_cancel_barrier(
// CHECK: call void @__kmpc_free(
// CHECK: ret void
