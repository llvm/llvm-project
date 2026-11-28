// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-linux-gnu -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-unknown-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-unknown-linux-gnu -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK-LABEL: define {{.*}}void @_Z22masked_taskloop_cancelv()
// CHECK: call i32 @__kmpc_masked(ptr [[DEFLOC:@.+]], i32 [[GTID:%.+]], i32 0)
// CHECK: call void @__kmpc_taskgroup(ptr [[DEFLOC]], i32 [[GTID]])
// CHECK: call ptr @__kmpc_omp_task_alloc(ptr [[DEFLOC]], i32 [[GTID]], i32 1, i64 80, i64 1, ptr [[TASK1:@.+]])
// CHECK: call void @__kmpc_taskloop(ptr [[DEFLOC]], i32 [[GTID]],
// CHECK: call void @__kmpc_end_taskgroup(ptr [[DEFLOC]], i32 [[GTID]])
// CHECK: call void @__kmpc_end_masked(ptr [[DEFLOC]], i32 [[GTID]])
void masked_taskloop_cancel() {
#pragma omp masked taskloop
  for (int j = 0; j < 42; j++) {
#pragma omp cancel taskgroup
  }
}

// CHECK: define internal {{.*}}i32 [[TASK1]](
// CHECK: [[RES:%.+]] = call i32 @__kmpc_cancel(ptr @{{.+}}, i32 %{{.+}}, i32 4)
// CHECK-NEXT: [[IS_CANCEL:%.+]] = icmp ne i32 [[RES]], 0
// CHECK-NEXT: br i1 [[IS_CANCEL]], label %[[EXIT:.+]], label %{{.+}}
// CHECK: [[EXIT]]:
// CHECK-NEXT: store i32 1, ptr {{%.+}},
// CHECK-NEXT: br label %[[DONE:[^,]+]]
// CHECK: [[DONE]]:
// CHECK: ret i32 0

// CHECK-LABEL: define {{.*}}void @_Z34masked_taskloop_cancellation_pointv()
// CHECK: call i32 @__kmpc_masked(ptr [[DEFLOC]], i32 [[GTID:%.+]], i32 0)
// CHECK: call ptr @__kmpc_omp_task_alloc(ptr [[DEFLOC]], i32 [[GTID]], i32 1, i64 80, i64 1, ptr [[TASK2:@.+]])
// CHECK: call void @__kmpc_taskloop(ptr [[DEFLOC]], i32 [[GTID]],
// CHECK: call void @__kmpc_end_masked(ptr [[DEFLOC]], i32 [[GTID]])
void masked_taskloop_cancellation_point() {
#pragma omp masked taskloop
  for (int j = 0; j < 42; j++) {
#pragma omp cancellation point taskgroup
  }
}

// CHECK: define internal {{.*}}i32 [[TASK2]](
// CHECK: [[RES:%.+]] = call i32 @__kmpc_cancellationpoint(ptr @{{.+}}, i32 %{{.+}}, i32 4)
// CHECK-NEXT: [[IS_CANCEL:%.+]] = icmp ne i32 [[RES]], 0
// CHECK-NEXT: br i1 [[IS_CANCEL]], label %[[EXIT:.+]], label %{{.+}}
// CHECK: [[EXIT]]:
// CHECK-NEXT: store i32 1, ptr {{%.+}},
// CHECK-NEXT: br label %[[DONE:[^,]+]]
// CHECK: [[DONE]]:
// CHECK: ret i32 0

// CHECK-LABEL: define {{.*}}void @_Z31parallel_masked_taskloop_cancelv()
// CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @{{.+}}, i32 0, ptr [[OUTLINED:@.+]])
void parallel_masked_taskloop_cancel() {
#pragma omp parallel masked taskloop
  for (int j = 0; j < 42; j++) {
#pragma omp cancel taskgroup
#pragma omp cancellation point taskgroup
  }
}

// CHECK: define internal void [[OUTLINED]](
// CHECK: call i32 @__kmpc_masked(ptr [[DEFLOC]], i32 [[GTID:%.+]], i32 0)
// CHECK: call void @__kmpc_taskgroup(ptr [[DEFLOC]], i32 [[GTID]])
// CHECK: call ptr @__kmpc_omp_task_alloc(ptr [[DEFLOC]], i32 [[GTID]], i32 1, i64 80, i64 1, ptr [[TASK3:@.+]])
// CHECK: call void @__kmpc_taskloop(ptr [[DEFLOC]], i32 [[GTID]],
// CHECK: call void @__kmpc_end_taskgroup(ptr [[DEFLOC]], i32 [[GTID]])
// CHECK: call void @__kmpc_end_masked(ptr [[DEFLOC]], i32 [[GTID]])

// CHECK: define internal {{.*}}i32 [[TASK3]](
// CHECK: [[RES:%.+]] = call i32 @__kmpc_cancel(ptr @{{.+}}, i32 %{{.+}}, i32 4)
// CHECK-NEXT: [[IS_CANCEL:%.+]] = icmp ne i32 [[RES]], 0
// CHECK-NEXT: br i1 [[IS_CANCEL]], label %[[EXIT:.+]], label %{{.+}}
// CHECK: [[EXIT]]:
// CHECK-NEXT: store i32 1, ptr [[CLEANUP_SLOT:%.+]],
// CHECK-NEXT: br label %[[DONE:[^,]+]]
// CHECK: [[RES:%.+]] = call i32 @__kmpc_cancellationpoint(ptr @{{.+}}, i32 %{{.+}}, i32 4)
// CHECK-NEXT: [[IS_CANCEL:%.+]] = icmp ne i32 [[RES]], 0
// CHECK-NEXT: br i1 [[IS_CANCEL]], label %[[EXIT2:.+]], label %{{.+}}
// CHECK: [[EXIT2]]:
// CHECK-NEXT: store i32 1, ptr [[CLEANUP_SLOT]],
// CHECK-NEXT: br label %[[DONE]]
// CHECK: [[DONE]]:
// CHECK: ret i32 0

// CHECK-LABEL: define {{.*}}void @_Z22master_taskloop_cancelv()
// CHECK: call i32 @__kmpc_master(ptr [[DEFLOC]], i32 [[GTID:%.+]])
// CHECK: call ptr @__kmpc_omp_task_alloc(ptr [[DEFLOC]], i32 [[GTID]], i32 1, i64 80, i64 1, ptr [[TASK4:@.+]])
// CHECK: call void @__kmpc_end_master(ptr [[DEFLOC]], i32 [[GTID]])
void master_taskloop_cancel() {
#pragma omp master taskloop
  for (int j = 0; j < 42; j++) {
#pragma omp cancel taskgroup
#pragma omp cancellation point taskgroup
  }
}

// CHECK: define internal {{.*}}i32 [[TASK4]](
// CHECK: call i32 @__kmpc_cancel(ptr @{{.+}}, i32 %{{.+}}, i32 4)
// CHECK: call i32 @__kmpc_cancellationpoint(ptr @{{.+}}, i32 %{{.+}}, i32 4)
// CHECK: ret i32 0

#endif
