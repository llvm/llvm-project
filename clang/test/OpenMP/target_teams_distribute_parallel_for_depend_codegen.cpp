// Test host codegen.
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32

// RUN: %clang_cc1 -verify -Wno-vla -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-64
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-64
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-32
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-32

// RUN: %clang_cc1 -verify -Wno-vla -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK-DAG: [[TT:%.+]] = type { i64, i8 }
// CHECK-DAG: [[ENTTY:%.+]] = type { ptr, ptr, i[[SZ:32|64]], i32, i32 }

// TCHECK: [[ENTTY:%.+]] = type { ptr, ptr, i{{32|64}}, i32, i32 }

// CHECK-DAG: [[SIZET:@.+]] = private unnamed_addr constant [2 x i64] [i64 0, i64 4]
// CHECK-DAG: [[MAPT:@.+]] = private unnamed_addr constant [2 x i64] [i64 544, i64 800]
// CHECK-DAG: @{{.*}} = weak constant i8 0

// TCHECK: @{{.+}} = weak constant [[ENTTY]]
// TCHECK: @{{.+}} = {{.*}}constant [[ENTTY]]
// TCHECK-NOT: @{{.+}} = weak constant [[ENTTY]]

// Check target registration is registered as a Ctor.
// CHECK: appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @.omp_offloading.requires_reg, ptr null }]


template<typename tx, typename ty>
struct TT{
  tx X;
  ty Y;
};

int global;
extern int global;

// CHECK: define {{.*}}[[FOO:@.+]](
int foo(int n) {
  int a = 0;
  short aa = 0;
  float b[10];
  float bn[n];
  double c[5][10];
  double cn[5][n];
  TT<long long, char> d;
  static long *plocal;

  // CHECK:       [[ADD:%.+]] = add nsw i32
  // CHECK:       store i32 [[ADD]], ptr [[DEVICE_CAP:%.+]],
  // CHECK:       [[GEP:%.+]] = getelementptr inbounds %{{.+}}, ptr %{{.+}}, i32 0, i32 0
  // CHECK:       [[DEV:%.+]] = load i32, ptr [[DEVICE_CAP]],
  // CHECK:       store i32 [[DEV]], ptr [[GEP]],
  // CHECK:       [[TASK:%.+]] = call ptr @__kmpc_omp_task_alloc(ptr [[ID:@.+]], i32 [[GTID:%.+]], i32 1, i[[SZ]] {{20|40}}, i[[SZ]] 4, ptr [[TASK_ENTRY0:@.+]])
  // CHECK:       getelementptr %struct.kmp_depend_info, ptr %{{.+}}, i[[SZ]] 0
  // CHECK:       getelementptr %struct.kmp_depend_info, ptr %{{.+}}, i[[SZ]] 1
  // CHECK:       getelementptr %struct.kmp_depend_info, ptr %{{.+}}, i[[SZ]] 2
  // CHECK:       getelementptr %struct.kmp_depend_info, ptr %{{.+}}, i[[SZ]] 3
  // CHECK:       call void @__kmpc_omp_taskwait_deps_51(ptr [[ID]], i32 [[GTID]], i32 4, ptr %{{.+}}, i32 0, ptr null, i32 0)
  // CHECK:       call void @__kmpc_omp_task_begin_if0(ptr [[ID]], i32 [[GTID]], ptr [[TASK]])
  // CHECK:       call i32 [[TASK_ENTRY0]](i32 [[GTID]], ptr [[TASK]])
  // CHECK:       call void @__kmpc_omp_task_complete_if0(ptr [[ID]], i32 [[GTID]], ptr [[TASK]])
  #pragma omp target teams distribute parallel for device(global + a) depend(in: global) depend(out: a, b, cn[4])
  for (int i = 0; i < 10; ++i) {
  }

  // CHECK:       [[ADD:%.+]] = add nsw i32
  // CHECK:       store i32 [[ADD]], ptr [[DEVICE_CAP:%.+]],

  // CHECK:       [[BOOL:%.+]] = icmp ne i32 %{{.+}}, 0
  // CHECK:       br i1 [[BOOL]], label %[[THEN:.+]], label %[[ELSE:.+]]
  // CHECK:       [[THEN]]:
  // CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [2 x ptr], ptr [[BP:%.+]], i32 0, i32 0
  // CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [2 x ptr], ptr [[P:%.+]], i32 0, i32 0
  // CHECK-DAG:   store ptr [[BP0:%[^,]+]], ptr [[BPADDR0]]
  // CHECK-DAG:   store ptr [[BP0]], ptr [[PADDR0]]

  // CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [2 x ptr], ptr [[BP]], i32 0, i32 1
  // CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [2 x ptr], ptr [[P]], i32 0, i32 1
  // CHECK-DAG:   store i[[SZ]] [[BP1:%[^,]+]], ptr [[BPADDR1]]
  // CHECK-DAG:   store i[[SZ]] [[BP1]], ptr [[PADDR1]]
  // CHECK-DAG:   getelementptr inbounds [2 x ptr], ptr [[BP]], i32 0, i32 0
  // CHECK-DAG:   getelementptr inbounds [2 x ptr], ptr [[P]], i32 0, i32 0
  // CHECK:       [[GEP:%.+]] = getelementptr inbounds %{{.+}}, ptr %{{.+}}, i32 0, i32 2
  // CHECK:       [[DEV:%.+]] = load i32, ptr [[DEVICE_CAP]],
  // CHECK:       store i32 [[DEV]], ptr [[GEP]],
  // CHECK:       [[DEV1:%.+]] = load i32, ptr [[DEVICE_CAP]],
  // CHECK:       [[DEV2:%.+]] = sext i32 [[DEV1]] to i64

  // CHECK:       [[TASK:%.+]] = call ptr @__kmpc_omp_target_task_alloc(ptr [[ID]], i32 [[GTID]], i32 1, i[[SZ]] {{104|60}}, i[[SZ]] {{16|12}}, ptr [[TASK_ENTRY1_:@.+]], i64 [[DEV2]])
  // CHECK:       getelementptr %struct.kmp_depend_info, ptr %{{.+}}, i[[SZ]] 0
  // CHECK:       getelementptr %struct.kmp_depend_info, ptr %{{.+}}, i[[SZ]] 1
  // CHECK:       getelementptr %struct.kmp_depend_info, ptr %{{.+}}, i[[SZ]] 2
  // CHECK:       call i32 @__kmpc_omp_task_with_deps(ptr [[ID]], i32 [[GTID]], ptr [[TASK]], i32 3, ptr %{{.+}}, i32 0, ptr null)
  // CHECK:       br label %[[EXIT:.+]]

  // CHECK:       [[ELSE]]:
  // CHECK-NOT:   getelementptr inbounds [2 x ptr], ptr
  // CHECK:       [[GEP:%.+]] = getelementptr inbounds %{{.+}}, ptr %{{.+}}, i32 0, i32 2
  // CHECK:       [[DEV:%.+]] = load i32, ptr [[DEVICE_CAP]],
  // CHECK:       store i32 [[DEV]], ptr [[GEP]],
  // CHECK:       [[DEV1:%.+]] = load i32, ptr [[DEVICE_CAP]],
  // CHECK:       [[DEV2:%.+]] = sext i32 [[DEV1]] to i64

  // CHECK:       [[TASK:%.+]] = call ptr @__kmpc_omp_target_task_alloc(ptr [[ID]], i32 [[GTID]], i32 1, i[[SZ]] {{56|28}}, i[[SZ]] {{16|12}}, ptr [[TASK_ENTRY1__:@.+]], i64 [[DEV2]])
  // CHECK:       getelementptr %struct.kmp_depend_info, ptr %{{.+}}, i[[SZ]] 0
  // CHECK:       getelementptr %struct.kmp_depend_info, ptr %{{.+}}, i[[SZ]] 1
  // CHECK:       getelementptr %struct.kmp_depend_info, ptr %{{.+}}, i[[SZ]] 2
  // CHECK:       call i32 @__kmpc_omp_task_with_deps(ptr [[ID]], i32 [[GTID]], ptr [[TASK]], i32 3, ptr %{{.+}}, i32 0, ptr null)
  // CHECK:       br label %[[EXIT:.+]]
  // CHECK:       [[EXIT]]:

  #pragma omp target teams distribute parallel for device(global + a) nowait depend(inout: global, a, bn) if(target:a)
  for (int i = 0; i < *plocal; ++i) {
    static int local1;
    *plocal = global;
    local1 = global;
  }

  // CHECK:       [[TASK:%.+]] = call ptr @__kmpc_omp_task_alloc(ptr [[ID]], i32 [[GTID]], i32 1, i[[SZ]] {{48|24}}, i[[SZ]] 4, ptr [[TASK_ENTRY2:@.+]])
  // CHECK:       getelementptr %struct.kmp_depend_info, ptr %{{.+}}, i[[SZ]] 0
  // CHECK:       call void @__kmpc_omp_taskwait_deps_51(ptr [[ID]], i32 [[GTID]], i32 1, ptr %{{.+}}, i32 0, ptr null, i32 0)
  // CHECK:       call void @__kmpc_omp_task_begin_if0(ptr [[ID]], i32 [[GTID]], ptr [[TASK]])
  // CHECK:       call i32 [[TASK_ENTRY2]](i32 [[GTID]], ptr [[TASK]])
  // CHECK:       call void @__kmpc_omp_task_complete_if0(ptr [[ID]], i32 [[GTID]], ptr [[TASK]])
  #pragma omp target teams distribute parallel for if(0) firstprivate(global) depend(out:global)
  for (int i = 0; i < global; ++i) {
    global += 1;
  }

  return a;
}

// Check that the offloading functions are emitted and that the arguments are
// correct and loaded correctly for the target regions in foo().

// CHECK:       define internal void [[HVT0:@.+]]()

// CHECK:       define internal{{.*}} i32 [[TASK_ENTRY0]](i32{{.*}}, ptr noalias noundef %1)
// CHECK:       store ptr null, ptr %
// CHECK:       [[DEVICE_CAP:%.+]] = load ptr,
// CHECK:       [[DEV:%.+]] = load i32, ptr [[DEVICE_CAP]],
// CHECK:       [[DEVICE:%.+]] = sext i32 [[DEV]] to i64
// CHECK:       [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 [[DEVICE]], i32 0, i32 0, ptr @.{{.+}}.region_id, ptr %{{.+}})
// CHECK-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT0]]()
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]
// CHECK:       ret i32 0

// CHECK:       define internal void [[HVT1:@.+]](ptr noundef %{{.+}}, i[[SZ]] noundef %{{.+}})

// CHECK:       define internal{{.*}} i32 [[TASK_ENTRY1_]](i32{{.*}}, ptr noalias noundef %1)
// CHECK:       call void {{%.*}}(
// CHECK:       [[DEVICE_CAP:%.+]] = getelementptr inbounds %{{.+}}, ptr %{{.+}}, i32 0, i32 2
// CHECK:       [[DEV:%.+]] = load i32, ptr [[DEVICE_CAP]],
// CHECK:       [[DEVICE:%.+]] = sext i32 [[DEV]] to i64
// CHECK:       [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 [[DEVICE]], i32 0, i32 0, ptr @.{{.+}}.region_id, ptr %{{.+}})

// CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]
// CHECK:       [[FAIL]]
// CHECK:       [[BP0:%.+]] = load ptr, ptr %
// CHECK:       [[BP1_I32:%.+]] = load i32, ptr @
// CHECK-64:    store i32 [[BP1_I32]], ptr [[BP1_PTR:%[^,]+]],
// CHECK-32:    store i32 [[BP1_I32]], ptr [[BP1_PTR:%[^,]+]],
// CHECK:       [[BP1:%.+]] = load i[[SZ]], ptr [[BP1_PTR]],
// CHECK:       call void [[HVT1]](ptr [[BP0]], i[[SZ]] [[BP1]])
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]
// CHECK:       ret i32 0

// CHECK:       define internal{{.*}} i32 [[TASK_ENTRY1__]](i32{{.*}}, ptr noalias noundef %1)
// CHECK:       call void {{%.*}}(
// CHECK:       [[DEVICE_CAP:%.+]] = getelementptr inbounds %{{.+}}, ptr %{{.+}}, i32 0, i32 2
// CHECK:       [[BP0:%.+]] = load ptr, ptr %
// CHECK:       [[BP1_I32:%.+]] = load i32, ptr @
// CHECK-64:    store i32 [[BP1_I32]], ptr [[BP1_PTR:%[^,]+]],
// CHECK-32:    store i32 [[BP1_I32]], ptr [[BP1_PTR:%[^,]+]],
// CHECK:       [[BP1:%.+]] = load i[[SZ]], ptr [[BP1_PTR]],
// CHECK:       call void [[HVT1]](ptr [[BP0]], i[[SZ]] [[BP1]])
// CHECK:       ret i32 0

// CHECK:       define internal void [[HVT2:@.+]](i[[SZ]] noundef %{{.+}})
// Create stack storage and store argument in there.
// CHECK:       [[AA_ADDR:%.+]] = alloca i[[SZ]], align
// CHECK:       store i[[SZ]] %{{.+}}, ptr [[AA_ADDR]], align
// CHECK-64:    load i32, ptr [[AA_ADDR]], align
// CHECK-32:    load i32, ptr [[AA_ADDR]], align

// CHECK:       define internal{{.*}} i32 [[TASK_ENTRY2]](i32{{.*}}, ptr noalias noundef %1)
// CHECK:       call void {{%.*}}(
// CHECK:       [[BP1_I32:%.+]] = load i32, ptr %
// CHECK-64:    store i32 [[BP1_I32]], ptr [[BP1_PTR:%[^,]+]],
// CHECK-32:    store i32 [[BP1_I32]], ptr [[BP1_PTR:%[^,]+]],
// CHECK:       [[BP1:%.+]] = load i[[SZ]], ptr [[BP1_PTR]],
// CHECK:       call void [[HVT2]](i[[SZ]] [[BP1]])
// CHECK:       ret i32 0


#endif
