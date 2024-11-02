// Test host codegen.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=50 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=99 -DOMP99 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,REV
// RUN: %clang_cc1 -fopenmp -fopenmp-version=99 -DOMP99 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=99 -DOMP99 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,REV

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=99 -DOMP99 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=SIMD-ONLY0
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=99 -DOMP99 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=99 -DOMP99 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefixes=SIMD-ONLY0

// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

#ifdef OMP99
#pragma omp requires reverse_offload
#endif

void foo(int n) {

// CHECK:       [[N:%.+]] = load i32, ptr [[N_ADDR:%.+]],
// CHECK:       store i32 [[N]], ptr [[DEVICE_CAP:%.+]],
// CHECK:       [[DEV:%.+]] = load i32, ptr [[DEVICE_CAP]],
// CHECK:       [[DEVICE:%.+]] = sext i32 [[DEV]] to i64
// CHECK:       [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 [[DEVICE]], i32 -1, i32 0, ptr @.[[TGT_REGION:.+]].region_id, ptr %[[KERNEL_ARGS:.+]])
// CHECK-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT0:@.+]]()
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]
#pragma omp target device(n)
  ;
// CHECK:       [[N:%.+]] = load i32, ptr [[N_ADDR]],
// CHECK:       store i32 [[N]], ptr [[DEVICE_CAP:%.+]],
// CHECK:       [[DEV:%.+]] = load i32, ptr [[DEVICE_CAP]],
// CHECK:       [[DEVICE:%.+]] = sext i32 [[DEV]] to i64
// CHECK:       [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 [[DEVICE]], i32 -1, i32 0, ptr @.[[TGT_REGION:.+]].region_id, ptr %[[KERNEL_ARGS:.+]])
// CHECK-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT0:@.+]]()
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]
#pragma omp target device(device_num \
                          : n)
  ;

#ifdef OMP99
// REV-NOT:   call i32 @__tgt_target_kernel(ptr @{{.+}},
// REV:       call void @__omp_offloading_{{.+}}_l62()
// REV-NOT:   call i32 @__tgt_target_kernel(ptr @{{.+}},
#pragma omp target device(ancestor \
                          : n)
  ;
#endif
}

#endif
