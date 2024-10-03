// RUN: %clang_cc1 -verify -fopenmp -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NORMAL
// RUN: %clang_cc1 -fopenmp -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -triple x86_64-unknown-unknown -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NORMAL

// RUN: %clang_cc1 -verify -fopenmp -triple x86_64-unknown-unknown -emit-llvm %s -o - -fopenmp-version=52 -DOMP52 | FileCheck %s --check-prefixes=CHECK,CHECK-NORMAL
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-enable-irbuilder -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-IRBUILDER
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-enable-irbuilder -triple x86_64-unknown-unknown -fopenmp-version=52 -DOMP52 -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-IRBUILDER
// RUN: %clang_cc1 -fopenmp -fopenmp-enable-irbuilder -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-enable-irbuilder -triple x86_64-unknown-unknown -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-IRBUILDER

// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -DOMP52 -fopenmp-enable-irbuilder -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -DOMP52 -fopenmp-enable-irbuilder -triple x86_64-unknown-unknown -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-IRBUILDER

// RUN: %clang_cc1 -verify -fopenmp-simd -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=52 -DOMP52 -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -triple x86_64-unknown-unknown -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=52 -DOMP52 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=52 -DOMP52 -triple x86_64-unknown-unknown -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// CHECK: [[KMP_DIM:%.+]] = type { i64, i64, i64 }
extern int n;
int a[10], b[10], c[10], d[10];
void foo(void);

// CHECK-LABEL: @main()
int main(void) {
  int i;
// CHECK: [[DIMS:%.+]] = alloca [1 x [[KMP_DIM]]],
// CHECK-NORMAL: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(ptr [[IDENT:@.+]])
// CHECK: icmp
// CHECK-NEXT: br i1 %
// CHECK: call void @llvm.memset.p0.i64(ptr align 8 [[DIMS]], i8 0, i64 24, i1 false)
// CHECK: [[DIM:%.+]] = getelementptr inbounds [1 x [[KMP_DIM]]], ptr [[DIMS]], i64 0, i64 0
// CHECK: getelementptr inbounds nuw [[KMP_DIM]], ptr [[DIM]], i32 0, i32 1
// CHECK: store i64 %{{.+}}, ptr %
// CHECK: getelementptr inbounds nuw [[KMP_DIM]], ptr [[DIM]], i32 0, i32 2
// CHECK: store i64 1, ptr %
// CHECK: [[DIM:%.+]] = getelementptr inbounds [1 x [[KMP_DIM]]], ptr [[DIMS]], i64 0, i64 0
// CHECK-NORMAL: call void @__kmpc_doacross_init(ptr [[IDENT]], i32 [[GTID]], i32 1, ptr [[DIM]])
// CHECK-NORMAL: call void @__kmpc_for_static_init_4(ptr @{{.+}}, i32 [[GTID]], i32 33, ptr %{{.+}}, ptr %{{.+}}, ptr %{{.+}}, ptr %{{.+}}, i32 1, i32 1)
#pragma omp for ordered(1)
  for (i = 0; i < n; ++i) {
    a[i] = b[i] + 1;
    foo();
// CHECK: call void @foo()
// CHECK: load i32, ptr [[I:%.+]],
// CHECK-NEXT: sub nsw i32 %{{.+}}, 0
// CHECK-NEXT: sdiv i32 %{{.+}}, 1
// CHECK-NEXT: sext i32 %{{.+}} to i64
// CHECK-NEXT: [[TMP:%.+]] = getelementptr inbounds [1 x i64], ptr [[CNT:%.+]], i64 0, i64 0
// CHECK-NEXT: store i64 %{{.+}}, ptr [[TMP]],
// CHECK-NEXT: [[TMP:%.+]] = getelementptr inbounds [1 x i64], ptr [[CNT]], i64 0, i64 0
// CHECK-NORMAL-NEXT: call void @__kmpc_doacross_post(ptr [[IDENT]], i32 [[GTID]], ptr [[TMP]])
// CHECK-IRBUILDER-NEXT: [[GTID1:%.+]] = call i32 @__kmpc_global_thread_num(ptr [[IDENT:@.+]])
// CHECK-IRBUILDER-NEXT: call void @__kmpc_doacross_post(ptr [[IDENT]], i32 [[GTID1]], ptr [[TMP]])
#ifdef OMP52
#pragma omp ordered doacross(source:)
#else
#pragma omp ordered depend(source)
#endif
    c[i] = c[i] + 1;
    foo();
// CHECK: call void @foo()
// CHECK: load i32, ptr [[I]],
// CHECK-NEXT: sub nsw i32 %{{.+}}, 2
// CHECK-NEXT: sub nsw i32 %{{.+}}, 0
// CHECK-NEXT: sdiv i32 %{{.+}}, 1
// CHECK-NEXT: sext i32 %{{.+}} to i64
// CHECK-NEXT: [[TMP:%.+]] = getelementptr inbounds [1 x i64], ptr [[CNT:%.+]], i64 0, i64 0
// CHECK-NEXT: store i64 %{{.+}}, ptr [[TMP]],
// CHECK-NEXT: [[TMP:%.+]] = getelementptr inbounds [1 x i64], ptr [[CNT]], i64 0, i64 0
// CHECK-NORMAL-NEXT: call void @__kmpc_doacross_wait(ptr [[IDENT]], i32 [[GTID]], ptr [[TMP]])
// CHECK-IRBUILDER-NEXT: [[GTID2:%.+]] = call i32 @__kmpc_global_thread_num(ptr [[IDENT:@.+]])
// CHECK-IRBUILDER-NEXT: call void @__kmpc_doacross_wait(ptr [[IDENT]], i32 [[GTID2]], ptr [[TMP]])
#ifdef OMP52
#pragma omp ordered doacross(sink : i - 2)
#else
#pragma omp ordered depend(sink : i - 2)
#endif
    d[i] = a[i - 2];
    foo();
// CHECK: call void @foo()
// CHECK: load i32, ptr [[I]],
// CHECK-NEXT: sub nsw i32 %{{.+}}, 1
// CHECK-NEXT: sub nsw i32 %{{.+}}, 0
// CHECK-NEXT: sdiv i32 %{{.+}}, 1
// CHECK-NEXT: sext i32 %{{.+}} to i64
// CHECK-NEXT: [[TMP:%.+]] = getelementptr inbounds [1 x i64], ptr [[CNT:%.+]], i64 0, i64 0
// CHECK-NEXT: store i64 %{{.+}}, ptr [[TMP]],
// CHECK-NEXT: [[TMP:%.+]] = getelementptr inbounds [1 x i64], ptr [[CNT]], i64 0, i64 0
// CHECK-NORMAL-NEXT: call void @__kmpc_doacross_wait(ptr [[IDENT]], i32 [[GTID]], ptr [[TMP]])
// CHECK-IRBUILDER-NEXT: [[GTID2:%.+]] = call i32 @__kmpc_global_thread_num(ptr [[IDENT:@.+]])
// CHECK-IRBUILDER-NEXT: call void @__kmpc_doacross_wait(ptr [[IDENT]], i32 [[GTID2]], ptr [[TMP]])
#ifdef OMP52
#pragma omp ordered doacross(sink :omp_cur_iteration - 1)
#else
#pragma omp ordered depend(sink : i - 1)
#endif
    d[i] = a[i - 1];
    foo();
// CHECK: call void @foo()
// CHECK: load i32, ptr [[I:%.+]],
// CHECK-NEXT: sub nsw i32 %{{.+}}, 0
// CHECK-NEXT: sdiv i32 %{{.+}}, 1
// CHECK-NEXT: sext i32 %{{.+}} to i64
// CHECK-NEXT: [[TMP:%.+]] = getelementptr inbounds [1 x i64], ptr [[CNT:%.+]], i64 0, i64 0
// CHECK-NEXT: store i64 %{{.+}}, ptr [[TMP]],
// CHECK-NEXT: [[TMP:%.+]] = getelementptr inbounds [1 x i64], ptr [[CNT]], i64 0, i64 0
// CHECK-NORMAL-NEXT: call void @__kmpc_doacross_post(ptr [[IDENT]], i32 [[GTID]], ptr [[TMP]])
// CHECK-IRBUILDER-NEXT: [[GTID1:%.+]] = call i32 @__kmpc_global_thread_num(ptr [[IDENT:@.+]])
// CHECK-IRBUILDER-NEXT: call void @__kmpc_doacross_post(ptr [[IDENT]], i32 [[GTID1]], ptr [[TMP]])
#ifdef OMP52
#pragma omp ordered doacross(source:omp_cur_iteration)
#else
#pragma omp ordered depend(source)
#endif
    c[i] = c[i] + 1;
  }
  // CHECK: call void @__kmpc_for_static_fini(
  // CHECK-NORMAL: call void @__kmpc_doacross_fini(ptr [[IDENT]], i32 [[GTID]])
  // CHECK: ret i32 0
  return 0;
}
#endif // HEADER
