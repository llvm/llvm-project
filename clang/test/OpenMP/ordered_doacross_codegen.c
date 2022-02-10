// RUN: %clang_cc1 -verify -fopenmp -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NORMAL
// RUN: %clang_cc1 -fopenmp -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -triple x86_64-unknown-unknown -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NORMAL

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-enable-irbuilder -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-IRBUILDER
// RUN: %clang_cc1 -fopenmp -fopenmp-enable-irbuilder -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-enable-irbuilder -triple x86_64-unknown-unknown -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-IRBUILDER

// RUN: %clang_cc1 -verify -fopenmp-simd -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -triple x86_64-unknown-unknown -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
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
// CHECK-NORMAL: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num([[IDENT:%.+]])
// CHECK: icmp
// CHECK-NEXT: br i1 %
// CHECK: [[CAST:%.+]] = bitcast [1 x [[KMP_DIM]]]* [[DIMS]] to i8*
// CHECK: call void @llvm.memset.p0i8.i64(i8* align 8 [[CAST]], i8 0, i64 24, i1 false)
// CHECK: [[DIM:%.+]] = getelementptr inbounds [1 x [[KMP_DIM]]], [1 x [[KMP_DIM]]]* [[DIMS]], i64 0, i64 0
// CHECK: getelementptr inbounds [[KMP_DIM]], [[KMP_DIM]]* [[DIM]], i32 0, i32 1
// CHECK: store i64 %{{.+}}, i64* %
// CHECK: getelementptr inbounds [[KMP_DIM]], [[KMP_DIM]]* [[DIM]], i32 0, i32 2
// CHECK: store i64 1, i64* %
// CHECK: [[DIM:%.+]] = getelementptr inbounds [1 x [[KMP_DIM]]], [1 x [[KMP_DIM]]]* [[DIMS]], i64 0, i64 0
// CHECK: [[CAST:%.+]] = bitcast [[KMP_DIM]]* [[DIM]] to i8*
// CHECK-NORMAL: call void @__kmpc_doacross_init([[IDENT]], i32 [[GTID]], i32 1, i8* [[CAST]])
// CHECK-NORMAL: call void @__kmpc_for_static_init_4(%struct.ident_t* @{{.+}}, i32 [[GTID]], i32 33, i32* %{{.+}}, i32* %{{.+}}, i32* %{{.+}}, i32* %{{.+}}, i32 1, i32 1)
#pragma omp for ordered(1)
  for (i = 0; i < n; ++i) {
    a[i] = b[i] + 1;
    foo();
// CHECK: call void @foo()
// CHECK: load i32, i32* [[I:%.+]],
// CHECK-NEXT: sub nsw i32 %{{.+}}, 0
// CHECK-NEXT: sdiv i32 %{{.+}}, 1
// CHECK-NEXT: sext i32 %{{.+}} to i64
// CHECK-NEXT: [[TMP:%.+]] = getelementptr inbounds [1 x i64], [1 x i64]* [[CNT:%.+]], i64 0, i64 0
// CHECK-NEXT: store i64 %{{.+}}, i64* [[TMP]],
// CHECK-NEXT: [[TMP:%.+]] = getelementptr inbounds [1 x i64], [1 x i64]* [[CNT]], i64 0, i64 0
// CHECK-NORMAL-NEXT: call void @__kmpc_doacross_post([[IDENT]], i32 [[GTID]], i64* [[TMP]])
// CHECK-IRBUILDER-NEXT: [[GTID1:%.+]] = call i32 @__kmpc_global_thread_num([[IDENT:%.+]])
// CHECK-IRBUILDER-NEXT: call void @__kmpc_doacross_post([[IDENT]], i32 [[GTID1]], i64* [[TMP]])
#pragma omp ordered depend(source)
    c[i] = c[i] + 1;
    foo();
// CHECK: call void @foo()
// CHECK: load i32, i32* [[I]],
// CHECK-NEXT: sub nsw i32 %{{.+}}, 2
// CHECK-NEXT: sub nsw i32 %{{.+}}, 0
// CHECK-NEXT: sdiv i32 %{{.+}}, 1
// CHECK-NEXT: sext i32 %{{.+}} to i64
// CHECK-NEXT: [[TMP:%.+]] = getelementptr inbounds [1 x i64], [1 x i64]* [[CNT:%.+]], i64 0, i64 0
// CHECK-NEXT: store i64 %{{.+}}, i64* [[TMP]],
// CHECK-NEXT: [[TMP:%.+]] = getelementptr inbounds [1 x i64], [1 x i64]* [[CNT]], i64 0, i64 0
// CHECK-NORMAL-NEXT: call void @__kmpc_doacross_wait([[IDENT]], i32 [[GTID]], i64* [[TMP]])
// CHECK-IRBUILDER-NEXT: [[GTID2:%.+]] = call i32 @__kmpc_global_thread_num([[IDENT:%.+]])
// CHECK-IRBUILDER-NEXT: call void @__kmpc_doacross_wait([[IDENT]], i32 [[GTID2]], i64* [[TMP]])
#pragma omp ordered depend(sink : i - 2)
    d[i] = a[i - 2];
  }
  // CHECK: call void @__kmpc_for_static_fini(
  // CHECK-NORMAL: call void @__kmpc_doacross_fini([[IDENT]], i32 [[GTID]])
  // CHECK: ret i32 0
  return 0;
}
#endif // HEADER
