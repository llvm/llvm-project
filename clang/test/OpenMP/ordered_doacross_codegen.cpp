// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NORMAL
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NORMAL

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-enable-irbuilder -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s --check-prefixes=CHECK,CHECK-IRBUILDER
// RUN: %clang_cc1 -fopenmp -fopenmp-enable-irbuilder -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-enable-irbuilder -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-IRBUILDER

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// CHECK: [[KMP_DIM:%.+]] = type { i64, i64, i64 }
extern int n;
int a[10], b[10], c[10], d[10];
void foo();

// CHECK-LABEL:bar
void bar() {
  int i,j;
// CHECK: call void @__kmpc_doacross_init(
// CHECK: call void @__kmpc_doacross_fini(
#pragma omp parallel for ordered(2)
  for (i = 0; i < n; ++i)
  for (j = 0; j < n; ++j)
    a[i] = b[i] + 1;
// CHECK: call void @__kmpc_doacross_init(
// CHECK: call void @__kmpc_doacross_fini(
 #pragma omp for collapse(2) ordered(2)
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      ;
}

// CHECK-LABEL: @main()
int main() {
  int i;
// CHECK: [[DIMS:%.+]] = alloca [1 x [[KMP_DIM]]],
// CHECK-NORMAL: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(ptr [[IDENT:@.+]])
// CHECK: icmp
// CHECK-NEXT: br i1 %
// CHECK: call void @llvm.memset.p0.i64(ptr align 8 [[DIMS]], i8 0, i64 24, i1 false)
// CHECK: [[DIM:%.+]] = getelementptr inbounds [1 x [[KMP_DIM]]], ptr [[DIMS]], i64 0, i64 0
// CHECK: getelementptr inbounds [[KMP_DIM]], ptr [[DIM]], i32 0, i32 1
// CHECK: store i64 %{{.+}}, ptr %
// CHECK: getelementptr inbounds [[KMP_DIM]], ptr [[DIM]], i32 0, i32 2
// CHECK: store i64 1, ptr %
// CHECK: [[DIM:%.+]] = getelementptr inbounds [1 x [[KMP_DIM]]], ptr [[DIMS]], i64 0, i64 0
// CHECK-NORMAL: call void @__kmpc_doacross_init(ptr [[IDENT]], i32 [[GTID]], i32 1, ptr [[DIM]])
// CHECK-NORMAL: call void @__kmpc_for_static_init_4(ptr @{{.+}}, i32 [[GTID]], i32 33, ptr %{{.+}}, ptr %{{.+}}, ptr %{{.+}}, ptr %{{.+}}, i32 1, i32 1)
#pragma omp for ordered(1)
  for (int i = 0; i < n; ++i) {
    a[i] = b[i] + 1;
    foo();
// CHECK: invoke void [[FOO:.+]](
// CHECK: load i32, ptr [[I:%.+]],
// CHECK-NEXT: sub nsw i32 %{{.+}}, 0
// CHECK-NEXT: sdiv i32 %{{.+}}, 1
// CHECK-NEXT: sext i32 %{{.+}} to i64
// CHECK-NEXT: [[TMP:%.+]] = getelementptr inbounds [1 x i64], ptr [[CNT:%.+]], i64 0, i64 0
// CHECK-NEXT: store i64 %{{.+}}, ptr [[TMP]],
// CHECK-NEXT: [[TMP:%.+]] = getelementptr inbounds [1 x i64], ptr [[CNT]], i64 0, i64 0
// CHECK-NORMAL-NEXT: call void @__kmpc_doacross_post(ptr [[IDENT]], i32 [[GTID]], ptr [[TMP]])
// CHECK-IRBUILDER-NEXT: [[GTID18:%.+]] = call i32 @__kmpc_global_thread_num(ptr [[IDENT:@.+]])
// CHECK-IRBUILDER-NEXT: call void @__kmpc_doacross_post(ptr [[IDENT]], i32 [[GTID18]], ptr [[TMP]])
#pragma omp ordered depend(source)
    c[i] = c[i] + 1;
    foo();
// CHECK: invoke void [[FOO]]
// CHECK: load i32, ptr [[I]],
// CHECK-NEXT: sub nsw i32 %{{.+}}, 2
// CHECK-NEXT: sub nsw i32 %{{.+}}, 0
// CHECK-NEXT: sdiv i32 %{{.+}}, 1
// CHECK-NEXT: sext i32 %{{.+}} to i64
// CHECK-NEXT: [[TMP:%.+]] = getelementptr inbounds [1 x i64], ptr [[CNT:%.+]], i64 0, i64 0
// CHECK-NEXT: store i64 %{{.+}}, ptr [[TMP]],
// CHECK-NEXT: [[TMP:%.+]] = getelementptr inbounds [1 x i64], ptr [[CNT]], i64 0, i64 0
// CHECK-NORMAL-NEXT: call void @__kmpc_doacross_wait(ptr [[IDENT]], i32 [[GTID]], ptr [[TMP]])
// CHECK-IRBUILDER-NEXT: [[GTID30:%.+]] = call i32 @__kmpc_global_thread_num(ptr [[IDENT:@.+]])
// CHECK-IRBUILDER-NEXT: call void @__kmpc_doacross_wait(ptr [[IDENT]], i32 [[GTID30]], ptr [[TMP]])
#pragma omp ordered depend(sink : i - 2)
    d[i] = a[i - 2];
  }

  // CHECK: call void @__kmpc_for_static_fini(
  // CHECK-NORMAL: call void @__kmpc_doacross_fini(ptr [[IDENT]], i32 [[GTID]])
  // CHECK: ret i32 0
  return 0;
}

// CHECK-LABEL: main1
int main1() {
// CHECK: [[DIMS:%.+]] = alloca [1 x [[KMP_DIM]]],
// CHECK-NORMAL: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(ptr [[IDENT:@.+]])
// CHECK: icmp
// CHECK-NEXT: br i1 %
// CHECK: call void @llvm.memset.p0.i64(ptr align 8 [[DIMS]], i8 0, i64 24, i1 false)
// CHECK: [[DIM:%.+]] = getelementptr inbounds [1 x [[KMP_DIM]]], ptr [[DIMS]], i64 0, i64 0
// CHECK: getelementptr inbounds [[KMP_DIM]], ptr [[DIM]], i32 0, i32 1
// CHECK: store i64 %{{.+}}, ptr %
// CHECK: getelementptr inbounds [[KMP_DIM]], ptr [[DIM]], i32 0, i32 2
// CHECK: store i64 1, ptr %
// CHECK: [[DIM:%.+]] = getelementptr inbounds [1 x [[KMP_DIM]]], ptr [[DIMS]], i64 0, i64 0
// CHECK-NORMAL: call void @__kmpc_doacross_init(ptr [[IDENT]], i32 [[GTID]], i32 1, ptr [[DIM]])
// CHECK-NORMAL: call void @__kmpc_for_static_init_4(ptr @{{.+}}, i32 [[GTID]], i32 33, ptr %{{.+}}, ptr %{{.+}}, ptr %{{.+}}, ptr %{{.+}}, i32 1, i32 1)
#pragma omp for ordered(1)
  for (int i = n; i > 0; --i) {
    a[i] = b[i] + 1;
    foo();
// CHECK: invoke void [[FOO:.+]](
// CHECK: [[UB_VAL:%.+]] = load i32, ptr [[UB:%.+]],
// CHECK-NEXT: [[I_VAL:%.+]] = load i32, ptr [[I:%.+]],
// CHECK-NEXT: sub i32 [[UB_VAL]], [[I_VAL]]
// CHECK-NEXT: udiv i32 %{{.+}}, 1
// CHECK-NEXT: zext i32 %{{.+}} to i64
// CHECK-NEXT: [[TMP:%.+]] = getelementptr inbounds [1 x i64], ptr [[CNT:%.+]], i64 0, i64 0
// CHECK-NEXT: store i64 %{{.+}}, ptr [[TMP]],
// CHECK-NEXT: [[TMP:%.+]] = getelementptr inbounds [1 x i64], ptr [[CNT]], i64 0, i64 0
// CHECK-NORMAL-NEXT: call void @__kmpc_doacross_post(ptr [[IDENT]], i32 [[GTID]], ptr [[TMP]])
// CHECK-IRBUILDER-NEXT: [[GTID17:%.+]] = call i32 @__kmpc_global_thread_num(ptr [[IDENT:@.+]])
// CHECK-IRBUILDER-NEXT: call void @__kmpc_doacross_post(ptr [[IDENT]], i32 [[GTID17]], ptr [[TMP]])
#pragma omp ordered depend(source)
    c[i] = c[i] + 1;
    foo();
// CHECK: invoke void [[FOO]]
// CHECK: [[UB_VAL:%.+]] = load i32, ptr [[UB]],
// CHECK-NEXT: [[I_VAL:%.+]] = load i32, ptr [[I]],
// CHECK-NEXT: [[SUB:%.+]] = sub nsw i32 [[I_VAL]], 2
// CHECK-NEXT: sub i32 [[UB_VAL]], [[SUB]]
// CHECK-NEXT: udiv i32 %{{.+}}, 1
// CHECK-NEXT: zext i32 %{{.+}} to i64
// CHECK-NEXT: [[TMP:%.+]] = getelementptr inbounds [1 x i64], ptr [[CNT:%.+]], i64 0, i64 0
// CHECK-NEXT: store i64 %{{.+}}, ptr [[TMP]],
// CHECK-NEXT: [[TMP:%.+]] = getelementptr inbounds [1 x i64], ptr [[CNT]], i64 0, i64 0
// CHECK-NORMAL-NEXT: call void @__kmpc_doacross_wait(ptr [[IDENT]], i32 [[GTID]], ptr [[TMP]])
// CHECK-IRBUILDER-NEXT: [[GTID29:%.+]] = call i32 @__kmpc_global_thread_num(ptr [[IDENT:@.+]])
// CHECK-IRBUILDER-NEXT: call void @__kmpc_doacross_wait(ptr [[IDENT]], i32 [[GTID29]], ptr [[TMP]])
#pragma omp ordered depend(sink : i - 2)
    d[i] = a[i - 2];
  }
  // CHECK: call void @__kmpc_for_static_fini(
  // CHECK-NORMAL: call void @__kmpc_doacross_fini(ptr [[IDENT]], i32 [[GTID]])
  // CHECK: ret i32 0
  return 0;
}

// CHECK: define {{.+}}TestStruct
template <typename T>
struct TestStruct {
  static const int M = 10;
  static const int N = 20;
  T i;
  T a[N][M];
  T b[N][M];
  T foo(T, T);
  T bar(T, T, T);
  void baz(T, T);
  TestStruct() {
// CHECK: [[DIMS:%.+]] = alloca [2 x [[KMP_DIM]]],
// CHECK-NORMAL: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(ptr [[IDENT:@.+]])
// CHECK: call void @llvm.memset.p0.i64(ptr align 8 [[DIMS]], i8 0, i64 48, i1 false)
// CHECK: [[DIM:%.+]] = getelementptr inbounds [2 x [[KMP_DIM]]], ptr [[DIMS]], i64 0, i64 0
// CHECK: getelementptr inbounds [[KMP_DIM]], ptr [[DIM]], i32 0, i32 1
// CHECK: store i64 10, ptr %
// CHECK: getelementptr inbounds [[KMP_DIM]], ptr [[DIM]], i32 0, i32 2
// CHECK: store i64 1, ptr %
// CHECK: [[DIM:%.+]] = getelementptr inbounds [2 x [[KMP_DIM]]], ptr [[DIMS]], i64 0, i64 1
// CHECK: getelementptr inbounds [[KMP_DIM]], ptr [[DIM]], i32 0, i32 1
// CHECK: store i64 %{{.+}}, ptr %
// CHECK: getelementptr inbounds [[KMP_DIM]], ptr [[DIM]], i32 0, i32 2
// CHECK: store i64 1, ptr %
// CHECK: [[DIM:%.+]] = getelementptr inbounds [2 x [[KMP_DIM]]], ptr [[DIMS]], i64 0, i64 0
// CHECK-NORMAL: call void @__kmpc_doacross_init(ptr [[IDENT]], i32 [[GTID]], i32 2, ptr [[DIM]])
// CHECK-NORMAL: call void @__kmpc_for_static_init_4(ptr @{{.+}}, i32 [[GTID]], i32 33, ptr %{{.+}}, ptr %{{.+}}, ptr %{{.+}}, ptr %{{.+}}, i32 1, i32 1)
#pragma omp for ordered(2)
    for (T j = 0; j < M; j++)
      for (i = 0; i < n; i += 2) {
        a[i][j] = foo(i, j);
// CHECK: invoke {{.+TestStruct.+foo}}
// CHECK: load ptr, ptr %
// CHECK: load i32, ptr %
// CHECK: load i32, ptr %
// CHECK: load i32, ptr [[J:%.+]],
// CHECK-NEXT: sub nsw i32 %{{.+}}, 0
// CHECK-NEXT: sdiv i32 %{{.+}}, 1
// CHECK-NEXT: sext i32 %{{.+}} to i64
// CHECK-NORMAL-NEXT: [[TMP:%.+]] = getelementptr inbounds [2 x i64], ptr [[CNT:%.+]], i64 0, i64 0
// CHECK-NORMAL-NEXT: store i64 %{{.+}}, ptr [[TMP:%.+]],
// CHECK-NEXT: [[I:%.+]] = load ptr, ptr [[I_REF:%.+]],
// CHECK-NEXT: load i32, ptr [[I]],
// CHECK-NEXT: sub nsw i32 %{{.+}}, 2
// CHECK-NEXT: sub nsw i32 %{{.+}}, 0
// CHECK-NEXT: sdiv i32 %{{.+}}, 2
// CHECK-NEXT: sext i32 %{{.+}} to i64
// CHECK-IRBUILDER-NEXT: [[TMP:%.+]] = getelementptr inbounds [2 x i64], ptr [[CNT:%.+]], i64 0, i64 0
// CHECK-IRBUILDER-NEXT: store i64 %{{.+}}, ptr [[TMP:%.+]],
// CHECK-NEXT: [[TMP:%.+]] = getelementptr inbounds [2 x i64], ptr [[CNT]], i64 0, i64 1
// CHECK-NEXT: store i64 %{{.+}}, ptr [[TMP]],
// CHECK-NEXT: [[TMP:%.+]] = getelementptr inbounds [2 x i64], ptr [[CNT]], i64 0, i64 0
// CHECK-NORMAL-NEXT: call void @__kmpc_doacross_wait(ptr [[IDENT]], i32 [[GTID]], ptr [[TMP]])
// CHECK-IRBUILDER-NEXT: [[GTID18:%.+]] = call i32 @__kmpc_global_thread_num(ptr [[IDENT:@.+]])
// CHECK-IRBUILDER-NEXT: call void @__kmpc_doacross_wait(ptr [[IDENT]], i32 [[GTID18]], ptr [[TMP]])
// CHECK-NEXT: load i32, ptr [[J:%.+]],
// CHECK-NEXT: sub nsw i32 %{{.+}}, 1
// CHECK-NEXT: sub nsw i32 %{{.+}}, 0
// CHECK-NEXT: sdiv i32 %{{.+}}, 1
// CHECK-NEXT: sext i32 %{{.+}} to i64
// CHECK-NORMAL-NEXT: [[TMP:%.+]] = getelementptr inbounds [2 x i64], ptr [[CNT:%.+]], i64 0, i64 0
// CHECK-NORMAL-NEXT: store i64 %{{.+}}, ptr [[TMP]],
// CHECK-NEXT: [[I:%.+]] = load ptr, ptr [[I_REF]],
// CHECK-NEXT: load i32, ptr [[I]],
// CHECK-NEXT: sub nsw i32 %{{.+}}, 0
// CHECK-NEXT: sdiv i32 %{{.+}}, 2
// CHECK-NEXT: sext i32 %{{.+}} to i64
// CHECK-IRBUILDER-NEXT: [[TMP:%.+]] = getelementptr inbounds [2 x i64], ptr [[CNT:%.+]], i64 0, i64 0
// CHECK-IRBUILDER-NEXT: store i64 %{{.+}}, ptr [[TMP]],
// CHECK-NEXT: [[TMP:%.+]] = getelementptr inbounds [2 x i64], ptr [[CNT]], i64 0, i64 1
// CHECK-NEXT: store i64 %{{.+}}, ptr [[TMP]],
// CHECK-NEXT: [[TMP:%.+]] = getelementptr inbounds [2 x i64], ptr [[CNT]], i64 0, i64 0
// CHECK-NORMAL-NEXT: call void @__kmpc_doacross_wait(ptr [[IDENT]], i32 [[GTID]], ptr [[TMP]])
// CHECK-IRBUILDER-NEXT: [[GTID27:%.+]] = call i32 @__kmpc_global_thread_num(ptr [[IDENT:@.+]])
// CHECK-IRBUILDER-NEXT: call void @__kmpc_doacross_wait(ptr [[IDENT]], i32 [[GTID27]], ptr [[TMP]])
#pragma omp ordered depend(sink : j, i - 2) depend(sink : j - 1, i)
        b[i][j] = bar(a[i][j], b[i - 1][j], b[i][j - 1]);
// CHECK: invoke {{.+TestStruct.+bar}}
// CHECK: load ptr, ptr %
// CHECK: load i32, ptr %
// CHECK: load i32, ptr %
// CHECK: load i32, ptr [[J]],
// CHECK-NEXT: sub nsw i32 %{{.+}}, 0
// CHECK-NEXT: sdiv i32 %{{.+}}, 1
// CHECK-NEXT: sext i32 %{{.+}} to i64
// CHECK-NORMAL-NEXT: [[TMP:%.+]] = getelementptr inbounds [2 x i64], ptr [[CNT:%.+]], i64 0, i64 0
// CHECK-NORMAL-NEXT: store i64 %{{.+}}, ptr [[TMP]],
// CHECK-NEXT: [[I:%.+]] = load ptr, ptr [[I_REF]],
// CHECK-NEXT: load i32, ptr [[I]],
// CHECK-NEXT: sub nsw i32 %{{.+}}, 0
// CHECK-NEXT: sdiv i32 %{{.+}}, 2
// CHECK-NEXT: sext i32 %{{.+}} to i64
// CHECK-IRBUILDER-NEXT: [[TMP:%.+]] = getelementptr inbounds [2 x i64], ptr [[CNT:%.+]], i64 0, i64 0
// CHECK-IRBUILDER-NEXT: store i64 %{{.+}}, ptr [[TMP]],
// CHECK-NEXT: [[TMP:%.+]] = getelementptr inbounds [2 x i64], ptr [[CNT]], i64 0, i64 1
// CHECK-NEXT: store i64 %{{.+}}, ptr [[TMP]],
// CHECK-NEXT: [[TMP:%.+]] = getelementptr inbounds [2 x i64], ptr [[CNT]], i64 0, i64 0
// CHECK-NORMAL-NEXT: call void @__kmpc_doacross_post(ptr [[IDENT]], i32 [[GTID]], ptr [[TMP]])
// CHECK-IRBUILDER-NEXT: [[GTID58:%.+]] = call i32 @__kmpc_global_thread_num(ptr [[IDENT:@.+]])
// CHECK-IRBUILDER-NEXT: call void @__kmpc_doacross_post(ptr [[IDENT]], i32 [[GTID58]], ptr [[TMP]])
#pragma omp ordered depend(source)
        baz(a[i][j], b[i][j]);
      }
  }
  // CHECK: call void @__kmpc_for_static_fini(
  // CHECK-NORMAL: call void @__kmpc_doacross_fini(ptr [[IDENT]], i32 [[GTID]])
  // CHECK: ret
};

TestStruct<int> s;
#endif // HEADER
