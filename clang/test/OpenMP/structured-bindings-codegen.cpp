// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 -x c++ -std=c++20 -emit-llvm %s -o - | FileCheck %s

// expected-no-diagnostics

void use(int);

// Struct binding.
struct Point {
  int x, y;
};
Point make_point() { return {1, 2}; }
void test_struct() {
  auto [m, n] = make_point();
#pragma omp parallel
  {
    use(m + n);
  }
}
// CHECK-LABEL: @{{.*}}test_struct{{.*}}()
// CHECK: call void {{.*}}@__kmpc_fork_call({{.*}}, i32 1, ptr @{{.*}}test_struct{{.*}}.omp_outlined", ptr {{.*}})

// CHECK-LABEL: @{{.*}}test_struct{{.*}}.omp_outlined"(
// CHECK-SAME: ptr {{.*}}, ptr {{.*}}, ptr noundef nonnull{{.*}}[[TMP0:%.*]])
// CHECK: [[TMP1:%.*]] = load ptr, ptr {{.*}}, align 8
// CHECK-NEXT: [[X:%.*]] = getelementptr inbounds nuw [[STRUCT_POINT:%.*]], ptr [[TMP1]], i32 0, i32 0
// CHECK-NEXT: [[TMP2:%.*]] = load i32, ptr [[X]], align 4
// CHECK-NEXT: [[Y:%.*]] = getelementptr inbounds nuw [[STRUCT_POINT]], ptr [[TMP1]], i32 0, i32 1
// CHECK-NEXT: [[TMP3:%.*]] = load i32, ptr [[Y]], align 4
//

// Pair binding.
struct pair {
  int first;
  int second;
};
pair make_pair(int a, int b) {
  return {a, b};
}
void test_pair() {
  auto [a, b] = make_pair(1, 2);
#pragma omp parallel
  {
    use(a);
  }
}
// CHECK-LABEL: @{{.*}}test_pair{{.*}}()
// CHECK: call void {{.*}}@__kmpc_fork_call({{.*}}, i32 1, ptr @{{.*}}test_pair{{.*}}.omp_outlined", ptr {{.*}})

// CHECK-LABEL: @{{.*}}test_pair{{.*}}.omp_outlined"(
// CHECK-SAME: ptr {{.*}}, ptr {{.*}}, ptr noundef nonnull{{.*}}[[TMP0:%.*]])
// CHECK: [[TMP1:%.*]] = load ptr, ptr {{.*}}, align 8
// CHECK-NEXT: [[FIRST:%.*]] = getelementptr inbounds nuw [[STRUCT_PAIR:%.*]], ptr [[TMP1]], i32 0, i32 0
// CHECK-NEXT: [[TMP2:%.*]] = load i32, ptr [[FIRST]], align 4
// CHECK-NEXT: call void {{.*}}use{{.*}}"(i32 noundef [[TMP2]])
//

// Array binding.
void test_array() {
  int arr[2] = {1, 2};
  auto [x, y] = arr;
#pragma omp parallel
  {
    use(x + y);
  }
}
// CHECK-LABEL: @{{.*}}test_array{{.*}}()
// CHECK: call void {{.*}}@__kmpc_fork_call({{.*}}, i32 1, ptr @{{.*}}test_array{{.*}}.omp_outlined", ptr {{.*}})

// CHECK-LABEL: @{{.*}}test_array{{.*}}.omp_outlined"(
// CHECK-SAME: ptr {{.*}}, ptr {{.*}}, ptr noundef nonnull{{.*}}[[TMP0:%.*]])
// CHECK: [[TMP1:%.*]] = load ptr, ptr {{.*}}, align 8
// CHECK-NEXT: [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x i32], ptr [[TMP1]], i32 0, i32 0
// CHECK-NEXT: [[TMP2:%.*]] = load i32, ptr [[ARRAYIDX]], align 4
// CHECK-NEXT: [[ARRAYIDX1:%.*]] = getelementptr inbounds [2 x i32], ptr [[TMP1]], i32 0, i32 1
// CHECK-NEXT:  [[TMP3:%.*]] = load i32, ptr [[ARRAYIDX1]], align 4
//

// Binding with bitfields.
struct S {
  int x : 4;
  int y : 4;
};
void test_bitfields() {
  S s{1, 2};
  auto [a, b] = s;
#pragma omp parallel
  {
    use(a + b);
  }
}
// CHECK-LABEL: @{{.*}}test_bitfields{{.*}}()
// CHECK: call void{{.*}}@__kmpc_fork_call({{.*}}, i32 1, ptr @{{.*}}test_bitfields{{.*}}.omp_outlined", ptr {{.*}})

// CHECK-LABEL: @{{.*}}test_bitfields{{.*}}.omp_outlined"(
// CHECK-SAME: ptr {{.*}}, ptr {{.*}}, ptr noundef nonnull{{.*}}[[TMP0:%.*]])
// CHECK: [[TMP1:%.*]] = load ptr, ptr {{.*}}, align 8
// CHECK-NEXT: [[BF_LOAD:%.*]] = load i32, ptr [[TMP1]], align 4
// CHECK-NEXT: [[BF_SHL:%.*]] = shl i32 [[BF_LOAD]], 28
// CHECK-NEXT: [[BF_ASHR:%.*]] = ashr i32 [[BF_SHL]], 28
// CHECK-NEXT: [[BF_LOAD1:%.*]] = load i32, ptr [[TMP1]], align 4
// CHECK-NEXT: [[BF_SHL2:%.*]] = shl i32 [[BF_LOAD1]], 24
// CHECK-NEXT: [[BF_ASHR3:%.*]] = ashr i32 [[BF_SHL2]], 28
//

// Lambda inside OpenMP with captured bindings.
void test_with_lambda() {
  auto [m, n] = make_point();
#pragma omp parallel for collapse(2)
  for (int i = 0; i < 10; i++)
    for (int j = 0; j < 10; j++)
      [m, n](int i, int j) -> void { return; }(i, j);
}
// CHECK-LABEL: @{{.*}}test_with_lambda{{.*}}()
// CHECK: call void{{.*}} @__kmpc_fork_call(ptr {{.*}}, i32 1, ptr @{{.*}}test_with_lambda{{.*}}.omp_outlined", ptr {{.*}})

// CHECK-LABEL: @{{.*}}test_with_lambda{{.*}}.omp_outlined"(
// CHECK-SAME: ptr {{.*}}, ptr {{.*}}, ptr noundef nonnull{{.*}}[[TMP0:%.*]])
// CHECK: [[TMP1:%.*]] = load ptr, ptr {{.*}}, align 8
// CHECK: [[X:%.*]] = getelementptr inbounds nuw [[STRUCT_POINT:%.*]], ptr [[TMP1]], i32 0, i32 0
// CHECK-NEXT: [[TMP13:%.*]] = load i32, ptr [[X]], align 4
// CHECK: [[Y:%.*]] = getelementptr inbounds nuw [[STRUCT_POINT]], ptr [[TMP1]], i32 0, i32 1
// CHECK-NEXT: [[TMP15:%.*]] = load i32, ptr [[Y]], align 4
//

// Only one binding used.
void test_partial_capture() {
  auto [a, b] = make_pair(1, 2);
#pragma omp parallel
  {
    use(a);
  }
}
// CHECK-LABEL: @{{.*}}test_partial_capture{{.*}}()
// CHECK: call void {{.*}}@__kmpc_fork_call(ptr {{.*}}, i32 1, ptr @{{.*}}test_partial_capture{{.*}}.omp_outlined", ptr {{.*}})

// CHECK-LABEL: @{{.*}}test_partial_capture{{.*}}.omp_outlined"(
// CHECK-SAME: ptr {{.*}}, ptr {{.*}}, ptr noundef nonnull{{.*}}[[TMP0:%.*]])
// CHECK: [[TMP1:%.*]] = load ptr, ptr {{.*}}, align 8
// CHECK-NEXT: [[FIRST:%.*]] = getelementptr inbounds nuw [[STRUCT_PAIR:%.*]], ptr [[TMP1]], i32 0, i32 0
// CHECK-NEXT: [[TMP2:%.*]] = load i32, ptr [[FIRST]], align 4
// CHECK-NEXT: call void {{.*}}use{{.*}}"(i32 noundef [[TMP2]])
//

// Nested parallel regions.
void test_nested() {
  auto [x, y] = make_point();
#pragma omp parallel
  {
    use(x);
#pragma omp parallel
    {
      use(y);
    }
  }
}
// CHECK-LABEL: @{{.*}}test_nested{{.*}}()
// CHECK: call void {{.*}}@__kmpc_fork_call(ptr {{.*}}, i32 2, ptr @{{.*}}test_nested{{.*}}.omp_outlined", ptr {{.*}}, ptr {{.*}})

// CHECK-LABEL: @{{.*}}test_nested{{.*}}.omp_outlined"(
// CHECK-SAME: ptr {{.*}}, ptr {{.*}}, ptr noundef nonnull{{.*}}[[TMP0:%.*]], ptr noundef nonnull{{.*}}[[TMP1:%.*]])
// CHECK: [[TMP2:%.*]] = load ptr, ptr {{.*}}, align 8
// CHECK-NEXT: [[TMP3:%.*]] = load ptr, ptr {{.*}}, align 8
// CHECK-NEXT: [[X:%.*]] = getelementptr inbounds nuw [[STRUCT_POINT:%.*]], ptr [[TMP2]], i32 0, i32 0
// CHECK-NEXT: [[TMP4:%.*]] = load i32, ptr [[X]], align 4
// CHECK-NEXT: call void @{{.*}}use{{.*}}"(i32 noundef [[TMP4]])
//

// Multiple bindings in same region.
void test_multiple() {
  auto [a, b] = make_point();
  auto [c, d] = make_pair(3, 4);
#pragma omp parallel
  {
    use(a + b + c + d);
  }
}
// CHECK-LABEL: define dso_local void @"?test_multiple@@YAXXZ"()
// CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 2, ptr @"?test_multiple@@YAXXZ.omp_outlined", ptr %0, ptr %1)

// CHECK-LABEL: define internal void @"?test_multiple@@YAXXZ.omp_outlined"(ptr noalias noundef %.global_tid., ptr noalias noundef %.bound_tid., ptr noundef nonnull align 4 dereferenceable(4) %0, ptr noundef nonnull align 4 dereferenceable(4) %1)
// CHECK: [[TMP2:%.*]] = load ptr, ptr {{.*}}, align 8
// CHECK-NEXT: [[TMP3:%.*]] = load ptr, ptr {{.*}}, align 8
// CHECK-NEXT: [[X:%.*]] = getelementptr inbounds nuw [[STRUCT_POINT:%.*]], ptr [[TMP2]], i32 0, i32 0
// CHECK-NEXT: [[TMP4:%.*]] = load i32, ptr [[X]], align 4
// CHECK-NEXT: [[Y:%.*]] = getelementptr inbounds nuw [[STRUCT_POINT]], ptr [[TMP2]], i32 0, i32 1
// CHECK-NEXT: [[TMP5:%.*]] = load i32, ptr [[Y]], align 4
// CHECK-NEXT: [[ADD:%.*]] = add nsw i32 [[TMP4]], [[TMP5]]
// CHECK-NEXT: [[FIRST:%.*]] = getelementptr inbounds nuw [[STRUCT_PAIR:%.*]], ptr [[TMP3]], i32 0, i32 0
// CHECK-NEXT: [[TMP6:%.*]] = load i32, ptr [[FIRST]], align 4
// CHECK-NEXT: [[ADD2:%.*]] = add nsw i32 [[ADD]], [[TMP6]]
// CHECK-NEXT: [[SECOND:%.*]] = getelementptr inbounds nuw [[STRUCT_PAIR]], ptr [[TMP3]], i32 0, i32 1
// CHECK-NEXT: [[TMP7:%.*]] = load i32, ptr [[SECOND]], align 4
// CHECK-NEXT: [[ADD3:%.*]] = add nsw i32 [[ADD2]], [[TMP7]]
// CHECK-NEXT: call void {{.*}}use{{.*}}(i32 noundef [[ADD3]])

