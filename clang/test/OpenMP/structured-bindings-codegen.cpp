// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 -x c++ -std=c++20 \
// RUN: -emit-llvm %s -o - | FileCheck %s

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
// CHECK-LABEL: @{{.*}}test_struct{{.*}}.omp_outlined{{.*}}(
// CHECK: getelementptr inbounds{{.*}}i32 0, i32 0
// CHECK: getelementptr inbounds{{.*}}i32 0, i32 1

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
// CHECK-LABEL: @{{.*}}test_pair{{.*}}.omp_outlined{{.*}}(
// CHECK: [[TMP1:%.*]] = load ptr, ptr {{.*}}, align 8
// CHECK: [[FIRST:%.*]] = getelementptr inbounds nuw [[STRUCT_PAIR:%.*]], ptr [[TMP1]], i32 0, i32 0
// CHECK: [[TMP2:%.*]] = load i32, ptr [[FIRST]], align 4
// CHECK: call void {{.*}}use{{.*}}"(i32 noundef [[TMP2]])
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
// CHECK-LABEL: @{{.*}}test_array{{.*}}.omp_outlined{{.*}}(
// CHECK: [[TMP1:%.*]] = load ptr, ptr {{.*}}, align 8
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x i32], ptr [[TMP1]], i32 0, i32 0
// CHECK: [[TMP2:%.*]] = load i32, ptr [[ARRAYIDX]], align 4
// CHECK: [[ARRAYIDX1:%.*]] = getelementptr inbounds [2 x i32], ptr [[TMP1]], i32 0, i32 1
// CHECK:  [[TMP3:%.*]] = load i32, ptr [[ARRAYIDX1]], align 4
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
// CHECK-LABEL: @{{.*}}test_bitfields{{.*}}.omp_outlined{{.*}}(
// CHECK: [[TMP1:%.*]] = load ptr, ptr {{.*}}, align 8
// CHECK: [[BF_LOAD:%.*]] = load i32, ptr [[TMP1]], align 4
// CHECK: [[BF_SHL:%.*]] = shl i32 [[BF_LOAD]], 28
// CHECK: [[BF_ASHR:%.*]] = ashr i32 [[BF_SHL]], 28
// CHECK: [[BF_LOAD1:%.*]] = load i32, ptr [[TMP1]], align 4
// CHECK: [[BF_SHL2:%.*]] = shl i32 [[BF_LOAD1]], 24
// CHECK: [[BF_ASHR3:%.*]] = ashr i32 [[BF_SHL2]], 28
//

// Lambda inside OpenMP with captured bindings.
void test_with_lambda() {
  auto [m, n] = make_point();
#pragma omp parallel for collapse(2)
  for (int i = 0; i < 10; i++)
    for (int j = 0; j < 10; j++)
      [m, n](int i, int j) -> void { return; }(i, j);
}
// CHECK-LABEL: @{{.*}}test_with_lambda{{.*}}.omp_outlined{{.*}}(
// CHECK: [[TMP1:%.*]] = load ptr, ptr {{.*}}, align 8
// CHECK: [[X:%.*]] = getelementptr inbounds nuw [[STRUCT_POINT:%.*]], ptr [[TMP1]], i32 0, i32 0
// CHECK: [[TMP13:%.*]] = load i32, ptr [[X]], align 4
// CHECK: [[Y:%.*]] = getelementptr inbounds nuw [[STRUCT_POINT]], ptr [[TMP1]], i32 0, i32 1
// CHECK: [[TMP15:%.*]] = load i32, ptr [[Y]], align 4
//

// Only one binding used.
void test_partial_capture() {
  auto [a, b] = make_pair(1, 2);
#pragma omp parallel
  {
    use(a);
  }
}
// CHECK-LABEL: @{{.*}}test_partial_capture{{.*}}.omp_outlined{{.*}}(
// CHECK: [[TMP1:%.*]] = load ptr, ptr {{.*}}, align 8
// CHECK: [[FIRST:%.*]] = getelementptr inbounds nuw [[STRUCT_PAIR:%.*]], ptr [[TMP1]], i32 0, i32 0
// CHECK: [[TMP2:%.*]] = load i32, ptr [[FIRST]], align 4
// CHECK: call void {{.*}}use{{.*}}"(i32 noundef [[TMP2]])
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
// CHECK-LABEL: @{{.*}}test_nested{{.*}}.omp_outlined{{.*}}(
// CHECK: [[TMP2:%.*]] = load ptr, ptr {{.*}}, align 8
// CHECK: [[TMP3:%.*]] = load ptr, ptr {{.*}}, align 8
// CHECK: [[X:%.*]] = getelementptr inbounds nuw [[STRUCT_POINT:%.*]], ptr [[TMP2]], i32 0, i32 0
// CHECK: [[TMP4:%.*]] = load i32, ptr [[X]], align 4
// CHECK: call void @{{.*}}use{{.*}}"(i32 noundef [[TMP4]])
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
// CHECK-LABEL: @{{.*}}test_multiple{{.*}}.omp_outlined{{.*}}(
// CHECK: [[TMP2:%.*]] = load ptr, ptr {{.*}}, align 8
// CHECK: [[TMP3:%.*]] = load ptr, ptr {{.*}}, align 8
// CHECK: [[X:%.*]] = getelementptr inbounds nuw [[STRUCT_POINT:%.*]], ptr [[TMP2]], i32 0, i32 0
// CHECK: [[TMP4:%.*]] = load i32, ptr [[X]], align 4
// CHECK: [[Y:%.*]] = getelementptr inbounds nuw [[STRUCT_POINT]], ptr [[TMP2]], i32 0, i32 1
// CHECK: [[TMP5:%.*]] = load i32, ptr [[Y]], align 4
// CHECK: [[ADD:%.*]] = add nsw i32 [[TMP4]], [[TMP5]]
// CHECK: [[FIRST:%.*]] = getelementptr inbounds nuw [[STRUCT_PAIR:%.*]], ptr [[TMP3]], i32 0, i32 0
// CHECK: [[TMP6:%.*]] = load i32, ptr [[FIRST]], align 4
// CHECK: [[ADD2:%.*]] = add nsw i32 [[ADD]], [[TMP6]]
// CHECK: [[SECOND:%.*]] = getelementptr inbounds nuw [[STRUCT_PAIR]], ptr [[TMP3]], i32 0, i32 1
// CHECK: [[TMP7:%.*]] = load i32, ptr [[SECOND]], align 4
// CHECK: [[ADD3:%.*]] = add nsw i32 [[ADD2]], [[TMP7]]
// CHECK: call void {{.*}}use{{.*}}(i32 noundef [[ADD3]])

