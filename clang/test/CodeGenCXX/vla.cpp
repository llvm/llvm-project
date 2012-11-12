// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -emit-llvm -o - | FileCheck %s

template<typename T>
struct S {
  static int n;
};
template<typename T> int S<T>::n = 5;

int f() {
  // Make sure that the reference here is enough to trigger the instantiation of
  // the static data member.
  // CHECK: @_ZN1SIiE1nE = weak_odr global i32 5
  int a[S<int>::n];
  return sizeof a;
}

// rdar://problem/9506377
void test0(void *array, int n) {
  // CHECK: define void @_Z5test0Pvi(
  // CHECK:      [[ARRAY:%.*]] = alloca i8*, align 8
  // CHECK-NEXT: [[N:%.*]] = alloca i32, align 4
  // CHECK-NEXT: [[REF:%.*]] = alloca i16*, align 8
  // CHECK-NEXT: [[S:%.*]] = alloca i16, align 2
  // CHECK-NEXT: store i8* 
  // CHECK-NEXT: store i32

  // Capture the bounds.
  // CHECK-NEXT: [[T0:%.*]] = load i32* [[N]], align 4
  // CHECK-NEXT: [[DIM0:%.*]] = zext i32 [[T0]] to i64
  // CHECK-NEXT: [[T0:%.*]] = load i32* [[N]], align 4
  // CHECK-NEXT: [[T1:%.*]] = add nsw i32 [[T0]], 1
  // CHECK-NEXT: [[DIM1:%.*]] = zext i32 [[T1]] to i64
  typedef short array_t[n][n+1];

  // CHECK-NEXT: [[T0:%.*]] = load i8** [[ARRAY]], align 8
  // CHECK-NEXT: [[T1:%.*]] = bitcast i8* [[T0]] to i16*
  // CHECK-NEXT: store i16* [[T1]], i16** [[REF]], align 8
  array_t &ref = *(array_t*) array;

  // CHECK-NEXT: [[T0:%.*]] = load i16** [[REF]]
  // CHECK-NEXT: [[T1:%.*]] = mul nsw i64 1, [[DIM1]]
  // CHECK-NEXT: [[T2:%.*]] = getelementptr inbounds i16* [[T0]], i64 [[T1]]
  // CHECK-NEXT: [[T3:%.*]] = getelementptr inbounds i16* [[T2]], i64 2
  // CHECK-NEXT: store i16 3, i16* [[T3]]
  ref[1][2] = 3;

  // CHECK-NEXT: [[T0:%.*]] = load i16** [[REF]]
  // CHECK-NEXT: [[T1:%.*]] = mul nsw i64 4, [[DIM1]]
  // CHECK-NEXT: [[T2:%.*]] = getelementptr inbounds i16* [[T0]], i64 [[T1]]
  // CHECK-NEXT: [[T3:%.*]] = getelementptr inbounds i16* [[T2]], i64 5
  // CHECK-NEXT: [[T4:%.*]] = load i16* [[T3]]
  // CHECK-NEXT: store i16 [[T4]], i16* [[S]], align 2
  short s = ref[4][5];

  // CHECK-NEXT: ret void
}
