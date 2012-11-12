// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -ftrapv %s -emit-llvm -o - | FileCheck %s

unsigned int ui, uj, uk;
int i, j, k;

// CHECK: define void @test0()
void test0() {
  // -ftrapv doesn't affect unsigned arithmetic.
  // CHECK:      [[T1:%.*]] = load i32* @uj
  // CHECK-NEXT: [[T2:%.*]] = load i32* @uk
  // CHECK-NEXT: [[T3:%.*]] = add i32 [[T1]], [[T2]]
  // CHECK-NEXT: store i32 [[T3]], i32* @ui
  ui = uj + uk;

  // CHECK:      [[T1:%.*]] = load i32* @j
  // CHECK-NEXT: [[T2:%.*]] = load i32* @k
  // CHECK-NEXT: [[T3:%.*]] = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 [[T1]], i32 [[T2]])
  // CHECK-NEXT: [[T4:%.*]] = extractvalue { i32, i1 } [[T3]], 0
  // CHECK-NEXT: [[T5:%.*]] = extractvalue { i32, i1 } [[T3]], 1
  // CHECK-NEXT: [[T6:%.*]] = xor i1 [[T5]], true
  // CHECK-NEXT: br i1 [[T6]]
  // CHECK:      call void @llvm.trap()
  i = j + k;
}

// CHECK: define void @test1()
void test1() {
  extern void opaque(int);
  opaque(i++);

  // CHECK:      [[T1:%.*]] = load i32* @i
  // CHECK-NEXT: [[T2:%.*]] = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 [[T1]], i32 1)
  // CHECK-NEXT: [[T3:%.*]] = extractvalue { i32, i1 } [[T2]], 0
  // CHECK-NEXT: [[T4:%.*]] = extractvalue { i32, i1 } [[T2]], 1
  // CHECK-NEXT: [[T5:%.*]] = xor i1 [[T4]], true
  // CHECK-NEXT: br i1 [[T5]]
  // CHECK:      call void @llvm.trap()
}

// CHECK: define void @test2()
void test2() {
  extern void opaque(int);
  opaque(++i);

  // CHECK:      [[T1:%.*]] = load i32* @i
  // CHECK-NEXT: [[T2:%.*]] = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 [[T1]], i32 1)
  // CHECK-NEXT: [[T3:%.*]] = extractvalue { i32, i1 } [[T2]], 0
  // CHECK-NEXT: [[T4:%.*]] = extractvalue { i32, i1 } [[T2]], 1
  // CHECK-NEXT: [[T5:%.*]] = xor i1 [[T4]], true
  // CHECK-NEXT: br i1 [[T5]]
  // CHECK:      call void @llvm.trap()
}

// CHECK: define void @test3(
void test3(int a, int b, float c, float d) {
  // CHECK-NOT:  @llvm.trap
  (void)(a / b);
  (void)(a % b);
  (void)(c / d);
  // CHECK: }
}
