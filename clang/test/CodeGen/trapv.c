// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -ftrapv %s -emit-llvm -o - | FileCheck %s

unsigned int ui, uj, uk;
int i, j, k;

// CHECK-LABEL: define{{.*}} void @test0()
void test0(void) {
  // -ftrapv doesn't affect unsigned arithmetic.
  // CHECK:      [[T1:%.*]] = load i32, ptr @uj
  // CHECK-NEXT: [[T2:%.*]] = load i32, ptr @uk
  // CHECK-NEXT: [[T3:%.*]] = add i32 [[T1]], [[T2]]
  // CHECK-NEXT: store i32 [[T3]], ptr @ui
  ui = uj + uk;

  // CHECK:      [[T1:%.*]] = load i32, ptr @j
  // CHECK-NEXT: [[T2:%.*]] = load i32, ptr @k
  // CHECK-NEXT: [[T3:%.*]] = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 [[T1]], i32 [[T2]])
  // CHECK-NEXT: [[T4:%.*]] = extractvalue { i32, i1 } [[T3]], 0
  // CHECK-NEXT: [[T5:%.*]] = extractvalue { i32, i1 } [[T3]], 1
  // CHECK-NEXT: [[T6:%.*]] = xor i1 [[T5]], true
  // CHECK-NEXT: br i1 [[T6]]
  // CHECK:      call void @llvm.ubsantrap(i8 0)
  i = j + k;
}

// CHECK-LABEL: define{{.*}} void @test1()
void test1(void) {
  extern void opaque(int);
  opaque(i++);

  // CHECK:      [[T1:%.*]] = load i32, ptr @i
  // CHECK-NEXT: [[T2:%.*]] = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 [[T1]], i32 1)
  // CHECK-NEXT: [[T3:%.*]] = extractvalue { i32, i1 } [[T2]], 0
  // CHECK-NEXT: [[T4:%.*]] = extractvalue { i32, i1 } [[T2]], 1
  // CHECK-NEXT: [[T5:%.*]] = xor i1 [[T4]], true
  // CHECK-NEXT: br i1 [[T5]]
  // CHECK:      call void @llvm.ubsantrap(i8 0)
}

// CHECK-LABEL: define{{.*}} void @test2()
void test2(void) {
  extern void opaque(int);
  opaque(++i);

  // CHECK:      [[T1:%.*]] = load i32, ptr @i
  // CHECK-NEXT: [[T2:%.*]] = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 [[T1]], i32 1)
  // CHECK-NEXT: [[T3:%.*]] = extractvalue { i32, i1 } [[T2]], 0
  // CHECK-NEXT: [[T4:%.*]] = extractvalue { i32, i1 } [[T2]], 1
  // CHECK-NEXT: [[T5:%.*]] = xor i1 [[T4]], true
  // CHECK-NEXT: br i1 [[T5]]
  // CHECK:      call void @llvm.ubsantrap(i8 0)
}

// CHECK-LABEL: define{{.*}} void @test3(
void test3(int a, int b, float c, float d) {
  // CHECK-NOT:  @llvm.trap
  (void)(a / b);
  (void)(a % b);
  (void)(c / d);
  // CHECK: }
}
