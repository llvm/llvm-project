// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-runtime-has-weak -fblocks -fobjc-arc -o - %s | FileCheck %s

// A test to ensure that we generate fused calls at -O0.

@class Test0;
Test0 *test0(void) {
  extern Test0 *test0_helper;
  return test0_helper;

  // CHECK:      [[LD:%.*]] = load [[TEST0:%.*]]** @test0_helper
  // CHECK-NEXT: [[T0:%.*]] = bitcast [[TEST0]]* [[LD]] to i8*
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retainAutoreleaseReturnValue(i8* [[T0]])
  // CHECK-NEXT: [[T2:%.*]] = bitcast i8* [[T1]] to [[TEST0]]*
  // CHECK-NEXT: ret [[TEST0]]* [[T2]]
}

id test1(void) {
  extern id test1_helper;
  return test1_helper;

  // CHECK:      [[LD:%.*]] = load i8** @test1_helper
  // CHECK-NEXT: [[T0:%.*]] = call i8* @objc_retainAutoreleaseReturnValue(i8* [[LD]])
  // CHECK-NEXT: ret i8* [[T0]]
}

void test2(void) {
  // CHECK:      [[X:%.*]] = alloca i8*
  // CHECK-NEXT: store i8* null, i8** [[X]]
  // CHECK-NEXT: call void @objc_destroyWeak(i8** [[X]])
  // CHECK-NEXT: ret void
  __weak id x;
}

id test3(void) {
  extern id test3_helper(void);
  // CHECK:      [[T0:%.*]] = call i8* @test3_helper()
  // CHECK-NEXT: ret i8* [[T0]]
  return test3_helper();
}

@interface Test4 { id x; } @end
@interface Test4_sub : Test4 { id y; } @end
Test4 *test4(void) {
  extern Test4_sub *test4_helper(void);
  // CHECK:      [[T0:%.*]] = call [[TEST4S:%.*]]* @test4_helper()
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST4S]]* [[T0]] to [[TEST4:%.*]]*
  // CHECK-NEXT: ret [[TEST4]]* [[T1]]
  return test4_helper();
}

// rdar://problem/9418404
@class Test5;
void test5(void) {
  Test5 *x, *y;
  if ((x = y))
    y = 0;

// CHECK:    define void @test5()
// CHECK:      [[X:%.*]] = alloca [[TEST5:%.*]]*,
// CHECK-NEXT: [[Y:%.*]] = alloca [[TEST5:%.*]]*,
// CHECK-NEXT: store [[TEST5]]* null, [[TEST5]]** [[X]],
// CHECK-NEXT: store [[TEST5]]* null, [[TEST5]]** [[Y]],
// CHECK-NEXT: [[T0:%.*]] = load [[TEST5]]** [[Y]],
// CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST5]]** [[X]] to i8**
// CHECK-NEXT: [[T2:%.*]] = bitcast [[TEST5]]* [[T0]] to i8*
// CHECK-NEXT: call void @objc_storeStrong(i8** [[T1]], i8* [[T2]])
// CHECK-NEXT: [[T3:%.*]] = icmp ne [[TEST5]]* [[T0]], null
// CHECK-NEXT: br i1 [[T3]],
}
