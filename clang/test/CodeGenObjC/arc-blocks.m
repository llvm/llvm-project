// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -O2 -disable-llvm-optzns -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -o - %s | FileCheck -check-prefix=CHECK-UNOPT %s

// This shouldn't crash.
void test0(id (^maker)(void)) {
  maker();
}

int (^test1(int x))(void) {
  // CHECK:    define i32 ()* @test1(
  // CHECK:      [[X:%.*]] = alloca i32,
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
  // CHECK-NEXT: store i32 {{%.*}}, i32* [[X]]
  // CHECK:      [[T0:%.*]] = bitcast [[BLOCK_T]]* [[BLOCK]] to i32 ()*
  // CHECK-NEXT: [[T1:%.*]] = bitcast i32 ()* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retainBlock(i8* [[T1]]) nounwind
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to i32 ()*
  // CHECK-NEXT: [[T4:%.*]] = bitcast i32 ()* [[T3]] to i8*
  // CHECK-NEXT: [[T5:%.*]] = call i8* @objc_autoreleaseReturnValue(i8* [[T4]]) nounwind
  // CHECK-NEXT: [[T6:%.*]] = bitcast i8* [[T5]] to i32 ()*
  // CHECK-NEXT: ret i32 ()* [[T6]]
  return ^{ return x; };
}

void test2(id x) {
// CHECK:    define void @test2(
// CHECK:      [[X:%.*]] = alloca i8*,
// CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
// CHECK-NEXT: [[PARM:%.*]] = call i8* @objc_retain(i8* {{%.*}})
// CHECK-NEXT: store i8* [[PARM]], i8** [[X]]
// CHECK-NEXT: [[SLOTREL:%.*]] = getelementptr inbounds [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
// CHECK:      [[SLOT:%.*]] = getelementptr inbounds [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
// CHECK-NEXT: [[T0:%.*]] = load i8** [[X]],
// CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retain(i8* [[T0]])
// CHECK-NEXT: store i8* [[T1]], i8** [[SLOT]],
// CHECK-NEXT: bitcast
// CHECK-NEXT: call void @test2_helper(
// CHECK-NEXT: [[T0:%.*]] = load i8** [[SLOTREL]]
// CHECK-NEXT: call void @objc_release(i8* [[T0]]) nounwind, !clang.imprecise_release
// CHECK-NEXT: [[T0:%.*]] = load i8** [[X]]
// CHECK-NEXT: call void @objc_release(i8* [[T0]]) nounwind, !clang.imprecise_release
// CHECK-NEXT: ret void
  extern void test2_helper(id (^)(void));
  test2_helper(^{ return x; });

// CHECK:    define internal void @__copy_helper_block_
// CHECK:      [[T0:%.*]] = load i8**
// CHECK-NEXT: [[SRC:%.*]] = bitcast i8* [[T0]] to [[BLOCK_T]]*
// CHECK-NEXT: [[T0:%.*]] = load i8**
// CHECK-NEXT: [[DST:%.*]] = bitcast i8* [[T0]] to [[BLOCK_T]]*
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[BLOCK_T]]* [[SRC]], i32 0, i32 5
// CHECK-NEXT: [[T1:%.*]] = load i8** [[T0]]
// CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retain(i8* [[T1]]) nounwind
// CHECK-NEXT: ret void

// CHECK:    define internal void @__destroy_helper_block_
// CHECK:      [[T0:%.*]] = load i8**
// CHECK-NEXT: [[T1:%.*]] = bitcast i8* [[T0]] to [[BLOCK_T]]*
// CHECK-NEXT: [[T2:%.*]] = getelementptr inbounds [[BLOCK_T]]* [[T1]], i32 0, i32 5
// CHECK-NEXT: [[T3:%.*]] = load i8** [[T2]]
// CHECK-NEXT: call void @objc_release(i8* [[T3]])
// CHECK-NEXT: ret void
}

void test3(void (^sink)(id*)) {
  __strong id strong;
  sink(&strong);

  // CHECK:    define void @test3(
  // CHECK:      [[SINK:%.*]] = alloca void (i8**)*
  // CHECK-NEXT: [[STRONG:%.*]] = alloca i8*
  // CHECK-NEXT: [[TEMP:%.*]] = alloca i8*
  // CHECK-NEXT: bitcast void (i8**)* {{%.*}} to i8*
  // CHECK-NEXT: call i8* @objc_retain(
  // CHECK-NEXT: bitcast i8*
  // CHECK-NEXT: store void (i8**)* {{%.*}}, void (i8**)** [[SINK]]
  // CHECK-NEXT: store i8* null, i8** [[STRONG]]

  // CHECK-NEXT: load void (i8**)** [[SINK]]
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: [[BLOCK:%.*]] = bitcast
  // CHECK-NEXT: [[T0:%.*]] = load i8** [[STRONG]]
  // CHECK-NEXT: store i8* [[T0]], i8** [[TEMP]]
  // CHECK-NEXT: [[F0:%.*]] = load i8**
  // CHECK-NEXT: [[F1:%.*]] = bitcast i8* [[F0]] to void (i8*, i8**)*
  // CHECK-NEXT: call void [[F1]](i8* [[BLOCK]], i8** [[TEMP]])
  // CHECK-NEXT: [[T0:%.*]] = load i8** [[TEMP]]
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retain(i8* [[T0]])
  // CHECK-NEXT: [[T2:%.*]] = load i8** [[STRONG]]
  // CHECK-NEXT: store i8* [[T1]], i8** [[STRONG]]
  // CHECK-NEXT: call void @objc_release(i8* [[T2]])

  // CHECK-NEXT: [[T0:%.*]] = load i8** [[STRONG]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]])

  // CHECK-NEXT: load void (i8**)** [[SINK]]
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: call void @objc_release
  // CHECK-NEXT: ret void

}

void test4(void) {
  id test4_source(void);
  void test4_helper(void (^)(void));
  __block id var = test4_source();
  test4_helper(^{ var = 0; });

  // CHECK:    define void @test4()
  // CHECK:      [[VAR:%.*]] = alloca [[BYREF_T:%.*]],
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BYREF_T]]* [[VAR]], i32 0, i32 2
  // 0x02000000 - has copy/dispose helpers
  // CHECK-NEXT: store i32 33554432, i32* [[T0]]
  // CHECK:      [[SLOT:%.*]] = getelementptr inbounds [[BYREF_T]]* [[VAR]], i32 0, i32 6
  // CHECK-NEXT: [[T0:%.*]] = call i8* @test4_source()
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T0]])
  // CHECK-NEXT: store i8* [[T1]], i8** [[SLOT]]
  // CHECK-NEXT: [[SLOT:%.*]] = getelementptr inbounds [[BYREF_T]]* [[VAR]], i32 0, i32 6
  // 0x42800000 - has signature, copy/dispose helpers, as well as BLOCK_HAS_EXTENDED_LAYOUT
  // CHECK:      store i32 -1040187392,
  // CHECK:      [[T0:%.*]] = bitcast [[BYREF_T]]* [[VAR]] to i8*
  // CHECK-NEXT: store i8* [[T0]], i8**
  // CHECK:      call void @test4_helper(
  // CHECK:      [[T0:%.*]] = bitcast [[BYREF_T]]* [[VAR]] to i8*
  // CHECK-NEXT: call void @_Block_object_dispose(i8* [[T0]], i32 8)
  // CHECK-NEXT: [[T0:%.*]] = load i8** [[SLOT]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]])
  // CHECK-NEXT: ret void

  // CHECK:    define internal void @__Block_byref_object_copy_
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BYREF_T]]* {{%.*}}, i32 0, i32 6
  // CHECK-NEXT: load i8**
  // CHECK-NEXT: bitcast i8* {{%.*}} to [[BYREF_T]]*
  // CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[BYREF_T]]* {{%.*}}, i32 0, i32 6
  // CHECK-NEXT: [[T2:%.*]] = load i8** [[T1]]
  // CHECK-NEXT: store i8* [[T2]], i8** [[T0]]
  // CHECK-NEXT: store i8* null, i8** [[T1]]

  // CHECK:    define internal void @__Block_byref_object_dispose_
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BYREF_T]]* {{%.*}}, i32 0, i32 6
  // CHECK-NEXT: [[T1:%.*]] = load i8** [[T0]]
  // CHECK-NEXT: call void @objc_release(i8* [[T1]])

  // CHECK:    define internal void @__test4_block_invoke
  // CHECK:      [[SLOT:%.*]] = getelementptr inbounds {{.*}}, i32 0, i32 6
  // CHECK-NEXT: [[T0:%.*]] = load i8** [[SLOT]], align 8
  // CHECK-NEXT: store i8* null, i8** [[SLOT]],
  // CHECK-NEXT: call void @objc_release(i8* [[T0]])
  // CHECK-NEXT: ret void

  // CHECK:    define internal void @__copy_helper_block_
  // CHECK:      call void @_Block_object_assign(i8* {{%.*}}, i8* {{%.*}}, i32 8)

  // CHECK:    define internal void @__destroy_helper_block_
  // CHECK:      call void @_Block_object_dispose(i8* {{%.*}}, i32 8)
}

void test5(void) {
  extern id test5_source(void);
  void test5_helper(void (^)(void));
  __unsafe_unretained id var = test5_source();
  test5_helper(^{ (void) var; });

  // CHECK:    define void @test5()
  // CHECK:      [[VAR:%.*]] = alloca i8*
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
  // CHECK: [[T0:%.*]] = call i8* @test5_source()
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T0]])
  // CHECK-NEXT: store i8* [[T1]], i8** [[VAR]],
  // CHECK-NEXT: call void @objc_release(i8* [[T1]])
  // 0x40800000 - has signature but no copy/dispose, as well as BLOCK_HAS_EXTENDED_LAYOUT
  // CHECK:      store i32 -1073741824, i32*
  // CHECK:      [[CAPTURE:%.*]] = getelementptr inbounds [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
  // CHECK-NEXT: [[T0:%.*]] = load i8** [[VAR]]
  // CHECK-NEXT: store i8* [[T0]], i8** [[CAPTURE]]
  // CHECK-NEXT: [[T0:%.*]] = bitcast [[BLOCK_T]]* [[BLOCK]] to
  // CHECK: call void @test5_helper
  // CHECK-NEXT: ret void
}

void test6(void) {
  id test6_source(void);
  void test6_helper(void (^)(void));
  __block __weak id var = test6_source();
  test6_helper(^{ var = 0; });

  // CHECK:    define void @test6()
  // CHECK:      [[VAR:%.*]] = alloca [[BYREF_T:%.*]],
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BYREF_T]]* [[VAR]], i32 0, i32 2
  // 0x02000000 - has copy/dispose helpers
  // CHECK-NEXT: store i32 33554432, i32* [[T0]]
  // CHECK:      [[SLOT:%.*]] = getelementptr inbounds [[BYREF_T]]* [[VAR]], i32 0, i32 6
  // CHECK-NEXT: [[T0:%.*]] = call i8* @test6_source()
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T0]])
  // CHECK-NEXT: call i8* @objc_initWeak(i8** [[SLOT]], i8* [[T1]])
  // CHECK-NEXT: call void @objc_release(i8* [[T1]])
  // CHECK-NEXT: [[SLOT:%.*]] = getelementptr inbounds [[BYREF_T]]* [[VAR]], i32 0, i32 6
  // 0x42800000 - has signature, copy/dispose helpers, as well as BLOCK_HAS_EXTENDED_LAYOUT
  // CHECK:      store i32 -1040187392,
  // CHECK:      [[T0:%.*]] = bitcast [[BYREF_T]]* [[VAR]] to i8*
  // CHECK-NEXT: store i8* [[T0]], i8**
  // CHECK:      call void @test6_helper(
  // CHECK:      [[T0:%.*]] = bitcast [[BYREF_T]]* [[VAR]] to i8*
  // CHECK-NEXT: call void @_Block_object_dispose(i8* [[T0]], i32 8)
  // CHECK-NEXT: call void @objc_destroyWeak(i8** [[SLOT]])
  // CHECK-NEXT: ret void

  // CHECK:    define internal void @__Block_byref_object_copy_
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BYREF_T]]* {{%.*}}, i32 0, i32 6
  // CHECK-NEXT: load i8**
  // CHECK-NEXT: bitcast i8* {{%.*}} to [[BYREF_T]]*
  // CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[BYREF_T]]* {{%.*}}, i32 0, i32 6
  // CHECK-NEXT: call void @objc_moveWeak(i8** [[T0]], i8** [[T1]])

  // CHECK:    define internal void @__Block_byref_object_dispose_
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BYREF_T]]* {{%.*}}, i32 0, i32 6
  // CHECK-NEXT: call void @objc_destroyWeak(i8** [[T0]])

  // CHECK:    define internal void @__test6_block_invoke
  // CHECK:      [[SLOT:%.*]] = getelementptr inbounds {{.*}}, i32 0, i32 6
  // CHECK-NEXT: call i8* @objc_storeWeak(i8** [[SLOT]], i8* null)
  // CHECK-NEXT: ret void

  // CHECK:    define internal void @__copy_helper_block_
  // 0x8 - FIELD_IS_BYREF (no FIELD_IS_WEAK because clang in control)
  // CHECK:      call void @_Block_object_assign(i8* {{%.*}}, i8* {{%.*}}, i32 8)

  // CHECK:    define internal void @__destroy_helper_block_
  // 0x8 - FIELD_IS_BYREF (no FIELD_IS_WEAK because clang in control)
  // CHECK:      call void @_Block_object_dispose(i8* {{%.*}}, i32 8)
}

void test7(void) {
  id test7_source(void);
  void test7_helper(void (^)(void));
  void test7_consume(id);
  __weak id var = test7_source();
  test7_helper(^{ test7_consume(var); });

  // CHECK:    define void @test7()
  // CHECK:      [[VAR:%.*]] = alloca i8*,
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
  // CHECK:      [[T0:%.*]] = call i8* @test7_source()
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T0]])
  // CHECK-NEXT: call i8* @objc_initWeak(i8** [[VAR]], i8* [[T1]])
  // CHECK-NEXT: call void @objc_release(i8* [[T1]])
  // 0x42800000 - has signature, copy/dispose helpers, as well as BLOCK_HAS_EXTENDED_LAYOUT
  // CHECK:      store i32 -1040187392,
  // CHECK:      [[SLOT:%.*]] = getelementptr inbounds [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
  // CHECK-NEXT: [[T0:%.*]] = call i8* @objc_loadWeak(i8** [[VAR]])
  // CHECK-NEXT: call i8* @objc_initWeak(i8** [[SLOT]], i8* [[T0]])
  // CHECK:      call void @test7_helper(
  // CHECK-NEXT: call void @objc_destroyWeak(i8** {{%.*}})
  // CHECK-NEXT: call void @objc_destroyWeak(i8** [[VAR]])
  // CHECK-NEXT: ret void

  // CHECK:    define internal void @__test7_block_invoke
  // CHECK:      [[SLOT:%.*]] = getelementptr inbounds [[BLOCK_T]]* {{%.*}}, i32 0, i32 5
  // CHECK-NEXT: [[T0:%.*]] = call i8* @objc_loadWeak(i8** [[SLOT]])
  // CHECK-NEXT: call void @test7_consume(i8* [[T0]])
  // CHECK-NEXT: ret void

  // CHECK:    define internal void @__copy_helper_block_
  // CHECK:      getelementptr
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: call void @objc_copyWeak(

  // CHECK:    define internal void @__destroy_helper_block_
  // CHECK:      getelementptr
  // CHECK-NEXT: call void @objc_destroyWeak(
}

@interface Test8 @end
@implementation Test8
- (void) test {
// CHECK:    define internal void @"\01-[Test8 test]"
// CHECK:      [[SELF:%.*]] = alloca [[TEST8:%.*]]*,
// CHECK-NEXT: alloca i8*
// CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
// CHECK: store
// CHECK-NEXT: store
// CHECK:      [[D0:%.*]] = getelementptr inbounds [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
// CHECK:      [[T0:%.*]] = getelementptr inbounds [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
// CHECK-NEXT: [[T1:%.*]] = load [[TEST8]]** [[SELF]],
// CHECK-NEXT: [[T2:%.*]] = bitcast [[TEST8]]* [[T1]] to i8*
// CHECK-NEXT: [[T3:%.*]] = call i8* @objc_retain(i8* [[T2]])
// CHECK-NEXT: [[T4:%.*]] = bitcast i8* [[T3]] to [[TEST8]]*
// CHECK-NEXT: store [[TEST8]]* [[T4]], [[TEST8]]** [[T0]]
// CHECK-NEXT: bitcast [[BLOCK_T]]* [[BLOCK]] to
// CHECK: call void @test8_helper(
// CHECK-NEXT: [[T1:%.*]] = load [[TEST8]]** [[D0]]
// CHECK-NEXT: [[T2:%.*]] = bitcast [[TEST8]]* [[T1]] to i8*
// CHECK-NEXT: call void @objc_release(i8* [[T2]])
// CHECK-NEXT: ret void

  extern void test8_helper(void (^)(void));
  test8_helper(^{ (void) self; });
}
@end

id test9(void) {
  typedef id __attribute__((ns_returns_retained)) blocktype(void);
  extern void test9_consume_block(blocktype^);
  return ^blocktype {
      extern id test9_produce(void);
      return test9_produce();
  }();

// CHECK:    define i8* @test9(
// CHECK:      load i8** getelementptr
// CHECK-NEXT: bitcast i8*
// CHECK-NEXT: call i8* 
// CHECK-NEXT: call i8* @objc_autoreleaseReturnValue
// CHECK-NEXT: ret i8*

// CHECK:      call i8* @test9_produce()
// CHECK-NEXT: call i8* @objc_retain
// CHECK-NEXT: ret i8*
}

// rdar://problem/9814099
// Test that we correctly initialize __block variables
// when the initialization captures the variable.
void test10a(void) {
  __block void (^block)(void) = ^{ block(); };
  // CHECK:    define void @test10a()
  // CHECK:      [[BYREF:%.*]] = alloca [[BYREF_T:%.*]],

  // Zero-initialization before running the initializer.
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BYREF_T]]* [[BYREF]], i32 0, i32 6
  // CHECK-NEXT: store void ()* null, void ()** [[T0]], align 8

  // Run the initializer as an assignment.
  // CHECK:      [[T0:%.*]] = bitcast void ()* {{%.*}} to i8*
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retainBlock(i8* [[T0]])
  // CHECK-NEXT: [[T2:%.*]] = bitcast i8* [[T1]] to void ()*
  // CHECK-NEXT: [[T3:%.*]] = getelementptr inbounds [[BYREF_T]]* [[BYREF]], i32 0, i32 1
  // CHECK-NEXT: [[T4:%.*]] = load [[BYREF_T]]** [[T3]]
  // CHECK-NEXT: [[T5:%.*]] = getelementptr inbounds [[BYREF_T]]* [[T4]], i32 0, i32 6
  // CHECK-NEXT: [[T6:%.*]] = load void ()** [[T5]], align 8
  // CHECK-NEXT: store void ()* {{%.*}}, void ()** [[T5]], align 8
  // CHECK-NEXT: [[T7:%.*]] = bitcast void ()* [[T6]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T7]])

  // Destroy at end of function.
  // CHECK-NEXT: [[SLOT:%.*]] = getelementptr inbounds [[BYREF_T]]* [[BYREF]], i32 0, i32 6
  // CHECK-NEXT: [[T0:%.*]] = bitcast [[BYREF_T]]* [[BYREF]] to i8*
  // CHECK-NEXT: call void @_Block_object_dispose(i8* [[T0]], i32 8)
  // CHECK-NEXT: [[T1:%.*]] = load void ()** [[SLOT]]
  // CHECK-NEXT: [[T2:%.*]] = bitcast void ()* [[T1]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T2]])
  // CHECK-NEXT: ret void
}

// <rdar://problem/10402698>: do this copy and dispose with
// objc_retainBlock/release instead of _Block_object_assign/destroy.
// We can also use _Block_object_assign/destroy with
// BLOCK_FIELD_IS_BLOCK as long as we don't pass BLOCK_BYREF_CALLER.

// CHECK: define internal void @__Block_byref_object_copy
// CHECK:      [[D0:%.*]] = load i8** {{%.*}}
// CHECK-NEXT: [[D1:%.*]] = bitcast i8* [[D0]] to [[BYREF_T]]*
// CHECK-NEXT: [[D2:%.*]] = getelementptr inbounds [[BYREF_T]]* [[D1]], i32 0, i32 6
// CHECK-NEXT: [[S0:%.*]] = load i8** {{%.*}}
// CHECK-NEXT: [[S1:%.*]] = bitcast i8* [[S0]] to [[BYREF_T]]*
// CHECK-NEXT: [[S2:%.*]] = getelementptr inbounds [[BYREF_T]]* [[S1]], i32 0, i32 6
// CHECK-NEXT: [[T0:%.*]] = load void ()** [[S2]], align 8
// CHECK-NEXT: [[T1:%.*]] = bitcast void ()* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retainBlock(i8* [[T1]])
// CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to void ()*
// CHECK-NEXT: store void ()* [[T3]], void ()** [[D2]], align 8
// CHECK-NEXT: ret void

// CHECK: define internal void @__Block_byref_object_dispose
// CHECK:      [[T0:%.*]] = load i8** {{%.*}}
// CHECK-NEXT: [[T1:%.*]] = bitcast i8* [[T0]] to [[BYREF_T]]*
// CHECK-NEXT: [[T2:%.*]] = getelementptr inbounds [[BYREF_T]]* [[T1]], i32 0, i32 6
// CHECK-NEXT: [[T3:%.*]] = load void ()** [[T2]]
// CHECK-NEXT: [[T4:%.*]] = bitcast void ()* [[T3]] to i8*
// CHECK-NEXT: call void @objc_release(i8* [[T4]])
// CHECK-NEXT: ret void

// Test that we correctly assign to __block variables when the
// assignment captures the variable.
void test10b(void) {
  __block void (^block)(void);
  block = ^{ block(); };

  // CHECK:    define void @test10b()
  // CHECK:      [[BYREF:%.*]] = alloca [[BYREF_T:%.*]],

  // Zero-initialize.
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BYREF_T]]* [[BYREF]], i32 0, i32 6
  // CHECK-NEXT: store void ()* null, void ()** [[T0]], align 8

  // CHECK-NEXT: [[SLOT:%.*]] = getelementptr inbounds [[BYREF_T]]* [[BYREF]], i32 0, i32 6

  // The assignment.
  // CHECK:      [[T0:%.*]] = bitcast void ()* {{%.*}} to i8*
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retainBlock(i8* [[T0]])
  // CHECK-NEXT: [[T2:%.*]] = bitcast i8* [[T1]] to void ()*
  // CHECK-NEXT: [[T3:%.*]] = getelementptr inbounds [[BYREF_T]]* [[BYREF]], i32 0, i32 1
  // CHECK-NEXT: [[T4:%.*]] = load [[BYREF_T]]** [[T3]]
  // CHECK-NEXT: [[T5:%.*]] = getelementptr inbounds [[BYREF_T]]* [[T4]], i32 0, i32 6
  // CHECK-NEXT: [[T6:%.*]] = load void ()** [[T5]], align 8
  // CHECK-NEXT: store void ()* {{%.*}}, void ()** [[T5]], align 8
  // CHECK-NEXT: [[T7:%.*]] = bitcast void ()* [[T6]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T7]])

  // Destroy at end of function.
  // CHECK-NEXT: [[T0:%.*]] = bitcast [[BYREF_T]]* [[BYREF]] to i8*
  // CHECK-NEXT: call void @_Block_object_dispose(i8* [[T0]], i32 8)
  // CHECK-NEXT: [[T1:%.*]] = load void ()** [[SLOT]]
  // CHECK-NEXT: [[T2:%.*]] = bitcast void ()* [[T1]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T2]])
  // CHECK-NEXT: ret void
}

// rdar://problem/10088932
void test11_helper(id);
void test11a(void) {
  int x;
  test11_helper(^{ (void) x; });

  // CHECK:    define void @test11a()
  // CHECK:      [[X:%.*]] = alloca i32, align 4
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]], align 8
  // CHECK:      [[T0:%.*]] = bitcast [[BLOCK_T]]* [[BLOCK]] to void ()*
  // CHECK-NEXT: [[T1:%.*]] = bitcast void ()* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retainBlock(i8* [[T1]])
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to void ()*
  // CHECK-NEXT: [[T4:%.*]] = bitcast void ()* [[T3]] to i8*
  // CHECK-NEXT: call void @test11_helper(i8* [[T4]])
  // CHECK-NEXT: [[T5:%.*]] = bitcast void ()* [[T3]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T5]])
  // CHECK-NEXT: ret void
}
void test11b(void) {
  int x;
  id b = ^{ (void) x; };

  // CHECK:    define void @test11b()
  // CHECK:      [[X:%.*]] = alloca i32, align 4
  // CHECK-NEXT: [[B:%.*]] = alloca i8*, align 8
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]], align 8
  // CHECK:      [[T0:%.*]] = bitcast [[BLOCK_T]]* [[BLOCK]] to void ()*
  // CHECK-NEXT: [[T1:%.*]] = bitcast void ()* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retainBlock(i8* [[T1]])
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to void ()*
  // CHECK-NEXT: [[T4:%.*]] = bitcast void ()* [[T3]] to i8*
  // CHECK-NEXT: store i8* [[T4]], i8** [[B]], align 8
  // CHECK-NEXT: [[T5:%.*]] = load i8** [[B]]
  // CHECK-NEXT: call void @objc_release(i8* [[T5]])
  // CHECK-NEXT: ret void
}

// rdar://problem/9979150
@interface Test12
@property (strong) void(^ablock)(void);
@property (nonatomic, strong) void(^nblock)(void);
@end
@implementation Test12
@synthesize ablock, nblock;
// CHECK:    define internal void ()* @"\01-[Test12 ablock]"(
// CHECK:    call i8* @objc_getProperty(i8* {{%.*}}, i8* {{%.*}}, i64 {{%.*}}, i1 zeroext true)

// CHECK:    define internal void @"\01-[Test12 setAblock:]"(
// CHECK:    call void @objc_setProperty(i8* {{%.*}}, i8* {{%.*}}, i64 {{%.*}}, i8* {{%.*}}, i1 zeroext true, i1 zeroext true)

// CHECK:    define internal void ()* @"\01-[Test12 nblock]"(
// CHECK:    call i8* @objc_getProperty(i8* {{%.*}}, i8* {{%.*}}, i64 {{%.*}}, i1 zeroext false)

// CHECK:    define internal void @"\01-[Test12 setNblock:]"(
// CHECK:    call void @objc_setProperty(i8* {{%.*}}, i8* {{%.*}}, i64 {{%.*}}, i8* {{%.*}}, i1 zeroext false, i1 zeroext true)
@end

// rdar://problem/10131784
void test13(id x) {
  extern void test13_helper(id);
  extern void test13_use(void(^)(void));

  void (^b)(void) = (x ? ^{test13_helper(x);} : 0);
  test13_use(b);

  // CHECK:    define void @test13(
  // CHECK:      [[X:%.*]] = alloca i8*, align 8
  // CHECK-NEXT: [[B:%.*]] = alloca void ()*, align 8
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:.*]], align 8
  // CHECK-NEXT: [[CLEANUP_ACTIVE:%.*]] = alloca i1
  // CHECK-NEXT: [[T0:%.*]] = call i8* @objc_retain(i8* {{%.*}})
  // CHECK-NEXT: store i8* [[T0]], i8** [[X]], align 8
  // CHECK-NEXT: [[CLEANUP_ADDR:%.*]] = getelementptr inbounds [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
  // CHECK-NEXT: [[T0:%.*]] = load i8** [[X]], align 8
  // CHECK-NEXT: [[T1:%.*]] = icmp ne i8* [[T0]], null
  // CHECK-NEXT: store i1 false, i1* [[CLEANUP_ACTIVE]]
  // CHECK-NEXT: br i1 [[T1]],

  // CHECK-NOT:  br
  // CHECK:      [[CAPTURE:%.*]] = getelementptr inbounds [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
  // CHECK-NEXT: [[T0:%.*]] = load i8** [[X]], align 8
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retain(i8* [[T0]])
  // CHECK-NEXT: store i8* [[T1]], i8** [[CAPTURE]], align 8
  // CHECK-NEXT: store i1 true, i1* [[CLEANUP_ACTIVE]]
  // CHECK-NEXT: bitcast [[BLOCK_T]]* [[BLOCK]] to void ()*
  // CHECK-NEXT: br label
  // CHECK:      br label
  // CHECK:      [[T0:%.*]] = phi void ()*
  // CHECK-NEXT: [[T1:%.*]] = bitcast void ()* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retainBlock(i8* [[T1]])
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to void ()*
  // CHECK-NEXT: store void ()* [[T3]], void ()** [[B]], align 8
  // CHECK-NEXT: [[T0:%.*]] = load void ()** [[B]], align 8
  // CHECK-NEXT: call void @test13_use(void ()* [[T0]])
  // CHECK-NEXT: [[T0:%.*]] = load void ()** [[B]]
  // CHECK-NEXT: [[T1:%.*]] = bitcast void ()* [[T0]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T1]])

  // CHECK-NEXT: [[T0:%.*]] = load i1* [[CLEANUP_ACTIVE]]
  // CHECK-NEXT: br i1 [[T0]]
  // CHECK:      [[T0:%.*]] = load i8** [[CLEANUP_ADDR]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]])
  // CHECK-NEXT: br label

  // CHECK:      [[T0:%.*]] = load i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]])
  // CHECK-NEXT: ret void
}

// <rdar://problem/10907510>
void test14() {
  void (^const x[1])(void) = { ^{} };
}

// rdar://11149025
// Don't make invalid ASTs and crash.
void test15_helper(void (^block)(void), int x);
void test15(int a) {
  test15_helper(^{ (void) a; }, ({ a; }));
}

// rdar://11016025
void test16() {
  void (^BLKVAR)(void) = ^{ BLKVAR(); };

  // CHECK: define void @test16(
  // CHECK: [[BLKVAR:%.*]]  = alloca void ()*, align 8
  // CHECK-NEXT:  [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
  // CHECK-NEXT:  [[SLOTREL:%.*]] = getelementptr inbounds [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
  // CHECK-NEXT:  store void ()* null, void ()** [[BLKVAR]], align 8
}

// rdar://12151005
//
// This is an intentional exception to our conservative jump-scope
// checking for full-expressions containing block literals with
// non-trivial cleanups: if the block literal appears in the operand
// of a return statement, there's no need to extend its lifetime.
id (^test17(id self, int which))(void) {
  switch (which) {
  case 1: return ^{ return self; };
  case 0: return ^{ return self; };
  }
  return (void*) 0;
}
// CHECK:    define i8* ()* @test17(
// CHECK:      [[RET:%.*]] = alloca i8* ()*, align
// CHECK-NEXT: [[SELF:%.*]] = alloca i8*,
// CHECK:      [[B0:%.*]] = alloca [[BLOCK:<.*>]], align
// CHECK:      [[B1:%.*]] = alloca [[BLOCK]], align
// CHECK:      [[T0:%.*]] = call i8* @objc_retain(i8*
// CHECK-NEXT: store i8* [[T0]], i8** [[SELF]], align
// CHECK-NOT:  objc_retain
// CHECK-NOT:  objc_release
// CHECK:      [[DESTROY:%.*]] = getelementptr inbounds [[BLOCK]]* [[B0]], i32 0, i32 5
// CHECK-NOT:  objc_retain
// CHECK-NOT:  objc_release
// CHECK:      [[T0:%.*]] = getelementptr inbounds [[BLOCK]]* [[B0]], i32 0, i32 5
// CHECK-NEXT: [[T1:%.*]] = load i8** [[SELF]], align
// CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retain(i8* [[T1]])
// CHECK-NEXT: store i8* [[T2]], i8** [[T0]],
// CHECK-NEXT: [[T0:%.*]] = bitcast [[BLOCK]]* [[B0]] to i8* ()*
// CHECK-NEXT: [[T1:%.*]] = bitcast i8* ()* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retainBlock(i8* [[T1]])
// CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to i8* ()*
// CHECK-NEXT: store i8* ()* [[T3]], i8* ()** [[RET]]
// CHECK-NEXT: [[T0:%.*]] = load i8** [[DESTROY]]
// CHECK-NEXT: call void @objc_release(i8* [[T0]])
// CHECK-NEXT: store i32
// CHECK-NEXT: br label
// CHECK-NOT:  objc_retain
// CHECK-NOT:  objc_release
// CHECK:      [[DESTROY:%.*]] = getelementptr inbounds [[BLOCK]]* [[B1]], i32 0, i32 5
// CHECK-NOT:  objc_retain
// CHECK-NOT:  objc_release
// CHECK:      [[T0:%.*]] = getelementptr inbounds [[BLOCK]]* [[B1]], i32 0, i32 5
// CHECK-NEXT: [[T1:%.*]] = load i8** [[SELF]], align
// CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retain(i8* [[T1]])
// CHECK-NEXT: store i8* [[T2]], i8** [[T0]],
// CHECK-NEXT: [[T0:%.*]] = bitcast [[BLOCK]]* [[B1]] to i8* ()*
// CHECK-NEXT: [[T1:%.*]] = bitcast i8* ()* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retainBlock(i8* [[T1]])
// CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to i8* ()*
// CHECK-NEXT: store i8* ()* [[T3]], i8* ()** [[RET]]
// CHECK-NEXT: [[T0:%.*]] = load i8** [[DESTROY]]
// CHECK-NEXT: call void @objc_release(i8* [[T0]])
// CHECK-NEXT: store i32
// CHECK-NEXT: br label

void test18(id x) {
// CHECK-UNOPT:    define void @test18(
// CHECK-UNOPT:      [[X:%.*]] = alloca i8*,
// CHECK-UNOPT-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
// CHECK-UNOPT-NEXT: [[PARM:%.*]] = call i8* @objc_retain(i8* {{%.*}})
// CHECK-UNOPT-NEXT: store i8* [[PARM]], i8** [[X]]
// CHECK-UNOPT-NEXT: [[SLOTREL:%.*]] = getelementptr inbounds [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
// CHECK-UNOPT:      [[SLOT:%.*]] = getelementptr inbounds [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
// CHECK-UNOPT-NEXT: [[T0:%.*]] = load i8** [[X]],
// CHECK-UNOPT-NEXT: [[T1:%.*]] = call i8* @objc_retain(i8* [[T0]])
// CHECK-UNOPT-NEXT: store i8* [[T1]], i8** [[SLOT]],
// CHECK-UNOPT-NEXT: bitcast
// CHECK-UNOPT-NEXT: call void @test18_helper(
// CHECK-UNOPT-NEXT: call void @objc_storeStrong(i8** [[SLOTREL]], i8* null) nounwind
// CHECK-UNOPT-NEXT: call void @objc_storeStrong(i8** [[X]], i8* null) nounwind
// CHECK-UNOPT-NEXT: ret void
  extern void test18_helper(id (^)(void));
  test18_helper(^{ return x; });

// CHECK-UNOPT:    define internal void @__copy_helper_block_
// CHECK-UNOPT:      [[T0:%.*]] = load i8**
// CHECK-UNOPT-NEXT: [[SRC:%.*]] = bitcast i8* [[T0]] to [[BLOCK_T]]*
// CHECK-UNOPT-NEXT: [[T0:%.*]] = load i8**
// CHECK-UNOPT-NEXT: [[DST:%.*]] = bitcast i8* [[T0]] to [[BLOCK_T]]*
// CHECK-UNOPT-NEXT: [[T0:%.*]] = getelementptr inbounds [[BLOCK_T]]* [[SRC]], i32 0, i32 5
// CHECK-UNOPT-NEXT: [[T1:%.*]] = getelementptr inbounds [[BLOCK_T]]* [[DST]], i32 0, i32 5
// CHECK-UNOPT-NEXT: [[T2:%.*]] = load i8** [[T0]]
// CHECK-UNOPT-NEXT: store i8* null, i8** [[T1]]
// CHECK-UNOPT-NEXT: call void @objc_storeStrong(i8** [[T1]], i8* [[T2]]) nounwind
// CHECK-UNOPT-NEXT: ret void

// CHECK-UNOPT:    define internal void @__destroy_helper_block_
// CHECK-UNOPT:      [[T0:%.*]] = load i8**
// CHECK-UNOPT-NEXT: [[T1:%.*]] = bitcast i8* [[T0]] to [[BLOCK_T]]*
// CHECK-UNOPT-NEXT: [[T2:%.*]] = getelementptr inbounds [[BLOCK_T]]* [[T1]], i32 0, i32 5
// CHECK-UNOPT-NEXT: call void @objc_storeStrong(i8** [[T2]], i8* null)
// CHECK-UNOPT-NEXT: ret void
}
