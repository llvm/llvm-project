// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -O2 -disable-llvm-passes -o - %s | FileCheck -check-prefix=CHECK -check-prefix=CHECK-COMMON %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -o - %s | FileCheck -check-prefix=CHECK-UNOPT -check-prefix=CHECK-COMMON %s

// CHECK-COMMON: %[[STRUCT_BLOCK_DESCRIPTOR:.*]] = type { i64, i64 }
// CHECK-COMMON: @[[BLOCK_DESCRIPTOR_TMP44:.*]] = linkonce_odr hidden unnamed_addr constant { i64, i64, i8*, i8*, i8*, i64 } { i64 0, i64 40, i8* bitcast (void (i8*, i8*)* @__copy_helper_block_8_32s to i8*), i8* bitcast (void (i8*)* @__destroy_helper_block_8_32s to i8*), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @{{.*}}, i32 0, i32 0), i64 256 }, align 8
// CHECK-COMMON: @[[BLOCK_DESCRIPTOR_TMP9:.*]] = linkonce_odr hidden unnamed_addr constant { i64, i64, i8*, i8*, i8*, i64 } { i64 0, i64 40, i8* bitcast (void (i8*, i8*)* @__copy_helper_block_8_32r to i8*), i8* bitcast (void (i8*)* @__destroy_helper_block_8_32r to i8*), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @{{.*}}, i32 0, i32 0), i64 16 }, align 8
// CHECK-COMMON: @[[BLOCK_DESCRIPTOR_TMP46:.*]] = linkonce_odr hidden unnamed_addr constant { i64, i64, i8*, i8*, i8*, i8* } { i64 0, i64 48, i8* bitcast (void (i8*, i8*)* @__copy_helper_block_8_32s to i8*), i8* bitcast (void (i8*)* @__destroy_helper_block_8_32s to i8*), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @{{.*}}, i32 0, i32 0), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @{{.*}}, i32 0, i32 0) }, align 8
// CHECK-COMMON: @[[BLOCK_DESCRIPTOR_TMP48:.*]] = linkonce_odr hidden unnamed_addr constant { i64, i64, i8*, i8*, i8*, i64 } { i64 0, i64 40, i8* bitcast (void (i8*, i8*)* @__copy_helper_block_8_32b to i8*), i8* bitcast (void (i8*)* @__destroy_helper_block_8_32s to i8*), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @{{.*}}, i32 0, i32 0), i64 256 }, align 8

// This shouldn't crash.
void test0(id (^maker)(void)) {
  maker();
}

int (^test1(int x))(void) {
  // CHECK-LABEL:    define{{.*}} i32 ()* @test1(
  // CHECK:      [[X:%.*]] = alloca i32,
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
  // CHECK-NEXT: store i32 {{%.*}}, i32* [[X]]
  // CHECK:      [[T0:%.*]] = bitcast [[BLOCK_T]]* [[BLOCK]] to i32 ()*
  // CHECK-NEXT: [[T1:%.*]] = bitcast i32 ()* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @llvm.objc.retainBlock(i8* [[T1]]) [[NUW:#[0-9]+]]
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to i32 ()*
  // CHECK-NEXT: [[T4:%.*]] = bitcast i32 ()* [[T3]] to i8*
  // CHECK-NEXT: [[T5:%.*]] = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* [[T4]]) [[NUW]]
  // CHECK-NEXT: [[T6:%.*]] = bitcast i8* [[T5]] to i32 ()*
  // CHECK-NEXT: ret i32 ()* [[T6]]
  return ^{ return x; };
}

void test2(id x) {
// CHECK-LABEL:    define{{.*}} void @test2(
// CHECK:      [[X:%.*]] = alloca i8*,
// CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
// CHECK-NEXT: [[PARM:%.*]] = call i8* @llvm.objc.retain(i8* {{%.*}})
// CHECK-NEXT: store i8* [[PARM]], i8** [[X]]
// CHECK:      [[SLOT:%.*]] = getelementptr inbounds [[BLOCK_T]], [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
// CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[X]],
// CHECK-NEXT: [[T1:%.*]] = call i8* @llvm.objc.retain(i8* [[T0]])
// CHECK-NEXT: store i8* [[T1]], i8** [[SLOT]],
// CHECK-NEXT: bitcast
// CHECK-NEXT: call void @test2_helper(
// CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[SLOT]]
// CHECK-NEXT: call void @llvm.objc.release(i8* [[T0]]) [[NUW]], !clang.imprecise_release
// CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[X]]
// CHECK-NEXT: call void @llvm.objc.release(i8* [[T0]]) [[NUW]], !clang.imprecise_release
// CHECK-NEXT: ret void
  extern void test2_helper(id (^)(void));
  test2_helper(^{ return x; });

// CHECK:    define linkonce_odr hidden void @__copy_helper_block_8_32s(i8* %0, i8* %1) unnamed_addr #{{[0-9]+}} {
// CHECK:      [[T0:%.*]] = load i8*, i8**
// CHECK-NEXT: [[SRC:%.*]] = bitcast i8* [[T0]] to [[BLOCK_T]]*
// CHECK-NEXT: [[T0:%.*]] = load i8*, i8**
// CHECK-NEXT: [[DST:%.*]] = bitcast i8* [[T0]] to [[BLOCK_T]]*
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[BLOCK_T]], [[BLOCK_T]]* [[SRC]], i32 0, i32 5
// CHECK-NEXT: [[T1:%.*]] = load i8*, i8** [[T0]]
// CHECK-NEXT: [[T2:%.*]] = call i8* @llvm.objc.retain(i8* [[T1]]) [[NUW]]
// CHECK-NEXT: ret void


// CHECK:    define linkonce_odr hidden void @__destroy_helper_block_8_32s(i8* %0) unnamed_addr #{{[0-9]+}} {
// CHECK:      [[T0:%.*]] = load i8*, i8**
// CHECK-NEXT: [[T1:%.*]] = bitcast i8* [[T0]] to [[BLOCK_T]]*
// CHECK-NEXT: [[T2:%.*]] = getelementptr inbounds [[BLOCK_T]], [[BLOCK_T]]* [[T1]], i32 0, i32 5
// CHECK-NEXT: [[T3:%.*]] = load i8*, i8** [[T2]]
// CHECK-NEXT: call void @llvm.objc.release(i8* [[T3]])
// CHECK-NEXT: ret void
}

void test3(void (^sink)(id*)) {
  __strong id strong;
  sink(&strong);

  // CHECK-LABEL:    define{{.*}} void @test3(
  // CHECK:      [[SINK:%.*]] = alloca void (i8**)*
  // CHECK-NEXT: [[STRONG:%.*]] = alloca i8*
  // CHECK-NEXT: [[TEMP:%.*]] = alloca i8*
  // CHECK-NEXT: bitcast void (i8**)* {{%.*}} to i8*
  // CHECK-NEXT: call i8* @llvm.objc.retain(
  // CHECK-NEXT: bitcast i8*
  // CHECK-NEXT: store void (i8**)* {{%.*}}, void (i8**)** [[SINK]]
  // CHECK-NEXT: [[STRONGPTR1:%.*]] = bitcast i8** [[STRONG]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 8, i8* [[STRONGPTR1]])
  // CHECK-NEXT: store i8* null, i8** [[STRONG]]

  // CHECK-NEXT: load void (i8**)*, void (i8**)** [[SINK]]
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: [[BLOCK:%.*]] = bitcast
  // CHECK-NEXT: [[V:%.*]] = load i8*, i8** [[STRONG]]
  // CHECK-NEXT: store i8* [[V]], i8** [[TEMP]]
  // CHECK-NEXT: [[F0:%.*]] = load i8*, i8**
  // CHECK-NEXT: [[F1:%.*]] = bitcast i8* [[F0]] to void (i8*, i8**)*
  // CHECK-NEXT: call void [[F1]](i8* [[BLOCK]], i8** [[TEMP]])
  // CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[TEMP]]
  // CHECK-NEXT: [[T1:%.*]] = call i8* @llvm.objc.retain(i8* [[T0]])
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.use(i8* [[V]]) [[NUW]]
  // CHECK-NEXT: [[T2:%.*]] = load i8*, i8** [[STRONG]]
  // CHECK-NEXT: store i8* [[T1]], i8** [[STRONG]]
  // CHECK-NEXT: call void @llvm.objc.release(i8* [[T2]])

  // CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[STRONG]]
  // CHECK-NEXT: call void @llvm.objc.release(i8* [[T0]])
  // CHECK-NEXT: [[STRONGPTR2:%.*]] = bitcast i8** [[STRONG]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* [[STRONGPTR2]])

  // CHECK-NEXT: load void (i8**)*, void (i8**)** [[SINK]]
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: call void @llvm.objc.release
  // CHECK-NEXT: ret void

}

void test4(void) {
  id test4_source(void);
  void test4_helper(void (^)(void));
  __block id var = test4_source();
  test4_helper(^{ var = 0; });

  // CHECK-LABEL:    define{{.*}} void @test4()
  // CHECK:      [[VAR:%.*]] = alloca [[BYREF_T:%.*]],
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BYREF_T]], [[BYREF_T]]* [[VAR]], i32 0, i32 2
  // 0x02000000 - has copy/dispose helpers strong
  // CHECK-NEXT: store i32 838860800, i32* [[T0]]
  // CHECK:      [[SLOT:%.*]] = getelementptr inbounds [[BYREF_T]], [[BYREF_T]]* [[VAR]], i32 0, i32 6
  // CHECK-NEXT: [[T0:%.*]] = call i8* @test4_source()
  // CHECK-NEXT: [[T1:%.*]] = notail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* [[T0]])
  // CHECK-NEXT: store i8* [[T1]], i8** [[SLOT]]
  // CHECK-NEXT: [[SLOT:%.*]] = getelementptr inbounds [[BYREF_T]], [[BYREF_T]]* [[VAR]], i32 0, i32 6
  // 0x42800000 - has signature, copy/dispose helpers, as well as BLOCK_HAS_EXTENDED_LAYOUT
  // CHECK:      store i32 -1040187392,
  // CHECK:      [[T0:%.*]] = bitcast [[BYREF_T]]* [[VAR]] to i8*
  // CHECK-NEXT: store i8* [[T0]], i8**
  // CHECK:      call void @test4_helper(
  // CHECK:      [[T0:%.*]] = bitcast [[BYREF_T]]* [[VAR]] to i8*
  // CHECK-NEXT: call void @_Block_object_dispose(i8* [[T0]], i32 8)
  // CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[SLOT]]
  // CHECK-NEXT: call void @llvm.objc.release(i8* [[T0]])
  // CHECK: ret void

  // CHECK-LABEL:    define internal void @__Block_byref_object_copy_(i8* %0, i8* %1) #{{[0-9]+}} {
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BYREF_T]], [[BYREF_T]]* {{%.*}}, i32 0, i32 6
  // CHECK-NEXT: load i8*, i8**
  // CHECK-NEXT: bitcast i8* {{%.*}} to [[BYREF_T]]*
  // CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[BYREF_T]], [[BYREF_T]]* {{%.*}}, i32 0, i32 6
  // CHECK-NEXT: [[T2:%.*]] = load i8*, i8** [[T1]]
  // CHECK-NEXT: store i8* [[T2]], i8** [[T0]]
  // CHECK-NEXT: store i8* null, i8** [[T1]]

  // CHECK-LABEL:    define internal void @__Block_byref_object_dispose_(i8* %0) #{{[0-9]+}} {
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BYREF_T]], [[BYREF_T]]* {{%.*}}, i32 0, i32 6
  // CHECK-NEXT: [[T1:%.*]] = load i8*, i8** [[T0]]
  // CHECK-NEXT: call void @llvm.objc.release(i8* [[T1]])

  // CHECK-LABEL:    define internal void @__test4_block_invoke
  // CHECK:      [[SLOT:%.*]] = getelementptr inbounds {{.*}}, i32 0, i32 6
  // CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[SLOT]], align 8
  // CHECK-NEXT: store i8* null, i8** [[SLOT]],
  // CHECK-NEXT: call void @llvm.objc.release(i8* [[T0]])
  // CHECK-NEXT: ret void

  // CHECK-LABEL:    define linkonce_odr hidden void @__copy_helper_block_8_32r(i8* %0, i8* %1) unnamed_addr #{{[0-9]+}} {
  // CHECK:      call void @_Block_object_assign(i8* {{%.*}}, i8* {{%.*}}, i32 8)

  // CHECK-LABEL:    define linkonce_odr hidden void @__destroy_helper_block_8_32r(i8* %0) unnamed_addr #{{[0-9]+}} {
  // CHECK:      call void @_Block_object_dispose(i8* {{%.*}}, i32 8)
}

void test5(void) {
  extern id test5_source(void);
  void test5_helper(void (^)(void));
  __unsafe_unretained id var = test5_source();
  test5_helper(^{ (void) var; });

  // CHECK-LABEL:    define{{.*}} void @test5()
  // CHECK:      [[VAR:%.*]] = alloca i8*
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
  // CHECK-NEXT: [[VARPTR1:%.*]] = bitcast i8** [[VAR]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 8, i8* [[VARPTR1]])
  // CHECK: [[T0:%.*]] = call i8* @test5_source()
  // CHECK-NEXT: [[T1:%.*]] = notail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* [[T0]])
  // CHECK-NEXT: store i8* [[T1]], i8** [[VAR]],
  // CHECK-NEXT: call void @llvm.objc.release(i8* [[T1]])
  // 0x40800000 - has signature but no copy/dispose, as well as BLOCK_HAS_EXTENDED_LAYOUT
  // CHECK:      store i32 -1073741824, i32*
  // CHECK:      [[CAPTURE:%.*]] = getelementptr inbounds [[BLOCK_T]], [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
  // CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[VAR]]
  // CHECK-NEXT: store i8* [[T0]], i8** [[CAPTURE]]
  // CHECK-NEXT: [[T0:%.*]] = bitcast [[BLOCK_T]]* [[BLOCK]] to
  // CHECK: call void @test5_helper
  // CHECK-NEXT: [[VARPTR2:%.*]] = bitcast i8** [[VAR]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* [[VARPTR2]])
  // CHECK-NEXT: ret void
}

void test6(void) {
  id test6_source(void);
  void test6_helper(void (^)(void));
  __block __weak id var = test6_source();
  test6_helper(^{ var = 0; });

  // CHECK-LABEL:    define{{.*}} void @test6()
  // CHECK:      [[VAR:%.*]] = alloca [[BYREF_T:%.*]],
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
  // CHECK-NEXT: [[VARPTR1:%.*]] = bitcast [[BYREF_T]]* [[VAR]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 48, i8* [[VARPTR1]])
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BYREF_T]], [[BYREF_T]]* [[VAR]], i32 0, i32 2
  // 0x02000000 - has copy/dispose helpers weak
  // CHECK-NEXT: store i32 1107296256, i32* [[T0]]
  // CHECK:      [[SLOT:%.*]] = getelementptr inbounds [[BYREF_T]], [[BYREF_T]]* [[VAR]], i32 0, i32 6
  // CHECK-NEXT: [[T0:%.*]] = call i8* @test6_source()
  // CHECK-NEXT: [[T1:%.*]] = notail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* [[T0]])
  // CHECK-NEXT: call i8* @llvm.objc.initWeak(i8** [[SLOT]], i8* [[T1]])
  // CHECK-NEXT: call void @llvm.objc.release(i8* [[T1]])
  // CHECK-NEXT: [[SLOT:%.*]] = getelementptr inbounds [[BYREF_T]], [[BYREF_T]]* [[VAR]], i32 0, i32 6
  // 0x42800000 - has signature, copy/dispose helpers, as well as BLOCK_HAS_EXTENDED_LAYOUT
  // CHECK:      store i32 -1040187392,
  // CHECK: %[[BLOCK_DESCRIPTOR:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>* %{{.*}}, i32 0, i32 4
// CHECK: store %[[STRUCT_BLOCK_DESCRIPTOR]]* bitcast ({ i64, i64, i8*, i8*, i8*, i64 }* @[[BLOCK_DESCRIPTOR_TMP9]] to %[[STRUCT_BLOCK_DESCRIPTOR]]*), %[[STRUCT_BLOCK_DESCRIPTOR]]** %[[BLOCK_DESCRIPTOR]], align 8
  // CHECK:      [[T0:%.*]] = bitcast [[BYREF_T]]* [[VAR]] to i8*
  // CHECK-NEXT: store i8* [[T0]], i8**
  // CHECK:      call void @test6_helper(
  // CHECK:      [[T0:%.*]] = bitcast [[BYREF_T]]* [[VAR]] to i8*
  // CHECK-NEXT: call void @_Block_object_dispose(i8* [[T0]], i32 8)
  // CHECK-NEXT: call void @llvm.objc.destroyWeak(i8** [[SLOT]])
  // CHECK-NEXT: [[VARPTR2:%.*]] = bitcast [[BYREF_T]]* [[VAR]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 48, i8* [[VARPTR2]])
  // CHECK-NEXT: ret void

  // CHECK-LABEL:    define internal void @__Block_byref_object_copy_.{{[0-9]+}}(i8* %0, i8* %1) #{{[0-9]+}} {
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BYREF_T]], [[BYREF_T]]* {{%.*}}, i32 0, i32 6
  // CHECK-NEXT: load i8*, i8**
  // CHECK-NEXT: bitcast i8* {{%.*}} to [[BYREF_T]]*
  // CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[BYREF_T]], [[BYREF_T]]* {{%.*}}, i32 0, i32 6
  // CHECK-NEXT: call void @llvm.objc.moveWeak(i8** [[T0]], i8** [[T1]])

  // CHECK-LABEL:    define internal void @__Block_byref_object_dispose_.{{[0-9]+}}(i8* %0) #{{[0-9]+}} {
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BYREF_T]], [[BYREF_T]]* {{%.*}}, i32 0, i32 6
  // CHECK-NEXT: call void @llvm.objc.destroyWeak(i8** [[T0]])

  // CHECK-LABEL:    define internal void @__test6_block_invoke
  // CHECK:      [[SLOT:%.*]] = getelementptr inbounds {{.*}}, i32 0, i32 6
  // CHECK-NEXT: call i8* @llvm.objc.storeWeak(i8** [[SLOT]], i8* null)
  // CHECK-NEXT: ret void
}

void test7(void) {
  id test7_source(void);
  void test7_helper(void (^)(void));
  void test7_consume(id);
  __weak id var = test7_source();
  test7_helper(^{ test7_consume(var); });

  // CHECK-LABEL:    define{{.*}} void @test7()
  // CHECK:      [[VAR:%.*]] = alloca i8*,
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
  // CHECK:      [[T0:%.*]] = call i8* @test7_source()
  // CHECK-NEXT: [[T1:%.*]] = notail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* [[T0]])
  // CHECK-NEXT: call i8* @llvm.objc.initWeak(i8** [[VAR]], i8* [[T1]])
  // CHECK-NEXT: call void @llvm.objc.release(i8* [[T1]])
  // 0x42800000 - has signature, copy/dispose helpers, as well as BLOCK_HAS_EXTENDED_LAYOUT
  // CHECK:      store i32 -1040187392,
  // CHECK:      [[SLOT:%.*]] = getelementptr inbounds [[BLOCK_T]], [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
  // CHECK-NEXT: call void @llvm.objc.copyWeak(i8** [[SLOT]], i8** [[VAR]])
  // CHECK:      call void @test7_helper(
  // CHECK-NEXT: call void @llvm.objc.destroyWeak(i8** {{%.*}})
  // CHECK-NEXT: call void @llvm.objc.destroyWeak(i8** [[VAR]])
  // CHECK: ret void

  // CHECK-LABEL:    define internal void @__test7_block_invoke
  // CHECK:      [[SLOT:%.*]] = getelementptr inbounds [[BLOCK_T]], [[BLOCK_T]]* {{%.*}}, i32 0, i32 5
  // CHECK-NEXT: [[T0:%.*]] = call i8* @llvm.objc.loadWeakRetained(i8** [[SLOT]])
  // CHECK-NEXT: call void @test7_consume(i8* [[T0]])
  // CHECK-NEXT: call void @llvm.objc.release(i8* [[T0]])
  // CHECK: ret void

  // CHECK-LABEL:    define linkonce_odr hidden void @__copy_helper_block_8_32w(i8* %0, i8* %1) unnamed_addr #{{[0-9]+}} {
  // CHECK:      getelementptr
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: call void @llvm.objc.copyWeak(

  // CHECK-LABEL:    define linkonce_odr hidden void @__destroy_helper_block_8_32w(i8* %0) unnamed_addr #{{[0-9]+}} {
  // CHECK:      getelementptr
  // CHECK-NEXT: call void @llvm.objc.destroyWeak(
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
// CHECK:      [[T0:%.*]] = getelementptr inbounds [[BLOCK_T]], [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
// CHECK-NEXT: [[T1:%.*]] = load [[TEST8]]*, [[TEST8]]** [[SELF]],
// CHECK-NEXT: store %0* [[T1]], %0** [[T0]]
// CHECK-NEXT: bitcast [[BLOCK_T]]* [[BLOCK]] to
// CHECK: call void @test8_helper(
// CHECK-NEXT: [[T2:%.*]] = load [[TEST8]]*, [[TEST8]]** [[T0]]
// CHECK-NEXT: call void (...) @llvm.objc.clang.arc.use([[TEST8]]* [[T2]])
// CHECK: ret void

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

// CHECK-LABEL:    define{{.*}} i8* @test9(
// CHECK:      load i8*, i8** getelementptr
// CHECK-NEXT: bitcast i8*
// CHECK-NEXT: call i8* 
// CHECK-NEXT: tail call i8* @llvm.objc.autoreleaseReturnValue
// CHECK-NEXT: ret i8*

// CHECK:      call i8* @test9_produce()
// CHECK-NEXT: call i8* @llvm.objc.retain
// CHECK-NEXT: ret i8*
}

// rdar://problem/9814099
// Test that we correctly initialize __block variables
// when the initialization captures the variable.
void test10a(void) {
  __block void (^block)(void) = ^{ block(); };
  // CHECK-LABEL:    define{{.*}} void @test10a()
  // CHECK:      [[BYREF:%.*]] = alloca [[BYREF_T:%.*]],
  // CHECK:      [[BLOCK1:%.*]] = alloca <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>, align 8

  // Zero-initialization before running the initializer.
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BYREF_T]], [[BYREF_T]]* [[BYREF]], i32 0, i32 6
  // CHECK-NEXT: store void ()* null, void ()** [[T0]], align 8

  // Run the initializer as an assignment.
  // CHECK:      [[T2:%.*]] = bitcast <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>* [[BLOCK1]] to void ()*
  // CHECK-NEXT: [[T3:%.*]] = getelementptr inbounds [[BYREF_T]], [[BYREF_T]]* [[BYREF]], i32 0, i32 1
  // CHECK-NEXT: [[T4:%.*]] = load [[BYREF_T]]*, [[BYREF_T]]** [[T3]]
  // CHECK-NEXT: [[T5:%.*]] = getelementptr inbounds [[BYREF_T]], [[BYREF_T]]* [[T4]], i32 0, i32 6
  // CHECK-NEXT: [[T6:%.*]] = load void ()*, void ()** [[T5]], align 8
  // CHECK-NEXT: store void ()* [[T2]], void ()** [[T5]], align 8
  // CHECK-NEXT: [[T7:%.*]] = bitcast void ()* [[T6]] to i8*
  // CHECK-NEXT: call void @llvm.objc.release(i8* [[T7]])

  // Destroy at end of function.
  // CHECK-NEXT: [[SLOT:%.*]] = getelementptr inbounds [[BYREF_T]], [[BYREF_T]]* [[BYREF]], i32 0, i32 6
  // CHECK-NEXT: [[T0:%.*]] = bitcast [[BYREF_T]]* [[BYREF]] to i8*
  // CHECK-NEXT: call void @_Block_object_dispose(i8* [[T0]], i32 8)
  // CHECK-NEXT: [[T1:%.*]] = load void ()*, void ()** [[SLOT]]
  // CHECK-NEXT: [[T2:%.*]] = bitcast void ()* [[T1]] to i8*
  // CHECK-NEXT: call void @llvm.objc.release(i8* [[T2]])
  // CHECK: ret void
}

// <rdar://problem/10402698>: do this copy and dispose with
// objc_retainBlock/release instead of _Block_object_assign/destroy.
// We can also use _Block_object_assign/destroy with
// BLOCK_FIELD_IS_BLOCK as long as we don't pass BLOCK_BYREF_CALLER.

// CHECK-LABEL: define internal void @__Block_byref_object_copy_.{{[0-9]+}}(i8* %0, i8* %1) #{{[0-9]+}} {
// CHECK:      [[D0:%.*]] = load i8*, i8** {{%.*}}
// CHECK-NEXT: [[D1:%.*]] = bitcast i8* [[D0]] to [[BYREF_T]]*
// CHECK-NEXT: [[D2:%.*]] = getelementptr inbounds [[BYREF_T]], [[BYREF_T]]* [[D1]], i32 0, i32 6
// CHECK-NEXT: [[S0:%.*]] = load i8*, i8** {{%.*}}
// CHECK-NEXT: [[S1:%.*]] = bitcast i8* [[S0]] to [[BYREF_T]]*
// CHECK-NEXT: [[S2:%.*]] = getelementptr inbounds [[BYREF_T]], [[BYREF_T]]* [[S1]], i32 0, i32 6
// CHECK-NEXT: [[T0:%.*]] = load void ()*, void ()** [[S2]], align 8
// CHECK-NEXT: [[T1:%.*]] = bitcast void ()* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = call i8* @llvm.objc.retainBlock(i8* [[T1]])
// CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to void ()*
// CHECK-NEXT: store void ()* [[T3]], void ()** [[D2]], align 8
// CHECK: ret void

// CHECK-LABEL: define internal void @__Block_byref_object_dispose_.{{[0-9]+}}(i8* %0) #{{[0-9]+}} {
// CHECK:      [[T0:%.*]] = load i8*, i8** {{%.*}}
// CHECK-NEXT: [[T1:%.*]] = bitcast i8* [[T0]] to [[BYREF_T]]*
// CHECK-NEXT: [[T2:%.*]] = getelementptr inbounds [[BYREF_T]], [[BYREF_T]]* [[T1]], i32 0, i32 6
// CHECK-NEXT: [[T3:%.*]] = load void ()*, void ()** [[T2]]
// CHECK-NEXT: [[T4:%.*]] = bitcast void ()* [[T3]] to i8*
// CHECK-NEXT: call void @llvm.objc.release(i8* [[T4]])
// CHECK-NEXT: ret void

// Test that we correctly assign to __block variables when the
// assignment captures the variable.
void test10b(void) {
  __block void (^block)(void);
  block = ^{ block(); };

  // CHECK-LABEL:    define{{.*}} void @test10b()
  // CHECK:      [[BYREF:%.*]] = alloca [[BYREF_T:%.*]],
  // CHECK:      [[BLOCK3:%.*]] = alloca <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>, align 8

  // Zero-initialize.
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BYREF_T]], [[BYREF_T]]* [[BYREF]], i32 0, i32 6
  // CHECK-NEXT: store void ()* null, void ()** [[T0]], align 8

  // CHECK-NEXT: [[SLOT:%.*]] = getelementptr inbounds [[BYREF_T]], [[BYREF_T]]* [[BYREF]], i32 0, i32 6

  // The assignment.
  // CHECK:      [[T2:%.*]] = bitcast <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>* [[BLOCK3]] to void ()*
  // CHECK-NEXT: [[T3:%.*]] = getelementptr inbounds [[BYREF_T]], [[BYREF_T]]* [[BYREF]], i32 0, i32 1
  // CHECK-NEXT: [[T4:%.*]] = load [[BYREF_T]]*, [[BYREF_T]]** [[T3]]
  // CHECK-NEXT: [[T5:%.*]] = getelementptr inbounds [[BYREF_T]], [[BYREF_T]]* [[T4]], i32 0, i32 6
  // CHECK-NEXT: [[T6:%.*]] = load void ()*, void ()** [[T5]], align 8
  // CHECK-NEXT: store void ()* [[T2]], void ()** [[T5]], align 8
  // CHECK-NEXT: [[T7:%.*]] = bitcast void ()* [[T6]] to i8*
  // CHECK-NEXT: call void @llvm.objc.release(i8* [[T7]])

  // Destroy at end of function.
  // CHECK-NEXT: [[T0:%.*]] = bitcast [[BYREF_T]]* [[BYREF]] to i8*
  // CHECK-NEXT: call void @_Block_object_dispose(i8* [[T0]], i32 8)
  // CHECK-NEXT: [[T1:%.*]] = load void ()*, void ()** [[SLOT]]
  // CHECK-NEXT: [[T2:%.*]] = bitcast void ()* [[T1]] to i8*
  // CHECK-NEXT: call void @llvm.objc.release(i8* [[T2]])
  // CHECK: ret void
}

// rdar://problem/10088932
void test11_helper(id);
void test11a(void) {
  int x;
  test11_helper(^{ (void) x; });

  // CHECK-LABEL:    define{{.*}} void @test11a()
  // CHECK:      [[X:%.*]] = alloca i32, align 4
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]], align 8
  // CHECK:      [[T0:%.*]] = bitcast [[BLOCK_T]]* [[BLOCK]] to void ()*
  // CHECK-NEXT: [[T1:%.*]] = bitcast void ()* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @llvm.objc.retainBlock(i8* [[T1]])
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to void ()*
  // CHECK-NEXT: [[T4:%.*]] = bitcast void ()* [[T3]] to i8*
  // CHECK-NEXT: call void @test11_helper(i8* [[T4]])
  // CHECK-NEXT: [[T5:%.*]] = bitcast void ()* [[T3]] to i8*
  // CHECK-NEXT: call void @llvm.objc.release(i8* [[T5]])
  // CHECK: ret void
}
void test11b(void) {
  int x;
  id b = ^{ (void) x; };

  // CHECK-LABEL:    define{{.*}} void @test11b()
  // CHECK:      [[X:%.*]] = alloca i32, align 4
  // CHECK-NEXT: [[B:%.*]] = alloca i8*, align 8
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]], align 8
  // CHECK:      [[T0:%.*]] = bitcast [[BLOCK_T]]* [[BLOCK]] to void ()*
  // CHECK-NEXT: [[T1:%.*]] = bitcast void ()* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @llvm.objc.retainBlock(i8* [[T1]])
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to void ()*
  // CHECK-NEXT: [[T4:%.*]] = bitcast void ()* [[T3]] to i8*
  // CHECK-NEXT: store i8* [[T4]], i8** [[B]], align 8
  // CHECK-NEXT: [[T5:%.*]] = load i8*, i8** [[B]]
  // CHECK-NEXT: call void @llvm.objc.release(i8* [[T5]])
  // CHECK: ret void
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
// CHECK:    %add.ptr = getelementptr inbounds i8, i8* %1, i64 %ivar

// CHECK:    define internal void @"\01-[Test12 setNblock:]"(
// CHECK:    call void @objc_setProperty(i8* {{%.*}}, i8* {{%.*}}, i64 {{%.*}}, i8* {{%.*}}, i1 zeroext false, i1 zeroext true)
@end

// rdar://problem/10131784
void test13(id x) {
  extern void test13_helper(id);
  extern void test13_use(void(^)(void));

  void (^b)(void) = (x ? ^{test13_helper(x);} : 0);
  test13_use(b);

  // CHECK-LABEL:    define{{.*}} void @test13(
  // CHECK:      [[X:%.*]] = alloca i8*, align 8
  // CHECK-NEXT: [[B:%.*]] = alloca void ()*, align 8
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:.*]], align 8
  // CHECK-NEXT: [[CLEANUP_ACTIVE:%.*]] = alloca i1
  // CHECK-NEXT: [[COND_CLEANUP_SAVE:%.*]] = alloca i8**,
  // CHECK-NEXT: [[T0:%.*]] = call i8* @llvm.objc.retain(i8* {{%.*}})
  // CHECK-NEXT: store i8* [[T0]], i8** [[X]], align 8
  // CHECK-NEXT: [[BPTR1:%.*]] = bitcast void ()** [[B]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 8, i8* [[BPTR1]])
  // CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[X]], align 8
  // CHECK-NEXT: [[T1:%.*]] = icmp ne i8* [[T0]], null
  // CHECK-NEXT: store i1 false, i1* [[CLEANUP_ACTIVE]]
  // CHECK-NEXT: br i1 [[T1]],

  // CHECK-NOT:  br
  // CHECK:      [[CAPTURE:%.*]] = getelementptr inbounds [[BLOCK_T]], [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
  // CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[X]], align 8
  // CHECK-NEXT: [[T1:%.*]] = call i8* @llvm.objc.retain(i8* [[T0]])
  // CHECK-NEXT: store i8* [[T1]], i8** [[CAPTURE]], align 8
  // CHECK-NEXT: store i1 true, i1* [[CLEANUP_ACTIVE]]
  // CHECK-NEXT: store i8** [[CAPTURE]], i8*** [[COND_CLEANUP_SAVE]], align 8
  // CHECK-NEXT: bitcast [[BLOCK_T]]* [[BLOCK]] to void ()*
  // CHECK-NEXT: br label
  // CHECK:      br label
  // CHECK:      [[T0:%.*]] = phi void ()*
  // CHECK-NEXT: [[T1:%.*]] = bitcast void ()* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @llvm.objc.retainBlock(i8* [[T1]])
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to void ()*
  // CHECK-NEXT: store void ()* [[T3]], void ()** [[B]], align 8
  // CHECK-NEXT: [[T0:%.*]] = load void ()*, void ()** [[B]], align 8
  // CHECK-NEXT: call void @test13_use(void ()* [[T0]])
  // CHECK-NEXT: [[T0:%.*]] = load void ()*, void ()** [[B]]
  // CHECK-NEXT: [[T1:%.*]] = bitcast void ()* [[T0]] to i8*
  // CHECK-NEXT: call void @llvm.objc.release(i8* [[T1]])

  // CHECK-NEXT: [[T0:%.*]] = load i1, i1* [[CLEANUP_ACTIVE]]
  // CHECK-NEXT: br i1 [[T0]]
  // CHECK:      [[V12:%.*]] = load i8**, i8*** [[COND_CLEANUP_SAVE]], align 8
  // CHECK:      [[T0:%.*]] = load i8*, i8** [[V12]]
  // CHECK-NEXT: call void @llvm.objc.release(i8* [[T0]])
  // CHECK-NEXT: br label

  // CHECK:      [[BPTR2:%.*]] = bitcast void ()** [[B]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* [[BPTR2]])
  // CHECK-NEXT:      [[T0:%.*]] = load i8*, i8** [[X]]
  // CHECK-NEXT: call void @llvm.objc.release(i8* [[T0]])
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

  // CHECK-LABEL: define{{.*}} void @test16(
  // CHECK: [[BLKVAR:%.*]]  = alloca void ()*, align 8
  // CHECK-NEXT:  [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
  // CHECK-NEXT:  [[BLKVARPTR1:%.*]] = bitcast void ()** [[BLKVAR]] to i8*
  // CHECK-NEXT:  call void @llvm.lifetime.start.p0i8(i64 8, i8* [[BLKVARPTR1]])
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
// CHECK-LABEL:    define{{.*}} i8* ()* @test17(
// CHECK:      [[RET:%.*]] = alloca i8* ()*, align
// CHECK-NEXT: [[SELF:%.*]] = alloca i8*,
// CHECK:      [[B0:%.*]] = alloca [[BLOCK:<.*>]], align
// CHECK:      [[B1:%.*]] = alloca [[BLOCK]], align
// CHECK:      [[T0:%.*]] = call i8* @llvm.objc.retain(i8*
// CHECK-NEXT: store i8* [[T0]], i8** [[SELF]], align
// CHECK-NOT:  objc_retain
// CHECK-NOT:  objc_release
// CHECK:      [[CAPTURED:%.*]] = getelementptr inbounds [[BLOCK]], [[BLOCK]]* [[B0]], i32 0, i32 5
// CHECK-NEXT: [[T1:%.*]] = load i8*, i8** [[SELF]], align
// CHECK-NEXT: [[T2:%.*]] = call i8* @llvm.objc.retain(i8* [[T1]])
// CHECK-NEXT: store i8* [[T2]], i8** [[CAPTURED]],
// CHECK-NEXT: [[T0:%.*]] = bitcast [[BLOCK]]* [[B0]] to i8* ()*
// CHECK-NEXT: [[T1:%.*]] = bitcast i8* ()* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = call i8* @llvm.objc.retainBlock(i8* [[T1]])
// CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to i8* ()*
// CHECK-NEXT: store i8* ()* [[T3]], i8* ()** [[RET]]
// CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[CAPTURED]]
// CHECK-NEXT: call void @llvm.objc.release(i8* [[T0]])
// CHECK-NEXT: store i32
// CHECK-NEXT: br label
// CHECK-NOT:  objc_retain
// CHECK-NOT:  objc_release
// CHECK:      [[CAPTURED:%.*]] = getelementptr inbounds [[BLOCK]], [[BLOCK]]* [[B1]], i32 0, i32 5
// CHECK-NEXT: [[T1:%.*]] = load i8*, i8** [[SELF]], align
// CHECK-NEXT: [[T2:%.*]] = call i8* @llvm.objc.retain(i8* [[T1]])
// CHECK-NEXT: store i8* [[T2]], i8** [[CAPTURED]],
// CHECK-NEXT: [[T0:%.*]] = bitcast [[BLOCK]]* [[B1]] to i8* ()*
// CHECK-NEXT: [[T1:%.*]] = bitcast i8* ()* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = call i8* @llvm.objc.retainBlock(i8* [[T1]])
// CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to i8* ()*
// CHECK-NEXT: store i8* ()* [[T3]], i8* ()** [[RET]]
// CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[CAPTURED]]
// CHECK-NEXT: call void @llvm.objc.release(i8* [[T0]])
// CHECK-NEXT: store i32
// CHECK-NEXT: br label

void test18(id x) {
// CHECK-UNOPT-LABEL:    define{{.*}} void @test18(
// CHECK-UNOPT:      [[X:%.*]] = alloca i8*,
// CHECK-UNOPT-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
// CHECK-UNOPT-NEXT: store i8* null, i8** [[X]]
// CHECK-UNOPT-NEXT: call void @llvm.objc.storeStrong(i8** [[X]], 
// CHECK-UNOPT: %[[BLOCK_DESCRIPTOR:.*]] = getelementptr inbounds [[BLOCK_T]], [[BLOCK_T]]* [[BLOCK]], i32 0, i32 4
// CHECK-UNOPT: store %[[STRUCT_BLOCK_DESCRIPTOR]]* bitcast ({ i64, i64, i8*, i8*, i8*, i64 }* @[[BLOCK_DESCRIPTOR_TMP44]] to %[[STRUCT_BLOCK_DESCRIPTOR]]*), %[[STRUCT_BLOCK_DESCRIPTOR]]** %[[BLOCK_DESCRIPTOR]], align 8
// CHECK-UNOPT:      [[SLOT:%.*]] = getelementptr inbounds [[BLOCK_T]], [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
// CHECK-UNOPT-NEXT: [[T0:%.*]] = load i8*, i8** [[X]],
// CHECK-UNOPT-NEXT: [[T1:%.*]] = call i8* @llvm.objc.retain(i8* [[T0]])
// CHECK-UNOPT-NEXT: store i8* [[T1]], i8** [[SLOT]],
// CHECK-UNOPT-NEXT: bitcast
// CHECK-UNOPT-NEXT: call void @test18_helper(
// CHECK-UNOPT-NEXT: call void @llvm.objc.storeStrong(i8** [[SLOT]], i8* null) [[NUW:#[0-9]+]]
// CHECK-UNOPT-NEXT: call void @llvm.objc.storeStrong(i8** [[X]], i8* null) [[NUW]]
// CHECK-UNOPT-NEXT: ret void
  extern void test18_helper(id (^)(void));
  test18_helper(^{ return x; });
}

// Ensure that we don't emit helper code in copy/dispose routines for variables
// that are const-captured.
void testUnsafeUnretainedLifetimeInCopyAndDestroyHelpers(id x, id y) {
  id __unsafe_unretained unsafeObject = x;
  (^ { testUnsafeUnretainedLifetimeInCopyAndDestroyHelpers(x, unsafeObject); })();
}

// CHECK-LABEL: define{{.*}} void @testUnsafeUnretainedLifetimeInCopyAndDestroyHelpers
// %[[BLOCK_DESCRIPTOR:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8* }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8* }>* %{{.*}}, i32 0, i32 4
// CHECK: store %[[STRUCT_BLOCK_DESCRIPTOR]]* bitcast ({ i64, i64, i8*, i8*, i8*, i8* }* @[[BLOCK_DESCRIPTOR_TMP46]] to %[[STRUCT_BLOCK_DESCRIPTOR]]*), %[[STRUCT_BLOCK_DESCRIPTOR]]** %[[BLOCK_DESCRIPTOR]], align 8

// CHECK-LABEL: define internal void @__testUnsafeUnretainedLifetimeInCopyAndDestroyHelpers_block_invoke
// CHECK-UNOPT-LABEL: define internal void @__testUnsafeUnretainedLifetimeInCopyAndDestroyHelpers_block_invoke

// rdar://13588325
void test19_sink(void (^)(int));
void test19(void (^b)(void)) {
// CHECK-LABEL:    define{{.*}} void @test19(
//   Prologue.
// CHECK:      [[B:%.*]] = alloca void ()*,
// CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
// CHECK-NEXT: [[T0:%.*]] = bitcast void ()* {{%.*}} to i8*
// CHECK-NEXT: [[T1:%.*]] = call i8* @llvm.objc.retain(i8* [[T0]])
// CHECK-NEXT: [[T2:%.*]] = bitcast i8* [[T1]] to void ()*
// CHECK-NEXT: store void ()* [[T2]], void ()** [[B]]

//   Block setup.  We skip most of this.  Note the bare retain.
// CHECK: %[[BLOCK_DESCRIPTOR:.*]] = getelementptr inbounds [[BLOCK_T]], [[BLOCK_T]]* [[BLOCK]], i32 0, i32 4
// CHECK: store %[[STRUCT_BLOCK_DESCRIPTOR]]* bitcast ({ i64, i64, i8*, i8*, i8*, i64 }* @[[BLOCK_DESCRIPTOR_TMP48]] to %[[STRUCT_BLOCK_DESCRIPTOR]]*), %[[STRUCT_BLOCK_DESCRIPTOR]]** %[[BLOCK_DESCRIPTOR]], align 8
// CHECK:      [[SLOT:%.*]] = getelementptr inbounds [[BLOCK_T]], [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
// CHECK-NEXT: [[T0:%.*]] = load void ()*, void ()** [[B]],
// CHECK-NEXT: [[T1:%.*]] = bitcast void ()* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = call i8* @llvm.objc.retain(i8* [[T1]])
// CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to void ()*
// CHECK-NEXT: store void ()* [[T3]], void ()** [[SLOT]],
//   Call.
// CHECK-NEXT: [[T0:%.*]] = bitcast [[BLOCK_T]]* [[BLOCK]] to void (i32)*
// CHECK-NEXT: call void @test19_sink(void (i32)* [[T0]])

  test19_sink(^(int x) { b(); });

//   Block teardown.
// CHECK-NEXT: [[T0:%.*]] = load void ()*, void ()** [[SLOT]]
// CHECK-NEXT: [[T1:%.*]] = bitcast void ()* [[T0]] to i8*
// CHECK-NEXT: call void @llvm.objc.release(i8* [[T1]])

//   Local cleanup.
// CHECK-NEXT: [[T0:%.*]] = load void ()*, void ()** [[B]]
// CHECK-NEXT: [[T1:%.*]] = bitcast void ()* [[T0]] to i8*
// CHECK-NEXT: call void @llvm.objc.release(i8* [[T1]])

// CHECK-NEXT: ret void
}

// CHECK-LABEL: define{{.*}} void @test20(
// CHECK: [[XADDR:%.*]] = alloca i8*
// CHECK-NEXT: [[BLOCK:%.*]] = alloca <[[BLOCKTY:.*]]>
// CHECK-NEXT: [[RETAINEDX:%.*]] = call i8* @llvm.objc.retain(i8* %{{.*}})
// CHECK-NEXT: store i8* [[RETAINEDX]], i8** [[XADDR]]
// CHECK: [[BLOCKCAPTURED:%.*]] = getelementptr inbounds <[[BLOCKTY]]>, <[[BLOCKTY]]>* [[BLOCK]], i32 0, i32 5
// CHECK: [[CAPTURED:%.*]] = load i8*, i8** [[XADDR]]
// CHECK: store i8* [[CAPTURED]], i8** [[BLOCKCAPTURED]]
// CHECK: [[CAPTURE:%.*]] = load i8*, i8** [[BLOCKCAPTURED]]
// CHECK-NEXT: call void (...) @llvm.objc.clang.arc.use(i8* [[CAPTURE]])
// CHECK-NEXT: [[X:%.*]] = load i8*, i8** [[XADDR]]
// CHECK-NEXT: call void @llvm.objc.release(i8* [[X]])
// CHECK-NEXT: ret void

// CHECK-UNOPT-LABEL: define{{.*}} void @test20(
// CHECK-UNOPT: [[XADDR:%.*]] = alloca i8*
// CHECK-UNOPT-NEXT: [[BLOCK:%.*]] = alloca <[[BLOCKTY:.*]]>
// CHECK-UNOPT: [[BLOCKCAPTURED:%.*]] = getelementptr inbounds <[[BLOCKTY]]>, <[[BLOCKTY]]>* [[BLOCK]], i32 0, i32 5
// CHECK-UNOPT: [[CAPTURED:%.*]] = load i8*, i8** [[XADDR]]
// CHECK-UNOPT: [[RETAINED:%.*]] = call i8* @llvm.objc.retain(i8* [[CAPTURED]])
// CHECK-UNOPT: store i8* [[RETAINED]], i8** [[BLOCKCAPTURED]]
// CHECK-UNOPT: call void @llvm.objc.storeStrong(i8** [[BLOCKCAPTURED]], i8* null)

void test20_callee(void (^)());
void test20(const id x) {
  test20_callee(^{ (void)x; });
}

// CHECK-LABEL: define{{.*}} void @test21(
// CHECK: %[[V6:.*]] = call i8* @llvm.objc.retainBlock(
// CHECK: %[[V7:.*]] = bitcast i8* %[[V6]] to void ()*
// CHECK: call void (i32, ...) @test21_callee(i32 1, void ()* %[[V7]]),

void test21_callee(int n, ...);
void test21(id x) {
  test21_callee(1, ^{ (void)x; });
}

// The lifetime of 'x', which is captured by the block in the statement
// expression, should be extended.

// CHECK-COMMON-LABEL: define{{.*}} i8* @test22(
// CHECK-COMMON: %[[BLOCK_CAPTURED:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %{{.*}}*, i8* }>, <{ i8*, i32, i32, i8*, %{{.*}}*, i8* }>* %{{.*}}, i32 0, i32 5
// CHECK-COMMON: %[[V3:.*]] = call i8* @llvm.objc.retain(i8* %{{.*}})
// CHECK-COMMON: store i8* %[[V3]], i8** %[[BLOCK_CAPTURED]], align 8
// CHECK-COMMON: call void @test22_1()
// CHECK-UNOPT: call void @llvm.objc.storeStrong(i8** %[[BLOCK_CAPTURED]], i8* null)
// CHECK: %[[V15:.*]] = load i8*, i8** %[[BLOCK_CAPTURED]], align 8
// CHECK: call void @llvm.objc.release(i8* %[[V15]])

id test22(int c, id x) {
  extern id test22_0(void);
  extern void test22_1(void);
  return c ? test22_0() : ({ id (^b)(void) = ^{ return x; }; test22_1(); b(); });
}

@interface Test23
-(void)m:(int)i, ...;
@end

// CHECK-COMMON-LABEL: define{{.*}} void @test23(
// CHECK-COMMON: %[[V9:.*]] = call i8* @llvm.objc.retainBlock(
// CHECK-COMMON: %[[V10:.*]] = bitcast i8* %[[V9]] to void ()*
// CHECK-COMMON: call void (i8*, i8*, i32, ...) bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, i32, ...)*)(i8* %{{.*}}, i8* %{{.*}}, i32 123, void ()* %[[V10]])

void test23(id x, Test23 *t) {
  [t m:123, ^{ (void)x; }];
}

// CHECK: attributes [[NUW]] = { nounwind }
// CHECK-UNOPT: attributes [[NUW]] = { nounwind }
