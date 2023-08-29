// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -O2 -disable-llvm-passes -o - %s | FileCheck -check-prefix=CHECK -check-prefix=CHECK-HEAP -check-prefix=CHECK-COMMON %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -O2 -disable-llvm-passes -fobjc-avoid-heapify-local-blocks -o - %s | FileCheck -check-prefix=CHECK -check-prefix=CHECK-NOHEAP -check-prefix=CHECK-COMMON %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -o - %s | FileCheck -check-prefix=CHECK-UNOPT -check-prefix=CHECK-COMMON %s

// CHECK-COMMON: @[[BLOCK_DESCRIPTOR_TMP44:.*]] = linkonce_odr hidden unnamed_addr constant { i64, i64, ptr, ptr, ptr, i64 } { i64 0, i64 40, ptr @__copy_helper_block_8_32s, ptr @__destroy_helper_block_8_32s, ptr @{{.*}}, i64 256 }, align 8
// CHECK-COMMON: @[[BLOCK_DESCRIPTOR_TMP9:.*]] = linkonce_odr hidden unnamed_addr constant { i64, i64, ptr, ptr, ptr, i64 } { i64 0, i64 40, ptr @__copy_helper_block_8_32r, ptr @__destroy_helper_block_8_32r, ptr @{{.*}}, i64 16 }, align 8
// CHECK-COMMON: @[[BLOCK_DESCRIPTOR_TMP46:.*]] = linkonce_odr hidden unnamed_addr constant { i64, i64, ptr, ptr, ptr, ptr } { i64 0, i64 48, ptr @__copy_helper_block_8_32s, ptr @__destroy_helper_block_8_32s, ptr @{{.*}}, ptr @{{.*}} }, align 8
// CHECK-COMMON: @[[BLOCK_DESCRIPTOR_TMP48:.*]] = linkonce_odr hidden unnamed_addr constant { i64, i64, ptr, ptr, ptr, i64 } { i64 0, i64 40, ptr @__copy_helper_block_8_32b, ptr @__destroy_helper_block_8_32s, ptr @{{.*}}, i64 256 }, align 8

// Check that no copy/dispose helpers are emitted for this block.

// CHECK-COMMON: @[[BLOCK_DESCRIPTOR_TMP10:.*]] = linkonce_odr hidden unnamed_addr constant { i64, i64, ptr, ptr } { i64 0, i64 40, ptr @{{.*}}, ptr @{{.*}} }, align 8

// This shouldn't crash.
void test0(id (^maker)(void)) {
  maker();
}

int (^test1(int x))(void) {
  // CHECK-LABEL:    define{{.*}} ptr @test1(
  // CHECK:      [[X:%.*]] = alloca i32,
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
  // CHECK-NEXT: store i32 {{%.*}}, ptr [[X]]
  // CHECK: [[T2:%.*]] = call ptr @llvm.objc.retainBlock(ptr [[BLOCK]]) [[NUW:#[0-9]+]]
  // CHECK-NEXT: [[T5:%.*]] = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr [[T2]]) [[NUW]]
  // CHECK-NEXT: ret ptr [[T5]]
  return ^{ return x; };
}

void test2(id x) {
// CHECK-LABEL:    define{{.*}} void @test2(
// CHECK:      [[X:%.*]] = alloca ptr,
// CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
// CHECK-NEXT: [[PARM:%.*]] = call ptr @llvm.objc.retain(ptr {{%.*}})
// CHECK-NEXT: store ptr [[PARM]], ptr [[X]]
// CHECK:      [[SLOT:%.*]] = getelementptr inbounds [[BLOCK_T]], ptr [[BLOCK]], i32 0, i32 5
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[X]],
// CHECK-NEXT: [[T1:%.*]] = call ptr @llvm.objc.retain(ptr [[T0]])
// CHECK-NEXT: store ptr [[T1]], ptr [[SLOT]],
// CHECK-NEXT: call void @test2_helper(
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[SLOT]]
// CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]]) [[NUW]], !clang.imprecise_release
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[X]]
// CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]]) [[NUW]], !clang.imprecise_release
// CHECK-NEXT: ret void
  extern void test2_helper(id (^)(void));
  test2_helper(^{ return x; });

// CHECK:    define linkonce_odr hidden void @__copy_helper_block_8_32s(ptr noundef %0, ptr noundef %1) unnamed_addr #{{[0-9]+}} {
// CHECK:      [[SRC:%.*]] = load ptr, ptr
// CHECK-NEXT: [[DST:%.*]] = load ptr, ptr
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[BLOCK_T]], ptr [[SRC]], i32 0, i32 5
// CHECK-NEXT: [[T1:%.*]] = load ptr, ptr [[T0]]
// CHECK-NEXT: [[T2:%.*]] = call ptr @llvm.objc.retain(ptr [[T1]]) [[NUW]]
// CHECK-NEXT: ret void


// CHECK:    define linkonce_odr hidden void @__destroy_helper_block_8_32s(ptr noundef %0) unnamed_addr #{{[0-9]+}} {
// CHECK:      [[T0:%.*]] = load ptr, ptr
// CHECK-NEXT: [[T2:%.*]] = getelementptr inbounds [[BLOCK_T]], ptr [[T0]], i32 0, i32 5
// CHECK-NEXT: [[T3:%.*]] = load ptr, ptr [[T2]]
// CHECK-NEXT: call void @llvm.objc.release(ptr [[T3]])
// CHECK-NEXT: ret void
}

void test3(void (^sink)(id*)) {
  __strong id strong;
  sink(&strong);

  // CHECK-LABEL:    define{{.*}} void @test3(
  // CHECK:      [[SINK:%.*]] = alloca ptr
  // CHECK-NEXT: [[STRONG:%.*]] = alloca ptr
  // CHECK-NEXT: [[TEMP:%.*]] = alloca ptr
  // CHECK-NEXT: call ptr @llvm.objc.retain(
  // CHECK-NEXT: store ptr {{%.*}}, ptr [[SINK]]
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[STRONG]])
  // CHECK-NEXT: store ptr null, ptr [[STRONG]]

  // CHECK-NEXT: [[BLOCK:%.*]] = load ptr, ptr [[SINK]]
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: [[V:%.*]] = load ptr, ptr [[STRONG]]
  // CHECK-NEXT: store ptr [[V]], ptr [[TEMP]]
  // CHECK-NEXT: [[F0:%.*]] = load ptr, ptr
  // CHECK-NEXT: call void [[F0]](ptr noundef [[BLOCK]], ptr noundef [[TEMP]])
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[TEMP]]
  // CHECK-NEXT: [[T1:%.*]] = call ptr @llvm.objc.retain(ptr [[T0]])
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.use(ptr [[V]]) [[NUW]]
  // CHECK-NEXT: [[T2:%.*]] = load ptr, ptr [[STRONG]]
  // CHECK-NEXT: store ptr [[T1]], ptr [[STRONG]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T2]])

  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[STRONG]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]])
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr [[STRONG]])

  // CHECK-NEXT: load ptr, ptr [[SINK]]
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
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BYREF_T]], ptr [[VAR]], i32 0, i32 2
  // 0x02000000 - has copy/dispose helpers strong
  // CHECK-NEXT: store i32 838860800, ptr [[T0]]
  // CHECK:      [[SLOT:%.*]] = getelementptr inbounds [[BYREF_T]], ptr [[VAR]], i32 0, i32 6
  // CHECK-NEXT: [[T0:%.*]] = call ptr @test4_source() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T0]])
  // CHECK-NEXT: store ptr [[T0]], ptr [[SLOT]]
  // CHECK-NEXT: [[SLOT:%.*]] = getelementptr inbounds [[BYREF_T]], ptr [[VAR]], i32 0, i32 6
  // 0x42800000 - has signature, copy/dispose helpers, as well as BLOCK_HAS_EXTENDED_LAYOUT
  // CHECK:      store i32 -1040187392,
  // CHECK: store ptr [[VAR]], ptr
  // CHECK:      call void @test4_helper(
  // CHECK: call void @_Block_object_dispose(ptr [[VAR]], i32 8)
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[SLOT]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]])
  // CHECK: ret void

  // CHECK-LABEL:    define internal void @__Block_byref_object_copy_(ptr noundef %0, ptr noundef %1) #{{[0-9]+}} {
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BYREF_T]], ptr {{%.*}}, i32 0, i32 6
  // CHECK-NEXT: load ptr, ptr
  // CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[BYREF_T]], ptr {{%.*}}, i32 0, i32 6
  // CHECK-NEXT: [[T2:%.*]] = load ptr, ptr [[T1]]
  // CHECK-NEXT: store ptr [[T2]], ptr [[T0]]
  // CHECK-NEXT: store ptr null, ptr [[T1]]

  // CHECK-LABEL:    define internal void @__Block_byref_object_dispose_(ptr noundef %0) #{{[0-9]+}} {
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BYREF_T]], ptr {{%.*}}, i32 0, i32 6
  // CHECK-NEXT: [[T1:%.*]] = load ptr, ptr [[T0]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T1]])

  // CHECK-LABEL:    define internal void @__test4_block_invoke
  // CHECK:      [[SLOT:%.*]] = getelementptr inbounds {{.*}}, i32 0, i32 6
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[SLOT]], align 8
  // CHECK-NEXT: store ptr null, ptr [[SLOT]],
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]])
  // CHECK-NEXT: ret void

  // CHECK-LABEL:    define linkonce_odr hidden void @__copy_helper_block_8_32r(ptr noundef %0, ptr noundef %1) unnamed_addr #{{[0-9]+}} {
  // CHECK:      call void @_Block_object_assign(ptr {{%.*}}, ptr {{%.*}}, i32 8)

  // CHECK-LABEL:    define linkonce_odr hidden void @__destroy_helper_block_8_32r(ptr noundef %0) unnamed_addr #{{[0-9]+}} {
  // CHECK:      call void @_Block_object_dispose(ptr {{%.*}}, i32 8)
}

void test5(void) {
  extern id test5_source(void);
  void test5_helper(void (^)(void));
  __unsafe_unretained id var = test5_source();
  test5_helper(^{ (void) var; });

  // CHECK-LABEL:    define{{.*}} void @test5()
  // CHECK:      [[VAR:%.*]] = alloca ptr
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[VAR]])
  // CHECK: [[T1:%.*]] = call ptr @test5_source() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T1]])
  // CHECK-NEXT: store ptr [[T1]], ptr [[VAR]],
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T1]])
  // 0x40800000 - has signature but no copy/dispose, as well as BLOCK_HAS_EXTENDED_LAYOUT
  // CHECK:      store i32 -1073741824, ptr
  // CHECK:      [[CAPTURE:%.*]] = getelementptr inbounds [[BLOCK_T]], ptr [[BLOCK]], i32 0, i32 5
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[VAR]]
  // CHECK-NEXT: store ptr [[T0]], ptr [[CAPTURE]]
  // CHECK: call void @test5_helper
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr [[VAR]])
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
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 48, ptr [[VAR]])
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BYREF_T]], ptr [[VAR]], i32 0, i32 2
  // 0x02000000 - has copy/dispose helpers weak
  // CHECK-NEXT: store i32 1107296256, ptr [[T0]]
  // CHECK:      [[SLOT:%.*]] = getelementptr inbounds [[BYREF_T]], ptr [[VAR]], i32 0, i32 6
  // CHECK-NEXT: [[T1:%.*]] = call ptr @test6_source() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T1]])
  // CHECK-NEXT: call ptr @llvm.objc.initWeak(ptr [[SLOT]], ptr [[T1]])
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T1]])
  // CHECK-NEXT: [[SLOT:%.*]] = getelementptr inbounds [[BYREF_T]], ptr [[VAR]], i32 0, i32 6
  // 0x42800000 - has signature, copy/dispose helpers, as well as BLOCK_HAS_EXTENDED_LAYOUT
  // CHECK:      store i32 -1040187392,
  // CHECK: %[[BLOCK_DESCRIPTOR:.*]] = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %{{.*}}, i32 0, i32 4
// CHECK: store ptr @[[BLOCK_DESCRIPTOR_TMP9]], ptr %[[BLOCK_DESCRIPTOR]], align 8
  // CHECK: store ptr [[VAR]], ptr
  // CHECK:      call void @test6_helper(
  // CHECK: call void @_Block_object_dispose(ptr [[VAR]], i32 8)
  // CHECK-NEXT: call void @llvm.objc.destroyWeak(ptr [[SLOT]])
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 48, ptr [[VAR]])
  // CHECK-NEXT: ret void

  // CHECK-LABEL:    define internal void @__Block_byref_object_copy_.{{[0-9]+}}(ptr noundef %0, ptr noundef %1) #{{[0-9]+}} {
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BYREF_T]], ptr {{%.*}}, i32 0, i32 6
  // CHECK-NEXT: load ptr, ptr
  // CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[BYREF_T]], ptr {{%.*}}, i32 0, i32 6
  // CHECK-NEXT: call void @llvm.objc.moveWeak(ptr [[T0]], ptr [[T1]])

  // CHECK-LABEL:    define internal void @__Block_byref_object_dispose_.{{[0-9]+}}(ptr noundef %0) #{{[0-9]+}} {
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BYREF_T]], ptr {{%.*}}, i32 0, i32 6
  // CHECK-NEXT: call void @llvm.objc.destroyWeak(ptr [[T0]])

  // CHECK-LABEL:    define internal void @__test6_block_invoke
  // CHECK:      [[SLOT:%.*]] = getelementptr inbounds {{.*}}, i32 0, i32 6
  // CHECK-NEXT: call ptr @llvm.objc.storeWeak(ptr [[SLOT]], ptr null)
  // CHECK-NEXT: ret void
}

void test7(void) {
  id test7_source(void);
  void test7_helper(void (^)(void));
  void test7_consume(id);
  __weak id var = test7_source();
  test7_helper(^{ test7_consume(var); });

  // CHECK-LABEL:    define{{.*}} void @test7()
  // CHECK:      [[VAR:%.*]] = alloca ptr,
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
  // CHECK:      [[T1:%.*]] = call ptr @test7_source() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T1]])
  // CHECK-NEXT: call ptr @llvm.objc.initWeak(ptr [[VAR]], ptr [[T1]])
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T1]])
  // 0x42800000 - has signature, copy/dispose helpers, as well as BLOCK_HAS_EXTENDED_LAYOUT
  // CHECK:      store i32 -1040187392,
  // CHECK:      [[SLOT:%.*]] = getelementptr inbounds [[BLOCK_T]], ptr [[BLOCK]], i32 0, i32 5
  // CHECK-NEXT: call void @llvm.objc.copyWeak(ptr [[SLOT]], ptr [[VAR]])
  // CHECK:      call void @test7_helper(
  // CHECK-NEXT: call void @llvm.objc.destroyWeak(ptr {{%.*}})
  // CHECK-NEXT: call void @llvm.objc.destroyWeak(ptr [[VAR]])
  // CHECK: ret void

  // CHECK-LABEL:    define internal void @__test7_block_invoke
  // CHECK:      [[SLOT:%.*]] = getelementptr inbounds [[BLOCK_T]], ptr {{%.*}}, i32 0, i32 5
  // CHECK-NEXT: [[T0:%.*]] = call ptr @llvm.objc.loadWeakRetained(ptr [[SLOT]])
  // CHECK-NEXT: call void @test7_consume(ptr noundef [[T0]])
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]])
  // CHECK: ret void

  // CHECK-LABEL:    define linkonce_odr hidden void @__copy_helper_block_8_32w(ptr noundef %0, ptr noundef %1) unnamed_addr #{{[0-9]+}} {
  // CHECK:      getelementptr
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: call void @llvm.objc.copyWeak(

  // CHECK-LABEL:    define linkonce_odr hidden void @__destroy_helper_block_8_32w(ptr noundef %0) unnamed_addr #{{[0-9]+}} {
  // CHECK:      getelementptr
  // CHECK-NEXT: call void @llvm.objc.destroyWeak(
}

@interface Test8 @end
@implementation Test8
- (void) test {
// CHECK:    define internal void @"\01-[Test8 test]"
// CHECK:      [[SELF:%.*]] = alloca ptr,
// CHECK-NEXT: alloca ptr
// CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
// CHECK: store
// CHECK-NEXT: store
// CHECK:      [[T0:%.*]] = getelementptr inbounds [[BLOCK_T]], ptr [[BLOCK]], i32 0, i32 5
// CHECK-NEXT: [[T1:%.*]] = load ptr, ptr [[SELF]],
// CHECK-NEXT: store ptr [[T1]], ptr [[T0]]
// CHECK: call void @test8_helper(
// CHECK-NEXT: [[T2:%.*]] = load ptr, ptr [[T0]]
// CHECK-NEXT: call void (...) @llvm.objc.clang.arc.use(ptr [[T2]])
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

// CHECK-LABEL:    define{{.*}} ptr @test9(
// CHECK:      load ptr, ptr getelementptr
// CHECK-NEXT: call ptr
// CHECK-NEXT: tail call ptr @llvm.objc.autoreleaseReturnValue
// CHECK-NEXT: ret ptr

// CHECK:      call ptr @test9_produce() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(
// CHECK-NEXT: ret ptr
}

// Test that we correctly initialize __block variables
// when the initialization captures the variable.
void test10a(void) {
  __block void (^block)(void) = ^{ block(); };
  // CHECK-LABEL:       define{{.*}} void @test10a()
  // CHECK:             [[BYREF:%.*]] = alloca [[BYREF_T:%.*]],
  // CHECK-NOHEAP:      [[BLOCK1:%.*]] = alloca <{ ptr, i32, i32, ptr, ptr, ptr }>, align 8

  // Zero-initialization before running the initializer.
  // CHECK:             [[T0:%.*]] = getelementptr inbounds [[BYREF_T]], ptr [[BYREF]], i32 0, i32 6
  // CHECK-NEXT:        store ptr null, ptr [[T0]], align 8

  // Run the initializer as an assignment.
  // CHECK-HEAP:   [[T1:%.*]] = call ptr @llvm.objc.retainBlock(ptr {{%.*}})
  // CHECK:        [[T3:%.*]] = getelementptr inbounds [[BYREF_T]], ptr [[BYREF]], i32 0, i32 1
  // CHECK-NEXT:        [[T4:%.*]] = load ptr, ptr [[T3]]
  // CHECK-NEXT:        [[T5:%.*]] = getelementptr inbounds [[BYREF_T]], ptr [[T4]], i32 0, i32 6
  // CHECK-NEXT:        [[T6:%.*]] = load ptr, ptr [[T5]], align 8
  // CHECK-HEAP-NEXT:   store ptr {{%.*}}, ptr [[T5]], align 8
  // CHECK-NOHEAP-NEXT: store ptr [[BLOCK1]], ptr [[T5]], align 8
  // CHECK-NEXT:        call void @llvm.objc.release(ptr [[T6]])

  // Destroy at end of function.
  // CHECK-NEXT:        [[SLOT:%.*]] = getelementptr inbounds [[BYREF_T]], ptr [[BYREF]], i32 0, i32 6
  // CHECK-NEXT:        call void @_Block_object_dispose(ptr [[BYREF]], i32 8)
  // CHECK-NEXT:        [[T1:%.*]] = load ptr, ptr [[SLOT]]
  // CHECK-NEXT:        call void @llvm.objc.release(ptr [[T1]])
  // CHECK: ret void
}

// do this copy and dispose with objc_retainBlock/release instead of
// _Block_object_assign/destroy. We can also use _Block_object_assign/destroy
// with BLOCK_FIELD_IS_BLOCK as long as we don't pass BLOCK_BYREF_CALLER.

// CHECK-LABEL: define internal void @__Block_byref_object_copy_.{{[0-9]+}}(ptr noundef %0, ptr noundef %1) #{{[0-9]+}} {
// CHECK:      [[D0:%.*]] = load ptr, ptr {{%.*}}
// CHECK-NEXT: [[D2:%.*]] = getelementptr inbounds [[BYREF_T]], ptr [[D0]], i32 0, i32 6
// CHECK-NEXT: [[S0:%.*]] = load ptr, ptr {{%.*}}
// CHECK-NEXT: [[S2:%.*]] = getelementptr inbounds [[BYREF_T]], ptr [[S0]], i32 0, i32 6
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[S2]], align 8
// CHECK-NEXT: [[T2:%.*]] = call ptr @llvm.objc.retainBlock(ptr [[T0]])
// CHECK-NEXT: store ptr [[T2]], ptr [[D2]], align 8
// CHECK: ret void

// CHECK-LABEL: define internal void @__Block_byref_object_dispose_.{{[0-9]+}}(ptr noundef %0) #{{[0-9]+}} {
// CHECK:      [[T0:%.*]] = load ptr, ptr {{%.*}}
// CHECK-NEXT: [[T2:%.*]] = getelementptr inbounds [[BYREF_T]], ptr [[T0]], i32 0, i32 6
// CHECK-NEXT: [[T3:%.*]] = load ptr, ptr [[T2]]
// CHECK-NEXT: call void @llvm.objc.release(ptr [[T3]])
// CHECK-NEXT: ret void

// Test that we correctly assign to __block variables when the
// assignment captures the variable.
void test10b(void) {
  __block void (^block)(void);
  block = ^{ block(); };

  // CHECK-LABEL:       define{{.*}} void @test10b()
  // CHECK:             [[BYREF:%.*]] = alloca [[BYREF_T:%.*]],
  // CHECK-NOHEAP:      [[BLOCK3:%.*]] = alloca <{ ptr, i32, i32, ptr, ptr, ptr }>, align 8

  // Zero-initialize.
  // CHECK:             [[T0:%.*]] = getelementptr inbounds [[BYREF_T]], ptr [[BYREF]], i32 0, i32 6
  // CHECK-NEXT:        store ptr null, ptr [[T0]], align 8

  // CHECK-NEXT:        [[SLOT:%.*]] = getelementptr inbounds [[BYREF_T]], ptr [[BYREF]], i32 0, i32 6

  // The assignment.
  // CHECK-HEAP:   [[T1:%.*]] = call ptr @llvm.objc.retainBlock(ptr {{%.*}})
  // CHECK:        [[T3:%.*]] = getelementptr inbounds [[BYREF_T]], ptr [[BYREF]], i32 0, i32 1
  // CHECK-NEXT:        [[T4:%.*]] = load ptr, ptr [[T3]]
  // CHECK-NEXT:        [[T5:%.*]] = getelementptr inbounds [[BYREF_T]], ptr [[T4]], i32 0, i32 6
  // CHECK-NEXT:        [[T6:%.*]] = load ptr, ptr [[T5]], align 8
  // CHECK-HEAP-NEXT:   store ptr {{%.*}}, ptr [[T5]], align 8
  // CHECK-NOHEAP-NEXT: store ptr [[BLOCK3]], ptr [[T5]], align 8
  // CHECK-NEXT:        call void @llvm.objc.release(ptr [[T6]])

  // Destroy at end of function.
  // CHECK-NEXT:        call void @_Block_object_dispose(ptr [[BYREF]], i32 8)
  // CHECK-NEXT:        [[T1:%.*]] = load ptr, ptr [[SLOT]]
  // CHECK-NEXT:        call void @llvm.objc.release(ptr [[T1]])
  // CHECK: ret void
}

void test11_helper(id);
void test11a(void) {
  int x;
  test11_helper(^{ (void) x; });

  // CHECK-LABEL:    define{{.*}} void @test11a()
  // CHECK:      [[X:%.*]] = alloca i32, align 4
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]], align 8
  // CHECK: [[T2:%.*]] = call ptr @llvm.objc.retainBlock(ptr [[BLOCK]])
  // CHECK-NEXT: call void @test11_helper(ptr noundef [[T2]])
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T2]])
  // CHECK: ret void
}
void test11b(void) {
  int x;
  id b = ^{ (void) x; };

  // CHECK-LABEL:    define{{.*}} void @test11b()
  // CHECK:      [[X:%.*]] = alloca i32, align 4
  // CHECK-NEXT: [[B:%.*]] = alloca ptr, align 8
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]], align 8
  // CHECK: [[T2:%.*]] = call ptr @llvm.objc.retainBlock(ptr [[BLOCK]])
  // CHECK-NEXT: store ptr [[T2]], ptr [[B]], align 8
  // CHECK-NEXT: [[T5:%.*]] = load ptr, ptr [[B]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T5]])
  // CHECK: ret void
}

@interface Test12
@property (strong) void(^ablock)(void);
@property (nonatomic, strong) void(^nblock)(void);
@end
@implementation Test12
@synthesize ablock, nblock;
// CHECK:    define internal ptr @"\01-[Test12 ablock]"(
// CHECK:    call ptr @objc_getProperty(ptr noundef {{%.*}}, ptr noundef {{%.*}}, i64 noundef {{%.*}}, i1 noundef zeroext true)

// CHECK:    define internal void @"\01-[Test12 setAblock:]"(
// CHECK:    call void @objc_setProperty(ptr noundef {{%.*}}, ptr noundef {{%.*}}, i64 noundef {{%.*}}, ptr noundef {{%.*}}, i1 noundef zeroext true, i1 noundef zeroext true)

// CHECK:    define internal ptr @"\01-[Test12 nblock]"(
// CHECK:    %add.ptr = getelementptr inbounds i8, ptr %0, i64 %ivar

// CHECK:    define internal void @"\01-[Test12 setNblock:]"(
// CHECK:    call void @objc_setProperty(ptr noundef {{%.*}}, ptr noundef {{%.*}}, i64 noundef {{%.*}}, ptr noundef {{%.*}}, i1 noundef zeroext false, i1 noundef zeroext true)
@end

void test13(id x) {
  extern void test13_helper(id);
  extern void test13_use(void(^)(void));

  void (^b)(void) = (x ? ^{test13_helper(x);} : 0);
  test13_use(b);

  // CHECK-LABEL:    define{{.*}} void @test13(
  // CHECK:      [[X:%.*]] = alloca ptr, align 8
  // CHECK-NEXT: [[B:%.*]] = alloca ptr, align 8
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:.*]], align 8
  // CHECK-NEXT: [[CLEANUP_ACTIVE:%.*]] = alloca i1
  // CHECK-NEXT: [[COND_CLEANUP_SAVE:%.*]] = alloca ptr,
  // CHECK-NEXT: [[T0:%.*]] = call ptr @llvm.objc.retain(ptr {{%.*}})
  // CHECK-NEXT: store ptr [[T0]], ptr [[X]], align 8
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[B]])
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[X]], align 8
  // CHECK-NEXT: [[T1:%.*]] = icmp ne ptr [[T0]], null
  // CHECK-NEXT: store i1 false, ptr [[CLEANUP_ACTIVE]]
  // CHECK-NEXT: br i1 [[T1]],

  // CHECK-NOT:  br
  // CHECK:      [[CAPTURE:%.*]] = getelementptr inbounds [[BLOCK_T]], ptr [[BLOCK]], i32 0, i32 5
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[X]], align 8
  // CHECK-NEXT: [[T1:%.*]] = call ptr @llvm.objc.retain(ptr [[T0]])
  // CHECK-NEXT: store ptr [[T1]], ptr [[CAPTURE]], align 8
  // CHECK-NEXT: store i1 true, ptr [[CLEANUP_ACTIVE]]
  // CHECK-NEXT: store ptr [[CAPTURE]], ptr [[COND_CLEANUP_SAVE]], align 8
  // CHECK-NEXT: br label
  // CHECK:      br label
  // CHECK:      [[T0:%.*]] = phi ptr
  // CHECK-NEXT: [[T2:%.*]] = call ptr @llvm.objc.retainBlock(ptr [[T0]])
  // CHECK-NEXT: store ptr [[T2]], ptr [[B]], align 8
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[B]], align 8
  // CHECK-NEXT: call void @test13_use(ptr noundef [[T0]])
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[B]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]])

  // CHECK-NEXT: [[T0:%.*]] = load i1, ptr [[CLEANUP_ACTIVE]]
  // CHECK-NEXT: br i1 [[T0]]
  // CHECK:      [[V12:%.*]] = load ptr, ptr [[COND_CLEANUP_SAVE]], align 8
  // CHECK:      [[T0:%.*]] = load ptr, ptr [[V12]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]])
  // CHECK-NEXT: br label

  // CHECK: call void @llvm.lifetime.end.p0(i64 8, ptr [[B]])
  // CHECK-NEXT:      [[T0:%.*]] = load ptr, ptr [[X]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]])
  // CHECK-NEXT: ret void
}

void test14(void) {
  void (^const x[1])(void) = { ^{} };
}

// Don't make invalid ASTs and crash.
void test15_helper(void (^block)(void), int x);
void test15(int a) {
  test15_helper(^{ (void) a; }, ({ a; }));
}

void test16(void) {
  void (^BLKVAR)(void) = ^{ BLKVAR(); };

  // CHECK-LABEL: define{{.*}} void @test16(
  // CHECK: [[BLKVAR:%.*]]  = alloca ptr, align 8
  // CHECK-NEXT:  [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
  // CHECK-NEXT:  call void @llvm.lifetime.start.p0(i64 8, ptr [[BLKVAR]])
  // CHECK-NEXT:  store ptr null, ptr [[BLKVAR]], align 8
}

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
// CHECK-LABEL:    define{{.*}} ptr @test17(
// CHECK:      [[RET:%.*]] = alloca ptr, align
// CHECK-NEXT: [[SELF:%.*]] = alloca ptr,
// CHECK:      [[B0:%.*]] = alloca [[BLOCK:<.*>]], align
// CHECK:      [[B1:%.*]] = alloca [[BLOCK]], align
// CHECK:      [[T0:%.*]] = call ptr @llvm.objc.retain(ptr
// CHECK-NEXT: store ptr [[T0]], ptr [[SELF]], align
// CHECK-NOT:  objc_retain
// CHECK-NOT:  objc_release
// CHECK:      [[CAPTURED:%.*]] = getelementptr inbounds [[BLOCK]], ptr [[B0]], i32 0, i32 5
// CHECK-NEXT: [[T1:%.*]] = load ptr, ptr [[SELF]], align
// CHECK-NEXT: [[T2:%.*]] = call ptr @llvm.objc.retain(ptr [[T1]])
// CHECK-NEXT: store ptr [[T2]], ptr [[CAPTURED]],
// CHECK-NEXT: [[T2:%.*]] = call ptr @llvm.objc.retainBlock(ptr [[B0]])
// CHECK-NEXT: store ptr [[T2]], ptr [[RET]]
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[CAPTURED]]
// CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]])
// CHECK-NEXT: store i32
// CHECK-NEXT: br label
// CHECK-NOT:  objc_retain
// CHECK-NOT:  objc_release
// CHECK:      [[CAPTURED:%.*]] = getelementptr inbounds [[BLOCK]], ptr [[B1]], i32 0, i32 5
// CHECK-NEXT: [[T1:%.*]] = load ptr, ptr [[SELF]], align
// CHECK-NEXT: [[T2:%.*]] = call ptr @llvm.objc.retain(ptr [[T1]])
// CHECK-NEXT: store ptr [[T2]], ptr [[CAPTURED]],
// CHECK-NEXT: [[T2:%.*]] = call ptr @llvm.objc.retainBlock(ptr [[B1]])
// CHECK-NEXT: store ptr [[T2]], ptr [[RET]]
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[CAPTURED]]
// CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]])
// CHECK-NEXT: store i32
// CHECK-NEXT: br label

void test18(id x) {
// CHECK-UNOPT-LABEL:    define{{.*}} void @test18(
// CHECK-UNOPT:      [[X:%.*]] = alloca ptr,
// CHECK-UNOPT-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
// CHECK-UNOPT-NEXT: store ptr null, ptr [[X]]
// CHECK-UNOPT-NEXT: call void @llvm.objc.storeStrong(ptr [[X]], 
// CHECK-UNOPT: %[[BLOCK_DESCRIPTOR:.*]] = getelementptr inbounds [[BLOCK_T]], ptr [[BLOCK]], i32 0, i32 4
// CHECK-UNOPT: store ptr @[[BLOCK_DESCRIPTOR_TMP44]], ptr %[[BLOCK_DESCRIPTOR]], align 8
// CHECK-UNOPT:      [[SLOT:%.*]] = getelementptr inbounds [[BLOCK_T]], ptr [[BLOCK]], i32 0, i32 5
// CHECK-UNOPT-NEXT: [[T0:%.*]] = load ptr, ptr [[X]],
// CHECK-UNOPT-NEXT: [[T1:%.*]] = call ptr @llvm.objc.retain(ptr [[T0]])
// CHECK-UNOPT-NEXT: store ptr [[T1]], ptr [[SLOT]],
// CHECK-UNOPT-NEXT: call void @test18_helper(
// CHECK-UNOPT-NEXT: call void @llvm.objc.storeStrong(ptr [[SLOT]], ptr null) [[NUW:#[0-9]+]]
// CHECK-UNOPT-NEXT: call void @llvm.objc.storeStrong(ptr [[X]], ptr null) [[NUW]]
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
// %[[BLOCK_DESCRIPTOR:.*]] = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr, ptr }>, ptr %{{.*}}, i32 0, i32 4
// CHECK: store ptr @[[BLOCK_DESCRIPTOR_TMP46]], ptr %[[BLOCK_DESCRIPTOR]], align 8

// CHECK-LABEL: define internal void @__testUnsafeUnretainedLifetimeInCopyAndDestroyHelpers_block_invoke
// CHECK-UNOPT-LABEL: define internal void @__testUnsafeUnretainedLifetimeInCopyAndDestroyHelpers_block_invoke

void test19_sink(void (^)(int));
void test19(void (^b)(void)) {
// CHECK-LABEL:    define{{.*}} void @test19(
//   Prologue.
// CHECK:      [[B:%.*]] = alloca ptr,
// CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
// CHECK-NEXT: [[T1:%.*]] = call ptr @llvm.objc.retain(ptr {{%.*}})
// CHECK-NEXT: store ptr [[T1]], ptr [[B]]

//   Block setup.  We skip most of this.  Note the bare retain.
// CHECK: %[[BLOCK_DESCRIPTOR:.*]] = getelementptr inbounds [[BLOCK_T]], ptr [[BLOCK]], i32 0, i32 4
// CHECK: store ptr @[[BLOCK_DESCRIPTOR_TMP48]], ptr %[[BLOCK_DESCRIPTOR]], align 8
// CHECK:      [[SLOT:%.*]] = getelementptr inbounds [[BLOCK_T]], ptr [[BLOCK]], i32 0, i32 5
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[B]],
// CHECK-NEXT: [[T2:%.*]] = call ptr @llvm.objc.retain(ptr [[T0]])
// CHECK-NEXT: store ptr [[T2]], ptr [[SLOT]],
//   Call.
// CHECK-NEXT: call void @test19_sink(ptr noundef [[BLOCK]])

  test19_sink(^(int x) { b(); });

//   Block teardown.
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[SLOT]]
// CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]])

//   Local cleanup.
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[B]]
// CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]])

// CHECK-NEXT: ret void
}

// CHECK-LABEL: define{{.*}} void @test20(
// CHECK: [[XADDR:%.*]] = alloca ptr
// CHECK-NEXT: [[BLOCK:%.*]] = alloca <[[BLOCKTY:.*]]>
// CHECK-NEXT: [[RETAINEDX:%.*]] = call ptr @llvm.objc.retain(ptr %{{.*}})
// CHECK-NEXT: store ptr [[RETAINEDX]], ptr [[XADDR]]
// CHECK: [[BLOCKCAPTURED:%.*]] = getelementptr inbounds <[[BLOCKTY]]>, ptr [[BLOCK]], i32 0, i32 5
// CHECK: [[CAPTURED:%.*]] = load ptr, ptr [[XADDR]]
// CHECK: store ptr [[CAPTURED]], ptr [[BLOCKCAPTURED]]
// CHECK: [[CAPTURE:%.*]] = load ptr, ptr [[BLOCKCAPTURED]]
// CHECK-NEXT: call void (...) @llvm.objc.clang.arc.use(ptr [[CAPTURE]])
// CHECK-NEXT: [[X:%.*]] = load ptr, ptr [[XADDR]]
// CHECK-NEXT: call void @llvm.objc.release(ptr [[X]])
// CHECK-NEXT: ret void

// CHECK-UNOPT-LABEL: define{{.*}} void @test20(
// CHECK-UNOPT: [[XADDR:%.*]] = alloca ptr
// CHECK-UNOPT-NEXT: [[BLOCK:%.*]] = alloca <[[BLOCKTY:.*]]>
// CHECK-UNOPT: [[BLOCKCAPTURED:%.*]] = getelementptr inbounds <[[BLOCKTY]]>, ptr [[BLOCK]], i32 0, i32 5
// CHECK-UNOPT: [[CAPTURED:%.*]] = load ptr, ptr [[XADDR]]
// CHECK-UNOPT: [[RETAINED:%.*]] = call ptr @llvm.objc.retain(ptr [[CAPTURED]])
// CHECK-UNOPT: store ptr [[RETAINED]], ptr [[BLOCKCAPTURED]]
// CHECK-UNOPT: call void @llvm.objc.storeStrong(ptr [[BLOCKCAPTURED]], ptr null)

void test20_callee(void (^)(void));
void test20(const id x) {
  test20_callee(^{ (void)x; });
}

// CHECK-LABEL: define{{.*}} void @test21(
// CHECK: %[[V6:.*]] = call ptr @llvm.objc.retainBlock(
// CHECK: call void (i32, ...) @test21_callee(i32 noundef 1, ptr noundef %[[V6]]),

void test21_callee(int n, ...);
void test21(id x) {
  test21_callee(1, ^{ (void)x; });
}

// The lifetime of 'x', which is captured by the block in the statement
// expression, should be extended.

// CHECK-COMMON-LABEL: define{{.*}} ptr @test22(
// CHECK-COMMON: %[[BLOCK_CAPTURED:.*]] = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %{{.*}}, i32 0, i32 5
// CHECK-COMMON: %[[V3:.*]] = call ptr @llvm.objc.retain(ptr %{{.*}})
// CHECK-COMMON: store ptr %[[V3]], ptr %[[BLOCK_CAPTURED]], align 8
// CHECK-COMMON: call void @test22_1()
// CHECK-UNOPT: call void @llvm.objc.storeStrong(ptr %[[BLOCK_CAPTURED]], ptr null)
// CHECK: %[[V15:.*]] = load ptr, ptr %[[BLOCK_CAPTURED]], align 8
// CHECK: call void @llvm.objc.release(ptr %[[V15]])

id test22(int c, id x) {
  extern id test22_0(void);
  extern void test22_1(void);
  return c ? test22_0() : ({ id (^b)(void) = ^{ return x; }; test22_1(); b(); });
}

@interface Test23
-(void)m:(int)i, ...;
@end

// CHECK-COMMON-LABEL: define{{.*}} void @test23(
// CHECK-COMMON: %[[V9:.*]] = call ptr @llvm.objc.retainBlock(
// CHECK-COMMON: call void (ptr, ptr, i32, ...) @objc_msgSend(ptr noundef %{{.*}}, ptr noundef %{{.*}}, i32 noundef 123, ptr noundef %[[V9]])

void test23(id x, Test23 *t) {
  [t m:123, ^{ (void)x; }];
}

// CHECK-COMMON-LABEL: define internal void @"\01+[Test24 m]"(
// CHECK-COMMON: %[[BLOCK_DESCRIPTOR:.*]] = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %{{.*}}, i32 0, i32 4
// CHECK: store ptr @[[BLOCK_DESCRIPTOR_TMP10]], ptr %[[BLOCK_DESCRIPTOR]],

@interface Test24
@property (class) void (^block)(void);
+(void)m;
@end

@implementation Test24
+(void)m {
  self.block = ^{ (void)self; };
}
@end

// CHECK: attributes [[NUW]] = { nounwind }
// CHECK-UNOPT: attributes [[NUW]] = { nounwind }
