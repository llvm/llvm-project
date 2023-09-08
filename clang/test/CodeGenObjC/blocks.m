// RUN: %clang_cc1 -triple i386-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -fblocks -Wno-strict-prototypes -o - %s | FileCheck %s

// Check that there is only one capture (20o) in the copy/dispose function
// names.

// CHECK: @[[BLOCK_DESCRIPTOR0:.*]] = linkonce_odr hidden unnamed_addr constant { i32, i32, ptr, ptr, ptr, i32 } { i32 0, i32 28, ptr @__copy_helper_block_4_20o, ptr @__destroy_helper_block_4_20o, ptr @{{.*}}, i32 512 },

void (^gb0)(void);

struct S {
  void (^F)(struct S*);
} P;


@interface T
  - (int)foo: (T* (^)(T*)) x;
@end

void foo(T *P) {
 [P foo: 0];
}

@interface A 
-(void) im0;
@end

// CHECK: define internal i32 @"__8-[A im0]_block_invoke"(
@implementation A
-(void) im0 {
  (void) ^{ return 1; }();
}
@end

@interface B : A @end
@implementation B
-(void) im1 {
  ^(void) { [self im0]; }();
}
-(void) im2 {
  ^{ [super im0]; }();
}
-(void) im3 {
  ^{ ^{[super im0];}(); }();
}
@end

// In-depth test for the initialization of a __weak __block variable.
@interface Test2 -(void) destroy; @end
void test2(Test2 *x) {
  extern void test2_helper(void (^)(void));
  // CHECK-LABEL:    define{{.*}} void @test2(
  // CHECK:      [[X:%.*]] = alloca ptr,
  // CHECK-NEXT: [[WEAKX:%.*]] = alloca [[WEAK_T:%.*]],
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
  // CHECK-NEXT: store ptr

  // isa=1 for weak byrefs.
  // CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[WEAK_T]], ptr [[WEAKX]], i32 0, i32 0
  // CHECK-NEXT: store ptr inttoptr (i32 1 to ptr), ptr [[T0]]

  // Forwarding.
  // CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[WEAK_T]], ptr [[WEAKX]], i32 0, i32 1
  // CHECK-NEXT: store ptr [[WEAKX]], ptr [[T1]]

  // Flags.  This is just BLOCK_HAS_COPY_DISPOSE BLOCK_BYREF_LAYOUT_UNRETAINED
  // CHECK-NEXT: [[T2:%.*]] = getelementptr inbounds [[WEAK_T]], ptr [[WEAKX]], i32 0, i32 2
  // CHECK-NEXT: store i32 1375731712, ptr [[T2]]

  // Size.
  // CHECK-NEXT: [[T3:%.*]] = getelementptr inbounds [[WEAK_T]], ptr [[WEAKX]], i32 0, i32 3
  // CHECK-NEXT: store i32 28, ptr [[T3]]

  // Copy and dispose helpers.
  // CHECK-NEXT: [[T4:%.*]] = getelementptr inbounds [[WEAK_T]], ptr [[WEAKX]], i32 0, i32 4
  // CHECK-NEXT: store ptr @__Block_byref_object_copy_{{.*}}, ptr [[T4]]
  // CHECK-NEXT: [[T5:%.*]] = getelementptr inbounds [[WEAK_T]], ptr [[WEAKX]], i32 0, i32 5
  // CHECK-NEXT: store ptr @__Block_byref_object_dispose_{{.*}}, ptr [[T5]]

  // Actually capture the value.
  // CHECK-NEXT: [[T6:%.*]] = getelementptr inbounds [[WEAK_T]], ptr [[WEAKX]], i32 0, i32 6
  // CHECK-NEXT: [[CAPTURE:%.*]] = load ptr, ptr [[X]]
  // CHECK-NEXT: store ptr [[CAPTURE]], ptr [[T6]]

  // Then we initialize the block, blah blah blah.
  // CHECK:      call void @test2_helper(

  // Finally, kill the variable with BLOCK_FIELD_IS_BYREF.
  // CHECK:      call void @_Block_object_dispose(ptr [[WEAKX]], i32 24)

  __attribute__((objc_gc(weak))) __block Test2 *weakX = x;
  test2_helper(^{ [weakX destroy]; });
}

// In the test above, check that the use in the invocation function
// doesn't require a read barrier.
// CHECK-LABEL:    define internal void @__test2_block_invoke
// CHECK:      [[T0:%.*]] = getelementptr inbounds [[BLOCK_T]], ptr {{%.*}}, i32 0, i32 5
// CHECK-NEXT: [[T1:%.*]] = load ptr, ptr [[T0]]
// CHECK-NEXT: [[T3:%.*]] = getelementptr inbounds [[WEAK_T]]{{.*}}, ptr [[T1]], i32 0, i32 1
// CHECK-NEXT: [[T4:%.*]] = load ptr, ptr [[T3]]
// CHECK-NEXT: [[WEAKX:%.*]] = getelementptr inbounds [[WEAK_T]]{{.*}}, ptr [[T4]], i32 0, i32 6
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[WEAKX]], align 4

// Make sure that ... is appropriately positioned in a block call.
void test3(void (^block)(int, ...)) {
  block(0, 1, 2, 3);
}
// CHECK-LABEL:    define{{.*}} void @test3(
// CHECK:      [[BLOCK:%.*]] = alloca ptr, align 4
// CHECK-NEXT: store ptr
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[BLOCK]], align 4
// CHECK-NEXT: [[T2:%.*]] = getelementptr inbounds [[BLOCK_T:%.*]], ptr [[T0]], i32 0, i32 3
// CHECK-NEXT: [[T4:%.*]] = load ptr, ptr [[T2]]
// CHECK-NEXT: call void (ptr, i32, ...) [[T4]](ptr noundef [[T0]], i32 noundef 0, i32 noundef 1, i32 noundef 2, i32 noundef 3)
// CHECK-NEXT: ret void

void test4(void (^block)()) {
  block(0, 1, 2, 3);
}
// CHECK-LABEL:    define{{.*}} void @test4(
// CHECK:      [[BLOCK:%.*]] = alloca ptr, align 4
// CHECK-NEXT: store ptr
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[BLOCK]], align 4
// CHECK-NEXT: [[T2:%.*]] = getelementptr inbounds [[BLOCK_T:%.*]], ptr [[T0]], i32 0, i32 3
// CHECK-NEXT: [[T4:%.*]] = load ptr, ptr [[T2]]
// CHECK-NEXT: call void [[T4]](ptr noundef [[T0]], i32 noundef 0, i32 noundef 1, i32 noundef 2, i32 noundef 3)
// CHECK-NEXT: ret void

void test5(A *a) {
  __unsafe_unretained A *t = a;
  gb0 = ^{ (void)a; (void)t; };
}

// CHECK-LABEL: define void @test5(
// CHECK: %[[V0:.*]] = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr, ptr }>, ptr %{{.*}}, i32 0, i32 4
// CHECK: store ptr @[[BLOCK_DESCRIPTOR0]], ptr %[[V0]],

void test6(id a, long long b) {
  void (^block)() = ^{ (void)b; (void)a; };
}

// Check that the block literal doesn't have two fields for capture 'a'.

// CHECK-LABEL: define void @test6(
// CHECK: alloca <{ ptr, i32, i32, ptr, ptr, ptr, i64 }>,
