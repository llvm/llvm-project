// RUN: %clang_cc1 -triple i386-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -fblocks -o - %s | FileCheck %s

// test1.  All of this is somehow testing rdar://6676764
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

// rdar://problem/9006315
// In-depth test for the initialization of a __weak __block variable.
@interface Test2 -(void) destroy; @end
void test2(Test2 *x) {
  extern void test2_helper(void (^)(void));
  // CHECK-LABEL:    define{{.*}} void @test2(
  // CHECK:      [[X:%.*]] = alloca [[TEST2:%.*]]*,
  // CHECK-NEXT: [[WEAKX:%.*]] = alloca [[WEAK_T:%.*]],
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
  // CHECK-NEXT: store [[TEST2]]*

  // isa=1 for weak byrefs.
  // CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[WEAK_T]], [[WEAK_T]]* [[WEAKX]], i32 0, i32 0
  // CHECK-NEXT: store i8* inttoptr (i32 1 to i8*), i8** [[T0]]

  // Forwarding.
  // CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[WEAK_T]], [[WEAK_T]]* [[WEAKX]], i32 0, i32 1
  // CHECK-NEXT: store [[WEAK_T]]* [[WEAKX]], [[WEAK_T]]** [[T1]]

  // Flags.  This is just BLOCK_HAS_COPY_DISPOSE BLOCK_BYREF_LAYOUT_UNRETAINED
  // CHECK-NEXT: [[T2:%.*]] = getelementptr inbounds [[WEAK_T]], [[WEAK_T]]* [[WEAKX]], i32 0, i32 2
  // CHECK-NEXT: store i32 1375731712, i32* [[T2]]

  // Size.
  // CHECK-NEXT: [[T3:%.*]] = getelementptr inbounds [[WEAK_T]], [[WEAK_T]]* [[WEAKX]], i32 0, i32 3
  // CHECK-NEXT: store i32 28, i32* [[T3]]

  // Copy and dispose helpers.
  // CHECK-NEXT: [[T4:%.*]] = getelementptr inbounds [[WEAK_T]], [[WEAK_T]]* [[WEAKX]], i32 0, i32 4
  // CHECK-NEXT: store i8* bitcast (void (i8*, i8*)* @__Block_byref_object_copy_{{.*}} to i8*), i8** [[T4]]
  // CHECK-NEXT: [[T5:%.*]] = getelementptr inbounds [[WEAK_T]], [[WEAK_T]]* [[WEAKX]], i32 0, i32 5
  // CHECK-NEXT: store i8* bitcast (void (i8*)* @__Block_byref_object_dispose_{{.*}} to i8*), i8** [[T5]]

  // Actually capture the value.
  // CHECK-NEXT: [[T6:%.*]] = getelementptr inbounds [[WEAK_T]], [[WEAK_T]]* [[WEAKX]], i32 0, i32 6
  // CHECK-NEXT: [[CAPTURE:%.*]] = load [[TEST2]]*, [[TEST2]]** [[X]]
  // CHECK-NEXT: store [[TEST2]]* [[CAPTURE]], [[TEST2]]** [[T6]]

  // Then we initialize the block, blah blah blah.
  // CHECK:      call void @test2_helper(

  // Finally, kill the variable with BLOCK_FIELD_IS_BYREF.
  // CHECK:      [[T0:%.*]] = bitcast [[WEAK_T]]* [[WEAKX]] to i8*
  // CHECK:      call void @_Block_object_dispose(i8* [[T0]], i32 24)

  __attribute__((objc_gc(weak))) __block Test2 *weakX = x;
  test2_helper(^{ [weakX destroy]; });
}

// rdar://problem/9124263
// In the test above, check that the use in the invocation function
// doesn't require a read barrier.
// CHECK-LABEL:    define internal void @__test2_block_invoke
// CHECK:      [[BLOCK:%.*]] = bitcast i8* {{%.*}} to [[BLOCK_T]]*
// CHECK-NOT:  bitcast
// CHECK:      [[T0:%.*]] = getelementptr inbounds [[BLOCK_T]], [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
// CHECK-NEXT: [[T1:%.*]] = load i8*, i8** [[T0]]
// CHECK-NEXT: [[T2:%.*]] = bitcast i8* [[T1]] to [[WEAK_T]]{{.*}}*
// CHECK-NEXT: [[T3:%.*]] = getelementptr inbounds [[WEAK_T]]{{.*}}, [[WEAK_T]]{{.*}}* [[T2]], i32 0, i32 1
// CHECK-NEXT: [[T4:%.*]] = load [[WEAK_T]]{{.*}}*, [[WEAK_T]]{{.*}}** [[T3]]
// CHECK-NEXT: [[WEAKX:%.*]] = getelementptr inbounds [[WEAK_T]]{{.*}}, [[WEAK_T]]{{.*}}* [[T4]], i32 0, i32 6
// CHECK-NEXT: [[T0:%.*]] = load [[TEST2]]*, [[TEST2]]** [[WEAKX]], align 4

// rdar://problem/12722954
// Make sure that ... is appropriately positioned in a block call.
void test3(void (^block)(int, ...)) {
  block(0, 1, 2, 3);
}
// CHECK-LABEL:    define{{.*}} void @test3(
// CHECK:      [[BLOCK:%.*]] = alloca void (i32, ...)*, align 4
// CHECK-NEXT: store void (i32, ...)*
// CHECK-NEXT: [[T0:%.*]] = load void (i32, ...)*, void (i32, ...)** [[BLOCK]], align 4
// CHECK-NEXT: [[T1:%.*]] = bitcast void (i32, ...)* [[T0]] to [[BLOCK_T:%.*]]*
// CHECK-NEXT: [[T2:%.*]] = getelementptr inbounds [[BLOCK_T]], [[BLOCK_T]]* [[T1]], i32 0, i32 3
// CHECK-NEXT: [[T3:%.*]] = bitcast [[BLOCK_T]]* [[T1]] to i8*
// CHECK-NEXT: [[T4:%.*]] = load i8*, i8** [[T2]]
// CHECK-NEXT: [[T5:%.*]] = bitcast i8* [[T4]] to void (i8*, i32, ...)*
// CHECK-NEXT: call void (i8*, i32, ...) [[T5]](i8* noundef [[T3]], i32 noundef 0, i32 noundef 1, i32 noundef 2, i32 noundef 3)
// CHECK-NEXT: ret void

void test4(void (^block)()) {
  block(0, 1, 2, 3);
}
// CHECK-LABEL:    define{{.*}} void @test4(
// CHECK:      [[BLOCK:%.*]] = alloca void (...)*, align 4
// CHECK-NEXT: store void (...)*
// CHECK-NEXT: [[T0:%.*]] = load void (...)*, void (...)** [[BLOCK]], align 4
// CHECK-NEXT: [[T1:%.*]] = bitcast void (...)* [[T0]] to [[BLOCK_T:%.*]]*
// CHECK-NEXT: [[T2:%.*]] = getelementptr inbounds [[BLOCK_T]], [[BLOCK_T]]* [[T1]], i32 0, i32 3
// CHECK-NEXT: [[T3:%.*]] = bitcast [[BLOCK_T]]* [[T1]] to i8*
// CHECK-NEXT: [[T4:%.*]] = load i8*, i8** [[T2]]
// CHECK-NEXT: [[T5:%.*]] = bitcast i8* [[T4]] to void (i8*, i32, i32, i32, i32)*
// CHECK-NEXT: call void [[T5]](i8* noundef [[T3]], i32 noundef 0, i32 noundef 1, i32 noundef 2, i32 noundef 3)
// CHECK-NEXT: ret void
