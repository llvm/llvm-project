// RUN: %clang_cc1 -no-enable-noundef-analysis -triple x86_64-apple-darwin -fblocks -fobjc-arc -fobjc-runtime-has-weak -emit-llvm %s -o - | FileCheck -check-prefix CHECK-LP64 %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple x86_64-apple-darwin -O1 -fblocks -fobjc-arc -fobjc-runtime-has-weak -emit-llvm %s -o - | FileCheck -check-prefix CHECK-LP64-OPT %s

extern void use(id);
extern void use_block(void (^)(void));

struct NSFastEnumerationState;
@interface NSArray
- (unsigned long) countByEnumeratingWithState: (struct NSFastEnumerationState*) state
                  objects: (id*) buffer
                  count: (unsigned long) bufferSize;
@end;

void test0(NSArray *array) {
  // 'x' should be initialized without a retain.
  // We should actually do a non-constant capture, and that
  // capture should require a retain.
  for (id x in array) {
    use_block(^{ use(x); });
  }
}

// CHECK-LP64-LABEL:    define{{.*}} void @test0(
// CHECK-LP64:      [[ARRAY:%.*]] = alloca ptr,
// CHECK-LP64-NEXT: [[X:%.*]] = alloca ptr,
// CHECK-LP64-NEXT: [[STATE:%.*]] = alloca [[STATE_T:%.*]],
// CHECK-LP64-NEXT: [[BUFFER:%.*]] = alloca [16 x ptr], align 8
// CHECK-LP64-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],

// CHECK-LP64-OPT-LABEL: define{{.*}} void @test0
// CHECK-LP64-OPT: [[STATE:%.*]] = alloca [[STATE_T:%.*]], align 8
// CHECK-LP64-OPT-NEXT: [[BUFFER:%.*]] = alloca [16 x ptr], align 8
// CHECK-LP64-OPT-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]], align 8

// Initialize 'array'.
// CHECK-LP64-NEXT: store ptr null, ptr [[ARRAY]]
// CHECK-LP64-NEXT: call void @llvm.objc.storeStrong(ptr [[ARRAY]], ptr {{%.*}}) [[NUW:#[0-9]+]]

// Initialize the fast enumaration state.
// CHECK-LP64-NEXT: call void @llvm.memset.p0.i64(ptr align 8 [[STATE]], i8 0, i64 64, i1 false)

// Evaluate the collection expression and retain.
// CHECK-LP64-NEXT: [[T0:%.*]] = load ptr, ptr [[ARRAY]], align 8
// CHECK-LP64-NEXT: [[SAVED_ARRAY:%.*]] = call ptr @llvm.objc.retain(ptr [[T0]])

// Call the enumeration method.
// CHECK-LP64-NEXT: [[T1:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_
// CHECK-LP64-NEXT: [[SIZE:%.*]] = call i64 @objc_msgSend(ptr [[SAVED_ARRAY]], ptr [[T1]], ptr [[STATE]], ptr [[BUFFER]], i64 16)

// Check for a nonzero result.
// CHECK-LP64-NEXT: [[T0:%.*]] = icmp eq i64 [[SIZE]], 0
// CHECK-LP64-NEXT: br i1 [[T0]]

// CHECK-LP64:      [[T0:%.*]] = getelementptr inbounds nuw [[STATE_T]], ptr [[STATE]], i32 0, i32 1
// CHECK-LP64-NEXT: [[T1:%.*]] = load ptr, ptr [[T0]]
// CHECK-LP64-NEXT: [[T2:%.*]] = getelementptr inbounds ptr, ptr [[T1]], i64
// CHECK-LP64-NEXT: [[T3:%.*]] = load ptr, ptr [[T2]]
// CHECK-LP64-NEXT: store ptr [[T3]], ptr [[X]]

// CHECK-LP64:      [[CAPTURED:%.*]] = getelementptr inbounds nuw [[BLOCK_T]], ptr [[BLOCK]], i32 0, i32 5
// CHECK-LP64-NEXT: [[T1:%.*]] = load ptr, ptr [[X]]
// CHECK-LP64-NEXT: [[T2:%.*]] = call ptr @llvm.objc.retain(ptr [[T1]])
// CHECK-LP64-NEXT: store ptr [[T2]], ptr [[CAPTURED]]
// CHECK-LP64-NEXT: call void @use_block(ptr [[BLOCK]])
// CHECK-LP64-NEXT: call void @llvm.objc.storeStrong(ptr [[CAPTURED]], ptr null)
// CHECK-LP64-NOT:  call void (...) @llvm.objc.clang.arc.use(

// CHECK-LP64-OPT: [[D0:%.*]] = getelementptr inbounds nuw i8, ptr [[BLOCK]], i64 32
// CHECK-LP64-OPT: [[CAPTURE:%.*]] = load ptr, ptr [[D0]]
// CHECK-LP64-OPT: call void (...) @llvm.objc.clang.arc.use(ptr [[CAPTURE]])

// CHECK-LP64: [[T1:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_
// CHECK-LP64-NEXT: [[SIZE:%.*]] = call i64 @objc_msgSend(ptr [[SAVED_ARRAY]], ptr [[T1]], ptr [[STATE]], ptr [[BUFFER]], i64 16)

// Release the array.
// CHECK-LP64: call void @llvm.objc.release(ptr [[SAVED_ARRAY]])

// Destroy 'array'.
// CHECK-LP64: call void @llvm.objc.storeStrong(ptr [[ARRAY]], ptr null)
// CHECK-LP64-NEXT: ret void

// CHECK-LP64-LABEL:    define internal void @__test0_block_invoke
// CHECK-LP64-NOT:  ret
// CHECK-LP64:      [[T0:%.*]] = getelementptr inbounds nuw [[BLOCK_T]], ptr {{%.*}}, i32 0, i32 5
// CHECK-LP64-NEXT: [[T2:%.*]] = load ptr, ptr [[T0]], align 8 
// CHECK-LP64-NEXT: call void @use(ptr [[T2]])

void test1(NSArray *array) {
  for (__weak id x in array) {
    use_block(^{ use(x); });
  }
}

// CHECK-LP64-LABEL:    define{{.*}} void @test1(
// CHECK-LP64:      alloca ptr,
// CHECK-LP64-NEXT: [[X:%.*]] = alloca ptr,
// CHECK-LP64-NEXT: [[STATE:%.*]] = alloca [[STATE_T:%.*]],
// CHECK-LP64-NEXT: alloca [16 x ptr], align 8
// CHECK-LP64-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],

// CHECK-LP64:      [[T0:%.*]] = getelementptr inbounds nuw [[STATE_T]], ptr [[STATE]], i32 0, i32 1
// CHECK-LP64-NEXT: [[T1:%.*]] = load ptr, ptr [[T0]]
// CHECK-LP64-NEXT: [[T2:%.*]] = getelementptr inbounds ptr, ptr [[T1]], i64
// CHECK-LP64-NEXT: [[T3:%.*]] = load ptr, ptr [[T2]]
// CHECK-LP64-NEXT: call ptr @llvm.objc.initWeak(ptr [[X]], ptr [[T3]])

// CHECK-LP64:      [[T0:%.*]] = getelementptr inbounds nuw [[BLOCK_T]], ptr [[BLOCK]], i32 0, i32 5
// CHECK-LP64: call void @llvm.objc.copyWeak(ptr [[T0]], ptr [[X]])
// CHECK-LP64: call void @use_block
// CHECK-LP64-NEXT: call void @llvm.objc.destroyWeak(ptr [[T0]])
// CHECK-LP64-NEXT: call void @llvm.objc.destroyWeak(ptr [[X]])

@interface Test2
- (NSArray *) array;
@end
void test2(Test2 *a) {
  for (id x in a.array) {
    use(x);
  }
}

// CHECK-LP64-LABEL:    define{{.*}} void @test2(
// CHECK-LP64:      [[T0:%.*]] = call ptr @objc_msgSend(
// CHECK-LP64-NEXT: [[T2:%.*]] = notail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr [[T0]])

// Make sure it's not immediately released before starting the iteration.
// CHECK-LP64-NEXT: load ptr, ptr @OBJC_SELECTOR_REFERENCES_
// CHECK-LP64-NEXT: @objc_msgSend

// CHECK-LP64: @objc_enumerationMutation

// CHECK-LP64: load ptr, ptr @OBJC_SELECTOR_REFERENCES_
// CHECK-LP64-NEXT: @objc_msgSend

// CHECK-LP64: call void @llvm.objc.release(ptr [[T2]])


// Check that the 'continue' label is positioned appropriately
// relative to the collection clenaup.
void test3(NSArray *array) {
  for (id x in array) {
    if (!x) continue;
    use(x);
  }

  // CHECK-LP64-LABEL:    define{{.*}} void @test3(
  // CHECK-LP64:      [[ARRAY:%.*]] = alloca ptr, align 8
  // CHECK-LP64-NEXT: [[X:%.*]] = alloca ptr, align 8
  // CHECK-LP64:      [[T0:%.*]] = load ptr, ptr [[X]], align 8
  // CHECK-LP64-NEXT: [[T1:%.*]] = icmp ne ptr [[T0]], null
  // CHECK-LP64-NEXT: br i1 [[T1]],
  // CHECK-LP64:      br label [[L:%[^ ]+]]
  // CHECK-LP64:      [[T0:%.*]] = load ptr, ptr [[X]], align 8
  // CHECK-LP64-NEXT: call void @use(ptr [[T0]])
  // CHECK-LP64-NEXT: br label [[L]]
}

@interface NSObject @end

@interface I1 : NSObject
- (NSArray *) foo1:(void (^)(void))block;
- (void) foo2;
@end

NSArray *array4;

@implementation I1 : NSObject
- (NSArray *) foo1:(void (^)(void))block {
  block();
  return array4;
}

- (void) foo2 {
  for (id x in [self foo1:^{ use(self); }]) {
    use(x);
    break;
  }
}
@end

// CHECK-LP64-LABEL: define internal void @"\01-[I1 foo2]"(
// CHECK-LP64:         [[SELF_ADDR:%.*]] = alloca ptr,
// CHECK-LP64:         [[BLOCK:%.*]] = alloca <{ ptr, i32, i32, ptr, ptr, ptr }>,
// CHECK-LP64:         store ptr %self, ptr [[SELF_ADDR]]
// CHECK-LP64:         [[BC:%.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr [[BLOCK]], i32 0, i32 5
// CHECK-LP64:         [[T1:%.*]] = load ptr, ptr [[SELF_ADDR]]
// CHECK-LP64:         call ptr @llvm.objc.retain(ptr [[T1]])

// CHECK-LP64-OPT-LABEL: define internal void @"\01-[I1 foo2]"(
// CHECK-LP64-OPT: ptr %self
// CHECK-LP64-OPT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
// CHECK-LP64-OPT: [[T0:%.*]] = getelementptr inbounds nuw i8, ptr [[BLOCK]], i64 32

// CHECK-LP64:         call void @llvm.objc.storeStrong(ptr [[BC]], ptr null)
// CHECK-LP64-NOT:     call void (...) @llvm.objc.clang.arc.use(ptr [[BC]])
// CHECK-LP64:         switch i32 {{%.*}}, label %[[UNREACHABLE:.*]] [
// CHECK-LP64-NEXT:      i32 0, label %[[CLEANUP_CONT:.*]]
// CHECK-LP64-NEXT:      i32 2, label %[[FORCOLL_END:.*]]
// CHECK-LP64-NEXT:    ]

// CHECK-LP64-OPT: [[T5:%.*]] = load ptr, ptr [[T0]]
// CHECK-LP64-OPT: call void (...) @llvm.objc.clang.arc.use(ptr [[T5]])

// CHECK-LP64:       {{^|:}}[[CLEANUP_CONT]]
// CHECK-LP64-NEXT:    br label %[[FORCOLL_END]]

// CHECK-LP64:       {{^|:}}[[FORCOLL_END]]
// CHECK-LP64-NEXT:    ret void

// CHECK-LP64:       {{^|:}}[[UNREACHABLE]]
// CHECK-LP64-NEXT:    unreachable

// CHECK-LP64: attributes [[NUW]] = { nounwind }
