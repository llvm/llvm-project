// RUN: %clang_cc1 -no-enable-noundef-analysis -triple x86_64-apple-darwin -fblocks -fobjc-arc -fobjc-runtime-has-weak -emit-llvm %s -o - | FileCheck -check-prefix CHECK-LP64 %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple x86_64-apple-darwin -O1 -fblocks -fobjc-arc -fobjc-runtime-has-weak -emit-llvm %s -o - | FileCheck -check-prefix CHECK-LP64-OPT %s
// rdar://9503326
// rdar://9606600

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
// CHECK-LP64:      [[ARRAY:%.*]] = alloca [[ARRAY_T:%.*]]*,
// CHECK-LP64-NEXT: [[X:%.*]] = alloca i8*,
// CHECK-LP64-NEXT: [[STATE:%.*]] = alloca [[STATE_T:%.*]],
// CHECK-LP64-NEXT: [[BUFFER:%.*]] = alloca [16 x i8*], align 8
// CHECK-LP64-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],

// CHECK-LP64-OPT-LABEL: define{{.*}} void @test0
// CHECK-LP64-OPT: [[STATE:%.*]] = alloca [[STATE_T:%.*]], align 8
// CHECK-LP64-OPT-NEXT: [[BUFFER:%.*]] = alloca [16 x i8*], align 8
// CHECK-LP64-OPT-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]], align 8

// Initialize 'array'.
// CHECK-LP64-NEXT: store [[ARRAY_T]]* null, [[ARRAY_T]]** [[ARRAY]]
// CHECK-LP64-NEXT: [[ZERO:%.*]] = bitcast [[ARRAY_T]]** [[ARRAY]] to i8**
// CHECK-LP64-NEXT: [[ONE:%.*]] = bitcast [[ARRAY_T]]* {{%.*}} to i8*
// CHECK-LP64-NEXT: call void @llvm.objc.storeStrong(i8** [[ZERO]], i8* [[ONE]]) [[NUW:#[0-9]+]]

// Initialize the fast enumaration state.
// CHECK-LP64-NEXT: [[T0:%.*]] = bitcast [[STATE_T]]* [[STATE]] to i8*
// CHECK-LP64-NEXT: call void @llvm.memset.p0i8.i64(i8* align 8 [[T0]], i8 0, i64 64, i1 false)

// Evaluate the collection expression and retain.
// CHECK-LP64-NEXT: [[T0:%.*]] = load [[ARRAY_T]]*, [[ARRAY_T]]** [[ARRAY]], align 8
// CHECK-LP64-NEXT: [[T1:%.*]] = bitcast [[ARRAY_T]]* [[T0]] to i8*
// CHECK-LP64-NEXT: [[T2:%.*]] = call i8* @llvm.objc.retain(i8* [[T1]])
// CHECK-LP64-NEXT: [[SAVED_ARRAY:%.*]] = bitcast i8* [[T2]] to [[ARRAY_T]]*

// Call the enumeration method.
// CHECK-LP64-NEXT: [[T0:%.*]] = bitcast [[ARRAY_T]]* [[SAVED_ARRAY]] to i8*
// CHECK-LP64-NEXT: [[T1:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_
// CHECK-LP64-NEXT: [[SIZE:%.*]] = call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*, [[STATE_T]]*, [16 x i8*]*, i64)*)(i8* [[T0]], i8* [[T1]], [[STATE_T]]* [[STATE]], [16 x i8*]* [[BUFFER]], i64 16)

// Check for a nonzero result.
// CHECK-LP64-NEXT: [[T0:%.*]] = icmp eq i64 [[SIZE]], 0
// CHECK-LP64-NEXT: br i1 [[T0]]

// CHECK-LP64:      [[T0:%.*]] = getelementptr inbounds [[STATE_T]], [[STATE_T]]* [[STATE]], i32 0, i32 1
// CHECK-LP64-NEXT: [[T1:%.*]] = load i8**, i8*** [[T0]]
// CHECK-LP64-NEXT: [[T2:%.*]] = getelementptr i8*, i8** [[T1]], i64
// CHECK-LP64-NEXT: [[T3:%.*]] = load i8*, i8** [[T2]]
// CHECK-LP64-NEXT: store i8* [[T3]], i8** [[X]]

// CHECK-LP64:      [[CAPTURED:%.*]] = getelementptr inbounds [[BLOCK_T]], [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
// CHECK-LP64-NEXT: [[T1:%.*]] = load i8*, i8** [[X]]
// CHECK-LP64-NEXT: [[T2:%.*]] = call i8* @llvm.objc.retain(i8* [[T1]])
// CHECK-LP64-NEXT: store i8* [[T2]], i8** [[CAPTURED]]
// CHECK-LP64-NEXT: [[BLOCK1:%.*]] = bitcast [[BLOCK_T]]* [[BLOCK]]
// CHECK-LP64-NEXT: call void @use_block(void ()* [[BLOCK1]])
// CHECK-LP64-NEXT: call void @llvm.objc.storeStrong(i8** [[CAPTURED]], i8* null)
// CHECK-LP64-NOT:  call void (...) @llvm.objc.clang.arc.use(

// CHECK-LP64-OPT: [[D0:%.*]] = getelementptr inbounds [[BLOCK_T]], [[BLOCK_T]]* [[BLOCK]], i64 0, i32 5
// CHECK-LP64-OPT: [[CAPTURE:%.*]] = load i8*, i8** [[D0]]
// CHECK-LP64-OPT: call void (...) @llvm.objc.clang.arc.use(i8* [[CAPTURE]])

// CHECK-LP64:      [[T0:%.*]] = bitcast [[ARRAY_T]]* [[SAVED_ARRAY]] to i8*
// CHECK-LP64-NEXT: [[T1:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_
// CHECK-LP64-NEXT: [[SIZE:%.*]] = call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*, [[STATE_T]]*, [16 x i8*]*, i64)*)(i8* [[T0]], i8* [[T1]], [[STATE_T]]* [[STATE]], [16 x i8*]* [[BUFFER]], i64 16)

// Release the array.
// CHECK-LP64:      [[T0:%.*]] = bitcast [[ARRAY_T]]* [[SAVED_ARRAY]] to i8*
// CHECK-LP64-NEXT: call void @llvm.objc.release(i8* [[T0]])

// Destroy 'array'.
// CHECK-LP64:      [[T0:%.*]] = bitcast [[ARRAY_T]]** [[ARRAY]] to i8**
// CHECK-LP64-NEXT: call void @llvm.objc.storeStrong(i8** [[T0]], i8* null)
// CHECK-LP64-NEXT: ret void

// CHECK-LP64-LABEL:    define internal void @__test0_block_invoke
// CHECK-LP64:      [[BLOCK:%.*]] = bitcast i8* {{%.*}} to [[BLOCK_T]]*
// CHECK-LP64-NOT:  ret
// CHECK-LP64:      [[T0:%.*]] = getelementptr inbounds [[BLOCK_T]], [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
// CHECK-LP64-NEXT: [[T2:%.*]] = load i8*, i8** [[T0]], align 8 
// CHECK-LP64-NEXT: call void @use(i8* [[T2]])

void test1(NSArray *array) {
  for (__weak id x in array) {
    use_block(^{ use(x); });
  }
}

// CHECK-LP64-LABEL:    define{{.*}} void @test1(
// CHECK-LP64:      alloca [[ARRAY_T:%.*]]*,
// CHECK-LP64-NEXT: [[X:%.*]] = alloca i8*,
// CHECK-LP64-NEXT: [[STATE:%.*]] = alloca [[STATE_T:%.*]],
// CHECK-LP64-NEXT: alloca [16 x i8*], align 8
// CHECK-LP64-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],

// CHECK-LP64:      [[T0:%.*]] = getelementptr inbounds [[STATE_T]], [[STATE_T]]* [[STATE]], i32 0, i32 1
// CHECK-LP64-NEXT: [[T1:%.*]] = load i8**, i8*** [[T0]]
// CHECK-LP64-NEXT: [[T2:%.*]] = getelementptr i8*, i8** [[T1]], i64
// CHECK-LP64-NEXT: [[T3:%.*]] = load i8*, i8** [[T2]]
// CHECK-LP64-NEXT: call i8* @llvm.objc.initWeak(i8** [[X]], i8* [[T3]])

// CHECK-LP64:      [[T0:%.*]] = getelementptr inbounds [[BLOCK_T]], [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
// CHECK-LP64-NEXT: call void @llvm.objc.copyWeak(i8** [[T0]], i8** [[X]])
// CHECK-LP64-NEXT: [[T1:%.*]] = bitcast [[BLOCK_T]]* [[BLOCK]] to
// CHECK-LP64: call void @use_block
// CHECK-LP64-NEXT: call void @llvm.objc.destroyWeak(i8** [[T0]])
// CHECK-LP64-NEXT: call void @llvm.objc.destroyWeak(i8** [[X]])

// rdar://problem/9817306
@interface Test2
- (NSArray *) array;
@end
void test2(Test2 *a) {
  for (id x in a.array) {
    use(x);
  }
}

// CHECK-LP64-LABEL:    define{{.*}} void @test2(
// CHECK-LP64:      [[T0:%.*]] = call [[ARRAY_T]]* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to [[ARRAY_T]]* (i8*, i8*)*)(
// CHECK-LP64-NEXT: [[T1:%.*]] = bitcast [[ARRAY_T]]* [[T0]] to i8*
// CHECK-LP64-NEXT: [[T2:%.*]] = notail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* [[T1]])
// CHECK-LP64-NEXT: [[COLL:%.*]] = bitcast i8* [[T2]] to [[ARRAY_T]]*

// Make sure it's not immediately released before starting the iteration.
// CHECK-LP64-NEXT: [[T0:%.*]] = bitcast [[ARRAY_T]]* [[COLL]] to i8*
// CHECK-LP64-NEXT: load i8*, i8** @OBJC_SELECTOR_REFERENCES_
// CHECK-LP64-NEXT: @objc_msgSend

// This bitcast is for the mutation check.
// CHECK-LP64:      [[T0:%.*]] = bitcast [[ARRAY_T]]* [[COLL]] to i8*
// CHECK-LP64-NEXT: @objc_enumerationMutation

// This bitcast is for the 'next' message send.
// CHECK-LP64:      [[T0:%.*]] = bitcast [[ARRAY_T]]* [[COLL]] to i8*
// CHECK-LP64-NEXT: load i8*, i8** @OBJC_SELECTOR_REFERENCES_
// CHECK-LP64-NEXT: @objc_msgSend

// This bitcast is for the final release.
// CHECK-LP64:      [[T0:%.*]] = bitcast [[ARRAY_T]]* [[COLL]] to i8*
// CHECK-LP64-NEXT: call void @llvm.objc.release(i8* [[T0]])


// Check that the 'continue' label is positioned appropriately
// relative to the collection clenaup.
void test3(NSArray *array) {
  for (id x in array) {
    if (!x) continue;
    use(x);
  }

  // CHECK-LP64-LABEL:    define{{.*}} void @test3(
  // CHECK-LP64:      [[ARRAY:%.*]] = alloca [[ARRAY_T]]*, align 8
  // CHECK-LP64-NEXT: [[X:%.*]] = alloca i8*, align 8
  // CHECK-LP64:      [[T0:%.*]] = load i8*, i8** [[X]], align 8
  // CHECK-LP64-NEXT: [[T1:%.*]] = icmp ne i8* [[T0]], null
  // CHECK-LP64-NEXT: br i1 [[T1]],
  // CHECK-LP64:      br label [[L:%[^ ]+]]
  // CHECK-LP64:      [[T0:%.*]] = load i8*, i8** [[X]], align 8
  // CHECK-LP64-NEXT: call void @use(i8* [[T0]])
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
// CHECK-LP64:         [[SELF_ADDR:%.*]] = alloca [[TY:%.*]]*,
// CHECK-LP64:         [[BLOCK:%.*]] = alloca <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, [[TY]]* }>,
// CHECK-LP64:         store [[TY]]* %self, [[TY]]** [[SELF_ADDR]]
// CHECK-LP64:         [[BC:%.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, [[TY]]* }>, <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, [[TY]]* }>* [[BLOCK]], i32 0, i32 5
// CHECK-LP64:         [[T1:%.*]] = load [[TY]]*, [[TY]]** [[SELF_ADDR]]
// CHECK-LP64:         [[T2:%.*]] = bitcast [[TY]]* [[T1]] to i8*
// CHECK-LP64:         call i8* @llvm.objc.retain(i8* [[T2]])

// CHECK-LP64-OPT-LABEL: define internal void @"\01-[I1 foo2]"(
// CHECK-LP64-OPT: [[TY:%.*]]* %self
// CHECK-LP64-OPT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
// CHECK-LP64-OPT: [[T0:%.*]] = getelementptr inbounds [[BLOCK_T]], [[BLOCK_T]]* [[BLOCK]], i64 0, i32 5

// CHECK-LP64:         [[T5:%.*]] = bitcast [[TY]]** [[BC]] to i8**
// CHECK-LP64:         call void @llvm.objc.storeStrong(i8** [[T5]], i8* null)
// CHECK-LP64-NOT:     call void (...) @llvm.objc.clang.arc.use([[TY]]* [[T5]])
// CHECK-LP64:         switch i32 {{%.*}}, label %[[UNREACHABLE:.*]] [
// CHECK-LP64-NEXT:      i32 0, label %[[CLEANUP_CONT:.*]]
// CHECK-LP64-NEXT:      i32 2, label %[[FORCOLL_END:.*]]
// CHECK-LP64-NEXT:    ]

// CHECK-LP64-OPT: [[T5:%.*]] = load [[TY]]*, [[TY]]** [[T0]]
// CHECK-LP64-OPT: call void (...) @llvm.objc.clang.arc.use([[TY]]* [[T5]])

// CHECK-LP64:       {{^|:}}[[CLEANUP_CONT]]
// CHECK-LP64-NEXT:    br label %[[FORCOLL_END]]

// CHECK-LP64:       {{^|:}}[[FORCOLL_END]]
// CHECK-LP64-NEXT:    ret void

// CHECK-LP64:       {{^|:}}[[UNREACHABLE]]
// CHECK-LP64-NEXT:    unreachable

// CHECK-LP64: attributes [[NUW]] = { nounwind }
