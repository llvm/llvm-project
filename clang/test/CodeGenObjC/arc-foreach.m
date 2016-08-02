// RUN: %clang_cc1 -fblocks -fobjc-arc -fobjc-runtime-has-weak -triple x86_64-apple-darwin -emit-llvm %s -o %t-64.s
// RUN: FileCheck -check-prefix CHECK-LP64 --input-file=%t-64.s %s
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

// CHECK-LP64-LABEL:    define void @test0(
// CHECK-LP64:      [[ARRAY:%.*]] = alloca [[ARRAY_T:%.*]]*,
// CHECK-LP64-NEXT: [[X:%.*]] = alloca i8*,
// CHECK-LP64-NEXT: [[STATE:%.*]] = alloca [[STATE_T:%.*]],
// CHECK-LP64-NEXT: [[BUFFER:%.*]] = alloca [16 x i8*], align 8
// CHECK-LP64-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],

// Initialize 'array'.
// CHECK-LP64-NEXT: store [[ARRAY_T]]* null, [[ARRAY_T]]** [[ARRAY]]
// CHECK-LP64-NEXT: [[ZERO:%.*]] = bitcast [[ARRAY_T]]** [[ARRAY]] to i8**
// CHECK-LP64-NEXT: [[ONE:%.*]] = bitcast [[ARRAY_T]]* {{%.*}} to i8*
// CHECK-LP64-NEXT: call void @objc_storeStrong(i8** [[ZERO]], i8* [[ONE]]) [[NUW:#[0-9]+]]

// Initialize the fast enumaration state.
// CHECK-LP64-NEXT: [[T0:%.*]] = bitcast [[STATE_T]]* [[STATE]] to i8*
// CHECK-LP64-NEXT: call void @llvm.memset.p0i8.i64(i8* [[T0]], i8 0, i64 64, i32 8, i1 false)

// Evaluate the collection expression and retain.
// CHECK-LP64-NEXT: [[T0:%.*]] = load [[ARRAY_T]]*, [[ARRAY_T]]** [[ARRAY]], align 8
// CHECK-LP64-NEXT: [[T1:%.*]] = bitcast [[ARRAY_T]]* [[T0]] to i8*
// CHECK-LP64-NEXT: [[T2:%.*]] = call i8* @objc_retain(i8* [[T1]])
// CHECK-LP64-NEXT: [[SAVED_ARRAY:%.*]] = bitcast i8* [[T2]] to [[ARRAY_T]]*

// Call the enumeration method.
// CHECK-LP64-NEXT: [[T0:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_
// CHECK-LP64-NEXT: [[T1:%.*]] = bitcast [[ARRAY_T]]* [[SAVED_ARRAY]] to i8*
// CHECK-LP64-NEXT: [[SIZE:%.*]] = call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*, [[STATE_T]]*, [16 x i8*]*, i64)*)(i8* [[T1]], i8* [[T0]], [[STATE_T]]* [[STATE]], [16 x i8*]* [[BUFFER]], i64 16)

// Check for a nonzero result.
// CHECK-LP64-NEXT: [[T0:%.*]] = icmp eq i64 [[SIZE]], 0
// CHECK-LP64-NEXT: br i1 [[T0]]

// CHECK-LP64:      [[T0:%.*]] = getelementptr inbounds [[STATE_T]], [[STATE_T]]* [[STATE]], i32 0, i32 1
// CHECK-LP64-NEXT: [[T1:%.*]] = load i8**, i8*** [[T0]]
// CHECK-LP64-NEXT: [[T2:%.*]] = getelementptr i8*, i8** [[T1]], i64
// CHECK-LP64-NEXT: [[T3:%.*]] = load i8*, i8** [[T2]]
// CHECK-LP64-NEXT: store i8* [[T3]], i8** [[X]]

// CHECK-LP64:      [[D0:%.*]] = getelementptr inbounds [[BLOCK_T]], [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
// CHECK-LP64:      [[T0:%.*]] = getelementptr inbounds [[BLOCK_T]], [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
// CHECK-LP64-NEXT: [[T1:%.*]] = load i8*, i8** [[X]]
// CHECK-LP64-NEXT: [[T2:%.*]] = call i8* @objc_retain(i8* [[T1]])
// CHECK-LP64-NEXT: store i8* [[T2]], i8** [[T0]]
// CHECK-LP64-NEXT: [[T1:%.*]] = bitcast [[BLOCK_T]]* [[BLOCK]] 
// CHECK-LP64: call void @use_block(
// CHECK-LP64-NEXT: call void @objc_storeStrong(i8** [[D0]], i8* null)

// CHECK-LP64:      [[T0:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_
// CHECK-LP64-NEXT: [[T1:%.*]] = bitcast [[ARRAY_T]]* [[SAVED_ARRAY]] to i8*
// CHECK-LP64-NEXT: [[SIZE:%.*]] = call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*, [[STATE_T]]*, [16 x i8*]*, i64)*)(i8* [[T1]], i8* [[T0]], [[STATE_T]]* [[STATE]], [16 x i8*]* [[BUFFER]], i64 16)

// Release the array.
// CHECK-LP64:      [[T0:%.*]] = bitcast [[ARRAY_T]]* [[SAVED_ARRAY]] to i8*
// CHECK-LP64-NEXT: call void @objc_release(i8* [[T0]])

// Destroy 'array'.
// CHECK-LP64:      [[T0:%.*]] = bitcast [[ARRAY_T]]** [[ARRAY]] to i8**
// CHECK-LP64-NEXT: call void @objc_storeStrong(i8** [[T0]], i8* null)
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

// CHECK-LP64-LABEL:    define void @test1(
// CHECK-LP64:      alloca [[ARRAY_T:%.*]]*,
// CHECK-LP64-NEXT: [[X:%.*]] = alloca i8*,
// CHECK-LP64-NEXT: [[STATE:%.*]] = alloca [[STATE_T:%.*]],
// CHECK-LP64-NEXT: alloca [16 x i8*], align 8
// CHECK-LP64-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],

// CHECK-LP64:      [[T0:%.*]] = getelementptr inbounds [[STATE_T]], [[STATE_T]]* [[STATE]], i32 0, i32 1
// CHECK-LP64-NEXT: [[T1:%.*]] = load i8**, i8*** [[T0]]
// CHECK-LP64-NEXT: [[T2:%.*]] = getelementptr i8*, i8** [[T1]], i64
// CHECK-LP64-NEXT: [[T3:%.*]] = load i8*, i8** [[T2]]
// CHECK-LP64-NEXT: call i8* @objc_initWeak(i8** [[X]], i8* [[T3]])

// CHECK-LP64:      [[D0:%.*]] = getelementptr inbounds [[BLOCK_T]], [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
// CHECK-LP64:      [[T0:%.*]] = getelementptr inbounds [[BLOCK_T]], [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
// CHECK-LP64-NEXT: call void @objc_copyWeak(i8** [[T0]], i8** [[X]])
// CHECK-LP64-NEXT: [[T1:%.*]] = bitcast [[BLOCK_T]]* [[BLOCK]] to
// CHECK-LP64: call void @use_block
// CHECK-LP64-NEXT: call void @objc_destroyWeak(i8** [[D0]])
// CHECK-LP64-NEXT: call void @objc_destroyWeak(i8** [[X]])

// rdar://problem/9817306
@interface Test2
- (NSArray *) array;
@end
void test2(Test2 *a) {
  for (id x in a.array) {
    use(x);
  }
}

// CHECK-LP64-LABEL:    define void @test2(
// CHECK-LP64:      [[T0:%.*]] = call [[ARRAY_T]]* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to [[ARRAY_T]]* (i8*, i8*)*)(
// CHECK-LP64-NEXT: [[T1:%.*]] = bitcast [[ARRAY_T]]* [[T0]] to i8*
// CHECK-LP64-NEXT: [[T2:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T1]])
// CHECK-LP64-NEXT: [[COLL:%.*]] = bitcast i8* [[T2]] to [[ARRAY_T]]*

// Make sure it's not immediately released before starting the iteration.
// CHECK-LP64-NEXT: load i8*, i8** @OBJC_SELECTOR_REFERENCES_
// CHECK-LP64-NEXT: [[T0:%.*]] = bitcast [[ARRAY_T]]* [[COLL]] to i8*
// CHECK-LP64-NEXT: @objc_msgSend

// This bitcast is for the mutation check.
// CHECK-LP64:      [[T0:%.*]] = bitcast [[ARRAY_T]]* [[COLL]] to i8*
// CHECK-LP64-NEXT: @objc_enumerationMutation

// This bitcast is for the 'next' message send.
// CHECK-LP64:      [[T0:%.*]] = bitcast [[ARRAY_T]]* [[COLL]] to i8*
// CHECK-LP64-NEXT: @objc_msgSend

// This bitcast is for the final release.
// CHECK-LP64:      [[T0:%.*]] = bitcast [[ARRAY_T]]* [[COLL]] to i8*
// CHECK-LP64-NEXT: call void @objc_release(i8* [[T0]])


// Check that the 'continue' label is positioned appropriately
// relative to the collection clenaup.
void test3(NSArray *array) {
  for (id x in array) {
    if (!x) continue;
    use(x);
  }

  // CHECK-LP64-LABEL:    define void @test3(
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

// CHECK-LP64: attributes [[NUW]] = { nounwind }
