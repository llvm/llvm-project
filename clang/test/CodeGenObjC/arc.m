// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -Wno-objc-root-class -Wno-incompatible-pointer-types -Wno-arc-unsafe-retained-assign -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -O2 -disable-llvm-passes -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -Wno-objc-root-class -Wno-incompatible-pointer-types -Wno-arc-unsafe-retained-assign -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -o - %s | FileCheck -check-prefix=CHECK-GLOBALS %s

// Check both native/non-native arc platforms. Here we check that they treat
// nonlazybind differently.
// RUN: %clang_cc1 -fobjc-runtime=macosx-10.6.0 -triple x86_64-apple-darwin10 -Wno-objc-root-class -Wno-incompatible-pointer-types -Wno-arc-unsafe-retained-assign -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -o - %s | FileCheck -check-prefix=ARC-ALIEN %s
// RUN: %clang_cc1 -fobjc-runtime=macosx-10.7.0 -triple x86_64-apple-darwin11 -Wno-objc-root-class -Wno-incompatible-pointer-types -Wno-arc-unsafe-retained-assign -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -o - %s | FileCheck -check-prefix=ARC-NATIVE %s

// ARC-ALIEN: declare extern_weak void @llvm.objc.storeStrong(ptr, ptr)
// ARC-ALIEN: declare extern_weak ptr @llvm.objc.retain(ptr)
// ARC-ALIEN: declare extern_weak ptr @llvm.objc.autoreleaseReturnValue(ptr)
// ARC-ALIEN: declare ptr @objc_msgSend(ptr, ptr, ...) [[NLB:#[0-9]+]]
// ARC-ALIEN: declare extern_weak void @llvm.objc.release(ptr)
// ARC-ALIEN: declare extern_weak ptr @llvm.objc.retainAutoreleasedReturnValue(ptr)
// ARC-ALIEN: declare extern_weak ptr @llvm.objc.initWeak(ptr, ptr)
// ARC-ALIEN: declare extern_weak ptr @llvm.objc.storeWeak(ptr, ptr)
// ARC-ALIEN: declare extern_weak ptr @llvm.objc.loadWeakRetained(ptr)
// ARC-ALIEN: declare extern_weak void @llvm.objc.destroyWeak(ptr)
// ARC-ALIEN: declare extern_weak ptr @llvm.objc.autorelease(ptr)
// ARC-ALIEN: declare extern_weak ptr @llvm.objc.retainAutorelease(ptr)

// ARC-NATIVE: declare void @llvm.objc.storeStrong(ptr, ptr)
// ARC-NATIVE: declare ptr @llvm.objc.retain(ptr)
// ARC-NATIVE: declare ptr @llvm.objc.autoreleaseReturnValue(ptr)
// ARC-NATIVE: declare ptr @objc_msgSend(ptr, ptr, ...) [[NLB:#[0-9]+]]
// ARC-NATIVE: declare void @llvm.objc.release(ptr)
// ARC-NATIVE: declare ptr @llvm.objc.retainAutoreleasedReturnValue(ptr)
// ARC-NATIVE: declare ptr @llvm.objc.initWeak(ptr, ptr)
// ARC-NATIVE: declare ptr @llvm.objc.storeWeak(ptr, ptr)
// ARC-NATIVE: declare ptr @llvm.objc.loadWeakRetained(ptr)
// ARC-NATIVE: declare void @llvm.objc.destroyWeak(ptr)
// ARC-NATIVE: declare ptr @llvm.objc.autorelease(ptr)
// ARC-NATIVE: declare ptr @llvm.objc.retainAutorelease(ptr)

// CHECK-LABEL: define{{.*}} void @test0
void test0(id x) {
  // CHECK:      [[X:%.*]] = alloca ptr
  // CHECK-NEXT: [[PARM:%.*]] = call ptr @llvm.objc.retain(ptr {{.*}})
  // CHECK-NEXT: store ptr [[PARM]], ptr [[X]]
  // CHECK-NEXT: [[TMP:%.*]] = load ptr, ptr [[X]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[TMP]])
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define{{.*}} ptr @test1(ptr
id test1(id x) {
  // CHECK:      [[X:%.*]] = alloca ptr
  // CHECK-NEXT: [[Y:%.*]] = alloca ptr
  // CHECK-NEXT: [[PARM:%.*]] = call ptr @llvm.objc.retain(ptr {{%.*}})
  // CHECK-NEXT: store ptr [[PARM]], ptr [[X]]
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[Y]])
  // CHECK-NEXT: store ptr null, ptr [[Y]]
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[Y]]
  // CHECK-NEXT: [[RET:%.*]] = call ptr @llvm.objc.retain(ptr [[T0]])
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[Y]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]])
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr [[Y]])
  // CHECK-NEXT: [[T1:%.*]] = load ptr, ptr [[X]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T1]])
  // CHECK-NEXT: [[T1:%.*]] = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr [[RET]])
  // CHECK-NEXT: ret ptr [[T1]]
  id y;
  return y;
}

@interface Test2
+ (void) class_method;
- (void) inst_method;
@end
@implementation Test2

// The self pointer of a class method is not retained.
// CHECK: define internal void @"\01+[Test2 class_method]"
// CHECK:      alloca
// CHECK-NEXT: alloca
// CHECK-NEXT: store
// CHECK-NEXT: store
// CHECK-NEXT: ret void
+ (void) class_method {}

// The self pointer of an instance method is not retained.
// CHECK: define internal void @"\01-[Test2 inst_method]"
// CHECK:      alloca
// CHECK-NEXT: alloca
// CHECK-NEXT: store
// CHECK-NEXT: store
// CHECK-NEXT: ret void
- (void) inst_method {}
@end

@interface Test3
+ (id) alloc;
- (id) initWith: (int) x;
- (id) copy;
@end

// CHECK-LABEL: define{{.*}} void @test3_unelided()
void test3_unelided(void) {
  extern void test3_helper(void);

  // CHECK:      [[X:%.*]] = alloca ptr
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[X]])
  // CHECK-NEXT: store ptr null, ptr [[X]], align
  Test3 *x;

  // Call to +alloc.
  // CHECK-NEXT: load {{.*}}, ptr @"OBJC_CLASSLIST_REFERENCES_
  // CHECK-NEXT: load ptr, ptr @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[ALLOC:%.*]] = call ptr @objc_msgSend
  // CHECK-NEXT: call void @llvm.objc.release(ptr
  [Test3 alloc];

  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[X]]
  // CHECK-NEXT: load ptr, ptr @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[COPY:%.*]] = call ptr @objc_msgSend(ptr noundef [[T0]],
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[COPY]]) [[NUW:#[0-9]+]]
  [x copy];

  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[X]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]]) [[NUW]]
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr [[X]])
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define{{.*}} void @test3()
void test3(void) {
  // CHECK:      [[X:%.*]] = alloca ptr
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[X]])

  id x = [[Test3 alloc] initWith: 5];

  // Call to +alloc.
  // CHECK-NEXT: load {{.*}}, ptr @"OBJC_CLASSLIST_REFERENCES_
  // CHECK-NEXT: load ptr, ptr @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[ALLOC:%.*]] = call ptr @objc_msgSend

  // Call to -initWith: with elided retain of consumed argument.
  // CHECK-NEXT: load ptr, ptr @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[INIT:%.*]] = call ptr @objc_msgSend(ptr
  // Assignment for initialization, retention elided.
  // CHECK-NEXT: store ptr [[INIT]], ptr [[X]]

  // Call to -copy.
  // CHECK-NEXT: [[V:%.*]] = load ptr, ptr [[X]]
  // CHECK-NEXT: load ptr, ptr @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[COPY:%.*]] = call ptr @objc_msgSend(ptr noundef [[V]],

  // Assignment to x.
  // CHECK-NEXT: [[TMP:%.*]] = load ptr, ptr [[X]]
  // CHECK-NEXT: store ptr [[COPY]], ptr [[X]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[TMP]]) [[NUW]]

  x = [x copy];

  // Cleanup for x.
  // CHECK-NEXT: [[TMP:%.*]] = load ptr, ptr [[X]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[TMP]]) [[NUW]]
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr [[X]])
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define{{.*}} ptr @test4()
id test4(void) {
  // Call to +alloc.
  // CHECK:      load {{.*}}, ptr @"OBJC_CLASSLIST_REFERENCES_
  // CHECK-NEXT: load ptr, ptr @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[ALLOC:%.*]] = call ptr @objc_msgSend

  // Call to -initWith: with elided retain of consumed argument.
  // CHECK-NEXT: load ptr, ptr @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[INIT:%.*]] = call ptr @objc_msgSend(ptr noundef [[ALLOC]],

  // Initialization of return value, occurring within full-expression.
  // Retain/release elided.
  // CHECK-NEXT: [[RET:%.*]] = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr [[INIT]])

  // CHECK-NEXT: ret ptr [[RET]]

  return [[Test3 alloc] initWith: 6];
}

@interface Test5 {
@public
  id var;
}
@end

// CHECK-LABEL: define{{.*}} void @test5
void test5(Test5 *x, id y) {
  // Prologue.
  // CHECK:      [[X:%.*]] = alloca ptr,
  // CHECK-NEXT: [[Y:%.*]] = alloca ptr
  // CHECK-NEXT: call ptr @llvm.objc.retain
  // CHECK-NEXT: store ptr {{%.*}}, ptr [[X]]
  // CHECK-NEXT: call ptr @llvm.objc.retain
  // CHECK-NEXT: store

  // CHECK-NEXT: load ptr, ptr [[X]]
  // CHECK-NEXT: load i64, ptr @"OBJC_IVAR_$_Test5.var"
  // CHECK-NEXT: [[VAR:%.*]] = getelementptr
  // CHECK-NEXT: [[TMP:%.*]] = load ptr, ptr [[VAR]]
  // CHECK-NEXT: store ptr null, ptr [[VAR]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[TMP]]) [[NUW]]
  x->var = 0;

  // CHECK-NEXT: [[YVAL:%.*]] = load ptr, ptr [[Y]]
  // CHECK-NEXT: load ptr, ptr [[X]]
  // CHECK-NEXT: load i64, ptr @"OBJC_IVAR_$_Test5.var"
  // CHECK-NEXT: [[VAR:%.*]] = getelementptr
  // CHECK-NEXT: [[T0:%.*]] = call ptr @llvm.objc.retain(ptr [[YVAL]]) [[NUW]]
  // CHECK-NEXT: [[TMP:%.*]] = load ptr, ptr [[VAR]]
  // CHECK-NEXT: store ptr [[T0]], ptr [[VAR]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[TMP]]) [[NUW]]
  x->var = y;

  // Epilogue.
  // CHECK-NEXT: [[TMP:%.*]] = load ptr, ptr [[Y]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[TMP]]) [[NUW]]
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[X]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]]) [[NUW]]
  // CHECK-NEXT: ret void
}

id test6_helper(void) __attribute__((ns_returns_retained));
// CHECK-LABEL: define{{.*}} void @test6()
void test6(void) {
  // CHECK:      [[X:%.*]] = alloca ptr
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[X]])
  // CHECK-NEXT: [[CALL:%.*]] = call ptr @test6_helper()
  // CHECK-NEXT: store ptr [[CALL]], ptr [[X]]
  // CHECK-NEXT: [[T1:%.*]] = load ptr, ptr [[X]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T1]]) [[NUW]], !clang.imprecise_release
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr [[X]])
  // CHECK-NEXT: ret void
  id x = test6_helper();
}

void test7_helper(id __attribute__((ns_consumed)));
// CHECK-LABEL: define{{.*}} void @test7()
void test7(void) {
  // CHECK:      [[X:%.*]] = alloca ptr
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[X]])
  // CHECK-NEXT: store ptr null, ptr [[X]]
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[X]]
  // CHECK-NEXT: [[T1:%.*]] = call ptr @llvm.objc.retain(ptr [[T0]]) [[NUW]]
  // CHECK-NEXT: call void @test7_helper(ptr noundef [[T1]])
  // CHECK-NEXT: [[T1:%.*]] = load ptr, ptr [[X]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T1]]) [[NUW]], !clang.imprecise_release
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr [[X]])
  // CHECK-NEXT: ret void
  id x;
  test7_helper(x);
}

id test8_helper(void) __attribute__((ns_returns_retained));
void test8(void) {
  __unsafe_unretained id x = test8_helper();
  // CHECK:      [[X:%.*]] = alloca ptr
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[X]])
  // CHECK-NEXT: [[T0:%.*]] = call ptr @test8_helper()
  // CHECK-NEXT: store ptr [[T0]], ptr [[X]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]]) [[NUW]], !clang.imprecise_release
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr [[X]])
  // CHECK-NEXT: ret void
}

@interface Test10
@property (retain) Test10 *me;
@end
void test10(void) {
  Test10 *x;
  id y = x.me.me;

  // CHECK-LABEL:      define{{.*}} void @test10()
  // CHECK:      [[X:%.*]] = alloca ptr, align
  // CHECK-NEXT: [[Y:%.*]] = alloca ptr, align
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[X]])
  // CHECK-NEXT: store ptr null, ptr [[X]]
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[Y]])
  // CHECK-NEXT: load ptr, ptr [[X]], align
  // CHECK-NEXT: load ptr, ptr @OBJC_SELECTOR_REFERENCES_{{[0-9]*}}
  // CHECK-NEXT: [[V:%.*]] = call ptr @objc_msgSend{{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[V]])
  // CHECK-NEXT: load ptr, ptr @OBJC_SELECTOR_REFERENCES_{{[0-9]*}}
  // CHECK-NEXT: [[T3:%.*]] = call ptr @objc_msgSend{{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T3]])
  // CHECK-NEXT: store ptr [[T3]], ptr [[Y]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[V]])
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[Y]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]])
  // CHECK-NEXT: void @llvm.lifetime.end.p0(i64 8, ptr [[Y]])
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[X]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]])
  // CHECK-NEXT: void @llvm.lifetime.end.p0(i64 8, ptr [[X]])
  // CHECK-NEXT: ret void
}

void test11(id (*f)(void) __attribute__((ns_returns_retained))) {
  // CHECK-LABEL:      define{{.*}} void @test11(
  // CHECK:      [[F:%.*]] = alloca ptr, align
  // CHECK-NEXT: [[X:%.*]] = alloca ptr, align
  // CHECK-NEXT: store ptr {{%.*}}, ptr [[F]], align
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[X]])
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[F]], align
  // CHECK-NEXT: [[T1:%.*]] = call ptr [[T0]]()
  // CHECK-NEXT: store ptr [[T1]], ptr [[X]], align
  // CHECK-NEXT: [[T3:%.*]] = load ptr, ptr [[X]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T3]]) [[NUW]], !clang.imprecise_release
  // CHECK-NEXT: void @llvm.lifetime.end.p0(i64 8, ptr [[X]])
  // CHECK-NEXT: ret void
  id x = f();
}

void test12(void) {
  extern id test12_helper(void);

  // CHECK-LABEL:      define{{.*}} void @test12()
  // CHECK:      [[X:%.*]] = alloca ptr, align
  // CHECK-NEXT: [[Y:%.*]] = alloca ptr, align

  __weak id x = test12_helper();
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[X]])
  // CHECK-NEXT: [[T1:%.*]] = call ptr @test12_helper(){{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T1]])
  // CHECK-NEXT: call ptr @llvm.objc.initWeak(ptr [[X]], ptr [[T1]])
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T1]])

  x = test12_helper();
  // CHECK-NEXT: [[T1:%.*]] = call ptr @test12_helper(){{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T1]])
  // CHECK-NEXT: call ptr @llvm.objc.storeWeak(ptr [[X]], ptr [[T1]])
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T1]])

  id y = x;
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[Y]])
  // CHECK-NEXT: [[T2:%.*]] = call ptr @llvm.objc.loadWeakRetained(ptr [[X]])
  // CHECK-NEXT: store ptr [[T2]], ptr [[Y]], align

  // CHECK-NEXT: [[T4:%.*]] = load ptr, ptr [[Y]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T4]]) [[NUW]], !clang.imprecise_release
  // CHECK-NEXT: void @llvm.lifetime.end.p0(i64 8, ptr [[Y]])
  // CHECK-NEXT: call void @llvm.objc.destroyWeak(ptr [[X]])
  // CHECK-NEXT: void @llvm.lifetime.end.p0(i64 8, ptr [[X]])
  // CHECK: ret void
}

// Indirect consuming calls.
void test13(void) {
  // CHECK-LABEL:      define{{.*}} void @test13()
  // CHECK:      [[X:%.*]] = alloca ptr, align
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[X]])
  // CHECK-NEXT: store ptr null, ptr [[X]], align
  id x;

  typedef void fnty(id __attribute__((ns_consumed)));
  extern fnty *test13_func;
  // CHECK-NEXT: [[FN:%.*]] = load ptr, ptr @test13_func, align
  // CHECK-NEXT: [[X_VAL:%.*]] = load ptr, ptr [[X]], align
  // CHECK-NEXT: [[X_TMP:%.*]] = call ptr @llvm.objc.retain(ptr [[X_VAL]]) [[NUW]]
  // CHECK-NEXT: call void [[FN]](ptr noundef [[X_TMP]])
  test13_func(x);

  extern fnty ^test13_block;
  // CHECK-NEXT: [[TMP:%.*]] = load ptr, ptr @test13_block, align
  // CHECK-NEXT: [[BLOCK_FN_PTR:%.*]] = getelementptr inbounds [[BLOCKTY:%.*]], ptr [[TMP]], i32 0, i32 3
  // CHECK-NEXT: [[X_VAL:%.*]] = load ptr, ptr [[X]], align
  // CHECK-NEXT: [[X_TMP:%.*]] = call ptr @llvm.objc.retain(ptr [[X_VAL]]) [[NUW]]
  // CHECK-NEXT: [[BLOCK_FN_TMP:%.*]] = load ptr, ptr [[BLOCK_FN_PTR]]
  // CHECK-NEXT: call void [[BLOCK_FN_TMP]](ptr noundef [[TMP]], ptr noundef [[X_TMP]])
  test13_block(x);

  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[X]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]]) [[NUW]]
  // CHECK-NEXT: void @llvm.lifetime.end.p0(i64 8, ptr [[X]])
  // CHECK-NEXT: ret void
}

@interface Test16_super @end
@interface Test16 : Test16_super {
  id z;
}
@property (assign) int x;
@property (retain) id y;
- (void) dealloc;
@end
@implementation Test16
@synthesize x;
@synthesize y;
- (void) dealloc {
  // CHECK:    define internal void @"\01-[Test16 dealloc]"(
  // CHECK:      [[SELF:%.*]] = alloca ptr, align
  // CHECK-NEXT: [[CMD:%.*]] = alloca ptr, align
  // CHECK-NEXT: alloca
  // CHECK-NEXT: store ptr {{%.*}}, ptr [[SELF]], align
  // CHECK-NEXT: store ptr {{%.*}}, ptr [[CMD]]
  // CHECK-NEXT: [[BASE:%.*]] = load ptr, ptr [[SELF]]

  // Call super.
  // CHECK-NEXT: [[T0:%.*]] = getelementptr
  // CHECK-NEXT: store ptr [[BASE]], ptr [[T0]]
  // CHECK-NEXT: load ptr, ptr @"OBJC_CLASSLIST_SUP_REFS_$_
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: store
  // CHECK-NEXT: load ptr, ptr @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: call void @objc_msgSendSuper2(
  // CHECK-NEXT: ret void
}

// .cxx_destruct
  // CHECK:    define internal void @"\01-[Test16 .cxx_destruct]"(
  // CHECK:      [[SELF:%.*]] = alloca ptr, align
  // CHECK-NEXT: [[CMD:%.*]] = alloca ptr, align
  // CHECK-NEXT: store ptr {{%.*}}, ptr [[SELF]], align
  // CHECK-NEXT: store ptr {{%.*}}, ptr [[CMD]]
  // CHECK-NEXT: [[BASE:%.*]] = load ptr, ptr [[SELF]]

  // Destroy y.
  // CHECK-NEXT: [[Y_OFF:%.*]] = load i64, ptr @"OBJC_IVAR_$_Test16.y"
  // CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds i8, ptr [[BASE]], i64 [[Y_OFF]]
  // CHECK-NEXT: call void @llvm.objc.storeStrong(ptr [[T1]], ptr null) [[NUW]]

  // Destroy z.
  // CHECK-NEXT: [[Z_OFF:%.*]] = load i64, ptr @"OBJC_IVAR_$_Test16.z"
  // CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds i8, ptr [[BASE]], i64 [[Z_OFF]]
  // CHECK-NEXT: call void @llvm.objc.storeStrong(ptr [[T1]], ptr null) [[NUW]]

  // CHECK-NEXT: ret void

@end

// This shouldn't crash.
@interface Test17A
@property (assign) int x;
@end
@interface Test17B : Test17A
@end
@implementation Test17B
- (int) x { return super.x + 1; }
@end

void test19(void) {
  // CHECK-LABEL: define{{.*}} void @test19()
  // CHECK:      [[X:%.*]] = alloca [5 x ptr], align 16
  // CHECK: call void @llvm.lifetime.start
  // CHECK: call void @llvm.memset.p0.i64(ptr align 16 [[X]], i8 0, i64 40, i1 false)
  id x[5];

  extern id test19_helper(void);
  x[2] = test19_helper();

  // CHECK-NEXT: [[T1:%.*]] = call ptr @test19_helper(){{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T1]])
  // CHECK-NEXT: [[SLOT:%.*]] = getelementptr inbounds [5 x ptr], ptr [[X]], i64 0, i64 2
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[SLOT]]
  // CHECK-NEXT: store ptr [[T1]], ptr [[SLOT]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]]) [[NUW]]

  // CHECK-NEXT: [[BEGIN:%.*]] = getelementptr inbounds [5 x ptr], ptr [[X]], i32 0, i32 0
  // CHECK-NEXT: [[END:%.*]] = getelementptr inbounds ptr, ptr [[BEGIN]], i64 5
  // CHECK-NEXT: br label

  // CHECK:      [[AFTER:%.*]] = phi ptr [ [[END]], {{%.*}} ], [ [[NEXT:%.*]], {{%.*}} ]
  // CHECK-NEXT: [[CUR:%.*]] = getelementptr inbounds ptr, ptr [[AFTER]], i64 -1
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[CUR]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]]) [[NUW]], !clang.imprecise_release
  // CHECK-NEXT: [[EQ:%.*]] = icmp eq ptr [[CUR]], [[BEGIN]]
  // CHECK-NEXT: br i1 [[EQ]],

  // CHECK:      ret void
}

void test20(unsigned n) {
  // CHECK-LABEL: define{{.*}} void @test20
  // CHECK:      [[N:%.*]] = alloca i32, align 4
  // CHECK-NEXT: [[SAVED_STACK:%.*]] = alloca ptr
  // CHECK-NEXT: [[VLA_EXPR:%.*]] = alloca i64, align 8
  // CHECK-NEXT: store i32 {{%.*}}, ptr [[N]], align 4

  id x[n];

  // Capture the VLA size.
  // CHECK-NEXT: [[T0:%.*]] = load i32, ptr [[N]], align 4
  // CHECK-NEXT: [[DIM:%.*]] = zext i32 [[T0]] to i64

  // Save the stack pointer.
  // CHECK-NEXT: [[T0:%.*]] = call ptr @llvm.stacksave.p0()
  // CHECK-NEXT: store ptr [[T0]], ptr [[SAVED_STACK]]

  // Allocate the VLA.
  // CHECK-NEXT: [[VLA:%.*]] = alloca ptr, i64 [[DIM]], align 16

  // Store the VLA #elements expression.
  // CHECK-NEXT: store i64 [[DIM]], ptr [[VLA_EXPR]], align 8

  // Zero-initialize.
  // CHECK-NEXT: [[T1:%.*]] = mul nuw i64 [[DIM]], 8
  // CHECK-NEXT: call void @llvm.memset.p0.i64(ptr align 16 [[VLA]], i8 0, i64 [[T1]], i1 false)

  // Destroy.
  // CHECK-NEXT: [[END:%.*]] = getelementptr inbounds ptr, ptr [[VLA]], i64 [[DIM]]
  // CHECK-NEXT: [[EMPTY:%.*]] = icmp eq ptr [[VLA]], [[END]]
  // CHECK-NEXT: br i1 [[EMPTY]]

  // CHECK:      [[AFTER:%.*]] = phi ptr [ [[END]], {{%.*}} ], [ [[CUR:%.*]], {{%.*}} ]
  // CHECK-NEXT: [[CUR:%.*]] = getelementptr inbounds ptr, ptr [[AFTER]], i64 -1
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[CUR]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]]) [[NUW]], !clang.imprecise_release
  // CHECK-NEXT: [[EQ:%.*]] = icmp eq ptr [[CUR]], [[VLA]]
  // CHECK-NEXT: br i1 [[EQ]],

  // CHECK:      [[T0:%.*]] = load ptr, ptr [[SAVED_STACK]]
  // CHECK-NEXT: call void @llvm.stackrestore.p0(ptr [[T0]])
  // CHECK-NEXT: ret void
}

void test21(unsigned n) {
  // CHECK-LABEL: define{{.*}} void @test21
  // CHECK:      [[N:%.*]] = alloca i32, align 4
  // CHECK-NEXT: [[SAVED_STACK:%.*]] = alloca ptr
  // CHECK-NEXT: [[VLA_EXPR:%.*]] = alloca i64, align 8
  // CHECK-NEXT: store i32 {{%.*}}, ptr [[N]], align 4

  id x[2][n][3];

  // Capture the VLA size.
  // CHECK-NEXT: [[T0:%.*]] = load i32, ptr [[N]], align 4
  // CHECK-NEXT: [[DIM:%.*]] = zext i32 [[T0]] to i64

  // CHECK-NEXT: [[T0:%.*]] = call ptr @llvm.stacksave.p0()
  // CHECK-NEXT: store ptr [[T0]], ptr [[SAVED_STACK]]


  // Allocate the VLA.
  // CHECK-NEXT: [[T0:%.*]] = mul nuw i64 2, [[DIM]]
  // CHECK-NEXT: [[VLA:%.*]] = alloca [3 x ptr], i64 [[T0]], align 16

  // Store the VLA #elements expression.
  // CHECK-NEXT: store i64 [[DIM]], ptr [[VLA_EXPR]], align 8

  // Zero-initialize.
  // CHECK-NEXT: [[T1:%.*]] = mul nuw i64 2, [[DIM]]
  // CHECK-NEXT: [[T2:%.*]] = mul nuw i64 [[T1]], 24
  // CHECK-NEXT: call void @llvm.memset.p0.i64(ptr align 16 [[VLA]], i8 0, i64 [[T2]], i1 false)

  // Destroy.
  // CHECK-NEXT: [[T0:%.*]] = mul nuw i64 2, [[DIM]]
  // CHECK-NEXT: [[BEGIN:%.*]] = getelementptr inbounds [3 x ptr], ptr [[VLA]], i32 0, i32 0
  // CHECK-NEXT: [[T1:%.*]] = mul nuw i64 [[T0]], 3
  // CHECK-NEXT: [[END:%.*]] = getelementptr inbounds ptr, ptr [[BEGIN]], i64 [[T1]]
  // CHECK-NEXT: [[EMPTY:%.*]] = icmp eq ptr [[BEGIN]], [[END]]
  // CHECK-NEXT: br i1 [[EMPTY]]

  // CHECK:      [[AFTER:%.*]] = phi ptr [ [[END]], {{%.*}} ], [ [[CUR:%.*]], {{%.*}} ]
  // CHECK-NEXT: [[CUR:%.*]] = getelementptr inbounds ptr, ptr [[AFTER]], i64 -1
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[CUR]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]]) [[NUW]], !clang.imprecise_release
  // CHECK-NEXT: [[EQ:%.*]] = icmp eq ptr [[CUR]], [[BEGIN]]
  // CHECK-NEXT: br i1 [[EQ]],

  // CHECK:      [[T0:%.*]] = load ptr, ptr [[SAVED_STACK]]
  // CHECK-NEXT: call void @llvm.stackrestore.p0(ptr [[T0]])
  // CHECK-NEXT: ret void
}

//   Note that we no longer emit .release_ivars flags.
//   Note that we set the flag saying that we need destruction *and*
//   the flag saying that we don't also need construction.
// CHECK-GLOBALS: @"_OBJC_CLASS_RO_$_Test23" = internal global [[RO_T:%.*]] { i32 390,
@interface Test23 { id x; } @end
@implementation Test23 @end

// CHECK-GLOBALS: @"_OBJC_CLASS_RO_$_Test24" = internal global [[RO_T:%.*]] { i32 130,
@interface Test24 {} @end
@implementation Test24 @end

@interface Test26 { id x[4]; } @end
@implementation Test26 @end
// CHECK:    define internal void @"\01-[Test26 .cxx_destruct]"(
// CHECK:      [[SELF:%.*]] = load ptr, ptr
// CHECK-NEXT: [[OFFSET:%.*]] = load i64, ptr @"OBJC_IVAR_$_Test26.x"
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds i8, ptr [[SELF]], i64 [[OFFSET]]
// CHECK-NEXT: [[BEGIN:%.*]] = getelementptr inbounds [4 x ptr], ptr [[T1]], i32 0, i32 0
// CHECK-NEXT: [[END:%.*]] = getelementptr inbounds ptr, ptr [[BEGIN]], i64 4
// CHECK-NEXT: br label
// CHECK:      [[PAST:%.*]] = phi ptr [ [[END]], {{%.*}} ], [ [[CUR:%.*]], {{%.*}} ]
// CHECK-NEXT: [[CUR]] = getelementptr inbounds ptr, ptr [[PAST]], i64 -1
// CHECK-NEXT: call void @llvm.objc.storeStrong(ptr [[CUR]], ptr null)
// CHECK-NEXT: [[ISDONE:%.*]] = icmp eq ptr [[CUR]], [[BEGIN]]
// CHECK-NEXT: br i1 [[ISDONE]],
// CHECK:      ret void

// Check that 'init' retains self.
@interface Test27
- (id) init;
@end
@implementation Test27
- (id) init { return self; }
// CHECK:    define internal ptr @"\01-[Test27 init]"
// CHECK:      [[SELF:%.*]] = alloca ptr,
// CHECK-NEXT: [[CMD:%.*]] = alloca ptr,
// CHECK-NEXT: store ptr {{%.*}}, ptr [[SELF]]
// CHECK-NEXT: store ptr {{%.*}}, ptr [[CMD]]
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[SELF]]
// CHECK-NEXT: [[T2:%.*]] = call ptr @llvm.objc.retain(ptr [[T0]])
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[SELF]]
// CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]])
// CHECK-NEXT: ret ptr [[T2]]

@end

@interface Test28
@property (copy) id prop;
@end
@implementation Test28
@synthesize prop;
@end
// CHECK:    define internal void @"\01-[Test28 .cxx_destruct]"
// CHECK:      [[SELF:%.*]] = load ptr, ptr
// CHECK-NEXT: [[OFFSET:%.*]] = load i64, ptr @"OBJC_IVAR_$_Test28.prop"
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds i8, ptr [[SELF]], i64 [[OFFSET]]
// CHECK-NEXT: call void @llvm.objc.storeStrong(ptr [[T1]], ptr null)
// CHECK-NEXT: ret void

@interface Test29_super
- (id) initWithAllocator: (id) allocator;
@end
@interface Test29 : Test29_super
- (id) init;
- (id) initWithAllocator: (id) allocator;
@end
@implementation Test29
static id _test29_allocator = 0;
- (id) init {
// CHECK:    define internal ptr @"\01-[Test29 init]"(ptr noundef {{%.*}},
// CHECK:      [[SELF:%.*]] = alloca ptr, align 8
// CHECK-NEXT: [[CMD:%.*]] = alloca ptr, align 8
// CHECK-NEXT: store ptr {{%.*}}, ptr [[SELF]]
// CHECK-NEXT: store ptr {{%.*}}, ptr [[CMD]]

// Evaluate arguments.  Note that the send argument is evaluated
// before the zeroing of self.
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[SELF]], align 8
// CHECK-NEXT: [[T1:%.*]] = load ptr, ptr @_test29_allocator, align 8

// Implicit null of 'self', i.e. direct transfer of ownership.
// CHECK-NEXT: store ptr null, ptr [[SELF]]

// Actual message send.
// CHECK-NEXT: [[SEL:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_
// CHECK-NEXT: [[CALL:%.*]] = call ptr @objc_msgSend(ptr noundef [[T0]], ptr noundef [[SEL]], ptr noundef [[T1]])

// Implicit write of result back into 'self'.  This is not supposed to
// be detectable because we're supposed to ban accesses to the old
// self value past the delegate init call.
// CHECK-NEXT: store ptr [[CALL]], ptr [[SELF]]

// Return statement.
// CHECK-NEXT: [[T1:%.*]] = call ptr @llvm.objc.retain(ptr [[CALL]]) [[NUW]]

// Cleanup.
// CHECK-NEXT: [[T2:%.*]] = load ptr, ptr [[SELF]]
// CHECK-NEXT: call void @llvm.objc.release(ptr [[T2]]) [[NUW]], !clang.imprecise_release

// Return.
// CHECK-NEXT: ret ptr [[T1]]
  return [self initWithAllocator: _test29_allocator];
}
- (id) initWithAllocator: (id) allocator {
// CHECK:    define internal ptr @"\01-[Test29 initWithAllocator:]"(
// CHECK:      [[SELF:%.*]] = alloca ptr, align 8
// CHECK-NEXT: [[CMD:%.*]] = alloca ptr, align 8
// CHECK-NEXT: [[ALLOCATOR:%.*]] = alloca ptr, align 8
// CHECK-NEXT: alloca
// CHECK-NEXT: store ptr {{%.*}}, ptr [[SELF]]
// CHECK-NEXT: store ptr {{%.*}}, ptr [[CMD]]
// CHECK-NEXT: [[T0:%.*]] = call ptr @llvm.objc.retain(ptr {{%.*}})
// CHECK-NEXT: store ptr [[T0]], ptr [[ALLOCATOR]]

// Evaluate arguments.  Note that the send argument is evaluated
// before the zeroing of self.
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[SELF]]
// CHECK-NEXT: [[T1:%.*]] = load ptr, ptr [[ALLOCATOR]], align 8

// Implicit null of 'self', i.e. direct transfer of ownership.
// CHECK-NEXT: store ptr null, ptr [[SELF]]

// Actual message send.
// CHECK:      [[CALL:%.*]] = call {{.*}} @objc_msgSendSuper2

// Implicit write of result back into 'self'.  This is not supposed to
// be detectable because we're supposed to ban accesses to the old
// self value past the delegate init call.
// CHECK-NEXT: store ptr [[CALL]], ptr [[SELF]]

// Assignment.
// CHECK-NEXT: [[T2:%.*]] = call ptr @llvm.objc.retain(ptr [[CALL]]) [[NUW]]
// CHECK-NEXT: [[T4:%.*]] = load ptr, ptr [[SELF]], align
// CHECK-NEXT: store ptr [[T2]], ptr [[SELF]], align
// CHECK-NEXT: call void @llvm.objc.release(ptr [[T4]])

// Return statement.
// CHECK-NEXT: [[T3:%.*]] = load ptr, ptr [[SELF]]
// CHECK-NEXT: [[T4:%.*]] = call ptr @llvm.objc.retain(ptr [[T3]]) [[NUW]]

// Cleanup.
// CHECK-NEXT: [[T5:%.*]] = load ptr, ptr [[ALLOCATOR]]
// CHECK-NEXT: call void @llvm.objc.release(ptr [[T5]]) [[NUW]], !clang.imprecise_release

// CHECK-NEXT: [[T6:%.*]] = load ptr, ptr [[SELF]]
// CHECK-NEXT: call void @llvm.objc.release(ptr [[T6]]) [[NUW]], !clang.imprecise_release

// Return.
// CHECK-NEXT: ret ptr [[T4]]
  self = [super initWithAllocator: allocator];
  return self;
}
@end

typedef struct Test30_helper Test30_helper;
@interface Test30
- (id) init;
- (Test30_helper*) initHelper;
@end
@implementation Test30 {
char *helper;
}
- (id) init {
// CHECK:    define internal ptr @"\01-[Test30 init]"(ptr noundef {{%.*}},
// CHECK:      [[RET:%.*]] = alloca ptr
// CHECK-NEXT: alloca ptr
// CHECK-NEXT: store ptr {{%.*}}, ptr [[SELF]]
// CHECK-NEXT: store

// Call.
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[SELF]]
// CHECK-NEXT: [[SEL:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_
// CHECK-NEXT: [[CALL:%.*]] = call ptr @objc_msgSend(ptr noundef [[T0]], ptr noundef [[SEL]])

// Assignment.
// CHECK-NEXT: [[T1:%.*]] = load ptr, ptr [[SELF]]
// CHECK-NEXT: [[IVAR:%.*]] = load i64, ptr @"OBJC_IVAR_$_Test30.helper"
// CHECK-NEXT: [[T3:%.*]] = getelementptr inbounds i8, ptr [[T1]], i64 [[IVAR]]
// CHECK-NEXT#: [[T5:%.*]] = load ptr, ptr [[T3]]
// CHECK-NEXT#: [[T6:%.*]] = call ptr @llvm.objc.retain(ptr [[CALL]])
// CHECK-NEXT#: call void @llvm.objc.release(ptr [[T5]])
// CHECK-NEXT: store ptr [[CALL]], ptr [[T3]]

// Return.
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[SELF]]
// CHECK-NEXT: [[T1:%.*]] = call ptr @llvm.objc.retain(ptr [[T0]])

// Cleanup.
// CHECK-NEXT: [[T2:%.*]] = load ptr, ptr [[SELF]]
// CHECK-NEXT: call void @llvm.objc.release(ptr [[T2]])

// Epilogue.
// CHECK-NEXT: ret ptr [[T1]]
  self->helper = [self initHelper];
  return self;
}
- (Test30_helper*) initHelper {
// CHECK:    define internal ptr @"\01-[Test30 initHelper]"(
// CHECK:      alloca
// CHECK-NEXT: alloca
// CHECK-NEXT: store
// CHECK-NEXT: store
// CHECK-NEXT: ret ptr null
  return 0;
}

@end

__attribute__((ns_returns_retained)) id test32(void) {
// CHECK-LABEL:    define{{.*}} ptr @test32()
// CHECK:      [[T0:%.*]] = call ptr @test32_helper(){{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T0]])
// CHECK-NEXT: ret ptr [[T0]]
  extern id test32_helper(void);
  return test32_helper();
}

@class Test33_a;
@interface Test33
- (void) give: (Test33_a **) x;
- (void) take: (Test33_a **) x;
- (void) giveStrong: (out __strong Test33_a **) x;
- (void) takeStrong: (inout __strong Test33_a **) x;
- (void) giveOut: (out Test33_a **) x;
@end
void test33(Test33 *ptr) {
  Test33_a *a;
  [ptr give: &a];
  [ptr take: &a];
  [ptr giveStrong: &a];
  [ptr takeStrong: &a];
  [ptr giveOut: &a];

  // CHECK:    define{{.*}} void @test33(ptr
  // CHECK:      [[PTR:%.*]] = alloca ptr
  // CHECK-NEXT: [[A:%.*]] = alloca ptr
  // CHECK-NEXT: [[TEMP0:%.*]] = alloca ptr
  // CHECK-NEXT: [[TEMP1:%.*]] = alloca ptr
  // CHECK-NEXT: [[TEMP2:%.*]] = alloca ptr
  // CHECK-NEXT: llvm.objc.retain
  // CHECK-NEXT: store
  // CHECK-NEXT: call void @llvm.lifetime.start
  // CHECK-NEXT: store ptr null, ptr [[A]]

  // CHECK-NEXT: load ptr, ptr [[PTR]]
  // CHECK-NEXT: [[W0:%.*]] = load ptr, ptr [[A]]
  // CHECK-NEXT: store ptr [[W0]], ptr [[TEMP0]]
  // CHECK-NEXT: load ptr, ptr @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: objc_msgSend{{.*}}, ptr noundef [[TEMP0]])
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[TEMP0]]
  // CHECK-NEXT: [[T2:%.*]] = call ptr @llvm.objc.retain(ptr [[T0]])
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.use(ptr [[W0]]) [[NUW]]
  // CHECK-NEXT: [[T4:%.*]] = load ptr, ptr [[A]]
  // CHECK-NEXT: store ptr [[T2]], ptr [[A]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T4]])

  // CHECK-NEXT: load ptr, ptr [[PTR]]
  // CHECK-NEXT: [[W0:%.*]] = load ptr, ptr [[A]]
  // CHECK-NEXT: store ptr [[W0]], ptr [[TEMP1]]
  // CHECK-NEXT: load ptr, ptr @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: objc_msgSend{{.*}}, ptr noundef [[TEMP1]])
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[TEMP1]]
  // CHECK-NEXT: [[T2:%.*]] = call ptr @llvm.objc.retain(ptr [[T0]])
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.use(ptr [[W0]]) [[NUW]]
  // CHECK-NEXT: [[T4:%.*]] = load ptr, ptr [[A]]
  // CHECK-NEXT: store ptr [[T2]], ptr [[A]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T4]])

  // CHECK-NEXT: load ptr, ptr [[PTR]]
  // CHECK-NEXT: load ptr, ptr @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: objc_msgSend{{.*}}, ptr noundef [[A]])

  // CHECK-NEXT: load ptr, ptr [[PTR]]
  // CHECK-NEXT: load ptr, ptr @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: objc_msgSend{{.*}}, ptr noundef [[A]])

  // 'out'
  // CHECK-NEXT: load ptr, ptr [[PTR]]
  // CHECK-NEXT: store ptr null, ptr [[TEMP2]]
  // CHECK-NEXT: load ptr, ptr @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: objc_msgSend{{.*}}, ptr noundef [[TEMP2]])
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[TEMP2]]
  // CHECK-NEXT: [[T2:%.*]] = call ptr @llvm.objc.retain(ptr [[T0]])
  // CHECK-NEXT: [[T4:%.*]] = load ptr, ptr [[A]]
  // CHECK-NEXT: store ptr [[T2]], ptr [[A]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T4]])

  // CHECK-NEXT: load
  // CHECK-NEXT: llvm.objc.release
  // CHECK-NEXT: call void @llvm.lifetime.end
  // CHECK-NEXT: load
  // CHECK-NEXT: llvm.objc.release
  // CHECK-NEXT: ret void
}


// CHECK-LABEL: define{{.*}} void @test36
void test36(id x) {
  // CHECK: [[X:%.*]] = alloca ptr

  // CHECK: call ptr @llvm.objc.retain
  // CHECK: call ptr @llvm.objc.retain
  // CHECK: call ptr @llvm.objc.retain
  id array[3] = { @"A", x, @"y" };

  // CHECK:      [[T0:%.*]] = load ptr, ptr [[X]]
  // CHECK-NEXT: store ptr null, ptr [[X]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]])
  x = 0;

  // CHECK: br label
  // CHECK: call void @llvm.objc.release
  // CHECK: br i1

  // CHECK: call void @llvm.objc.release
  // CHECK-NEXT: ret void
}

@class Test37;
void test37(void) {
  extern void test37_helper(id *);
  Test37 *var;
  test37_helper(&var);

  // CHECK-LABEL:    define{{.*}} void @test37()
  // CHECK:      [[VAR:%.*]] = alloca ptr,
  // CHECK-NEXT: [[TEMP:%.*]] = alloca ptr
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[VAR]])
  // CHECK-NEXT: store ptr null, ptr [[VAR]]

  // CHECK-NEXT: [[W0:%.*]] = load ptr, ptr [[VAR]]
  // CHECK-NEXT: store ptr [[W0]], ptr [[TEMP]]
  // CHECK-NEXT: call void @test37_helper(ptr noundef [[TEMP]])
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[TEMP]]
  // CHECK-NEXT: [[T3:%.*]] = call ptr @llvm.objc.retain(ptr [[T0]])
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.use(ptr [[W0]]) [[NUW]]
  // CHECK-NEXT: [[T5:%.*]] = load ptr, ptr [[VAR]]
  // CHECK-NEXT: store ptr [[T3]], ptr [[VAR]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T5]])

  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[VAR]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]])
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr [[VAR]])
  // CHECK-NEXT: ret void
}

@interface Test43 @end
@implementation Test43
- (id) test __attribute__((ns_returns_retained)) {
  extern id test43_produce(void);
  return test43_produce();
  // CHECK:      call ptr @test43_produce(){{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(
  // CHECK-NEXT: ret
}
@end

@interface Test45
@property (retain) id x;
@end
@implementation Test45
@synthesize x;
@end
// CHECK:    define internal ptr @"\01-[Test45 x]"(
// CHECK:      [[CALL:%.*]] = tail call ptr @objc_getProperty(
// CHECK-NEXT: ret ptr [[CALL]]

void test46(__weak id *wp, __weak volatile id *wvp) {
  extern id test46_helper(void);

  // TODO: this is sub-optimal, we should retain at the actual call site.

  // CHECK:      [[T1:%.*]] = call ptr @test46_helper(){{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T1]])
  // CHECK-NEXT: [[T2:%.*]] = load ptr, ptr {{%.*}}, align 8
  // CHECK-NEXT: [[T3:%.*]] = call ptr @llvm.objc.storeWeak(ptr [[T2]], ptr [[T1]])
  // CHECK-NEXT: [[T4:%.*]] = call ptr @llvm.objc.retain(ptr [[T3]])
  // CHECK-NEXT: store ptr [[T4]], ptr
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T1]])
  id x = *wp = test46_helper();

  // CHECK:      [[T1:%.*]] = call ptr @test46_helper(){{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T1]])
  // CHECK-NEXT: [[T2:%.*]] = load ptr, ptr {{%.*}}, align 8
  // CHECK-NEXT: [[T3:%.*]] = call ptr @llvm.objc.storeWeak(ptr [[T2]], ptr [[T1]])
  // CHECK-NEXT: [[T4:%.*]] = call ptr @llvm.objc.retain(ptr [[T3]])
  // CHECK-NEXT: store ptr [[T4]], ptr
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T1]])
  id y = *wvp = test46_helper();
}

void test47(void) {
  extern id test47_helper(void);
  id x = x = test47_helper();

  // CHECK-LABEL:    define{{.*}} void @test47()
  // CHECK:      [[X:%.*]] = alloca ptr
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[X]])
  // CHECK-NEXT: store ptr null, ptr [[X]]
  // CHECK-NEXT: [[T0:%.*]] = call ptr @test47_helper(){{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T0]])
  // CHECK-NEXT: [[T1:%.*]] = load ptr, ptr [[X]]
  // CHECK-NEXT: store ptr [[T0]], ptr [[X]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T1]])
  // CHECK-NEXT: [[T2:%.*]] = call ptr @llvm.objc.retain(ptr [[T0]])
  // CHECK-NEXT: [[T3:%.*]] = load ptr, ptr [[X]]
  // CHECK-NEXT: store ptr [[T2]], ptr [[X]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T3]])
  // CHECK-NEXT: [[T4:%.*]] = load ptr, ptr [[X]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T4]])
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr [[X]])
  // CHECK-NEXT: ret void
}

void test48(void) {
  extern id test48_helper(void);
  __weak id x = x = test48_helper();
  // CHECK-LABEL:    define{{.*}} void @test48()
  // CHECK:      [[X:%.*]] = alloca ptr
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[X]])
  // CHECK-NEXT: [[T0:%.*]] = call ptr @llvm.objc.initWeak(ptr [[X]], ptr null)
  // CHECK-NEXT: [[T2:%.*]] = call ptr @test48_helper(){{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T2]])
  // CHECK-NEXT: [[T3:%.*]] = call ptr @llvm.objc.storeWeak(ptr [[X]], ptr [[T2]])
  // CHECK-NEXT: [[T4:%.*]] = call ptr @llvm.objc.storeWeak(ptr [[X]], ptr [[T3]])
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T2]])
  // CHECK-NEXT: call void @llvm.objc.destroyWeak(ptr [[X]])
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr [[X]])
  // CHECK-NEXT: ret void
}

void test49(void) {
  extern id test49_helper(void);
  __autoreleasing id x = x = test49_helper();
  // CHECK-LABEL:    define{{.*}} void @test49()
  // CHECK:      [[X:%.*]] = alloca ptr
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[X]])
  // CHECK-NEXT: store ptr null, ptr [[X]]
  // CHECK-NEXT: [[T0:%.*]] = call ptr @test49_helper(){{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T0]])
  // CHECK-NEXT: [[T1:%.*]] = call ptr @llvm.objc.autorelease(ptr [[T0]])
  // CHECK-NEXT: store ptr [[T1]], ptr [[X]]
  // CHECK-NEXT: [[T3:%.*]] = call ptr @llvm.objc.retainAutorelease(ptr [[T1]])
  // CHECK-NEXT: store ptr [[T3]], ptr [[X]]
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr [[X]])
  // CHECK-NEXT: ret void
}

id x(void);
void test50(id y) {
  ({x();});
// CHECK: [[T0:%.*]] = call ptr @llvm.objc.retain
// CHECK: call void @llvm.objc.release
}

struct CGPoint {
  float x;
  float y;
};
typedef struct CGPoint CGPoint;

@interface Foo
@property (assign) CGPoint point;
@end

@implementation Foo
@synthesize point;
@end

id test52(void) {
  id test52_helper(int) __attribute__((ns_returns_retained));
  return ({ int x = 5; test52_helper(x); });

// CHECK-LABEL:    define{{.*}} ptr @test52()
// CHECK:      [[X:%.*]] = alloca i32
// CHECK-NEXT: [[TMPALLOCA:%.*]] = alloca ptr
// CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 4, ptr [[X]])
// CHECK-NEXT: store i32 5, ptr [[X]],
// CHECK-NEXT: [[T0:%.*]] = load i32, ptr [[X]],
// CHECK-NEXT: [[T1:%.*]] = call ptr @test52_helper(i32 noundef [[T0]])
// CHECK-NEXT: store ptr [[T1]], ptr [[TMPALLOCA]]
// CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 4, ptr [[X]])
// CHECK-NEXT: [[T2:%.*]] = load ptr, ptr [[TMPALLOCA]]
// CHECK-NEXT: [[T3:%.*]] = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr [[T2]])
// CHECK-NEXT: ret ptr [[T3]]
}

void test53(void) {
  id test53_helper(void);
  id x = ({ id y = test53_helper(); y; });
  (void) x;
// CHECK-LABEL:    define{{.*}} void @test53()
// CHECK:      [[X:%.*]] = alloca ptr,
// CHECK-NEXT: [[Y:%.*]] = alloca ptr,
// CHECK-NEXT: [[TMPALLOCA:%.*]] = alloca ptr,
// CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[X]])
// CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[Y]])
// CHECK-NEXT: [[T1:%.*]] = call ptr @test53_helper(){{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T1]])
// CHECK-NEXT: store ptr [[T1]], ptr [[Y]],
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[Y]],
// CHECK-NEXT: [[T1:%.*]] = call ptr @llvm.objc.retain(ptr [[T0]])
// CHECK-NEXT: store ptr [[T1]], ptr [[TMPALLOCA]]
// CHECK-NEXT: [[T2:%.*]] = load ptr, ptr [[Y]]
// CHECK-NEXT: call void @llvm.objc.release(ptr [[T2]])
// CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr [[Y]])
// CHECK-NEXT: [[T3:%.*]] = load ptr, ptr [[TMPALLOCA]]
// CHECK-NEXT: store ptr [[T3]], ptr [[X]],
// CHECK-NEXT: load ptr, ptr [[X]],
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[X]]
// CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]])
// CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr [[X]])
// CHECK-NEXT: ret void
}

// CHECK-LABEL: define{{.*}} void @test54(i32 noundef %first, ...)
void test54(int first, ...) {
  __builtin_va_list arglist;
  // CHECK: call void @llvm.va_start
  __builtin_va_start(arglist, first);
  // CHECK: call ptr @llvm.objc.retain
  id obj = __builtin_va_arg(arglist, id);
  // CHECK: call void @llvm.va_end
  __builtin_va_end(arglist);
  // CHECK: call void @llvm.objc.release
  // CHECK: ret void
}

// PR10228
@interface Test55Base @end
@interface Test55 : Test55Base @end
@implementation Test55 (Category)
- (void) dealloc {}
@end
// CHECK:   define internal void @"\01-[Test55(Category) dealloc]"(
// CHECK-NOT: ret
// CHECK:     call void @objc_msgSendSuper2(

@protocol Test56Protocol
+ (id) make __attribute__((ns_returns_retained));
@end
@interface Test56<Test56Protocol> @end
@implementation Test56
// CHECK: define internal ptr @"\01+[Test56 make]"(
+ (id) make {
  extern id test56_helper(void);
  // CHECK:      [[T1:%.*]] = call ptr @test56_helper(){{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T1]])
  // CHECK-NEXT: ret ptr [[T1]]
  return test56_helper();
}
@end
void test56_test(void) {
  id x = [Test56 make];
  // CHECK-LABEL: define{{.*}} void @test56_test()
  // CHECK:      [[X:%.*]] = alloca ptr, align 8
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[X]])
  // CHECK:      [[T0:%.*]] = call ptr @objc_msgSend(
  // CHECK-NEXT: store ptr [[T0]], ptr [[X]]
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[X]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]])
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr [[X]])
  // CHECK-NEXT: ret void
}

@interface Test57
@property (nonatomic, strong) id strong;
@property (nonatomic, weak) id weak;
@property (nonatomic, unsafe_unretained) id unsafe;
@end
@implementation Test57
@synthesize strong, weak, unsafe;
@end
// CHECK: define internal ptr @"\01-[Test57 strong]"(
// CHECK:      [[T0:%.*]] = load ptr, ptr {{%.*}}
// CHECK-NEXT: [[T1:%.*]] = load i64, ptr @"OBJC_IVAR_$_Test57.strong"
// CHECK-NEXT: [[T3:%.*]] = getelementptr inbounds i8, ptr [[T0]], i64 [[T1]]
// CHECK-NEXT: [[T5:%.*]] = load ptr, ptr [[T3]]
// CHECK-NEXT: ret ptr [[T5]]

// CHECK: define internal ptr @"\01-[Test57 weak]"(
// CHECK:      [[T0:%.*]] = load ptr, ptr {{%.*}}
// CHECK-NEXT: [[T1:%.*]] = load i64, ptr @"OBJC_IVAR_$_Test57.weak"
// CHECK-NEXT: [[T3:%.*]] = getelementptr inbounds i8, ptr [[T0]], i64 [[T1]]
// CHECK-NEXT: [[T5:%.*]] = call ptr @llvm.objc.loadWeakRetained(ptr [[T3]])
// CHECK-NEXT: [[T6:%.*]] = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr [[T5]])
// CHECK-NEXT: ret ptr [[T6]]

// CHECK: define internal ptr @"\01-[Test57 unsafe]"(
// CHECK:      [[T0:%.*]] = load ptr, ptr {{%.*}}
// CHECK-NEXT: [[T1:%.*]] = load i64, ptr @"OBJC_IVAR_$_Test57.unsafe"
// CHECK-NEXT: [[T3:%.*]] = getelementptr inbounds i8, ptr [[T0]], i64 [[T1]]
// CHECK-NEXT: [[T5:%.*]] = load ptr, ptr [[T3]]
// CHECK-NEXT: ret ptr [[T5]]

void test59(void) {
  extern id test59_getlock(void);
  extern void test59_body(void);
  @synchronized (test59_getlock()) {
    test59_body();
  }

  // CHECK-LABEL:    define{{.*}} void @test59()
  // CHECK:      [[T1:%.*]] = call ptr @test59_getlock(){{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T1]])
  // CHECK-NEXT: call i32 @objc_sync_enter(ptr [[T1]])
  // CHECK-NEXT: call void @test59_body()
  // CHECK-NEXT: call i32 @objc_sync_exit(ptr [[T1]])
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T1]])
  // CHECK-NEXT: ret void
}

// Verify that we don't try to reclaim the result of performSelector.
@interface Test61
- (id) performSelector: (SEL) selector;
- (void) test61_void;
- (id) test61_id;
@end
void test61(void) {
  // CHECK-LABEL:    define{{.*}} void @test61()
  // CHECK:      [[Y:%.*]] = alloca ptr, align 8

  extern id test61_make(void);

  // CHECK-NEXT: [[T0:%.*]] = call ptr @test61_make(){{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T0]])
  // CHECK-NEXT: [[T2:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[T3:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[T4:%.*]] = call ptr @objc_msgSend(ptr noundef [[T1]], ptr noundef [[T3]], ptr noundef [[T2]])
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T1]])
  [test61_make() performSelector: @selector(test61_void)];

  // CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[Y]])
  // CHECK-NEXT: [[T1:%.*]] = call ptr @test61_make(){{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T1]])
  // CHECK-NEXT: [[T2:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[T3:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[T5:%.*]] = call ptr @objc_msgSend(ptr noundef [[T1]], ptr noundef [[T3]], ptr noundef [[T2]]){{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T5]])
  // CHECK-NEXT: store ptr [[T5]], ptr [[Y]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T1]])
  id y = [test61_make() performSelector: @selector(test61_id)];

  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[Y]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]])
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr [[Y]])
  // CHECK-NEXT: ret void
}

void test62(void) {
  // CHECK-LABEL:    define{{.*}} void @test62()
  // CHECK:      [[I:%.*]] = alloca i32, align 4
  // CHECK-NEXT: [[CLEANUP_VALUE:%.*]] = alloca ptr
  // CHECK-NEXT: [[CLEANUP_REQUIRED:%.*]] = alloca i1
  extern id test62_make(void);
  extern void test62_body(void);

  // CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 4, ptr [[I]])
  // CHECK-NEXT: store i32 0, ptr [[I]], align 4
  // CHECK-NEXT: br label

  // CHECK:      [[T0:%.*]] = load i32, ptr [[I]], align 4
  // CHECK-NEXT: [[T1:%.*]] = icmp ne i32 [[T0]], 20
  // CHECK-NEXT: br i1 [[T1]],

  for (unsigned i = 0; i != 20; ++i) {
    // CHECK:      [[T0:%.*]] = load i32, ptr [[I]], align 4
    // CHECK-NEXT: [[T1:%.*]] = icmp ne i32 [[T0]], 0
    // CHECK-NEXT: store i1 false, ptr [[CLEANUP_REQUIRED]]
    // CHECK-NEXT: br i1 [[T1]],
    // CHECK:      [[T1:%.*]] = call ptr @test62_make(){{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
    // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T1]])
    // CHECK-NEXT: store ptr [[T1]], ptr [[CLEANUP_VALUE]]
    // CHECK-NEXT: store i1 true, ptr [[CLEANUP_REQUIRED]]
    // CHECK-NEXT: [[T2:%.*]] = icmp ne ptr [[T1]], null
    // CHECK-NEXT: br label
    // CHECK:      [[COND:%.*]] = phi i1 [ false, {{%.*}} ], [ [[T2]], {{%.*}} ]
    // CHECK-NEXT: [[T0:%.*]] = load i1, ptr [[CLEANUP_REQUIRED]]
    // CHECK-NEXT: br i1 [[T0]],
    // CHECK:      [[T0:%.*]] = load ptr, ptr [[CLEANUP_VALUE]]
    // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]])
    // CHECK-NEXT: br label
    // CHECK:      br i1 [[COND]]
    // CHECK:      call void @test62_body()
    // CHECK-NEXT: br label
    // CHECK:      br label
    if (i != 0 && test62_make() != 0)
      test62_body();
  }

  // CHECK:      [[T0:%.*]] = load i32, ptr [[I]], align 4
  // CHECK-NEXT: [[T1:%.*]] = add i32 [[T0]], 1
  // CHECK-NEXT: store i32 [[T1]], ptr [[I]]
  // CHECK-NEXT: br label

  // CHECK:      ret void
}

@class NSString;

@interface Person  {
  NSString *name;
}
@property NSString *address;
@end

@implementation Person
@synthesize address;
@end
// CHECK: tail call ptr @objc_getProperty
// CHECK: call void @objc_setProperty

// Verify that we successfully parse and preserve this attribute in
// this position.
@interface Test66
- (void) consume: (id __attribute__((ns_consumed))) ptr;
@end
void test66(void) {
  extern Test66 *test66_receiver(void);
  extern id test66_arg(void);
  [test66_receiver() consume: test66_arg()];
}
// CHECK-LABEL:    define{{.*}} void @test66()
// CHECK:      [[T3:%.*]] = call ptr @test66_receiver(){{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T3]])
// CHECK-NEXT: [[T4:%.*]] = call ptr @test66_arg(){{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T4]])
// CHECK-NEXT: [[SIX:%.*]] = icmp eq ptr [[T3]], null
// CHECK-NEXT: br i1 [[SIX]], label [[NULINIT:%.*]], label %[[CALL:.*]]
// CHECK:      [[CALL]]:
// CHECK-NEXT: [[SEL:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES
// CHECK-NEXT: call void @objc_msgSend(ptr noundef [[T3]], ptr noundef [[SEL]], ptr noundef [[T4]])
// CHECK-NEXT: br label [[CONT:%.*]]
// CHECK: call void @llvm.objc.release(ptr [[T4]]) [[NUW]]
// CHECK-NEXT: br label [[CONT:%.*]]
// CHECK: call void @llvm.objc.release(ptr [[T3]])
// CHECK-NEXT: ret void

Class test67_helper(void);
void test67(void) {
  Class cl = test67_helper();
}
// CHECK-LABEL:    define{{.*}} void @test67()
// CHECK:      [[CL:%.*]] = alloca ptr, align 8
// CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[CL]])
// CHECK-NEXT: [[T0:%.*]] = call ptr @test67_helper()
// CHECK-NEXT: store ptr [[T0]], ptr [[CL]], align 8
// CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr [[CL]])
// CHECK-NEXT: ret void

Class test68_helper(void);
void test68(void) {
  __strong Class cl = test67_helper();
}
// CHECK-LABEL:    define{{.*}} void @test68()
// CHECK:      [[CL:%.*]] = alloca ptr, align 8
// CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[CL]])
// CHECK-NEXT: [[T1:%.*]] = call ptr @test67_helper(){{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T1]])
// CHECK-NEXT: store ptr [[T1]], ptr [[CL]], align 8
// CHECK-NEXT: [[T2:%.*]] = load ptr, ptr [[CL]]
// CHECK-NEXT: call void @llvm.objc.release(ptr [[T2]])
// CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr [[CL]])
// CHECK-NEXT: ret void

@interface Test69 @end
@implementation Test69
- (id) foo { return self; }
@end
// CHECK: define internal ptr @"\01-[Test69 foo]"(
// CHECK:      [[SELF:%.*]] = alloca ptr, align 8
// CHECK:      [[T0:%.*]] = load ptr, ptr [[SELF]], align 8
// CHECK-NEXT: ret ptr [[T0]]

void test70(id i) {
  // CHECK-LABEL: define{{.*}} void @test70
  // CHECK: store ptr null, ptr
  // CHECK: store ptr null, ptr
  // CHECK: [[ID:%.*]] = call ptr @llvm.objc.retain(ptr
  // CHECK: store ptr [[ID]], ptr
  id x[3] = {
    [2] = i
  };
}

// Be sure that we emit lifetime intrinsics only after dtors
struct AggDtor {
  char cs[40];
  id x;
};

struct AggDtor getAggDtor(void);

// CHECK-LABEL: define{{.*}} void @test71
void test71(void) {
  // CHECK: call void @llvm.lifetime.start.p0({{[^,]+}}, ptr %[[T:.*]])
  // CHECK: call void @getAggDtor(ptr sret(%struct.AggDtor) align 8 %[[T]])
  // CHECK: call void @__destructor_8_s40(ptr %[[T]])
  // CHECK: call void @llvm.lifetime.end.p0({{[^,]+}}, ptr %[[T]])
  // CHECK: call void @llvm.lifetime.start.p0({{[^,]+}}, ptr %[[T2:.*]])
  // CHECK: call void @getAggDtor(ptr sret(%struct.AggDtor) align 8 %[[T2]])
  // CHECK: call void @__destructor_8_s40(ptr %[[T2]])
  // CHECK: call void @llvm.lifetime.end.p0({{[^,]+}}, ptr %[[T2]])
  getAggDtor();
  getAggDtor();
}

// Check that no extra release calls are emitted to detruct the compond literal.

// CHECK: define{{.*}} void @test72(ptr noundef %[[A:.*]], ptr noundef %[[B:.*]])
// CHECK: %[[A_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[B_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[T:.*]] = alloca [2 x ptr], align 16
// CHECK: %[[V0:.*]] = call ptr @llvm.objc.retain(ptr %[[A]])
// CHECK: %[[V1:.*]] = call ptr @llvm.objc.retain(ptr %[[B]]) #2
// CHECK: %[[ARRAYINIT_BEGIN:.*]] = getelementptr inbounds [2 x ptr], ptr %[[T]], i64 0, i64 0
// CHECK: %[[V3:.*]] = load ptr, ptr %[[A_ADDR]], align 8, !tbaa !7
// CHECK: %[[V4:.*]] = call ptr @llvm.objc.retain(ptr %[[V3]]) #2
// CHECK: store ptr %[[V4]], ptr %[[ARRAYINIT_BEGIN]], align 8, !tbaa !7
// CHECK: %[[ARRAYINIT_ELEMENT:.*]] = getelementptr inbounds ptr, ptr %[[ARRAYINIT_BEGIN]], i64 1
// CHECK: %[[V5:.*]] = load ptr, ptr %[[B_ADDR]], align 8, !tbaa !7
// CHECK: %[[V6:.*]] = call ptr @llvm.objc.retain(ptr %[[V5]]) #2
// CHECK: store ptr %[[V6]], ptr %[[ARRAYINIT_ELEMENT]], align 8, !tbaa !7
// CHECK: %[[ARRAY_BEGIN:.*]] = getelementptr inbounds [2 x ptr], ptr %[[T]], i32 0, i32 0
// CHECK: %[[V7:.*]] = getelementptr inbounds ptr, ptr %[[ARRAY_BEGIN]], i64 2

// CHECK-NOT: call void @llvm.objc.release

// CHECK: %[[ARRAYDESTROY_ELEMENTPAST:.*]] = phi ptr [ %[[V7]], %{{.*}} ], [ %[[ARRAYDESTROY_ELEMENT:.*]], %{{.*}} ]
// CHECK: %[[ARRAYDESTROY_ELEMENT]] = getelementptr inbounds ptr, ptr %[[ARRAYDESTROY_ELEMENTPAST]], i64 -1
// CHECK: %[[V8:.*]] = load ptr, ptr %[[ARRAYDESTROY_ELEMENT]], align 8
// CHECK: call void @llvm.objc.release(ptr %[[V8]]) #2, !clang.imprecise_release

// CHECK-NOT: call void @llvm.objc.release

// CHECK: %[[V10:.*]] = load ptr, ptr %[[B_ADDR]], align 8
// CHECK: call void @llvm.objc.release(ptr %[[V10]]) #2, !clang.imprecise_release
// CHECK: %[[V11:.*]] = load ptr, ptr %[[A_ADDR]], align 8
// CHECK: call void @llvm.objc.release(ptr %[[V11]]) #2, !clang.imprecise_release

void test72(id a, id b) {
  __strong id t[] = (__strong id[]){a, b};
}

// ARC-ALIEN: attributes [[NLB]] = { nonlazybind }
// ARC-NATIVE: attributes [[NLB]] = { nonlazybind }
// CHECK: attributes [[NUW]] = { nounwind }
