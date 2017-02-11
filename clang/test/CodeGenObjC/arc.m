// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -Wno-objc-root-class -Wno-incompatible-pointer-types -Wno-arc-unsafe-retained-assign -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -O2 -disable-llvm-passes -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -Wno-objc-root-class -Wno-incompatible-pointer-types -Wno-arc-unsafe-retained-assign -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -o - %s | FileCheck -check-prefix=CHECK-GLOBALS %s

// rdar://13129783. Check both native/non-native arc platforms. Here we check
// that they treat nonlazybind differently.
// RUN: %clang_cc1 -fobjc-runtime=macosx-10.6.0 -triple x86_64-apple-darwin10 -Wno-objc-root-class -Wno-incompatible-pointer-types -Wno-arc-unsafe-retained-assign -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -o - %s | FileCheck -check-prefix=ARC-ALIEN %s
// RUN: %clang_cc1 -fobjc-runtime=macosx-10.7.0 -triple x86_64-apple-darwin11 -Wno-objc-root-class -Wno-incompatible-pointer-types -Wno-arc-unsafe-retained-assign -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -o - %s | FileCheck -check-prefix=ARC-NATIVE %s

// ARC-ALIEN: declare extern_weak void @objc_storeStrong(i8**, i8*)
// ARC-ALIEN: declare extern_weak i8* @objc_retain(i8* returned)
// ARC-ALIEN: declare extern_weak i8* @objc_autoreleaseReturnValue(i8* returned)
// ARC-ALIEN: declare i8* @objc_msgSend(i8*, i8*, ...) [[NLB:#[0-9]+]]
// ARC-ALIEN: declare extern_weak void @objc_release(i8*)
// ARC-ALIEN: declare extern_weak i8* @objc_retainAutoreleasedReturnValue(i8* returned)
// ARC-ALIEN: declare extern_weak i8* @objc_initWeak(i8**, i8*)
// ARC-ALIEN: declare extern_weak i8* @objc_storeWeak(i8**, i8*)
// ARC-ALIEN: declare extern_weak i8* @objc_loadWeakRetained(i8**)
// ARC-ALIEN: declare extern_weak void @objc_destroyWeak(i8**)
// declare extern_weak i8* @objc_autorelease(i8*)
// ARC-ALIEN: declare extern_weak i8* @objc_retainAutorelease(i8* returned)

// ARC-NATIVE: declare void @objc_storeStrong(i8**, i8*)
// ARC-NATIVE: declare i8* @objc_retain(i8* returned) [[NLB:#[0-9]+]]
// ARC-NATIVE: declare i8* @objc_autoreleaseReturnValue(i8* returned)
// ARC-NATIVE: declare i8* @objc_msgSend(i8*, i8*, ...) [[NLB]]
// ARC-NATIVE: declare void @objc_release(i8*) [[NLB]]
// ARC-NATIVE: declare i8* @objc_retainAutoreleasedReturnValue(i8* returned)
// ARC-NATIVE: declare i8* @objc_initWeak(i8**, i8*)
// ARC-NATIVE: declare i8* @objc_storeWeak(i8**, i8*)
// ARC-NATIVE: declare i8* @objc_loadWeakRetained(i8**)
// ARC-NATIVE: declare void @objc_destroyWeak(i8**)
// declare i8* @objc_autorelease(i8*)
// ARC-NATIVE: declare i8* @objc_retainAutorelease(i8* returned)

// CHECK-LABEL: define void @test0
void test0(id x) {
  // CHECK:      [[X:%.*]] = alloca i8*
  // CHECK-NEXT: [[PARM:%.*]] = call i8* @objc_retain(i8* {{.*}})
  // CHECK-NEXT: store i8* [[PARM]], i8** [[X]]
  // CHECK-NEXT: [[TMP:%.*]] = load i8*, i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[TMP]])
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define i8* @test1(i8*
id test1(id x) {
  // CHECK:      [[X:%.*]] = alloca i8*
  // CHECK-NEXT: [[Y:%.*]] = alloca i8*
  // CHECK-NEXT: [[PARM:%.*]] = call i8* @objc_retain(i8* {{%.*}})
  // CHECK-NEXT: store i8* [[PARM]], i8** [[X]]
  // CHECK-NEXT: [[YPTR1:%.*]] = bitcast i8** [[Y]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start(i64 8, i8* [[YPTR1]])
  // CHECK-NEXT: store i8* null, i8** [[Y]]
  // CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[Y]]
  // CHECK-NEXT: [[RET:%.*]] = call i8* @objc_retain(i8* [[T0]])
  // CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[Y]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]])
  // CHECK-NEXT: [[YPTR2:%.*]] = bitcast i8** [[Y]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end(i64 8, i8* [[YPTR2]])
  // CHECK-NEXT: [[T1:%.*]] = load i8*, i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[T1]])
  // CHECK-NEXT: [[T1:%.*]] = tail call i8* @objc_autoreleaseReturnValue(i8* [[RET]])
  // CHECK-NEXT: ret i8* [[T1]]
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

// CHECK-LABEL: define void @test3_unelided()
void test3_unelided() {
  extern void test3_helper(void);

  // CHECK:      [[X:%.*]] = alloca [[TEST3:%.*]]*
  // CHECK-NEXT: [[XPTR1:%.*]] = bitcast [[TEST3]]** [[X]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start(i64 8, i8* [[XPTR1]])
  // CHECK-NEXT: store [[TEST3]]* null, [[TEST3]]** [[X]], align
  Test3 *x;

  // Call to +alloc.
  // CHECK-NEXT: load {{.*}}, {{.*}}* @"OBJC_CLASSLIST_REFERENCES_
  // CHECK-NEXT: load i8*, i8** @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: [[ALLOC:%.*]] = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: call void @objc_release(i8*
  [Test3 alloc];

  // CHECK-NEXT: [[T0:%.*]] = load [[TEST3]]*, [[TEST3]]** [[X]]
  // CHECK-NEXT: load i8*, i8** @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST3]]* [[T0]] to i8*
  // CHECK-NEXT: [[COPY:%.*]] = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend {{.*}})(i8* [[T1]],
  // CHECK-NEXT: call void @objc_release(i8* [[COPY]]) [[NUW:#[0-9]+]]
  [x copy];

  // CHECK-NEXT: [[T0:%.*]] = load [[TEST3]]*, [[TEST3]]** [[X]]
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST3]]* [[T0]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T1]]) [[NUW]]
  // CHECK-NEXT: [[XPTR2:%.*]] = bitcast [[TEST3]]** [[X]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end(i64 8, i8* [[XPTR2]])
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define void @test3()
void test3() {
  // CHECK:      [[X:%.*]] = alloca i8*
  // CHECK-NEXT: [[XPTR1:%.*]] = bitcast i8** [[X]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start(i64 8, i8* [[XPTR1]])

  id x = [[Test3 alloc] initWith: 5];

  // Call to +alloc.
  // CHECK-NEXT: load {{.*}}, {{.*}}* @"OBJC_CLASSLIST_REFERENCES_
  // CHECK-NEXT: load i8*, i8** @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: [[ALLOC:%.*]] = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend
  // CHECK-NEXT: bitcast

  // Call to -initWith: with elided retain of consumed argument.
  // CHECK-NEXT: load i8*, i8** @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: [[INIT:%.*]] = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i32)*)(i8* 
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: [[INIT:%.*]] = bitcast
  // Assignment for initialization, retention elided.
  // CHECK-NEXT: store i8* [[INIT]], i8** [[X]]

  // Call to -copy.
  // CHECK-NEXT: [[V:%.*]] = load i8*, i8** [[X]]
  // CHECK-NEXT: load i8*, i8** @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[COPY:%.*]] = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend {{.*}})(i8* [[V]],

  // Assignment to x.
  // CHECK-NEXT: [[TMP:%.*]] = load i8*, i8** [[X]]
  // CHECK-NEXT: store i8* [[COPY]], i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[TMP]]) [[NUW]]

  x = [x copy];

  // Cleanup for x.
  // CHECK-NEXT: [[TMP:%.*]] = load i8*, i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[TMP]]) [[NUW]]
  // CHECK-NEXT: [[XPTR2:%.*]] = bitcast i8** [[X]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end(i64 8, i8* [[XPTR2]])
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define i8* @test4()
id test4() {
  // Call to +alloc.
  // CHECK:      load {{.*}}, {{.*}}* @"OBJC_CLASSLIST_REFERENCES_
  // CHECK-NEXT: load i8*, i8** @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: [[ALLOC:%.*]] = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend
  // CHECK-NEXT: [[ALLOC:%.*]] = bitcast

  // Call to -initWith: with elided retain of consumed argument.
  // CHECK-NEXT: load i8*, i8** @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[ALLOC:%.*]] = bitcast
  // CHECK-NEXT: [[INIT:%.*]] = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i32)*)(i8* [[ALLOC]],

  // Initialization of return value, occurring within full-expression.
  // Retain/release elided.
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: [[INIT:%.*]] = bitcast
  // CHECK-NEXT: [[RET:%.*]] = tail call i8* @objc_autoreleaseReturnValue(i8* [[INIT]])

  // CHECK-NEXT: ret i8* [[RET]]

  return [[Test3 alloc] initWith: 6];
}

@interface Test5 {
@public
  id var;
}
@end

// CHECK-LABEL: define void @test5
void test5(Test5 *x, id y) {
  // Prologue.
  // CHECK:      [[X:%.*]] = alloca [[TEST5:%.*]]*,
  // CHECK-NEXT: [[Y:%.*]] = alloca i8*
  // CHECK-NEXT: bitcast [[TEST5]]* {{%.*}} to i8*
  // CHECK-NEXT: call i8* @objc_retain
  // CHECK-NEXT: [[PARMX:%.*]] = bitcast i8* {{%.*}} to [[TEST5]]*
  // CHECK-NEXT: store [[TEST5]]* [[PARMX]], [[TEST5]]** [[X]]
  // CHECK-NEXT: call i8* @objc_retain
  // CHECK-NEXT: store

  // CHECK-NEXT: load [[TEST5]]*, [[TEST5]]** [[X]]
  // CHECK-NEXT: load i64, i64* @"OBJC_IVAR_$_Test5.var"
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: [[VAR:%.*]] = bitcast
  // CHECK-NEXT: [[TMP:%.*]] = load i8*, i8** [[VAR]]
  // CHECK-NEXT: store i8* null, i8** [[VAR]]
  // CHECK-NEXT: call void @objc_release(i8* [[TMP]]) [[NUW]]
  x->var = 0;

  // CHECK-NEXT: [[YVAL:%.*]] = load i8*, i8** [[Y]]
  // CHECK-NEXT: load [[TEST5]]*, [[TEST5]]** [[X]]
  // CHECK-NEXT: load i64, i64* @"OBJC_IVAR_$_Test5.var"
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: [[VAR:%.*]] = bitcast
  // CHECK-NEXT: [[T0:%.*]] = call i8* @objc_retain(i8* [[YVAL]]) [[NUW]]
  // CHECK-NEXT: [[TMP:%.*]] = load i8*, i8** [[VAR]]
  // CHECK-NEXT: store i8* [[T0]], i8** [[VAR]]
  // CHECK-NEXT: call void @objc_release(i8* [[TMP]]) [[NUW]]
  x->var = y;

  // Epilogue.
  // CHECK-NEXT: [[TMP:%.*]] = load i8*, i8** [[Y]]
  // CHECK-NEXT: call void @objc_release(i8* [[TMP]]) [[NUW]]
  // CHECK-NEXT: [[T0:%.*]] = load [[TEST5]]*, [[TEST5]]** [[X]]
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST5]]* [[T0]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T1]]) [[NUW]]
  // CHECK-NEXT: ret void
}

id test6_helper(void) __attribute__((ns_returns_retained));
// CHECK-LABEL: define void @test6()
void test6() {
  // CHECK:      [[X:%.*]] = alloca i8*
  // CHECK-NEXT: [[XPTR1:%.*]] = bitcast i8** [[X]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start(i64 8, i8* [[XPTR1]])
  // CHECK-NEXT: [[CALL:%.*]] = call i8* @test6_helper()
  // CHECK-NEXT: store i8* [[CALL]], i8** [[X]]
  // CHECK-NEXT: [[T1:%.*]] = load i8*, i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[T1]]) [[NUW]], !clang.imprecise_release
  // CHECK-NEXT: [[XPTR2:%.*]] = bitcast i8** [[X]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end(i64 8, i8* [[XPTR2]])
  // CHECK-NEXT: ret void
  id x = test6_helper();
}

void test7_helper(id __attribute__((ns_consumed)));
// CHECK-LABEL: define void @test7()
void test7() {
  // CHECK:      [[X:%.*]] = alloca i8*
  // CHECK-NEXT: [[XPTR1:%.*]] = bitcast i8** [[X]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start(i64 8, i8* [[XPTR1]])
  // CHECK-NEXT: store i8* null, i8** [[X]]
  // CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[X]]
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retain(i8* [[T0]]) [[NUW]]
  // CHECK-NEXT: call void @test7_helper(i8* [[T1]])
  // CHECK-NEXT: [[T1:%.*]] = load i8*, i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[T1]]) [[NUW]], !clang.imprecise_release
  // CHECK-NEXT: [[XPTR2:%.*]] = bitcast i8** [[X]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end(i64 8, i8* [[XPTR2]])
  // CHECK-NEXT: ret void
  id x;
  test7_helper(x);
}

id test8_helper(void) __attribute__((ns_returns_retained));
void test8() {
  __unsafe_unretained id x = test8_helper();
  // CHECK:      [[X:%.*]] = alloca i8*
  // CHECK-NEXT: [[XPTR1:%.*]] = bitcast i8** [[X]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start(i64 8, i8* [[XPTR1]])
  // CHECK-NEXT: [[T0:%.*]] = call i8* @test8_helper()
  // CHECK-NEXT: store i8* [[T0]], i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]]) [[NUW]], !clang.imprecise_release
  // CHECK-NEXT: [[XPTR2:%.*]] = bitcast i8** [[X]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end(i64 8, i8* [[XPTR2]])
  // CHECK-NEXT: ret void
}

@interface Test10
@property (retain) Test10 *me;
@end
void test10() {
  Test10 *x;
  id y = x.me.me;

  // CHECK-LABEL:      define void @test10()
  // CHECK:      [[X:%.*]] = alloca [[TEST10:%.*]]*, align
  // CHECK-NEXT: [[Y:%.*]] = alloca i8*, align
  // CHECK-NEXT: [[XPTR1:%.*]] = bitcast [[TEST10]]** [[X]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start(i64 8, i8* [[XPTR1]])
  // CHECK-NEXT: store [[TEST10]]* null, [[TEST10]]** [[X]]
  // CHECK-NEXT: [[YPTR1:%.*]] = bitcast i8** [[Y]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start(i64 8, i8* [[YPTR1]])
  // CHECK-NEXT: load [[TEST10]]*, [[TEST10]]** [[X]], align
  // CHECK-NEXT: load i8*, i8** @OBJC_SELECTOR_REFERENCES_{{[0-9]*}}
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: [[T0:%.*]] = call [[TEST10]]* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST10]]* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T1]])
  // CHECK-NEXT: [[V:%.*]] = bitcast i8* [[T2]] to [[TEST10]]*
  // CHECK-NEXT: load i8*, i8** @OBJC_SELECTOR_REFERENCES_{{[0-9]*}}
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: [[T0:%.*]] = call [[TEST10]]* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST10]]* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T1]])
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to [[TEST10]]*
  // CHECK-NEXT: [[T4:%.*]] = bitcast [[TEST10]]* [[T3]] to i8*
  // CHECK-NEXT: store i8* [[T4]], i8** [[Y]]
  // CHECK-NEXT: [[T0:%.*]] = bitcast [[TEST10]]* [[V]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T0]])
  // CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[Y]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]])
  // CHECK-NEXT: [[YPTR2:%.*]] = bitcast i8** [[Y]] to i8*
  // CHECK-NEXT: void @llvm.lifetime.end(i64 8, i8* [[YPTR2]])
  // CHECK-NEXT: [[T0:%.*]] = load [[TEST10]]*, [[TEST10]]** [[X]]
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST10]]* [[T0]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T1]])
  // CHECK-NEXT: [[XPTR2:%.*]] = bitcast [[TEST10]]** [[X]] to i8*
  // CHECK-NEXT: void @llvm.lifetime.end(i64 8, i8* [[XPTR2]])
  // CHECK-NEXT: ret void
}

void test11(id (*f)(void) __attribute__((ns_returns_retained))) {
  // CHECK-LABEL:      define void @test11(
  // CHECK:      [[F:%.*]] = alloca i8* ()*, align
  // CHECK-NEXT: [[X:%.*]] = alloca i8*, align
  // CHECK-NEXT: store i8* ()* {{%.*}}, i8* ()** [[F]], align
  // CHECK-NEXT: [[XPTR1:%.*]] = bitcast i8** [[X]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start(i64 8, i8* [[XPTR1]])
  // CHECK-NEXT: [[T0:%.*]] = load i8* ()*, i8* ()** [[F]], align
  // CHECK-NEXT: [[T1:%.*]] = call i8* [[T0]]()
  // CHECK-NEXT: store i8* [[T1]], i8** [[X]], align
  // CHECK-NEXT: [[T3:%.*]] = load i8*, i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[T3]]) [[NUW]], !clang.imprecise_release
  // CHECK-NEXT: [[XPTR2:%.*]] = bitcast i8** [[X]] to i8*
  // CHECK-NEXT: void @llvm.lifetime.end(i64 8, i8* [[XPTR2]])
  // CHECK-NEXT: ret void
  id x = f();
}

void test12(void) {
  extern id test12_helper(void);

  // CHECK-LABEL:      define void @test12()
  // CHECK:      [[X:%.*]] = alloca i8*, align
  // CHECK-NEXT: [[Y:%.*]] = alloca i8*, align

  __weak id x = test12_helper();
  // CHECK-NEXT: [[XPTR1:%.*]] = bitcast i8** [[X]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start(i64 8, i8* [[XPTR1]])
  // CHECK-NEXT: [[T0:%.*]] = call i8* @test12_helper()
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T0]])
  // CHECK-NEXT: call i8* @objc_initWeak(i8** [[X]], i8* [[T1]])
  // CHECK-NEXT: call void @objc_release(i8* [[T1]])

  x = test12_helper();
  // CHECK-NEXT: [[T0:%.*]] = call i8* @test12_helper()
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T0]])
  // CHECK-NEXT: call i8* @objc_storeWeak(i8** [[X]], i8* [[T1]])
  // CHECK-NEXT: call void @objc_release(i8* [[T1]])

  id y = x;
  // CHECK-NEXT: [[YPTR1:%.*]] = bitcast i8** [[Y]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start(i64 8, i8* [[YPTR1]])
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_loadWeakRetained(i8** [[X]])
  // CHECK-NEXT: store i8* [[T2]], i8** [[Y]], align

  // CHECK-NEXT: [[T4:%.*]] = load i8*, i8** [[Y]]
  // CHECK-NEXT: call void @objc_release(i8* [[T4]]) [[NUW]], !clang.imprecise_release
  // CHECK-NEXT: [[YPTR2:%.*]] = bitcast i8** [[Y]] to i8*
  // CHECK-NEXT: void @llvm.lifetime.end(i64 8, i8* [[YPTR2]])
  // CHECK-NEXT: call void @objc_destroyWeak(i8** [[X]])
  // CHECK-NEXT: [[XPTR2:%.*]] = bitcast i8** [[X]] to i8*
  // CHECK-NEXT: void @llvm.lifetime.end(i64 8, i8* [[XPTR2]])
  // CHECK: ret void
}

// Indirect consuming calls.
void test13(void) {
  // CHECK-LABEL:      define void @test13()
  // CHECK:      [[X:%.*]] = alloca i8*, align
  // CHECK-NEXT: [[XPTR1:%.*]] = bitcast i8** [[X]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start(i64 8, i8* [[XPTR1]])
  // CHECK-NEXT: store i8* null, i8** [[X]], align
  id x;

  typedef void fnty(id __attribute__((ns_consumed)));
  extern fnty *test13_func;
  // CHECK-NEXT: [[FN:%.*]] = load void (i8*)*, void (i8*)** @test13_func, align
  // CHECK-NEXT: [[X_VAL:%.*]] = load i8*, i8** [[X]], align
  // CHECK-NEXT: [[X_TMP:%.*]] = call i8* @objc_retain(i8* [[X_VAL]]) [[NUW]]
  // CHECK-NEXT: call void [[FN]](i8* [[X_TMP]])
  test13_func(x);

  extern fnty ^test13_block;
  // CHECK-NEXT: [[TMP:%.*]] = load void (i8*)*, void (i8*)** @test13_block, align
  // CHECK-NEXT: [[BLOCK:%.*]] = bitcast void (i8*)* [[TMP]] to [[BLOCKTY:%.*]]*
  // CHECK-NEXT: [[BLOCK_FN_PTR:%.*]] = getelementptr inbounds [[BLOCKTY]], [[BLOCKTY]]* [[BLOCK]], i32 0, i32 3
  // CHECK-NEXT: [[BLOCK_OPAQUE:%.*]] = bitcast [[BLOCKTY]]* [[BLOCK]] to i8*
  // CHECK-NEXT: [[X_VAL:%.*]] = load i8*, i8** [[X]], align
  // CHECK-NEXT: [[X_TMP:%.*]] = call i8* @objc_retain(i8* [[X_VAL]]) [[NUW]]
  // CHECK-NEXT: [[BLOCK_FN_TMP:%.*]] = load i8*, i8** [[BLOCK_FN_PTR]]
  // CHECK-NEXT: [[BLOCK_FN:%.*]] = bitcast i8* [[BLOCK_FN_TMP]] to void (i8*, i8*)*
  // CHECK-NEXT: call void [[BLOCK_FN]](i8* [[BLOCK_OPAQUE]], i8* [[X_TMP]])
  test13_block(x);

  // CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]]) [[NUW]]
  // CHECK-NEXT: [[XPTR2:%.*]] = bitcast i8** [[X]] to i8*
  // CHECK-NEXT: void @llvm.lifetime.end(i64 8, i8* [[XPTR2]])
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
  // CHECK:      [[SELF:%.*]] = alloca [[TEST16:%.*]]*, align
  // CHECK-NEXT: [[CMD:%.*]] = alloca i8*, align
  // CHECK-NEXT: alloca
  // CHECK-NEXT: store [[TEST16]]* {{%.*}}, [[TEST16]]** [[SELF]], align
  // CHECK-NEXT: store i8* {{%.*}}, i8** [[CMD]]
  // CHECK-NEXT: [[BASE:%.*]] = load [[TEST16]]*, [[TEST16]]** [[SELF]]

  // Call super.
  // CHECK-NEXT: [[BASE2:%.*]] = bitcast [[TEST16]]* [[BASE]] to i8*
  // CHECK-NEXT: [[T0:%.*]] = getelementptr
  // CHECK-NEXT: store i8* [[BASE2]], i8** [[T0]]
  // CHECK-NEXT: load {{%.*}}*, {{%.*}}** @"OBJC_CLASSLIST_SUP_REFS_$_
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: store
  // CHECK-NEXT: load i8*, i8** @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: call void bitcast (i8* ({{.*}})* @objc_msgSendSuper2 to void (
  // CHECK-NEXT: ret void
}

// .cxx_destruct
  // CHECK:    define internal void @"\01-[Test16 .cxx_destruct]"(
  // CHECK:      [[SELF:%.*]] = alloca [[TEST16:%.*]]*, align
  // CHECK-NEXT: [[CMD:%.*]] = alloca i8*, align
  // CHECK-NEXT: store [[TEST16]]* {{%.*}}, [[TEST16]]** [[SELF]], align
  // CHECK-NEXT: store i8* {{%.*}}, i8** [[CMD]]
  // CHECK-NEXT: [[BASE:%.*]] = load [[TEST16]]*, [[TEST16]]** [[SELF]]

  // Destroy y.
  // CHECK-NEXT: [[Y_OFF:%.*]] = load i64, i64* @"OBJC_IVAR_$_Test16.y"
  // CHECK-NEXT: [[T0:%.*]] = bitcast [[TEST16]]* [[BASE]] to i8*
  // CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds i8, i8* [[T0]], i64 [[Y_OFF]]
  // CHECK-NEXT: [[T2:%.*]] = bitcast i8* [[T1]] to i8**
  // CHECK-NEXT: call void @objc_storeStrong(i8** [[T2]], i8* null) [[NUW]]

  // Destroy z.
  // CHECK-NEXT: [[Z_OFF:%.*]] = load i64, i64* @"OBJC_IVAR_$_Test16.z"
  // CHECK-NEXT: [[T0:%.*]] = bitcast [[TEST16]]* [[BASE]] to i8*
  // CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds i8, i8* [[T0]], i64 [[Z_OFF]]
  // CHECK-NEXT: [[T2:%.*]] = bitcast i8* [[T1]] to i8**
  // CHECK-NEXT: call void @objc_storeStrong(i8** [[T2]], i8* null) [[NUW]]

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

void test19() {
  // CHECK-LABEL: define void @test19()
  // CHECK:      [[X:%.*]] = alloca [5 x i8*], align 16
  // CHECK: call void @llvm.lifetime.start
  // CHECK-NEXT: [[T0:%.*]] = bitcast [5 x i8*]* [[X]] to i8*
  // CHECK: call void @llvm.memset.p0i8.i64(i8* [[T0]], i8 0, i64 40, i32 16, i1 false)
  id x[5];

  extern id test19_helper(void);
  x[2] = test19_helper();

  // CHECK-NEXT: [[CALL:%.*]] = call i8* @test19_helper()
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[CALL]]) [[NUW]]
  // CHECK-NEXT: [[SLOT:%.*]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[X]], i64 0, i64 2
  // CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[SLOT]]
  // CHECK-NEXT: store i8* [[T1]], i8** [[SLOT]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]]) [[NUW]]

  // CHECK-NEXT: [[BEGIN:%.*]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[X]], i32 0, i32 0
  // CHECK-NEXT: [[END:%.*]] = getelementptr inbounds i8*, i8** [[BEGIN]], i64 5
  // CHECK-NEXT: br label

  // CHECK:      [[AFTER:%.*]] = phi i8** [ [[END]], {{%.*}} ], [ [[NEXT:%.*]], {{%.*}} ]
  // CHECK-NEXT: [[CUR:%.*]] = getelementptr inbounds i8*, i8** [[AFTER]], i64 -1
  // CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[CUR]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]]) [[NUW]], !clang.imprecise_release
  // CHECK-NEXT: [[EQ:%.*]] = icmp eq i8** [[CUR]], [[BEGIN]]
  // CHECK-NEXT: br i1 [[EQ]],

  // CHECK:      ret void
}

void test20(unsigned n) {
  // CHECK-LABEL: define void @test20
  // CHECK:      [[N:%.*]] = alloca i32, align 4
  // CHECK-NEXT: [[SAVED_STACK:%.*]] = alloca i8*
  // CHECK-NEXT: store i32 {{%.*}}, i32* [[N]], align 4

  id x[n];

  // Capture the VLA size.
  // CHECK-NEXT: [[T0:%.*]] = load i32, i32* [[N]], align 4
  // CHECK-NEXT: [[DIM:%.*]] = zext i32 [[T0]] to i64

  // Save the stack pointer.
  // CHECK-NEXT: [[T0:%.*]] = call i8* @llvm.stacksave()
  // CHECK-NEXT: store i8* [[T0]], i8** [[SAVED_STACK]]

  // Allocate the VLA.
  // CHECK-NEXT: [[VLA:%.*]] = alloca i8*, i64 [[DIM]], align 16

  // Zero-initialize.
  // CHECK-NEXT: [[T0:%.*]] = bitcast i8** [[VLA]] to i8*
  // CHECK-NEXT: [[T1:%.*]] = mul nuw i64 [[DIM]], 8
  // CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* [[T0]], i8 0, i64 [[T1]], i32 16, i1 false)

  // Destroy.
  // CHECK-NEXT: [[END:%.*]] = getelementptr inbounds i8*, i8** [[VLA]], i64 [[DIM]]
  // CHECK-NEXT: [[EMPTY:%.*]] = icmp eq i8** [[VLA]], [[END]]
  // CHECK-NEXT: br i1 [[EMPTY]]

  // CHECK:      [[AFTER:%.*]] = phi i8** [ [[END]], {{%.*}} ], [ [[CUR:%.*]], {{%.*}} ]
  // CHECK-NEXT: [[CUR:%.*]] = getelementptr inbounds i8*, i8** [[AFTER]], i64 -1
  // CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[CUR]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]]) [[NUW]], !clang.imprecise_release
  // CHECK-NEXT: [[EQ:%.*]] = icmp eq i8** [[CUR]], [[VLA]]
  // CHECK-NEXT: br i1 [[EQ]],

  // CHECK:      [[T0:%.*]] = load i8*, i8** [[SAVED_STACK]]
  // CHECK-NEXT: call void @llvm.stackrestore(i8* [[T0]])
  // CHECK-NEXT: ret void
}

void test21(unsigned n) {
  // CHECK-LABEL: define void @test21
  // CHECK:      [[N:%.*]] = alloca i32, align 4
  // CHECK-NEXT: [[SAVED_STACK:%.*]] = alloca i8*
  // CHECK-NEXT: store i32 {{%.*}}, i32* [[N]], align 4

  id x[2][n][3];

  // Capture the VLA size.
  // CHECK-NEXT: [[T0:%.*]] = load i32, i32* [[N]], align 4
  // CHECK-NEXT: [[DIM:%.*]] = zext i32 [[T0]] to i64

  // CHECK-NEXT: [[T0:%.*]] = call i8* @llvm.stacksave()
  // CHECK-NEXT: store i8* [[T0]], i8** [[SAVED_STACK]]


  // Allocate the VLA.
  // CHECK-NEXT: [[T0:%.*]] = mul nuw i64 2, [[DIM]]
  // CHECK-NEXT: [[VLA:%.*]] = alloca [3 x i8*], i64 [[T0]], align 16

  // Zero-initialize.
  // CHECK-NEXT: [[T0:%.*]] = bitcast [3 x i8*]* [[VLA]] to i8*
  // CHECK-NEXT: [[T1:%.*]] = mul nuw i64 2, [[DIM]]
  // CHECK-NEXT: [[T2:%.*]] = mul nuw i64 [[T1]], 24
  // CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* [[T0]], i8 0, i64 [[T2]], i32 16, i1 false)

  // Destroy.
  // CHECK-NEXT: [[T0:%.*]] = mul nuw i64 2, [[DIM]]
  // CHECK-NEXT: [[BEGIN:%.*]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[VLA]], i32 0, i32 0
  // CHECK-NEXT: [[T1:%.*]] = mul nuw i64 [[T0]], 3
  // CHECK-NEXT: [[END:%.*]] = getelementptr inbounds i8*, i8** [[BEGIN]], i64 [[T1]]
  // CHECK-NEXT: [[EMPTY:%.*]] = icmp eq i8** [[BEGIN]], [[END]]
  // CHECK-NEXT: br i1 [[EMPTY]]

  // CHECK:      [[AFTER:%.*]] = phi i8** [ [[END]], {{%.*}} ], [ [[CUR:%.*]], {{%.*}} ]
  // CHECK-NEXT: [[CUR:%.*]] = getelementptr inbounds i8*, i8** [[AFTER]], i64 -1
  // CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[CUR]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]]) [[NUW]], !clang.imprecise_release
  // CHECK-NEXT: [[EQ:%.*]] = icmp eq i8** [[CUR]], [[BEGIN]]
  // CHECK-NEXT: br i1 [[EQ]],

  // CHECK:      [[T0:%.*]] = load i8*, i8** [[SAVED_STACK]]
  // CHECK-NEXT: call void @llvm.stackrestore(i8* [[T0]])
  // CHECK-NEXT: ret void
}

// rdar://problem/8922540
//   Note that we no longer emit .release_ivars flags.
// rdar://problem/12492434
//   Note that we set the flag saying that we need destruction *and*
//   the flag saying that we don't also need construction.
// CHECK-GLOBALS: @"\01l_OBJC_CLASS_RO_$_Test23" = private global [[RO_T:%.*]] { i32 390,
@interface Test23 { id x; } @end
@implementation Test23 @end

// CHECK-GLOBALS: @"\01l_OBJC_CLASS_RO_$_Test24" = private global [[RO_T:%.*]] { i32 130,
@interface Test24 {} @end
@implementation Test24 @end

// rdar://problem/8941012
@interface Test26 { id x[4]; } @end
@implementation Test26 @end
// CHECK:    define internal void @"\01-[Test26 .cxx_destruct]"(
// CHECK:      [[SELF:%.*]] = load [[TEST26:%.*]]*, [[TEST26:%.*]]**
// CHECK-NEXT: [[OFFSET:%.*]] = load i64, i64* @"OBJC_IVAR_$_Test26.x"
// CHECK-NEXT: [[T0:%.*]] = bitcast [[TEST26]]* [[SELF]] to i8*
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds i8, i8* [[T0]], i64 [[OFFSET]]
// CHECK-NEXT: [[X:%.*]] = bitcast i8* [[T1]] to [4 x i8*]*
// CHECK-NEXT: [[BEGIN:%.*]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[X]], i32 0, i32 0
// CHECK-NEXT: [[END:%.*]] = getelementptr inbounds i8*, i8** [[BEGIN]], i64 4
// CHECK-NEXT: br label
// CHECK:      [[PAST:%.*]] = phi i8** [ [[END]], {{%.*}} ], [ [[CUR:%.*]], {{%.*}} ]
// CHECK-NEXT: [[CUR]] = getelementptr inbounds i8*, i8** [[PAST]], i64 -1
// CHECK-NEXT: call void @objc_storeStrong(i8** [[CUR]], i8* null)
// CHECK-NEXT: [[ISDONE:%.*]] = icmp eq i8** [[CUR]], [[BEGIN]]
// CHECK-NEXT: br i1 [[ISDONE]],
// CHECK:      ret void

// Check that 'init' retains self.
@interface Test27
- (id) init;
@end
@implementation Test27
- (id) init { return self; }
// CHECK:    define internal i8* @"\01-[Test27 init]"
// CHECK:      [[SELF:%.*]] = alloca [[TEST27:%.*]]*,
// CHECK-NEXT: [[CMD:%.*]] = alloca i8*,
// CHECK-NEXT: store [[TEST27]]* {{%.*}}, [[TEST27]]** [[SELF]]
// CHECK-NEXT: store i8* {{%.*}}, i8** [[CMD]]
// CHECK-NEXT: [[T0:%.*]] = load [[TEST27]]*, [[TEST27]]** [[SELF]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST27]]* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retain(i8* [[T1]])
// CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to [[TEST27]]*
// CHECK-NEXT: [[RET:%.*]] = bitcast [[TEST27]]* [[T3]] to i8*
// CHECK-NEXT: [[T0:%.*]] = load [[TEST27]]*, [[TEST27]]** [[SELF]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST27]]* [[T0]] to i8*
// CHECK-NEXT: call void @objc_release(i8* [[T1]])
// CHECK-NEXT: ret i8* [[RET]]

@end

// rdar://problem/8087194
@interface Test28
@property (copy) id prop;
@end
@implementation Test28
@synthesize prop;
@end
// CHECK:    define internal void @"\01-[Test28 .cxx_destruct]"
// CHECK:      [[SELF:%.*]] = load [[TEST28:%.*]]*, [[TEST28:%.*]]**
// CHECK-NEXT: [[OFFSET:%.*]] = load i64, i64* @"OBJC_IVAR_$_Test28.prop"
// CHECK-NEXT: [[T0:%.*]] = bitcast [[TEST28]]* [[SELF]] to i8*
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds i8, i8* [[T0]], i64 [[OFFSET]]
// CHECK-NEXT: [[T2:%.*]] = bitcast i8* [[T1]] to i8**
// CHECK-NEXT: call void @objc_storeStrong(i8** [[T2]], i8* null)
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
// CHECK:    define internal i8* @"\01-[Test29 init]"([[TEST29:%[^*]*]]* {{%.*}},
// CHECK:      [[SELF:%.*]] = alloca [[TEST29]]*, align 8
// CHECK-NEXT: [[CMD:%.*]] = alloca i8*, align 8
// CHECK-NEXT: store [[TEST29]]* {{%.*}}, [[TEST29]]** [[SELF]]
// CHECK-NEXT: store i8* {{%.*}}, i8** [[CMD]]

// Evaluate arguments.  Note that the send argument is evaluated
// before the zeroing of self.
// CHECK-NEXT: [[T0:%.*]] = load [[TEST29]]*, [[TEST29]]** [[SELF]], align 8
// CHECK-NEXT: [[T1:%.*]] = load i8*, i8** @_test29_allocator, align 8

// Implicit null of 'self', i.e. direct transfer of ownership.
// CHECK-NEXT: store [[TEST29]]* null, [[TEST29]]** [[SELF]]

// Actual message send.
// CHECK-NEXT: [[T2:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_
// CHECK-NEXT: [[T3:%.*]] = bitcast [[TEST29]]* [[T0]] to i8*
// CHECK-NEXT: [[CALL:%.*]] = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8*)*)(i8* [[T3]], i8* [[T2]], i8* [[T1]])

// Implicit write of result back into 'self'.  This is not supposed to
// be detectable because we're supposed to ban accesses to the old
// self value past the delegate init call.
// CHECK-NEXT: [[T0:%.*]] = bitcast i8* [[CALL]] to [[TEST29]]*
// CHECK-NEXT: store [[TEST29]]* [[T0]], [[TEST29]]** [[SELF]]

// Return statement.
// CHECK-NEXT: [[T2:%.*]] = bitcast i8* [[CALL]]
// CHECK-NEXT: [[CALL:%.*]] = bitcast
// CHECK-NEXT: [[T0:%.*]] = call i8* @objc_retain(i8* [[CALL]]) [[NUW]]
// CHECK-NEXT: [[T1:%.*]] = bitcast i8* [[T0]] to [[TEST29]]*
// CHECK-NEXT: [[RET:%.*]] = bitcast [[TEST29]]* [[T1]] to i8*

// Cleanup.
// CHECK-NEXT: [[T0:%.*]] = load [[TEST29]]*, [[TEST29]]** [[SELF]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST29]]* [[T0]] to i8*
// CHECK-NEXT: call void @objc_release(i8* [[T1]]) [[NUW]], !clang.imprecise_release

// Return.
// CHECK-NEXT: ret i8* [[RET]]
  return [self initWithAllocator: _test29_allocator];
}
- (id) initWithAllocator: (id) allocator {
// CHECK:    define internal i8* @"\01-[Test29 initWithAllocator:]"(
// CHECK:      [[SELF:%.*]] = alloca [[TEST29]]*, align 8
// CHECK-NEXT: [[CMD:%.*]] = alloca i8*, align 8
// CHECK-NEXT: [[ALLOCATOR:%.*]] = alloca i8*, align 8
// CHECK-NEXT: alloca
// CHECK-NEXT: store [[TEST29]]* {{%.*}}, [[TEST29]]** [[SELF]]
// CHECK-NEXT: store i8* {{%.*}}, i8** [[CMD]]
// CHECK-NEXT: [[T0:%.*]] = call i8* @objc_retain(i8* {{%.*}})
// CHECK-NEXT: store i8* [[T0]], i8** [[ALLOCATOR]]

// Evaluate arguments.  Note that the send argument is evaluated
// before the zeroing of self.
// CHECK-NEXT: [[T0:%.*]] = load [[TEST29]]*, [[TEST29]]** [[SELF]]
// CHECK-NEXT: [[T1:%.*]] = load i8*, i8** [[ALLOCATOR]], align 8

// Implicit null of 'self', i.e. direct transfer of ownership.
// CHECK-NEXT: store [[TEST29]]* null, [[TEST29]]** [[SELF]]

// Actual message send.
// CHECK:      [[CALL:%.*]] = call {{.*}} @objc_msgSendSuper2

// Implicit write of result back into 'self'.  This is not supposed to
// be detectable because we're supposed to ban accesses to the old
// self value past the delegate init call.
// CHECK-NEXT: [[T0:%.*]] = bitcast i8* [[CALL]] to [[TEST29]]*
// CHECK-NEXT: store [[TEST29]]* [[T0]], [[TEST29]]** [[SELF]]

// Assignment.
// CHECK-NEXT: [[T0:%.*]] = bitcast i8* [[CALL]] to [[TEST29]]*
// CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST29]]* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retain(i8* [[T1]]) [[NUW]]
// CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to [[TEST29]]*
// CHECK-NEXT: [[T4:%.*]] = load [[TEST29]]*, [[TEST29]]** [[SELF]], align
// CHECK-NEXT: store [[TEST29]]* [[T3]], [[TEST29]]** [[SELF]], align
// CHECK-NEXT: [[T5:%.*]] = bitcast [[TEST29]]* [[T4]] to i8*
// CHECK-NEXT: call void @objc_release(i8* [[T5]])

// Return statement.
// CHECK-NEXT: [[T0:%.*]] = load [[TEST29]]*, [[TEST29]]** [[SELF]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST29]]* [[T0]] to i8*
// CHECK-NEXT: [[T0:%.*]] = call i8* @objc_retain(i8* [[T1]]) [[NUW]]
// CHECK-NEXT: [[T1:%.*]] = bitcast i8* [[T0]] to [[TEST29]]*
// CHECK-NEXT: [[RET:%.*]] = bitcast [[TEST29]]* [[T1]] to i8*

// Cleanup.
// CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[ALLOCATOR]]
// CHECK-NEXT: call void @objc_release(i8* [[T0]]) [[NUW]], !clang.imprecise_release

// CHECK-NEXT: [[T0:%.*]] = load [[TEST29]]*, [[TEST29]]** [[SELF]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST29]]* [[T0]] to i8*
// CHECK-NEXT: call void @objc_release(i8* [[T1]]) [[NUW]], !clang.imprecise_release

// Return.
// CHECK-NEXT: ret i8* [[RET]]
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
// CHECK:    define internal i8* @"\01-[Test30 init]"([[TEST30:%[^*]*]]* {{%.*}},
// CHECK:      [[RET:%.*]] = alloca [[TEST30]]*
// CHECK-NEXT: alloca i8*
// CHECK-NEXT: store [[TEST30]]* {{%.*}}, [[TEST30]]** [[SELF]]
// CHECK-NEXT: store

// Call.
// CHECK-NEXT: [[T0:%.*]] = load [[TEST30]]*, [[TEST30]]** [[SELF]]
// CHECK-NEXT: [[T1:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_
// CHECK-NEXT: [[T2:%.*]] = bitcast [[TEST30]]* [[T0]] to i8*
// CHECK-NEXT: [[CALL:%.*]] = call [[TEST30_HELPER:%.*]]* bitcast {{.*}} @objc_msgSend {{.*}}(i8* [[T2]], i8* [[T1]])

// Assignment.
// CHECK-NEXT: [[T0:%.*]] = bitcast [[TEST30_HELPER]]* [[CALL]] to i8*
// CHECK-NEXT: [[T1:%.*]] = load [[TEST30]]*, [[TEST30]]** [[SELF]]
// CHECK-NEXT: [[IVAR:%.*]] = load i64, i64* @"OBJC_IVAR_$_Test30.helper"
// CHECK-NEXT: [[T2:%.*]] = bitcast [[TEST30]]* [[T1]] to i8*
// CHECK-NEXT: [[T3:%.*]] = getelementptr inbounds i8, i8* [[T2]], i64 [[IVAR]]
// CHECK-NEXT: [[T4:%.*]] = bitcast i8* [[T3]] to i8**
// CHECK-NEXT#: [[T5:%.*]] = load i8*, i8** [[T4]]
// CHECK-NEXT#: [[T6:%.*]] = call i8* @objc_retain(i8* [[T0]])
// CHECK-NEXT#: call void @objc_release(i8* [[T5]])
// CHECK-NEXT: store i8* [[T0]], i8** [[T4]]

// Return.
// CHECK-NEXT: [[T0:%.*]] = load [[TEST30]]*, [[TEST30]]** [[SELF]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST30]]* [[T0]] to i8*
// CHECK-NEXT: [[T0:%.*]] = call i8* @objc_retain(i8* [[T1]])
// CHECK-NEXT: [[T1:%.*]] = bitcast i8* [[T0]] to [[TEST30]]*
// CHECK-NEXT: [[RET:%.*]] = bitcast [[TEST30]]* [[T1]] to i8*

// Cleanup.
// CHECK-NEXT: [[T0:%.*]] = load [[TEST30]]*, [[TEST30]]** [[SELF]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST30]]* [[T0]] to i8*
// CHECK-NEXT: call void @objc_release(i8* [[T1]])

// Epilogue.
// CHECK-NEXT: ret i8* [[RET]]
  self->helper = [self initHelper];
  return self;
}
- (Test30_helper*) initHelper {
// CHECK:    define internal [[TEST30_HELPER]]* @"\01-[Test30 initHelper]"(
// CHECK:      alloca
// CHECK-NEXT: alloca
// CHECK-NEXT: store
// CHECK-NEXT: store
// CHECK-NEXT: ret [[TEST30_HELPER]]* null
  return 0;
}

@end

__attribute__((ns_returns_retained)) id test32(void) {
// CHECK-LABEL:    define i8* @test32()
// CHECK:      [[CALL:%.*]] = call i8* @test32_helper()
// CHECK-NEXT: [[T0:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[CALL]])
// CHECK-NEXT: ret i8* [[T0]]
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

  // CHECK:    define void @test33([[TEST33:%.*]]*
  // CHECK:      [[PTR:%.*]] = alloca [[TEST33]]*
  // CHECK-NEXT: [[A:%.*]] = alloca [[A_T:%.*]]*
  // CHECK-NEXT: [[TEMP0:%.*]] = alloca [[A_T]]*
  // CHECK-NEXT: [[TEMP1:%.*]] = alloca [[A_T]]*
  // CHECK-NEXT: [[TEMP2:%.*]] = alloca [[A_T]]*
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: objc_retain
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: store
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: call void @llvm.lifetime.start
  // CHECK-NEXT: store [[A_T]]* null, [[A_T]]** [[A]]

  // CHECK-NEXT: load [[TEST33]]*, [[TEST33]]** [[PTR]]
  // CHECK-NEXT: [[W0:%.*]] = load [[A_T]]*, [[A_T]]** [[A]]
  // CHECK-NEXT: store [[A_T]]* [[W0]], [[A_T]]** [[TEMP0]]
  // CHECK-NEXT: load i8*, i8** @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: objc_msgSend{{.*}}, [[A_T]]** [[TEMP0]])
  // CHECK-NEXT: [[T0:%.*]] = load [[A_T]]*, [[A_T]]** [[TEMP0]]
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[A_T]]* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retain(i8* [[T1]])
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to [[A_T]]*
  // CHECK-NEXT: call void (...) @clang.arc.use([[A_T]]* [[W0]]) [[NUW]]
  // CHECK-NEXT: [[T4:%.*]] = load [[A_T]]*, [[A_T]]** [[A]]
  // CHECK-NEXT: store [[A_T]]* [[T3]], [[A_T]]** [[A]]
  // CHECK-NEXT: [[T5:%.*]] = bitcast [[A_T]]* [[T4]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T5]])

  // CHECK-NEXT: load [[TEST33]]*, [[TEST33]]** [[PTR]]
  // CHECK-NEXT: [[W0:%.*]] = load [[A_T]]*, [[A_T]]** [[A]]
  // CHECK-NEXT: store [[A_T]]* [[W0]], [[A_T]]** [[TEMP1]]
  // CHECK-NEXT: load i8*, i8** @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: objc_msgSend{{.*}}, [[A_T]]** [[TEMP1]])
  // CHECK-NEXT: [[T0:%.*]] = load [[A_T]]*, [[A_T]]** [[TEMP1]]
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[A_T]]* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retain(i8* [[T1]])
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to [[A_T]]*
  // CHECK-NEXT: call void (...) @clang.arc.use([[A_T]]* [[W0]]) [[NUW]]
  // CHECK-NEXT: [[T4:%.*]] = load [[A_T]]*, [[A_T]]** [[A]]
  // CHECK-NEXT: store [[A_T]]* [[T3]], [[A_T]]** [[A]]
  // CHECK-NEXT: [[T5:%.*]] = bitcast [[A_T]]* [[T4]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T5]])

  // CHECK-NEXT: load [[TEST33]]*, [[TEST33]]** [[PTR]]
  // CHECK-NEXT: load i8*, i8** @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: objc_msgSend{{.*}}, [[A_T]]** [[A]])

  // CHECK-NEXT: load [[TEST33]]*, [[TEST33]]** [[PTR]]
  // CHECK-NEXT: load i8*, i8** @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: objc_msgSend{{.*}}, [[A_T]]** [[A]])

  // 'out'
  // CHECK-NEXT: load [[TEST33]]*, [[TEST33]]** [[PTR]]
  // CHECK-NEXT: store [[A_T]]* null, [[A_T]]** [[TEMP2]]
  // CHECK-NEXT: load i8*, i8** @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: objc_msgSend{{.*}}, [[A_T]]** [[TEMP2]])
  // CHECK-NEXT: [[T0:%.*]] = load [[A_T]]*, [[A_T]]** [[TEMP2]]
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[A_T]]* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retain(i8* [[T1]])
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to [[A_T]]*
  // CHECK-NEXT: [[T4:%.*]] = load [[A_T]]*, [[A_T]]** [[A]]
  // CHECK-NEXT: store [[A_T]]* [[T3]], [[A_T]]** [[A]]
  // CHECK-NEXT: [[T5:%.*]] = bitcast [[A_T]]* [[T4]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T5]])

  // CHECK-NEXT: load
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: objc_release
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: call void @llvm.lifetime.end
  // CHECK-NEXT: load
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: objc_release
  // CHECK-NEXT: ret void
}


// CHECK-LABEL: define void @test36
void test36(id x) {
  // CHECK: [[X:%.*]] = alloca i8*

  // CHECK: call i8* @objc_retain
  // CHECK: call i8* @objc_retain
  // CHECK: call i8* @objc_retain
  id array[3] = { @"A", x, @"y" };

  // CHECK:      [[T0:%.*]] = load i8*, i8** [[X]]
  // CHECK-NEXT: store i8* null, i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]])
  x = 0;

  // CHECK: br label
  // CHECK: call void @objc_release
  // CHECK: br i1

  // CHECK: call void @objc_release
  // CHECK-NEXT: ret void
}

@class Test37;
void test37(void) {
  extern void test37_helper(id *);
  Test37 *var;
  test37_helper(&var);

  // CHECK-LABEL:    define void @test37()
  // CHECK:      [[VAR:%.*]] = alloca [[TEST37:%.*]]*,
  // CHECK-NEXT: [[TEMP:%.*]] = alloca i8*
  // CHECK-NEXT: [[VARPTR1:%.*]] = bitcast [[TEST37]]** [[VAR]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start(i64 8, i8* [[VARPTR1]])
  // CHECK-NEXT: store [[TEST37]]* null, [[TEST37]]** [[VAR]]

  // CHECK-NEXT: [[W0:%.*]] = load [[TEST37]]*, [[TEST37]]** [[VAR]]
  // CHECK-NEXT: [[W1:%.*]] = bitcast [[TEST37]]* [[W0]] to i8*
  // CHECK-NEXT: store i8* [[W1]], i8** [[TEMP]]
  // CHECK-NEXT: call void @test37_helper(i8** [[TEMP]])
  // CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[TEMP]]
  // CHECK-NEXT: [[T1:%.*]] = bitcast i8* [[T0]] to [[TEST37]]*
  // CHECK-NEXT: [[T2:%.*]] = bitcast [[TEST37]]* [[T1]] to i8*
  // CHECK-NEXT: [[T3:%.*]] = call i8* @objc_retain(i8* [[T2]])
  // CHECK-NEXT: [[T4:%.*]] = bitcast i8* [[T3]] to [[TEST37]]*
  // CHECK-NEXT: call void (...) @clang.arc.use(i8* [[W1]]) [[NUW]]
  // CHECK-NEXT: [[T5:%.*]] = load [[TEST37]]*, [[TEST37]]** [[VAR]]
  // CHECK-NEXT: store [[TEST37]]* [[T4]], [[TEST37]]** [[VAR]]
  // CHECK-NEXT: [[T6:%.*]] = bitcast [[TEST37]]* [[T5]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T6]])

  // CHECK-NEXT: [[T0:%.*]] = load [[TEST37]]*, [[TEST37]]** [[VAR]]
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST37]]* [[T0]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T1]])
  // CHECK-NEXT: [[VARPTR2:%.*]] = bitcast [[TEST37]]** [[VAR]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end(i64 8, i8* [[VARPTR2]])
  // CHECK-NEXT: ret void
}

@interface Test43 @end
@implementation Test43
- (id) test __attribute__((ns_returns_retained)) {
  extern id test43_produce(void);
  return test43_produce();
  // CHECK:      call i8* @test43_produce()
  // CHECK-NEXT: call i8* @objc_retainAutoreleasedReturnValue(
  // CHECK-NEXT: ret 
}
@end

@interface Test45
@property (retain) id x;
@end
@implementation Test45
@synthesize x;
@end
// CHECK:    define internal i8* @"\01-[Test45 x]"(
// CHECK:      [[CALL:%.*]] = tail call i8* @objc_getProperty(
// CHECK-NEXT: ret i8* [[CALL]]

// rdar://problem/9315552
void test46(__weak id *wp, __weak volatile id *wvp) {
  extern id test46_helper(void);

  // TODO: this is sub-optimal, we should retain at the actual call site.

  // CHECK:      [[T0:%.*]] = call i8* @test46_helper()
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T0]])
  // CHECK-NEXT: [[T2:%.*]] = load i8**, i8*** {{%.*}}, align 8
  // CHECK-NEXT: [[T3:%.*]] = call i8* @objc_storeWeak(i8** [[T2]], i8* [[T1]])
  // CHECK-NEXT: [[T4:%.*]] = call i8* @objc_retain(i8* [[T3]])
  // CHECK-NEXT: store i8* [[T4]], i8**
  // CHECK-NEXT: call void @objc_release(i8* [[T1]])
  id x = *wp = test46_helper();

  // CHECK:      [[T0:%.*]] = call i8* @test46_helper()
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T0]])
  // CHECK-NEXT: [[T2:%.*]] = load i8**, i8*** {{%.*}}, align 8
  // CHECK-NEXT: [[T3:%.*]] = call i8* @objc_storeWeak(i8** [[T2]], i8* [[T1]])
  // CHECK-NEXT: [[T4:%.*]] = call i8* @objc_retain(i8* [[T3]])
  // CHECK-NEXT: store i8* [[T4]], i8**
  // CHECK-NEXT: call void @objc_release(i8* [[T1]])
  id y = *wvp = test46_helper();
}

// rdar://problem/9378887
void test47(void) {
  extern id test47_helper(void);
  id x = x = test47_helper();

  // CHECK-LABEL:    define void @test47()
  // CHECK:      [[X:%.*]] = alloca i8*
  // CHECK-NEXT: [[XPTR1:%.*]] = bitcast i8** [[X]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start(i64 8, i8* [[XPTR1]])
  // CHECK-NEXT: store i8* null, i8** [[X]]
  // CHECK-NEXT: [[CALL:%.*]] = call i8* @test47_helper()
  // CHECK-NEXT: [[T0:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[CALL]])
  // CHECK-NEXT: [[T1:%.*]] = load i8*, i8** [[X]]
  // CHECK-NEXT: store i8* [[T0]], i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[T1]])
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retain(i8* [[T0]])
  // CHECK-NEXT: [[T3:%.*]] = load i8*, i8** [[X]]
  // CHECK-NEXT: store i8* [[T2]], i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[T3]])
  // CHECK-NEXT: [[T4:%.*]] = load i8*, i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[T4]])
  // CHECK-NEXT: [[XPTR2:%.*]] = bitcast i8** [[X]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end(i64 8, i8* [[XPTR2]])
  // CHECK-NEXT: ret void
}

void test48(void) {
  extern id test48_helper(void);
  __weak id x = x = test48_helper();
  // CHECK-LABEL:    define void @test48()
  // CHECK:      [[X:%.*]] = alloca i8*
  // CHECK-NEXT: [[XPTR1:%.*]] = bitcast i8** [[X]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start(i64 8, i8* [[XPTR1]])
  // CHECK-NEXT: [[T0:%.*]] = call i8* @objc_initWeak(i8** [[X]], i8* null)
  // CHECK-NEXT: [[T1:%.*]] = call i8* @test48_helper()
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T1]])
  // CHECK-NEXT: [[T3:%.*]] = call i8* @objc_storeWeak(i8** [[X]], i8* [[T2]])
  // CHECK-NEXT: [[T4:%.*]] = call i8* @objc_storeWeak(i8** [[X]], i8* [[T3]])
  // CHECK-NEXT: call void @objc_release(i8* [[T2]])
  // CHECK-NEXT: call void @objc_destroyWeak(i8** [[X]])
  // CHECK-NEXT: [[XPTR2:%.*]] = bitcast i8** [[X]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end(i64 8, i8* [[XPTR2]])
  // CHECK-NEXT: ret void
}

void test49(void) {
  extern id test49_helper(void);
  __autoreleasing id x = x = test49_helper();
  // CHECK-LABEL:    define void @test49()
  // CHECK:      [[X:%.*]] = alloca i8*
  // CHECK-NEXT: [[XPTR1:%.*]] = bitcast i8** [[X]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start(i64 8, i8* [[XPTR1]])
  // CHECK-NEXT: store i8* null, i8** [[X]]
  // CHECK-NEXT: [[CALL:%.*]] = call i8* @test49_helper()
  // CHECK-NEXT: [[T0:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[CALL]])
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_autorelease(i8* [[T0]])
  // CHECK-NEXT: store i8* [[T2]], i8** [[X]]
  // CHECK-NEXT: [[T3:%.*]] = call i8* @objc_retainAutorelease(i8* [[T1]])
  // CHECK-NEXT: store i8* [[T3]], i8** [[X]]
  // CHECK-NEXT: [[XPTR2:%.*]] = bitcast i8** [[X]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end(i64 8, i8* [[XPTR2]])
  // CHECK-NEXT: ret void
}

// rdar://9380136
id x();
void test50(id y) {
  ({x();});
// CHECK: [[T0:%.*]] = call i8* @objc_retain
// CHECK: call void @objc_release
}


// rdar://9400762
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

// rdar://problem/9400398
id test52(void) {
  id test52_helper(int) __attribute__((ns_returns_retained));
  return ({ int x = 5; test52_helper(x); });

// CHECK-LABEL:    define i8* @test52()
// CHECK:      [[X:%.*]] = alloca i32
// CHECK-NEXT: [[TMPALLOCA:%.*]] = alloca i8*
// CHECK-NEXT: [[XPTR1:%.*]] = bitcast i32* [[X]] to i8*
// CHECK-NEXT: call void @llvm.lifetime.start(i64 4, i8* [[XPTR1]])
// CHECK-NEXT: store i32 5, i32* [[X]],
// CHECK-NEXT: [[T0:%.*]] = load i32, i32* [[X]],
// CHECK-NEXT: [[T1:%.*]] = call i8* @test52_helper(i32 [[T0]])
// CHECK-NEXT: store i8* [[T1]], i8** [[TMPALLOCA]]
// CHECK-NEXT: [[XPTR2:%.*]] = bitcast i32* [[X]] to i8*
// CHECK-NEXT: call void @llvm.lifetime.end(i64 4, i8* [[XPTR2]])
// CHECK-NEXT: [[T2:%.*]] = load i8*, i8** [[TMPALLOCA]]
// CHECK-NEXT: [[T3:%.*]] = tail call i8* @objc_autoreleaseReturnValue(i8* [[T2]])
// CHECK-NEXT: ret i8* [[T3]]
}

// rdar://problem/9400644
void test53(void) {
  id test53_helper(void);
  id x = ({ id y = test53_helper(); y; });
  (void) x;
// CHECK-LABEL:    define void @test53()
// CHECK:      [[X:%.*]] = alloca i8*,
// CHECK-NEXT: [[Y:%.*]] = alloca i8*,
// CHECK-NEXT: [[TMPALLOCA:%.*]] = alloca i8*,
// CHECK-NEXT: [[XPTR1:%.*]] = bitcast i8** [[X]] to i8*
// CHECK-NEXT: call void @llvm.lifetime.start(i64 8, i8* [[XPTR1]])
// CHECK-NEXT: [[YPTR1:%.*]] = bitcast i8** [[Y]] to i8*
// CHECK-NEXT: call void @llvm.lifetime.start(i64 8, i8* [[YPTR1]])
// CHECK-NEXT: [[T0:%.*]] = call i8* @test53_helper()
// CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T0]])
// CHECK-NEXT: store i8* [[T1]], i8** [[Y]],
// CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[Y]],
// CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retain(i8* [[T0]])
// CHECK-NEXT: store i8* [[T1]], i8** [[TMPALLOCA]]
// CHECK-NEXT: [[T2:%.*]] = load i8*, i8** [[Y]]
// CHECK-NEXT: call void @objc_release(i8* [[T2]])
// CHECK-NEXT: [[YPTR2:%.*]] = bitcast i8** [[Y]] to i8*
// CHECK-NEXT: call void @llvm.lifetime.end(i64 8, i8* [[YPTR2]])
// CHECK-NEXT: [[T3:%.*]] = load i8*, i8** [[TMPALLOCA]]
// CHECK-NEXT: store i8* [[T3]], i8** [[X]],
// CHECK-NEXT: load i8*, i8** [[X]],
// CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[X]]
// CHECK-NEXT: call void @objc_release(i8* [[T0]])
// CHECK-NEXT: [[XPTR2:%.*]] = bitcast i8** [[X]] to i8*
// CHECK-NEXT: call void @llvm.lifetime.end(i64 8, i8* [[XPTR2]])
// CHECK-NEXT: ret void
}

// <rdar://problem/9758798>
// CHECK-LABEL: define void @test54(i32 %first, ...)
void test54(int first, ...) {
  __builtin_va_list arglist;
  // CHECK: call void @llvm.va_start
  __builtin_va_start(arglist, first);
  // CHECK: call i8* @objc_retain
  id obj = __builtin_va_arg(arglist, id);
  // CHECK: call void @llvm.va_end
  __builtin_va_end(arglist);
  // CHECK: call void @objc_release
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
// CHECK:     call void bitcast (i8* ({{%.*}}*, i8*, ...)* @objc_msgSendSuper2 to void ({{%.*}}*, i8*)*)(

// rdar://problem/8024350
@protocol Test56Protocol
+ (id) make __attribute__((ns_returns_retained));
@end
@interface Test56<Test56Protocol> @end
@implementation Test56
// CHECK: define internal i8* @"\01+[Test56 make]"(
+ (id) make {
  extern id test56_helper(void);
  // CHECK:      [[T0:%.*]] = call i8* @test56_helper()
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T0]])
  // CHECK-NEXT: ret i8* [[T1]]
  return test56_helper();
}
@end
void test56_test(void) {
  id x = [Test56 make];
  // CHECK-LABEL: define void @test56_test()
  // CHECK:      [[X:%.*]] = alloca i8*, align 8
  // CHECK-NEXT: [[XPTR1:%.*]] = bitcast i8** [[X]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start(i64 8, i8* [[XPTR1]])
  // CHECK:      [[T0:%.*]] = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(
  // CHECK-NEXT: store i8* [[T0]], i8** [[X]]
  // CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]])
  // CHECK-NEXT: [[XPTR2:%.*]] = bitcast i8** [[X]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end(i64 8, i8* [[XPTR2]])
  // CHECK-NEXT: ret void
}

// rdar://problem/9784964
@interface Test57
@property (nonatomic, strong) id strong;
@property (nonatomic, weak) id weak;
@property (nonatomic, unsafe_unretained) id unsafe;
@end
@implementation Test57
@synthesize strong, weak, unsafe;
@end
// CHECK: define internal i8* @"\01-[Test57 strong]"(
// CHECK:      [[T0:%.*]] = load [[TEST57:%.*]]*, [[TEST57:%.*]]** {{%.*}}
// CHECK-NEXT: [[T1:%.*]] = load i64, i64* @"OBJC_IVAR_$_Test57.strong"
// CHECK-NEXT: [[T2:%.*]] = bitcast [[TEST57]]* [[T0]] to i8*
// CHECK-NEXT: [[T3:%.*]] = getelementptr inbounds i8, i8* [[T2]], i64 [[T1]]
// CHECK-NEXT: [[T4:%.*]] = bitcast i8* [[T3]] to i8**
// CHECK-NEXT: [[T5:%.*]] = load i8*, i8** [[T4]]
// CHECK-NEXT: ret i8* [[T5]]

// CHECK: define internal i8* @"\01-[Test57 weak]"(
// CHECK:      [[T0:%.*]] = load [[TEST57]]*, [[TEST57]]** {{%.*}}
// CHECK-NEXT: [[T1:%.*]] = load i64, i64* @"OBJC_IVAR_$_Test57.weak"
// CHECK-NEXT: [[T2:%.*]] = bitcast [[TEST57]]* [[T0]] to i8*
// CHECK-NEXT: [[T3:%.*]] = getelementptr inbounds i8, i8* [[T2]], i64 [[T1]]
// CHECK-NEXT: [[T4:%.*]] = bitcast i8* [[T3]] to i8**
// CHECK-NEXT: [[T5:%.*]] = call i8* @objc_loadWeakRetained(i8** [[T4]])
// CHECK-NEXT: [[T6:%.*]] = tail call i8* @objc_autoreleaseReturnValue(i8* [[T5]])
// CHECK-NEXT: ret i8* [[T6]]

// CHECK: define internal i8* @"\01-[Test57 unsafe]"(
// CHECK:      [[T0:%.*]] = load [[TEST57]]*, [[TEST57]]** {{%.*}}
// CHECK-NEXT: [[T1:%.*]] = load i64, i64* @"OBJC_IVAR_$_Test57.unsafe"
// CHECK-NEXT: [[T2:%.*]] = bitcast [[TEST57]]* [[T0]] to i8*
// CHECK-NEXT: [[T3:%.*]] = getelementptr inbounds i8, i8* [[T2]], i64 [[T1]]
// CHECK-NEXT: [[T4:%.*]] = bitcast i8* [[T3]] to i8**
// CHECK-NEXT: [[T5:%.*]] = load i8*, i8** [[T4]]
// CHECK-NEXT: ret i8* [[T5]]

// rdar://problem/9842343
void test59(void) {
  extern id test59_getlock(void);
  extern void test59_body(void);
  @synchronized (test59_getlock()) {
    test59_body();
  }

  // CHECK-LABEL:    define void @test59()
  // CHECK:      [[T0:%.*]] = call i8* @test59_getlock()
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T0]])
  // CHECK-NEXT: call i32 @objc_sync_enter(i8* [[T1]])
  // CHECK-NEXT: call void @test59_body()
  // CHECK-NEXT: call i32 @objc_sync_exit(i8* [[T1]])
  // CHECK-NEXT: call void @objc_release(i8* [[T1]])
  // CHECK-NEXT: ret void
}

// Verify that we don't try to reclaim the result of performSelector.
// rdar://problem/9887545
@interface Test61
- (id) performSelector: (SEL) selector;
- (void) test61_void;
- (id) test61_id;
@end
void test61(void) {
  // CHECK-LABEL:    define void @test61()
  // CHECK:      [[Y:%.*]] = alloca i8*, align 8

  extern id test61_make(void);

  // CHECK-NEXT: [[T0:%.*]] = call i8* @test61_make()
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T0]])
  // CHECK-NEXT: [[T2:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[T3:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[T4:%.*]] = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8*)*)(i8* [[T1]], i8* [[T3]], i8* [[T2]])
  // CHECK-NEXT: call void @objc_release(i8* [[T1]])
  [test61_make() performSelector: @selector(test61_void)];

  // CHECK-NEXT: [[YPTR1:%.*]] = bitcast i8** [[Y]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start(i64 8, i8* [[YPTR1]])
  // CHECK-NEXT: [[T0:%.*]] = call i8* @test61_make()
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T0]])
  // CHECK-NEXT: [[T2:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[T3:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[T4:%.*]] = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8*)*)(i8* [[T1]], i8* [[T3]], i8* [[T2]])
  // CHECK-NEXT: [[T5:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T4]])
  // CHECK-NEXT: store i8* [[T5]], i8** [[Y]]
  // CHECK-NEXT: call void @objc_release(i8* [[T1]])
  id y = [test61_make() performSelector: @selector(test61_id)];

  // CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[Y]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]])
  // CHECK-NEXT: [[YPTR2:%.*]] = bitcast i8** [[Y]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end(i64 8, i8* [[YPTR2]])
  // CHECK-NEXT: ret void
}

// rdar://problem/9891815
void test62(void) {
  // CHECK-LABEL:    define void @test62()
  // CHECK:      [[I:%.*]] = alloca i32, align 4
  // CHECK-NEXT: [[CLEANUP_VALUE:%.*]] = alloca i8*
  // CHECK-NEXT: [[CLEANUP_REQUIRED:%.*]] = alloca i1
  extern id test62_make(void);
  extern void test62_body(void);

  // CHECK-NEXT: [[IPTR:%.*]] = bitcast i32* [[I]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start(i64 4, i8* [[IPTR]])
  // CHECK-NEXT: store i32 0, i32* [[I]], align 4
  // CHECK-NEXT: br label

  // CHECK:      [[T0:%.*]] = load i32, i32* [[I]], align 4
  // CHECK-NEXT: [[T1:%.*]] = icmp ne i32 [[T0]], 20
  // CHECK-NEXT: br i1 [[T1]],

  for (unsigned i = 0; i != 20; ++i) {
    // CHECK:      [[T0:%.*]] = load i32, i32* [[I]], align 4
    // CHECK-NEXT: [[T1:%.*]] = icmp ne i32 [[T0]], 0
    // CHECK-NEXT: store i1 false, i1* [[CLEANUP_REQUIRED]]
    // CHECK-NEXT: br i1 [[T1]],
    // CHECK:      [[T0:%.*]] = call i8* @test62_make()
    // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T0]])
    // CHECK-NEXT: store i8* [[T1]], i8** [[CLEANUP_VALUE]]
    // CHECK-NEXT: store i1 true, i1* [[CLEANUP_REQUIRED]]
    // CHECK-NEXT: [[T2:%.*]] = icmp ne i8* [[T1]], null
    // CHECK-NEXT: br label
    // CHECK:      [[COND:%.*]] = phi i1 [ false, {{%.*}} ], [ [[T2]], {{%.*}} ]
    // CHECK-NEXT: [[T0:%.*]] = load i1, i1* [[CLEANUP_REQUIRED]]
    // CHECK-NEXT: br i1 [[T0]],
    // CHECK:      [[T0:%.*]] = load i8*, i8** [[CLEANUP_VALUE]]
    // CHECK-NEXT: call void @objc_release(i8* [[T0]])
    // CHECK-NEXT: br label
    // CHECK:      br i1 [[COND]]
    // CHECK:      call void @test62_body()
    // CHECK-NEXT: br label
    // CHECK:      br label
    if (i != 0 && test62_make() != 0)
      test62_body();
  }

  // CHECK:      [[T0:%.*]] = load i32, i32* [[I]], align 4
  // CHECK-NEXT: [[T1:%.*]] = add i32 [[T0]], 1
  // CHECK-NEXT: store i32 [[T1]], i32* [[I]]
  // CHECK-NEXT: br label

  // CHECK:      ret void
}

// rdar://9971982
@class NSString;

@interface Person  {
  NSString *name;
}
@property NSString *address;
@end

@implementation Person
@synthesize address;
@end
// CHECK: tail call i8* @objc_getProperty
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
// CHECK-LABEL:    define void @test66()
// CHECK:      [[T0:%.*]] = call [[TEST66:%.*]]* @test66_receiver()
// CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST66]]* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T1]])
// CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to [[TEST66]]*
// CHECK-NEXT: [[T4:%.*]] = call i8* @test66_arg()
// CHECK-NEXT: [[T5:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T4]])
// CHECK-NEXT: [[T6:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES
// CHECK-NEXT: [[T7:%.*]] = bitcast [[TEST66]]* [[T3]] to i8*
// CHECK-NEXT: [[SIX:%.*]] = icmp eq i8* [[T7]], null
// CHECK-NEXT: br i1 [[SIX]], label [[NULINIT:%.*]], label [[CALL:%.*]]
// CHECK: call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, i8*)*)(i8* [[T7]], i8* [[T6]], i8* [[T5]])
// CHECK-NEXT: br label [[CONT:%.*]]
// CHECK: call void @objc_release(i8* [[T5]]) [[NUW]]
// CHECK-NEXT: br label [[CONT:%.*]]
// CHECK: [[T8:%.*]] = bitcast [[TEST66]]* [[T3]] to i8*
// CHECK-NEXT: call void @objc_release(i8* [[T8]])
// CHECK-NEXT: ret void

// rdar://problem/9953540
Class test67_helper(void);
void test67(void) {
  Class cl = test67_helper();
}
// CHECK-LABEL:    define void @test67()
// CHECK:      [[CL:%.*]] = alloca i8*, align 8
// CHECK-NEXT: [[CLPTR1:%.*]] = bitcast i8** [[CL]] to i8*
// CHECK-NEXT: call void @llvm.lifetime.start(i64 8, i8* [[CLPTR1]])
// CHECK-NEXT: [[T0:%.*]] = call i8* @test67_helper()
// CHECK-NEXT: store i8* [[T0]], i8** [[CL]], align 8
// CHECK-NEXT: [[CLPTR2:%.*]] = bitcast i8** [[CL]] to i8*
// CHECK-NEXT: call void @llvm.lifetime.end(i64 8, i8* [[CLPTR2]])
// CHECK-NEXT: ret void

Class test68_helper(void);
void test68(void) {
  __strong Class cl = test67_helper();
}
// CHECK-LABEL:    define void @test68()
// CHECK:      [[CL:%.*]] = alloca i8*, align 8
// CHECK-NEXT: [[CLPTR1:%.*]] = bitcast i8** [[CL]] to i8*
// CHECK-NEXT: call void @llvm.lifetime.start(i64 8, i8* [[CLPTR1]])
// CHECK-NEXT: [[T0:%.*]] = call i8* @test67_helper()
// CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T0]])
// CHECK-NEXT: store i8* [[T1]], i8** [[CL]], align 8
// CHECK-NEXT: [[T2:%.*]] = load i8*, i8** [[CL]]
// CHECK-NEXT: call void @objc_release(i8* [[T2]])
// CHECK-NEXT: [[CLPTR2:%.*]] = bitcast i8** [[CL]] to i8*
// CHECK-NEXT: call void @llvm.lifetime.end(i64 8, i8* [[CLPTR2]])
// CHECK-NEXT: ret void

// rdar://problem/10564852
@interface Test69 @end
@implementation Test69
- (id) foo { return self; }
@end
// CHECK: define internal i8* @"\01-[Test69 foo]"(
// CHECK:      [[SELF:%.*]] = alloca [[TEST69:%.*]]*, align 8
// CHECK:      [[T0:%.*]] = load [[TEST69]]*, [[TEST69]]** [[SELF]], align 8
// CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST69]]* [[T0]] to i8*
// CHECK-NEXT: [[RETAIN:%.*]] = call i8* @objc_retain(i8* [[T1]])
// CHECK-NEXT: [[AUTORELEASE:%.*]] = tail call i8* @objc_autoreleaseReturnValue(i8* [[RETAIN]])
// CHECK-NEXT: ret i8* [[AUTORELEASE]]

// rdar://problem/10907547
void test70(id i) {
  // CHECK-LABEL: define void @test70
  // CHECK: store i8* null, i8**
  // CHECK: store i8* null, i8**
  // CHECK: [[ID:%.*]] = call i8* @objc_retain(i8*
  // CHECK: store i8* [[ID]], i8**
  id x[3] = {
    [2] = i
  };
}

// ARC-ALIEN: attributes [[NLB]] = { nonlazybind }
// ARC-NATIVE: attributes [[NLB]] = { nonlazybind }
// CHECK: attributes [[NUW]] = { nounwind }
