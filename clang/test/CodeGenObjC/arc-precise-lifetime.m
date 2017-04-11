// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -O2 -disable-llvm-passes -o - %s | FileCheck %s

#define PRECISE_LIFETIME __attribute__((objc_precise_lifetime))

id test0_helper(void) __attribute__((ns_returns_retained));
void test0() {
  PRECISE_LIFETIME id x = test0_helper();
  x = 0;
  // CHECK:      [[X:%.*]] = alloca i8*
  // CHECK-NEXT: [[XPTR1:%.*]] = bitcast i8** [[X]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 8, i8* [[XPTR1]])
  // CHECK-NEXT: [[CALL:%.*]] = call i8* @test0_helper()
  // CHECK-NEXT: store i8* [[CALL]], i8** [[X]]

  // CHECK-NEXT: [[T1:%.*]] = load i8*, i8** [[X]]
  // CHECK-NEXT: store i8* null, i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[T1]]) [[NUW:#[0-9]+]]
  // CHECK-NOT:  clang.imprecise_release

  // CHECK-NEXT: [[T1:%.*]] = load i8*, i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[T1]]) [[NUW:#[0-9]+]]
  // CHECK-NOT:  clang.imprecise_release

  // CHECK-NEXT: [[XPTR2:%.*]] = bitcast i8** [[X]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* [[XPTR2]])
  // CHECK-NEXT: ret void
}

// rdar://problem/9821110 - precise lifetime should suppress extension
// rdar://problem/22172983 - should work for calls via property syntax, too
@interface Test1
- (char*) interior __attribute__((objc_returns_inner_pointer));
// Should we allow this on properties? Yes! see // rdar://14990439
@property (nonatomic, readonly) char * PropertyReturnsInnerPointer __attribute__((objc_returns_inner_pointer));
@end
extern Test1 *test1_helper(void);

// CHECK-LABEL: define void @test1a_message()
void test1a_message(void) {
  // CHECK:      [[PTR:%.*]] = alloca [[PTR_T:%.*]]*, align 8
  // CHECK:      [[C:%.*]] = alloca i8*, align 8
  // CHECK:      [[PTRPTR1:%.*]] = bitcast [[PTR_T]]** [[PTR]] to i8*
  // CHECK:      call void @llvm.lifetime.start.p0i8(i64 8, i8* [[PTRPTR1]])
  // CHECK:      [[T0:%.*]] = call [[TEST1:%.*]]* @test1_helper()
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST1]]* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T1]])
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to [[TEST1]]*
  // CHECK-NEXT: store [[TEST1]]* [[T3]]
  // CHECK-NEXT: [[CPTR1:%.*]] = bitcast i8** [[C]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 8, i8* [[CPTR1]])
  // CHECK-NEXT: [[T0:%.*]] = load [[TEST1]]*, [[TEST1]]**
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST1]]* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retainAutorelease(i8* [[T1]])
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to [[TEST1]]*
  // CHECK-NEXT: [[T4:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[T5:%.*]] = bitcast [[TEST1]]* [[T3]] to i8*
  // CHECK-NEXT: [[T6:%.*]] = call i8* bitcast
  // CHECK-NEXT: store i8* [[T6]], i8**
  // CHECK-NEXT: [[CPTR2:%.*]] = bitcast i8** [[C]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* [[CPTR2]])
  // CHECK-NEXT: [[T0:%.*]] = load [[TEST1]]*, [[TEST1]]**
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST1]]* [[T0]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T1]]) [[NUW]], !clang.imprecise_release
  // CHECK-NEXT: [[PTRPTR2:%.*]] = bitcast [[PTR_T]]** [[PTR]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* [[PTRPTR2]])
  // CHECK-NEXT: ret void
  Test1 *ptr = test1_helper();
  char *c = [(ptr) interior];
}


// CHECK-LABEL: define void @test1a_property()
void test1a_property(void) {
  // CHECK:      [[PTR:%.*]] = alloca [[PTR_T:%.*]]*, align 8
  // CHECK:      [[C:%.*]] = alloca i8*, align 8
  // CHECK:      [[PTRPTR1:%.*]] = bitcast [[PTR_T]]** [[PTR]] to i8*
  // CHECK:      call void @llvm.lifetime.start.p0i8(i64 8, i8* [[PTRPTR1]])
  // CHECK:      [[T0:%.*]] = call [[TEST1:%.*]]* @test1_helper()
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST1]]* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T1]])
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to [[TEST1]]*
  // CHECK-NEXT: store [[TEST1]]* [[T3]]
  // CHECK-NEXT: [[CPTR1:%.*]] = bitcast i8** [[C]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 8, i8* [[CPTR1]])
  // CHECK-NEXT: [[T0:%.*]] = load [[TEST1]]*, [[TEST1]]**
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST1]]* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retainAutorelease(i8* [[T1]])
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to [[TEST1]]*
  // CHECK-NEXT: [[T4:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[T5:%.*]] = bitcast [[TEST1]]* [[T3]] to i8*
  // CHECK-NEXT: [[T6:%.*]] = call i8* bitcast
  // CHECK-NEXT: store i8* [[T6]], i8**
  // CHECK-NEXT: [[CPTR2:%.*]] = bitcast i8** [[C]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* [[CPTR2]])
  // CHECK-NEXT: [[T0:%.*]] = load [[TEST1]]*, [[TEST1]]**
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST1]]* [[T0]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T1]]) [[NUW]], !clang.imprecise_release
  // CHECK-NEXT: [[PTRPTR2:%.*]] = bitcast [[PTR_T]]** [[PTR]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* [[PTRPTR2]])
  // CHECK-NEXT: ret void
  Test1 *ptr = test1_helper();
  char *c = ptr.interior;
}


// CHECK-LABEL: define void @test1b_message()
void test1b_message(void) {
  // CHECK:      [[PTR:%.*]] = alloca [[PTR_T:%.*]]*, align 8
  // CHECK:      [[C:%.*]] = alloca i8*, align 8
  // CHECK:      [[PTRPTR1:%.*]] = bitcast [[PTR_T]]** [[PTR]] to i8*
  // CHECK:      call void @llvm.lifetime.start.p0i8(i64 8, i8* [[PTRPTR1]])
  // CHECK:      [[T0:%.*]] = call [[TEST1:%.*]]* @test1_helper()
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST1]]* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T1]])
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to [[TEST1]]*
  // CHECK-NEXT: store [[TEST1]]* [[T3]]
  // CHECK-NEXT: [[CPTR1:%.*]] = bitcast i8** [[C]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 8, i8* [[CPTR1]])
  // CHECK-NEXT: [[T0:%.*]] = load [[TEST1]]*, [[TEST1]]**
  // CHECK-NEXT: [[T1:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[T2:%.*]] = bitcast [[TEST1]]* [[T0]] to i8*
  // CHECK-NEXT: [[T3:%.*]] = call i8* bitcast
  // CHECK-NEXT: store i8* [[T3]], i8**
  // CHECK-NEXT: [[CPTR2:%.*]] = bitcast i8** [[C]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* [[CPTR2]])
  // CHECK-NEXT: [[T0:%.*]] = load [[TEST1]]*, [[TEST1]]**
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST1]]* [[T0]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T1]]) [[NUW]]
  // CHECK-NOT:  clang.imprecise_release
  // CHECK-NEXT: [[PTRPTR2:%.*]] = bitcast [[PTR_T]]** [[PTR]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* [[PTRPTR2]])
  // CHECK-NEXT: ret void
  PRECISE_LIFETIME Test1 *ptr = test1_helper();
  char *c = [ptr interior];
}

// CHECK-LABEL: define void @test1b_property()
void test1b_property(void) {
  // CHECK:      [[PTR:%.*]] = alloca [[PTR_T:%.*]]*, align 8
  // CHECK:      [[C:%.*]] = alloca i8*, align 8
  // CHECK:      [[PTRPTR1:%.*]] = bitcast [[PTR_T]]** [[PTR]] to i8*
  // CHECK:      call void @llvm.lifetime.start.p0i8(i64 8, i8* [[PTRPTR1]])
  // CHECK:      [[T0:%.*]] = call [[TEST1:%.*]]* @test1_helper()
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST1]]* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T1]])
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to [[TEST1]]*
  // CHECK-NEXT: store [[TEST1]]* [[T3]]
  // CHECK-NEXT: [[CPTR1:%.*]] = bitcast i8** [[C]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 8, i8* [[CPTR1]])
  // CHECK-NEXT: [[T0:%.*]] = load [[TEST1]]*, [[TEST1]]**
  // CHECK-NEXT: [[T1:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[T2:%.*]] = bitcast [[TEST1]]* [[T0]] to i8*
  // CHECK-NEXT: [[T3:%.*]] = call i8* bitcast
  // CHECK-NEXT: store i8* [[T3]], i8**
  // CHECK-NEXT: [[CPTR2:%.*]] = bitcast i8** [[C]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* [[CPTR2]])
  // CHECK-NEXT: [[T0:%.*]] = load [[TEST1]]*, [[TEST1]]**
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST1]]* [[T0]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T1]]) [[NUW]]
  // CHECK-NOT:  clang.imprecise_release
  // CHECK-NEXT: [[PTRPTR2:%.*]] = bitcast [[PTR_T]]** [[PTR]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* [[PTRPTR2]])
  // CHECK-NEXT: ret void
  PRECISE_LIFETIME Test1 *ptr = test1_helper();
  char *c = ptr.interior;
}

// CHECK-LABEL: define void @test1c_message()
void test1c_message(void) {
  // CHECK:      [[PTR:%.*]] = alloca [[PTR_T:%.*]]*, align 8
  // CHECK:      [[PC:%.*]] = alloca i8*, align 8
  // CHECK:      [[PTRPTR1:%.*]] = bitcast [[PTR_T]]** [[PTR]] to i8*
  // CHECK:      call void @llvm.lifetime.start.p0i8(i64 8, i8* [[PTRPTR1]])
  // CHECK:      [[T0:%.*]] = call [[TEST1:%.*]]* @test1_helper()
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST1]]* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T1]])
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to [[TEST1]]*
  // CHECK-NEXT: store [[TEST1]]* [[T3]]
  // CHECK-NEXT: [[PCPTR1:%.*]] = bitcast i8** [[PC]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 8, i8* [[PCPTR1]])
  // CHECK-NEXT: [[T0:%.*]] = load [[TEST1]]*, [[TEST1]]**
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST1]]* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retainAutorelease(i8* [[T1]])
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to [[TEST1]]*
  // CHECK-NEXT: [[T4:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[T5:%.*]] = bitcast [[TEST1]]* [[T3]] to i8*
  // CHECK-NEXT: [[T6:%.*]] = call i8* bitcast
  // CHECK-NEXT: store i8* [[T6]], i8**
  // CHECK-NEXT: [[PCPTR2:%.*]] = bitcast i8** [[PC]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* [[PCPTR2]])
  // CHECK-NEXT: [[T0:%.*]] = load [[TEST1]]*, [[TEST1]]**
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST1]]* [[T0]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T1]]) [[NUW]], !clang.imprecise_release
  // CHECK-NEXT: [[PTRPTR2:%.*]] = bitcast [[PTR_T]]** [[PTR]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* [[PTRPTR2]])
  // CHECK-NEXT: ret void
  Test1 *ptr = test1_helper();
  char *pc = [ptr PropertyReturnsInnerPointer];
}

// CHECK-LABEL: define void @test1c_property()
void test1c_property(void) {
  // CHECK:      [[PTR:%.*]] = alloca [[PTR_T:%.*]]*, align 8
  // CHECK:      [[PC:%.*]] = alloca i8*, align 8
  // CHECK:      [[PTRPTR1:%.*]] = bitcast [[PTR_T]]** [[PTR]] to i8*
  // CHECK:      call void @llvm.lifetime.start.p0i8(i64 8, i8* [[PTRPTR1]])
  // CHECK:      [[T0:%.*]] = call [[TEST1:%.*]]* @test1_helper()
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST1]]* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T1]])
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to [[TEST1]]*
  // CHECK-NEXT: store [[TEST1]]* [[T3]]
  // CHECK-NEXT: [[PCPTR1:%.*]] = bitcast i8** [[PC]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 8, i8* [[PCPTR1]])
  // CHECK-NEXT: [[T0:%.*]] = load [[TEST1]]*, [[TEST1]]**
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST1]]* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retainAutorelease(i8* [[T1]])
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to [[TEST1]]*
  // CHECK-NEXT: [[T4:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[T5:%.*]] = bitcast [[TEST1]]* [[T3]] to i8*
  // CHECK-NEXT: [[T6:%.*]] = call i8* bitcast
  // CHECK-NEXT: store i8* [[T6]], i8**
  // CHECK-NEXT: [[PCPTR2:%.*]] = bitcast i8** [[PC]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* [[PCPTR2]])
  // CHECK-NEXT: [[T0:%.*]] = load [[TEST1]]*, [[TEST1]]**
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST1]]* [[T0]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T1]]) [[NUW]], !clang.imprecise_release
  // CHECK-NEXT: [[PTRPTR2:%.*]] = bitcast [[PTR_T]]** [[PTR]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* [[PTRPTR2]])
  // CHECK-NEXT: ret void
  Test1 *ptr = test1_helper();
  char *pc = ptr.PropertyReturnsInnerPointer;
}

// CHECK-LABEL: define void @test1d_message()
void test1d_message(void) {
  // CHECK:      [[PTR:%.*]] = alloca [[PTR_T:%.*]]*, align 8
  // CHECK:      [[PC:%.*]] = alloca i8*, align 8
  // CHECK:      [[PTRPTR1:%.*]] = bitcast [[PTR_T]]** [[PTR]] to i8*
  // CHECK:      call void @llvm.lifetime.start.p0i8(i64 8, i8* [[PTRPTR1]])
  // CHECK:      [[T0:%.*]] = call [[TEST1:%.*]]* @test1_helper()
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST1]]* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T1]])
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to [[TEST1]]*
  // CHECK-NEXT: store [[TEST1]]* [[T3]]
  // CHECK-NEXT: [[PCPTR1:%.*]] = bitcast i8** [[PC]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 8, i8* [[PCPTR1]])
  // CHECK-NEXT: [[T0:%.*]] = load [[TEST1]]*, [[TEST1]]**
  // CHECK-NEXT: [[SEVEN:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[EIGHT:%.*]] = bitcast [[TEST1]]* [[T0]] to i8*
  // CHECK-NEXT: [[CALL1:%.*]] = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* [[EIGHT]], i8* [[SEVEN]])
  // CHECK-NEXT: store i8* [[CALL1]], i8**
  // CHECK-NEXT: [[PCPTR2:%.*]] = bitcast i8** [[PC]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* [[PCPTR2]])
  // CHECK-NEXT: [[NINE:%.*]] = load [[TEST1]]*, [[TEST1]]**
  // CHECK-NEXT: [[TEN:%.*]] = bitcast [[TEST1]]* [[NINE]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[TEN]])
  // CHECK-NEXT: [[PTRPTR2:%.*]] = bitcast [[PTR_T]]** [[PTR]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* [[PTRPTR2]])
  // CHECK-NEXT: ret void
  PRECISE_LIFETIME Test1 *ptr = test1_helper();
  char *pc = [ptr PropertyReturnsInnerPointer];
}

// CHECK-LABEL: define void @test1d_property()
void test1d_property(void) {
  // CHECK:      [[PTR:%.*]] = alloca [[PTR_T:%.*]]*, align 8
  // CHECK:      [[PC:%.*]] = alloca i8*, align 8
  // CHECK:      [[PTRPTR1:%.*]] = bitcast [[PTR_T]]** [[PTR]] to i8*
  // CHECK:      call void @llvm.lifetime.start.p0i8(i64 8, i8* [[PTRPTR1]])
  // CHECK:      [[T0:%.*]] = call [[TEST1:%.*]]* @test1_helper()
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST1]]* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T1]])
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to [[TEST1]]*
  // CHECK-NEXT: store [[TEST1]]* [[T3]]
  // CHECK-NEXT: [[PCPTR1:%.*]] = bitcast i8** [[PC]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 8, i8* [[PCPTR1]])
  // CHECK-NEXT: [[T0:%.*]] = load [[TEST1]]*, [[TEST1]]**
  // CHECK-NEXT: [[SEVEN:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[EIGHT:%.*]] = bitcast [[TEST1]]* [[T0]] to i8*
  // CHECK-NEXT: [[CALL1:%.*]] = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* [[EIGHT]], i8* [[SEVEN]])
  // CHECK-NEXT: store i8* [[CALL1]], i8**
  // CHECK-NEXT: [[PCPTR2:%.*]] = bitcast i8** [[PC]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* [[PCPTR2]])
  // CHECK-NEXT: [[NINE:%.*]] = load [[TEST1]]*, [[TEST1]]**
  // CHECK-NEXT: [[TEN:%.*]] = bitcast [[TEST1]]* [[NINE]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[TEN]])
  // CHECK-NEXT: [[PTRPTR2:%.*]] = bitcast [[PTR_T]]** [[PTR]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* [[PTRPTR2]])
  // CHECK-NEXT: ret void
  PRECISE_LIFETIME Test1 *ptr = test1_helper();
  char *pc = ptr.PropertyReturnsInnerPointer;
}

@interface Test2 {
@public
  id ivar;
}
@end
// CHECK-LABEL:      define void @test2(
void test2(Test2 *x) {
  x->ivar = 0;
  // CHECK:      [[X:%.*]] = alloca [[TEST2:%.*]]*
  // CHECK-NEXT: [[T0:%.*]] = bitcast [[TEST2]]* {{%.*}} to i8*
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retain(i8* [[T0]]) [[NUW]]
  // CHECK-NEXT: [[T2:%.*]] = bitcast i8* [[T1]] to [[TEST2]]*
  // CHECK-NEXT: store [[TEST2]]* [[T2]], [[TEST2]]** [[X]],

  // CHECK-NEXT: [[T0:%.*]] = load [[TEST2]]*, [[TEST2]]** [[X]],
  // CHECK-NEXT: [[OFFSET:%.*]] = load i64, i64* @"OBJC_IVAR_$_Test2.ivar"
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST2]]* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = getelementptr inbounds i8, i8* [[T1]], i64 [[OFFSET]]
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to i8**
  // CHECK-NEXT: [[T4:%.*]] = load i8*, i8** [[T3]],
  // CHECK-NEXT: store i8* null, i8** [[T3]],
  // CHECK-NEXT: call void @objc_release(i8* [[T4]]) [[NUW]]
  // CHECK-NOT:  imprecise

  // CHECK-NEXT: [[T0:%.*]] = load [[TEST2]]*, [[TEST2]]** [[X]]
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST2]]* [[T0]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T1]]) [[NUW]], !clang.imprecise_release

  // CHECK-NEXT: ret void
}

// CHECK-LABEL:      define void @test3(i8*
void test3(PRECISE_LIFETIME id x) {
  // CHECK:      [[X:%.*]] = alloca i8*,
  // CHECK-NEXT: [[T0:%.*]] = call i8* @objc_retain(i8* {{%.*}}) [[NUW]]
  // CHECK-NEXT: store i8* [[T0]], i8** [[X]],

  // CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]]) [[NUW]]
  // CHECK-NOT:  imprecise_release

  // CHECK-NEXT: ret void  
}

// CHECK: attributes [[NUW]] = { nounwind }
