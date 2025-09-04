// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -O2 -disable-llvm-passes -o - %s | FileCheck %s

#define PRECISE_LIFETIME __attribute__((objc_precise_lifetime))

id test0_helper(void) __attribute__((ns_returns_retained));
void test0(void) {
  PRECISE_LIFETIME id x = test0_helper();
  x = 0;
  // CHECK:      [[X:%.*]] = alloca ptr
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr [[X]])
  // CHECK-NEXT: [[CALL:%.*]] = call ptr @test0_helper()
  // CHECK-NEXT: store ptr [[CALL]], ptr [[X]]

  // CHECK-NEXT: [[T1:%.*]] = load ptr, ptr [[X]]
  // CHECK-NEXT: store ptr null, ptr [[X]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T1]]) [[NUW:#[0-9]+]]
  // CHECK-NOT:  clang.imprecise_release

  // CHECK-NEXT: [[T1:%.*]] = load ptr, ptr [[X]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T1]]) [[NUW:#[0-9]+]]
  // CHECK-NOT:  clang.imprecise_release

  // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[X]])
  // CHECK-NEXT: ret void
}

// precise lifetime should suppress extension
// should work for calls via property syntax, too
@interface Test1
- (char*) interior __attribute__((objc_returns_inner_pointer));
// Should we allow this on properties? Yes!
@property (nonatomic, readonly) char * PropertyReturnsInnerPointer __attribute__((objc_returns_inner_pointer));
@end
extern Test1 *test1_helper(void);

// CHECK-LABEL: define{{.*}} void @test1a_message()
void test1a_message(void) {
  // CHECK:      [[PTR:%.*]] = alloca ptr, align 8
  // CHECK:      [[C:%.*]] = alloca ptr, align 8
  // CHECK:      call void @llvm.lifetime.start.p0(ptr [[PTR]])
  // CHECK:      [[T0:%.*]] = call ptr @test1_helper() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T0]])
  // CHECK-NEXT: store ptr [[T0]]
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr [[C]])
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr
  // CHECK-NEXT: [[T2:%.*]] = call ptr @llvm.objc.retainAutorelease(ptr [[T0]])
  // CHECK-NEXT: [[T4:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[T6:%.*]] = call ptr
  // CHECK-NEXT: store ptr [[T6]], ptr
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[C]])
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]]) [[NUW]], !clang.imprecise_release
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[PTR]])
  // CHECK-NEXT: ret void
  Test1 *ptr = test1_helper();
  char *c = [(ptr) interior];
}


// CHECK-LABEL: define{{.*}} void @test1a_property()
void test1a_property(void) {
  // CHECK:      [[PTR:%.*]] = alloca ptr, align 8
  // CHECK:      [[C:%.*]] = alloca ptr, align 8
  // CHECK:      call void @llvm.lifetime.start.p0(ptr [[PTR]])
  // CHECK:      [[T0:%.*]] = call ptr @test1_helper() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T0]])
  // CHECK-NEXT: store ptr [[T0]]
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr [[C]])
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr
  // CHECK-NEXT: [[T2:%.*]] = call ptr @llvm.objc.retainAutorelease(ptr [[T0]])
  // CHECK-NEXT: [[T4:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[T6:%.*]] = call ptr
  // CHECK-NEXT: store ptr [[T6]], ptr
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[C]])
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]]) [[NUW]], !clang.imprecise_release
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[PTR]])
  // CHECK-NEXT: ret void
  Test1 *ptr = test1_helper();
  char *c = ptr.interior;
}


// CHECK-LABEL: define{{.*}} void @test1b_message()
void test1b_message(void) {
  // CHECK:      [[PTR:%.*]] = alloca ptr, align 8
  // CHECK:      [[C:%.*]] = alloca ptr, align 8
  // CHECK:      call void @llvm.lifetime.start.p0(ptr [[PTR]])
  // CHECK:      [[T0:%.*]] = call ptr @test1_helper() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T0]])
  // CHECK-NEXT: store ptr [[T0]]
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr [[C]])
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr
  // CHECK-NEXT: [[T1:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[T3:%.*]] = call ptr
  // CHECK-NEXT: store ptr [[T3]], ptr
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[C]])
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]]) [[NUW]]
  // CHECK-NOT:  clang.imprecise_release
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[PTR]])
  // CHECK-NEXT: ret void
  PRECISE_LIFETIME Test1 *ptr = test1_helper();
  char *c = [ptr interior];
}

// CHECK-LABEL: define{{.*}} void @test1b_property()
void test1b_property(void) {
  // CHECK:      [[PTR:%.*]] = alloca ptr, align 8
  // CHECK:      [[C:%.*]] = alloca ptr, align 8
  // CHECK:      call void @llvm.lifetime.start.p0(ptr [[PTR]])
  // CHECK:      [[T0:%.*]] = call ptr @test1_helper() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T0]])
  // CHECK-NEXT: store ptr [[T0]]
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr [[C]])
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr
  // CHECK-NEXT: [[T1:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[T3:%.*]] = call ptr
  // CHECK-NEXT: store ptr [[T3]], ptr
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[C]])
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]]) [[NUW]]
  // CHECK-NOT:  clang.imprecise_release
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[PTR]])
  // CHECK-NEXT: ret void
  PRECISE_LIFETIME Test1 *ptr = test1_helper();
  char *c = ptr.interior;
}

// CHECK-LABEL: define{{.*}} void @test1c_message()
void test1c_message(void) {
  // CHECK:      [[PTR:%.*]] = alloca ptr, align 8
  // CHECK:      [[PC:%.*]] = alloca ptr, align 8
  // CHECK:      call void @llvm.lifetime.start.p0(ptr [[PTR]])
  // CHECK:      [[T0:%.*]] = call ptr @test1_helper() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T0]])
  // CHECK-NEXT: store ptr [[T0]]
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr [[PC]])
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr
  // CHECK-NEXT: [[T2:%.*]] = call ptr @llvm.objc.retainAutorelease(ptr [[T0]])
  // CHECK-NEXT: [[T4:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[T6:%.*]] = call ptr
  // CHECK-NEXT: store ptr [[T6]], ptr
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[PC]])
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]]) [[NUW]], !clang.imprecise_release
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[PTR]])
  // CHECK-NEXT: ret void
  Test1 *ptr = test1_helper();
  char *pc = [ptr PropertyReturnsInnerPointer];
}

// CHECK-LABEL: define{{.*}} void @test1c_property()
void test1c_property(void) {
  // CHECK:      [[PTR:%.*]] = alloca ptr, align 8
  // CHECK:      [[PC:%.*]] = alloca ptr, align 8
  // CHECK:      call void @llvm.lifetime.start.p0(ptr [[PTR]])
  // CHECK:      [[T0:%.*]] = call ptr @test1_helper() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T0]])
  // CHECK-NEXT: store ptr [[T0]]
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr [[PC]])
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr
  // CHECK-NEXT: [[T2:%.*]] = call ptr @llvm.objc.retainAutorelease(ptr [[T0]])
  // CHECK-NEXT: [[T4:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[T6:%.*]] = call ptr
  // CHECK-NEXT: store ptr [[T6]], ptr
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[PC]])
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]]) [[NUW]], !clang.imprecise_release
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[PTR]])
  // CHECK-NEXT: ret void
  Test1 *ptr = test1_helper();
  char *pc = ptr.PropertyReturnsInnerPointer;
}

// CHECK-LABEL: define{{.*}} void @test1d_message()
void test1d_message(void) {
  // CHECK:      [[PTR:%.*]] = alloca ptr, align 8
  // CHECK:      [[PC:%.*]] = alloca ptr, align 8
  // CHECK:      call void @llvm.lifetime.start.p0(ptr [[PTR]])
  // CHECK:      [[T0:%.*]] = call ptr @test1_helper() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T0]])
  // CHECK-NEXT: store ptr [[T0]]
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr [[PC]])
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr
  // CHECK-NEXT: [[SEL:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[CALL1:%.*]] = call ptr @objc_msgSend(ptr noundef [[T0]], ptr noundef [[SEL]])
  // CHECK-NEXT: store ptr [[CALL1]], ptr
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[PC]])
  // CHECK-NEXT: [[NINE:%.*]] = load ptr, ptr
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[NINE]])
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[PTR]])
  // CHECK-NEXT: ret void
  PRECISE_LIFETIME Test1 *ptr = test1_helper();
  char *pc = [ptr PropertyReturnsInnerPointer];
}

// CHECK-LABEL: define{{.*}} void @test1d_property()
void test1d_property(void) {
  // CHECK:      [[PTR:%.*]] = alloca ptr, align 8
  // CHECK:      [[PC:%.*]] = alloca ptr, align 8
  // CHECK:      call void @llvm.lifetime.start.p0(ptr [[PTR]])
  // CHECK:      [[T0:%.*]] = call ptr @test1_helper() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T0]])
  // CHECK-NEXT: store ptr [[T0]]
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr [[PC]])
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr
  // CHECK-NEXT: [[SEL:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[CALL1:%.*]] = call ptr @objc_msgSend(ptr noundef [[T0]], ptr noundef [[SEL]])
  // CHECK-NEXT: store ptr [[CALL1]], ptr
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[PC]])
  // CHECK-NEXT: [[NINE:%.*]] = load ptr, ptr
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[NINE]])
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[PTR]])
  // CHECK-NEXT: ret void
  PRECISE_LIFETIME Test1 *ptr = test1_helper();
  char *pc = ptr.PropertyReturnsInnerPointer;
}

@interface Test2 {
@public
  id ivar;
}
@end
// CHECK-LABEL:      define{{.*}} void @test2(
void test2(Test2 *x) {
  x->ivar = 0;
  // CHECK:      [[X:%.*]] = alloca ptr
  // CHECK-NEXT: [[T1:%.*]] = call ptr @llvm.objc.retain(ptr {{%.*}}) [[NUW]]
  // CHECK-NEXT: store ptr [[T1]], ptr [[X]],

  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[X]],
  // CHECK-NEXT: [[OFFSET:%.*]] = load i64, ptr @"OBJC_IVAR_$_Test2.ivar"
  // CHECK-NEXT: [[T2:%.*]] = getelementptr inbounds i8, ptr [[T0]], i64 [[OFFSET]]
  // CHECK-NEXT: [[T4:%.*]] = load ptr, ptr [[T2]],
  // CHECK-NEXT: store ptr null, ptr [[T2]],
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T4]]) [[NUW]]
  // CHECK-NOT:  imprecise

  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[X]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]]) [[NUW]], !clang.imprecise_release

  // CHECK-NEXT: ret void
}

// CHECK-LABEL:      define{{.*}} void @test3(ptr
void test3(PRECISE_LIFETIME id x) {
  // CHECK:      [[X:%.*]] = alloca ptr,
  // CHECK-NEXT: [[T0:%.*]] = call ptr @llvm.objc.retain(ptr {{%.*}}) [[NUW]]
  // CHECK-NEXT: store ptr [[T0]], ptr [[X]],

  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[X]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]]) [[NUW]]
  // CHECK-NOT:  imprecise_release

  // CHECK-NEXT: ret void  
}

// CHECK: attributes [[NUW]] = { nounwind }
