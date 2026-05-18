// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-arc -fobjc-dispatch-method=mixed -fobjc-runtime-has-weak -fexceptions -fobjc-exceptions -o - %s | FileCheck %s

@interface NSObject
- (id) new;
@end

@interface MyObject : NSObject
- (char)isEqual:(id) __attribute__((ns_consumed)) object;
- (_Complex float) asComplexWithArg: (id) __attribute__((ns_consumed)) object;
+(instancetype)m0:(id) __attribute__((ns_consumed)) object;
@end

MyObject *x;

void test0(void) {
  id obj = [NSObject new];
  [x isEqual : obj];
}
// CHECK-LABEL:     define{{.*}} void @test0()
// CHECK:       [[FIVE:%.*]] = call ptr @llvm.objc.retain
// CHECK-NEXT:  [[SEVEN:%.*]]  = icmp eq ptr {{.*}}, null
// CHECK-NEXT:  br i1 [[SEVEN]], label [[NULLINIT:%.*]], label [[CALL_LABEL:%.*]]
// CHECK:       [[FN:%.*]] = load ptr, ptr
// CHECK-NEXT:  [[CALL:%.*]] = call signext i8 [[FN]]
// CHECK-NEXT:  br label [[CONT:%.*]]
// CHECK:       call void @llvm.objc.release(ptr [[FIVE]]) [[NUW:#[0-9]+]]
// CHECK-NEXT:  br label [[CONT]]
// CHECK:       phi i8 [ [[CALL]], {{%.*}} ], [ 0, {{%.*}} ]

// Ensure that we build PHIs correctly in the presence of cleanups.
void test1(void) {
  id obj = [NSObject new];
  __weak id weakObj = obj;
  _Complex float result = [x asComplexWithArg: obj];
}
// CHECK-LABEL:    define{{.*}} void @test1()
// CHECK:      [[OBJ:%.*]] = alloca ptr, align 8
// CHECK-NEXT: [[WEAKOBJ:%.*]] = alloca ptr, align 8
// CHECK-NEXT: [[RESULT:%.*]] = alloca { float, float }, align 4
//   Various initializations.
// CHECK:      [[T0:%.*]] = call ptr
// CHECK-NEXT: store ptr [[T0]], ptr [[OBJ]]
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[OBJ]]
// CHECK-NEXT: call ptr @llvm.objc.initWeak(ptr [[WEAKOBJ]], ptr [[T0]]) [[NUW]]
//   Okay, start the message-send.
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr @x
// CHECK-NEXT: [[ARG:%.*]] = load ptr, ptr [[OBJ]]
// CHECK-NEXT: [[ARG_RETAINED:%.*]] = call ptr @llvm.objc.retain(ptr [[ARG]])
//   Null check.
// CHECK-NEXT: [[T1:%.*]] = icmp eq ptr [[T0]], null
// CHECK-NEXT: br i1 [[T1]], label [[FORNULL:%.*]], label %[[FORCALL:.*]]
//   Invoke and produce the return values.
// CHECK:     [[FORCALL]]:
// CHECK-NEXT: load ptr, ptr @OBJC_SELECTOR_REFERENCES_
// CHECK-NEXT: [[CALL:%.*]] = invoke <2 x float>
// CHECK-NEXT:   to label [[INVOKE_CONT:%.*]] unwind label {{%.*}}
// CHECK: store <2 x float> [[CALL]], ptr [[COERCE:%.*]],
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds nuw { float, float }, ptr [[COERCE]], i32 0, i32 0
// CHECK-NEXT: [[REALCALL:%.*]] = load float, ptr [[T0]]
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds nuw { float, float }, ptr [[COERCE]], i32 0, i32 1
// CHECK-NEXT: [[IMAGCALL:%.*]] = load float, ptr [[T0]]
// CHECK-NEXT: br label [[CONT:%.*]]{{$}}
//   Null path.
// CHECK:      call void @llvm.objc.release(ptr [[ARG_RETAINED]]) [[NUW]]
// CHECK-NEXT: br label [[CONT]]
//   Join point.
// CHECK:      [[REAL:%.*]] = phi float [ [[REALCALL]], [[INVOKE_CONT]] ], [ 0.000000e+00, [[FORNULL]] ]
// CHECK-NEXT: [[IMAG:%.*]] = phi float [ [[IMAGCALL]], [[INVOKE_CONT]] ], [ 0.000000e+00, [[FORNULL]] ]
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds nuw { float, float }, ptr [[RESULT]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds nuw { float, float }, ptr [[RESULT]], i32 0, i32 1
// CHECK-NEXT: store float [[REAL]], ptr [[T0]]
// CHECK-NEXT: store float [[IMAG]], ptr [[T1]]
//   Epilogue.
// CHECK-NEXT: call void @llvm.objc.destroyWeak(ptr [[WEAKOBJ]]) [[NUW]]
// CHECK-NEXT: call void @llvm.objc.storeStrong(ptr [[OBJ]], ptr null) [[NUW]]
// CHECK-NEXT: ret void
//   Cleanup.
// CHECK:      landingpad
// CHECK:      call void @llvm.objc.destroyWeak(ptr [[WEAKOBJ]]) [[NUW]]

void test2(id a) {
  id obj = [MyObject m0:a];
}

// CHECK-LABEL: define{{.*}} void @test2(
// CHECK: %[[CALL:.*]] = call ptr @objc_msgSend
// CHECK-NEXT: %[[V6:.*]] = {{.*}}call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %[[CALL]])

// CHECK: phi ptr [ %[[V6]], %{{.*}} ], [ null, %{{.*}} ]

void test3(id a) {
  @try {
    id obj = [MyObject m0:a];
  } @catch (id x) {
  }
}

// CHECK-LABEL: define{{.*}} void @test3(
// CHECK: %[[CALL:.*]] = invoke ptr @objc_msgSend
// CHECK: %[[V6:.*]] = {{.*}}call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %[[CALL]])

// CHECK: phi ptr [ %[[V6]], %{{.*}} ], [ null, %{{.*}} ]

// CHECK: attributes [[NUW]] = { nounwind }
