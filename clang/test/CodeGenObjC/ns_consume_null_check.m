// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin10 -emit-llvm -fobjc-arc -fobjc-dispatch-method=mixed -fobjc-runtime-has-weak -fexceptions -fobjc-exceptions -o - %s | FileCheck %s

@interface NSObject
- (id) new;
@end

@interface MyObject : NSObject
- (char)isEqual:(id) __attribute__((ns_consumed)) object;
- (_Complex float) asComplexWithArg: (id) __attribute__((ns_consumed)) object;
+(instancetype)m0:(id) __attribute__((ns_consumed)) object;
@end

MyObject *x;

// rdar://10444476
void test0(void) {
  id obj = [NSObject new];
  [x isEqual : obj];
}
// CHECK-LABEL:     define{{.*}} void @test0()
// CHECK:       [[FIVE:%.*]] = call i8* @llvm.objc.retain
// CHECK-NEXT:  [[SIX:%.*]] = bitcast
// CHECK-NEXT:  [[SEVEN:%.*]]  = icmp eq i8* [[SIX]], null
// CHECK-NEXT:  br i1 [[SEVEN]], label [[NULLINIT:%.*]], label [[CALL_LABEL:%.*]]
// CHECK:       [[FN:%.*]] = load i8*, i8** getelementptr inbounds
// CHECK-NEXT:  [[EIGHT:%.*]] = bitcast i8* [[FN]]
// CHECK-NEXT:  [[CALL:%.*]] = call signext i8 [[EIGHT]]
// CHECK-NEXT:  br label [[CONT:%.*]]
// CHECK:       call void @llvm.objc.release(i8* [[FIVE]]) [[NUW:#[0-9]+]]
// CHECK-NEXT:  br label [[CONT]]
// CHECK:       phi i8 [ [[CALL]], {{%.*}} ], [ 0, {{%.*}} ]

// Ensure that we build PHIs correctly in the presence of cleanups.
// rdar://12046763
void test1(void) {
  id obj = [NSObject new];
  __weak id weakObj = obj;
  _Complex float result = [x asComplexWithArg: obj];
}
// CHECK-LABEL:    define{{.*}} void @test1()
// CHECK:      [[OBJ:%.*]] = alloca i8*, align 8
// CHECK-NEXT: [[WEAKOBJ:%.*]] = alloca i8*, align 8
// CHECK-NEXT: [[RESULT:%.*]] = alloca { float, float }, align 4
//   Various initializations.
// CHECK:      [[T0:%.*]] = call i8* bitcast (
// CHECK-NEXT: store i8* [[T0]], i8** [[OBJ]]
// CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[OBJ]]
// CHECK-NEXT: call i8* @llvm.objc.initWeak(i8** [[WEAKOBJ]], i8* [[T0]]) [[NUW]]
//   Okay, start the message-send.
// CHECK-NEXT: [[T0:%.*]] = load [[MYOBJECT:%.*]]*, [[MYOBJECT:%.*]]** @x
// CHECK-NEXT: [[ARG:%.*]] = load i8*, i8** [[OBJ]]
// CHECK-NEXT: [[ARG_RETAINED:%.*]] = call i8* @llvm.objc.retain(i8* [[ARG]])
// CHECK-NEXT: [[SELF:%.*]] = bitcast [[MYOBJECT]]* [[T0]] to i8*
//   Null check.
// CHECK-NEXT: [[T0:%.*]] = icmp eq i8* [[SELF]], null
// CHECK-NEXT: br i1 [[T0]], label [[FORNULL:%.*]], label %[[FORCALL:.*]]
//   Invoke and produce the return values.
// CHECK:     [[FORCALL]]:
// CHECK-NEXT: load i8*, i8** @OBJC_SELECTOR_REFERENCES_
// CHECK-NEXT: [[CALL:%.*]] = invoke <2 x float> bitcast
// CHECK-NEXT:   to label [[INVOKE_CONT:%.*]] unwind label {{%.*}}
// CHECK:      [[T0:%.*]] = bitcast { float, float }* [[COERCE:%.*]] to <2 x float>*
// CHECK-NEXT: store <2 x float> [[CALL]], <2 x float>* [[T0]],
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds { float, float }, { float, float }* [[COERCE]], i32 0, i32 0
// CHECK-NEXT: [[REALCALL:%.*]] = load float, float* [[T0]]
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds { float, float }, { float, float }* [[COERCE]], i32 0, i32 1
// CHECK-NEXT: [[IMAGCALL:%.*]] = load float, float* [[T0]]
// CHECK-NEXT: br label [[CONT:%.*]]{{$}}
//   Null path.
// CHECK:      call void @llvm.objc.release(i8* [[ARG_RETAINED]]) [[NUW]]
// CHECK-NEXT: br label [[CONT]]
//   Join point.
// CHECK:      [[REAL:%.*]] = phi float [ [[REALCALL]], [[INVOKE_CONT]] ], [ 0.000000e+00, [[FORNULL]] ]
// CHECK-NEXT: [[IMAG:%.*]] = phi float [ [[IMAGCALL]], [[INVOKE_CONT]] ], [ 0.000000e+00, [[FORNULL]] ]
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds { float, float }, { float, float }* [[RESULT]], i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds { float, float }, { float, float }* [[RESULT]], i32 0, i32 1
// CHECK-NEXT: store float [[REAL]], float* [[T0]]
// CHECK-NEXT: store float [[IMAG]], float* [[T1]]
//   Epilogue.
// CHECK-NEXT: call void @llvm.objc.destroyWeak(i8** [[WEAKOBJ]]) [[NUW]]
// CHECK-NEXT: call void @llvm.objc.storeStrong(i8** [[OBJ]], i8* null) [[NUW]]
// CHECK-NEXT: ret void
//   Cleanup.
// CHECK:      landingpad
// CHECK:      call void @llvm.objc.destroyWeak(i8** [[WEAKOBJ]]) [[NUW]]

void test2(id a) {
  id obj = [MyObject m0:a];
}

// CHECK-LABEL: define{{.*}} void @test2(
// CHECK: %[[CALL:.*]] = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend
// CHECK-NEXT: %[[V6:.*]] = {{.*}}call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %[[CALL]])

// CHECK: phi i8* [ %[[V6]], %{{.*}} ], [ null, %{{.*}} ]

void test3(id a) {
  @try {
    id obj = [MyObject m0:a];
  } @catch (id x) {
  }
}

// CHECK-LABEL: define{{.*}} void @test3(
// CHECK: %[[CALL:.*]] = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend
// CHECK: %[[V6:.*]] = {{.*}}call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %[[CALL]])

// CHECK: phi i8* [ %[[V6]], %{{.*}} ], [ null, %{{.*}} ]

// CHECK: attributes [[NUW]] = { nounwind }
