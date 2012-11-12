// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-arc -fobjc-dispatch-method=mixed -o - %s | FileCheck %s
// rdar://10444476

@interface NSObject
- (id) new;
@end

@interface MyObject : NSObject
- (char)isEqual:(id) __attribute__((ns_consumed)) object;
@end

MyObject *x;

void foo()
{
        id obj = [NSObject new];
        [x isEqual : obj];
}

// CHECK: [[TMP:%.*]] = alloca i8{{$}}
// CHECK: [[FIVE:%.*]] = call i8* @objc_retain
// CHECK-NEXT:  [[SIX:%.*]] = bitcast
// CHECK-NEXT:  [[SEVEN:%.*]]  = icmp eq i8* [[SIX]], null
// CHECK-NEXT:  br i1 [[SEVEN]], label [[NULLINIT:%.*]], label [[CALL_LABEL:%.*]]
// CHECK:  [[FN:%.*]] = load i8** getelementptr inbounds
// CHECK-NEXT:  [[EIGHT:%.*]] = bitcast i8* [[FN]]
// CHECK-NEXT:  [[CALL:%.*]] = call signext i8 [[EIGHT]]
// CHECK-NEXT:  store i8 [[CALL]], i8* [[TMP]]
// CHECK-NEXT:  br label [[CONT:%.*]]
// CHECK:   call void @objc_release(i8* [[FIVE]]) nounwind
// CHECK-NEXT:   call void @llvm.memset
// CHECK-NEXT:  br label [[CONT]]
