// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-arc -fobjc-runtime=macosx-10.7 -fexceptions -fobjc-exceptions -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -emit-llvm -fobjc-runtime=macosx-10.7 -fexceptions -fobjc-exceptions -o - %s | FileCheck %s

@interface I
{
  id ivar;
}
- (id) Meth;
+ (id) MyAlloc;;
@end

@implementation I
- (id) Meth {
   @autoreleasepool {
      id p = [I MyAlloc];
      if (!p)
        return ivar;
   }
  return 0;
}
+ (id) MyAlloc {
    return 0;
}
@end

// CHECK: call ptr @llvm.objc.autoreleasePoolPush
// CHECK: [[T:%.*]] = load ptr, ptr [[A:%.*]]
// CHECK: call void @llvm.objc.autoreleasePoolPop

int tryTo(int (*f)(void)) {
  @try {
    @autoreleasepool {
      return f();
    }
  } @catch (...) {
    return 0;
  }
}
// CHECK-LABEL:    define{{.*}} i32 @tryTo(ptr
// CHECK:      [[RET:%.*]] = alloca i32,
// CHECK:      [[T0:%.*]] = call ptr @llvm.objc.autoreleasePoolPush()
// CHECK-NEXT: [[T1:%.*]] = load ptr, ptr {{%.*}},
// CHECK-NEXT: [[T2:%.*]] = invoke i32 [[T1]]()
// CHECK:      store i32 [[T2]], ptr [[RET]]
// CHECK:      invoke void @objc_autoreleasePoolPop(ptr [[T0]])
// CHECK:      landingpad { ptr, i32 }
// CHECK-NEXT:   catch ptr null
// CHECK:      call ptr @objc_begin_catch
// CHECK-NEXT: store i32 0, ptr [[RET]]
// CHECK:      call void @objc_end_catch()
