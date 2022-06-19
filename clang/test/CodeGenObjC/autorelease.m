// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin10 -emit-llvm -fobjc-arc -fobjc-runtime=macosx-10.7 -fexceptions -fobjc-exceptions -o - %s | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin11 -emit-llvm -fobjc-runtime=macosx-10.7 -fexceptions -fobjc-exceptions -o - %s | FileCheck %s
// rdar://8881826
// rdar://9412038

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

// CHECK: call i8* @llvm.objc.autoreleasePoolPush
// CHECK: [[T:%.*]] = load i8*, i8** [[A:%.*]]
// CHECK: call void @llvm.objc.autoreleasePoolPop

// rdar://13660038
int tryTo(int (*f)(void)) {
  @try {
    @autoreleasepool {
      return f();
    }
  } @catch (...) {
    return 0;
  }
}
// CHECK-LABEL:    define{{.*}} i32 @tryTo(i32 ()*
// CHECK:      [[RET:%.*]] = alloca i32,
// CHECK:      [[T0:%.*]] = call i8* @llvm.objc.autoreleasePoolPush()
// CHECK-NEXT: [[T1:%.*]] = load i32 ()*, i32 ()** {{%.*}},
// CHECK-NEXT: [[T2:%.*]] = invoke i32 [[T1]]()
// CHECK:      store i32 [[T2]], i32* [[RET]]
// CHECK:      invoke void @objc_autoreleasePoolPop(i8* [[T0]])
// CHECK:      landingpad { i8*, i32 }
// CHECK-NEXT:   catch i8* null
// CHECK:      call i8* @objc_begin_catch
// CHECK-NEXT: store i32 0, i32* [[RET]]
// CHECK:      call void @objc_end_catch()
