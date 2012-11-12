// RUN: %clang_cc1 -triple armv7-apple-darwin10 -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-ARM %s
// rdar://7761305

@interface I
@property long long LONG_PROP;
@end

@implementation I
@synthesize LONG_PROP;
@end
// CHECK-ARM: call arm_aapcscc  void @objc_copyStruct(i8* %{{.*}}, i8* %{{.*}}, i32 8, i1 zeroext true, i1 zeroext false)
// CHECK-ARM: call arm_aapcscc  void @objc_copyStruct(i8* %{{.*}}, i8* %{{.*}}, i32 8, i1 zeroext true, i1 zeroext false)

