// RUN: %clang_cc1 -triple armv7-apple-darwin10 -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-ARM %s
// rdar://7761305

@interface I
@property long long LONG_PROP;
@end

@implementation I
@synthesize LONG_PROP;
@end
// CHECK-ARM: call void @objc_copyStruct(ptr noundef %{{.*}}, ptr noundef %{{.*}}, i32 noundef 8, i1 noundef zeroext true, i1 noundef zeroext false)
// CHECK-ARM: call void @objc_copyStruct(ptr noundef %{{.*}}, ptr noundef %{{.*}}, i32 noundef 8, i1 noundef zeroext true, i1 noundef zeroext false)

