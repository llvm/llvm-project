// RUN: %clang_cc1 -triple i386-apple-macosx -fvisibility hidden -emit-llvm -o - %s | FileCheck %s
// CHECK: @"OBJC_IVAR_$_I.P" = hidden
// CHECK: @"OBJC_CLASS_$_I" = hidden
// CHECK: @"OBJC_METACLASS_$_I" = hidden
// CHECK: @"\01l_OBJC_PROTOCOL_$_Prot0" = weak hidden

@interface I {
  int P;
}

@property int P;
@end

@implementation I
@synthesize P;
@end


@protocol Prot0 @end

id f0() {
  return @protocol(Prot0);
}


