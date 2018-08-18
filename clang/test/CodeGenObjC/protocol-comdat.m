// RUN: %clang -cc1 -triple thumbv7--windows-itanium -fobjc-runtime=ios -emit-llvm -o - %s -Wno-objc-root-class | FileCheck %s

@protocol P
- (void) method;
@end

@protocol Q @end
@protocol R @end

@interface I<P>
@end

@implementation I
- (void) method { }
@end

_Bool f(void) {
  return @protocol(Q) == @protocol(R);
}

// CHECK: $"\01l_OBJC_PROTOCOL_$_P" = comdat any
// CHECK: $"\01l_OBJC_LABEL_PROTOCOL_$_P" = comdat any
// CHECK: $"\01l_OBJC_PROTOCOL_REFERENCE_$_Q" = comdat any
// CHECK: $"\01l_OBJC_PROTOCOL_REFERENCE_$_R" = comdat any

// CHECK: @"\01l_OBJC_PROTOCOL_$_P" = {{.*}}, comdat
// CHECK: @"\01l_OBJC_LABEL_PROTOCOL_$_P" = {{.*}}, comdat

