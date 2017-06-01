// RUN: %clang -cc1 -triple thumbv7--windows-itanium -fobjc-runtime=ios -emit-llvm -o - %s -Wno-objc-root-class | FileCheck %s

@protocol P
- (void) method;
@end

@interface I<P>
@end

@implementation I
- (void) method { }
@end


// CHECK: $"\01l_OBJC_PROTOCOL_$_P" = comdat any
// CHECK: $"\01l_OBJC_LABEL_PROTOCOL_$_P" = comdat any

// CHECK: @"\01l_OBJC_PROTOCOL_$_P" = {{.*}}, comdat
// CHECK: @"\01l_OBJC_LABEL_PROTOCOL_$_P" = {{.*}}, comdat

