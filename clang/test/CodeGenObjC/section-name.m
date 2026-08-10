// RUN: %clang_cc1 -triple thumbv7--windows-itanium -fdeclspec -fobjc-runtime=ios -emit-llvm -o - %s -Wno-objc-root-class | FileCheck %s

@protocol Protocol
- (void) protocol_method;
@end

__declspec(dllexport)
@interface Interface<Protocol>
@property(assign) id property;
+ (void) class_method;
- (void) instance_method;
@end


@implementation Interface
+ (void) class_method {
}

- (void) protocol_method {
}

- (void) instance_method {
}
@end

@implementation Interface(Category)
- (void) category_method {
}
@end

// CHECK-NOT: @"OBJC_IVAR_$_Interface._property" = {{.*}} section "__DATA, __objc_ivar"
// CHECK-NOT: @"OBJC_CLASS_$_Interface" = {{.*}} section "__DATA, __objc_data"
// CHECK-NOT: @"OBJC_METACLASS_$_Interface" = {{.*}} section "__DATA, __objc_data"
// CHECK-NOT: @"_OBJC_$_CLASS_METHODS_Interface" = {{.*}} section "__DATA, __objc_const"
// CHECK-NOT: @"_OBJC_$_PROTOCOL_INSTANCE_METHODS_Protocol" = {{.*}} section "__DATA, __objc_const"
// CHECK-NOT: @"_OBJC_$_PROTOCOL_METHOD_TYPES_Protocol" = {{.*}} section "__DATA, __objc_const"
// CHECK-NOT: @"_OBJC_CLASS_PROTOCOLS_$_Interface" = {{.*}} section "__DATA, __objc_const"
// CHECK-NOT: @"_OBJC_METACLASS_RO_$_" = {{.*}} section "__DATA, __objc_const"
// CHECK-NOT: @"_OBJC_$_INSTANCE_METHODS_Interface" = {{.*}} section "__DATA, __objc_const"
// CHECK-NOT: @"_OBJC_$_INSTANCE_VARIABLES_Interface" = {{.*}} section "__DATA, __objc_const"
// CHECK-NOT: @"_OBJC_$_PROP_LIST_Interface" = {{.*}} section "__DATA, __objc_const"
// CHECK-NOT: @"_OBJC_CLASS_RO_$_Interface" = {{.*}} section "__DATA, __objc_const"
// CHECK-NOT: @"_OBJC_$_CATEGORY_INSTANCE_METHODS_Interface_$_Category" = {{.*}} section "__DATA, __objc_const"
// CHECK-NOT: @"_OBJC_$_CATEGORY_Interface_$_Category" = {{.*}} section "__DATA, __objc_const"

