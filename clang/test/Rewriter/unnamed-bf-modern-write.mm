// RUN: %clang_cc1 -E %s -o %t.mm
// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %t.mm -o - | FileCheck %s 

@interface Foo {
@private
    int first;
    int :1;
    int third :1;
    int :1;
    int fifth :1;
}
@end
@implementation Foo 
@end

// CHECK: struct Foo__T_1 {
// CHECK-NEXT:         int : 1;
// CHECK-NEXT:         int third : 1;
// CHECK-NEXT:         int : 1;
// CHECK-NEXT:         int fifth : 1;
// CHECK-NEXT:         char : 0;
// CHECK-NEXT:         } ;
// CHECK: struct Foo_IMPL {
// CHECK-NEXT:         int first;
// CHECK-NEXT:         struct Foo__T_1 Foo__GRBF_1;
// CHECK-NEXT: };
