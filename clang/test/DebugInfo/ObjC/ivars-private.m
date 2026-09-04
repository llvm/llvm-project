// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s

// Debug symbols for private ivars. This test ensures that we are
// generating debug info for ivars added by the implementation.
__attribute((objc_root_class)) @interface NSObject {
  id isa;
}
@end

@protocol Protocol
@end

@interface Delegate : NSObject<Protocol> {
  @protected int foo;
}
@end

@interface Delegate(NSObject)
- (void)f;
@end

@implementation Delegate(NSObject)
- (void)f { return; }
@end

@implementation Delegate {
  int bar;
}

- (void)g:(NSObject*) anObject {
  bar = foo;
}
@end

// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "foo"
// CHECK-SAME:           line: 14
// CHECK-SAME:           baseType: ![[INT:[0-9]+]]
// CHECK-SAME:           size: 32,
// CHECK-NOT:            offset:
// CHECK-SAME:           flags: DIFlagProtected
// CHECK: ![[INT]] = !DIBasicType(name: "int"
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "bar"
// CHECK-SAME:           line: 27
// CHECK-SAME:           baseType: ![[INT:[0-9]+]]
// CHECK-SAME:           size: 32,
// CHECK-NOT:            offset:
// CHECK-SAME:           flags: DIFlagPrivate
