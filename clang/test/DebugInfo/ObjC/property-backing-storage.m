// Verifies that, in addition to the legacy DIObjCProperty node, Clang also
// emits a new DIProperty node whose backing_storage points at the ivar a
// synthesized property forwards to. Covers the three ways a property can be
// backed by an ivar:
//   1. declaredBacking   - @synthesize with a custom ivar name that IS
//                           declared in the @interface.
//   2. undeclaredBacking - @synthesize with a custom ivar name that is NOT
//                           declared; the compiler creates the ivar itself.
//   3. implicitBacking   - no @synthesize at all; the compiler
//                           auto-synthesizes both the accessors and a
//                           default-named ivar (_implicitBacking).

// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s

// CHECK-DAG: ![[DECLARED_PROP:[0-9]+]] = !DIObjCProperty(name: "declaredBacking", file: ![[FILE:[0-9]+]], line: {{[0-9]+}}, attributes: 2316, type: ![[INT_TY:[0-9]+]])
// CHECK-DAG: ![[UNDECLARED_PROP:[0-9]+]] = !DIObjCProperty(name: "undeclaredBacking", file: ![[FILE]], line: {{[0-9]+}}, attributes: 2316, type: ![[INT_TY]])
// CHECK-DAG: ![[IMPLICIT_PROP:[0-9]+]] = !DIObjCProperty(name: "implicitBacking", file: ![[FILE]], line: {{[0-9]+}}, attributes: 2316, type: ![[INT_TY]])
//
// CHECK-DAG: ![[DECLARED_IVAR:[0-9]+]] = !DIDerivedType(tag: DW_TAG_member, name: "_customDeclaredIvar", {{.*}}file: ![[FILE]], {{.*}}extraData: ![[DECLARED_PROP]])
// CHECK-DAG: !DIProperty(name: "declaredBacking", file: ![[FILE]], line: {{[0-9]+}}, type: ![[INT_TY]], backing_storage: ![[DECLARED_IVAR]])
//
// CHECK-DAG: ![[UNDECLARED_IVAR:[0-9]+]] = !DIDerivedType(tag: DW_TAG_member, name: "_customUndeclaredIvar", {{.*}}file: ![[FILE]], {{.*}}extraData: ![[UNDECLARED_PROP]])
// CHECK-DAG: !DIProperty(name: "undeclaredBacking", file: ![[FILE]], line: {{[0-9]+}}, type: ![[INT_TY]], backing_storage: ![[UNDECLARED_IVAR]])
//
// CHECK-DAG: ![[IMPLICIT_IVAR:[0-9]+]] = !DIDerivedType(tag: DW_TAG_member, name: "_implicitBacking", {{.*}}file: ![[FILE]], {{.*}}extraData: ![[IMPLICIT_PROP]])
// CHECK-DAG: !DIProperty(name: "implicitBacking", file: ![[FILE]], line: {{[0-9]+}}, type: ![[INT_TY]], backing_storage: ![[IMPLICIT_IVAR]])

@interface C {
  int _customDeclaredIvar;
}
@property int declaredBacking;
@property int undeclaredBacking;
@property int implicitBacking;
@end

@implementation C
@synthesize declaredBacking = _customDeclaredIvar;
@synthesize undeclaredBacking = _customUndeclaredIvar;
@end

void foo(C *cptr) {}
