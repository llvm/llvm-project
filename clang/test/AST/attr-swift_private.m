// RUN: %clang_cc1 -ast-dump %s | FileCheck %s

@interface I
- (void)method __attribute__((__swift_private__));
@end

// CHECK: ObjCInterfaceDecl {{.*}} I
// CHECK: ObjCMethodDecl {{.*}} method 'void'
// CHECK: SwiftPrivateAttr

@interface J : I
- (void)method;
@end

// CHECK: ObjCInterfaceDecl {{.*}} J
// CHECK: ObjCMethodDecl {{.*}} method 'void'
// CHECK: SwiftPrivateAttr {{.*}} Inherited

void f(void) __attribute__((__swift_private__));
// CHECK: FunctionDecl {{.*}} f 'void (void)'
// CHECK: SwiftPrivateAttr

void f(void) {
}
// CHECK: FunctionDecl {{.*}} f 'void (void)'
// CHECK: SwiftPrivateAttr {{.*}} Inherited
