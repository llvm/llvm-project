// RUN: %clang_cc1 -Wno-unused -fblocks -ast-dump -ast-dump-filter Test %s | FileCheck -strict-whitespace %s

@interface A
@end

@interface TestObjCImplementation : A
@end

@implementation TestObjCImplementation : A {
  struct X {
    int i;
  } X;
}
- (void) foo {
}
@end
// CHECK:      ObjCImplementationDecl{{.*}} TestObjCImplementation
// CHECK-NEXT:   super ObjCInterface{{.*}} 'A'
// CHECK-NEXT:   ObjCInterface{{.*}} 'TestObjCImplementation'
// CHECK-NEXT:   CXXCtorInitializer{{.*}} 'X'
// CHECK-NEXT:     CXXConstructExpr
// CHECK-NEXT:   ObjCIvarDecl{{.*}} X
// CHECK-NEXT:   ObjCMethodDecl{{.*}} foo

// @() boxing expressions.
template <typename T>
struct BoxingTest {
  static id box(T value) {
    return @(value);
  }
};

// CHECK: ObjCBoxedExpr{{.*}} '<dependent type>'{{$}}
