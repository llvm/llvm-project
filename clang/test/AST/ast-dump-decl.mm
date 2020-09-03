// Test without serialization:
// RUN: %clang_cc1 -Wno-unused -fblocks -ast-dump -ast-dump-filter Test %s \
// RUN: | FileCheck --strict-whitespace %s
//
// Test with serialization:
// RUN: %clang_cc1 -Wno-unused -fblocks -emit-pch -o %t %s
// RUN: %clang_cc1 -x objective-c++ -Wno-unused -fblocks -include-pch %t \
// RUN: -ast-dump-all -ast-dump-filter Test /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --strict-whitespace %s

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

struct Test {
  void f() {
    ^{ this->yada(); }();
    // CHECK:      ExprWithCleanups {{.*}} <line:[[@LINE-1]]:5, col:24> 'void'
    // CHECK-NEXT:   cleanup Block
    // CHECK-NEXT:   CallExpr {{.*}} <col:5, col:24> 'void'
    // CHECK-NEXT:     BlockExpr {{.*}} <col:5, col:22> 'void (^)()'
    // CHECK-NEXT:       BlockDecl {{.*}} <col:5, col:22> col:5 captures_this
    // CHECK-NEXT:         CompoundStmt {{.*}} <col:6, col:22>
    // CHECK-NEXT:           CXXMemberCallExpr {{.*}} <col:8, col:19> 'void'
    // CHECK-NEXT:             MemberExpr {{.*}} <col:8, col:14> '<bound member function type>' ->yada
    // CHECK-NEXT:               CXXThisExpr {{.*}} <col:8> 'Test *' this
  }
  void yada();
};
