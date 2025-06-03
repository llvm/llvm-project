// RUN: not %clang_cc1 -triple x86_64-unknown-unknown -frecovery-ast -frecovery-ast-type -fblocks -ast-dump %s | FileCheck -strict-whitespace %s

@interface Foo
- (void)method:(int)n;
@end

void k(Foo *foo) {
  // CHECK:       ObjCMessageExpr {{.*}} 'void' contains-errors
  // CHECK-NEXT:  |-ImplicitCastExpr {{.*}} 'Foo *' <LValueToRValue>
  // CHECK-NEXT:  | `-DeclRefExpr {{.*}} 'foo'
  // CHECK-NEXT:  `-RecoveryExpr {{.*}}
  [foo method:undef];

  // CHECK:      ImplicitCastExpr {{.*}} '<dependent type>' contains-errors
  // CHECK-NEXT: `-RecoveryExpr {{.*}} '<dependent type>' contains-errors
  // CHECK-NEXT:   `-DeclRefExpr {{.*}} 'foo'
  foo.undef;
}

// CHECK:      |-VarDecl {{.*}} 'int (^)()' cinit
// CHECK-NEXT: | `-RecoveryExpr {{.*}} '<dependent type> (^)(void)' contains-errors lvalue
// CHECK-NEXT: |   `-BlockExpr {{.*}} '<dependent type> (^)(void)'
// CHECK-NEXT: |     `-BlockDecl {{.*}} invalid
int (^gh63863)() = ^() {
  return undef;
};

// CHECK:      `-BlockExpr {{.*}} 'int (^)(int, int)'
// CHECK-NEXT:   `-BlockDecl {{.*}} invalid
int (^gh64005)(int, int) = ^(int, undefined b) {
   return 1;
};
