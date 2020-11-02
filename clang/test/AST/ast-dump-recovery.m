// RUN: not %clang_cc1 -triple x86_64-unknown-unknown -frecovery-ast -frecovery-ast-type -ast-dump %s | FileCheck -strict-whitespace %s

@interface Foo
- (void)method:(int)n;
@end

void k(Foo *foo) {
  // CHECK:       ObjCMessageExpr {{.*}} 'void' contains-errors
  // CHECK-CHECK:  |-ImplicitCastExpr {{.*}} 'Foo *' <LValueToRValue>
  // CHECK-CHECK:  | `-DeclRefExpr {{.*}} 'foo'
  // CHECK-CHECK:  `-RecoveryExpr {{.*}}
  [foo method:undef];

  // CHECK:      ImplicitCastExpr {{.*}} '<dependent type>' contains-errors
  // CHECK-NEXT: `-RecoveryExpr {{.*}} '<dependent type>' contains-errors
  // CHECK-NEXT:   `-DeclRefExpr {{.*}} 'foo'
  foo.undef;
}
