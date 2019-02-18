// RUN: %clang_analyze_cc1 -w -analyzer-checker=debug.DumpLiveStmts %s 2>&1\
// RUN:   | FileCheck %s

int coin();


int testThatDumperWorks(int x, int y, int z) {
  return x ? y : z;
}
// CHECK: [ B0 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B1 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B2 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-NEXT: DeclRefExpr {{.*}} 'y' 'int'
// CHECK-EMPTY:
// CHECK-NEXT: DeclRefExpr {{.*}} 'z' 'int'
// CHECK-EMPTY:
// CHECK-NEXT: ImplicitCastExpr {{.*}} <IntegralToBoolean>
// CHECK-NEXT: `-ImplicitCastExpr {{.*}} <LValueToRValue>
// CHECK-NEXT:   `-DeclRefExpr {{.*}} 'x' 'int'
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B3 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-NEXT: DeclRefExpr {{.*}} 'y' 'int'
// CHECK-EMPTY:
// CHECK-NEXT: DeclRefExpr {{.*}} 'z' 'int'
// CHECK-EMPTY:
// CHECK-NEXT: ImplicitCastExpr {{.*}} <IntegralToBoolean>
// CHECK-NEXT: `-ImplicitCastExpr {{.*}} <LValueToRValue>
// CHECK-NEXT:   `-DeclRefExpr {{.*}} 'x' 'int'
// CHECK: [ B4 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-NEXT: DeclRefExpr {{.*}} 'y' 'int'
// CHECK-EMPTY:
// CHECK-NEXT: DeclRefExpr {{.*}} 'z' 'int'
// CHECK-EMPTY:
// CHECK-NEXT: ImplicitCastExpr {{.*}} <IntegralToBoolean>
// CHECK-NEXT: `-ImplicitCastExpr {{.*}} <LValueToRValue>
// CHECK-NEXT:   `-DeclRefExpr {{.*}} 'x' 'int'
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B5 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-NEXT: DeclRefExpr {{.*}} 'y' 'int'
// CHECK-EMPTY:
// CHECK-NEXT: DeclRefExpr {{.*}} 'z' 'int'
// CHECK-EMPTY:
// CHECK-EMPTY:


void testIfBranchExpression(bool flag) {
  // No expressions should be carried over from one block to another here.
  while (flag) {
    int e = 1;
    if (true)
      e;
  }
}
// CHECK: [ B0 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B1 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B2 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B3 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B4 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B5 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:


void testWhileBodyExpression(bool flag) {
  // No expressions should be carried over from one block to another here.
  while (flag) {
    int e = 1;
    while (coin())
      e;
  }
}
// CHECK: [ B0 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B1 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B2 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B3 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B4 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B5 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:


void testDoWhileBodyExpression(bool flag) {
  // No expressions should be carried over from one block to another here.
  while (flag) {
    int e = 1;
    do
      e;
    while (coin());
  }
}
// CHECK: [ B0 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B1 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B2 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B3 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B4 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B5 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:


void testForBodyExpression(bool flag) {
  // No expressions should be carried over from one block to another here.
  while (flag) {
    int e = 1;
    for (; coin();)
      e;
  }
}
// CHECK: [ B0 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B1 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B2 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B3 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B4 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B5 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:

