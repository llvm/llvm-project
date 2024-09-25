// RUN: %clang_analyze_cc1 -w -analyzer-checker=debug.DumpLiveExprs %s 2>&1\
// RUN:   | FileCheck %s

int coin();


int testThatDumperWorks(int x, int y, int z) {
  return x ? y : z;
}

// [B5 (ENTRY)]
//    |
//    V
// [B4 (x)] ? [B2 (y)] : [B3 (z)]
//                \        /
//                 ---|----
//                    V
//                   [B1] --> [B0 (EXIT)]
//                  return

// CHECK: [ B0 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B1 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B2 (live expressions at block exit) ]
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
// CHECK: [ B3 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-NEXT: DeclRefExpr {{.*}} 'y' 'int'
// CHECK-EMPTY:
// CHECK-NEXT: DeclRefExpr {{.*}} 'z' 'int'
// CHECK-EMPTY:
// CHECK-NEXT: ImplicitCastExpr {{.*}} <IntegralToBoolean>
// CHECK-NEXT: `-ImplicitCastExpr {{.*}} <LValueToRValue>
// CHECK-NEXT:   `-DeclRefExpr {{.*}} 'x' 'int'
// CHECK: [ B4 (live expressions at block exit) ]
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
// CHECK: [ B5 (live expressions at block exit) ]
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
// CHECK: [ B0 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B1 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B2 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B3 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B4 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B5 (live expressions at block exit) ]
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
// CHECK: [ B0 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B1 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B2 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B3 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B4 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B5 (live expressions at block exit) ]
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
// CHECK: [ B0 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B1 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B2 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B3 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B4 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B5 (live expressions at block exit) ]
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
// CHECK: [ B0 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B1 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B2 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B3 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B4 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B5 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:

void clang_analyzer_eval(bool);

void test_lambda_refcapture() {
  int a = 6;
  [&](int &a) { a = 42; }(a);
  clang_analyzer_eval(a == 42); // expected-warning{{TRUE}}
}

// CHECK: [ B0 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT: [ B1 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT: [ B2 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:

int logicalOpInTernary(bool b) {
  return (b || b) ? 0 : 1;
}

// [B6 (ENTRY)]
//    |
//    V
// [B5 (b || ...)]
//   |            \
//   |             |
//   V             V
// [B4 (b||b)] ? [B2 (0)] : [B3 (1)]
//                \        /
//                 ---|----
//                    V
//                   [B1] --> [B0 (EXIT)]
//                  return

// CHECK: [ B0 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B1 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B2 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK: ImplicitCastExpr {{.*}} '_Bool' <LValueToRValue>
// CHECK: `-DeclRefExpr {{.*}} '_Bool' lvalue ParmVar {{.*}} 'b' '_Bool'
// CHECK-EMPTY:
// CHECK: ImplicitCastExpr {{.*}} '_Bool' <LValueToRValue>
// CHECK: `-DeclRefExpr {{.*}} '_Bool' lvalue ParmVar {{.*}} 'b' '_Bool'
// CHECK-EMPTY:
// CHECK: BinaryOperator {{.*}} '_Bool' '||'
// CHECK: |-ImplicitCastExpr {{.*}} '_Bool' <LValueToRValue>
// CHECK: | `-DeclRefExpr {{.*}} '_Bool' lvalue ParmVar {{.*}} 'b' '_Bool'
// CHECK: `-ImplicitCastExpr {{.*}} '_Bool' <LValueToRValue>
// CHECK:   `-DeclRefExpr {{.*}} '_Bool' lvalue ParmVar {{.*}} 'b' '_Bool'
// CHECK-EMPTY:
// CHECK: IntegerLiteral {{.*}} 'int' 0
// CHECK-EMPTY:
// CHECK: IntegerLiteral {{.*}} 'int' 1
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B3 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK: ImplicitCastExpr {{.*}} '_Bool' <LValueToRValue>
// CHECK: `-DeclRefExpr {{.*}} '_Bool' lvalue ParmVar {{.*}} 'b' '_Bool'
// CHECK-EMPTY:
// CHECK: ImplicitCastExpr {{.*}} '_Bool' <LValueToRValue>
// CHECK: `-DeclRefExpr {{.*}} '_Bool' lvalue ParmVar {{.*}} 'b' '_Bool'
// CHECK-EMPTY:
// CHECK: BinaryOperator {{.*}} '_Bool' '||'
// CHECK: |-ImplicitCastExpr {{.*}} '_Bool' <LValueToRValue>
// CHECK: | `-DeclRefExpr {{.*}} '_Bool' lvalue ParmVar {{.*}} 'b' '_Bool'
// CHECK: `-ImplicitCastExpr {{.*}} '_Bool' <LValueToRValue>
// CHECK:   `-DeclRefExpr {{.*}} '_Bool' lvalue ParmVar {{.*}} 'b' '_Bool'
// CHECK-EMPTY:
// CHECK: IntegerLiteral {{.*}} 'int' 0
// CHECK-EMPTY:
// CHECK: IntegerLiteral {{.*}} 'int' 1
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B4 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK: ImplicitCastExpr {{.*}} '_Bool' <LValueToRValue>
// CHECK: `-DeclRefExpr {{.*}} '_Bool' lvalue ParmVar {{.*}} 'b' '_Bool'
// CHECK-EMPTY:
// CHECK: ImplicitCastExpr {{.*}} '_Bool' <LValueToRValue>
// CHECK: `-DeclRefExpr {{.*}} '_Bool' lvalue ParmVar {{.*}} 'b' '_Bool'
// CHECK-EMPTY:
// CHECK: BinaryOperator {{.*}} '_Bool' '||'
// CHECK: |-ImplicitCastExpr {{.*}} '_Bool' <LValueToRValue>
// CHECK: | `-DeclRefExpr {{.*}} '_Bool' lvalue ParmVar {{.*}} 'b' '_Bool'
// CHECK: `-ImplicitCastExpr {{.*}} '_Bool' <LValueToRValue>
// CHECK:   `-DeclRefExpr {{.*}} '_Bool' lvalue ParmVar {{.*}} 'b' '_Bool'
// CHECK-EMPTY:
// CHECK: IntegerLiteral {{.*}} 'int' 0
// CHECK-EMPTY:
// CHECK: IntegerLiteral {{.*}} 'int' 1
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B5 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK: ImplicitCastExpr {{.*}} '_Bool' <LValueToRValue>
// CHECK: `-DeclRefExpr {{.*}} '_Bool' lvalue ParmVar {{.*}} 'b' '_Bool'
// CHECK-EMPTY:
// CHECK: ImplicitCastExpr {{.*}} '_Bool' <LValueToRValue>
// CHECK: `-DeclRefExpr {{.*}} '_Bool' lvalue ParmVar {{.*}} 'b' '_Bool'
// CHECK-EMPTY:
// CHECK: BinaryOperator {{.*}} '_Bool' '||'
// CHECK: |-ImplicitCastExpr {{.*}} '_Bool' <LValueToRValue>
// CHECK: | `-DeclRefExpr {{.*}} '_Bool' lvalue ParmVar {{.*}} 'b' '_Bool'
// CHECK: `-ImplicitCastExpr {{.*}} '_Bool' <LValueToRValue>
// CHECK:   `-DeclRefExpr {{.*}} '_Bool' lvalue ParmVar {{.*}} 'b' '_Bool'
// CHECK-EMPTY:
// CHECK: IntegerLiteral {{.*}} 'int' 0
// CHECK-EMPTY:
// CHECK: IntegerLiteral {{.*}} 'int' 1
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK: [ B6 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK: ImplicitCastExpr {{.*}} '_Bool' <LValueToRValue>
// CHECK: `-DeclRefExpr {{.*}} '_Bool' lvalue ParmVar {{.*}} 'b' '_Bool'
// CHECK-EMPTY:
// CHECK: IntegerLiteral {{.*}} 'int' 0
// CHECK-EMPTY:
// CHECK: IntegerLiteral {{.*}} 'int' 1
