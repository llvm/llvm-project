// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c++ -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c++ -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

// CHECK-NOT: BoundsCheckExpr
// CHECK-NOT: BoundsSafetyPointerPromotionExpr
// CHECK-NOT: MaterializeSequenceExpr

// CHECK: FunctionDecl {{.+}} foo 'void (int * __ended_by(end), int * /* __started_by(p) */ )'
void foo(int *__ended_by(end) p, int *end) {
  // CHECK:      UnaryOperator {{.+}} 'int' lvalue prefix '*' cannot overflow
  // CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'int * __ended_by(end)':'int *' <LValueToRValue>
  // CHECK-NEXT:   `-DeclRefExpr {{.+}} 'int * __ended_by(end)':'int *' lvalue ParmVar {{.+}} 'p' 'int * __ended_by(end)':'int *'
  (void)*p;

  // CHECK:      ArraySubscriptExpr {{.+}} 'int' lvalue
  // CHECK-NEXT: |-ImplicitCastExpr {{.+}} 'int * __ended_by(end)':'int *' <LValueToRValue>
  // CHECK-NEXT: | `-DeclRefExpr {{.+}} 'int * __ended_by(end)':'int *' lvalue ParmVar {{.+}} 'p' 'int * __ended_by(end)':'int *'
  // CHECK-NEXT: `-IntegerLiteral {{.+}} 'int' 42
  (void)p[42];

  // CHECK:      BinaryOperator {{.+}} 'int * __ended_by(end)':'int *'{{.*}} '='
  // CHECK-NEXT: |-DeclRefExpr {{.+}} 'int * __ended_by(end)':'int *' lvalue ParmVar {{.+}} 'p' 'int * __ended_by(end)':'int *'
  // CHECK-NEXT: `-BinaryOperator {{.+}} 'int * __ended_by(end)':'int *' '+'
  // CHECK-NEXT:   |-ImplicitCastExpr {{.+}} 'int * __ended_by(end)':'int *' <LValueToRValue>
  // CHECK-NEXT:   | `-DeclRefExpr {{.+}} 'int * __ended_by(end)':'int *' lvalue ParmVar {{.+}} 'p' 'int * __ended_by(end)':'int *'
  // CHECK-NEXT:   `-IntegerLiteral {{.+}} 'int' 42
  // CHECK-NEXT: BinaryOperator {{.+}} 'int * /* __started_by(p) */ ':'int *'{{.*}} '='
  // CHECK-NEXT: |-DeclRefExpr {{.+}} 'int * /* __started_by(p) */ ':'int *' lvalue ParmVar {{.+}} 'end' 'int * /* __started_by(p) */ ':'int *'
  // CHECK-NEXT: `-BinaryOperator {{.+}} 'int * /* __started_by(p) */ ':'int *' '-'
  // CHECK-NEXT:   |-ImplicitCastExpr {{.+}} 'int * /* __started_by(p) */ ':'int *' <LValueToRValue>
  // CHECK-NEXT:   | `-DeclRefExpr {{.+}} 'int * /* __started_by(p) */ ':'int *' lvalue ParmVar {{.+}} 'end' 'int * /* __started_by(p) */ ':'int *'
  // CHECK-NEXT:   `-IntegerLiteral {{.+}} 'int' 42
  p = p + 42;
  end = end - 42;
}

// CHECK:      RecordDecl {{.+}} struct bar definition
// CHECK:      |-FieldDecl {{.+}} q 'int * __ended_by(end)':'int *'
// CHECK-NEXT: `-FieldDecl {{.+}} end 'int * /* __started_by(q) */ ':'int *'
struct bar {
  int *__ended_by(end) q;
  int *end;
};

// CHECK: FunctionDecl {{.+}} baz 'void (struct bar *)'
void baz(struct bar *b) {
  // CHECK:      UnaryOperator {{.+}} 'int' lvalue prefix '*' cannot overflow
  // CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'int * __ended_by(end)':'int *' <LValueToRValue>
  // CHECK-NEXT:   `-MemberExpr {{.+}} 'int * __ended_by(end)':'int *' lvalue ->q
  // CHECK-NEXT:     `-ImplicitCastExpr {{.+}} 'struct bar *' <LValueToRValue>
  // CHECK-NEXT:       `-DeclRefExpr {{.+}} 'struct bar *' lvalue ParmVar {{.+}} 'b' 'struct bar *'
  (void)*b->q;

  // CHECK:      ArraySubscriptExpr {{.+}} 'int' lvalue
  // CHECK-NEXT: |-ImplicitCastExpr {{.+}} 'int * __ended_by(end)':'int *' <LValueToRValue>
  // CHECK-NEXT: | `-MemberExpr {{.+}} 'int * __ended_by(end)':'int *' lvalue ->q
  // CHECK-NEXT: |   `-ImplicitCastExpr {{.+}} 'struct bar *' <LValueToRValue>
  // CHECK-NEXT: |     `-DeclRefExpr {{.+}} 'struct bar *' lvalue ParmVar {{.+}} 'b' 'struct bar *'
  // CHECK-NEXT: `-IntegerLiteral {{.+}} 'int' 42
  (void)b->q[42];

  // CHECK:      BinaryOperator {{.+}} 'int * __ended_by(end)':'int *'{{.*}} '='
  // CHECK-NEXT: |-MemberExpr {{.+}} 'int * __ended_by(end)':'int *' lvalue ->q
  // CHECK-NEXT: | `-ImplicitCastExpr {{.+}} 'struct bar *' <LValueToRValue>
  // CHECK-NEXT: |   `-DeclRefExpr {{.+}} 'struct bar *' lvalue ParmVar {{.+}} 'b' 'struct bar *'
  // CHECK-NEXT: `-BinaryOperator {{.+}} 'int * __ended_by(end)':'int *' '+'
  // CHECK-NEXT:   |-ImplicitCastExpr {{.+}} 'int * __ended_by(end)':'int *' <LValueToRValue>
  // CHECK-NEXT:   | `-MemberExpr {{.+}} 'int * __ended_by(end)':'int *' lvalue ->q
  // CHECK-NEXT:   |   `-ImplicitCastExpr {{.+}} 'struct bar *' <LValueToRValue>
  // CHECK-NEXT:   |     `-DeclRefExpr {{.+}} 'struct bar *' lvalue ParmVar {{.+}} 'b' 'struct bar *'
  // CHECK-NEXT:   `-IntegerLiteral {{.+}} 'int' 42
  // CHECK-NEXT: BinaryOperator {{.+}} 'int * /* __started_by(q) */ ':'int *'{{.*}} '='
  // CHECK-NEXT: |-MemberExpr {{.+}} 'int * /* __started_by(q) */ ':'int *' lvalue ->end
  // CHECK-NEXT: | `-ImplicitCastExpr {{.+}} 'struct bar *' <LValueToRValue>
  // CHECK-NEXT: |   `-DeclRefExpr {{.+}} 'struct bar *' lvalue ParmVar {{.+}} 'b' 'struct bar *'
  // CHECK-NEXT: `-BinaryOperator {{.+}} 'int * /* __started_by(q) */ ':'int *' '-'
  // CHECK-NEXT:   |-ImplicitCastExpr {{.+}} 'int * /* __started_by(q) */ ':'int *' <LValueToRValue>
  // CHECK-NEXT:   | `-MemberExpr {{.+}} 'int * /* __started_by(q) */ ':'int *' lvalue ->end
  // CHECK-NEXT:   |   `-ImplicitCastExpr {{.+}} 'struct bar *' <LValueToRValue>
  // CHECK-NEXT:   |     `-DeclRefExpr {{.+}} 'struct bar *' lvalue ParmVar {{.+}} 'b' 'struct bar *'
  // CHECK-NEXT:   `-IntegerLiteral {{.+}} 'int' 42
  b->q = b->q + 42;
  b->end = b->end - 42;
}
