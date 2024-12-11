

// RUN: %clang_cc1 -triple arm64-apple-iphoneos -fexperimental-bounds-safety-attributes -x c -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-iphoneos -fexperimental-bounds-safety-attributes -x c++ -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-iphoneos -fexperimental-bounds-safety-attributes -x objective-c -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-iphoneos -fexperimental-bounds-safety-attributes -x objective-c++ -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

// CHECK-NOT: BoundsCheckExpr
// CHECK-NOT: BoundsSafetyPointerPromotionExpr
// CHECK-NOT: MaterializeSequenceExpr

// CHECK: FunctionDecl {{.+}} foo 'void (int * __counted_by(count), int)'
void foo(int *__counted_by(count) p, int count) {
  // CHECK:      UnaryOperator {{.+}} 'int' lvalue prefix '*' cannot overflow
  // CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'int * __counted_by(count)':'int *' <LValueToRValue>
  // CHECK-NEXT:   `-DeclRefExpr {{.+}} 'int * __counted_by(count)':'int *' lvalue ParmVar {{.+}} 'p' 'int * __counted_by(count)':'int *'
  (void)*p;

  // CHECK:      ArraySubscriptExpr {{.+}} 'int' lvalue
  // CHECK-NEXT: |-ImplicitCastExpr {{.+}} 'int * __counted_by(count)':'int *' <LValueToRValue>
  // CHECK-NEXT: | `-DeclRefExpr {{.+}} 'int * __counted_by(count)':'int *' lvalue ParmVar {{.+}} 'p' 'int * __counted_by(count)':'int *'
  // CHECK-NEXT: `-IntegerLiteral {{.+}} 'int' 42
  (void)p[42];

  // CHECK:      BinaryOperator {{.+}} 'int * __counted_by(count)':'int *'{{.*}} '='
  // CHECK-NEXT: |-DeclRefExpr {{.+}} 'int * __counted_by(count)':'int *' lvalue ParmVar {{.+}} 'p' 'int * __counted_by(count)':'int *'
  // CHECK-NEXT: `-BinaryOperator {{.+}} 'int * __counted_by(count)':'int *' '+'
  // CHECK-NEXT:   |-ImplicitCastExpr {{.+}} 'int * __counted_by(count)':'int *' <LValueToRValue>
  // CHECK-NEXT:   | `-DeclRefExpr {{.+}} 'int * __counted_by(count)':'int *' lvalue ParmVar {{.+}} 'p' 'int * __counted_by(count)':'int *'
  // CHECK-NEXT:   `-IntegerLiteral {{.+}} 'int' 42
  // CHECK-NEXT: BinaryOperator {{.+}} 'int'{{.*}} '='
  // CHECK-NEXT: |-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
  // CHECK-NEXT: `-BinaryOperator {{.+}} 'int' '-'
  // CHECK-NEXT:   |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
  // CHECK-NEXT:   | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
  // CHECK-NEXT:   `-IntegerLiteral {{.+}} 'int' 42
  p = p + 42;
  count = count - 42;
}

// CHECK:      RecordDecl {{.+}} struct bar definition
// CHECK:      |-FieldDecl {{.+}} q 'int * __sized_by(size)':'int *'
// CHECK-NEXT: `-FieldDecl {{.+}} referenced size 'int'
struct bar {
  int *__sized_by(size) q;
  int size;
};

// CHECK: FunctionDecl {{.+}} baz 'void (struct bar *)'
void baz(struct bar *b) {
  // CHECK:      UnaryOperator {{.+}} 'int' lvalue prefix '*' cannot overflow
  // CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'int * __sized_by(size)':'int *' <LValueToRValue>
  // CHECK-NEXT:   `-MemberExpr {{.+}} 'int * __sized_by(size)':'int *' lvalue ->q
  // CHECK-NEXT:     `-ImplicitCastExpr {{.+}} 'struct bar *' <LValueToRValue>
  // CHECK-NEXT:       `-DeclRefExpr {{.+}} 'struct bar *' lvalue ParmVar {{.+}} 'b' 'struct bar *'
  (void)*b->q;

  // CHECK:      ArraySubscriptExpr {{.+}} 'int' lvalue
  // CHECK-NEXT: |-ImplicitCastExpr {{.+}} 'int * __sized_by(size)':'int *' <LValueToRValue>
  // CHECK-NEXT: | `-MemberExpr {{.+}} 'int * __sized_by(size)':'int *' lvalue ->q
  // CHECK-NEXT: |   `-ImplicitCastExpr {{.+}} 'struct bar *' <LValueToRValue>
  // CHECK-NEXT: |     `-DeclRefExpr {{.+}} 'struct bar *' lvalue ParmVar {{.+}} 'b' 'struct bar *'
  // CHECK-NEXT: `-IntegerLiteral {{.+}} 'int' 42
  (void)b->q[42];

  // CHECK:      BinaryOperator {{.+}} 'int * __sized_by(size)':'int *'{{.*}} '='
  // CHECK-NEXT: |-MemberExpr {{.+}} 'int * __sized_by(size)':'int *' lvalue ->q
  // CHECK-NEXT: | `-ImplicitCastExpr {{.+}} 'struct bar *' <LValueToRValue>
  // CHECK-NEXT: |   `-DeclRefExpr {{.+}} 'struct bar *' lvalue ParmVar {{.+}} 'b' 'struct bar *'
  // CHECK-NEXT: `-BinaryOperator {{.+}} 'int * __sized_by(size)':'int *' '+'
  // CHECK-NEXT:   |-ImplicitCastExpr {{.+}} 'int * __sized_by(size)':'int *' <LValueToRValue>
  // CHECK-NEXT:   | `-MemberExpr {{.+}} 'int * __sized_by(size)':'int *' lvalue ->q
  // CHECK-NEXT:   |   `-ImplicitCastExpr {{.+}} 'struct bar *' <LValueToRValue>
  // CHECK-NEXT:   |     `-DeclRefExpr {{.+}} 'struct bar *' lvalue ParmVar {{.+}} 'b' 'struct bar *'
  // CHECK-NEXT:   `-IntegerLiteral {{.+}} 'int' 42
  // CHECK-NEXT: BinaryOperator {{.+}} 'int'{{.*}} '='
  // CHECK-NEXT: |-MemberExpr {{.+}} 'int' lvalue ->size
  // CHECK-NEXT: | `-ImplicitCastExpr {{.+}} 'struct bar *' <LValueToRValue>
  // CHECK-NEXT: |   `-DeclRefExpr {{.+}} 'struct bar *' lvalue ParmVar {{.+}} 'b' 'struct bar *'
  // CHECK-NEXT: `-BinaryOperator {{.+}} 'int' '-'
  // CHECK-NEXT:   |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
  // CHECK-NEXT:   | `-MemberExpr {{.+}} 'int' lvalue ->size
  // CHECK-NEXT:   |   `-ImplicitCastExpr {{.+}} 'struct bar *' <LValueToRValue>
  // CHECK-NEXT:   |     `-DeclRefExpr {{.+}} 'struct bar *' lvalue ParmVar {{.+}} 'b' 'struct bar *'
  // CHECK-NEXT:   `-IntegerLiteral {{.+}} 'int' 42
  b->q = b->q + 42;
  b->size = b->size - 42;
}
