

// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

// Unary operator should NOT convert __indexable to __bidi_indexable.
void unop(int *__indexable p) {
  // CHECK:      UnaryOperator {{.+}} 'int *__indexable' postfix '++'
  // CHECK-NEXT: `-DeclRefExpr {{.+}} 'int *__indexable' lvalue ParmVar {{.+}} 'p' 'int *__indexable'
  p++;

  // CHECK:      UnaryOperator {{.+}} 'int *__indexable' prefix '++'
  // CHECK-NEXT: `-DeclRefExpr {{.+}} 'int *__indexable' lvalue ParmVar {{.+}} 'p' 'int *__indexable'
  ++p;
}

// Binary operator should convert __indexable to __bidi_indexable.
void binop(int *__indexable p, int index) {
  // CHECK:      BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
  // CHECK-NEXT: |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
  // CHECK-NEXT: | `-ImplicitCastExpr {{.+}} 'int *__indexable' <LValueToRValue>
  // CHECK-NEXT: |   `-DeclRefExpr {{.+}} 'int *__indexable' lvalue ParmVar {{.+}} 'p' 'int *__indexable'
  // CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
  // CHECK-NEXT:   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'index' 'int'
  (void)(p + index);

  // CHECK:      BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
  // CHECK-NEXT: |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
  // CHECK-NEXT: | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'index' 'int'
  // CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
  // CHECK-NEXT:   `-ImplicitCastExpr {{.+}} 'int *__indexable' <LValueToRValue>
  // CHECK-NEXT:     `-DeclRefExpr {{.+}} 'int *__indexable' lvalue ParmVar {{.+}} 'p' 'int *__indexable'
  (void)(index + p);

  // CHECK:      BinaryOperator {{.+}} 'int *__bidi_indexable' '-'
  // CHECK-NEXT: |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
  // CHECK-NEXT: | `-ImplicitCastExpr {{.+}} 'int *__indexable' <LValueToRValue>
  // CHECK-NEXT: |   `-DeclRefExpr {{.+}} 'int *__indexable' lvalue ParmVar {{.+}} 'p' 'int *__indexable'
  // CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
  // CHECK-NEXT:   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'index' 'int'
  (void)(p - index);

  // CHECK:      BinaryOperator {{.+}} 'long' '-'
  // CHECK-NEXT: |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
  // CHECK-NEXT: | `-ImplicitCastExpr {{.+}} 'int *__indexable' <LValueToRValue>
  // CHECK-NEXT: |   `-DeclRefExpr {{.+}} 'int *__indexable' lvalue ParmVar {{.+}} 'p' 'int *__indexable'
  // CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
  // CHECK-NEXT:   `-ImplicitCastExpr {{.+}} 'int *__indexable' <LValueToRValue>
  // CHECK-NEXT:     `-DeclRefExpr {{.+}} 'int *__indexable' lvalue ParmVar {{.+}} 'p' 'int *__indexable'
  (void)(p - p);
}

// Pointer arithmetic with assignment should NOT convert __indexable to
// __bidi_indexable.
void assign(int *__indexable p, int index) {
  (void)p;

  // CHECK:      CompoundAssignOperator {{.+}} 'int *__indexable' '+=' ComputeLHSTy='int *__indexable' ComputeResultTy='int *__indexable'
  // CHECK-NEXT: |-DeclRefExpr {{.+}} 'int *__indexable' lvalue ParmVar {{.+}} 'p' 'int *__indexable'
  // CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
  // CHECK-NEXT:   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'index' 'int'
  p += index;

  // CHECK:      CompoundAssignOperator {{.+}} 'int *__indexable' '-=' ComputeLHSTy='int *__indexable' ComputeResultTy='int *__indexable'
  // CHECK-NEXT: |-DeclRefExpr {{.+}} 'int *__indexable' lvalue ParmVar {{.+}} 'p' 'int *__indexable'
  // CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
  // CHECK-NEXT:   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'index' 'int'
  p -= index;
}

// Array subscript should convert __indexable to __bidi_indexable.
void array_subscript(int *__indexable p, int index) {
  // CHECK:      ArraySubscriptExpr {{.+}} 'int' lvalue
  // CHECK-NEXT: |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
  // CHECK-NEXT: | `-ImplicitCastExpr {{.+}} 'int *__indexable' <LValueToRValue>
  // CHECK-NEXT: |   `-DeclRefExpr {{.+}} 'int *__indexable' lvalue ParmVar {{.+}} 'p' 'int *__indexable'
  // CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
  // CHECK-NEXT:   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'index' 'int'
  (void)&p[index];

  // CHECK:      ArraySubscriptExpr {{.+}} 'int' lvalue
  // CHECK-NEXT: |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
  // CHECK-NEXT: | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'index' 'int'
  // CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
  // CHECK-NEXT:   `-ImplicitCastExpr {{.+}} 'int *__indexable' <LValueToRValue>
  // CHECK-NEXT:     `-DeclRefExpr {{.+}} 'int *__indexable' lvalue ParmVar {{.+}} 'p' 'int *__indexable'
  (void)&index[p];
}
