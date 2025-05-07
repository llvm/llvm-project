
// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

// Make sure that BoundsSafetyPointerCast is emitted when an atomic op is used.

void unsafe_indexable(void) {
  // CHECK: AtomicExpr {{.+}} 'void'
  // CHECK: |-UnaryOperator {{.+}} 'int *__unsafe_indexable*__bidi_indexable' prefix '&' cannot overflow
  // CHECK: | `-DeclRefExpr {{.+}} 'int *__unsafe_indexable' lvalue Var {{.+}} 'p1' 'int *__unsafe_indexable'
  // CHECK: |-IntegerLiteral {{.+}} 'int' 5
  // CHECK: `-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable' <BoundsSafetyPointerCast>
  // CHECK:   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
  // CHECK:     `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue Var {{.+}} 'q1' 'int *__bidi_indexable'
  int *__unsafe_indexable p1;
  int *__bidi_indexable q1;
  __atomic_store_n(&p1, q1, __ATOMIC_SEQ_CST);

  // CHECK: AtomicExpr {{.+}} 'int *__unsafe_indexable'
  // CHECK: |-UnaryOperator {{.+}} 'int *__unsafe_indexable*__bidi_indexable' prefix '&' cannot overflow
  // CHECK: | `-DeclRefExpr {{.+}} 'int *__unsafe_indexable' lvalue Var {{.+}} 'p2' 'int *__unsafe_indexable'
  // CHECK: |-IntegerLiteral {{.+}} 'int' 5
  // CHECK: `-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable' <BoundsSafetyPointerCast>
  // CHECK:   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
  // CHECK:     `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue Var {{.+}} 'q2' 'int *__bidi_indexable'
  int *__unsafe_indexable p2;
  int *__bidi_indexable q2;
  __atomic_exchange_n(&p2, q2, __ATOMIC_SEQ_CST);

  // CHECK: `-AtomicExpr {{.+}} '_Bool'
  // CHECK:   |-UnaryOperator {{.+}} 'int *__unsafe_indexable*__bidi_indexable' prefix '&' cannot overflow
  // CHECK:   | `-DeclRefExpr {{.+}} 'int *__unsafe_indexable' lvalue Var {{.+}} 'p3' 'int *__unsafe_indexable'
  // CHECK:   |-IntegerLiteral {{.+}} 'int' 5
  // CHECK:   |-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable*' <BoundsSafetyPointerCast>
  // CHECK:   | `-UnaryOperator {{.+}} 'int *__unsafe_indexable*__bidi_indexable' prefix '&' cannot overflow
  // CHECK:   |   `-DeclRefExpr {{.+}} 'int *__unsafe_indexable' lvalue Var {{.+}} 'q3' 'int *__unsafe_indexable'
  // CHECK:   |-IntegerLiteral {{.+}} 'int' 5
  // CHECK:   |-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable' <BoundsSafetyPointerCast>
  // CHECK:   | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
  // CHECK:   |   `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue Var {{.+}} 'r3' 'int *__bidi_indexable'
  // CHECK:   `-ImplicitCastExpr {{.+}} '_Bool' <IntegralToBoolean>
  // CHECK:     `-IntegerLiteral {{.+}} 'int' 0
  int *__unsafe_indexable p3;
  int *__unsafe_indexable q3;
  int *__bidi_indexable r3;
  __atomic_compare_exchange_n(&p3, &q3, r3, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}

void single(void) {
  // CHECK: AtomicExpr {{.+}} 'void'
  // CHECK: |-UnaryOperator {{.+}} 'int *__single*__bidi_indexable' prefix '&' cannot overflow
  // CHECK: | `-DeclRefExpr {{.+}} 'int *__single' lvalue Var {{.+}} 'p1' 'int *__single'
  // CHECK: |-IntegerLiteral {{.+}} 'int' 5
  // CHECK: `-ImplicitCastExpr {{.+}} 'int *__single' <BoundsSafetyPointerCast>
  // CHECK:   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
  // CHECK:     `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue Var {{.+}} 'q1' 'int *__bidi_indexable'
  int *__single p1;
  int *__bidi_indexable q1;
  __atomic_store_n(&p1, q1, __ATOMIC_SEQ_CST);

  // CHECK: AtomicExpr {{.+}} 'int *__single'
  // CHECK: |-UnaryOperator {{.+}} 'int *__single*__bidi_indexable' prefix '&' cannot overflow
  // CHECK: | `-DeclRefExpr {{.+}} 'int *__single' lvalue Var {{.+}} 'p2' 'int *__single'
  // CHECK: |-IntegerLiteral {{.+}} 'int' 5
  // CHECK: `-ImplicitCastExpr {{.+}} 'int *__single' <BoundsSafetyPointerCast>
  // CHECK:   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
  // CHECK:     `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue Var {{.+}} 'q2' 'int *__bidi_indexable'
  int *__single p2;
  int *__bidi_indexable q2;
  __atomic_exchange_n(&p2, q2, __ATOMIC_SEQ_CST);

  // CHECK: `-AtomicExpr {{.+}} '_Bool'
  // CHECK:   |-UnaryOperator {{.+}} 'int *__single*__bidi_indexable' prefix '&' cannot overflow
  // CHECK:   | `-DeclRefExpr {{.+}} 'int *__single' lvalue Var {{.+}} 'p3' 'int *__single'
  // CHECK:   |-IntegerLiteral {{.+}} 'int' 5
  // CHECK:   |-ImplicitCastExpr {{.+}} 'int *__single*' <BoundsSafetyPointerCast>
  // CHECK:   | `-UnaryOperator {{.+}} 'int *__single*__bidi_indexable' prefix '&' cannot overflow
  // CHECK:   |   `-DeclRefExpr {{.+}} 'int *__single' lvalue Var {{.+}} 'q3' 'int *__single'
  // CHECK:   |-IntegerLiteral {{.+}} 'int' 5
  // CHECK:   |-ImplicitCastExpr {{.+}} 'int *__single' <BoundsSafetyPointerCast>
  // CHECK:   | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
  // CHECK:   |   `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue Var {{.+}} 'r3' 'int *__bidi_indexable'
  // CHECK:   `-ImplicitCastExpr {{.+}} '_Bool' <IntegralToBoolean>
  // CHECK:     `-IntegerLiteral {{.+}} 'int' 0
  int *__single p3;
  int *__single q3;
  int *__bidi_indexable r3;
  __atomic_compare_exchange_n(&p3, &q3, r3, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}
