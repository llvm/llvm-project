
// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

// Make sure that BoundsSafetyPointerCast is emitted when an atomic op is used.

void unsafe_indexable(void) {
  // CHECK: DeclStmt {{.+}}
  // CHECK: `-VarDecl {{.+}} p1 '_Atomic(int *__unsafe_indexable)' cinit
  // CHECK:   `-ImplicitCastExpr {{.+}} '_Atomic(int *__unsafe_indexable)' <NonAtomicToAtomic>
  // CHECK:     `-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable' <BoundsSafetyPointerCast>
  // CHECK:       `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
  // CHECK:         `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue Var {{.+}} 'q1' 'int *__bidi_indexable'
  int *__bidi_indexable q1;
  int *_Atomic __unsafe_indexable p1 = q1;

  // CHECK: AtomicExpr {{.+}} 'void'
  // CHECK: |-UnaryOperator {{.+}} '_Atomic(int *__unsafe_indexable) *__bidi_indexable' prefix '&' cannot overflow
  // CHECK: | `-DeclRefExpr {{.+}} '_Atomic(int *__unsafe_indexable)' lvalue Var {{.+}} 'p2' '_Atomic(int *__unsafe_indexable)'
  // CHECK: `-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable' <BoundsSafetyPointerCast>
  // CHECK:   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
  // CHECK:     `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue Var {{.+}} 'q2' 'int *__bidi_indexable'
  int *_Atomic __unsafe_indexable p2;
  int *__bidi_indexable q2;
  __c11_atomic_init(&p2, q2);

  // CHECK: AtomicExpr {{.+}} 'void'
  // CHECK: |-UnaryOperator {{.+}} '_Atomic(int *__unsafe_indexable) *__bidi_indexable' prefix '&' cannot overflow
  // CHECK: | `-DeclRefExpr {{.+}} '_Atomic(int *__unsafe_indexable)' lvalue Var {{.+}} 'p3' '_Atomic(int *__unsafe_indexable)'
  // CHECK: |-IntegerLiteral {{.+}} 'int' 5
  // CHECK: `-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable' <BoundsSafetyPointerCast>
  // CHECK:   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
  // CHECK:     `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue Var {{.+}} 'q3' 'int *__bidi_indexable'
  int *_Atomic __unsafe_indexable p3;
  int *__bidi_indexable q3;
  __c11_atomic_store(&p3, q3, __ATOMIC_SEQ_CST);

  // CHECK: AtomicExpr {{.+}} 'int *__unsafe_indexable'
  // CHECK: |-UnaryOperator {{.+}} '_Atomic(int *__unsafe_indexable) *__bidi_indexable' prefix '&' cannot overflow
  // CHECK: | `-DeclRefExpr {{.+}} '_Atomic(int *__unsafe_indexable)' lvalue Var {{.+}} 'p4' '_Atomic(int *__unsafe_indexable)'
  // CHECK: |-IntegerLiteral {{.+}} 'int' 5
  // CHECK: `-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable' <BoundsSafetyPointerCast>
  // CHECK:   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
  // CHECK:     `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue Var {{.+}} 'q4' 'int *__bidi_indexable'
  int *_Atomic __unsafe_indexable p4;
  int *__bidi_indexable q4;
  __c11_atomic_exchange(&p4, q4, __ATOMIC_SEQ_CST);

  // CHECK: AtomicExpr {{.+}} '_Bool'
  // CHECK: |-UnaryOperator {{.+}} '_Atomic(int *__unsafe_indexable) *__bidi_indexable' prefix '&' cannot overflow
  // CHECK: | `-DeclRefExpr {{.+}} '_Atomic(int *__unsafe_indexable)' lvalue Var {{.+}} 'p5' '_Atomic(int *__unsafe_indexable)'
  // CHECK: |-IntegerLiteral {{.+}} 'int' 5
  // CHECK: |-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable*' <BoundsSafetyPointerCast>
  // CHECK: | `-UnaryOperator {{.+}} 'int *__unsafe_indexable*__bidi_indexable' prefix '&' cannot overflow
  // CHECK: |   `-DeclRefExpr {{.+}} 'int *__unsafe_indexable' lvalue Var {{.+}} 'q5' 'int *__unsafe_indexable'
  // CHECK: |-IntegerLiteral {{.+}} 'int' 5
  // CHECK: `-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable' <BoundsSafetyPointerCast>
  // CHECK:   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
  // CHECK:     `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue Var {{.+}} 'r5' 'int *__bidi_indexable'
  int *_Atomic __unsafe_indexable p5;
  int *__unsafe_indexable q5;
  int *__bidi_indexable r5;
  __c11_atomic_compare_exchange_strong(&p5, &q5, r5, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}

void single(void) {
  // CHECK: DeclStmt {{.+}}
  // CHECK: `-VarDecl {{.+}} p1 '_Atomic(int *__single)' cinit
  // CHECK:   `-ImplicitCastExpr {{.+}} '_Atomic(int *__single)' <NonAtomicToAtomic>
  // CHECK:     `-ImplicitCastExpr {{.+}} 'int *__single' <BoundsSafetyPointerCast>
  // CHECK:       `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
  // CHECK:         `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue Var {{.+}} 'q1' 'int *__bidi_indexable'
  int *__bidi_indexable q1;
  int *_Atomic __single p1 = q1;

  // CHECK: AtomicExpr {{.+}} 'void'
  // CHECK: |-UnaryOperator {{.+}} '_Atomic(int *__single) *__bidi_indexable' prefix '&' cannot overflow
  // CHECK: | `-DeclRefExpr {{.+}} '_Atomic(int *__single)' lvalue Var {{.+}} 'p2' '_Atomic(int *__single)'
  // CHECK: `-ImplicitCastExpr {{.+}} 'int *__single' <BoundsSafetyPointerCast>
  // CHECK:   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
  // CHECK:     `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue Var {{.+}} 'q2' 'int *__bidi_indexable'
  int *_Atomic __single p2;
  int *__bidi_indexable q2;
  __c11_atomic_init(&p2, q2);

  // CHECK: AtomicExpr {{.+}} 'void'
  // CHECK: |-UnaryOperator {{.+}} '_Atomic(int *__single) *__bidi_indexable' prefix '&' cannot overflow
  // CHECK: | `-DeclRefExpr {{.+}} '_Atomic(int *__single)' lvalue Var {{.+}} 'p3' '_Atomic(int *__single)'
  // CHECK: |-IntegerLiteral {{.+}} 'int' 5
  // CHECK: `-ImplicitCastExpr {{.+}} 'int *__single' <BoundsSafetyPointerCast>
  // CHECK:   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
  // CHECK:     `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue Var {{.+}} 'q3' 'int *__bidi_indexable'
  int *_Atomic __single p3;
  int *__bidi_indexable q3;
  __c11_atomic_store(&p3, q3, __ATOMIC_SEQ_CST);

  // CHECK: AtomicExpr {{.+}} 'int *__single'
  // CHECK: |-UnaryOperator {{.+}} '_Atomic(int *__single) *__bidi_indexable' prefix '&' cannot overflow
  // CHECK: | `-DeclRefExpr {{.+}} '_Atomic(int *__single)' lvalue Var {{.+}} 'p4' '_Atomic(int *__single)'
  // CHECK: |-IntegerLiteral {{.+}} 'int' 5
  // CHECK: `-ImplicitCastExpr {{.+}} 'int *__single' <BoundsSafetyPointerCast>
  // CHECK:   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
  // CHECK:     `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue Var {{.+}} 'q4' 'int *__bidi_indexable'
  int *_Atomic __single p4;
  int *__bidi_indexable q4;
  __c11_atomic_exchange(&p4, q4, __ATOMIC_SEQ_CST);

  // CHECK: AtomicExpr {{.+}} '_Bool'
  // CHECK: |-UnaryOperator {{.+}} '_Atomic(int *__single) *__bidi_indexable' prefix '&' cannot overflow
  // CHECK: | `-DeclRefExpr {{.+}} '_Atomic(int *__single)' lvalue Var {{.+}} 'p5' '_Atomic(int *__single)'
  // CHECK: |-IntegerLiteral {{.+}} 'int' 5
  // CHECK: |-ImplicitCastExpr {{.+}} 'int *__single*' <BoundsSafetyPointerCast>
  // CHECK: | `-UnaryOperator {{.+}} 'int *__single*__bidi_indexable' prefix '&' cannot overflow
  // CHECK: |   `-DeclRefExpr {{.+}} 'int *__single' lvalue Var {{.+}} 'q5' 'int *__single'
  // CHECK: |-IntegerLiteral {{.+}} 'int' 5
  // CHECK: `-ImplicitCastExpr {{.+}} 'int *__single' <BoundsSafetyPointerCast>
  // CHECK:   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
  // CHECK:     `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue Var {{.+}} 'r5' 'int *__bidi_indexable'
  int *_Atomic __single p5;
  int *__single q5;
  int *__bidi_indexable r5;
  __c11_atomic_compare_exchange_strong(&p5, &q5, r5, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}
