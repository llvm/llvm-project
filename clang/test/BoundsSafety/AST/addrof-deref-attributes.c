

// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

void test(int *__bidi_indexable bidi, int *__indexable unidi,
          int *__unsafe_indexable unsafe, int *__null_terminated tb) {
  (void)&*bidi;
  (void)&bidi[4];

  (void)&*unidi;
  (void)&unidi[4];

  (void)&*unsafe;
  (void)&unsafe[4];

  (void)&*tb;
  (void)&tb[0];
}

// CHECK:      CStyleCastExpr {{.+}} 'void' <ToVoid>
// CHECK-NEXT: `-UnaryOperator {{.+}} 'int *__bidi_indexable' prefix '&' cannot overflow
// CHECK-NEXT:   `-UnaryOperator {{.+}} 'int' lvalue prefix '*' cannot overflow
// CHECK-NEXT:     `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT:       `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'bidi' 'int *__bidi_indexable'
// CHECK:      CStyleCastExpr {{.+}} 'void' <ToVoid>
// CHECK-NEXT: `-UnaryOperator {{.+}} 'int *__bidi_indexable' prefix '&' cannot overflow
// CHECK-NEXT:   `-ArraySubscriptExpr {{.+}} 'int' lvalue
// CHECK-NEXT:     |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT:     | `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'bidi' 'int *__bidi_indexable'
// CHECK-NEXT:     `-IntegerLiteral

// CHECK:      CStyleCastExpr {{.+}} 'void' <ToVoid>
// CHECK-NEXT: `-UnaryOperator {{.+}} 'int *__indexable' prefix '&' cannot overflow
// CHECK-NEXT:   `-UnaryOperator {{.+}} 'int' lvalue prefix '*' cannot overflow
// CHECK-NEXT:     `-ImplicitCastExpr {{.+}} 'int *__indexable' <LValueToRValue>
// CHECK-NEXT:       `-DeclRefExpr {{.+}} 'int *__indexable' lvalue ParmVar {{.+}} 'unidi' 'int *__indexable'
// CHECK:      CStyleCastExpr {{.+}} 'void' <ToVoid>
// CHECK-NEXT: `-UnaryOperator {{.+}} 'int *__bidi_indexable' prefix '&' cannot overflow
// CHECK-NEXT:   `-ArraySubscriptExpr {{.+}} 'int' lvalue
// CHECK-NEXT:     |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK-NEXT:     | `-ImplicitCastExpr {{.+}} 'int *__indexable' <LValueToRValue>
// CHECK-NEXT:     |   `-DeclRefExpr {{.+}} 'int *__indexable' lvalue ParmVar {{.+}} 'unidi' 'int *__indexable'
// CHECK-NEXT:     `-IntegerLiteral {{.+}} 'int' 4

// CHECK:      CStyleCastExpr {{.+}} 'void' <ToVoid>
// CHECK-NEXT: `-UnaryOperator {{.+}} 'int *__unsafe_indexable' prefix '&' cannot overflow
// CHECK-NEXT:   `-UnaryOperator {{.+}} 'int' lvalue prefix '*' cannot overflow
// CHECK-NEXT:     `-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable' <LValueToRValue>
// CHECK-NEXT:       `-DeclRefExpr {{.+}} 'int *__unsafe_indexable' lvalue ParmVar {{.+}} 'unsafe' 'int *__unsafe_indexable'
// CHECK:      CStyleCastExpr {{.+}} 'void' <ToVoid>
// CHECK-NEXT: `-UnaryOperator {{.+}} 'int *__unsafe_indexable' prefix '&' cannot overflow
// CHECK-NEXT:   `-ArraySubscriptExpr {{.+}} 'int' lvalue
// CHECK-NEXT:     |-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable' <LValueToRValue>
// CHECK-NEXT:     | `-DeclRefExpr {{.+}} 'int *__unsafe_indexable' lvalue ParmVar {{.+}} 'unsafe' 'int *__unsafe_indexable'
// CHECK-NEXT:     `-IntegerLiteral

// CHECK:      CStyleCastExpr {{.+}} 'void' <ToVoid>
// CHECK-NEXT: `-UnaryOperator {{.+}} 'int *__single __terminated_by(0)':'int *__single' prefix '&' cannot overflow
// CHECK-NEXT:   `-UnaryOperator {{.+}} 'int' lvalue prefix '*' cannot overflow
// CHECK-NEXT:     `-ImplicitCastExpr {{.+}} 'int *__single __terminated_by(0)':'int *__single' <LValueToRValue>
// CHECK-NEXT:       `-DeclRefExpr {{.+}} 'int *__single __terminated_by(0)':'int *__single' lvalue ParmVar {{.+}} 'tb' 'int *__single __terminated_by(0)':'int *__single'
// CHECK:      CStyleCastExpr {{.+}} 'void' <ToVoid>
// CHECK-NEXT: `-UnaryOperator {{.+}} 'int *__single __terminated_by(0)':'int *__single' prefix '&' cannot overflow
// CHECK-NEXT:   `-ArraySubscriptExpr {{.+}} 'int' lvalue
// CHECK-NEXT:     |-ImplicitCastExpr {{.+}} 'int *__single __terminated_by(0)':'int *__single' <LValueToRValue>
// CHECK-NEXT:     | `-DeclRefExpr {{.+}} 'int *__single __terminated_by(0)':'int *__single' lvalue ParmVar {{.+}} 'tb' 'int *__single __terminated_by(0)':'int *__single'
// CHECK-NEXT:     `-IntegerLiteral
