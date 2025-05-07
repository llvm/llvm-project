
// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

// CHECK:      TerminatedByFromIndexableExpr {{.+}} 'int *__single __terminated_by(0)':'int *__single'
// CHECK-NEXT: |-ImplicitCastExpr {{.+}} 'int *__indexable' <LValueToRValue>
// CHECK-NEXT: | `-DeclRefExpr {{.+}} 'int *__indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__indexable'
// CHECK-NEXT: `-<<<NULL>>>
void null(int *__indexable ptr) {
  __unsafe_null_terminated_from_indexable(ptr);
}

// CHECK:      TerminatedByFromIndexableExpr {{.+}} 'int *__single __terminated_by(0)':'int *__single'
// CHECK-NEXT: |-ImplicitCastExpr {{.+}} 'int *__indexable' <LValueToRValue>
// CHECK-NEXT: | `-DeclRefExpr {{.+}} 'int *__indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__indexable'
// CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'int *__indexable' <LValueToRValue>
// CHECK-NEXT:   `-DeclRefExpr {{.+}} 'int *__indexable' lvalue ParmVar {{.+}} 'ptr_to_term' 'int *__indexable'
void null_ptr_to_term(int *__indexable ptr, int *__indexable ptr_to_term) {
  __unsafe_null_terminated_from_indexable(ptr, ptr_to_term);
}

// CHECK:      TerminatedByFromIndexableExpr {{.+}} 'int *__single __terminated_by((42))':'int *__single'
// CHECK-NEXT: |-ImplicitCastExpr {{.+}} 'int *__indexable' <LValueToRValue>
// CHECK-NEXT: | `-DeclRefExpr {{.+}} 'int *__indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__indexable'
// CHECK-NEXT: `-<<<NULL>>>
void _42(int *__indexable ptr) {
  __unsafe_terminated_by_from_indexable(42, ptr);
}

static int array[42];

// CHECK:      TerminatedByFromIndexableExpr {{.+}} 'int *__single __terminated_by(0)':'int *__single'
// CHECK-NEXT: |-ImplicitCastExpr {{.+}} 'int *__indexable' <BoundsSafetyPointerCast>
// CHECK-NEXT: | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |   `-DeclRefExpr {{.+}} 'int[42]' lvalue Var {{.+}} 'array' 'int[42]'
// CHECK-NEXT: `-<<<NULL>>>
void decay(void) {
  __unsafe_null_terminated_from_indexable(array);
}

// CHECK:      TerminatedByFromIndexableExpr {{.+}} 'const int *__single __terminated_by(0)':'const int *__single'
// CHECK-NEXT: |-ImplicitCastExpr {{.+}} 'const int *__indexable' <LValueToRValue>
// CHECK-NEXT: | `-DeclRefExpr {{.+}} 'const int *__indexable' lvalue ParmVar {{.+}} 'ptr' 'const int *__indexable'
// CHECK-NEXT: `-<<<NULL>>>
void quals(const int *__indexable ptr) {
  __unsafe_null_terminated_from_indexable(ptr);
}
