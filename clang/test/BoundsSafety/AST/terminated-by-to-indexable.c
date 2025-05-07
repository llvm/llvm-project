
// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

// CHECK:      TerminatedByToIndexableExpr {{.+}} 'int *__indexable'
// CHECK-NEXT: |-ImplicitCastExpr {{.+}} 'int *__single __terminated_by(0)':'int *__single' <LValueToRValue>
// CHECK-NEXT: | `-DeclRefExpr {{.+}} 'int *__single __terminated_by(0)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __terminated_by(0)':'int *__single'
// CHECK-NEXT: `-<<<NULL>>>
void null(int *__null_terminated ptr) {
  __terminated_by_to_indexable(ptr);
}

// CHECK:      TerminatedByToIndexableExpr {{.+}} 'int *__indexable'
// CHECK-NEXT: |-ImplicitCastExpr {{.+}} 'int *__single __terminated_by(42)':'int *__single' <LValueToRValue>
// CHECK-NEXT: | `-DeclRefExpr {{.+}} 'int *__single __terminated_by(42)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __terminated_by(42)':'int *__single'
// CHECK-NEXT: `-<<<NULL>>>
void _42(int *__terminated_by(42) ptr) {
  __terminated_by_to_indexable(ptr);
}

// CHECK:      TerminatedByToIndexableExpr {{.+}} 'const int *__indexable'
// CHECK-NEXT: |-ImplicitCastExpr {{.+}} 'const int *__single __terminated_by(0)':'const int *__single' <LValueToRValue>
// CHECK-NEXT: | `-DeclRefExpr {{.+}} 'const int *__single __terminated_by(0)':'const int *__single' lvalue ParmVar {{.+}} 'ptr' 'const int *__single __terminated_by(0)':'const int *__single'
// CHECK-NEXT: `-<<<NULL>>>
void quals(const int *__null_terminated ptr) {
  __terminated_by_to_indexable(ptr);
}

// CHECK:      TerminatedByToIndexableExpr {{.+}} 'int *__indexable'
// CHECK-NEXT: |-ImplicitCastExpr {{.+}} 'int *__single __terminated_by(0)':'int *__single' <LValueToRValue>
// CHECK-NEXT: | `-DeclRefExpr {{.+}} 'int *__single __terminated_by(0)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __terminated_by(0)':'int *__single'
// CHECK-NEXT: `-<<<NULL>>>
void null_unsafe(int *__null_terminated ptr) {
  __unsafe_terminated_by_to_indexable(ptr);
}

// CHECK:      TerminatedByToIndexableExpr {{.+}} 'int *__indexable'
// CHECK-NEXT: |-ImplicitCastExpr {{.+}} 'int *__single __terminated_by(0)':'int *__single' <LValueToRValue>
// CHECK-NEXT: | `-ParenExpr {{.+}} 'int *__single __terminated_by(0)':'int *__single' lvalue
// CHECK-NEXT: |   `-DeclRefExpr {{.+}} 'int *__single __terminated_by(0)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __terminated_by(0)':'int *__single'
// CHECK-NEXT: `-IntegerLiteral {{.+}} 'int' 0
void explicit_null(int *__null_terminated ptr) {
  __null_terminated_to_indexable(ptr);
}

// CHECK:      TerminatedByToIndexableExpr {{.+}} 'int *__indexable'
// CHECK-NEXT: |-ImplicitCastExpr {{.+}} 'int *__single __terminated_by(0)':'int *__single' <LValueToRValue>
// CHECK-NEXT: | `-ParenExpr {{.+}} 'int *__single __terminated_by(0)':'int *__single' lvalue
// CHECK-NEXT: |   `-DeclRefExpr {{.+}} 'int *__single __terminated_by(0)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __terminated_by(0)':'int *__single'
// CHECK-NEXT: `-IntegerLiteral {{.+}} 'int' 0
void explicit_null_unsafe(int *__null_terminated ptr) {
  __unsafe_null_terminated_to_indexable(ptr);
}
