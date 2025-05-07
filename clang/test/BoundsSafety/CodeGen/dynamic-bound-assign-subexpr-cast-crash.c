

// RUN: %clang_cc1 -O0  -fbounds-safety %s -o /dev/null
// RUN: %clang_cc1 -O2  -fbounds-safety %s -o /dev/null
// RUN: %clang_cc1 -O0  -fbounds-safety -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -O0  -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s -o /dev/null
// RUN: %clang_cc1 -O2  -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s -o /dev/null
// RUN: %clang_cc1 -O0  -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s | FileCheck %s

#include <ptrcheck.h>

struct S { int *__counted_by(len) ptr; int len; };
int get_len(int *__counted_by(max_len) ptr, int max_len);

void foo(void) {
    struct S s;
    int arr[10] = {0};
    int *ptr = arr;
    s.len = get_len(ptr, 10);
    s.ptr = (ptr);
}

// CHECK-LABEL: bar
void bar(void) {
    struct S s;
    int arr[10] = {0};
    void *ptr = arr;
assignment:
    s.len = 10;
    s.ptr = (int *)ptr;
// CHECK-LABEL: assignment
// CHECK:    | `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:    |   |-BoundsCheckExpr
// CHECK:    |   | |-BinaryOperator {{.+}} 'int' '='
// CHECK:    |   | | |-MemberExpr {{.+}} .len
// CHECK:    |   | | | `-DeclRefExpr {{.+}} 's'
// CHECK:    |   | | `-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'int'
// CHECK:    |   | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:    |   |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:    |   |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:    |   |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:    |   |   | | | `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:    |   |   | | `-GetBoundExpr {{.+}} upper
// CHECK:    |   |   | |   `-OpaqueValueExpr [[ove_5]] {{.*}} 'int *__bidi_indexable'
// CHECK:    |   |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:    |   |   |   |-GetBoundExpr {{.+}} lower
// CHECK:    |   |   |   | `-OpaqueValueExpr [[ove_5]] {{.*}} 'int *__bidi_indexable'
// CHECK:    |   |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:    |   |   |     `-OpaqueValueExpr [[ove_5]] {{.*}} 'int *__bidi_indexable'
// CHECK:    |   |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:    |   |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK:    |   |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:    |   |     | | `-OpaqueValueExpr [[ove_4]] {{.*}} 'int'
// CHECK:    |   |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:    |   |     |   |-GetBoundExpr {{.+}} upper
// CHECK:    |   |     |   | `-OpaqueValueExpr [[ove_5]] {{.*}} 'int *__bidi_indexable'
// CHECK:    |   |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:    |   |     |     `-OpaqueValueExpr [[ove_5]] {{.*}} 'int *__bidi_indexable'
// CHECK:    |   |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK:    |   |       |-IntegerLiteral {{.+}} 0
// CHECK:    |   |       `-OpaqueValueExpr [[ove_4]] {{.*}} 'int'
// CHECK:    |   |-OpaqueValueExpr [[ove_4]]
// CHECK:    |   | `-IntegerLiteral {{.+}} 10
// CHECK:    |   `-OpaqueValueExpr [[ove_5]]
// CHECK:    |     `-CStyleCastExpr {{.+}} 'int *__bidi_indexable' <BitCast>
// CHECK:    |       `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <LValueToRValue>
// CHECK:    |         `-DeclRefExpr {{.+}} 'ptr'
// CHECK:    `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:      |-BinaryOperator {{.+}} 'int *__single __counted_by(len)':'int *__single' '='
// CHECK:      | |-MemberExpr {{.+}} .ptr
// CHECK:      | | `-DeclRefExpr {{.+}} 's'
// CHECK:      | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK:      |   `-OpaqueValueExpr [[ove_5]] {{.*}} 'int *__bidi_indexable'
// CHECK:      |-OpaqueValueExpr [[ove_4]] {{.*}} 'int'
// CHECK:      `-OpaqueValueExpr [[ove_5]] {{.*}} 'int *__bidi_indexable'
}
