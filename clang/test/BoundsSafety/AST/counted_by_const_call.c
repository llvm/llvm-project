

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s
#include <ptrcheck.h>

int data_const_count __unsafe_late_const;

// CHECK: |-FunctionDecl [[func_fun_const_count:0x[^ ]+]] {{.+}} fun_const_count
__attribute__((const)) int fun_const_count() {
  return data_const_count;
}

struct struct_const_count_call {
  int *__counted_by(fun_const_count()) ptr;
};

// CHECK-LABEL: fun_pointer_access
void fun_pointer_access(struct struct_const_count_call *sp) {
  *(sp->ptr) = 0;
}
// CHECK: | |-ParmVarDecl [[var_sp:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   `-BinaryOperator {{.+}} 'int' '='
// CHECK: |     |-UnaryOperator {{.+}} cannot overflow
// CHECK: |     | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |     |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |     |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: |     |   | | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int *__single __counted_by(fun_const_count())':'int *__single'
// CHECK: |     |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: |     |   | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |     |   | | | | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __counted_by(fun_const_count())':'int *__single'
// CHECK: |     |   | | | `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int'
// CHECK: |     |   | |-OpaqueValueExpr [[ove]]
// CHECK: |     |   | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(fun_const_count())':'int *__single' <LValueToRValue>
// CHECK: |     |   | |   `-ParenExpr
// CHECK: |     |   | |     `-MemberExpr {{.+}} ->ptr
// CHECK: |     |   | |       `-ImplicitCastExpr {{.+}} 'struct struct_const_count_call *__single' <LValueToRValue>
// CHECK: |     |   | |         `-DeclRefExpr {{.+}} [[var_sp]]
// CHECK: |     |   | `-OpaqueValueExpr [[ove_1]]
// CHECK: |     |   |   `-CallExpr
// CHECK: |     |   |     `-ImplicitCastExpr {{.+}} 'int (*__single)()' <FunctionToPointerDecay>
// CHECK: |     |   |       `-DeclRefExpr {{.+}} [[func_fun_const_count]]
// CHECK: |     |   |-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __counted_by(fun_const_count())':'int *__single'
// CHECK: |     |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK: |     `-IntegerLiteral {{.+}} 0

// CHECK-LABEL: fun_pointer_assignment
void fun_pointer_assignment(struct struct_const_count_call *sp, void *__bidi_indexable buf) {
  sp->ptr = buf;
}
// CHECK: | |-ParmVarDecl [[var_sp_1:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_buf:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |     |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |     | |-BoundsCheckExpr
// CHECK: |     | | |-BinaryOperator {{.+}} 'int *__single __counted_by(fun_const_count())':'int *__single' '='
// CHECK: |     | | | |-MemberExpr {{.+}} ->ptr
// CHECK: |     | | | | `-ImplicitCastExpr {{.+}} 'struct struct_const_count_call *__single' <LValueToRValue>
// CHECK: |     | | | |   `-DeclRefExpr {{.+}} [[var_sp_1]]
// CHECK: |     | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(fun_const_count())':'int *__single' <BoundsSafetyPointerCast>
// CHECK: |     | | |   `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     | | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |     | | | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |     | | | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |     | | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |     | | | | | | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     | | | | | `-GetBoundExpr {{.+}} upper
// CHECK: |     | | | | |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     | | | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |     | | | |   |-GetBoundExpr {{.+}} lower
// CHECK: |     | | | |   | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     | | | |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |     | | | |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     | | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |     | | |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |     | | |   | |-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'long'
// CHECK: |     | | |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |     | | |   |   |-GetBoundExpr {{.+}} upper
// CHECK: |     | | |   |   | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     | | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |     | | |   |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     | | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |     | | |     |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |     | | |     | `-IntegerLiteral {{.+}} 0
// CHECK: |     | | |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'long'
// CHECK: |     | | `-OpaqueValueExpr [[ove_3]]
// CHECK: |     | |   `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |     | |     `-CallExpr
// CHECK: |     | |       `-ImplicitCastExpr {{.+}} 'int (*__single)()' <FunctionToPointerDecay>
// CHECK: |     | |         `-DeclRefExpr {{.+}} [[func_fun_const_count]]
// CHECK: |     | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     `-OpaqueValueExpr [[ove_2]]
// CHECK: |       `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BitCast>
// CHECK: |         `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <LValueToRValue>
// CHECK: |           `-DeclRefExpr {{.+}} [[var_buf]]

// FIXME: rdar://85158790
// CHECK-LABEL: fun_struct_noinit
void fun_struct_noinit() {
  struct struct_const_count_call s;
}
// CHECK: | `-CompoundStmt
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl [[var_s:0x[^ ]+]]

// CHECK-LABEL: fun_struct_init
void fun_struct_init(int *__bidi_indexable buf) {
  struct struct_const_count_call s = { buf };
}
// CHECK: |-ParmVarDecl [[var_buf_1:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-DeclStmt
// CHECK:     `-VarDecl [[var_s_1:0x[^ ]+]]
// CHECK:       `-BoundsCheckExpr
// CHECK:         |-InitListExpr
// CHECK:         | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(fun_const_count())':'int *__single' <BoundsSafetyPointerCast>
// CHECK:         |   `-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:         |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:         | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:         | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:         | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:         | | | | `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:         | | | `-GetBoundExpr {{.+}} upper
// CHECK:         | | |   `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:         | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:         | |   |-GetBoundExpr {{.+}} lower
// CHECK:         | |   | `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:         | |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:         | |     `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:         | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:         |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK:         |   | |-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'long'
// CHECK:         |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:         |   |   |-GetBoundExpr {{.+}} upper
// CHECK:         |   |   | `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:         |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:         |   |     `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:         |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:         |     |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:         |     | `-IntegerLiteral {{.+}} 0
// CHECK:         |     `-OpaqueValueExpr [[ove_5]] {{.*}} 'long'
// CHECK:         |-OpaqueValueExpr [[ove_4]]
// CHECK:         | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK:         |   `-DeclRefExpr {{.+}} [[var_buf_1]]
// CHECK:         `-OpaqueValueExpr [[ove_5]]
// CHECK:           `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:             `-CallExpr
// CHECK:               `-ImplicitCastExpr {{.+}} 'int (*__single)()' <FunctionToPointerDecay>
// CHECK:                 `-DeclRefExpr {{.+}} [[func_fun_const_count]]

