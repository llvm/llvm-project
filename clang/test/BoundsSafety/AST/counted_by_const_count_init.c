

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s
#include <ptrcheck.h>

struct struct_const_count {
  unsigned const const_count;
  int *__counted_by(const_count) ptr;
};

enum enum_count { en_count = 10 };
// CHECK-LABEL: fun_enum_count
void fun_enum_count(int *__sized_by(en_count) ptr, int *__bidi_indexable buf) {
  ptr = buf;
}
// CHECK: | |-ParmVarDecl {{.*}} 'int *__single __sized_by(10)':'int *__single'
// CHECK: | |-ParmVarDecl {{.*}} 'int *__bidi_indexable'

// CHECK-LABEL: fun_pointer_access
void fun_pointer_access(struct struct_const_count *sp) {
  *(sp->ptr) = 0;
}
// CHECK: | |-ParmVarDecl [[var_sp:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   `-BinaryOperator {{.+}} 'int' '='
// CHECK: |     |-UnaryOperator {{.+}} cannot overflow
// CHECK: |     | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |     |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |     |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: |     |   | | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int *__single __counted_by(const_count)':'int *__single'
// CHECK: |     |   | | |     `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'struct struct_const_count *__single'
// CHECK: |     |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: |     |   | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |     |   | | | | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __counted_by(const_count)':'int *__single'
// CHECK: |     |   | | | `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'const unsigned int'
// CHECK: |     |   | |-OpaqueValueExpr [[ove_1]]
// CHECK: |     |   | | `-ImplicitCastExpr {{.+}} 'struct struct_const_count *__single' <LValueToRValue>
// CHECK: |     |   | |   `-DeclRefExpr {{.+}} [[var_sp]]
// CHECK: |     |   | |-OpaqueValueExpr [[ove_2]]
// CHECK: |     |   | | `-ImplicitCastExpr {{.+}} 'const unsigned int' <LValueToRValue>
// CHECK: |     |   | |   `-MemberExpr {{.+}} ->const_count
// CHECK: |     |   | |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'struct struct_const_count *__single'
// CHECK: |     |   | `-OpaqueValueExpr [[ove]]
// CHECK: |     |   |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(const_count)':'int *__single' <LValueToRValue>
// CHECK: |     |   |     `-MemberExpr {{.+}} ->ptr
// CHECK: |     |   |       `-OpaqueValueExpr [[ove_1]] {{.*}} 'struct struct_const_count *__single'
// CHECK: |     |   |-OpaqueValueExpr [[ove_1]] {{.*}} 'struct struct_const_count *__single'
// CHECK: |     |   |-OpaqueValueExpr [[ove_2]] {{.*}} 'const unsigned int'
// CHECK: |     |   `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __counted_by(const_count)':'int *__single'
// CHECK: |     `-IntegerLiteral {{.+}} 0

// CHECK-LABEL: fun_pointer_assignment
void fun_pointer_assignment(struct struct_const_count *sp, void *__bidi_indexable buf) {
  sp->ptr = buf;
}
// CHECK: | |-ParmVarDecl [[var_sp_1:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_buf:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |     |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |     | |-BoundsCheckExpr
// CHECK: |     | | |-BinaryOperator {{.+}} 'int *__single __counted_by(const_count)':'int *__single' '='
// CHECK: |     | | | |-MemberExpr {{.+}} ->ptr
// CHECK: |     | | | | `-ImplicitCastExpr {{.+}} 'struct struct_const_count *__single' <LValueToRValue>
// CHECK: |     | | | |   `-DeclRefExpr {{.+}} [[var_sp_1]]
// CHECK: |     | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(const_count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK: |     | | |   `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |     | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |     | |   | | | `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     | |   | | `-GetBoundExpr {{.+}} upper
// CHECK: |     | |   | |   `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |     | |   |   |-GetBoundExpr {{.+}} lower
// CHECK: |     | |   |   | `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |     | |   |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |     | |     |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |     | |     | `-ImplicitCastExpr {{.+}} 'unsigned int' <LValueToRValue>
// CHECK: |     | |     |   `-MemberExpr {{.+}} ->const_count
// CHECK: |     | |     |     `-ImplicitCastExpr {{.+}} 'struct struct_const_count *__single' <LValueToRValue>
// CHECK: |     | |     |       `-DeclRefExpr {{.+}} [[var_sp_1]]
// CHECK: |     | |     `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |     | |       |-GetBoundExpr {{.+}} upper
// CHECK: |     | |       | `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     | |       `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |     | |         `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     | `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     `-OpaqueValueExpr [[ove_3]]
// CHECK: |       `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BitCast>
// CHECK: |         `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <LValueToRValue>
// CHECK: |           `-DeclRefExpr {{.+}} [[var_buf]]

// FIXME: rdar://85158790
// CHECK-LABEL: fun_struct_noinit
void fun_struct_noinit() {
  struct struct_const_count s;
}
// CHECK: `-CompoundStmt
// CHECK:   `-DeclStmt
// CHECK:     `-VarDecl [[var_s:0x[^ ]+]]
