

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s
#include <ptrcheck.h>

// CHECK:      {{^}}|-VarDecl [[var_data_const_count:0x[^ ]+]]
// CHECK-NEXT: {{^}}| |-UnsafeLateConstAttr
// CHECK-NEXT: {{^}}| `-DependerDeclsAttr
unsigned data_const_count __unsafe_late_const;

// CHECK: {{^}}|-RecordDecl
// CHECK-NEXT: {{^}}| `-FieldDecl
struct struct_data_const_count {
  int *__counted_by(data_const_count) ptr;
};

// CHECK: {{^}}|-VarDecl [[var_data_const_count_flex:0x[^ ]+]]
// CHECK-NEXT: {{^}}| |-UnsafeLateConstAttr
// CHECK-NEXT: {{^}}| `-DependerDeclsAttr
unsigned data_const_count_flex __unsafe_late_const;

// CHECK: {{^}}|-RecordDecl
// CHECK-NEXT: {{^}}| |-FieldDecl
// CHECK-NEXT: {{^}}| `-FieldDecl
struct struct_data_const_count_flex {
  int count;
  int fam[__counted_by(data_const_count_flex)];
};

// CHECK-LABEL: fun_pointer_assignment
void fun_pointer_assignment(struct struct_data_const_count *sp, void *__bidi_indexable buf) {
  sp->ptr = buf;
}
// CHECK: | |-ParmVarDecl [[var_sp:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_buf:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |     |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |     | |-BoundsCheckExpr
// CHECK: |     | | |-BinaryOperator {{.+}} 'int *__single __counted_by(data_const_count)':'int *__single' '='
// CHECK: |     | | | |-MemberExpr {{.+}} ->ptr
// CHECK: |     | | | | `-ImplicitCastExpr {{.+}} 'struct struct_data_const_count *__single' <LValueToRValue>
// CHECK: |     | | | |   `-DeclRefExpr {{.+}} [[var_sp]]
// CHECK: |     | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(data_const_count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK: |     | | |   `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |     | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |     | |   | | | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     | |   | | `-GetBoundExpr {{.+}} upper
// CHECK: |     | |   | |   `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |     | |   |   |-GetBoundExpr {{.+}} lower
// CHECK: |     | |   |   | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |     | |   |     `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |     | |     |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |     | |     | `-ImplicitCastExpr {{.+}} 'unsigned int' <LValueToRValue>
// CHECK: |     | |     |   `-DeclRefExpr {{.+}} [[var_data_const_count]]
// CHECK: |     | |     `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |     | |       |-GetBoundExpr {{.+}} upper
// CHECK: |     | |       | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     | |       `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |     | |         `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     `-OpaqueValueExpr [[ove]]
// CHECK: |       `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BitCast>
// CHECK: |         `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <LValueToRValue>
// CHECK: |           `-DeclRefExpr {{.+}} [[var_buf]]

// CHECK-LABEL: fun_pointer_assignment2
void fun_pointer_assignment2(struct struct_data_const_count *sp, void *__bidi_indexable buf) {
  sp->ptr = buf;
  data_const_count = 10; // XXX: The assignment precheck at `buf` won't take into account this new count assignment.
}
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_sp_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_buf_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   | |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|   | | |-BoundsCheckExpr {{.+}} 'buf <= __builtin_get_pointer_upper_bound(buf) && __builtin_get_pointer_lower_bound(buf) <= buf && data_const_count <= __builtin_get_pointer_upper_bound(buf) - buf'
// CHECK-NEXT: {{^}}|   | | | |-BinaryOperator {{.+}} 'int *__single __counted_by(data_const_count)':'int *__single' '='
// CHECK-NEXT: {{^}}|   | | | | |-MemberExpr {{.+}} ->ptr
// CHECK-NEXT: {{^}}|   | | | | | `-ImplicitCastExpr {{.+}} 'struct struct_data_const_count *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}|   | | | | |   `-DeclRefExpr {{.+}} [[var_sp_1]]
// CHECK-NEXT: {{^}}|   | | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(data_const_count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | | | |   `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | | |   | | | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | | |   | | `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|   | | |   | |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | | |   |   |-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}|   | | |   |   | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | | |   |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | | |     |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|   | | |     | `-ImplicitCastExpr {{.+}} 'unsigned int' <LValueToRValue>
// CHECK-NEXT: {{^}}|   | | |     |   `-DeclRefExpr {{.+}} [[var_data_const_count]]
// CHECK-NEXT: {{^}}|   | | |     `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}|   | | |       |-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|   | | |       | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | | |       `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | | |         `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | | `-OpaqueValueExpr [[ove_1]]
// CHECK-NEXT: {{^}}|   | |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}|   | |     `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|   | |       `-DeclRefExpr {{.+}} [[var_buf_1]]
// CHECK-NEXT: {{^}}|   | `-OpaqueValueExpr [[ove_1]]
// CHECK-NEXT: {{^}}|   |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}|   |     `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |       `-DeclRefExpr {{.+}} [[var_buf_1]]
// CHECK-NEXT: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|     |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     | |-BinaryOperator {{.+}} 'unsigned int' '='
// CHECK-NEXT: {{^}}|     | | |-DeclRefExpr {{.+}} [[var_data_const_count]]
// CHECK-NEXT: {{^}}|     | | `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'unsigned int'
// CHECK:      {{^}}|     | `-OpaqueValueExpr [[ove_2]]
// CHECK-NEXT: {{^}}|     |   `-ImplicitCastExpr {{.+}} 'unsigned int' <IntegralCast>
// CHECK-NEXT: {{^}}|     |     `-IntegerLiteral {{.+}} 10
// CHECK-NEXT: {{^}}|     `-OpaqueValueExpr [[ove_2]]
// CHECK-NEXT: {{^}}|       `-ImplicitCastExpr {{.+}} 'unsigned int' <IntegralCast>
// CHECK-NEXT: {{^}}|         `-IntegerLiteral {{.+}} 10


// CHECK-LABEL: fun_flex_pointer_assignment
void fun_flex_pointer_assignment(struct struct_data_const_count_flex *sp, void *__bidi_indexable buf) {
  sp = buf;
}
// CHECK: |-ParmVarDecl [[var_sp_2:0x[^ ]+]]
// CHECK: |-ParmVarDecl [[var_buf_2:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     | |-BinaryOperator {{.+}} 'struct struct_data_const_count_flex *__single' '='
// CHECK:     | | |-DeclRefExpr {{.+}} [[var_sp_2]]
// CHECK:     | | `-ImplicitCastExpr {{.+}} 'struct struct_data_const_count_flex *__single' <BoundsSafetyPointerCast>
// CHECK:     | |   `-PredefinedBoundsCheckExpr {{.+}} 'struct struct_data_const_count_flex *__bidi_indexable' <FlexibleArrayCountAssign(BasePtr, FamPtr, Count)>
// CHECK:     | |     |-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'struct struct_data_const_count_flex *__bidi_indexable'
// CHECK:     | |     |-OpaqueValueExpr [[ove_2]] {{.*}} 'struct struct_data_const_count_flex *__bidi_indexable'
// CHECK:     | |     |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK:     | |     | `-MemberExpr {{.+}} ->fam
// CHECK:     | |     |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'struct struct_data_const_count_flex *__bidi_indexable'
// CHECK:     | |     `-ImplicitCastExpr {{.+}} 'unsigned int' <LValueToRValue>
// CHECK:     | |       `-DeclRefExpr {{.+}} [[var_data_const_count_flex]]
// CHECK:     | `-OpaqueValueExpr [[ove_2]] {{.*}} 'struct struct_data_const_count_flex *__bidi_indexable'
// CHECK:     `-OpaqueValueExpr [[ove_2]]
// CHECK:       `-ImplicitCastExpr {{.+}} 'struct struct_data_const_count_flex *__bidi_indexable' <BitCast>
// CHECK:         `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <LValueToRValue>
// CHECK:           `-DeclRefExpr {{.+}} [[var_buf_2]]

// CHECK-LABEL: fun_flex_pointer_assignment2
void fun_flex_pointer_assignment2(struct struct_data_const_count_flex *sp, void *__bidi_indexable buf) {
  sp = buf;
  data_const_count_flex = 100; // XXX: The assignment precheck at `buf` won't take into account this new count assignment.
}
// CHECK-NEXT: {{^}}  |-ParmVarDecl [[var_sp_3:0x[^ ]+]]
// CHECK-NEXT: {{^}}  |-ParmVarDecl [[var_buf_3:0x[^ ]+]]
// CHECK-NEXT: {{^}}  `-CompoundStmt
// CHECK-NEXT: {{^}}    |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}    | |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}    | | |-BinaryOperator {{.+}} 'struct struct_data_const_count_flex *__single' '='
// CHECK-NEXT: {{^}}    | | | |-DeclRefExpr {{.+}} [[var_sp_3]]
// CHECK-NEXT: {{^}}    | | | `-ImplicitCastExpr {{.+}} 'struct struct_data_const_count_flex *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    | | |   `-PredefinedBoundsCheckExpr {{.+}} 'struct struct_data_const_count_flex *__bidi_indexable' <FlexibleArrayCountAssign(BasePtr, FamPtr, Count)>
// CHECK-NEXT: {{^}}    | | |     |-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'struct struct_data_const_count_flex *__bidi_indexable'
// CHECK:      {{^}}    | | |     |-OpaqueValueExpr [[ove_4]] {{.*}} 'struct struct_data_const_count_flex *__bidi_indexable'
// CHECK:      {{^}}    | | |     |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: {{^}}    | | |     | `-MemberExpr {{.+}} ->fam
// CHECK-NEXT: {{^}}    | | |     |   `-OpaqueValueExpr [[ove_4]] {{.*}} 'struct struct_data_const_count_flex *__bidi_indexable'
// CHECK:      {{^}}    | | |     `-ImplicitCastExpr {{.+}} 'unsigned int' <LValueToRValue>
// CHECK-NEXT: {{^}}    | | |       `-DeclRefExpr {{.+}} [[var_data_const_count_flex]]
// CHECK-NEXT: {{^}}    | | `-OpaqueValueExpr [[ove_4]]
// CHECK-NEXT: {{^}}    | |   `-ImplicitCastExpr {{.+}} 'struct struct_data_const_count_flex *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}    | |     `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}    | |       `-DeclRefExpr {{.+}} [[var_buf_3]]
// CHECK-NEXT: {{^}}    | `-OpaqueValueExpr [[ove_4]]
// CHECK-NEXT: {{^}}    |   `-ImplicitCastExpr {{.+}} 'struct struct_data_const_count_flex *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}    |     `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}    |       `-DeclRefExpr {{.+}} [[var_buf_3]]
// CHECK-NEXT: {{^}}    `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}      |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}      | |-BinaryOperator {{.+}} 'unsigned int' '='
// CHECK-NEXT: {{^}}      | | |-DeclRefExpr {{.+}} [[var_data_const_count_flex]]
// CHECK-NEXT: {{^}}      | | `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'unsigned int'
// CHECK:      {{^}}      | `-OpaqueValueExpr [[ove_5]]
// CHECK-NEXT: {{^}}      |   `-ImplicitCastExpr {{.+}} 'unsigned int' <IntegralCast>
// CHECK-NEXT: {{^}}      |     `-IntegerLiteral {{.+}} 100
// CHECK-NEXT: {{^}}      `-OpaqueValueExpr [[ove_5]]
// CHECK-NEXT: {{^}}        `-ImplicitCastExpr {{.+}} 'unsigned int' <IntegralCast>
// CHECK-NEXT: {{^}}          `-IntegerLiteral {{.+}} 100
