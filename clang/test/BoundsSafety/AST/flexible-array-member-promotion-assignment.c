

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s

#include <ptrcheck.h>

struct flexible {
    int count;
    int elems[__counted_by(count)];
};

// CHECK-LABEL: promote_to_bidi_indexable
void promote_to_bidi_indexable(struct flexible *flex) {
  struct flexible *b = flex;
}
// CHECK: |-ParmVarDecl [[var_flex:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-DeclStmt
// CHECK:     `-VarDecl [[var_b:0x[^ ]+]]
// CHECK:       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK:         | | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'struct flexible *__single'
// CHECK:         | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK:         | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK:         | | | | `-MemberExpr {{.+}} ->elems
// CHECK:         | | | |   `-OpaqueValueExpr [[ove]] {{.*}} 'struct flexible *__single'
// CHECK:         | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:         | | |   `-MemberExpr {{.+}} ->count
// CHECK:         | | |     `-OpaqueValueExpr [[ove]] {{.*}} 'struct flexible *__single'
// CHECK:         | `-OpaqueValueExpr [[ove]]
// CHECK:         |   `-ImplicitCastExpr {{.+}} 'struct flexible *__single' <LValueToRValue>
// CHECK:         |     `-DeclRefExpr {{.+}} [[var_flex]]
// CHECK:         `-OpaqueValueExpr [[ove]] {{.*}} 'struct flexible *__single'

// CHECK-LABEL: promote_null_to_bidi_indexable
void promote_null_to_bidi_indexable(void) {
  struct flexible *b = (struct flexible *)0;
}
// CHECK: CompoundStmt
// CHECK: `-DeclStmt
// CHECK:   `-VarDecl [[var_b_1:0x[^ ]+]]
// CHECK:     `-ImplicitCastExpr {{.+}} 'struct flexible *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK:       `-CStyleCastExpr {{.+}} 'struct flexible *' <NullToPointer>
// CHECK:         `-IntegerLiteral {{.+}} 0

// CHECK-LABEL: promote_null_to_single
void promote_null_to_single() {
  struct flexible *__single b = (struct flexible *)0;
}
// CHECK: | `-CompoundStmt
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl [[var_b_2:0x[^ ]+]]
// CHECK: |       `-ImplicitCastExpr {{.+}} 'struct flexible *__single' <BoundsSafetyPointerCast>
// CHECK: |         `-CStyleCastExpr {{.+}} 'struct flexible *' <NullToPointer>
// CHECK: |           `-IntegerLiteral {{.+}} 0

// CHECK-LABEL: promote_to_single
void promote_to_single(struct flexible *flex) {
  struct flexible *__single s = flex;
}
// CHECK: |-ParmVarDecl [[var_flex_1:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-DeclStmt
// CHECK:     `-VarDecl [[var_s:0x[^ ]+]]
// CHECK:       `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:         |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:         | |-ImplicitCastExpr {{.+}} 'struct flexible *__single' <BoundsSafetyPointerCast>
// CHECK:         | | `-PredefinedBoundsCheckExpr {{.+}} 'struct flexible *__bidi_indexable' <FlexibleArrayCountAssign(BasePtr, FamPtr, Count)>
// CHECK:         | |   |-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK:         | |   |   | | |-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'struct flexible *__single'
// CHECK:         | |   |-OpaqueValueExpr [[ove_2]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK:         | |   |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK:         | |   | `-MemberExpr {{.+}} ->elems
// CHECK:         | |   |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK:         | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:         | |     `-MemberExpr {{.+}} ->count
// CHECK:         | |       `-OpaqueValueExpr [[ove_2]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK:         | `-OpaqueValueExpr [[ove_2]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK:         `-OpaqueValueExpr [[ove_2]]
// CHECK:           `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:             |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:             | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK:             | | |-OpaqueValueExpr [[ove_3]] {{.*}} 'struct flexible *__single'
// CHECK:             | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK:             | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK:             | | | | `-MemberExpr {{.+}} ->elems
// CHECK:             | | | |   `-OpaqueValueExpr [[ove_3]] {{.*}} 'struct flexible *__single'
// CHECK:             | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:             | | |   `-MemberExpr {{.+}} ->count
// CHECK:             | | |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'struct flexible *__single'
// CHECK:             | `-OpaqueValueExpr [[ove_3]]
// CHECK:             |   `-ImplicitCastExpr {{.+}} 'struct flexible *__single' <LValueToRValue>
// CHECK:             |     `-DeclRefExpr {{.+}} [[var_flex_1]]
// CHECK:             `-OpaqueValueExpr [[ove_3]] {{.*}} 'struct flexible *__single'

// CHECK-LABEL: double_promote_to_bidi_indexable
void double_promote_to_bidi_indexable(struct flexible *__sized_by(size) flex, int size) {
  struct flexible *b = flex;
}
// CHECK: |-ParmVarDecl [[var_flex_2:0x[^ ]+]]
// CHECK: |-ParmVarDecl [[var_size:0x[^ ]+]]
// CHECK: | `-DependerDeclsAttr
// CHECK: `-CompoundStmt
// CHECK:   `-DeclStmt
// CHECK:     `-VarDecl [[var_b_1:0x[^ ]+]]
// CHECK:       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK:         | | |-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single'
// CHECK:         | | |-ImplicitCastExpr {{.+}} 'struct flexible *' <BitCast>
// CHECK:         | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK:         | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK:         | | |   | `-ImplicitCastExpr {{.+}} 'struct flexible *' <BoundsSafetyPointerCast>
// CHECK:         | | |   |   `-OpaqueValueExpr [[ove_4]] {{.*}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single'
// CHECK:         | | |   `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'int'
// CHECK:         | |-OpaqueValueExpr [[ove_4]]
// CHECK:         | | `-ImplicitCastExpr {{.+}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single' <LValueToRValue>
// CHECK:         | |   `-DeclRefExpr {{.+}} [[var_flex_2]]
// CHECK:         | `-OpaqueValueExpr [[ove_5]]
// CHECK:         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:         |     `-DeclRefExpr {{.+}} [[var_size]]
// CHECK:         |-OpaqueValueExpr [[ove_4]] {{.*}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single'
// CHECK:         `-OpaqueValueExpr [[ove_5]] {{.*}} 'int'

// CHECK-LABEL: promote_to_sized_by
void promote_to_sized_by(struct flexible *flex) {
  unsigned long long siz;
  struct flexible *__sized_by(siz) s;

  siz = sizeof(struct flexible) + sizeof(int) * flex->count;
  s = flex;
}
// CHECK: |-ParmVarDecl [[var_flex_3:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   |-DeclStmt
// CHECK:   | `-VarDecl [[var_siz:0x[^ ]+]]
// CHECK:   |   `-DependerDeclsAttr
// CHECK:   |-DeclStmt
// CHECK:   | `-VarDecl [[var_s_1:0x[^ ]+]]
// CHECK:   |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   |   | |-BoundsCheckExpr {{.+}} 'flex <= __builtin_get_pointer_upper_bound(flex) && __builtin_get_pointer_lower_bound(flex) <= flex && sizeof(struct flexible) + sizeof(int) * flex->count <= (char *)__builtin_get_pointer_upper_bound(flex) - (char *__bidi_indexable)flex'
// CHECK:   |   | | |-BinaryOperator {{.+}} 'unsigned long long' '='
// CHECK:   |   | | | |-DeclRefExpr {{.+}} [[var_siz]]
// CHECK:   |   | | | `-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'unsigned long long'
// CHECK:   |   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   |   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   |   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:   |   | |   | | |-ImplicitCastExpr {{.+}} 'struct flexible *' <BoundsSafetyPointerCast>
// CHECK:   |   | |   | | | `-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK:   |   | |   | | |     | | |-OpaqueValueExpr [[ove_8:0x[^ ]+]] {{.*}} 'struct flexible *__single'
// CHECK:   |   | |   | | `-GetBoundExpr {{.+}} upper
// CHECK:   |   | |   | |   `-OpaqueValueExpr [[ove_7]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK:   |   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   |   | |   |   |-GetBoundExpr {{.+}} lower
// CHECK:   |   | |   |   | `-OpaqueValueExpr [[ove_7]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK:   |   | |   |   `-ImplicitCastExpr {{.+}} 'struct flexible *' <BoundsSafetyPointerCast>
// CHECK:   |   | |   |     `-OpaqueValueExpr [[ove_7]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK:   |   | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   |   | |     |-OpaqueValueExpr [[ove_6]] {{.*}} 'unsigned long long'
// CHECK:   |   | |     `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK:   |   | |       `-BinaryOperator {{.+}} 'long' '-'
// CHECK:   |   | |         |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK:   |   | |         | `-GetBoundExpr {{.+}} upper
// CHECK:   |   | |         |   `-OpaqueValueExpr [[ove_7]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK:   |   | |         `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:   |   | |           `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:   |   | |             `-OpaqueValueExpr [[ove_7]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK:   |   | |-OpaqueValueExpr [[ove_6]]
// CHECK:   |   | | `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK:   |   | |   `-BinaryOperator {{.+}} 'unsigned long' '+'
// CHECK:   |   | |     |-UnaryExprOrTypeTraitExpr
// CHECK:   |   | |     `-BinaryOperator {{.+}} 'unsigned long' '*'
// CHECK:   |   | |       |-UnaryExprOrTypeTraitExpr
// CHECK:   |   | |       `-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK:   |   | |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:   |   | |           `-MemberExpr {{.+}} ->count
// CHECK:   |   | |             `-ImplicitCastExpr {{.+}} 'struct flexible *__single' <LValueToRValue>
// CHECK:   |   | |               `-DeclRefExpr {{.+}} [[var_flex_3]]
// CHECK:   |   | `-OpaqueValueExpr [[ove_7]]
// CHECK:   |   |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:   |   |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK:   |   |     | | |-OpaqueValueExpr [[ove_8]] {{.*}} 'struct flexible *__single'
// CHECK:   |   |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK:   |   |     | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK:   |   |     | | | | `-MemberExpr {{.+}} ->elems
// CHECK:   |   |     | | | |   `-OpaqueValueExpr [[ove_8]] {{.*}} 'struct flexible *__single'
// CHECK:   |   |     | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:   |   |     | | |   `-MemberExpr {{.+}} ->count
// CHECK:   |   |     | | |     `-OpaqueValueExpr [[ove_8]] {{.*}} 'struct flexible *__single'
// CHECK:   |   |     | `-OpaqueValueExpr [[ove_8]]
// CHECK:   |   |     |   `-ImplicitCastExpr {{.+}} 'struct flexible *__single' <LValueToRValue>
// CHECK:   |   |     |     `-DeclRefExpr {{.+}} [[var_flex_3]]
// CHECK:   |   |     `-OpaqueValueExpr [[ove_8]] {{.*}} 'struct flexible *__single'
// CHECK:   |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:   |     |-BinaryOperator {{.+}} 'struct flexible *__single __sized_by(siz)':'struct flexible *__single' '='
// CHECK:   |     | |-DeclRefExpr {{.+}} [[var_s_1]]
// CHECK:   |     | `-ImplicitCastExpr {{.+}} 'struct flexible *__single __sized_by(siz)':'struct flexible *__single' <BoundsSafetyPointerCast>
// CHECK:   |     |   `-OpaqueValueExpr [[ove_7]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK:   |     |-OpaqueValueExpr [[ove_6]] {{.*}} 'unsigned long long'
// CHECK:   |     `-OpaqueValueExpr [[ove_7]] {{.*}} 'struct flexible *__bidi_indexable'

// CHECK-LABEL: promote_to_single_assign
void promote_to_single_assign(struct flexible *flex) {
  struct flexible *__single s = flex;
  s->count = flex->count;
}
// CHECK: |-ParmVarDecl [[var_flex_4:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   |-DeclStmt
// CHECK:   | `-VarDecl [[var_s_2:0x[^ ]+]]
// CHECK:   |   `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   |     |-ImplicitCastExpr {{.+}} 'struct flexible *__single' <BoundsSafetyPointerCast>
// CHECK:   |     | `-PredefinedBoundsCheckExpr {{.+}} 'struct flexible *__bidi_indexable' <FlexibleArrayCountAssign(BasePtr, FamPtr, Count)>
// CHECK:   |     |   |-OpaqueValueExpr [[ove_9:0x[^ ]+]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK:   |     |   |   | | |-OpaqueValueExpr [[ove_10:0x[^ ]+]] {{.*}} 'struct flexible *__single'
// CHECK:   |     |   |-OpaqueValueExpr [[ove_9]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK:   |     |   |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK:   |     |   | `-MemberExpr {{.+}} ->elems
// CHECK:   |     |   |   `-OpaqueValueExpr [[ove_9]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK:   |     |   `-OpaqueValueExpr [[ove_11:0x[^ ]+]] {{.*}} 'int'
// CHECK:   |     |-OpaqueValueExpr [[ove_9]]
// CHECK:   |     | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:   |     |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   |     |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK:   |     |   | | |-OpaqueValueExpr [[ove_10]] {{.*}} 'struct flexible *__single'
// CHECK:   |     |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK:   |     |   | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK:   |     |   | | | | `-MemberExpr {{.+}} ->elems
// CHECK:   |     |   | | | |   `-OpaqueValueExpr [[ove_10]] {{.*}} 'struct flexible *__single'
// CHECK:   |     |   | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:   |     |   | | |   `-MemberExpr {{.+}} ->count
// CHECK:   |     |   | | |     `-OpaqueValueExpr [[ove_10]] {{.*}} 'struct flexible *__single'
// CHECK:   |     |   | `-OpaqueValueExpr [[ove_10]]
// CHECK:   |     |   |   `-ImplicitCastExpr {{.+}} 'struct flexible *__single' <LValueToRValue>
// CHECK:   |     |   |     `-DeclRefExpr {{.+}} [[var_flex_4]]
// CHECK:   |     |   `-OpaqueValueExpr [[ove_10]] {{.*}} 'struct flexible *__single'
// CHECK:   |     `-OpaqueValueExpr [[ove_11]]
// CHECK:   |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:   |         `-MemberExpr {{.+}} ->count
// CHECK:   |           `-ImplicitCastExpr {{.+}} 'struct flexible *__single' <LValueToRValue>
// CHECK:   |             `-DeclRefExpr {{.+}} [[var_flex_4]]
// CHECK:   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     |-BinaryOperator {{.+}} 'int' '='
// CHECK:     | |-MemberExpr {{.+}} ->count
// CHECK:     | | `-ImplicitCastExpr {{.+}} 'struct flexible *__single' <LValueToRValue>
// CHECK:     | |   `-DeclRefExpr {{.+}} [[var_s_2]]
// CHECK:     | `-OpaqueValueExpr [[ove_11]] {{.*}} 'int'
// CHECK:     |-OpaqueValueExpr [[ove_9]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK:     `-OpaqueValueExpr [[ove_11]] {{.*}} 'int'
