

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s

#include <ptrcheck.h>

typedef struct {
  int count;
  int elems[__counted_by(count)];
} flex_inner_t;

typedef struct {
  unsigned dummy;
  flex_inner_t flex;
} flex_t;


// CHECK-LABEL: test_fam_base
void test_fam_base(flex_t *f, void *__bidi_indexable buf) {
  f = buf;
}
// CHECK: |-ParmVarDecl [[var_f:0x[^ ]+]]
// CHECK: |-ParmVarDecl [[var_buf:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     | |-BinaryOperator {{.+}} 'flex_t *__single' '='
// CHECK:     | | |-DeclRefExpr {{.+}} [[var_f]]
// CHECK:     | | `-ImplicitCastExpr {{.+}} 'flex_t *__single' <BoundsSafetyPointerCast>
// CHECK:     | |   `-PredefinedBoundsCheckExpr {{.+}} 'flex_t *__bidi_indexable' <FlexibleArrayCountAssign(BasePtr, FamPtr, Count)>
// CHECK:     | |     |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'flex_t *__bidi_indexable'
// CHECK:     | |     |-OpaqueValueExpr [[ove]] {{.*}} 'flex_t *__bidi_indexable'
// CHECK:     | |     |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK:     | |     | `-MemberExpr {{.+}} .elems
// CHECK:     | |     |   `-MemberExpr {{.+}} ->flex
// CHECK:     | |     |     `-OpaqueValueExpr [[ove]] {{.*}} 'flex_t *__bidi_indexable'
// CHECK:     | |     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     | |       `-MemberExpr {{.+}} .count
// CHECK:     | |         `-MemberExpr {{.+}} ->flex
// CHECK:     | |           `-OpaqueValueExpr [[ove]] {{.*}} 'flex_t *__bidi_indexable'
// CHECK:     | `-OpaqueValueExpr [[ove]] {{.*}} 'flex_t *__bidi_indexable'
// CHECK:     `-OpaqueValueExpr [[ove]]
// CHECK:       `-ImplicitCastExpr {{.+}} 'flex_t *__bidi_indexable' <BitCast>
// CHECK:         `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <LValueToRValue>
// CHECK:           `-DeclRefExpr {{.+}} [[var_buf]]

// CHECK-LABEL: test_fam_base_with_count
void test_fam_base_with_count(flex_t *f, void *__bidi_indexable buf) {
  f = buf;
  f->flex.count = 10;
}

// CHECK: |-ParmVarDecl [[var_f_1:0x[^ ]+]]
// CHECK: |-ParmVarDecl [[var_buf_1:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   | |-BinaryOperator {{.+}} 'flex_t *__single' '='
// CHECK:   | | |-DeclRefExpr {{.+}} [[var_f_1]]
// CHECK:   | | `-ImplicitCastExpr {{.+}} 'flex_t *__single' <BoundsSafetyPointerCast>
// CHECK:   | |   `-PredefinedBoundsCheckExpr {{.+}} 'flex_t *__bidi_indexable' <FlexibleArrayCountAssign(BasePtr, FamPtr, Count)>
// CHECK:   | |     |-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'flex_t *__bidi_indexable'
// CHECK:   | |     |-OpaqueValueExpr [[ove_1]] {{.*}} 'flex_t *__bidi_indexable'
// CHECK:   | |     |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK:   | |     | `-MemberExpr {{.+}} .elems
// CHECK:   | |     |   `-MemberExpr {{.+}} ->flex
// CHECK:   | |     |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'flex_t *__bidi_indexable'
// CHECK:   | |     `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int'
// CHECK:   | |-OpaqueValueExpr [[ove_1]]
// CHECK:   | | `-ImplicitCastExpr {{.+}} 'flex_t *__bidi_indexable' <BitCast>
// CHECK:   | |   `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <LValueToRValue>
// CHECK:   | |     `-DeclRefExpr {{.+}} [[var_buf_1]]
// CHECK:   | `-OpaqueValueExpr [[ove_2]]
// CHECK:   |   `-IntegerLiteral {{.+}} 10
// CHECK:   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     |-BinaryOperator {{.+}} 'int' '='
// CHECK:     | |-MemberExpr {{.+}} .count
// CHECK:     | | `-MemberExpr {{.+}} ->flex
// CHECK:     | |   `-ImplicitCastExpr {{.+}} 'flex_t *__single' <LValueToRValue>
// CHECK:     | |     `-DeclRefExpr {{.+}} [[var_f_1]]
// CHECK:     | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int'
// CHECK:     |-OpaqueValueExpr [[ove_1]] {{.*}} 'flex_t *__bidi_indexable'
// CHECK:     `-OpaqueValueExpr [[ove_2]] {{.*}} 'int'


// CHECK-LABEL: test_fam_base_init
void test_fam_base_init(void *__bidi_indexable buf) {
  flex_t *__single f = buf;
}
// CHECK: |-ParmVarDecl [[var_buf_2:0x[^ ]+]]
// CHECK:   `-CompoundStmt
// CHECK:     `-DeclStmt
// CHECK:       `-VarDecl [[var_f_2:0x[^ ]+]]
// CHECK:         `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:           |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:           | |-ImplicitCastExpr {{.+}} 'flex_t *__single' <BoundsSafetyPointerCast>
// CHECK:           | | `-PredefinedBoundsCheckExpr {{.+}} 'flex_t *__bidi_indexable' <FlexibleArrayCountAssign(BasePtr, FamPtr, Count)>
// CHECK:           | |   |-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'flex_t *__bidi_indexable'
// CHECK:           | |   |-OpaqueValueExpr [[ove_3]] {{.*}} 'flex_t *__bidi_indexable'
// CHECK:           | |   |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK:           | |   | `-MemberExpr {{.+}} .elems
// CHECK:           | |   |   `-MemberExpr {{.+}} ->flex
// CHECK:           | |   |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'flex_t *__bidi_indexable'
// CHECK:           | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:           | |     `-MemberExpr {{.+}} .count
// CHECK:           | |       `-MemberExpr {{.+}} ->flex
// CHECK:           | |         `-OpaqueValueExpr [[ove_3]] {{.*}} 'flex_t *__bidi_indexable'
// CHECK:           | `-OpaqueValueExpr [[ove_3]] {{.*}} 'flex_t *__bidi_indexable'
// CHECK:           `-OpaqueValueExpr [[ove_3]]
// CHECK:             `-ImplicitCastExpr {{.+}} 'flex_t *__bidi_indexable' <BitCast>
// CHECK:               `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <LValueToRValue>
// CHECK:                 `-DeclRefExpr {{.+}} [[var_buf_2]]

// CHECK-LABEL: test_fam_base_init_with_count
void test_fam_base_init_with_count(void *__bidi_indexable buf) {
  flex_t *__single f = buf;
  f->flex.count = 10;
}
// CHECK: |-ParmVarDecl [[var_buf_3:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   |-DeclStmt
// CHECK:   | `-VarDecl [[var_f_3:0x[^ ]+]]
// CHECK:   |   `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   |     |-ImplicitCastExpr {{.+}} 'flex_t *__single' <BoundsSafetyPointerCast>
// CHECK:   |     | `-PredefinedBoundsCheckExpr {{.+}} 'flex_t *__bidi_indexable' <FlexibleArrayCountAssign(BasePtr, FamPtr, Count)>
// CHECK:   |     |   |-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'flex_t *__bidi_indexable'
// CHECK:   |     |   |-OpaqueValueExpr [[ove_4]] {{.*}} 'flex_t *__bidi_indexable'
// CHECK:   |     |   |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK:   |     |   | `-MemberExpr {{.+}} .elems
// CHECK:   |     |   |   `-MemberExpr {{.+}} ->flex
// CHECK:   |     |   |     `-OpaqueValueExpr [[ove_4]] {{.*}} 'flex_t *__bidi_indexable'
// CHECK:   |     |   `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'int'
// CHECK:   |     |-OpaqueValueExpr [[ove_4]]
// CHECK:   |     | `-ImplicitCastExpr {{.+}} 'flex_t *__bidi_indexable' <BitCast>
// CHECK:   |     |   `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <LValueToRValue>
// CHECK:   |     |     `-DeclRefExpr {{.+}} [[var_buf_3]]
// CHECK:   |     `-OpaqueValueExpr [[ove_5]]
// CHECK:   |       `-IntegerLiteral {{.+}} 10
// CHECK:   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     |-BinaryOperator {{.+}} 'int' '='
// CHECK:     | |-MemberExpr {{.+}} .count
// CHECK:     | | `-MemberExpr {{.+}} ->flex
// CHECK:     | |   `-ImplicitCastExpr {{.+}} 'flex_t *__single' <LValueToRValue>
// CHECK:     | |     `-DeclRefExpr {{.+}} [[var_f_3]]
// CHECK:     | `-OpaqueValueExpr [[ove_5]] {{.*}} 'int'
// CHECK:     |-OpaqueValueExpr [[ove_4]] {{.*}} 'flex_t *__bidi_indexable'
// CHECK:     `-OpaqueValueExpr [[ove_5]] {{.*}} 'int'


// FIXME: rdar://84810920
// CHECK-LABEL: test_fam_base_init_deref_with_count
void test_fam_base_init_deref_with_count(void *__bidi_indexable buf) {
  flex_t *__single f = buf;
  (*f).flex.count = 10;
}
// CHECK: |-ParmVarDecl [[var_buf_4:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   |-DeclStmt
// CHECK:   | `-VarDecl [[var_f_4:0x[^ ]+]]
// CHECK:   |   `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   |     |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:   |     | |-ImplicitCastExpr {{.+}} 'flex_t *__single' <BoundsSafetyPointerCast>
// CHECK:   |     | | `-PredefinedBoundsCheckExpr {{.+}} 'flex_t *__bidi_indexable' <FlexibleArrayCountAssign(BasePtr, FamPtr, Count)>
// CHECK:   |     | |   |-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'flex_t *__bidi_indexable'
// CHECK:   |     | |   |-OpaqueValueExpr [[ove_6]] {{.*}} 'flex_t *__bidi_indexable'
// CHECK:   |     | |   |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK:   |     | |   | `-MemberExpr {{.+}} .elems
// CHECK:   |     | |   |   `-MemberExpr {{.+}} ->flex
// CHECK:   |     | |   |     `-OpaqueValueExpr [[ove_6]] {{.*}} 'flex_t *__bidi_indexable'
// CHECK:   |     | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:   |     | |     `-MemberExpr {{.+}} .count
// CHECK:   |     | |       `-MemberExpr {{.+}} ->flex
// CHECK:   |     | |         `-OpaqueValueExpr [[ove_6]] {{.*}} 'flex_t *__bidi_indexable'
// CHECK:   |     | `-OpaqueValueExpr [[ove_6]] {{.*}} 'flex_t *__bidi_indexable'
// CHECK:   |     `-OpaqueValueExpr [[ove_6]]
// CHECK:   |       `-ImplicitCastExpr {{.+}} 'flex_t *__bidi_indexable' <BitCast>
// CHECK:   |         `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <LValueToRValue>
// CHECK:   |           `-DeclRefExpr {{.+}} [[var_buf_4]]
// CHECK:   `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     | |-BoundsCheckExpr {{.+}} '10 <= (*f).flex.count && 0 <= 10'
// FIXME: Ignoring the rest of AST for now. It seems file check doesn't like
// the dump produced by our ast simplfier for this function. rdar://103050286

// CHECK: VarDecl [[var_g_flex:0x[^ ]+]]
flex_inner_t g_flex = {4, {1, 2, 3, 4}};

// CHECK-LABEL: test_fam_lvalue_base_count_assign
void test_fam_lvalue_base_count_assign(unsigned arg) {
  g_flex.count = arg;
}
// CHECK: | |-ParmVarDecl [[var_arg:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |     |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |     | |-BoundsCheckExpr {{.+}} 'arg <= g_flex.count && 0 <= arg'
// CHECK: |     | | |-BinaryOperator {{.+}} 'int' '='
// CHECK: |     | | | |-MemberExpr {{.+}} .count
// CHECK: |     | | | | `-OpaqueValueExpr [[ove_17:0x[^ ]+]] {{.*}} lvalue
// CHECK: |     | | | `-OpaqueValueExpr [[ove_18:0x[^ ]+]] {{.*}} 'int'
// CHECK: |     | | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |     | | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |     | | | | |-OpaqueValueExpr [[ove_18]] {{.*}} 'int'
// CHECK: |     | | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |     | | | |   `-MemberExpr {{.+}} .count
// CHECK: |     | | | |     `-OpaqueValueExpr [[ove_17]] {{.*}} lvalue
// CHECK: |     | | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |     | | |   |-IntegerLiteral {{.+}} 0
// CHECK: |     | | |   `-OpaqueValueExpr [[ove_18]] {{.*}} 'int'
// CHECK: |     | | `-OpaqueValueExpr [[ove_17]]
// CHECK: |     | |   `-DeclRefExpr {{.+}} [[var_g_flex]]
// CHECK: |     | `-OpaqueValueExpr [[ove_18]] {{.*}} 'int'
// CHECK: |     `-OpaqueValueExpr [[ove_18]]
// CHECK: |       `-ImplicitCastExpr {{.+}} 'int' <IntegralCast>
// CHECK: |         `-ImplicitCastExpr {{.+}} 'unsigned int' <LValueToRValue>
// CHECK: |           `-DeclRefExpr {{.+}} [[var_arg]]

// CHECK-LABEL: test_fam_lvalue_base_count_decrement
void test_fam_lvalue_base_count_decrement() {
  g_flex.count--;
}
// CHECK: | `-CompoundStmt
// CHECK: |   `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |     |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |     | |-BoundsCheckExpr
// CHECK: |     | | |-UnaryOperator {{.+}} postfix '--'
// CHECK: |     | | | `-OpaqueValueExpr [[ove_19:0x[^ ]+]] {{.*}} lvalue
// CHECK: |     | | |     `-OpaqueValueExpr [[ove_20:0x[^ ]+]] {{.*}} lvalue
// CHECK: |     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |     | |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |     | |   | |-OpaqueValueExpr [[ove_21:0x[^ ]+]] {{.*}} 'int'
// CHECK: |     | |   | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |     | |   |   `-MemberExpr {{.+}} .count
// CHECK: |     | |   |     `-OpaqueValueExpr [[ove_20]] {{.*}} lvalue
// CHECK: |     | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |     | |     |-IntegerLiteral {{.+}} 0
// CHECK: |     | |     `-OpaqueValueExpr [[ove_21]] {{.*}} 'int'
// CHECK: |     | |-OpaqueValueExpr [[ove_20]] {{.*}} lvalue
// CHECK: |     | |-OpaqueValueExpr [[ove_19]] {{.*}} lvalue
// CHECK: |     | `-OpaqueValueExpr [[ove_21]] {{.*}} 'int'
// CHECK: |     |-OpaqueValueExpr [[ove_20]]
// CHECK: |     | `-DeclRefExpr {{.+}} [[var_g_flex]]
// CHECK: |     |-OpaqueValueExpr [[ove_19]]
// CHECK: |     | `-MemberExpr {{.+}} .count
// CHECK: |     |   `-OpaqueValueExpr [[ove_20]] {{.*}} lvalue
// CHECK: |     `-OpaqueValueExpr [[ove_21]]
// CHECK: |       `-BinaryOperator {{.+}} 'int' '-'
// CHECK: |         |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |         | `-OpaqueValueExpr [[ove_19]] {{.*}} lvalue
// CHECK: |         `-IntegerLiteral {{.+}} 1

// CHECK-LABEL: test_fam_lvalue_base_count_compound
void test_fam_lvalue_base_count_compound(unsigned arg) {
  g_flex.count -= arg;
}
// CHECK: | |-ParmVarDecl [[var_arg_1:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |     |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |     | |-BoundsCheckExpr
// CHECK: |     | | |-CompoundAssignOperator {{.+}} ComputeLHSTy='unsigned int'
// CHECK: |     | | | |-OpaqueValueExpr [[ove_22:0x[^ ]+]] {{.*}} lvalue
// CHECK: |     | | | |   `-OpaqueValueExpr [[ove_23:0x[^ ]+]] {{.*}} lvalue
// CHECK: |     | | | `-OpaqueValueExpr [[ove_24:0x[^ ]+]] {{.*}} 'unsigned int'
// CHECK: |     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |     | |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |     | |   | |-OpaqueValueExpr [[ove_25:0x[^ ]+]] {{.*}} 'unsigned int'
// CHECK: |     | |   | `-ImplicitCastExpr {{.+}} 'unsigned int' <IntegralCast>
// CHECK: |     | |   |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |     | |   |     `-MemberExpr {{.+}} .count
// CHECK: |     | |   |       `-OpaqueValueExpr [[ove_23]] {{.*}} lvalue
// CHECK: |     | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |     | |     |-ImplicitCastExpr {{.+}} 'unsigned int' <IntegralCast>
// CHECK: |     | |     | `-IntegerLiteral {{.+}} 0
// CHECK: |     | |     `-OpaqueValueExpr [[ove_25]] {{.*}} 'unsigned int'
// CHECK: |     | |-OpaqueValueExpr [[ove_24]] {{.*}} 'unsigned int'
// CHECK: |     | |-OpaqueValueExpr [[ove_23]] {{.*}} lvalue
// CHECK: |     | |-OpaqueValueExpr [[ove_22]] {{.*}} lvalue
// CHECK: |     | `-OpaqueValueExpr [[ove_25]] {{.*}} 'unsigned int'
// CHECK: |     |-OpaqueValueExpr [[ove_24]]
// CHECK: |     | `-ImplicitCastExpr {{.+}} 'unsigned int' <LValueToRValue>
// CHECK: |     |   `-DeclRefExpr {{.+}} [[var_arg_1]]
// CHECK: |     |-OpaqueValueExpr [[ove_23]]
// CHECK: |     | `-DeclRefExpr {{.+}} [[var_g_flex]]
// CHECK: |     |-OpaqueValueExpr [[ove_22]]
// CHECK: |     | `-MemberExpr {{.+}} .count
// CHECK: |     |   `-OpaqueValueExpr [[ove_23]] {{.*}} lvalue
// CHECK: |     `-OpaqueValueExpr [[ove_25]]
// CHECK: |       `-BinaryOperator {{.+}} 'unsigned int' '-'
// CHECK: |         |-ImplicitCastExpr {{.+}} 'unsigned int' <IntegralCast>
// CHECK: |         | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |         |   `-OpaqueValueExpr [[ove_22]] {{.*}} lvalue
// CHECK: |         `-OpaqueValueExpr [[ove_24]] {{.*}} 'unsigned int'

typedef struct {
  unsigned char count;
  int elems[__counted_by(count - 1)];
} flex_uchar_t;

void test_flex_uchar_count_conversion(flex_uchar_t *flex, int arg) {
  flex = flex;
  flex->count = arg;
}
// CHECK: |-FunctionDecl [[func_test_flex_uchar_count_conversion:0x[^ ]+]] {{.+}} test_flex_uchar_count_conversion
// CHECK: | |-ParmVarDecl [[var_flex:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_arg_2:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   | |-BinaryOperator {{.+}} 'flex_uchar_t *__single' '='
// CHECK: |   | | |-DeclRefExpr {{.+}} [[var_flex]]
// CHECK: |   | | `-ImplicitCastExpr {{.+}} 'flex_uchar_t *__single' <BoundsSafetyPointerCast>
// CHECK: |   | |   `-PredefinedBoundsCheckExpr {{.+}} 'flex_uchar_t *__bidi_indexable' <FlexibleArrayCountAssign(BasePtr, FamPtr, Count)>
// CHECK: |   | |     |-OpaqueValueExpr [[ove_25:0x[^ ]+]] {{.*}} 'flex_uchar_t *__bidi_indexable'
// CHECK: |   | |     |   | | |-OpaqueValueExpr [[ove_26:0x[^ ]+]] {{.*}} 'flex_uchar_t *__single'
// CHECK: |   | |     |-OpaqueValueExpr [[ove_25]] {{.*}} 'flex_uchar_t *__bidi_indexable'
// CHECK: |   | |     |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: |   | |     | `-MemberExpr {{.+}} ->elems
// CHECK: |   | |     |   `-OpaqueValueExpr [[ove_25]] {{.*}} 'flex_uchar_t *__bidi_indexable'
// CHECK: |   | |     `-BinaryOperator {{.+}} 'int' '-'
// CHECK: |   | |       |-ImplicitCastExpr {{.+}} 'int' <IntegralCast>
// CHECK: |   | |       | `-OpaqueValueExpr [[ove_27:0x[^ ]+]] {{.*}} 'unsigned char'
// CHECK: |   | |       `-IntegerLiteral {{.+}} 1
// CHECK: |   | |-OpaqueValueExpr [[ove_25]]
// CHECK: |   | | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   | |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   | |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'flex_uchar_t *__bidi_indexable'
// CHECK: |   | |   | | |-OpaqueValueExpr [[ove_26]] {{.*}} 'flex_uchar_t *__single'
// CHECK: |   | |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: |   | |   | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: |   | |   | | | | `-MemberExpr {{.+}} ->elems
// CHECK: |   | |   | | | |   `-OpaqueValueExpr [[ove_26]] {{.*}} 'flex_uchar_t *__single'
// CHECK: |   | |   | | |  `-BinaryOperator {{.+}} 'int' '-'
// CHECK: |   | |   | | |    |-ImplicitCastExpr {{.+}} 'int' <IntegralCast>
// CHECK: |   | |   | | |    | `-ImplicitCastExpr {{.+}} 'unsigned char' <LValueToRValue>
// CHECK: |   | |   | | |    |   `-MemberExpr {{.+}} ->count
// CHECK: |   | |   | | |    |     `-OpaqueValueExpr [[ove_26]] {{.*}} 'flex_uchar_t *__single'
// CHECK: |   | |   | | |     `-IntegerLiteral {{.+}} 1
// CHECK: |   | |   | `-OpaqueValueExpr [[ove_26]]
// CHECK: |   | |   |   `-ImplicitCastExpr {{.+}} 'flex_uchar_t *__single' <LValueToRValue>
// CHECK: |   | |   |     `-DeclRefExpr {{.+}} [[var_flex]]
// CHECK: |   | |   `-OpaqueValueExpr [[ove_26]] {{.*}} 'flex_uchar_t *__single'
// CHECK: |   | `-OpaqueValueExpr [[ove_27]]
// CHECK: |   |   `-ImplicitCastExpr {{.+}} 'unsigned char' <IntegralCast>
// CHECK: |   |     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   |       `-DeclRefExpr {{.+}} [[var_arg_2]]
// CHECK: |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |     |-BinaryOperator {{.+}} 'unsigned char' '='
// CHECK: |     | |-MemberExpr {{.+}} ->count
// CHECK: |     | | `-ImplicitCastExpr {{.+}} 'flex_uchar_t *__single' <LValueToRValue>
// CHECK: |     | |   `-DeclRefExpr {{.+}} [[var_flex]]
// CHECK: |     | `-OpaqueValueExpr [[ove_27]] {{.*}} 'unsigned char'
// CHECK: |     |-OpaqueValueExpr [[ove_25]] {{.*}} 'flex_uchar_t *__bidi_indexable'
// CHECK: |     `-OpaqueValueExpr [[ove_27]] {{.*}} 'unsigned char'
;
