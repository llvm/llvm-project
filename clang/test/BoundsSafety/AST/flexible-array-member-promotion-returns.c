

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s

#include <ptrcheck.h>

struct flexible {
    int count;
    int elems[__counted_by(count)];
};

// CHECK: FunctionDecl [[func_return_flex:0x[^ ]+]] {{.+}} return_flex
struct flexible *return_flex();

// CHECK-LABEL: single_init
void single_init() {
  struct flexible *__single s = return_flex();
}
// CHECK: `-CompoundStmt
// CHECK:   `-DeclStmt
// CHECK:     `-VarDecl [[var_s:0x[^ ]+]]
// CHECK:       `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:         |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:         | |-ImplicitCastExpr {{.+}} 'struct flexible *__single' <BoundsSafetyPointerCast>
// CHECK:         | | `-PredefinedBoundsCheckExpr {{.+}} 'struct flexible *__bidi_indexable' <FlexibleArrayCountAssign(BasePtr, FamPtr, Count)>
// CHECK:         | |   |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK:         | |   |   | | |-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'struct flexible *__single'
// CHECK:         | |   |-OpaqueValueExpr [[ove]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK:         | |   |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK:         | |   | `-MemberExpr {{.+}} ->elems
// CHECK:         | |   |   `-OpaqueValueExpr [[ove]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK:         | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:         | |     `-MemberExpr {{.+}} ->count
// CHECK:         | |       `-OpaqueValueExpr [[ove]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK:         | `-OpaqueValueExpr [[ove]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK:         `-OpaqueValueExpr [[ove]]
// CHECK:           `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:             |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:             | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK:             | | |-OpaqueValueExpr [[ove_1]] {{.*}} 'struct flexible *__single'
// CHECK:             | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK:             | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK:             | | | | `-MemberExpr {{.+}} ->elems
// CHECK:             | | | |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'struct flexible *__single'
// CHECK:             | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:             | | |   `-MemberExpr {{.+}} ->count
// CHECK:             | | |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'struct flexible *__single'
// CHECK:             | `-OpaqueValueExpr [[ove_1]]
// CHECK:             |   `-CallExpr
// CHECK:             |     `-ImplicitCastExpr {{.+}} 'struct flexible *__single(*__single)()' <FunctionToPointerDecay>
// CHECK:             |       `-DeclRefExpr {{.+}} [[func_return_flex]]
// CHECK:             `-OpaqueValueExpr [[ove_1]] {{.*}} 'struct flexible *__single'


// CHECK-LABEL: single_assign
void single_assign() {
  struct flexible *__single s;
  s = return_flex();
}
// CHECK: CompoundStmt
// CHECK: |-DeclStmt
// CHECK: | `-VarDecl [[var_s_1:0x[^ ]+]]
// CHECK: `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:   | |-BinaryOperator {{.+}} 'struct flexible *__single' '='
// CHECK:   | | |-DeclRefExpr {{.+}} [[var_s_1]]
// CHECK:   | | `-ImplicitCastExpr {{.+}} 'struct flexible *__single' <BoundsSafetyPointerCast>
// CHECK:   | |   `-PredefinedBoundsCheckExpr {{.+}} 'struct flexible *__bidi_indexable' <FlexibleArrayCountAssign(BasePtr, FamPtr, Count)>
// CHECK:   | |     |-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK:   | |     |   | | |-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'struct flexible *__single'
// CHECK:   | |     |-OpaqueValueExpr [[ove_2]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK:   | |     |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK:   | |     | `-MemberExpr {{.+}} ->elems
// CHECK:   | |     |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK:   | |     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:   | |       `-MemberExpr {{.+}} ->count
// CHECK:   | |         `-OpaqueValueExpr [[ove_2]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK:   | `-OpaqueValueExpr [[ove_2]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK:   `-OpaqueValueExpr [[ove_2]]
// CHECK:     `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:       |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK:       | | |-OpaqueValueExpr [[ove_3]] {{.*}} 'struct flexible *__single'
// CHECK:       | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK:       | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK:       | | | | `-MemberExpr {{.+}} ->elems
// CHECK:       | | | |   `-OpaqueValueExpr [[ove_3]] {{.*}} 'struct flexible *__single'
// CHECK:       | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:       | | |   `-MemberExpr {{.+}} ->count
// CHECK:       | | |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'struct flexible *__single'
// CHECK:       | `-OpaqueValueExpr [[ove_3]]
// CHECK:       |   `-CallExpr
// CHECK:       |     `-ImplicitCastExpr {{.+}} 'struct flexible *__single(*__single)()' <FunctionToPointerDecay>
// CHECK:       |       `-DeclRefExpr {{.+}} [[func_return_flex]]
// CHECK:       `-OpaqueValueExpr [[ove_3]] {{.*}} 'struct flexible *__single'

// CHECK-LABEL: single_assign_with_count
void single_assign_with_count() {
  struct flexible *__single s;
  s = return_flex();
  s->count = 10;
}
// CHECK: CompoundStmt
// CHECK: |-DeclStmt
// CHECK: | `-VarDecl [[var_s_2:0x[^ ]+]]
// CHECK: |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: | |-BinaryOperator {{.+}} 'struct flexible *__single' '='
// CHECK: | | |-DeclRefExpr {{.+}} [[var_s_2]]
// CHECK: | | `-ImplicitCastExpr {{.+}} 'struct flexible *__single' <BoundsSafetyPointerCast>
// CHECK: | |   `-PredefinedBoundsCheckExpr {{.+}} 'struct flexible *__bidi_indexable' <FlexibleArrayCountAssign(BasePtr, FamPtr, Count)>
// CHECK: | |     |-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: | |     |   | | |-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'struct flexible *__single'
// CHECK: | |     |-OpaqueValueExpr [[ove_4]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: | |     |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: | |     | `-MemberExpr {{.+}} ->elems
// CHECK: | |     |   `-OpaqueValueExpr [[ove_4]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: | |     `-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'int'
// CHECK: | |-OpaqueValueExpr [[ove_4]]
// CHECK: | | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: | |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: | |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK: | |   | | |-OpaqueValueExpr [[ove_5]] {{.*}} 'struct flexible *__single'
// CHECK: | |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: | |   | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: | |   | | | | `-MemberExpr {{.+}} ->elems
// CHECK: | |   | | | |   `-OpaqueValueExpr [[ove_5]] {{.*}} 'struct flexible *__single'
// CHECK: | |   | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: | |   | | |   `-MemberExpr {{.+}} ->count
// CHECK: | |   | | |     `-OpaqueValueExpr [[ove_5]] {{.*}} 'struct flexible *__single'
// CHECK: | |   | `-OpaqueValueExpr [[ove_5]]
// CHECK: | |   |   `-CallExpr
// CHECK: | |   |     `-ImplicitCastExpr {{.+}} 'struct flexible *__single(*__single)()' <FunctionToPointerDecay>
// CHECK: | |   |       `-DeclRefExpr {{.+}} [[func_return_flex]]
// CHECK: | |   `-OpaqueValueExpr [[ove_5]] {{.*}} 'struct flexible *__single'
// CHECK: | `-OpaqueValueExpr [[ove_6]]
// CHECK: |   `-IntegerLiteral {{.+}} 10
// CHECK: `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:   |-BinaryOperator {{.+}} 'int' '='
// CHECK:   | |-MemberExpr {{.+}} ->count
// CHECK:   | | `-ImplicitCastExpr {{.+}} 'struct flexible *__single' <LValueToRValue>
// CHECK:   | |   `-DeclRefExpr {{.+}} [[var_s_2]]
// CHECK:   | `-OpaqueValueExpr [[ove_6]] {{.*}} 'int'
// CHECK:   |-OpaqueValueExpr [[ove_4]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK:   `-OpaqueValueExpr [[ove_6]] {{.*}} 'int'

// CHECK-LABEL: bidi_init
void bidi_init() {
  struct flexible *b = return_flex();
}
// CHECK: CompoundStmt
// CHECK: `-DeclStmt
// CHECK:   `-VarDecl [[var_b:0x[^ ]+]]
// CHECK:     `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:       |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK:       | | |-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'struct flexible *__single'
// CHECK:       | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK:       | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK:       | | | | `-MemberExpr {{.+}} ->elems
// CHECK:       | | | |   `-OpaqueValueExpr [[ove_7]] {{.*}} 'struct flexible *__single'
// CHECK:       | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:       | | |   `-MemberExpr {{.+}} ->count
// CHECK:       | | |     `-OpaqueValueExpr [[ove_7]] {{.*}} 'struct flexible *__single'
// CHECK:       | `-OpaqueValueExpr [[ove_7]]
// CHECK:       |   `-CallExpr
// CHECK:       |     `-ImplicitCastExpr {{.+}} 'struct flexible *__single(*__single)()' <FunctionToPointerDecay>
// CHECK:       |       `-DeclRefExpr {{.+}} [[func_return_flex]]
// CHECK:       `-OpaqueValueExpr [[ove_7]] {{.*}} 'struct flexible *__single'

// CHECK-LABEL: bidi_assign
void bidi_assign() {
  struct flexible *b;
  b = return_flex();
}
// CHECK: CompoundStmt
// CHECK: |-DeclStmt
// CHECK: | `-VarDecl [[var_b_1:0x[^ ]+]]
// CHECK: `-BinaryOperator {{.+}} 'struct flexible *__bidi_indexable' '='
// CHECK:   |-DeclRefExpr {{.+}} [[var_b_1]]
// CHECK:   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK:     | | |-OpaqueValueExpr [[ove_8:0x[^ ]+]] {{.*}} 'struct flexible *__single'
// CHECK:     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK:     | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK:     | | | | `-MemberExpr {{.+}} ->elems
// CHECK:     | | | |   `-OpaqueValueExpr [[ove_8]] {{.*}} 'struct flexible *__single'
// CHECK:     | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     | | |   `-MemberExpr {{.+}} ->count
// CHECK:     | | |     `-OpaqueValueExpr [[ove_8]] {{.*}} 'struct flexible *__single'
// CHECK:     | `-OpaqueValueExpr [[ove_8]]
// CHECK:     |   `-CallExpr
// CHECK:     |     `-ImplicitCastExpr {{.+}} 'struct flexible *__single(*__single)()' <FunctionToPointerDecay>
// CHECK:     |       `-DeclRefExpr {{.+}} [[func_return_flex]]
// CHECK:     `-OpaqueValueExpr [[ove_8]] {{.*}} 'struct flexible *__single'
