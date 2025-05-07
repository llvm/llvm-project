

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s

#include <ptrcheck.h>

struct flexible {
    int count;
    int elems[__counted_by(count)];
};

void sink(struct flexible *__single flex);
// CHECK: |-FunctionDecl [[func_sink:0x[^ ]+]] {{.+}} sink

// CHECK-LABEL: checking_count_single
void checking_count_single(struct flexible *__single flex) {
// CHECK: | |-ParmVarDecl [[var_flex_1:0x[^ ]+]]
    sink(flex);
// CHECK: |   |-CallExpr
// CHECK: |   | |-ImplicitCastExpr {{.+}} 'void (*__single)(struct flexible *__single)' <FunctionToPointerDecay>
// CHECK: |   | | `-DeclRefExpr {{.+}} [[func_sink]]
// CHECK: |   | | | `-ImplicitCastExpr {{.+}} 'struct flexible *__single' <BoundsSafetyPointerCast>
// CHECK: |   | | |   `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: |   | | |       | | |-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'struct flexible *__single'
// CHECK: |   | | `-OpaqueValueExpr [[ove]]
// CHECK: |   | |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   | |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK: |   | |     | | |-OpaqueValueExpr [[ove_1]] {{.*}} 'struct flexible *__single'
// CHECK: |   | |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: |   | |     | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: |   | |     | | | | `-MemberExpr {{.+}} ->elems
// CHECK: |   | |     | | | |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'struct flexible *__single'
// CHECK: |   | |     | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   | |     | | |   `-MemberExpr {{.+}} ->count
// CHECK: |   | |     | | |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'struct flexible *__single'
// CHECK: |   | |     | `-OpaqueValueExpr [[ove_1]]
// CHECK: |   | |     |   `-ImplicitCastExpr {{.+}} 'struct flexible *__single' <LValueToRValue>
// CHECK: |   | |     |     `-DeclRefExpr {{.+}} [[var_flex_1]]
// CHECK: |   | |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'struct flexible *__single'
// CHECK: |   | `-OpaqueValueExpr [[ove]] {{.*}} 'struct flexible *__bidi_indexable'

    (void)(struct flexible *__single)flex;
// CHECK: |   `-CStyleCastExpr {{.+}} 'void' <ToVoid>
// CHECK: |     `-CStyleCastExpr {{.+}} 'struct flexible *__single' <BoundsSafetyPointerCast>
// CHECK: |       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK: |         | | |-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'struct flexible *__single'
// CHECK: |         | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: |         | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: |         | | | | `-MemberExpr {{.+}} ->elems
// CHECK: |         | | | |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'struct flexible *__single'
// CHECK: |         | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |         | | |   `-MemberExpr {{.+}} ->count
// CHECK: |         | | |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'struct flexible *__single'
// CHECK: |         | `-OpaqueValueExpr [[ove_2]]
// CHECK: |         |   `-ImplicitCastExpr {{.+}} 'struct flexible *__single' <LValueToRValue>
// CHECK: |         |     `-DeclRefExpr {{.+}} [[var_flex_1]]
// CHECK: |         `-OpaqueValueExpr [[ove_2]] {{.*}} 'struct flexible *__single'

}

// CHECK-LABEL: checking_count_indexable
void checking_count_indexable(struct flexible *__indexable flex) {
// CHECK: | |-ParmVarDecl [[var_flex_2:0x[^ ]+]]
    sink(flex);
// CHECK:  |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:  | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   |-CallExpr
// CHECK: |   | |-ImplicitCastExpr {{.+}} 'void (*__single)(struct flexible *__single)' <FunctionToPointerDecay>
// CHECK: |   | | `-DeclRefExpr {{.+}} [[func_sink]]
// CHECK: |   | | | `-ImplicitCastExpr {{.+}} 'struct flexible *__single' <BoundsSafetyPointerCast>
// CHECK: |   | | |   `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'struct flexible *__indexable'
// CHECK: |   | | |       | | |-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'struct flexible *__indexable'
// CHECK: |   | | `-OpaqueValueExpr [[ove_3]]
// CHECK: |   | |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   | |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   | |     | |-PredefinedBoundsCheckExpr {{.+}} 'struct flexible *__indexable' <FlexibleArrayCountCast(BasePtr, FamPtr, Count)>
// CHECK: |   | |     | | |-OpaqueValueExpr [[ove_4]] {{.*}} 'struct flexible *__indexable'
// CHECK: |   | |     | | |-OpaqueValueExpr [[ove_4]] {{.*}} 'struct flexible *__indexable'
// CHECK: |   | |     | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: |   | |     | | | `-MemberExpr {{.+}} ->elems
// CHECK: |   | |     | | |   `-OpaqueValueExpr [[ove_4]] {{.*}} 'struct flexible *__indexable'
// CHECK: |   | |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   | |     | |   `-MemberExpr {{.+}} ->count
// CHECK: |   | |     | |     `-OpaqueValueExpr [[ove_4]] {{.*}} 'struct flexible *__indexable'
// CHECK: |   | |     | `-OpaqueValueExpr [[ove_4]]
// CHECK: |   | |     |   `-ImplicitCastExpr {{.+}} 'struct flexible *__indexable' <LValueToRValue>
// CHECK: |   | |     |     `-DeclRefExpr {{.+}} [[var_flex_2]]
// CHECK: |   | |     `-OpaqueValueExpr [[ove_4]] {{.*}} 'struct flexible *__indexable'
// CHECK: |   | `-OpaqueValueExpr [[ove_3]] {{.*}} 'struct flexible *__indexable'

    (void)(struct flexible *__single)flex;
// CHECK: |   `-CStyleCastExpr {{.+}} 'void' <ToVoid>
// CHECK: |     `-CStyleCastExpr {{.+}} 'struct flexible *__single' <BoundsSafetyPointerCast>
// CHECK: |       `-ImplicitCastExpr {{.+}} 'struct flexible *__indexable' <LValueToRValue>
// CHECK: |         `-DeclRefExpr {{.+}} [[var_flex_2]]
}

// CHECK-LABEL: checking_count_bidi_indexable
void checking_count_bidi_indexable(struct flexible *__indexable flex) {
// CHECK: {{^}}| |-ParmVarDecl [[var_flex_3:0x[^ ]+]]
    sink(flex);
// CHECK: | `-CompoundStmt
// CHECK: |   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   | | |-CallExpr
// CHECK: |   | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(struct flexible *__single)' <FunctionToPointerDecay>
// CHECK: |   | | | | `-DeclRefExpr {{.+}} [[func_sink]]
// CHECK: |   | | | `-ImplicitCastExpr {{.+}} 'struct flexible *__single' <BoundsSafetyPointerCast>
// CHECK: |   | | |   `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'struct flexible *__indexable'
// CHECK: |   | | |       | | |-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'struct flexible *__indexable'
// CHECK: |   | | `-OpaqueValueExpr [[ove_5]]
// CHECK: |   | |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   | |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   | |     | |-PredefinedBoundsCheckExpr {{.+}} 'struct flexible *__indexable' <FlexibleArrayCountCast(BasePtr, FamPtr, Count)>
// CHECK: |   | |     | | |-OpaqueValueExpr [[ove_6]] {{.*}} 'struct flexible *__indexable'
// CHECK: |   | |     | | |-OpaqueValueExpr [[ove_6]] {{.*}} 'struct flexible *__indexable'
// CHECK: |   | |     | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: |   | |     | | | `-MemberExpr {{.+}} ->elems
// CHECK: |   | |     | | |   `-OpaqueValueExpr [[ove_6]] {{.*}} 'struct flexible *__indexable'
// CHECK: |   | |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   | |     | |   `-MemberExpr {{.+}} ->count
// CHECK: |   | |     | |     `-OpaqueValueExpr [[ove_6]] {{.*}} 'struct flexible *__indexable'
// CHECK: |   | |     | `-OpaqueValueExpr [[ove_6]]
// CHECK: |   | |     |   `-ImplicitCastExpr {{.+}} 'struct flexible *__indexable' <LValueToRValue>
// CHECK: |   | |     |     `-DeclRefExpr {{.+}} [[var_flex_3]]
// CHECK: |   | |     `-OpaqueValueExpr [[ove_6]] {{.*}} 'struct flexible *__indexable'
// CHECK: |   | `-OpaqueValueExpr [[ove_5]] {{.*}} 'struct flexible *__indexable'

    (void)(struct flexible *__single)flex;
// CHECK: |   `-CStyleCastExpr {{.+}} 'void' <ToVoid>
// CHECK: |     `-CStyleCastExpr {{.+}} 'struct flexible *__single' <BoundsSafetyPointerCast>
// CHECK: |       `-ImplicitCastExpr {{.+}} 'struct flexible *__indexable' <LValueToRValue>
// CHECK: |         `-DeclRefExpr {{.+}} [[var_flex_3]]
}

// CHECK-LABEL: checking_count_sized_by
void checking_count_sized_by(struct flexible *__sized_by(size) flex, int size) {
// CHECK: {{^}}| |-ParmVarDecl [[var_flex_4:0x[^ ]+]]
// CHECK: {{^}}| |-ParmVarDecl [[var_size:0x[^ ]+]]
    sink(flex);
// CHECK: | `-CompoundStmt
// CHECK: |   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   | | |-CallExpr
// CHECK: |   | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(struct flexible *__single)' <FunctionToPointerDecay>
// CHECK: |   | | | | `-DeclRefExpr {{.+}} [[func_sink]]
// CHECK: |   | | | `-ImplicitCastExpr {{.+}} 'struct flexible *__single' <BoundsSafetyPointerCast>
// CHECK: |   | | |   `-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: |   | | |       | | |-OpaqueValueExpr [[ove_8:0x[^ ]+]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: |   | | |       | | |   | | |-OpaqueValueExpr [[ove_9:0x[^ ]+]] {{.*}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single'
// CHECK: |   | | |       | | |   | | |   `-OpaqueValueExpr [[ove_10:0x[^ ]+]] {{.*}} 'int'
// CHECK: |   | | `-OpaqueValueExpr [[ove_7]]
// CHECK: |   | |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   | |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   | |     | |-PredefinedBoundsCheckExpr {{.+}} 'struct flexible *__bidi_indexable' <FlexibleArrayCountCast(BasePtr, FamPtr, Count)>
// CHECK: |   | |     | | |-OpaqueValueExpr [[ove_8]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: |   | |     | | |-OpaqueValueExpr [[ove_8]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: |   | |     | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: |   | |     | | | `-MemberExpr {{.+}} ->elems
// CHECK: |   | |     | | |   `-OpaqueValueExpr [[ove_8]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: |   | |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   | |     | |   `-MemberExpr {{.+}} ->count
// CHECK: |   | |     | |     `-OpaqueValueExpr [[ove_8]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: |   | |     | `-OpaqueValueExpr [[ove_8]]
// CHECK: |   | |     |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   | |     |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   | |     |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK: |   | |     |     | | |-OpaqueValueExpr [[ove_9]] {{.*}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single'
// CHECK: |   | |     |     | | |-ImplicitCastExpr {{.+}} 'struct flexible *' <BitCast>
// CHECK: |   | |     |     | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: |   | |     |     | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: |   | |     |     | | |   | `-ImplicitCastExpr {{.+}} 'struct flexible *' <BoundsSafetyPointerCast>
// CHECK: |   | |     |     | | |   |   `-OpaqueValueExpr [[ove_9]] {{.*}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single'
// CHECK: |   | |     |     | | |   `-OpaqueValueExpr [[ove_10]] {{.*}} 'int'
// CHECK: |   | |     |     | |-OpaqueValueExpr [[ove_9]]
// CHECK: |   | |     |     | | `-ImplicitCastExpr {{.+}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single' <LValueToRValue>
// CHECK: |   | |     |     | |   `-DeclRefExpr {{.+}} [[var_flex_4]]
// CHECK: |   | |     |     | `-OpaqueValueExpr [[ove_10]]
// CHECK: |   | |     |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   | |     |     |     `-DeclRefExpr {{.+}} [[var_size]]
// CHECK: |   | |     |     |-OpaqueValueExpr [[ove_9]] {{.*}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single'
// CHECK: |   | |     |     `-OpaqueValueExpr [[ove_10]] {{.*}} 'int'
// CHECK: |   | |     `-OpaqueValueExpr [[ove_8]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: |   | `-OpaqueValueExpr [[ove_7]] {{.*}} 'struct flexible *__bidi_indexable'


    (void)(struct flexible *__single)flex;
// CHECK: |   `-CStyleCastExpr {{.+}} 'void' <ToVoid>
// CHECK: |     `-CStyleCastExpr {{.+}} 'struct flexible *__single' <BoundsSafetyPointerCast>
// CHECK: |       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK: |         | | |-OpaqueValueExpr [[ove_11:0x[^ ]+]] {{.*}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single'
// CHECK: |         | | |-ImplicitCastExpr {{.+}} 'struct flexible *' <BitCast>
// CHECK: |         | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: |         | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: |         | | |   | `-ImplicitCastExpr {{.+}} 'struct flexible *' <BoundsSafetyPointerCast>
// CHECK: |         | | |   |   `-OpaqueValueExpr [[ove_11]] {{.*}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single'
// CHECK: |         | | |   `-OpaqueValueExpr [[ove_12:0x[^ ]+]] {{.*}} 'int'
// CHECK: |         | |-OpaqueValueExpr [[ove_11]]
// CHECK: |         | | `-ImplicitCastExpr {{.+}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single' <LValueToRValue>
// CHECK: |         | |   `-DeclRefExpr {{.+}} [[var_flex_4]]
// CHECK: |         | `-OpaqueValueExpr [[ove_12]]
// CHECK: |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |         |     `-DeclRefExpr {{.+}} [[var_size]]
// CHECK: |         |-OpaqueValueExpr [[ove_11]] {{.*}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single'
// CHECK: |         `-OpaqueValueExpr [[ove_12]] {{.*}} 'int'

}

// FIXME: Unnecessary promotion
// CHECK-LABEL: checking_count_single_return
struct flexible *checking_count_single_return(struct flexible *__single flex) {
// CHECK: | |-ParmVarDecl [[var_flex_5:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} 'struct flexible *__single' <BoundsSafetyPointerCast>
// CHECK: |       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK: |         | | |-OpaqueValueExpr [[ove_13:0x[^ ]+]] {{.*}} 'struct flexible *__single'
// CHECK: |         | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: |         | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: |         | | | | `-MemberExpr {{.+}} ->elems
// CHECK: |         | | | |   `-OpaqueValueExpr [[ove_13]] {{.*}} 'struct flexible *__single'
// CHECK: |         | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |         | | |   `-MemberExpr {{.+}} ->count
// CHECK: |         | | |     `-OpaqueValueExpr [[ove_13]] {{.*}} 'struct flexible *__single'
// CHECK: |         | `-OpaqueValueExpr [[ove_13]]
// CHECK: |         |   `-ImplicitCastExpr {{.+}} 'struct flexible *__single' <LValueToRValue>
// CHECK: |         |     `-DeclRefExpr {{.+}} [[var_flex_5]]
// CHECK: |         `-OpaqueValueExpr [[ove_13]] {{.*}} 'struct flexible *__single'
  return flex;
}

// CHECK-LABEL: checking_count_indexable_return
struct flexible *checking_count_indexable_return(struct flexible *__indexable flex) {
// CHECK: | |-ParmVarDecl [[var_flex_6:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} 'struct flexible *__single' <BoundsSafetyPointerCast>
// CHECK: |       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         | |-PredefinedBoundsCheckExpr {{.+}} 'struct flexible *__indexable' <FlexibleArrayCountCast(BasePtr, FamPtr, Count)>
// CHECK: |         | | |-OpaqueValueExpr [[ove_14:0x[^ ]+]] {{.*}} 'struct flexible *__indexable'
// CHECK: |         | | |-OpaqueValueExpr [[ove_14]] {{.*}} 'struct flexible *__indexable'
// CHECK: |         | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: |         | | | `-MemberExpr {{.+}} ->elems
// CHECK: |         | | |   `-OpaqueValueExpr [[ove_14]] {{.*}} 'struct flexible *__indexable'
// CHECK: |         | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |         | |   `-MemberExpr {{.+}} ->count
// CHECK: |         | |     `-OpaqueValueExpr [[ove_14]] {{.*}} 'struct flexible *__indexable'
// CHECK: |         | `-OpaqueValueExpr [[ove_14]]
// CHECK: |         |   `-ImplicitCastExpr {{.+}} 'struct flexible *__indexable' <LValueToRValue>
// CHECK: |         |     `-DeclRefExpr {{.+}} [[var_flex_6]]
// CHECK: |         `-OpaqueValueExpr [[ove_14]] {{.*}} 'struct flexible *__indexable'
  return flex;
}

// CHECK-LABEL: checking_count_bidi_indexable_return
struct flexible *checking_count_bidi_indexable_return(struct flexible *__bidi_indexable flex) {
// CHECK: | |-ParmVarDecl [[var_flex_7:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} 'struct flexible *__single' <BoundsSafetyPointerCast>
// CHECK: |       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         | |-PredefinedBoundsCheckExpr {{.+}} 'struct flexible *__bidi_indexable' <FlexibleArrayCountCast(BasePtr, FamPtr, Count)>
// CHECK: |         | | |-OpaqueValueExpr [[ove_15:0x[^ ]+]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: |         | | |-OpaqueValueExpr [[ove_15]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: |         | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: |         | | | `-MemberExpr {{.+}} ->elems
// CHECK: |         | | |   `-OpaqueValueExpr [[ove_15]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: |         | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |         | |   `-MemberExpr {{.+}} ->count
// CHECK: |         | |     `-OpaqueValueExpr [[ove_15]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: |         | `-OpaqueValueExpr [[ove_15]]
// CHECK: |         |   `-ImplicitCastExpr {{.+}} 'struct flexible *__bidi_indexable' <LValueToRValue>
// CHECK: |         |     `-DeclRefExpr {{.+}} [[var_flex_7]]
// CHECK: |         `-OpaqueValueExpr [[ove_15]] {{.*}} 'struct flexible *__bidi_indexable'
  return flex;
}

// CHECK-LABEL: checking_count_sized_by_return
struct flexible *checking_count_sized_by_return(struct flexible *__sized_by(size) flex, unsigned long long size) {
// CHECK: | |-ParmVarDecl [[var_flex_8:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_size_1:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
// CHECK: | `-CompoundStmt
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} 'struct flexible *__single' <BoundsSafetyPointerCast>
// CHECK: |       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         | |-PredefinedBoundsCheckExpr {{.+}} 'struct flexible *__bidi_indexable' <FlexibleArrayCountCast(BasePtr, FamPtr, Count)>
// CHECK: |         | | |-OpaqueValueExpr [[ove_16:0x[^ ]+]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: |         | | |   | | |-OpaqueValueExpr [[ove_17:0x[^ ]+]] {{.*}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single'
// CHECK: |         | | |   | | |     |-OpaqueValueExpr [[ove_18:0x[^ ]+]] {{.*}} 'unsigned long long'
// CHECK: |         | | |-OpaqueValueExpr [[ove_16]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: |         | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: |         | | | `-MemberExpr {{.+}} ->elems
// CHECK: |         | | |   `-OpaqueValueExpr [[ove_16]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: |         | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |         | |   `-MemberExpr {{.+}} ->count
// CHECK: |         | |     `-OpaqueValueExpr [[ove_16]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: |         | `-OpaqueValueExpr [[ove_16]]
// CHECK: |         |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK: |         |     | | |-OpaqueValueExpr [[ove_17]] {{.*}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single'
// CHECK: |         |     | | |-ImplicitCastExpr {{.+}} 'struct flexible *' <BitCast>
// CHECK: |         |     | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: |         |     | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: |         |     | | |   | `-ImplicitCastExpr {{.+}} 'struct flexible *' <BoundsSafetyPointerCast>
// CHECK: |         |     | | |   |   `-OpaqueValueExpr [[ove_17]] {{.*}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single'
// CHECK: |         |     | | |   `-AssumptionExpr
// CHECK: |         |     | | |     |-OpaqueValueExpr [[ove_18]] {{.*}} 'unsigned long long'
// CHECK: |         |     | | |     `-BinaryOperator {{.+}} 'int' '>='
// CHECK: |         |     | | |       |-ImplicitCastExpr {{.+}} 'long long' <IntegralCast>
// CHECK: |         |     | | |       | `-OpaqueValueExpr [[ove_18]] {{.*}} 'unsigned long long'
// CHECK: |         |     | | |       `-ImplicitCastExpr {{.+}} 'long long' <IntegralCast>
// CHECK: |         |     | | |         `-IntegerLiteral {{.+}} 0
// CHECK: |         |     | |-OpaqueValueExpr [[ove_17]]
// CHECK: |         |     | | `-ImplicitCastExpr {{.+}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single' <LValueToRValue>
// CHECK: |         |     | |   `-DeclRefExpr {{.+}} [[var_flex_8]]
// CHECK: |         |     | `-OpaqueValueExpr [[ove_18]]
// CHECK: |         |     |   `-ImplicitCastExpr {{.+}} 'unsigned long long' <LValueToRValue>
// CHECK: |         |     |     `-DeclRefExpr {{.+}} [[var_size_1]]
// CHECK: |         |     |-OpaqueValueExpr [[ove_17]] {{.*}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single'
// CHECK: |         |     `-OpaqueValueExpr [[ove_18]] {{.*}} 'unsigned long long'
// CHECK: |         `-OpaqueValueExpr [[ove_16]] {{.*}} 'struct flexible *__bidi_indexable'

  return flex;
}

;
