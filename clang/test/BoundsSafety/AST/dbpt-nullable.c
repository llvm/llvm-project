
// RUN: %clang_cc1 -triple x86_64-apple-mac -ast-dump -fbounds-safety %s | FileCheck %s --check-prefix=CHECK
// RUN: %clang_cc1 -triple x86_64-apple-mac -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s --check-prefix=CHECK

#include <ptrcheck.h>

void counted(int *_Nullable __counted_by(count) array, int count) {
	array[10];
}

// CHECK: |-FunctionDecl [[func_counted:0x[^ ]+]] {{.+}} counted
// CHECK: | |-ParmVarDecl [[var_array:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_count:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
// CHECK: | `-CompoundStmt
// CHECK: |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |     `-ArraySubscriptExpr
// CHECK: |       |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |       | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |       | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: |       | | | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int *__single __counted_by(count) _Nullable':'int *__single'
// CHECK: |       | | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: |       | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |       | | | | | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __counted_by(count) _Nullable':'int *__single'
// CHECK: |       | | | | `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int'
// CHECK: |       | | |-OpaqueValueExpr [[ove]]
// CHECK: |       | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count) _Nullable':'int *__single' <LValueToRValue>
// CHECK: |       | | |   `-DeclRefExpr {{.+}} [[var_array]]
// CHECK: |       | | `-OpaqueValueExpr [[ove_1]]
// CHECK: |       | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |       | |     `-DeclRefExpr {{.+}} [[var_count]]
// CHECK: |       | |-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __counted_by(count) _Nullable':'int *__single'
// CHECK: |       | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK: |       `-IntegerLiteral {{.+}} 10

void sized(int *_Nullable __sized_by(count) array, int count) {
	array[10];
}

// CHECK: |-FunctionDecl [[func_sized:0x[^ ]+]] {{.+}} sized
// CHECK: | |-ParmVarDecl [[var_array_1:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_count_1:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
// CHECK: | `-CompoundStmt
// CHECK: |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |     `-ArraySubscriptExpr
// CHECK: |       |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |       | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |       | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: |       | | | |-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int *__single __sized_by(count) _Nullable':'int *__single'
// CHECK: |       | | | |-ImplicitCastExpr {{.+}} 'int *' <BitCast>
// CHECK: |       | | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: |       | | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: |       | | | |   | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |       | | | |   |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __sized_by(count) _Nullable':'int *__single'
// CHECK: |       | | | |   `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'int'
// CHECK: |       | | |-OpaqueValueExpr [[ove_2]]
// CHECK: |       | | | `-ImplicitCastExpr {{.+}} 'int *__single __sized_by(count) _Nullable':'int *__single' <LValueToRValue>
// CHECK: |       | | |   `-DeclRefExpr {{.+}} [[var_array_1]]
// CHECK: |       | | `-OpaqueValueExpr [[ove_3]]
// CHECK: |       | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |       | |     `-DeclRefExpr {{.+}} [[var_count_1]]
// CHECK: |       | |-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __sized_by(count) _Nullable':'int *__single'
// CHECK: |       | `-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// CHECK: |       `-IntegerLiteral {{.+}} 10

void ended(int *_Nullable __ended_by(end) array, int *end) {
	array[10];
}

// CHECK: `-FunctionDecl [[func_ended:0x[^ ]+]] {{.+}} ended
// CHECK:   |-ParmVarDecl [[var_array_2:0x[^ ]+]]
// CHECK:   |-ParmVarDecl [[var_end:0x[^ ]+]]
// CHECK:   `-CompoundStmt
// CHECK:     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:       `-ArraySubscriptExpr
// CHECK:         |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK:         | |-DeclRefExpr {{.+}} [[var_array_2]]
// CHECK:         | |-ImplicitCastExpr {{.+}} 'int *__single /* __started_by(array) */ ':'int *__single' <LValueToRValue>
// CHECK:         | | `-DeclRefExpr {{.+}} [[var_end]]
// CHECK:         | `-ImplicitCastExpr {{.+}} 'int *__single __ended_by(end) _Nullable':'int *__single' <LValueToRValue>
// CHECK:         |   `-DeclRefExpr {{.+}} [[var_array_2]]
// CHECK:         `-IntegerLiteral {{.+}} 10
