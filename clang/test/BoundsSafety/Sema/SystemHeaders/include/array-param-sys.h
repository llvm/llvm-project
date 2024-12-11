#include <ptrcheck.h>

void extFunc(int size, int arr[size]);
void extFunc3(int size, int * __null_terminated arr); // strict-note{{passing argument to parameter 'arr' here}}

#pragma clang system_header

extern void extFunc2(int size, int *sizeLessArr);

// CHECK: |-FunctionDecl [[func_extFunc:0x[^ ]+]] {{.+}} extFunc
// CHECK: | |-ParmVarDecl [[var_size:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
// CHECK: | `-ParmVarDecl [[var_arr:0x[^ ]+]]
// CHECK: |-FunctionDecl [[func_extFunc3:0x[^ ]+]] {{.+}} extFunc3
// CHECK: | |-ParmVarDecl [[var_size_1:0x[^ ]+]]
// CHECK: | `-ParmVarDecl [[var_arr_1:0x[^ ]+]]
// CHECK: |-FunctionDecl [[func_extFunc2:0x[^ ]+]] {{.+}} 'void (int, int *)'
// CHECK: | |-ParmVarDecl [[var_size_2:0x[^ ]+]]
// CHECK: | `-ParmVarDecl [[var_sizeLessArr:0x[^ ]+]]

static inline void funcInSDK(int size, int arr[size]) {
    extFunc(size, arr);
    extFunc2(size, arr);
}

// CHECK: |-FunctionDecl [[func_static:0x[^ ]+]] {{.+}} static
// CHECK: | |-ParmVarDecl [[var_size_3:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
// CHECK: | |-ParmVarDecl [[var_arr_2:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   | | |-BoundsCheckExpr {{.+}} 'arr <= __builtin_get_pointer_upper_bound(arr) && __builtin_get_pointer_lower_bound(arr) <= arr && size <= __builtin_get_pointer_upper_bound(arr) - arr && 0 <= size'
// CHECK: |   | | | |-CallExpr
// CHECK: |   | | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int, int *__single __counted_by(size))' <FunctionToPointerDecay>
// CHECK: |   | | | | | `-DeclRefExpr {{.+}} [[func_extFunc]]
// CHECK: |   | | | | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int'
// CHECK: |   | | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(size)':'int *__single' <BoundsSafetyPointerCast>
// CHECK: |   | | | |   `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | | | |       | | |-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int *__single __counted_by(size)':'int *__single'
// CHECK: |   | | | |       | | | `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'int'
// CHECK: |   | | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   | | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   | | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | | |   | | | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | | |   | | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK: |   | | |   | |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK: |   | | |   |   |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | | |   |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   | | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |   | | |     | | `-OpaqueValueExpr [[ove]] {{.*}} 'int'
// CHECK: |   | | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |   | | |     |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | | |     |   | `-GetBoundExpr {{.+}} upper
// CHECK: |   | | |     |   |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | | |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | | |     |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | | |       |-IntegerLiteral {{.+}} 0
// CHECK: |   | | |       `-OpaqueValueExpr [[ove]] {{.*}} 'int'
// CHECK: |   | | |-OpaqueValueExpr [[ove]]
// CHECK: |   | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   | | |   `-DeclRefExpr {{.+}} [[var_size_3]]
// CHECK: |   | | `-OpaqueValueExpr [[ove_1]]
// CHECK: |   | |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   | |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: |   | |     | | |-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __counted_by(size)':'int *__single'
// CHECK: |   | |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: |   | |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | |     | | | | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __counted_by(size)':'int *__single'
// CHECK: |   | |     | | | `-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// CHECK: |   | |     | |-OpaqueValueExpr [[ove_2]]
// CHECK: |   | |     | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(size)':'int *__single' <LValueToRValue>
// CHECK: |   | |     | |   `-DeclRefExpr {{.+}} [[var_arr_2]]
// CHECK: |   | |     | `-OpaqueValueExpr [[ove_3]]
// CHECK: |   | |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   | |     |     `-DeclRefExpr {{.+}} [[var_size_3]]
// CHECK: |   | |     |-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __counted_by(size)':'int *__single'
// CHECK: |   | |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// CHECK: |   | |-OpaqueValueExpr [[ove]] {{.*}} 'int'
// CHECK: |   | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   `-CallExpr
// CHECK: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(int, int *)' <FunctionToPointerDecay>
// CHECK: |     | `-DeclRefExpr {{.+}} [[func_extFunc2]]
// CHECK: |     |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |     | `-DeclRefExpr {{.+}} [[var_size_3]]
// CHECK: |     `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: |         | | |-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'int *__single __counted_by(size)':'int *__single'
// CHECK: |         | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: |         | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |         | | | | `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__single __counted_by(size)':'int *__single'
// CHECK: |         | | | `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'int'
// CHECK: |         | |-OpaqueValueExpr [[ove_4]]
// CHECK: |         | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(size)':'int *__single' <LValueToRValue>
// CHECK: |         | |   `-DeclRefExpr {{.+}} [[var_arr_2]]
// CHECK: |         | `-OpaqueValueExpr [[ove_5]]
// CHECK: |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |         |     `-DeclRefExpr {{.+}} [[var_size_3]]
// CHECK: |         |-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__single __counted_by(size)':'int *__single'
// CHECK: |         `-OpaqueValueExpr [[ove_5]] {{.*}} 'int'

static inline void funcInSDK2(int size, int *sizeLessArr) {
    // strict-error@+1{{passing 'int *' to parameter of incompatible type 'int *__single __counted_by(size)' (aka 'int *__single') casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    funcInSDK(size, sizeLessArr);
}

// CHECK: |-FunctionDecl [[func_static_1:0x[^ ]+]] {{.+}} static
// CHECK: | |-ParmVarDecl [[var_size_4:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_sizeLessArr_1:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |     | |-CallExpr
// CHECK: |     | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int, int *__single __counted_by(size))' <FunctionToPointerDecay>
// CHECK: |     | | | `-DeclRefExpr {{.+}} [[func_static]]
// CHECK: |     | | |-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'int'
// CHECK: |     | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(size)':'int *__single' <BoundsSafetyPointerCast>
// CHECK: |     | |   `-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'int *'
// CHECK: |     | |-OpaqueValueExpr [[ove_6]]
// CHECK: |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |     | |   `-DeclRefExpr {{.+}} [[var_size_4]]
// CHECK: |     | `-OpaqueValueExpr [[ove_7]]
// CHECK: |     |   `-ImplicitCastExpr {{.+}} 'int *' <LValueToRValue>
// CHECK: |     |     `-DeclRefExpr {{.+}} [[var_sizeLessArr_1]]
// CHECK: |     |-OpaqueValueExpr [[ove_6]] {{.*}} 'int'
// CHECK: |     `-OpaqueValueExpr [[ove_7]] {{.*}} 'int *'

static inline void funcInSDK3(int size, int arr[size]) {
    funcInSDK(size+1, arr);
    // strict-error@+2{{passing 'int *__single __counted_by(size)' (aka 'int *__single') to parameter of incompatible type 'int *__single __terminated_by(0)' (aka 'int *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
    // strict-note@21{{passing argument to parameter 'arr' here}}
    extFunc3(size+1, arr);
    // strict-error@+1{{assignment to 'size' requires corresponding assignment to 'int *__single __counted_by(size)' (aka 'int *__single') 'arr'; add self assignment 'arr = arr' if the value has not changed}}
    size++;
}

// CHECK: |-FunctionDecl [[func_static_2:0x[^ ]+]] {{.+}} static
// CHECK: | |-ParmVarDecl [[var_size_5:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
// CHECK: | |-ParmVarDecl [[var_arr_3:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   | | |-BoundsCheckExpr {{.+}} 'arr <= __builtin_get_pointer_upper_bound(arr) && __builtin_get_pointer_lower_bound(arr) <= arr && size + 1 <= __builtin_get_pointer_upper_bound(arr) - arr && 0 <= size + 1'
// CHECK: |   | | | |-CallExpr
// CHECK: |   | | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int, int *__single __counted_by(size))' <FunctionToPointerDecay>
// CHECK: |   | | | | | `-DeclRefExpr {{.+}} [[func_static]]
// CHECK: |   | | | | |-OpaqueValueExpr [[ove_8:0x[^ ]+]] {{.*}} 'int'
// CHECK: |   | | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(size)':'int *__single' <BoundsSafetyPointerCast>
// CHECK: |   | | | |   `-OpaqueValueExpr [[ove_9:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | | | |       | | |-OpaqueValueExpr [[ove_10:0x[^ ]+]] {{.*}} 'int *__single __counted_by(size)':'int *__single'
// CHECK: |   | | | |       | | | `-OpaqueValueExpr [[ove_11:0x[^ ]+]] {{.*}} 'int'
// CHECK: |   | | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   | | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   | | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | | |   | | | `-OpaqueValueExpr [[ove_9]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | | |   | | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK: |   | | |   | |     `-OpaqueValueExpr [[ove_9]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK: |   | | |   |   |   `-OpaqueValueExpr [[ove_9]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | | |   |     `-OpaqueValueExpr [[ove_9]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   | | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |   | | |     | | `-OpaqueValueExpr [[ove_8]] {{.*}} 'int'
// CHECK: |   | | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |   | | |     |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | | |     |   | `-GetBoundExpr {{.+}} upper
// CHECK: |   | | |     |   |   `-OpaqueValueExpr [[ove_9]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | | |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | | |     |     `-OpaqueValueExpr [[ove_9]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | | |       |-IntegerLiteral {{.+}} 0
// CHECK: |   | | |       `-OpaqueValueExpr [[ove_8]] {{.*}} 'int'
// CHECK: |   | | |-OpaqueValueExpr [[ove_8]]
// CHECK: |   | | | `-BinaryOperator {{.+}} 'int' '+'
// CHECK: |   | | |   |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   | | |   | `-DeclRefExpr {{.+}} [[var_size_5]]
// CHECK: |   | | |   `-IntegerLiteral {{.+}} 1
// CHECK: |   | | `-OpaqueValueExpr [[ove_9]]
// CHECK: |   | |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   | |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: |   | |     | | |-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__single __counted_by(size)':'int *__single'
// CHECK: |   | |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: |   | |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | |     | | | | `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__single __counted_by(size)':'int *__single'
// CHECK: |   | |     | | | `-OpaqueValueExpr [[ove_11]] {{.*}} 'int'
// CHECK: |   | |     | |-OpaqueValueExpr [[ove_10]]
// CHECK: |   | |     | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(size)':'int *__single' <LValueToRValue>
// CHECK: |   | |     | |   `-DeclRefExpr {{.+}} [[var_arr_3]]
// CHECK: |   | |     | `-OpaqueValueExpr [[ove_11]]
// CHECK: |   | |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   | |     |     `-DeclRefExpr {{.+}} [[var_size_5]]
// CHECK: |   | |     |-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__single __counted_by(size)':'int *__single'
// CHECK: |   | |     `-OpaqueValueExpr [[ove_11]] {{.*}} 'int'
// CHECK: |   | |-OpaqueValueExpr [[ove_8]] {{.*}} 'int'
// CHECK: |   | `-OpaqueValueExpr [[ove_9]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |-CallExpr
// CHECK: |   | |-ImplicitCastExpr {{.+}} 'void (*__single)(int, int *__single __terminated_by(0))' <FunctionToPointerDecay>
// CHECK: |   | | `-DeclRefExpr {{.+}} [[func_extFunc3]]
// CHECK: |   | |-BinaryOperator {{.+}} 'int' '+'
// CHECK: |   | | |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   | | | `-DeclRefExpr {{.+}} [[var_size_5]]
// CHECK: |   | | `-IntegerLiteral {{.+}} 1
// CHECK: |   | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: |   |   | | |-OpaqueValueExpr [[ove_12:0x[^ ]+]] {{.*}} 'int *__single __counted_by(size)':'int *__single'
// CHECK: |   |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: |   |   | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | | | | `-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__single __counted_by(size)':'int *__single'
// CHECK: |   |   | | | `-OpaqueValueExpr [[ove_13:0x[^ ]+]] {{.*}} 'int'
// CHECK: |   |   | |-OpaqueValueExpr [[ove_12]]
// CHECK: |   |   | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(size)':'int *__single' <LValueToRValue>
// CHECK: |   |   | |   `-DeclRefExpr {{.+}} [[var_arr_3]]
// CHECK: |   |   | `-OpaqueValueExpr [[ove_13]]
// CHECK: |   |   |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   |   |     `-DeclRefExpr {{.+}} [[var_size_5]]
// CHECK: |   |   |-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__single __counted_by(size)':'int *__single'
// CHECK: |   |   `-OpaqueValueExpr [[ove_13]] {{.*}} 'int'
// CHECK: |   `-UnaryOperator {{.+}} postfix '++'
// CHECK: |     `-OpaqueValueExpr [[ove_14:0x[^ ]+]] {{.*}} lvalue

static void tmp() {
}

