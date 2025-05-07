

// RUN: %clang_cc1 -fbounds-safety -fbounds-safety-bringup-missing-checks=return_size -Wno-bounds-safety-single-to-count -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -fbounds-safety-bringup-missing-checks=return_size -Wno-bounds-safety-single-to-count -ast-dump %s 2>&1 | FileCheck %s
#include <ptrcheck.h>

// CHECK:      {{^}}|-FunctionDecl [[func_cb_0:0x[^ ]+]] {{.+}} cb_0
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_len:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-BoundsCheckExpr {{.+}} 'len == 0'
// CHECK-NEXT: {{^}}|       |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <NullToPointer>
// CHECK-NEXT: {{^}}|       | `-OpaqueValueExpr [[ove:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|       | |-OpaqueValueExpr [[ove_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       | |   `-DeclRefExpr {{.+}} [[var_len]]
// CHECK-NEXT: {{^}}|       | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |-OpaqueValueExpr [[ove]]
// CHECK-NEXT: {{^}}|       | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       `-OpaqueValueExpr [[ove_1]]
// CHECK-NEXT: {{^}}|         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|           `-DeclRefExpr {{.+}} [[var_len]]
int *__counted_by(len) cb_0(int len) {
  return 0;
}

// CHECK: {{^}}|-FunctionDecl [[func_cb_NULL:0x[^ ]+]] {{.+}} cb_NULL
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_len_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-BoundsCheckExpr {{.+}} 'len == 0'
// CHECK-NEXT: {{^}}|       |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | `-OpaqueValueExpr [[ove_2:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       |   `-CStyleCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT: {{^}}|       |     `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|       | |-OpaqueValueExpr [[ove_3:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       | |   `-DeclRefExpr {{.+}} [[var_len_1]]
// CHECK-NEXT: {{^}}|       | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |-OpaqueValueExpr [[ove_2]]
// CHECK-NEXT: {{^}}|       | `-CStyleCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT: {{^}}|       |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       `-OpaqueValueExpr [[ove_3]]
// CHECK-NEXT: {{^}}|         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|           `-DeclRefExpr {{.+}} [[var_len_1]]
int *__counted_by(len) cb_NULL(int len) {
  return (void *)0;
}

// CHECK: {{^}}|-FunctionDecl [[func_cb_int_cast:0x[^ ]+]] {{.+}} cb_int_cast
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_len_2:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-BoundsCheckExpr {{.+}} 'len == 0'
// CHECK-NEXT: {{^}}|       |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | `-OpaqueValueExpr [[ove_4:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       |   `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}|       |     `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|       | |-OpaqueValueExpr [[ove_5:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       | |   `-DeclRefExpr {{.+}} [[var_len_2]]
// CHECK-NEXT: {{^}}|       | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |-OpaqueValueExpr [[ove_4]]
// CHECK-NEXT: {{^}}|       | `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}|       |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       `-OpaqueValueExpr [[ove_5]]
// CHECK-NEXT: {{^}}|         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|           `-DeclRefExpr {{.+}} [[var_len_2]]
int *__counted_by(len) cb_int_cast(int len) {
  return (int *)0;
}

// CHECK: {{^}}|-FunctionDecl [[func_cb_int_cast_NULL:0x[^ ]+]] {{.+}} cb_int_cast_NULL
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_len_3:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-BoundsCheckExpr {{.+}} 'len == 0'
// CHECK-NEXT: {{^}}|       |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | `-OpaqueValueExpr [[ove_6:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       |   `-CStyleCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}|       |     `-CStyleCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT: {{^}}|       |       `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|       | |-OpaqueValueExpr [[ove_7:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       | |   `-DeclRefExpr {{.+}} [[var_len_3]]
// CHECK-NEXT: {{^}}|       | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |-OpaqueValueExpr [[ove_6]]
// CHECK-NEXT: {{^}}|       | `-CStyleCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}|       |   `-CStyleCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT: {{^}}|       |     `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       `-OpaqueValueExpr [[ove_7]]
// CHECK-NEXT: {{^}}|         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|           `-DeclRefExpr {{.+}} [[var_len_3]]
int *__counted_by(len) cb_int_cast_NULL(int len) {
  return (int *)(void*)0;
}

// CHECK: {{^}}|-FunctionDecl [[func_cb_int_void_int_cast_NULL:0x[^ ]+]] {{.+}} cb_int_void_int_cast_NULL
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_len_4:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-BoundsCheckExpr {{.+}} 'len == 0'
// CHECK-NEXT: {{^}}|       |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | `-OpaqueValueExpr [[ove_8:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       |   `-ParenExpr
// CHECK-NEXT: {{^}}|       |     `-CStyleCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}|       |       `-CStyleCastExpr {{.+}} 'void *' <BitCast>
// CHECK-NEXT: {{^}}|       |         `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}|       |           `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|       | |-OpaqueValueExpr [[ove_9:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       | |   `-DeclRefExpr {{.+}} [[var_len_4]]
// CHECK-NEXT: {{^}}|       | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |-OpaqueValueExpr [[ove_8]]
// CHECK-NEXT: {{^}}|       | `-ParenExpr
// CHECK-NEXT: {{^}}|       |   `-CStyleCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}|       |     `-CStyleCastExpr {{.+}} 'void *' <BitCast>
// CHECK-NEXT: {{^}}|       |       `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}|       |         `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       `-OpaqueValueExpr [[ove_9]]
// CHECK-NEXT: {{^}}|         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|           `-DeclRefExpr {{.+}} [[var_len_4]]
int *__counted_by(len) cb_int_void_int_cast_NULL(int len) {
  return ((int*)(void*)(int*)0);
}

// CHECK: {{^}}|-FunctionDecl [[func_sb_0:0x[^ ]+]] {{.+}} sb_0
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_size:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-BoundsCheckExpr {{.+}} 'size == 0'
// CHECK-NEXT: {{^}}|       |-ImplicitCastExpr {{.+}} 'int *__single __sized_by(size)':'int *__single' <NullToPointer>
// CHECK-NEXT: {{^}}|       | `-OpaqueValueExpr [[ove_10:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|       | |-OpaqueValueExpr [[ove_11:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       | |   `-DeclRefExpr {{.+}} [[var_size]]
// CHECK-NEXT: {{^}}|       | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |-OpaqueValueExpr [[ove_10]]
// CHECK-NEXT: {{^}}|       | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       `-OpaqueValueExpr [[ove_11]]
// CHECK-NEXT: {{^}}|         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|           `-DeclRefExpr {{.+}} [[var_size]]
int *__sized_by(size) sb_0(int size) {
  return 0;
}

// CHECK: {{^}}|-FunctionDecl [[func_sb_int_cast:0x[^ ]+]] {{.+}} sb_int_cast
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_len_5:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-BoundsCheckExpr {{.+}} 'len == 0'
// CHECK-NEXT: {{^}}|       |-ImplicitCastExpr {{.+}} 'int *__single __sized_by(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | `-OpaqueValueExpr [[ove_12:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       |   `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}|       |     `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|       | |-OpaqueValueExpr [[ove_13:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       | |   `-DeclRefExpr {{.+}} [[var_len_5]]
// CHECK-NEXT: {{^}}|       | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |-OpaqueValueExpr [[ove_12]]
// CHECK-NEXT: {{^}}|       | `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}|       |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       `-OpaqueValueExpr [[ove_13]]
// CHECK-NEXT: {{^}}|         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|           `-DeclRefExpr {{.+}} [[var_len_5]]
int *__sized_by(len) sb_int_cast(int len) {
  return (int *)0;
}

// CHECK: {{^}}|-FunctionDecl [[func_sb_int_cast_NULL:0x[^ ]+]] {{.+}} sb_int_cast_NULL
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_len_6:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-BoundsCheckExpr {{.+}} 'len == 0'
// CHECK-NEXT: {{^}}|       |-ImplicitCastExpr {{.+}} 'int *__single __sized_by(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | `-OpaqueValueExpr [[ove_14:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       |   `-CStyleCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}|       |     `-CStyleCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT: {{^}}|       |       `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|       | |-OpaqueValueExpr [[ove_15:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       | |   `-DeclRefExpr {{.+}} [[var_len_6]]
// CHECK-NEXT: {{^}}|       | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |-OpaqueValueExpr [[ove_14]]
// CHECK-NEXT: {{^}}|       | `-CStyleCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}|       |   `-CStyleCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT: {{^}}|       |     `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       `-OpaqueValueExpr [[ove_15]]
// CHECK-NEXT: {{^}}|         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|           `-DeclRefExpr {{.+}} [[var_len_6]]
int *__sized_by(len) sb_int_cast_NULL(int len) {
  return (int *)(void*)0;
}

// CHECK: {{^}}|-FunctionDecl [[func_sb_int_void_int_cast_NULL:0x[^ ]+]] {{.+}} sb_int_void_int_cast_NULL
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_len_7:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-BoundsCheckExpr {{.+}} 'len == 0'
// CHECK-NEXT: {{^}}|       |-ImplicitCastExpr {{.+}} 'int *__single __sized_by(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | `-OpaqueValueExpr [[ove_16:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       |   `-ParenExpr
// CHECK-NEXT: {{^}}|       |     `-CStyleCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}|       |       `-CStyleCastExpr {{.+}} 'void *' <BitCast>
// CHECK-NEXT: {{^}}|       |         `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}|       |           `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|       | |-OpaqueValueExpr [[ove_17:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       | |   `-DeclRefExpr {{.+}} [[var_len_7]]
// CHECK-NEXT: {{^}}|       | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |-OpaqueValueExpr [[ove_16]]
// CHECK-NEXT: {{^}}|       | `-ParenExpr
// CHECK-NEXT: {{^}}|       |   `-CStyleCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}|       |     `-CStyleCastExpr {{.+}} 'void *' <BitCast>
// CHECK-NEXT: {{^}}|       |       `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}|       |         `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       `-OpaqueValueExpr [[ove_17]]
// CHECK-NEXT: {{^}}|         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|           `-DeclRefExpr {{.+}} [[var_len_7]]
int *__sized_by(len) sb_int_void_int_cast_NULL(int len) {
  return ((int*)(void*)(int*)0);
}

// CHECK: {{^}}|-FunctionDecl [[func_cbn_0:0x[^ ]+]] {{.+}} cbn_0
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_len_8:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|       |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|       | |-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(len)':'int *__single' <NullToPointer>
// CHECK-NEXT: {{^}}|       | | `-OpaqueValueExpr [[ove_18:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       | |-OpaqueValueExpr [[ove_18]]
// CHECK-NEXT: {{^}}|       | | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       | `-OpaqueValueExpr [[ove_19:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       |     `-DeclRefExpr {{.+}} [[var_len_8]]
// CHECK-NEXT: {{^}}|       |-OpaqueValueExpr [[ove_18]] {{.*}} 'int'
// CHECK:      {{^}}|       `-OpaqueValueExpr [[ove_19]] {{.*}} 'int'
int *__counted_by_or_null(len) cbn_0(int len) {
  return 0;
}

// CHECK:      {{^}}|-FunctionDecl [[func_cbn_int_cast:0x[^ ]+]] {{.+}} cbn_int_cast
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_len_9:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|       |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|       | |-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | | `-OpaqueValueExpr [[ove_20:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       | |   `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}|       | |     `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       | |-OpaqueValueExpr [[ove_20]]
// CHECK-NEXT: {{^}}|       | | `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}|       | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       | `-OpaqueValueExpr [[ove_21:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       |     `-DeclRefExpr {{.+}} [[var_len_9]]
// CHECK-NEXT: {{^}}|       |-OpaqueValueExpr [[ove_20]] {{.*}} 'int *'
// CHECK:      {{^}}|       `-OpaqueValueExpr [[ove_21]] {{.*}} 'int'
int *__counted_by_or_null(len) cbn_int_cast(int len) {
  return (int*)0;
}

// CHECK:      {{^}}|-FunctionDecl [[func_cbn_int_cast_NULL:0x[^ ]+]] {{.+}} cbn_int_cast_NULL
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_len_10:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|       |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|       | |-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | | `-OpaqueValueExpr [[ove_22:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       | |   `-CStyleCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}|       | |     `-CStyleCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT: {{^}}|       | |       `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       | |-OpaqueValueExpr [[ove_22]]
// CHECK-NEXT: {{^}}|       | | `-CStyleCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}|       | |   `-CStyleCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT: {{^}}|       | |     `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       | `-OpaqueValueExpr [[ove_23:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       |     `-DeclRefExpr {{.+}} [[var_len_10]]
// CHECK-NEXT: {{^}}|       |-OpaqueValueExpr [[ove_22]] {{.*}} 'int *'
// CHECK:      {{^}}|       `-OpaqueValueExpr [[ove_23]] {{.*}} 'int'
int *__counted_by_or_null(len) cbn_int_cast_NULL(int len) {
  return (int*)(void*)0;
}

// CHECK:      {{^}}|-FunctionDecl [[func_cbn_int_void_int_cast_NULL:0x[^ ]+]] {{.+}} cbn_int_void_int_cast_NULL
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_len_11:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|       |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|       | |-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | | `-OpaqueValueExpr [[ove_24:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       | |   `-ParenExpr
// CHECK-NEXT: {{^}}|       | |     `-CStyleCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}|       | |       `-CStyleCastExpr {{.+}} 'void *' <BitCast>
// CHECK-NEXT: {{^}}|       | |         `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}|       | |           `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       | |-OpaqueValueExpr [[ove_24]]
// CHECK-NEXT: {{^}}|       | | `-ParenExpr
// CHECK-NEXT: {{^}}|       | |   `-CStyleCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}|       | |     `-CStyleCastExpr {{.+}} 'void *' <BitCast>
// CHECK-NEXT: {{^}}|       | |       `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}|       | |         `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       | `-OpaqueValueExpr [[ove_25:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       |     `-DeclRefExpr {{.+}} [[var_len_11]]
// CHECK-NEXT: {{^}}|       |-OpaqueValueExpr [[ove_24]] {{.*}} 'int *'
// CHECK:      {{^}}|       `-OpaqueValueExpr [[ove_25]] {{.*}} 'int'
int *__counted_by_or_null(len) cbn_int_void_int_cast_NULL(int len) {
  return ((int*)(void*)(int*)0);
}

// CHECK:      {{^}}|-FunctionDecl [[func_sbn_0:0x[^ ]+]] {{.+}} sbn_0
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_len_12:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|       |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|       | |-ImplicitCastExpr {{.+}} 'int *__single __sized_by_or_null(len)':'int *__single' <NullToPointer>
// CHECK-NEXT: {{^}}|       | | `-OpaqueValueExpr [[ove_26:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       | |-OpaqueValueExpr [[ove_26]]
// CHECK-NEXT: {{^}}|       | | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       | `-OpaqueValueExpr [[ove_27:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       |     `-DeclRefExpr {{.+}} [[var_len_12]]
// CHECK-NEXT: {{^}}|       |-OpaqueValueExpr [[ove_26]] {{.*}} 'int'
// CHECK:      {{^}}|       `-OpaqueValueExpr [[ove_27]] {{.*}} 'int'
int *__sized_by_or_null(len) sbn_0(int len) {
  return 0;
}

// CHECK:      {{^}}|-FunctionDecl [[func_sbn_int_cast:0x[^ ]+]] {{.+}} sbn_int_cast
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_len_13:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|       |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|       | |-ImplicitCastExpr {{.+}} 'int *__single __sized_by_or_null(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | | `-OpaqueValueExpr [[ove_28:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       | |   `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}|       | |     `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       | |-OpaqueValueExpr [[ove_28]]
// CHECK-NEXT: {{^}}|       | | `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}|       | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       | `-OpaqueValueExpr [[ove_29:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       |     `-DeclRefExpr {{.+}} [[var_len_13]]
// CHECK-NEXT: {{^}}|       |-OpaqueValueExpr [[ove_28]] {{.*}} 'int *'
// CHECK:      {{^}}|       `-OpaqueValueExpr [[ove_29]] {{.*}} 'int'
int *__sized_by_or_null(len) sbn_int_cast(int len) {
  return (int*)0;
}

// CHECK:      {{^}}|-FunctionDecl [[func_sbn_int_cast_NULL:0x[^ ]+]] {{.+}} sbn_int_cast_NULL
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_len_14:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|       |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|       | |-ImplicitCastExpr {{.+}} 'int *__single __sized_by_or_null(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | | `-OpaqueValueExpr [[ove_30:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       | |   `-CStyleCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}|       | |     `-CStyleCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT: {{^}}|       | |       `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       | |-OpaqueValueExpr [[ove_30]]
// CHECK-NEXT: {{^}}|       | | `-CStyleCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}|       | |   `-CStyleCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT: {{^}}|       | |     `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       | `-OpaqueValueExpr [[ove_31:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       |     `-DeclRefExpr {{.+}} [[var_len_14]]
// CHECK-NEXT: {{^}}|       |-OpaqueValueExpr [[ove_30]] {{.*}} 'int *'
// CHECK:      {{^}}|       `-OpaqueValueExpr [[ove_31]] {{.*}} 'int'
int *__sized_by_or_null(len) sbn_int_cast_NULL(int len) {
  return (int*)(void*)0;
}

// CHECK:      {{^}}`-FunctionDecl [[func_sbn_int_void_int_cast_NULL:0x[^ ]+]] {{.+}} sbn_int_void_int_cast_NULL
// CHECK-NEXT: {{^}}  |-ParmVarDecl [[var_len_15:0x[^ ]+]]
// CHECK-NEXT: {{^}}  `-CompoundStmt
// CHECK-NEXT: {{^}}    `-ReturnStmt
// CHECK-NEXT: {{^}}      `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}        |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}        | |-ImplicitCastExpr {{.+}} 'int *__single __sized_by_or_null(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}        | | `-OpaqueValueExpr [[ove_32:0x[^ ]+]]
// CHECK-NEXT: {{^}}        | |   `-ParenExpr
// CHECK-NEXT: {{^}}        | |     `-CStyleCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}        | |       `-CStyleCastExpr {{.+}} 'void *' <BitCast>
// CHECK-NEXT: {{^}}        | |         `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}        | |           `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}        | |-OpaqueValueExpr [[ove_32]]
// CHECK-NEXT: {{^}}        | | `-ParenExpr
// CHECK-NEXT: {{^}}        | |   `-CStyleCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}        | |     `-CStyleCastExpr {{.+}} 'void *' <BitCast>
// CHECK-NEXT: {{^}}        | |       `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}        | |         `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}        | `-OpaqueValueExpr [[ove_33:0x[^ ]+]]
// CHECK-NEXT: {{^}}        |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}        |     `-DeclRefExpr {{.+}} [[var_len_15]]
// CHECK-NEXT: {{^}}        |-OpaqueValueExpr [[ove_32]] {{.*}} 'int *'
// CHECK:      {{^}}        `-OpaqueValueExpr [[ove_33]] {{.*}} 'int'
int *__sized_by_or_null(len) sbn_int_void_int_cast_NULL(int len) {
  return ((int*)(void*)(int*)0);
}
