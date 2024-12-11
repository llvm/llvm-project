// Pretend this is a system header
#pragma clang system_header
#include <ptrcheck.h>

// FIXME: We might not want bounds checks here because this file is in something
// that hasn't fully adopted (rdar://139815437)

// CHECK:      {{^}}|-FunctionDecl [[func_inline_header_ret_explicit_unspecified_cast_0:0x[^ ]+]] {{.+}} inline_header_ret_explicit_unspecified_cast_0
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_count:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-BoundsCheckExpr {{.+}} 'count == 0'
// CHECK-NEXT: {{^}}|       |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | `-OpaqueValueExpr [[ove:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       |   `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}|       |     `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|       | |-OpaqueValueExpr [[ove_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       | |   `-DeclRefExpr {{.+}} [[var_count]]
// CHECK-NEXT: {{^}}|       | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |-OpaqueValueExpr [[ove]]
// CHECK-NEXT: {{^}}|       | `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}|       |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       `-OpaqueValueExpr [[ove_1]]
// CHECK-NEXT: {{^}}|         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|           `-DeclRefExpr {{.+}} [[var_count]]
inline int* __counted_by(count) inline_header_ret_explicit_unspecified_cast_0(int count) {
  // Outside of system headers this implicit conversion **is allowed**
  return (int*)0;
}

// CHECK: {{^}}|-FunctionDecl [[func_inline_header_ret_explicit_unsafe_indexable_cast_0:0x[^ ]+]] {{.+}} inline_header_ret_explicit_unsafe_indexable_cast_0
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_count_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-BoundsCheckExpr {{.+}} 'count == 0'
// CHECK-NEXT: {{^}}|       |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | `-OpaqueValueExpr [[ove_2:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       |   `-CStyleCastExpr {{.+}} 'int *__unsafe_indexable' <NullToPointer>
// CHECK-NEXT: {{^}}|       |     `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|       | |-OpaqueValueExpr [[ove_3:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       | |   `-DeclRefExpr {{.+}} [[var_count_1]]
// CHECK-NEXT: {{^}}|       | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |-OpaqueValueExpr [[ove_2]]
// CHECK-NEXT: {{^}}|       | `-CStyleCastExpr {{.+}} 'int *__unsafe_indexable' <NullToPointer>
// CHECK-NEXT: {{^}}|       |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       `-OpaqueValueExpr [[ove_3]]
// CHECK-NEXT: {{^}}|         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|           `-DeclRefExpr {{.+}} [[var_count_1]]
inline int* __counted_by(count) inline_header_ret_explicit_unsafe_indexable_cast_0(int count) {
  // Outside of system headers this implicit conversion is not
  // allowed but it's allowed in system headers.
  return (int* __unsafe_indexable)0;
}

// CHECK: {{^}}|-FunctionDecl [[func_inline_header_ret_0:0x[^ ]+]] {{.+}} inline_header_ret_0
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_count_2:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-BoundsCheckExpr {{.+}} 'count == 0'
// CHECK-NEXT: {{^}}|       |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <NullToPointer>
// CHECK-NEXT: {{^}}|       | `-OpaqueValueExpr [[ove_4:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|       | |-OpaqueValueExpr [[ove_5:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       | |   `-DeclRefExpr {{.+}} [[var_count_2]]
// CHECK-NEXT: {{^}}|       | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |-OpaqueValueExpr [[ove_4]]
// CHECK-NEXT: {{^}}|       | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       `-OpaqueValueExpr [[ove_5]]
// CHECK-NEXT: {{^}}|         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|           `-DeclRefExpr {{.+}} [[var_count_2]]
inline int* __counted_by(count) inline_header_ret_0(int count) {
  // Outside of system headers this implicit conversion **is allowed**
  return 0;
}

// CHECK: {{^}}|-FunctionDecl [[func_inline_header_ret_void_star_unspecified_0:0x[^ ]+]] {{.+}} inline_header_ret_void_star_unspecified_0
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_count_3:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-BoundsCheckExpr {{.+}} 'count == 0'
// CHECK-NEXT: {{^}}|       |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | `-OpaqueValueExpr [[ove_6:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       |   `-CStyleCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT: {{^}}|       |     `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|       | |-OpaqueValueExpr [[ove_7:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       | |   `-DeclRefExpr {{.+}} [[var_count_3]]
// CHECK-NEXT: {{^}}|       | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |-OpaqueValueExpr [[ove_6]]
// CHECK-NEXT: {{^}}|       | `-CStyleCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT: {{^}}|       |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       `-OpaqueValueExpr [[ove_7]]
// CHECK-NEXT: {{^}}|         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|           `-DeclRefExpr {{.+}} [[var_count_3]]
inline int* __counted_by(count) inline_header_ret_void_star_unspecified_0(int count) {
  // Outside of system headers this implicit conversion **is allowed**
  return (void*)0;
}

// CHECK-NEXT: {{^}}|-FunctionDecl [[func_inline_header_ret_void_star_unsafe_indexable_0:0x[^ ]+]] {{.+}} inline_header_ret_void_star_unsafe_indexable_0
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_count_4:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-BoundsCheckExpr {{.+}} 'count == 0'
// CHECK-NEXT: {{^}}|       |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | `-OpaqueValueExpr [[ove_8:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       |   `-CStyleCastExpr {{.+}} 'void *__unsafe_indexable' <NullToPointer>
// CHECK-NEXT: {{^}}|       |     `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|       | |-OpaqueValueExpr [[ove_9:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       | |   `-DeclRefExpr {{.+}} [[var_count_4]]
// CHECK-NEXT: {{^}}|       | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |-OpaqueValueExpr [[ove_8]]
// CHECK-NEXT: {{^}}|       | `-CStyleCastExpr {{.+}} 'void *__unsafe_indexable' <NullToPointer>
// CHECK-NEXT: {{^}}|       |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       `-OpaqueValueExpr [[ove_9]]
// CHECK-NEXT: {{^}}|         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|           `-DeclRefExpr {{.+}} [[var_count_4]]
inline int* __counted_by(count) inline_header_ret_void_star_unsafe_indexable_0(int count) {
  // Outside of system headers this implicit conversion is not
  // allowed but it's allowed in system headers.
  return (void* __unsafe_indexable) 0;
}

// These are the checks for the AST from the `bounds-attributed-in-return-null-system-header.c`.
// They have to be here because that's what FileCheck requires.

// CHECK-NEXT: {{^}}|-FunctionDecl [[func_test_explicit_unspecified_cast_0:0x[^ ]+]] {{.+}} test_explicit_unspecified_cast_0
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_count_5:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-BoundsCheckExpr {{.+}} 'inline_header_ret_explicit_unspecified_cast_0(count) <= __builtin_get_pointer_upper_bound(inline_header_ret_explicit_unspecified_cast_0(count)) && __builtin_get_pointer_lower_bound(inline_header_ret_explicit_unspecified_cast_0(count)) <= inline_header_ret_explicit_unspecified_cast_0(count) && count <= __builtin_get_pointer_upper_bound(inline_header_ret_explicit_unspecified_cast_0(count)) - inline_header_ret_explicit_unspecified_cast_0(count) && 0 <= count'
// CHECK-NEXT: {{^}}|       |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | `-OpaqueValueExpr [[ove_10:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|       |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|       |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: {{^}}|       |     | | |-OpaqueValueExpr [[ove_11:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       |     | | | `-CallExpr
// CHECK-NEXT: {{^}}|       |     | | |   |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|       |     | | |   | `-DeclRefExpr {{.+}} [[func_inline_header_ret_explicit_unspecified_cast_0]]
// CHECK-NEXT: {{^}}|       |     | | |   `-OpaqueValueExpr [[ove_12:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|       |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: {{^}}|       |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       |     | | | | `-OpaqueValueExpr [[ove_11]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}|       |     | | | `-OpaqueValueExpr [[ove_12]] {{.*}} 'int'
// CHECK:      {{^}}|       |     | |-OpaqueValueExpr [[ove_12]]
// CHECK-NEXT: {{^}}|       |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       |     | |   `-DeclRefExpr {{.+}} [[var_count_5]]
// CHECK-NEXT: {{^}}|       |     | `-OpaqueValueExpr [[ove_11]]
// CHECK-NEXT: {{^}}|       |     |   `-CallExpr
// CHECK-NEXT: {{^}}|       |     |     |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|       |     |     | `-DeclRefExpr {{.+}} [[func_inline_header_ret_explicit_unspecified_cast_0]]
// CHECK-NEXT: {{^}}|       |     |     `-OpaqueValueExpr [[ove_12]] {{.*}} 'int'
// CHECK:      {{^}}|       |     |-OpaqueValueExpr [[ove_12]] {{.*}} 'int'
// CHECK:      {{^}}|       |     `-OpaqueValueExpr [[ove_11]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}|       |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|       | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|       | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|       | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | | | | `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|       | | | `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|       | | |   `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|       | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|       | |   |-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}|       | |   | `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|       | |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | |     `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|       | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|       |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|       |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|       |   | | `-OpaqueValueExpr [[ove_13:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|       |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}|       |   |   |-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|       |   |   | `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|       |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       |   |     `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|       |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|       |     |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |     `-OpaqueValueExpr [[ove_13]] {{.*}} 'int'
// CHECK:      {{^}}|       |-OpaqueValueExpr [[ove_10]]
// CHECK-NEXT: {{^}}|       | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|       |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|       |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: {{^}}|       |   | | |-OpaqueValueExpr [[ove_11]]
// CHECK-NEXT: {{^}}|       |   | | | `-CallExpr
// CHECK-NEXT: {{^}}|       |   | | |   |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|       |   | | |   | `-DeclRefExpr {{.+}} [[func_inline_header_ret_explicit_unspecified_cast_0]]
// CHECK-NEXT: {{^}}|       |   | | |   `-OpaqueValueExpr [[ove_12]] {{.*}} 'int'
// CHECK:      {{^}}|       |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: {{^}}|       |   | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       |   | | | | `-OpaqueValueExpr [[ove_11]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}|       |   | | | `-OpaqueValueExpr [[ove_12]] {{.*}} 'int'
// CHECK:      {{^}}|       |   | |-OpaqueValueExpr [[ove_12]]
// CHECK-NEXT: {{^}}|       |   | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       |   | |   `-DeclRefExpr {{.+}} [[var_count_5]]
// CHECK-NEXT: {{^}}|       |   | `-OpaqueValueExpr [[ove_11]]
// CHECK-NEXT: {{^}}|       |   |   `-CallExpr
// CHECK-NEXT: {{^}}|       |   |     |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|       |   |     | `-DeclRefExpr {{.+}} [[func_inline_header_ret_explicit_unspecified_cast_0]]
// CHECK-NEXT: {{^}}|       |   |     `-OpaqueValueExpr [[ove_12]] {{.*}} 'int'
// CHECK:      {{^}}|       |   |-OpaqueValueExpr [[ove_12]] {{.*}} 'int'
// CHECK:      {{^}}|       |   `-OpaqueValueExpr [[ove_11]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}|       `-OpaqueValueExpr [[ove_13]]
// CHECK-NEXT: {{^}}|         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|           `-DeclRefExpr {{.+}} [[var_count_5]]
// CHECK-NEXT: {{^}}|-FunctionDecl [[func_test_explicit_unsafe_indexable_cast_0:0x[^ ]+]] {{.+}} test_explicit_unsafe_indexable_cast_0
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_count_6:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-BoundsCheckExpr {{.+}} 'inline_header_ret_explicit_unsafe_indexable_cast_0(count) <= __builtin_get_pointer_upper_bound(inline_header_ret_explicit_unsafe_indexable_cast_0(count)) && __builtin_get_pointer_lower_bound(inline_header_ret_explicit_unsafe_indexable_cast_0(count)) <= inline_header_ret_explicit_unsafe_indexable_cast_0(count) && count <= __builtin_get_pointer_upper_bound(inline_header_ret_explicit_unsafe_indexable_cast_0(count)) - inline_header_ret_explicit_unsafe_indexable_cast_0(count) && 0 <= count'
// CHECK-NEXT: {{^}}|       |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | `-OpaqueValueExpr [[ove_14:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|       |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|       |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: {{^}}|       |     | | |-OpaqueValueExpr [[ove_15:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       |     | | | `-CallExpr
// CHECK-NEXT: {{^}}|       |     | | |   |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|       |     | | |   | `-DeclRefExpr {{.+}} [[func_inline_header_ret_explicit_unsafe_indexable_cast_0]]
// CHECK-NEXT: {{^}}|       |     | | |   `-OpaqueValueExpr [[ove_16:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|       |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: {{^}}|       |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       |     | | | | `-OpaqueValueExpr [[ove_15]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}|       |     | | | `-OpaqueValueExpr [[ove_16]] {{.*}} 'int'
// CHECK:      {{^}}|       |     | |-OpaqueValueExpr [[ove_16]]
// CHECK-NEXT: {{^}}|       |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       |     | |   `-DeclRefExpr {{.+}} [[var_count_6]]
// CHECK-NEXT: {{^}}|       |     | `-OpaqueValueExpr [[ove_15]]
// CHECK-NEXT: {{^}}|       |     |   `-CallExpr
// CHECK-NEXT: {{^}}|       |     |     |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|       |     |     | `-DeclRefExpr {{.+}} [[func_inline_header_ret_explicit_unsafe_indexable_cast_0]]
// CHECK-NEXT: {{^}}|       |     |     `-OpaqueValueExpr [[ove_16]] {{.*}} 'int'
// CHECK:      {{^}}|       |     |-OpaqueValueExpr [[ove_16]] {{.*}} 'int'
// CHECK:      {{^}}|       |     `-OpaqueValueExpr [[ove_15]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}|       |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|       | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|       | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|       | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | | | | `-OpaqueValueExpr [[ove_14]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|       | | | `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|       | | |   `-OpaqueValueExpr [[ove_14]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|       | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|       | |   |-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}|       | |   | `-OpaqueValueExpr [[ove_14]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|       | |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | |     `-OpaqueValueExpr [[ove_14]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|       | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|       |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|       |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|       |   | | `-OpaqueValueExpr [[ove_17:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|       |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}|       |   |   |-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|       |   |   | `-OpaqueValueExpr [[ove_14]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|       |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       |   |     `-OpaqueValueExpr [[ove_14]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|       |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|       |     |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |     `-OpaqueValueExpr [[ove_17]] {{.*}} 'int'
// CHECK:      {{^}}|       |-OpaqueValueExpr [[ove_14]]
// CHECK-NEXT: {{^}}|       | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|       |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|       |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: {{^}}|       |   | | |-OpaqueValueExpr [[ove_15]]
// CHECK-NEXT: {{^}}|       |   | | | `-CallExpr
// CHECK-NEXT: {{^}}|       |   | | |   |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|       |   | | |   | `-DeclRefExpr {{.+}} [[func_inline_header_ret_explicit_unsafe_indexable_cast_0]]
// CHECK-NEXT: {{^}}|       |   | | |   `-OpaqueValueExpr [[ove_16]] {{.*}} 'int'
// CHECK:      {{^}}|       |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: {{^}}|       |   | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       |   | | | | `-OpaqueValueExpr [[ove_15]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}|       |   | | | `-OpaqueValueExpr [[ove_16]] {{.*}} 'int'
// CHECK:      {{^}}|       |   | |-OpaqueValueExpr [[ove_16]]
// CHECK-NEXT: {{^}}|       |   | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       |   | |   `-DeclRefExpr {{.+}} [[var_count_6]]
// CHECK-NEXT: {{^}}|       |   | `-OpaqueValueExpr [[ove_15]]
// CHECK-NEXT: {{^}}|       |   |   `-CallExpr
// CHECK-NEXT: {{^}}|       |   |     |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|       |   |     | `-DeclRefExpr {{.+}} [[func_inline_header_ret_explicit_unsafe_indexable_cast_0]]
// CHECK-NEXT: {{^}}|       |   |     `-OpaqueValueExpr [[ove_16]] {{.*}} 'int'
// CHECK:      {{^}}|       |   |-OpaqueValueExpr [[ove_16]] {{.*}} 'int'
// CHECK:      {{^}}|       |   `-OpaqueValueExpr [[ove_15]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}|       `-OpaqueValueExpr [[ove_17]]
// CHECK-NEXT: {{^}}|         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|           `-DeclRefExpr {{.+}} [[var_count_6]]
// CHECK-NEXT: {{^}}|-FunctionDecl [[func_test_0:0x[^ ]+]] {{.+}} test_0
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_count_7:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-BoundsCheckExpr {{.+}} 'inline_header_ret_0(count) <= __builtin_get_pointer_upper_bound(inline_header_ret_0(count)) && __builtin_get_pointer_lower_bound(inline_header_ret_0(count)) <= inline_header_ret_0(count) && count <= __builtin_get_pointer_upper_bound(inline_header_ret_0(count)) - inline_header_ret_0(count) && 0 <= count'
// CHECK-NEXT: {{^}}|       |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | `-OpaqueValueExpr [[ove_18:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|       |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|       |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: {{^}}|       |     | | |-OpaqueValueExpr [[ove_19:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       |     | | | `-CallExpr
// CHECK-NEXT: {{^}}|       |     | | |   |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|       |     | | |   | `-DeclRefExpr {{.+}} [[func_inline_header_ret_0]]
// CHECK-NEXT: {{^}}|       |     | | |   `-OpaqueValueExpr [[ove_20:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|       |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: {{^}}|       |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       |     | | | | `-OpaqueValueExpr [[ove_19]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}|       |     | | | `-OpaqueValueExpr [[ove_20]] {{.*}} 'int'
// CHECK:      {{^}}|       |     | |-OpaqueValueExpr [[ove_20]]
// CHECK-NEXT: {{^}}|       |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       |     | |   `-DeclRefExpr {{.+}} [[var_count_7]]
// CHECK-NEXT: {{^}}|       |     | `-OpaqueValueExpr [[ove_19]]
// CHECK-NEXT: {{^}}|       |     |   `-CallExpr
// CHECK-NEXT: {{^}}|       |     |     |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|       |     |     | `-DeclRefExpr {{.+}} [[func_inline_header_ret_0]]
// CHECK-NEXT: {{^}}|       |     |     `-OpaqueValueExpr [[ove_20]] {{.*}} 'int'
// CHECK:      {{^}}|       |     |-OpaqueValueExpr [[ove_20]] {{.*}} 'int'
// CHECK:      {{^}}|       |     `-OpaqueValueExpr [[ove_19]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}|       |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|       | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|       | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|       | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | | | | `-OpaqueValueExpr [[ove_18]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|       | | | `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|       | | |   `-OpaqueValueExpr [[ove_18]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|       | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|       | |   |-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}|       | |   | `-OpaqueValueExpr [[ove_18]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|       | |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | |     `-OpaqueValueExpr [[ove_18]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|       | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|       |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|       |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|       |   | | `-OpaqueValueExpr [[ove_21:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|       |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}|       |   |   |-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|       |   |   | `-OpaqueValueExpr [[ove_18]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|       |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       |   |     `-OpaqueValueExpr [[ove_18]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|       |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|       |     |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |     `-OpaqueValueExpr [[ove_21]] {{.*}} 'int'
// CHECK:      {{^}}|       |-OpaqueValueExpr [[ove_18]]
// CHECK-NEXT: {{^}}|       | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|       |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|       |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: {{^}}|       |   | | |-OpaqueValueExpr [[ove_19]]
// CHECK-NEXT: {{^}}|       |   | | | `-CallExpr
// CHECK-NEXT: {{^}}|       |   | | |   |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|       |   | | |   | `-DeclRefExpr {{.+}} [[func_inline_header_ret_0]]
// CHECK-NEXT: {{^}}|       |   | | |   `-OpaqueValueExpr [[ove_20]] {{.*}} 'int'
// CHECK:      {{^}}|       |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: {{^}}|       |   | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       |   | | | | `-OpaqueValueExpr [[ove_19]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}|       |   | | | `-OpaqueValueExpr [[ove_20]] {{.*}} 'int'
// CHECK:      {{^}}|       |   | |-OpaqueValueExpr [[ove_20]]
// CHECK-NEXT: {{^}}|       |   | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       |   | |   `-DeclRefExpr {{.+}} [[var_count_7]]
// CHECK-NEXT: {{^}}|       |   | `-OpaqueValueExpr [[ove_19]]
// CHECK-NEXT: {{^}}|       |   |   `-CallExpr
// CHECK-NEXT: {{^}}|       |   |     |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|       |   |     | `-DeclRefExpr {{.+}} [[func_inline_header_ret_0]]
// CHECK-NEXT: {{^}}|       |   |     `-OpaqueValueExpr [[ove_20]] {{.*}} 'int'
// CHECK:      {{^}}|       |   |-OpaqueValueExpr [[ove_20]] {{.*}} 'int'
// CHECK:      {{^}}|       |   `-OpaqueValueExpr [[ove_19]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}|       `-OpaqueValueExpr [[ove_21]]
// CHECK-NEXT: {{^}}|         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|           `-DeclRefExpr {{.+}} [[var_count_7]]
// CHECK-NEXT: {{^}}|-FunctionDecl [[func_test_void_star_unspecified_0:0x[^ ]+]] {{.+}} test_void_star_unspecified_0
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_count_8:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-BoundsCheckExpr {{.+}} 'inline_header_ret_void_star_unspecified_0(count) <= __builtin_get_pointer_upper_bound(inline_header_ret_void_star_unspecified_0(count)) && __builtin_get_pointer_lower_bound(inline_header_ret_void_star_unspecified_0(count)) <= inline_header_ret_void_star_unspecified_0(count) && count <= __builtin_get_pointer_upper_bound(inline_header_ret_void_star_unspecified_0(count)) - inline_header_ret_void_star_unspecified_0(count) && 0 <= count'
// CHECK-NEXT: {{^}}|       |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | `-OpaqueValueExpr [[ove_22:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|       |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|       |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: {{^}}|       |     | | |-OpaqueValueExpr [[ove_23:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       |     | | | `-CallExpr
// CHECK-NEXT: {{^}}|       |     | | |   |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|       |     | | |   | `-DeclRefExpr {{.+}} [[func_inline_header_ret_void_star_unspecified_0]]
// CHECK-NEXT: {{^}}|       |     | | |   `-OpaqueValueExpr [[ove_24:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|       |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: {{^}}|       |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       |     | | | | `-OpaqueValueExpr [[ove_23]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}|       |     | | | `-OpaqueValueExpr [[ove_24]] {{.*}} 'int'
// CHECK:      {{^}}|       |     | |-OpaqueValueExpr [[ove_24]]
// CHECK-NEXT: {{^}}|       |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       |     | |   `-DeclRefExpr {{.+}} [[var_count_8]]
// CHECK-NEXT: {{^}}|       |     | `-OpaqueValueExpr [[ove_23]]
// CHECK-NEXT: {{^}}|       |     |   `-CallExpr
// CHECK-NEXT: {{^}}|       |     |     |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|       |     |     | `-DeclRefExpr {{.+}} [[func_inline_header_ret_void_star_unspecified_0]]
// CHECK-NEXT: {{^}}|       |     |     `-OpaqueValueExpr [[ove_24]] {{.*}} 'int'
// CHECK:      {{^}}|       |     |-OpaqueValueExpr [[ove_24]] {{.*}} 'int'
// CHECK:      {{^}}|       |     `-OpaqueValueExpr [[ove_23]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}|       |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|       | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|       | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|       | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | | | | `-OpaqueValueExpr [[ove_22]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|       | | | `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|       | | |   `-OpaqueValueExpr [[ove_22]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|       | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|       | |   |-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}|       | |   | `-OpaqueValueExpr [[ove_22]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|       | |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | |     `-OpaqueValueExpr [[ove_22]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|       | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|       |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|       |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|       |   | | `-OpaqueValueExpr [[ove_25:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|       |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}|       |   |   |-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|       |   |   | `-OpaqueValueExpr [[ove_22]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|       |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       |   |     `-OpaqueValueExpr [[ove_22]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|       |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|       |     |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |     `-OpaqueValueExpr [[ove_25]] {{.*}} 'int'
// CHECK:      {{^}}|       |-OpaqueValueExpr [[ove_22]]
// CHECK-NEXT: {{^}}|       | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|       |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|       |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: {{^}}|       |   | | |-OpaqueValueExpr [[ove_23]]
// CHECK-NEXT: {{^}}|       |   | | | `-CallExpr
// CHECK-NEXT: {{^}}|       |   | | |   |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|       |   | | |   | `-DeclRefExpr {{.+}} [[func_inline_header_ret_void_star_unspecified_0]]
// CHECK-NEXT: {{^}}|       |   | | |   `-OpaqueValueExpr [[ove_24]] {{.*}} 'int'
// CHECK:      {{^}}|       |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: {{^}}|       |   | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       |   | | | | `-OpaqueValueExpr [[ove_23]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}|       |   | | | `-OpaqueValueExpr [[ove_24]] {{.*}} 'int'
// CHECK:      {{^}}|       |   | |-OpaqueValueExpr [[ove_24]]
// CHECK-NEXT: {{^}}|       |   | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       |   | |   `-DeclRefExpr {{.+}} [[var_count_8]]
// CHECK-NEXT: {{^}}|       |   | `-OpaqueValueExpr [[ove_23]]
// CHECK-NEXT: {{^}}|       |   |   `-CallExpr
// CHECK-NEXT: {{^}}|       |   |     |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|       |   |     | `-DeclRefExpr {{.+}} [[func_inline_header_ret_void_star_unspecified_0]]
// CHECK-NEXT: {{^}}|       |   |     `-OpaqueValueExpr [[ove_24]] {{.*}} 'int'
// CHECK:      {{^}}|       |   |-OpaqueValueExpr [[ove_24]] {{.*}} 'int'
// CHECK:      {{^}}|       |   `-OpaqueValueExpr [[ove_23]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}|       `-OpaqueValueExpr [[ove_25]]
// CHECK-NEXT: {{^}}|         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|           `-DeclRefExpr {{.+}} [[var_count_8]]
// CHECK-NEXT: {{^}}`-FunctionDecl [[func_test_void_star_unsafe_indexable_0:0x[^ ]+]] {{.+}} test_void_star_unsafe_indexable_0
// CHECK-NEXT: {{^}}  |-ParmVarDecl [[var_count_9:0x[^ ]+]]
// CHECK-NEXT: {{^}}  `-CompoundStmt
// CHECK-NEXT: {{^}}    `-ReturnStmt
// CHECK-NEXT: {{^}}      `-BoundsCheckExpr {{.+}} 'inline_header_ret_void_star_unsafe_indexable_0(count) <= __builtin_get_pointer_upper_bound(inline_header_ret_void_star_unsafe_indexable_0(count)) && __builtin_get_pointer_lower_bound(inline_header_ret_void_star_unsafe_indexable_0(count)) <= inline_header_ret_void_star_unsafe_indexable_0(count) && count <= __builtin_get_pointer_upper_bound(inline_header_ret_void_star_unsafe_indexable_0(count)) - inline_header_ret_void_star_unsafe_indexable_0(count) && 0 <= count'
// CHECK-NEXT: {{^}}        |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}        | `-OpaqueValueExpr [[ove_26:0x[^ ]+]]
// CHECK-NEXT: {{^}}        |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}        |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}        |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: {{^}}        |     | | |-OpaqueValueExpr [[ove_27:0x[^ ]+]]
// CHECK-NEXT: {{^}}        |     | | | `-CallExpr
// CHECK-NEXT: {{^}}        |     | | |   |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}        |     | | |   | `-DeclRefExpr {{.+}} [[func_inline_header_ret_void_star_unsafe_indexable_0]]
// CHECK-NEXT: {{^}}        |     | | |   `-OpaqueValueExpr [[ove_28:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}        |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: {{^}}        |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}        |     | | | | `-OpaqueValueExpr [[ove_27]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}        |     | | | `-OpaqueValueExpr [[ove_28]] {{.*}} 'int'
// CHECK:      {{^}}        |     | |-OpaqueValueExpr [[ove_28]]
// CHECK-NEXT: {{^}}        |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}        |     | |   `-DeclRefExpr {{.+}} [[var_count_9]]
// CHECK-NEXT: {{^}}        |     | `-OpaqueValueExpr [[ove_27]]
// CHECK-NEXT: {{^}}        |     |   `-CallExpr
// CHECK-NEXT: {{^}}        |     |     |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}        |     |     | `-DeclRefExpr {{.+}} [[func_inline_header_ret_void_star_unsafe_indexable_0]]
// CHECK-NEXT: {{^}}        |     |     `-OpaqueValueExpr [[ove_28]] {{.*}} 'int'
// CHECK:      {{^}}        |     |-OpaqueValueExpr [[ove_28]] {{.*}} 'int'
// CHECK:      {{^}}        |     `-OpaqueValueExpr [[ove_27]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}        |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}        | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}        | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}        | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}        | | | | `-OpaqueValueExpr [[ove_26]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}        | | | `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}        | | |   `-OpaqueValueExpr [[ove_26]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}        | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}        | |   |-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}        | |   | `-OpaqueValueExpr [[ove_26]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}        | |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}        | |     `-OpaqueValueExpr [[ove_26]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}        | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}        |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}        |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}        |   | | `-OpaqueValueExpr [[ove_29:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}        |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}        |   |   |-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}        |   |   | `-OpaqueValueExpr [[ove_26]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}        |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}        |   |     `-OpaqueValueExpr [[ove_26]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}        |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}        |     |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}        |     `-OpaqueValueExpr [[ove_29]] {{.*}} 'int'
// CHECK:      {{^}}        |-OpaqueValueExpr [[ove_26]]
// CHECK-NEXT: {{^}}        | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}        |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}        |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: {{^}}        |   | | |-OpaqueValueExpr [[ove_27]]
// CHECK-NEXT: {{^}}        |   | | | `-CallExpr
// CHECK-NEXT: {{^}}        |   | | |   |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}        |   | | |   | `-DeclRefExpr {{.+}} [[func_inline_header_ret_void_star_unsafe_indexable_0]]
// CHECK-NEXT: {{^}}        |   | | |   `-OpaqueValueExpr [[ove_28]] {{.*}} 'int'
// CHECK:      {{^}}        |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: {{^}}        |   | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}        |   | | | | `-OpaqueValueExpr [[ove_27]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}        |   | | | `-OpaqueValueExpr [[ove_28]] {{.*}} 'int'
// CHECK:      {{^}}        |   | |-OpaqueValueExpr [[ove_28]]
// CHECK-NEXT: {{^}}        |   | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}        |   | |   `-DeclRefExpr {{.+}} [[var_count_9]]
// CHECK-NEXT: {{^}}        |   | `-OpaqueValueExpr [[ove_27]]
// CHECK-NEXT: {{^}}        |   |   `-CallExpr
// CHECK-NEXT: {{^}}        |   |     |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}        |   |     | `-DeclRefExpr {{.+}} [[func_inline_header_ret_void_star_unsafe_indexable_0]]
// CHECK-NEXT: {{^}}        |   |     `-OpaqueValueExpr [[ove_28]] {{.*}} 'int'
// CHECK:      {{^}}        |   |-OpaqueValueExpr [[ove_28]] {{.*}} 'int'
// CHECK:      {{^}}        |   `-OpaqueValueExpr [[ove_27]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}        `-OpaqueValueExpr [[ove_29]]
// CHECK-NEXT: {{^}}          `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}            `-DeclRefExpr {{.+}} [[var_count_9]]
