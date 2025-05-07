#include <ptrcheck.h>
#include <builtin-function-sys.h>

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s -I %S/include | FileCheck %s --implicit-check-not "GetBoundExpr {{.+}} 'char *__single'" --implicit-check-not "GetBoundExpr {{.+}} 'char *'"
// RUN: %clang_cc1 -ast-dump -fbounds-safety %s -I %S/include -x objective-c -fexperimental-bounds-safety-objc | FileCheck %s --implicit-check-not "GetBoundExpr {{.+}} 'char *__single'" --implicit-check-not "GetBoundExpr {{.+}} 'char *'"

char * __counted_by(len) func(char * __counted_by(len) src_str, int len) {
  int len2 = 0;
  char * __counted_by(len2) dst_str;
  dst_str = __unsafe_forge_bidi_indexable(char*, malloc(len), len);
  len2 = len;
  memcpy(dst_str, src_str, len);
  return dst_str;
}

// CHECK:      {{^}}TranslationUnitDecl
// CHECK:      {{^}}|-FunctionDecl [[func_static:0x[^ ]+]] {{.+}} static
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_dst:0x[^ ]+]]
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_src:0x[^ ]+]]
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_size:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'void *__bidi_indexable'
// CHECK-NEXT: {{^}}|     | | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single'
// CHECK:      {{^}}|     | | |   | `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'void *'
// CHECK:      {{^}}|     | | |   | `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'void *'
// CHECK:      {{^}}|     | | |   `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'unsigned long'
// CHECK:      {{^}}|     | | |-ImplicitCastExpr {{.+}} 'void *' <BitCast>
// CHECK-NEXT: {{^}}|     | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: {{^}}|     | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK-NEXT: {{^}}|     | | |   | `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | | |   |   `-OpaqueValueExpr [[ove]] {{.*}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single'
// CHECK:      {{^}}|     | | |   `-AssumptionExpr
// CHECK-NEXT: {{^}}|     | | |     |-OpaqueValueExpr [[ove_3]] {{.*}} 'unsigned long'
// CHECK:      {{^}}|     | | |     `-BinaryOperator {{.+}} 'int' '>='
// CHECK-NEXT: {{^}}|     | | |       |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|     | | |       | `-OpaqueValueExpr [[ove_3]] {{.*}} 'unsigned long'
// CHECK:      {{^}}|     | | |       `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|     | | |         `-IntegerLiteral {{.+}} 0
// CHECK:      {{^}}|     | |-OpaqueValueExpr [[ove_1]]
// CHECK-NEXT: {{^}}|     | | `-ImplicitCastExpr {{.+}} 'void *' <LValueToRValue>
// CHECK-NEXT: {{^}}|     | |   `-DeclRefExpr {{.+}} [[var_dst]]
// CHECK-NEXT: {{^}}|     | |-OpaqueValueExpr [[ove_2]]
// CHECK-NEXT: {{^}}|     | | `-ImplicitCastExpr {{.+}} 'void *' <LValueToRValue>
// CHECK-NEXT: {{^}}|     | |   `-DeclRefExpr {{.+}} [[var_src]]
// CHECK-NEXT: {{^}}|     | |-OpaqueValueExpr [[ove_3]]
// CHECK-NEXT: {{^}}|     | | `-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK-NEXT: {{^}}|     | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|     | |     `-DeclRefExpr {{.+}} [[var_size]]
// CHECK-NEXT: {{^}}|     | `-OpaqueValueExpr [[ove]]
// CHECK-NEXT: {{^}}|     |   `-CallExpr
// CHECK-NEXT: {{^}}|     |     |-ImplicitCastExpr {{.+}} 'void *__single __sized_by(function-parameter-0-2)(*)(void *__single __sized_by(function-parameter-0-2), const void *__single __sized_by(function-parameter-0-2), unsigned long)' <BuiltinFnToFnPtr>
// CHECK-NEXT: {{^}}|     |     | `-DeclRefExpr {{.+}}
// CHECK-NEXT: {{^}}|     |     |-ImplicitCastExpr {{.+}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     |     | `-OpaqueValueExpr [[ove_1]] {{.*}} 'void *'
// CHECK:      {{^}}|     |     |-ImplicitCastExpr {{.+}} 'const void *__single __sized_by(function-parameter-0-2)':'const void *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     |     | `-OpaqueValueExpr [[ove_2]] {{.*}} 'void *'
// CHECK:      {{^}}|     |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'unsigned long'
// CHECK:      {{^}}|     |-OpaqueValueExpr [[ove_1]] {{.*}} 'void *'
// CHECK:      {{^}}|     |-OpaqueValueExpr [[ove_2]] {{.*}} 'void *'
// CHECK:      {{^}}|     |-OpaqueValueExpr [[ove_3]] {{.*}} 'unsigned long'
// CHECK:      {{^}}|     `-OpaqueValueExpr [[ove]] {{.*}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single'
// CHECK:      {{^}}|-FunctionDecl [[func_static_1:0x[^ ]+]] {{.+}} static
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_dst_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_src_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_size_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'void *__bidi_indexable'
// CHECK-NEXT: {{^}}|         | | |-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single'
// CHECK:      {{^}}|         | | |   | `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'void *'
// CHECK:      {{^}}|         | | |   | `-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'void *'
// CHECK:      {{^}}|         | | |   `-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'unsigned long'
// CHECK:      {{^}}|         | | |-ImplicitCastExpr {{.+}} 'void *' <BitCast>
// CHECK-NEXT: {{^}}|         | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: {{^}}|         | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK-NEXT: {{^}}|         | | |   | `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|         | | |   |   `-OpaqueValueExpr [[ove_4]] {{.*}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single'
// CHECK:      {{^}}|         | | |   `-AssumptionExpr
// CHECK-NEXT: {{^}}|         | | |     |-OpaqueValueExpr [[ove_7]] {{.*}} 'unsigned long'
// CHECK:      {{^}}|         | | |     `-BinaryOperator {{.+}} 'int' '>='
// CHECK-NEXT: {{^}}|         | | |       |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|         | | |       | `-OpaqueValueExpr [[ove_7]] {{.*}} 'unsigned long'
// CHECK:      {{^}}|         | | |       `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|         | | |         `-IntegerLiteral {{.+}} 0
// CHECK:      {{^}}|         | |-OpaqueValueExpr [[ove_5]]
// CHECK-NEXT: {{^}}|         | | `-ImplicitCastExpr {{.+}} 'void *' <LValueToRValue>
// CHECK-NEXT: {{^}}|         | |   `-DeclRefExpr {{.+}} [[var_dst_1]]
// CHECK-NEXT: {{^}}|         | |-OpaqueValueExpr [[ove_6]]
// CHECK-NEXT: {{^}}|         | | `-ImplicitCastExpr {{.+}} 'void *' <LValueToRValue>
// CHECK-NEXT: {{^}}|         | |   `-DeclRefExpr {{.+}} [[var_src_1]]
// CHECK-NEXT: {{^}}|         | |-OpaqueValueExpr [[ove_7]]
// CHECK-NEXT: {{^}}|         | | `-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK-NEXT: {{^}}|         | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|         | |     `-DeclRefExpr {{.+}} [[var_size_1]]
// CHECK-NEXT: {{^}}|         | `-OpaqueValueExpr [[ove_4]]
// CHECK-NEXT: {{^}}|         |   `-CallExpr
// CHECK-NEXT: {{^}}|         |     |-ImplicitCastExpr {{.+}} 'void *__single __sized_by(function-parameter-0-2)(*)(void *__single __sized_by(function-parameter-0-2), const void *__single __sized_by(function-parameter-0-2), unsigned long)' <BuiltinFnToFnPtr>
// CHECK-NEXT: {{^}}|         |     | `-DeclRefExpr {{.+}}
// CHECK-NEXT: {{^}}|         |     |-ImplicitCastExpr {{.+}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|         |     | `-OpaqueValueExpr [[ove_5]] {{.*}} 'void *'
// CHECK:      {{^}}|         |     |-ImplicitCastExpr {{.+}} 'const void *__single __sized_by(function-parameter-0-2)':'const void *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|         |     | `-OpaqueValueExpr [[ove_6]] {{.*}} 'void *'
// CHECK:      {{^}}|         |     `-OpaqueValueExpr [[ove_7]] {{.*}} 'unsigned long'
// CHECK:      {{^}}|         |-OpaqueValueExpr [[ove_5]] {{.*}} 'void *'
// CHECK:      {{^}}|         |-OpaqueValueExpr [[ove_6]] {{.*}} 'void *'
// CHECK:      {{^}}|         |-OpaqueValueExpr [[ove_7]] {{.*}} 'unsigned long'
// CHECK:      {{^}}|         `-OpaqueValueExpr [[ove_4]] {{.*}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single'
// CHECK:      {{^}}|-FunctionDecl [[func_static_2:0x[^ ]+]] {{.+}} static
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_dst_2:0x[^ ]+]]
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_src_2:0x[^ ]+]]
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_size_2:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_tmp:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   |     `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|   |       |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'void *__bidi_indexable'
// CHECK-NEXT: {{^}}|   |       | | |-OpaqueValueExpr [[ove_8:0x[^ ]+]] {{.*}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single'
// CHECK:      {{^}}|   |       | | |   | `-OpaqueValueExpr [[ove_9:0x[^ ]+]] {{.*}} 'void *'
// CHECK:      {{^}}|   |       | | |   | `-OpaqueValueExpr [[ove_10:0x[^ ]+]] {{.*}} 'void *'
// CHECK:      {{^}}|   |       | | |   `-OpaqueValueExpr [[ove_11:0x[^ ]+]] {{.*}} 'unsigned long'
// CHECK:      {{^}}|   |       | | |-ImplicitCastExpr {{.+}} 'void *' <BitCast>
// CHECK-NEXT: {{^}}|   |       | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: {{^}}|   |       | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK-NEXT: {{^}}|   |       | | |   | `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   |       | | |   |   `-OpaqueValueExpr [[ove_8]] {{.*}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single'
// CHECK:      {{^}}|   |       | | |   `-AssumptionExpr
// CHECK-NEXT: {{^}}|   |       | | |     |-OpaqueValueExpr [[ove_11]] {{.*}} 'unsigned long'
// CHECK:      {{^}}|   |       | | |     `-BinaryOperator {{.+}} 'int' '>='
// CHECK-NEXT: {{^}}|   |       | | |       |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|   |       | | |       | `-OpaqueValueExpr [[ove_11]] {{.*}} 'unsigned long'
// CHECK:      {{^}}|   |       | | |       `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|   |       | | |         `-IntegerLiteral {{.+}} 0
// CHECK:      {{^}}|   |       | |-OpaqueValueExpr [[ove_9]]
// CHECK-NEXT: {{^}}|   |       | | `-ImplicitCastExpr {{.+}} 'void *' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |       | |   `-DeclRefExpr {{.+}} [[var_dst_2]]
// CHECK-NEXT: {{^}}|   |       | |-OpaqueValueExpr [[ove_10]]
// CHECK-NEXT: {{^}}|   |       | | `-ImplicitCastExpr {{.+}} 'void *' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |       | |   `-DeclRefExpr {{.+}} [[var_src_2]]
// CHECK-NEXT: {{^}}|   |       | |-OpaqueValueExpr [[ove_11]]
// CHECK-NEXT: {{^}}|   |       | | `-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK-NEXT: {{^}}|   |       | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |       | |     `-DeclRefExpr {{.+}} [[var_size_2]]
// CHECK-NEXT: {{^}}|   |       | `-OpaqueValueExpr [[ove_8]]
// CHECK-NEXT: {{^}}|   |       |   `-CallExpr
// CHECK-NEXT: {{^}}|   |       |     |-ImplicitCastExpr {{.+}} 'void *__single __sized_by(function-parameter-0-2)(*)(void *__single __sized_by(function-parameter-0-2), const void *__single __sized_by(function-parameter-0-2), unsigned long)' <BuiltinFnToFnPtr>
// CHECK-NEXT: {{^}}|   |       |     | `-DeclRefExpr {{.+}}
// CHECK-NEXT: {{^}}|   |       |     |-ImplicitCastExpr {{.+}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   |       |     | `-OpaqueValueExpr [[ove_9]] {{.*}} 'void *'
// CHECK:      {{^}}|   |       |     |-ImplicitCastExpr {{.+}} 'const void *__single __sized_by(function-parameter-0-2)':'const void *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   |       |     | `-OpaqueValueExpr [[ove_10]] {{.*}} 'void *'
// CHECK:      {{^}}|   |       |     `-OpaqueValueExpr [[ove_11]] {{.*}} 'unsigned long'
// CHECK:      {{^}}|   |       |-OpaqueValueExpr [[ove_9]] {{.*}} 'void *'
// CHECK:      {{^}}|   |       |-OpaqueValueExpr [[ove_10]] {{.*}} 'void *'
// CHECK:      {{^}}|   |       |-OpaqueValueExpr [[ove_11]] {{.*}} 'unsigned long'
// CHECK:      {{^}}|   |       `-OpaqueValueExpr [[ove_8]] {{.*}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single'
// CHECK:      {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-ImplicitCastExpr {{.+}} 'void *' <LValueToRValue>
// CHECK-NEXT: {{^}}|       `-DeclRefExpr {{.+}} [[var_tmp]]
// CHECK-NEXT: {{^}}|-FunctionDecl [[func_static_3:0x[^ ]+]] {{.+}} static
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_size_3:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-CallExpr
// CHECK-NEXT: {{^}}|       |-ImplicitCastExpr {{.+}} 'void *(*)(unsigned long)' <BuiltinFnToFnPtr>
// CHECK-NEXT: {{^}}|       | `-DeclRefExpr {{.+}}
// CHECK-NEXT: {{^}}|       `-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK-NEXT: {{^}}|         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|           `-DeclRefExpr {{.+}} [[var_size_3]]
// CHECK:      {{^}}|-FunctionDecl [[func_static_4:0x[^ ]+]] {{.+}} static
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_size_4:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_tmp_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-CallExpr
// CHECK-NEXT: {{^}}|   |     |-ImplicitCastExpr {{.+}} 'void *(*)(unsigned long)' <BuiltinFnToFnPtr>
// CHECK-NEXT: {{^}}|   |     | `-DeclRefExpr {{.+}}
// CHECK-NEXT: {{^}}|   |     `-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK-NEXT: {{^}}|   |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |         `-DeclRefExpr {{.+}} [[var_size_4]]
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-ImplicitCastExpr {{.+}} 'void *' <LValueToRValue>
// CHECK-NEXT: {{^}}|       `-DeclRefExpr {{.+}} [[var_tmp_1]]
// CHECK-NEXT: {{^}}`-FunctionDecl [[func_func:0x[^ ]+]] {{.+}} func
// CHECK-NEXT: {{^}}  |-ParmVarDecl [[var_src_str:0x[^ ]+]]
// CHECK-NEXT: {{^}}  |-ParmVarDecl [[var_len:0x[^ ]+]]
// CHECK-NEXT: {{^}}  | `-DependerDeclsAttr
// CHECK-NEXT: {{^}}  `-CompoundStmt
// CHECK-NEXT: {{^}}    |-DeclStmt
// CHECK-NEXT: {{^}}    | `-VarDecl [[var_len2:0x[^ ]+]]
// CHECK-NEXT: {{^}}    |   |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}    |   `-DependerDeclsAttr
// CHECK-NEXT: {{^}}    |-DeclStmt
// CHECK-NEXT: {{^}}    | `-VarDecl [[var_dst_str:0x[^ ]+]]
// CHECK-NEXT: {{^}}    |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}    | |-BoundsCheckExpr {{.+}} '((char *__bidi_indexable)__builtin_unsafe_forge_bidi_indexable((malloc(len)), (len))) <= __builtin_get_pointer_upper_bound(((char *__bidi_indexable)__builtin_unsafe_forge_bidi_indexable((malloc(len)), (len)))) && __builtin_get_pointer_lower_bound(((char *__bidi_indexable)__builtin_unsafe_forge_bidi_indexable((malloc(len)), (len)))) <= ((char *__bidi_indexable)__builtin_unsafe_forge_bidi_indexable((malloc(len)), (len))) && len <= __builtin_get_pointer_upper_bound(((char *__bidi_indexable)__builtin_unsafe_forge_bidi_indexable((malloc(len)), (len)))) - ((char *__bidi_indexable)__builtin_unsafe_forge_bidi_indexable((malloc(len)), (len))) && 0 <= len'
// CHECK-NEXT: {{^}}    | | |-BinaryOperator {{.+}} 'char *__single __counted_by(len2)':'char *__single' '='
// CHECK-NEXT: {{^}}    | | | |-DeclRefExpr {{.+}} [[var_dst_str]]
// CHECK-NEXT: {{^}}    | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(len2)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    | | |   `-OpaqueValueExpr [[ove_12:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}    | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}    | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}    | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    | |   | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    | |   | | | `-OpaqueValueExpr [[ove_12]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}    | |   | | `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}    | |   | |   `-OpaqueValueExpr [[ove_12]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}    | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    | |   |   |-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}    | |   |   | `-OpaqueValueExpr [[ove_12]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}    | |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    | |   |     `-OpaqueValueExpr [[ove_12]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}    | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}    | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}    | |     | | `-OpaqueValueExpr [[ove_13:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}    | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}    | |     |   |-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}    | |     |   | `-OpaqueValueExpr [[ove_12]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}    | |     |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    | |     |     `-OpaqueValueExpr [[ove_12]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}    | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    | |       |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}    | |       `-OpaqueValueExpr [[ove_13]] {{.*}} 'int'
// CHECK:      {{^}}    | |-OpaqueValueExpr [[ove_12]]
// CHECK-NEXT: {{^}}    | | `-ParenExpr
// CHECK-NEXT: {{^}}    | |   `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}    | |     `-ForgePtrExpr
// CHECK-NEXT: {{^}}    | |       |-ParenExpr
// CHECK-NEXT: {{^}}    | |       | `-CallExpr
// CHECK-NEXT: {{^}}    | |       |   |-ImplicitCastExpr {{.+}} 'void *(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}    | |       |   | `-DeclRefExpr {{.+}} [[func_static_3]]
// CHECK-NEXT: {{^}}    | |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}    | |       |     `-DeclRefExpr {{.+}} [[var_len]]
// CHECK-NEXT: {{^}}    | |       |-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK-NEXT: {{^}}    | |       | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}    | |       |   `-ParenExpr
// CHECK-NEXT: {{^}}    | |       |     `-DeclRefExpr {{.+}} [[var_len]]
// CHECK:      {{^}}    | `-OpaqueValueExpr [[ove_13]]
// CHECK-NEXT: {{^}}    |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}    |     `-DeclRefExpr {{.+}} [[var_len]]
// CHECK-NEXT: {{^}}    |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}    | |-BinaryOperator {{.+}} 'int' '='
// CHECK-NEXT: {{^}}    | | |-DeclRefExpr {{.+}} [[var_len2]]
// CHECK-NEXT: {{^}}    | | `-OpaqueValueExpr [[ove_13]] {{.*}} 'int'
// CHECK:      {{^}}    | |-OpaqueValueExpr [[ove_12]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}    | `-OpaqueValueExpr [[ove_13]] {{.*}} 'int'
// CHECK:      {{^}}    |-CallExpr
// CHECK-NEXT: {{^}}    | |-ImplicitCastExpr {{.+}} 'void (*__single)(void *restrict, void *restrict, int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}    | | `-DeclRefExpr {{.+}} [[func_static]]
// CHECK-NEXT: {{^}}    | |-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    | | `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}    | |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}    | |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}    | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: {{^}}    | |     | | |-OpaqueValueExpr [[ove_14:0x[^ ]+]] {{.*}} 'char *__single __counted_by(len2)':'char *__single'
// CHECK:      {{^}}    | |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: {{^}}    | |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    | |     | | | | `-OpaqueValueExpr [[ove_14]] {{.*}} 'char *__single __counted_by(len2)':'char *__single'
// CHECK:      {{^}}    | |     | | | `-OpaqueValueExpr [[ove_15:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}    | |     | |-OpaqueValueExpr [[ove_14]]
// CHECK-NEXT: {{^}}    | |     | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(len2)':'char *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}    | |     | |   `-DeclRefExpr {{.+}} [[var_dst_str]]
// CHECK-NEXT: {{^}}    | |     | `-OpaqueValueExpr [[ove_15]]
// CHECK-NEXT: {{^}}    | |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}    | |     |     `-DeclRefExpr {{.+}} [[var_len2]]
// CHECK-NEXT: {{^}}    | |     |-OpaqueValueExpr [[ove_14]] {{.*}} 'char *__single __counted_by(len2)':'char *__single'
// CHECK:      {{^}}    | |     `-OpaqueValueExpr [[ove_15]] {{.*}} 'int'
// CHECK:      {{^}}    | |-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    | | `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}    | |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}    | |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}    | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: {{^}}    | |     | | |-OpaqueValueExpr [[ove_16:0x[^ ]+]] {{.*}} 'char *__single __counted_by(len)':'char *__single'
// CHECK:      {{^}}    | |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: {{^}}    | |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    | |     | | | | `-OpaqueValueExpr [[ove_16]] {{.*}} 'char *__single __counted_by(len)':'char *__single'
// CHECK:      {{^}}    | |     | | | `-OpaqueValueExpr [[ove_17:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}    | |     | |-OpaqueValueExpr [[ove_16]]
// CHECK-NEXT: {{^}}    | |     | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(len)':'char *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}    | |     | |   `-DeclRefExpr {{.+}} [[var_src_str]]
// CHECK-NEXT: {{^}}    | |     | `-OpaqueValueExpr [[ove_17]]
// CHECK-NEXT: {{^}}    | |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}    | |     |     `-DeclRefExpr {{.+}} [[var_len]]
// CHECK-NEXT: {{^}}    | |     |-OpaqueValueExpr [[ove_16]] {{.*}} 'char *__single __counted_by(len)':'char *__single'
// CHECK:      {{^}}    | |     `-OpaqueValueExpr [[ove_17]] {{.*}} 'int'
// CHECK:      {{^}}    | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}    |   `-DeclRefExpr {{.+}} [[var_len]]
// CHECK-NEXT: {{^}}    `-ReturnStmt
// CHECK-NEXT: {{^}}      `-BoundsCheckExpr {{.+}} 'dst_str <= __builtin_get_pointer_upper_bound(dst_str) && __builtin_get_pointer_lower_bound(dst_str) <= dst_str && len <= __builtin_get_pointer_upper_bound(dst_str) - dst_str && 0 <= len'
// CHECK-NEXT: {{^}}        |-ImplicitCastExpr {{.+}} 'char *__single __counted_by(len)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}        | `-OpaqueValueExpr [[ove_18:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}        |     | | |-OpaqueValueExpr [[ove_19:0x[^ ]+]] {{.*}} 'char *__single __counted_by(len2)':'char *__single'
// CHECK:      {{^}}        |     | | | `-OpaqueValueExpr [[ove_20:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}        |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}        | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}        | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}        | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}        | | | | `-OpaqueValueExpr [[ove_18]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}        | | | `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}        | | |   `-OpaqueValueExpr [[ove_18]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}        | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}        | |   |-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}        | |   | `-OpaqueValueExpr [[ove_18]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}        | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}        | |     `-OpaqueValueExpr [[ove_18]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}        | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}        |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}        |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}        |   | | `-OpaqueValueExpr [[ove_21:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}        |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}        |   |   |-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}        |   |   | `-OpaqueValueExpr [[ove_18]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}        |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}        |   |     `-OpaqueValueExpr [[ove_18]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}        |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}        |     |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}        |     `-OpaqueValueExpr [[ove_21]] {{.*}} 'int'
// CHECK:      {{^}}        |-OpaqueValueExpr [[ove_18]]
// CHECK-NEXT: {{^}}        | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}        |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}        |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: {{^}}        |   | | |-OpaqueValueExpr [[ove_19]] {{.*}} 'char *__single __counted_by(len2)':'char *__single'
// CHECK:      {{^}}        |   | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: {{^}}        |   | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}        |   | | | | `-OpaqueValueExpr [[ove_19]] {{.*}} 'char *__single __counted_by(len2)':'char *__single'
// CHECK:      {{^}}        |   | | | `-OpaqueValueExpr [[ove_20]] {{.*}} 'int'
// CHECK:      {{^}}        |   | |-OpaqueValueExpr [[ove_19]]
// CHECK-NEXT: {{^}}        |   | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(len2)':'char *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}        |   | |   `-DeclRefExpr {{.+}} [[var_dst_str]]
// CHECK-NEXT: {{^}}        |   | `-OpaqueValueExpr [[ove_20]]
// CHECK-NEXT: {{^}}        |   |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}        |   |     `-DeclRefExpr {{.+}} [[var_len2]]
// CHECK-NEXT: {{^}}        |   |-OpaqueValueExpr [[ove_19]] {{.*}} 'char *__single __counted_by(len2)':'char *__single'
// CHECK:      {{^}}        |   `-OpaqueValueExpr [[ove_20]] {{.*}} 'int'
// CHECK:      {{^}}        `-OpaqueValueExpr [[ove_21]]
// CHECK-NEXT: {{^}}          `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}            `-DeclRefExpr {{.+}} [[var_len]]
