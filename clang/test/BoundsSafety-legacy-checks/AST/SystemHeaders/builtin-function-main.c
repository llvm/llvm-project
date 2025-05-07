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

// CHECK: TranslationUnitDecl
// CHECK: |-FunctionDecl [[func_static:0x[^ ]+]] {{.+}} static
// CHECK: | |-ParmVarDecl [[var_dst:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_src:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_size:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'void *__bidi_indexable'
// CHECK: |     | | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single'
// CHECK: |     | | |   | `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'void *'
// CHECK: |     | | |   | `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'void *'
// CHECK: |     | | |   `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'unsigned long'
// CHECK: |     | | |-ImplicitCastExpr {{.+}} 'void *' <BitCast>
// CHECK: |     | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: |     | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: |     | | |   | `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK: |     | | |   |   `-OpaqueValueExpr [[ove]] {{.*}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single'
// CHECK: |     | | |   `-AssumptionExpr
// CHECK: |     | | |     |-OpaqueValueExpr [[ove_3]] {{.*}} 'unsigned long'
// CHECK: |     | | |     `-BinaryOperator {{.+}} 'int' '>='
// CHECK: |     | | |       |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |     | | |       | `-OpaqueValueExpr [[ove_3]] {{.*}} 'unsigned long'
// CHECK: |     | | |       `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |     | | |         `-IntegerLiteral {{.+}} 0
// CHECK: |     | |-OpaqueValueExpr [[ove_1]]
// CHECK: |     | | `-ImplicitCastExpr {{.+}} 'void *' <LValueToRValue>
// CHECK: |     | |   `-DeclRefExpr {{.+}} [[var_dst]]
// CHECK: |     | |-OpaqueValueExpr [[ove_2]]
// CHECK: |     | | `-ImplicitCastExpr {{.+}} 'void *' <LValueToRValue>
// CHECK: |     | |   `-DeclRefExpr {{.+}} [[var_src]]
// CHECK: |     | |-OpaqueValueExpr [[ove_3]]
// CHECK: |     | | `-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK: |     | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |     | |     `-DeclRefExpr {{.+}} [[var_size]]
// CHECK: |     | `-OpaqueValueExpr [[ove]]
// CHECK: |     |   `-CallExpr
// CHECK: |     |     |-ImplicitCastExpr {{.+}} 'void *__single __sized_by(function-parameter-0-2)(*)(void *__single __sized_by(function-parameter-0-2), const void *__single __sized_by(function-parameter-0-2), unsigned long)' <BuiltinFnToFnPtr>
// CHECK: |     |     | `-DeclRefExpr {{.+}}
// CHECK: |     |     |-ImplicitCastExpr {{.+}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single' <BoundsSafetyPointerCast>
// CHECK: |     |     | `-OpaqueValueExpr [[ove_1]] {{.*}} 'void *'
// CHECK: |     |     |-ImplicitCastExpr {{.+}} 'const void *__single __sized_by(function-parameter-0-2)':'const void *__single' <BoundsSafetyPointerCast>
// CHECK: |     |     | `-OpaqueValueExpr [[ove_2]] {{.*}} 'void *'
// CHECK: |     |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'unsigned long'
// CHECK: |     |-OpaqueValueExpr [[ove_1]] {{.*}} 'void *'
// CHECK: |     |-OpaqueValueExpr [[ove_2]] {{.*}} 'void *'
// CHECK: |     |-OpaqueValueExpr [[ove_3]] {{.*}} 'unsigned long'
// CHECK: |     `-OpaqueValueExpr [[ove]] {{.*}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single'
// CHECK: |-FunctionDecl [[func_static_1:0x[^ ]+]] {{.+}} static
// CHECK: | |-ParmVarDecl [[var_dst_1:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_src_1:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_size_1:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK: |       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'void *__bidi_indexable'
// CHECK: |         | | |-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single'
// CHECK: |         | | |   | `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'void *'
// CHECK: |         | | |   | `-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'void *'
// CHECK: |         | | |   `-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'unsigned long'
// CHECK: |         | | |-ImplicitCastExpr {{.+}} 'void *' <BitCast>
// CHECK: |         | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: |         | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: |         | | |   | `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK: |         | | |   |   `-OpaqueValueExpr [[ove_4]] {{.*}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single'
// CHECK: |         | | |   `-AssumptionExpr
// CHECK: |         | | |     |-OpaqueValueExpr [[ove_7]] {{.*}} 'unsigned long'
// CHECK: |         | | |     `-BinaryOperator {{.+}} 'int' '>='
// CHECK: |         | | |       |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |         | | |       | `-OpaqueValueExpr [[ove_7]] {{.*}} 'unsigned long'
// CHECK: |         | | |       `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |         | | |         `-IntegerLiteral {{.+}} 0
// CHECK: |         | |-OpaqueValueExpr [[ove_5]]
// CHECK: |         | | `-ImplicitCastExpr {{.+}} 'void *' <LValueToRValue>
// CHECK: |         | |   `-DeclRefExpr {{.+}} [[var_dst_1]]
// CHECK: |         | |-OpaqueValueExpr [[ove_6]]
// CHECK: |         | | `-ImplicitCastExpr {{.+}} 'void *' <LValueToRValue>
// CHECK: |         | |   `-DeclRefExpr {{.+}} [[var_src_1]]
// CHECK: |         | |-OpaqueValueExpr [[ove_7]]
// CHECK: |         | | `-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK: |         | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |         | |     `-DeclRefExpr {{.+}} [[var_size_1]]
// CHECK: |         | `-OpaqueValueExpr [[ove_4]]
// CHECK: |         |   `-CallExpr
// CHECK: |         |     |-ImplicitCastExpr {{.+}} 'void *__single __sized_by(function-parameter-0-2)(*)(void *__single __sized_by(function-parameter-0-2), const void *__single __sized_by(function-parameter-0-2), unsigned long)' <BuiltinFnToFnPtr>
// CHECK: |         |     | `-DeclRefExpr {{.+}}
// CHECK: |         |     |-ImplicitCastExpr {{.+}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single' <BoundsSafetyPointerCast>
// CHECK: |         |     | `-OpaqueValueExpr [[ove_5]] {{.*}} 'void *'
// CHECK: |         |     |-ImplicitCastExpr {{.+}} 'const void *__single __sized_by(function-parameter-0-2)':'const void *__single' <BoundsSafetyPointerCast>
// CHECK: |         |     | `-OpaqueValueExpr [[ove_6]] {{.*}} 'void *'
// CHECK: |         |     `-OpaqueValueExpr [[ove_7]] {{.*}} 'unsigned long'
// CHECK: |         |-OpaqueValueExpr [[ove_5]] {{.*}} 'void *'
// CHECK: |         |-OpaqueValueExpr [[ove_6]] {{.*}} 'void *'
// CHECK: |         |-OpaqueValueExpr [[ove_7]] {{.*}} 'unsigned long'
// CHECK: |         `-OpaqueValueExpr [[ove_4]] {{.*}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single'
// CHECK: |-FunctionDecl [[func_static_2:0x[^ ]+]] {{.+}} static
// CHECK: | |-ParmVarDecl [[var_dst_2:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_src_2:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_size_2:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_tmp:0x[^ ]+]]
// CHECK: |   |   `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK: |   |     `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   |       |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'void *__bidi_indexable'
// CHECK: |   |       | | |-OpaqueValueExpr [[ove_8:0x[^ ]+]] {{.*}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single'
// CHECK: |   |       | | |   | `-OpaqueValueExpr [[ove_9:0x[^ ]+]] {{.*}} 'void *'
// CHECK: |   |       | | |   | `-OpaqueValueExpr [[ove_10:0x[^ ]+]] {{.*}} 'void *'
// CHECK: |   |       | | |   `-OpaqueValueExpr [[ove_11:0x[^ ]+]] {{.*}} 'unsigned long'
// CHECK: |   |       | | |-ImplicitCastExpr {{.+}} 'void *' <BitCast>
// CHECK: |   |       | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: |   |       | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: |   |       | | |   | `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK: |   |       | | |   |   `-OpaqueValueExpr [[ove_8]] {{.*}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single'
// CHECK: |   |       | | |   `-AssumptionExpr
// CHECK: |   |       | | |     |-OpaqueValueExpr [[ove_11]] {{.*}} 'unsigned long'
// CHECK: |   |       | | |     `-BinaryOperator {{.+}} 'int' '>='
// CHECK: |   |       | | |       |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |   |       | | |       | `-OpaqueValueExpr [[ove_11]] {{.*}} 'unsigned long'
// CHECK: |   |       | | |       `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |   |       | | |         `-IntegerLiteral {{.+}} 0
// CHECK: |   |       | |-OpaqueValueExpr [[ove_9]]
// CHECK: |   |       | | `-ImplicitCastExpr {{.+}} 'void *' <LValueToRValue>
// CHECK: |   |       | |   `-DeclRefExpr {{.+}} [[var_dst_2]]
// CHECK: |   |       | |-OpaqueValueExpr [[ove_10]]
// CHECK: |   |       | | `-ImplicitCastExpr {{.+}} 'void *' <LValueToRValue>
// CHECK: |   |       | |   `-DeclRefExpr {{.+}} [[var_src_2]]
// CHECK: |   |       | |-OpaqueValueExpr [[ove_11]]
// CHECK: |   |       | | `-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK: |   |       | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   |       | |     `-DeclRefExpr {{.+}} [[var_size_2]]
// CHECK: |   |       | `-OpaqueValueExpr [[ove_8]]
// CHECK: |   |       |   `-CallExpr
// CHECK: |   |       |     |-ImplicitCastExpr {{.+}} 'void *__single __sized_by(function-parameter-0-2)(*)(void *__single __sized_by(function-parameter-0-2), const void *__single __sized_by(function-parameter-0-2), unsigned long)' <BuiltinFnToFnPtr>
// CHECK: |   |       |     | `-DeclRefExpr {{.+}}
// CHECK: |   |       |     |-ImplicitCastExpr {{.+}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single' <BoundsSafetyPointerCast>
// CHECK: |   |       |     | `-OpaqueValueExpr [[ove_9]] {{.*}} 'void *'
// CHECK: |   |       |     |-ImplicitCastExpr {{.+}} 'const void *__single __sized_by(function-parameter-0-2)':'const void *__single' <BoundsSafetyPointerCast>
// CHECK: |   |       |     | `-OpaqueValueExpr [[ove_10]] {{.*}} 'void *'
// CHECK: |   |       |     `-OpaqueValueExpr [[ove_11]] {{.*}} 'unsigned long'
// CHECK: |   |       |-OpaqueValueExpr [[ove_9]] {{.*}} 'void *'
// CHECK: |   |       |-OpaqueValueExpr [[ove_10]] {{.*}} 'void *'
// CHECK: |   |       |-OpaqueValueExpr [[ove_11]] {{.*}} 'unsigned long'
// CHECK: |   |       `-OpaqueValueExpr [[ove_8]] {{.*}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single'
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} 'void *' <LValueToRValue>
// CHECK: |       `-DeclRefExpr {{.+}} [[var_tmp]]
// CHECK: |-FunctionDecl [[func_static_3:0x[^ ]+]] {{.+}} static
// CHECK: | |-ParmVarDecl [[var_size_3:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   `-ReturnStmt
// CHECK: |     `-CallExpr
// CHECK: |       |-ImplicitCastExpr {{.+}} 'void *(*)(unsigned long)' <BuiltinFnToFnPtr>
// CHECK: |       | `-DeclRefExpr {{.+}}
// CHECK: |       `-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK: |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |           `-DeclRefExpr {{.+}} [[var_size_3]]
// CHECK: |-FunctionDecl [[func_static_4:0x[^ ]+]] {{.+}} static
// CHECK: | |-ParmVarDecl [[var_size_4:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_tmp_1:0x[^ ]+]]
// CHECK: |   |   `-CallExpr
// CHECK: |   |     |-ImplicitCastExpr {{.+}} 'void *(*)(unsigned long)' <BuiltinFnToFnPtr>
// CHECK: |   |     | `-DeclRefExpr {{.+}}
// CHECK: |   |     `-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK: |   |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   |         `-DeclRefExpr {{.+}} [[var_size_4]]
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} 'void *' <LValueToRValue>
// CHECK: |       `-DeclRefExpr {{.+}} [[var_tmp_1]]
// CHECK: `-FunctionDecl [[func_func:0x[^ ]+]] {{.+}} func
// CHECK:   |-ParmVarDecl [[var_src_str:0x[^ ]+]]
// CHECK:   |-ParmVarDecl [[var_len:0x[^ ]+]]
// CHECK:   | `-DependerDeclsAttr
// CHECK:   `-CompoundStmt
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl [[var_len2:0x[^ ]+]]
// CHECK:     |   |-IntegerLiteral {{.+}} 0
// CHECK:     |   `-DependerDeclsAttr
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl [[var_dst_str:0x[^ ]+]]
// CHECK:     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     | |-BoundsCheckExpr {{.+}} '((char *__bidi_indexable)__builtin_unsafe_forge_bidi_indexable((malloc(len)), (len))) <= __builtin_get_pointer_upper_bound(((char *__bidi_indexable)__builtin_unsafe_forge_bidi_indexable((malloc(len)), (len)))) && __builtin_get_pointer_lower_bound(((char *__bidi_indexable)__builtin_unsafe_forge_bidi_indexable((malloc(len)), (len)))) <= ((char *__bidi_indexable)__builtin_unsafe_forge_bidi_indexable((malloc(len)), (len))) && len <= __builtin_get_pointer_upper_bound(((char *__bidi_indexable)__builtin_unsafe_forge_bidi_indexable((malloc(len)), (len)))) - ((char *__bidi_indexable)__builtin_unsafe_forge_bidi_indexable((malloc(len)), (len))) && 0 <= len'
// CHECK:     | | |-BinaryOperator {{.+}} 'char *__single __counted_by(len2)':'char *__single' '='
// CHECK:     | | | |-DeclRefExpr {{.+}} [[var_dst_str]]
// CHECK:     | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(len2)':'char *__single' <BoundsSafetyPointerCast>
// CHECK:     | | |   `-OpaqueValueExpr [[ove_12:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |   | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:     | |   | | | `-OpaqueValueExpr [[ove_12]] {{.*}} 'char *__bidi_indexable'
// CHECK:     | |   | | `-GetBoundExpr {{.+}} upper
// CHECK:     | |   | |   `-OpaqueValueExpr [[ove_12]] {{.*}} 'char *__bidi_indexable'
// CHECK:     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |   |   |-GetBoundExpr {{.+}} lower
// CHECK:     | |   |   | `-OpaqueValueExpr [[ove_12]] {{.*}} 'char *__bidi_indexable'
// CHECK:     | |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:     | |   |     `-OpaqueValueExpr [[ove_12]] {{.*}} 'char *__bidi_indexable'
// CHECK:     | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:     | |     | | `-OpaqueValueExpr [[ove_13:0x[^ ]+]] {{.*}} 'int'
// CHECK:     | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:     | |     |   |-GetBoundExpr {{.+}} upper
// CHECK:     | |     |   | `-OpaqueValueExpr [[ove_12]] {{.*}} 'char *__bidi_indexable'
// CHECK:     | |     |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:     | |     |     `-OpaqueValueExpr [[ove_12]] {{.*}} 'char *__bidi_indexable'
// CHECK:     | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |       |-IntegerLiteral {{.+}} 0
// CHECK:     | |       `-OpaqueValueExpr [[ove_13]] {{.*}} 'int'
// CHECK:     | |-OpaqueValueExpr [[ove_12]]
// CHECK:     | | `-ParenExpr
// CHECK:     | |   `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:     | |     `-ForgePtrExpr
// CHECK:     | |       |-ParenExpr
// CHECK:     | |       | `-CallExpr
// CHECK:     | |       |   |-ImplicitCastExpr {{.+}} 'void *(*__single)(int)' <FunctionToPointerDecay>
// CHECK:     | |       |   | `-DeclRefExpr {{.+}} [[func_static_3]]
// CHECK:     | |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     | |       |     `-DeclRefExpr {{.+}} [[var_len]]
// CHECK:     | |       |-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK:     | |       | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     | |       |   `-ParenExpr
// CHECK:     | |       |     `-DeclRefExpr {{.+}} [[var_len]]
// CHECK:     | `-OpaqueValueExpr [[ove_13]]
// CHECK:     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     |     `-DeclRefExpr {{.+}} [[var_len]]
// CHECK:     |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     | |-BinaryOperator {{.+}} 'int' '='
// CHECK:     | | |-DeclRefExpr {{.+}} [[var_len2]]
// CHECK:     | | `-OpaqueValueExpr [[ove_13]] {{.*}} 'int'
// CHECK:     | |-OpaqueValueExpr [[ove_12]] {{.*}} 'char *__bidi_indexable'
// CHECK:     | `-OpaqueValueExpr [[ove_13]] {{.*}} 'int'
// CHECK:     |-CallExpr
// CHECK:     | |-ImplicitCastExpr {{.+}} 'void (*__single)(void *restrict, void *restrict, int)' <FunctionToPointerDecay>
// CHECK:     | | `-DeclRefExpr {{.+}} [[func_static]]
// CHECK:     | |-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:     | | `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <BitCast>
// CHECK:     | |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     | |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK:     | |     | | |-OpaqueValueExpr [[ove_14:0x[^ ]+]] {{.*}} 'char *__single __counted_by(len2)':'char *__single'
// CHECK:     | |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK:     | |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:     | |     | | | | `-OpaqueValueExpr [[ove_14]] {{.*}} 'char *__single __counted_by(len2)':'char *__single'
// CHECK:     | |     | | | `-OpaqueValueExpr [[ove_15:0x[^ ]+]] {{.*}} 'int'
// CHECK:     | |     | |-OpaqueValueExpr [[ove_14]]
// CHECK:     | |     | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(len2)':'char *__single' <LValueToRValue>
// CHECK:     | |     | |   `-DeclRefExpr {{.+}} [[var_dst_str]]
// CHECK:     | |     | `-OpaqueValueExpr [[ove_15]]
// CHECK:     | |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     | |     |     `-DeclRefExpr {{.+}} [[var_len2]]
// CHECK:     | |     |-OpaqueValueExpr [[ove_14]] {{.*}} 'char *__single __counted_by(len2)':'char *__single'
// CHECK:     | |     `-OpaqueValueExpr [[ove_15]] {{.*}} 'int'
// CHECK:     | |-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:     | | `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <BitCast>
// CHECK:     | |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     | |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK:     | |     | | |-OpaqueValueExpr [[ove_16:0x[^ ]+]] {{.*}} 'char *__single __counted_by(len)':'char *__single'
// CHECK:     | |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK:     | |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:     | |     | | | | `-OpaqueValueExpr [[ove_16]] {{.*}} 'char *__single __counted_by(len)':'char *__single'
// CHECK:     | |     | | | `-OpaqueValueExpr [[ove_17:0x[^ ]+]] {{.*}} 'int'
// CHECK:     | |     | |-OpaqueValueExpr [[ove_16]]
// CHECK:     | |     | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(len)':'char *__single' <LValueToRValue>
// CHECK:     | |     | |   `-DeclRefExpr {{.+}} [[var_src_str]]
// CHECK:     | |     | `-OpaqueValueExpr [[ove_17]]
// CHECK:     | |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     | |     |     `-DeclRefExpr {{.+}} [[var_len]]
// CHECK:     | |     |-OpaqueValueExpr [[ove_16]] {{.*}} 'char *__single __counted_by(len)':'char *__single'
// CHECK:     | |     `-OpaqueValueExpr [[ove_17]] {{.*}} 'int'
// CHECK:     | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     |   `-DeclRefExpr {{.+}} [[var_len]]
// CHECK:     `-ReturnStmt
// CHECK:       `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(len)':'char *__single' <BoundsSafetyPointerCast>
// CHECK:         `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:           |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:           | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK:           | | |-OpaqueValueExpr [[ove_18:0x[^ ]+]] {{.*}} 'char *__single __counted_by(len2)':'char *__single'
// CHECK:           | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK:           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:           | | | | `-OpaqueValueExpr [[ove_18]] {{.*}} 'char *__single __counted_by(len2)':'char *__single'
// CHECK:           | | | `-OpaqueValueExpr [[ove_19:0x[^ ]+]] {{.*}} 'int'
// CHECK:           | |-OpaqueValueExpr [[ove_18]]
// CHECK:           | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(len2)':'char *__single' <LValueToRValue>
// CHECK:           | |   `-DeclRefExpr {{.+}} [[var_dst_str]]
// CHECK:           | `-OpaqueValueExpr [[ove_19]]
// CHECK:           |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:           |     `-DeclRefExpr {{.+}} [[var_len2]]
// CHECK:           |-OpaqueValueExpr [[ove_18]] {{.*}} 'char *__single __counted_by(len2)':'char *__single'
// CHECK:           `-OpaqueValueExpr [[ove_19]] {{.*}} 'int'
