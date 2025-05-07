
// RUN: %clang_cc1 -fbounds-safety -Wno-default-const-init-unsafe -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -Wno-default-const-init-unsafe -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

typedef int *__counted_by(count) cnt_t(int count);

void test_cnt_fn(cnt_t cnt_fn) {
  int *p = cnt_fn(16);
}
// CHECK: FunctionDecl [[func_test_cnt_fn:0x[^ ]+]] {{.+}} test_cnt_fn
// CHECK: |-ParmVarDecl [[var_cnt_fn:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-DeclStmt
// CHECK:     `-VarDecl [[var_p:0x[^ ]+]]
// CHECK:       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK:         | | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:         | | |   `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int'
// CHECK:         | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK:         | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:         | | | | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:         | | | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK:         | |-OpaqueValueExpr [[ove_1]]
// CHECK:         | | `-IntegerLiteral {{.+}} 16
// CHECK:         | `-OpaqueValueExpr [[ove]]
// CHECK:         |   `-CallExpr
// CHECK:         |     |-ImplicitCastExpr {{.+}} 'cnt_t *__single' <LValueToRValue>
// CHECK:         |     | `-DeclRefExpr {{.+}} [[var_cnt_fn]]
// CHECK:         |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK:         |-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK:         `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __counted_by(count)':'int *__single'

typedef int *__counted_by(count) (*cnt_ptr_t)(int count);

void test_cnt_fn_ptr(cnt_ptr_t cnt_fn_ptr) {
  int *q = cnt_fn_ptr(32);
}
// CHECK: FunctionDecl [[func_test_cnt_fn_ptr:0x[^ ]+]] {{.+}} test_cnt_fn_ptr
// CHECK: |-ParmVarDecl [[var_cnt_fn_ptr:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-DeclStmt
// CHECK:     `-VarDecl [[var_q:0x[^ ]+]]
// CHECK:       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK:         | | |-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:         | | |   `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'int'
// CHECK:         | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK:         | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:         | | | | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:         | | | `-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// CHECK:         | |-OpaqueValueExpr [[ove_3]]
// CHECK:         | | `-IntegerLiteral {{.+}} 32
// CHECK:         | `-OpaqueValueExpr [[ove_2]]
// CHECK:         |   `-CallExpr
// CHECK:         |     |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)(*__single)(int)' <LValueToRValue>
// CHECK:         |     | `-DeclRefExpr {{.+}} [[var_cnt_fn_ptr]]
// CHECK:         |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// CHECK:         |-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// CHECK:         `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __counted_by(count)':'int *__single'

typedef void *__sized_by(size) sz_t(unsigned size);

void test_sz_fn(sz_t sz_fn) {
  void *r = sz_fn(64);
}
// CHECK: FunctionDecl [[func_test_sz_fn:0x[^ ]+]] {{.+}} test_sz_fn
// CHECK: |-ParmVarDecl [[var_sz_fn:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-DeclStmt
// CHECK:     `-VarDecl [[var_r:0x[^ ]+]]
// CHECK:       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'void *__bidi_indexable'
// CHECK:         | | |-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'void *__single __sized_by(size)':'void *__single'
// CHECK:         | | |   `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'unsigned int'
// CHECK:         | | |-ImplicitCastExpr {{.+}} 'void *' <BitCast>
// CHECK:         | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK:         | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK:         | | |   | `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:         | | |   |   `-OpaqueValueExpr [[ove_4]] {{.*}} 'void *__single __sized_by(size)':'void *__single'
// CHECK:         | | |   `-OpaqueValueExpr [[ove_5]] {{.*}} 'unsigned int'
// CHECK:         | |-OpaqueValueExpr [[ove_5]]
// CHECK:         | | `-ImplicitCastExpr {{.+}} 'unsigned int' <IntegralCast>
// CHECK:         | |   `-IntegerLiteral {{.+}} 64
// CHECK:         | `-OpaqueValueExpr [[ove_4]]
// CHECK:         |   `-CallExpr
// CHECK:         |     |-ImplicitCastExpr {{.+}} 'sz_t *__single' <LValueToRValue>
// CHECK:         |     | `-DeclRefExpr {{.+}} [[var_sz_fn]]
// CHECK:         |     `-OpaqueValueExpr [[ove_5]] {{.*}} 'unsigned int'
// CHECK:         |-OpaqueValueExpr [[ove_5]] {{.*}} 'unsigned int'
// CHECK:         `-OpaqueValueExpr [[ove_4]] {{.*}} 'void *__single __sized_by(size)':'void *__single'

typedef void *__sized_by(size) (*sz_ptr_t)(unsigned size);

void test_sz_fn_ptr(sz_ptr_t sz_fn_ptr) {
  void *s = sz_fn_ptr(128);
}
// CHECK: FunctionDecl [[func_test_sz_fn_ptr:0x[^ ]+]] {{.+}} test_sz_fn_ptr
// CHECK: |-ParmVarDecl [[var_sz_fn_ptr:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-DeclStmt
// CHECK:     `-VarDecl [[var_s:0x[^ ]+]]
// CHECK:       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'void *__bidi_indexable'
// CHECK:         | | |-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'void *__single __sized_by(size)':'void *__single'
// CHECK:         | | |   `-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'unsigned int'
// CHECK:         | | |-ImplicitCastExpr {{.+}} 'void *' <BitCast>
// CHECK:         | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK:         | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK:         | | |   | `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:         | | |   |   `-OpaqueValueExpr [[ove_6]] {{.*}} 'void *__single __sized_by(size)':'void *__single'
// CHECK:         | | |   `-OpaqueValueExpr [[ove_7]] {{.*}} 'unsigned int'
// CHECK:         | |-OpaqueValueExpr [[ove_7]]
// CHECK:         | | `-ImplicitCastExpr {{.+}} 'unsigned int' <IntegralCast>
// CHECK:         | |   `-IntegerLiteral {{.+}} 128
// CHECK:         | `-OpaqueValueExpr [[ove_6]]
// CHECK:         |   `-CallExpr
// CHECK:         |     |-ImplicitCastExpr {{.+}} 'void *__single __sized_by(size)(*__single)(unsigned int)' <LValueToRValue>
// CHECK:         |     | `-DeclRefExpr {{.+}} [[var_sz_fn_ptr]]
// CHECK:         |     `-OpaqueValueExpr [[ove_7]] {{.*}} 'unsigned int'
// CHECK:         |-OpaqueValueExpr [[ove_7]] {{.*}} 'unsigned int'
// CHECK:         `-OpaqueValueExpr [[ove_6]] {{.*}} 'void *__single __sized_by(size)':'void *__single'

int *endptr;
// CHECK: VarDecl [[var_endptr:0x[^ ]+]]
typedef int *__ended_by(end) ent_t(int *end);

void test_ent_fn(ent_t ent_fn) {
  int *p = ent_fn(endptr);
}
// CHECK: FunctionDecl [[func_test_ent_fn:0x[^ ]+]] {{.+}} test_ent_fn
// CHECK: |-ParmVarDecl [[var_ent_fn:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-DeclStmt
// CHECK:     `-VarDecl [[var_p_1:0x[^ ]+]]
// CHECK:       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK:         | | |-CallExpr
// CHECK:         | | | |-ImplicitCastExpr {{.+}} 'ent_t *__single' <LValueToRValue>
// CHECK:         | | | | `-DeclRefExpr {{.+}} [[var_ent_fn]]
// CHECK:         | | | `-OpaqueValueExpr [[ove_8:0x[^ ]+]] {{.*}} 'int *__single'
// CHECK:         | | |-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__single'
// CHECK:         | `-OpaqueValueExpr [[ove_8]]
// CHECK:         |   `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// CHECK:         |     `-DeclRefExpr {{.+}} [[var_endptr]]
// CHECK:         `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__single'

typedef int *__ended_by(end) (*ent_ptr_t)(int *end);

void test_ent_fn_ptr(ent_ptr_t ent_fn_ptr) {
  int *q = ent_fn_ptr(endptr);
}
// CHECK: FunctionDecl [[func_test_ent_fn_ptr:0x[^ ]+]] {{.+}} test_ent_fn_ptr
// CHECK: |-ParmVarDecl [[var_ent_fn_ptr:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-DeclStmt
// CHECK:     `-VarDecl [[var_q_1:0x[^ ]+]]
// CHECK:       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK:         | | |-CallExpr
// CHECK:         | | | |-ImplicitCastExpr {{.+}} 'int *__single __ended_by(end)(*__single)(int *__single)' <LValueToRValue>
// CHECK:         | | | | `-DeclRefExpr {{.+}} [[var_ent_fn_ptr]]
// CHECK:         | | | `-OpaqueValueExpr [[ove_9:0x[^ ]+]] {{.*}} 'int *__single'
// CHECK:         | | |-OpaqueValueExpr [[ove_9]] {{.*}} 'int *__single'
// CHECK:         | `-OpaqueValueExpr [[ove_9]]
// CHECK:         |   `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// CHECK:         |     `-DeclRefExpr {{.+}} [[var_endptr]]
// CHECK:         `-OpaqueValueExpr [[ove_9]] {{.*}} 'int *__single'

typedef void ent_param_t(const char *__ended_by(end) start, const char *end);

void test_ent_param_fn(ent_param_t ent_param_fn) {
  const char array[10];
  ent_param_fn(array, array + 10);
}
// CHECK: FunctionDecl [[func_test_ent_param_fn:0x[^ ]+]] {{.+}} test_ent_param_fn
// CHECK: |-ParmVarDecl [[var_ent_param_fn:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   |-DeclStmt
// CHECK:   | `-VarDecl [[var_array:0x[^ ]+]]
// CHECK:   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     | |-BoundsCheckExpr
// CHECK:     | | |-CallExpr
// CHECK:     | | | |-ImplicitCastExpr {{.+}} 'ent_param_t *__single' <LValueToRValue>
// CHECK:     | | | | `-DeclRefExpr {{.+}} [[var_ent_param_fn]]
// CHECK:     | | | |-ImplicitCastExpr {{.+}} 'const char *__single __ended_by(end)':'const char *__single' <BoundsSafetyPointerCast>
// CHECK:     | | | | `-OpaqueValueExpr [[ove_10:0x[^ ]+]] {{.*}} 'const char *__bidi_indexable'
// CHECK:     | | | `-ImplicitCastExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' <BoundsSafetyPointerCast>
// CHECK:     | | |   `-OpaqueValueExpr [[ove_11:0x[^ ]+]] {{.*}} 'const char *__bidi_indexable'
// CHECK:     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |   | |-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK:     | |   | | `-OpaqueValueExpr [[ove_11]] {{.*}} 'const char *__bidi_indexable'
// CHECK:     | |   | `-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK:     | |   |   `-GetBoundExpr {{.+}} upper
// CHECK:     | |   |     `-OpaqueValueExpr [[ove_10]] {{.*}} 'const char *__bidi_indexable'
// CHECK:     | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |     |-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK:     | |     | `-OpaqueValueExpr [[ove_10]] {{.*}} 'const char *__bidi_indexable'
// CHECK:     | |     `-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK:     | |       `-OpaqueValueExpr [[ove_11]] {{.*}} 'const char *__bidi_indexable'
// CHECK:     | |-OpaqueValueExpr [[ove_10]]
// CHECK:     | | `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:     | |   `-DeclRefExpr {{.+}} [[var_array]]
// CHECK:     | `-OpaqueValueExpr [[ove_11]]
// CHECK:     |   `-BinaryOperator {{.+}} 'const char *__bidi_indexable' '+'
// CHECK:     |     |-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:     |     | `-DeclRefExpr {{.+}} [[var_array]]
// CHECK:     |     `-IntegerLiteral {{.+}} 10
// CHECK:     |-OpaqueValueExpr [[ove_10]] {{.*}} 'const char *__bidi_indexable'
// CHECK:     `-OpaqueValueExpr [[ove_11]] {{.*}} 'const char *__bidi_indexable'

typedef void (*ent_param_ptr_t)(const char *__ended_by(end) start, const char *end);

void test_ent_param_fn_ptr(ent_param_ptr_t ent_param_fn_ptr) {
  const char array[10];
  ent_param_fn_ptr(array, array + 10);
}
// CHECK: FunctionDecl [[func_test_ent_param_fn_ptr:0x[^ ]+]] {{.+}} test_ent_param_fn_ptr
// CHECK: |-ParmVarDecl [[var_ent_param_fn_ptr:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   |-DeclStmt
// CHECK:   | `-VarDecl [[var_array_1:0x[^ ]+]]
// CHECK:   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     | |-BoundsCheckExpr
// CHECK:     | | |-CallExpr
// CHECK:     | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(const char *__single __ended_by(end), const char *__single /* __started_by(start) */ )' <LValueToRValue>
// CHECK:     | | | | `-DeclRefExpr {{.+}} [[var_ent_param_fn_ptr]]
// CHECK:     | | | |-ImplicitCastExpr {{.+}} 'const char *__single __ended_by(end)':'const char *__single' <BoundsSafetyPointerCast>
// CHECK:     | | | | `-OpaqueValueExpr [[ove_12:0x[^ ]+]] {{.*}} 'const char *__bidi_indexable'
// CHECK:     | | | `-ImplicitCastExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' <BoundsSafetyPointerCast>
// CHECK:     | | |   `-OpaqueValueExpr [[ove_13:0x[^ ]+]] {{.*}} 'const char *__bidi_indexable'
// CHECK:     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |   | |-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK:     | |   | | `-OpaqueValueExpr [[ove_13]] {{.*}} 'const char *__bidi_indexable'
// CHECK:     | |   | `-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK:     | |   |   `-GetBoundExpr {{.+}} upper
// CHECK:     | |   |     `-OpaqueValueExpr [[ove_12]] {{.*}} 'const char *__bidi_indexable'
// CHECK:     | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |     |-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK:     | |     | `-OpaqueValueExpr [[ove_12]] {{.*}} 'const char *__bidi_indexable'
// CHECK:     | |     `-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK:     | |       `-OpaqueValueExpr [[ove_13]] {{.*}} 'const char *__bidi_indexable'
// CHECK:     | |-OpaqueValueExpr [[ove_12]]
// CHECK:     | | `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:     | |   `-DeclRefExpr {{.+}} [[var_array_1]]
// CHECK:     | `-OpaqueValueExpr [[ove_13]]
// CHECK:     |   `-BinaryOperator {{.+}} 'const char *__bidi_indexable' '+'
// CHECK:     |     |-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:     |     | `-DeclRefExpr {{.+}} [[var_array_1]]
// CHECK:     |     `-IntegerLiteral {{.+}} 10
// CHECK:     |-OpaqueValueExpr [[ove_12]] {{.*}} 'const char *__bidi_indexable'
// CHECK:     `-OpaqueValueExpr [[ove_13]] {{.*}} 'const char *__bidi_indexable'
