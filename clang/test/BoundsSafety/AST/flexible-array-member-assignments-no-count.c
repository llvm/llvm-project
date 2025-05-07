

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s

#include <ptrcheck.h>

typedef struct {
  int count;
  int elems[];
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
// CHECK:   `-BinaryOperator {{.+}} 'flex_t *__single' '='
// CHECK:     |-DeclRefExpr {{.+}} [[var_f]]
// CHECK:     `-ImplicitCastExpr {{.+}} 'flex_t *__single' <BoundsSafetyPointerCast>
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
// CHECK:   |-BinaryOperator {{.+}} 'flex_t *__single' '='
// CHECK:   | |-DeclRefExpr {{.+}} [[var_f_1]]
// CHECK:   | `-ImplicitCastExpr {{.+}} 'flex_t *__single' <BoundsSafetyPointerCast>
// CHECK:   |   `-ImplicitCastExpr {{.+}} 'flex_t *__bidi_indexable' <BitCast>
// CHECK:   |     `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <LValueToRValue>
// CHECK:   |       `-DeclRefExpr {{.+}} [[var_buf_1]]
// CHECK:   `-BinaryOperator {{.+}} 'int' '='
// CHECK:     |-MemberExpr {{.+}} .count
// CHECK:     | `-MemberExpr {{.+}} ->flex
// CHECK:     |   `-ImplicitCastExpr {{.+}} 'flex_t *__single' <LValueToRValue>
// CHECK:     |     `-DeclRefExpr {{.+}} [[var_f_1]]
// CHECK:     `-IntegerLiteral {{.+}} 10

// CHECK-LABEL: test_fam_base_init
void test_fam_base_init(void *__bidi_indexable buf) {
  flex_t *__single f = buf;
}
// CHECK: |-ParmVarDecl [[var_buf_2:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-DeclStmt
// CHECK:     `-VarDecl [[var_f_2:0x[^ ]+]]
// CHECK:       `-ImplicitCastExpr {{.+}} 'flex_t *__single' <BoundsSafetyPointerCast>
// CHECK:         `-ImplicitCastExpr {{.+}} 'flex_t *__bidi_indexable' <BitCast>
// CHECK:           `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <LValueToRValue>
// CHECK:             `-DeclRefExpr {{.+}} [[var_buf_2]]


// CHECK-LABEL: test_fam_base_init_with_count
void test_fam_base_init_with_count(void *__bidi_indexable buf) {
  flex_t *__single f = buf;
  f->flex.count = 10;
}
// CHECK: |-ParmVarDecl [[var_buf_3:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   |-DeclStmt
// CHECK:   | `-VarDecl [[var_f_3:0x[^ ]+]]
// CHECK:   |   `-ImplicitCastExpr {{.+}} 'flex_t *__single' <BoundsSafetyPointerCast>
// CHECK:   |     `-ImplicitCastExpr {{.+}} 'flex_t *__bidi_indexable' <BitCast>
// CHECK:   |       `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <LValueToRValue>
// CHECK:   |         `-DeclRefExpr {{.+}} [[var_buf_3]]
// CHECK:   `-BinaryOperator {{.+}} 'int' '='
// CHECK:     |-MemberExpr {{.+}} .count
// CHECK:     | `-MemberExpr {{.+}} ->flex
// CHECK:     |   `-ImplicitCastExpr {{.+}} 'flex_t *__single' <LValueToRValue>
// CHECK:     |     `-DeclRefExpr {{.+}} [[var_f_3]]
// CHECK:     `-IntegerLiteral {{.+}} 10

// CHECK: VarDecl [[var_g_flex:0x[^ ]+]]
flex_inner_t g_flex;

// CHECK-LABEL: test_fam_lvalue_base_count_assign
void test_fam_lvalue_base_count_assign(unsigned arg) {
  g_flex.count = arg;
}
// CHECK: |-ParmVarDecl [[var_arg:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-BinaryOperator {{.+}} 'int' '='
// CHECK:     |-MemberExpr {{.+}} .count
// CHECK:     | `-DeclRefExpr {{.+}} [[var_g_flex]]
// CHECK:     `-ImplicitCastExpr {{.+}} 'int' <IntegralCast>
// CHECK:       `-ImplicitCastExpr {{.+}} 'unsigned int' <LValueToRValue>
// CHECK:         `-DeclRefExpr {{.+}} [[var_arg]]

// CHECK-LABEL: test_fam_lvalue_base_count_decrement
void test_fam_lvalue_base_count_decrement() {
  g_flex.count--;
}
// CHECK: | `-CompoundStmt
// CHECK: |   `-UnaryOperator {{.+}} postfix '--'
// CHECK: |     `-MemberExpr {{.+}} .count
// CHECK: |       `-DeclRefExpr {{.+}} [[var_g_flex]]

// CHECK-LABEL: test_fam_lvalue_base_count_compound
void test_fam_lvalue_base_count_compound(unsigned arg) {
  g_flex.count -= arg;
}
// CHECK: |-ParmVarDecl [[var_arg_1:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-CompoundAssignOperator {{.+}} ComputeLHSTy='unsigned int'
// CHECK:     |-MemberExpr {{.+}} .count
// CHECK:     | `-DeclRefExpr {{.+}} [[var_g_flex]]
// CHECK:     `-ImplicitCastExpr {{.+}} 'unsigned int' <LValueToRValue>
// CHECK:       `-DeclRefExpr {{.+}} [[var_arg_1]]
