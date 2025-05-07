

// RUN: %clang_cc1 -ast-dump -fbounds-safety -verify %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

struct s_no_annot {
  int len;
  int *ptr;
};

int getlen();
int *getptr();

// CHECK: |-FunctionDecl [[func_getlen:0x[^ ]+]] {{.+}} getlen
// CHECK: |-FunctionDecl [[func_getptr:0x[^ ]+]] {{.+}} getptr

// CHECK: |-FunctionDecl [[func_test_no_annot:0x[^ ]+]] {{.+}} test_no_annot
// CHECK: | `-CompoundStmt
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_s1:0x[^ ]+]]
// CHECK: |   |   `-InitListExpr
// CHECK: |   |     |-CallExpr
// CHECK: |   |     | `-ImplicitCastExpr {{.+}} 'int (*__single)()' <FunctionToPointerDecay>
// CHECK: |   |     |   `-DeclRefExpr {{.+}} [[func_getlen]]
// CHECK: |   |     `-CallExpr
// CHECK: |   |       `-ImplicitCastExpr {{.+}} 'int *__single(*__single)()' <FunctionToPointerDecay>
// CHECK: |   |         `-DeclRefExpr {{.+}} [[func_getptr]]
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl [[var_s2:0x[^ ]+]]
void test_no_annot() {
  struct s_no_annot s1 = {
    .len = getlen(),
    .ptr = getptr()
  };
  struct s_no_annot s2;
}


struct s_count_annot {
  int len;
  int dummy;
  int *__counted_by(len) ptr;
};

int *__counted_by(len) getcountptr(int len);

// CHECK: |-FunctionDecl [[func_getcountptr:0x[^ ]+]] {{.+}} getcountptr
// CHECK: | `-ParmVarDecl [[var_len:0x[^ ]+]]
// CHECK: |-FunctionDecl [[func_test_count_annot:0x[^ ]+]] {{.+}} test_count_annot
// CHECK: | `-CompoundStmt
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_arr:0x[^ ]+]]
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_s1_1:0x[^ ]+]]
// CHECK: |   |   `-BoundsCheckExpr
// CHECK: |   |     |-InitListExpr
// CHECK: |   |     | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int'
// CHECK: |   |     | |-CallExpr
// CHECK: |   |     | | `-ImplicitCastExpr {{.+}} 'int (*__single)()' <FunctionToPointerDecay>
// CHECK: |   |     | |   `-DeclRefExpr {{.+}} [[func_getlen]]
// CHECK: |   |     | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK: |   |     |   `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |     |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |     | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |     | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |     | | | | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |     | | | `-GetBoundExpr {{.+}} upper
// CHECK: |   |     | | |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |     | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |     | |   |-GetBoundExpr {{.+}} lower
// CHECK: |   |     | |   | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |     | |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |     | |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |     | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |     |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |     |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |   |     |   | | `-OpaqueValueExpr [[ove]] {{.*}} 'int'
// CHECK: |   |     |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |   |     |   |   |-GetBoundExpr {{.+}} upper
// CHECK: |   |     |   |   | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |     |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |     |   |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |     |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |     |     |-IntegerLiteral {{.+}} 0
// CHECK: |   |     |     `-OpaqueValueExpr [[ove]] {{.*}} 'int'
// CHECK: |   |     |-OpaqueValueExpr [[ove]]
// CHECK: |   |     | `-IntegerLiteral {{.+}} 10
// CHECK: |   |     `-OpaqueValueExpr [[ove_1]]
// CHECK: |   |       `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK: |   |         `-DeclRefExpr {{.+}} [[var_arr]]
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl [[var_s2_1:0x[^ ]+]]
void test_count_annot() {
  int arr[10];
  struct s_count_annot s1 = {
    .len = 10,
    .ptr = arr,
    // expected-warning@+1{{initializer 'getlen()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
    .dummy = getlen()
  };
  struct s_count_annot s2;
}

struct s_range_annot {
  int *end;
  int dummy;
  int *__ended_by(end) start;
};

// CHECK: FunctionDecl [[func_test_range_annot:0x[^ ]+]] {{.+}} test_range_annot
// CHECK: `-CompoundStmt
// CHECK:   |-DeclStmt
// CHECK:   | `-VarDecl [[var_arr_1:0x[^ ]+]]
// CHECK:   |-DeclStmt
// CHECK:   | `-VarDecl [[var_s1_2:0x[^ ]+]]
// CHECK:   |   `-BoundsCheckExpr
// CHECK:   |     |-InitListExpr
// CHECK:   |     | |-ImplicitCastExpr {{.+}} 'int *__single /* __started_by(start) */ ':'int *__single' <BoundsSafetyPointerCast>
// CHECK:   |     | | `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:   |     | |-CallExpr
// CHECK:   |     | | `-ImplicitCastExpr {{.+}} 'int (*__single)()' <FunctionToPointerDecay>
// CHECK:   |     | |   `-DeclRefExpr {{.+}} [[func_getlen]]
// CHECK:   |     | `-ImplicitCastExpr {{.+}} 'int *__single __ended_by(end)':'int *__single' <BoundsSafetyPointerCast>
// CHECK:   |     |   `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:   |     |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   |     | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   |     | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:   |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   |     | | | | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK:   |     | | | `-GetBoundExpr {{.+}} upper
// CHECK:   |     | | |   `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__bidi_indexable'
// CHECK:   |     | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   |     | |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   |     | |   | `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__bidi_indexable'
// CHECK:   |     | |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   |     | |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK:   |     | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   |     |   |-GetBoundExpr {{.+}} lower
// CHECK:   |     |   | `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__bidi_indexable'
// CHECK:   |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   |     |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__bidi_indexable'
// CHECK:   |     |-OpaqueValueExpr [[ove_2]]
// CHECK:   |     | `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK:   |     |   |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:   |     |   | `-DeclRefExpr {{.+}} [[var_arr_1]]
// CHECK:   |     |   `-IntegerLiteral {{.+}} 10
// CHECK:   |     `-OpaqueValueExpr [[ove_3]]
// CHECK:   |       `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:   |         `-DeclRefExpr {{.+}} [[var_arr_1]]
// CHECK:   `-DeclStmt
// CHECK:     `-VarDecl [[var_s2_2:0x[^ ]+]]
void test_range_annot() {
  int arr[10];
  struct s_range_annot s1 = {
    .end = arr + 10,
    .start = arr,
    // expected-warning@+1{{initializer 'getlen()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
    .dummy = getlen()
  };
  struct s_count_annot s2;
}
