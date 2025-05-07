

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s

#include <ptrcheck.h>

struct flexible {
    int count;
    int elems[__counted_by(count)];
};

// CHECK-LABEL: init_single
void init_single(void *p) {
  struct flexible *__single s = p;
}
// CHECK-NEXT: | |-ParmVarDecl [[var_p:0x[^ ]+]]
// CHECK-NEXT: | `-CompoundStmt
// CHECK-NEXT: |   `-DeclStmt
// CHECK-NEXT: |     `-VarDecl [[var_s:0x[^ ]+]]
// CHECK-NEXT: |       `-ImplicitCastExpr {{.+}} 'struct flexible *__single' <BitCast>
// CHECK-NEXT: |         `-ImplicitCastExpr {{.+}} 'void *__single' <LValueToRValue>
// CHECK-NEXT: |           `-DeclRefExpr {{.+}} [[var_p]]

// CHECK-LABEL: init_casted_single
void init_casted_single(void *p) {
  struct flexible *__single s = (struct flexible *)p;
}
// CHECK-NEXT: | |-ParmVarDecl [[var_p_1:0x[^ ]+]]
// CHECK-NEXT: | `-CompoundStmt
// CHECK-NEXT: |   `-DeclStmt
// CHECK-NEXT: |     `-VarDecl [[var_s_1:0x[^ ]+]]
// CHECK-NEXT: |       `-CStyleCastExpr {{.+}} 'struct flexible *__single' <BitCast>
// CHECK-NEXT: |         `-ImplicitCastExpr {{.+}} 'void *__single' <LValueToRValue>
// CHECK-NEXT: |           `-DeclRefExpr {{.+}} [[var_p_1]]

// CHECK-LABEL: assign_single
void assign_single(void *p) {
  struct flexible *__single s;
  s = p;
}
// CHECK-NEXT: | |-ParmVarDecl [[var_p_2:0x[^ ]+]]
// CHECK-NEXT: | `-CompoundStmt
// CHECK-NEXT: |   |-DeclStmt
// CHECK-NEXT: |   | `-VarDecl [[var_s_2:0x[^ ]+]]
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct flexible *__single' '='
// CHECK-NEXT: |     |-DeclRefExpr {{.+}} [[var_s_2]]
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct flexible *__single' <BitCast>
// CHECK-NEXT: |       `-ImplicitCastExpr {{.+}} 'void *__single' <LValueToRValue>
// CHECK-NEXT: |         `-DeclRefExpr {{.+}} [[var_p_2]]

// CHECK-LABEL: assign_casted_single
void assign_casted_single(void *p) {
  struct flexible *__single s;
  s = (struct flexible *)p;
}
// CHECK-NEXT: |-ParmVarDecl [[var_p_3:0x[^ ]+]]
// CHECK-NEXT: `-CompoundStmt
// CHECK-NEXT:   |-DeclStmt
// CHECK-NEXT:   | `-VarDecl [[var_s_3:0x[^ ]+]]
// CHECK-NEXT:   `-BinaryOperator {{.+}} 'struct flexible *__single' '='
// CHECK-NEXT:     |-DeclRefExpr {{.+}} [[var_s_3]]
// CHECK-NEXT:     `-CStyleCastExpr {{.+}} 'struct flexible *__single' <BitCast>
// CHECK-NEXT:       `-ImplicitCastExpr {{.+}} 'void *__single' <LValueToRValue>
// CHECK-NEXT:         `-DeclRefExpr {{.+}} [[var_p_3]]
