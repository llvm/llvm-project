


// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

typedef long a;
typedef struct {
  a *b;
  a *c;
} d, *e;
#define f(g, h)                                                                \
  a *i;                                                                        \
  d j = {&i[0], &i[h]};                                                        \
  e g = &j;
#define k(g) sizeof(g)
#define l(g) k(g)
void m(void) {
  int h;
  f(g, h) l(g);
}

// CHECK: `-FunctionDecl {{.+}} m 'void (void)'
// CHECK:   `-CompoundStmt
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.+}} used h 'int'
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.+}} used i 'a *__bidi_indexable'
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.+}} used j 'd' cinit
// CHECK:     |   `-InitListExpr {{.+}} 'd'
// CHECK:     |     |-ImplicitCastExpr {{.+}} 'a *__single' <BoundsSafetyPointerCast>
// CHECK:     |     | `-UnaryOperator {{.+}} 'a *__bidi_indexable' prefix '&' cannot overflow
// CHECK:     |     |   `-ArraySubscriptExpr {{.+}} 'a':'long' lvalue
// CHECK:     |     |     |-ImplicitCastExpr {{.+}} 'a *__bidi_indexable'{{.*}} <LValueToRValue>
// CHECK:     |     |     | `-DeclRefExpr {{.+}} 'a *__bidi_indexable'{{.*}} lvalue Var {{.+}} 'i' 'a *__bidi_indexable'
// CHECK:     |     |     `-IntegerLiteral {{.+}} 'int' 0
// CHECK:     |     `-ImplicitCastExpr {{.+}} 'a *__single' <BoundsSafetyPointerCast>
// CHECK:     |       `-UnaryOperator {{.+}} 'a *__bidi_indexable' prefix '&' cannot overflow
// CHECK:     |         `-ArraySubscriptExpr {{.+}} 'a':'long' lvalue
// CHECK:     |           |-ImplicitCastExpr {{.+}} 'a *__bidi_indexable'{{.*}} <LValueToRValue>
// CHECK:     |           | `-DeclRefExpr {{.+}} 'a *__bidi_indexable'{{.*}} lvalue Var {{.+}} 'i' 'a *__bidi_indexable'
// CHECK:     |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     |             `-DeclRefExpr {{.+}} 'int' lvalue Var {{.+}} 'h' 'int'
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.+}} referenced g 'struct d *__bidi_indexable'{{.*}} cinit
// CHECK:     |   `-UnaryOperator {{.+}} 'd *__bidi_indexable' prefix '&' cannot overflow
// CHECK:     |     `-DeclRefExpr {{.+}} 'd' lvalue Var {{.+}} 'j' 'd'
// CHECK:     `-UnaryExprOrTypeTraitExpr {{.+}} 'unsigned long' sizeof
// CHECK:       `-ParenExpr {{.+}} 'struct d *__bidi_indexable'
// CHECK:         `-DeclRefExpr {{.+}} 'struct d *__bidi_indexable'{{.*}} 'g' 'struct d *__bidi_indexable'{{.*}} non_odr_use_unevaluated
