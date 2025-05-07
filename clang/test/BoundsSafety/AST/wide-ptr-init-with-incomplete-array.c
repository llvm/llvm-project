
// RUN: %clang_cc1 -ast-dump -verify -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -verify -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s

#include <ptrcheck.h>

extern float foo[];
// CHECK:|-VarDecl {{.*}} used foo 'float[]' extern

// expected-warning@+1{{accessing elements of an unannotated incomplete array always fails at runtime}}
float *__indexable wide_f[] = {foo};
// CHECK-NEXT:|-VarDecl {{.*}} wide_f 'float *__indexable[1]' cinit
// CHECK-NEXT:| `-InitListExpr {{.*}} 'float *__indexable[1]'
// CHECK-NEXT:|   `-ImplicitCastExpr {{.*}} 'float *__indexable' <BoundsSafetyPointerCast>
// CHECK-NEXT:|     `-ImplicitCastExpr {{.*}} 'float *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT:|       `-DeclRefExpr {{.*}} 'float[]' lvalue Var {{.*}} 'foo' 'float[]'


extern float bar[];
float bar[] = {1, 2, 3, 4};
float *__indexable wide_f2[] = {bar};
// CHECK-NEXT:|-VarDecl {{.*}} used bar 'float[]' extern
// CHECK-NEXT:|-VarDecl {{.*}} prev {{.*}} used bar 'float[4]' cinit
// CHECK-NEXT:| `-InitListExpr {{.*}} 'float[4]'
// CHECK-NEXT:|   |-ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT:|   | `-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:|   |-ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT:|   | `-IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT:|   |-ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT:|   | `-IntegerLiteral {{.*}} 'int' 3
// CHECK-NEXT:|   `-ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT:|     `-IntegerLiteral {{.*}} 'int' 4
// CHECK-NEXT:|-VarDecl {{.*}} wide_f2 'float *__indexable[1]' cinit
// CHECK-NEXT:| `-InitListExpr {{.*}} 'float *__indexable[1]'
// CHECK-NEXT:|   `-ImplicitCastExpr {{.*}} 'float *__indexable' <BoundsSafetyPointerCast>
// CHECK-NEXT:|     `-ImplicitCastExpr {{.*}} 'float *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT:|       `-DeclRefExpr {{.*}} 'float[4]' lvalue Var {{.*}} 'bar' 'float[4]'

extern float baz[];
float baz[] = {1, 2, 3, 4};
extern float baz[];
float *__indexable wide_f3[] = {baz};
// CHECK-NEXT:|-VarDecl {{.*}} used baz 'float[]' extern
// CHECK-NEXT:|-VarDecl {{.*}} prev {{.*}} used baz 'float[4]' cinit
// CHECK-NEXT:| `-InitListExpr {{.*}} 'float[4]'
// CHECK-NEXT:|   |-ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT:|   | `-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:|   |-ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT:|   | `-IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT:|   |-ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT:|   | `-IntegerLiteral {{.*}} 'int' 3
// CHECK-NEXT:|   `-ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT:|     `-IntegerLiteral {{.*}} 'int' 4
// CHECK-NEXT:|-VarDecl {{.*}} prev {{.*}} used baz 'float[4]' extern
// CHECK-NEXT:|-VarDecl {{.*}} wide_f3 'float *__indexable[1]' cinit
// CHECK-NEXT:| `-InitListExpr {{.*}} 'float *__indexable[1]'
// CHECK-NEXT:|   `-ImplicitCastExpr {{.*}} 'float *__indexable' <BoundsSafetyPointerCast>
// CHECK-NEXT:|     `-ImplicitCastExpr {{.*}} 'float *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT:|       `-DeclRefExpr {{.*}} 'float[4]' lvalue Var {{.*}} 'baz' 'float[4]'

// expected-note@+2{{use __attribute__((visibility("hidden"))) attribute instead}}
// expected-warning@+1{{use of __private_extern__ on a declaration may not produce external symbol private to the linkage unit and is deprecated}}
__private_extern__ float quz[];
float quz[] = {1, 2, 3, 4};
// expected-note@+2{{use __attribute__((visibility("hidden"))) attribute instead}}
// expected-warning@+1{{use of __private_extern__ on a declaration may not produce external symbol private to the linkage unit and is deprecated}}
__private_extern__ float quz[];
float *__indexable wide_f4[] = {quz};
// CHECK-NEXT:|-VarDecl {{.*}} used quz 'float[]' __private_extern__
// CHECK-NEXT:|-VarDecl {{.*}} prev {{.*}} used quz 'float[4]' cinit
// CHECK-NEXT:| `-InitListExpr {{.*}} 'float[4]'
// CHECK-NEXT:|   |-ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT:|   | `-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:|   |-ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT:|   | `-IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT:|   |-ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT:|   | `-IntegerLiteral {{.*}} 'int' 3
// CHECK-NEXT:|   `-ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT:|     `-IntegerLiteral {{.*}} 'int' 4
// CHECK-NEXT:|-VarDecl {{.*}} prev {{.*}} used quz 'float[4]' __private_extern__
// CHECK-NEXT:|-VarDecl {{.*}} wide_f4 'float *__indexable[1]' cinit
// CHECK-NEXT:| `-InitListExpr {{.*}} 'float *__indexable[1]'
// CHECK-NEXT:|   `-ImplicitCastExpr {{.*}} 'float *__indexable' <BoundsSafetyPointerCast>
// CHECK-NEXT:|     `-ImplicitCastExpr {{.*}} 'float *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT:|       `-DeclRefExpr {{.*}} 'float[4]' lvalue Var {{.*}} 'quz' 'float[4]'

// expected-warning@+1{{tentative array definition assumed to have one element}}
static float qux[];
// expected-warning@+1{{accessing elements of an unannotated incomplete array always fails at runtime}}
float *__bidi_indexable wide_f5[] = {qux};
// CHECK-NEXT:|-VarDecl {{.*}} used qux 'float[1]' static
// CHECK-NEXT:|-VarDecl {{.*}} wide_f5 'float *__bidi_indexable[1]' cinit
// CHECK-NEXT:| `-InitListExpr {{.*}} 'float *__bidi_indexable[1]'
// CHECK-NEXT:|   `-ImplicitCastExpr {{.*}} 'float *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT:|     `-DeclRefExpr {{.*}} 'float[]' lvalue Var {{.*}} 'qux' 'float[1]'

void f(void) {
  extern float bar[];
  extern float quxx[];
  // expected-warning@+1{{accessing elements of an unannotated incomplete array always fails at runtime}}
  static float *__bidi_indexable wide_f6[] = {quxx, bar, baz};
}
// CHECK-LABEL: f 'void (void)'
// CHECK-NEXT:  `-CompoundStmt {{.*}}
// CHECK-NEXT:    |-DeclStmt {{.*}}
// CHECK-NEXT:    | `-VarDecl {{.*}} parent {{.*}} prev {{.*}} used bar 'float[4]' extern
// CHECK-NEXT:    |-DeclStmt {{.*}}
// CHECK-NEXT:    | `-VarDecl {{.*}} parent {{.*}} used quxx 'float[]' extern
// CHECK-NEXT:    `-DeclStmt {{.*}}
// CHECK-NEXT:      `-VarDecl {{.*}} wide_f6 'float *__bidi_indexable[3]' static cinit
// CHECK-NEXT:        `-InitListExpr {{.*}} 'float *__bidi_indexable[3]'
// CHECK-NEXT:          |-ImplicitCastExpr {{.*}} 'float *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT:          | `-DeclRefExpr {{.*}} 'float[]' lvalue Var {{.*}} 'quxx' 'float[]'
// CHECK-NEXT:          |-ImplicitCastExpr {{.*}} 'float *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT:          | `-DeclRefExpr {{.*}} 'float[4]' lvalue Var {{.*}} 'bar' 'float[4]'
// CHECK-NEXT:          `-ImplicitCastExpr {{.*}} 'float *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT:            `-DeclRefExpr {{.*}} 'float[4]' lvalue Var {{.*}} 'baz' 'float[4]'
