

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s

#include <ptrcheck.h>

typedef struct {
  int count;
  int elems[];
} flex_t;

// CHECK-LABEL: init_null 'void (void)'
void init_null(void) {
  flex_t *__single s = 0;
}
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: `-DeclStmt
// CHECK-NEXT:   `-VarDecl {{.*}} s 'flex_t *__single' cinit
// CHECK-NEXT:     `-ImplicitCastExpr {{.*}} 'flex_t *__single' <NullToPointer>
// CHECK-NEXT:       `-IntegerLiteral {{.*}} 'int' 0


// CHECK-LABEL: init_casted_null 'void (void)'
void init_casted_null(void) {
  flex_t *__single s = (flex_t *)0;
}
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: `-DeclStmt
// CHECK-NEXT:   `-VarDecl {{.*}} s 'flex_t *__single' cinit
// CHECK-NEXT:     `-ImplicitCastExpr {{.*}} 'flex_t *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT:       `-CStyleCastExpr {{.*}} 'flex_t *' <NullToPointer>
// CHECK-NEXT:         `-IntegerLiteral {{.*}} 'int' 0


// CHECK-LABEL: assign_null 'void (void)'
void assign_null(void) {
  flex_t *__single s;
  s = 0;
}
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: |-DeclStmt
// CHECK-NEXT: | `-VarDecl {{.*}} used s 'flex_t *__single'
// CHECK-NEXT: `-BinaryOperator {{.*}} 'flex_t *__single' '='
// CHECK-NEXT:   |-DeclRefExpr {{.*}} 'flex_t *__single' lvalue Var {{.*}} 's' 'flex_t *__single'
// CHECK-NEXT:   `-ImplicitCastExpr {{.*}} 'flex_t *__single' <NullToPointer>
// CHECK-NEXT:     `-IntegerLiteral {{.*}} 'int' 0


// CHECK-LABEL: assign_casted_null 'void (void)'
void assign_casted_null(void) {
  flex_t *__single s;
  s = (flex_t *)0;
}
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: |-DeclStmt
// CHECK-NEXT: | `-VarDecl {{.*}} used s 'flex_t *__single'
// CHECK-NEXT: `-BinaryOperator {{.*}} 'flex_t *__single' '='
// CHECK-NEXT:   |-DeclRefExpr {{.*}} 'flex_t *__single' lvalue Var {{.*}} 's' 'flex_t *__single'
// CHECK-NEXT:   `-ImplicitCastExpr {{.*}} 'flex_t *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT:     `-CStyleCastExpr {{.*}} 'flex_t *' <NullToPointer>
// CHECK-NEXT:       `-IntegerLiteral {{.*}} 'int' 0
