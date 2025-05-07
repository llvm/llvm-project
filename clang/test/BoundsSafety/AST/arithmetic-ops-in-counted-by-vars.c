
// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>
#include <stddef.h>

int len = -1;
int *__counted_by(len + 1) buf;
// CHECK:     |-VarDecl {{.*}} used len 'int' cinit
// CHECK-NEXT:| |-UnaryOperator {{.*}} 'int' prefix '-'
// CHECK-NEXT:| | `-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:| `-DependerDeclsAttr {{.*}} <<invalid sloc>> Implicit {{.*}} 0
// CHECK-NEXT:|-VarDecl {{.*}} buf 'int *__single __counted_by(len + 1)':'int *__single'

unsigned size;
unsigned count;
void *__sized_by(size * count) buf2;
// CHECK:     |-VarDecl {{.*}} used size 'unsigned int'
// CHECK-NEXT:| `-DependerDeclsAttr {{.*}} <<invalid sloc>> Implicit {{.*}} 0
// CHECK-NEXT:|-VarDecl {{.*}} used count 'unsigned int'
// CHECK-NEXT:| `-DependerDeclsAttr {{.*}} <<invalid sloc>> Implicit {{.*}} 0
// CHECK-NEXT:|-VarDecl {{.*}} buf2 'void *__single __sized_by(size * count)':'void *__single'

void f(void) {
  int len3 = 10;
  int *__counted_by(len3 - 10) buf3; 
}
// CHECK-LABEL: f 'void (void)'
// CHECK-NEXT:| `-CompoundStmt {{.*}}
// CHECK-NEXT:|   |-DeclStmt {{.*}}
// CHECK-NEXT:|   | `-VarDecl {{.*}} used len3 'int' cinit
// CHECK-NEXT:|   |   |-IntegerLiteral {{.*}} 'int' 10
// CHECK-NEXT:|   |   `-DependerDeclsAttr {{.*}} <<invalid sloc>> Implicit {{.*}} 0
// CHECK-NEXT:|   `-DeclStmt {{.*}}
// CHECK-NEXT:|     `-VarDecl {{.*}} buf3 'int *__single __counted_by(len3 - 10)':'int *__single'

void f2(int *__counted_by(10 * order1 + order0) buf, int order1, unsigned order0);
// CHECK-LABEL: f2 'void (int *__single __counted_by(10 * order1 + order0), int, unsigned int)'
// CHECK-NEXT:| |-ParmVarDecl {{.*}} buf 'int *__single __counted_by(10 * order1 + order0)':'int *__single'
// CHECK-NEXT:| |-ParmVarDecl {{.*}} used order1 'int'
// CHECK-NEXT:| | `-DependerDeclsAttr {{.*}} <<invalid sloc>> Implicit {{.*}} 0
// CHECK-NEXT:| `-ParmVarDecl {{.*}} used order0 'unsigned int'
// CHECK-NEXT:|   `-DependerDeclsAttr {{.*}} <<invalid sloc>> Implicit {{.*}} 0

void *__sized_by(nitems * size) mycalloc(size_t nitems, size_t size);
// CHECK-LABEL: mycalloc 'void *__single __sized_by(nitems * size)(size_t, size_t)'
// CHECK-NEXT:  |-ParmVarDecl {{.*}} used nitems 'size_t':'unsigned long'
// CHECK-NEXT:  `-ParmVarDecl {{.*}} used size 'size_t':'unsigned long'
