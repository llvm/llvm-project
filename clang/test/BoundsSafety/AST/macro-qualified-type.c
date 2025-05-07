

// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

#define my_unused __attribute__((__unused__))

// CHECK:      FunctionDecl {{.+}} foo 'void (my_unused int *__single)'
// CHECK-NEXT: |-ParmVarDecl {{.+}} p 'my_unused int *__single':'int *__single'
// CHECK-NEXT: | `-UnusedAttr {{.+}} unused
// CHECK-NEXT: `-CompoundStmt
// CHECK-NEXT:   `-DeclStmt
// CHECK-NEXT:     `-VarDecl {{.+}} local 'my_unused int *__bidi_indexable':'int *__bidi_indexable'
// CHECK-NEXT:       `-UnusedAttr {{.+}} unused
void foo(int *_Nullable p my_unused) {
  int *_Nullable local my_unused;
}
