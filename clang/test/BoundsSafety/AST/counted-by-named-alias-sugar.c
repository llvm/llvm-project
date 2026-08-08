// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x c -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c++ -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c++ -ast-dump %s 2>&1 | FileCheck %s

// Confirms the CountAttributedType is actually built when __counted_by names a
// pointer through typedef / __typeof__ sugar. The 'int *{{.*}} __counted_by(n)'
// pattern matches both -fbounds-safety ('int *__single __counted_by(n)') and
// attribute-only mode ('int * __counted_by(n)'), so one CHECK prefix serves
// every RUN line.

#include <ptrcheck.h>

typedef int *ptr_to_int_t;
extern int *gp;

// FIXME: Type aliases (e.g. `ptr_to_int_t`) shouldn't be dropped in the AST
// (rdar://185140320)

// CHECK: FieldDecl {{.+}} buf 'int *{{.*}} __counted_by(n)'
struct GoodTypedef {
  int n;
  ptr_to_int_t buf __counted_by(n);
};

// CHECK: FieldDecl {{.+}} buf 'int *{{.*}} __counted_by(n)'
struct GoodTypeofExpr {
  int n;
  __typeof__(gp) buf __counted_by(n);
};

// CHECK: FieldDecl {{.+}} buf 'int *{{.*}} __counted_by(n)'
struct GoodTypeofType {
  int n;
  __typeof__(int *) buf __counted_by(n);
};
