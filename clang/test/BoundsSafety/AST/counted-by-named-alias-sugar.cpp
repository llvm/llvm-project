// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -std=c++17 -x c++ -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -std=c++17 -x objective-c++ -ast-dump %s 2>&1 | FileCheck %s

// C++-only named-alias sugar: using-alias (TypedefType), using-declaration
// (UsingType), and decltype. Confirms the CountAttributedType is built when
// __counted_by names a pointer through each. Companion to the .c file.

// FIXME: Type aliases (e.g. `ualias_t`) shouldn't be dropped in the AST
// (rdar://185140320)

#include <ptrcheck.h>

extern int *gp;
using ualias_t = int *;
namespace ns { typedef int *ptr_to_int_t; }
using ns::ptr_to_int_t;

// CHECK: FieldDecl {{.+}} buf 'int *{{.*}} __counted_by(n)'
struct GoodUsingAlias {
  int n;
  ualias_t buf __counted_by(n);
};

// CHECK: FieldDecl {{.+}} buf 'int *{{.*}} __counted_by(n)'
struct GoodUsingDecl {
  int n;
  ptr_to_int_t buf __counted_by(n);
};

// CHECK: FieldDecl {{.+}} buf 'int *{{.*}} __counted_by(n)'
struct GoodDecltype {
  int n;
  decltype(gp) buf __counted_by(n);
};
