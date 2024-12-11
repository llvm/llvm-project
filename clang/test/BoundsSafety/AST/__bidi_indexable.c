

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -x c++ -ast-dump -fbounds-safety -fbounds-attributes-cxx-experimental %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -x objective-c -ast-dump -fbounds-safety -fbounds-attributes-objc-experimental %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

struct Foo {
  int *__bidi_indexable foo;
  // CHECK: FieldDecl {{.+}} foo 'int *__bidi_indexable'
};
