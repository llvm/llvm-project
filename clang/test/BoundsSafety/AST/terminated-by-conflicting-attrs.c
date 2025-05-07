
// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

#include "terminated-by-conflicting-attrs.h"

/* terminated-by-conflicting-attrs.h:

 void foo(int *__counted_by(len) p, int len);
 void foo(int *__null_terminated p, int len);

 void bar(int *__null_terminated p, int len);
 void bar(int *__counted_by(len) p, int len);

*/

// CHECK: FunctionDecl {{.+}} foo 'void (int *__single __counted_by(len), int)'
// CHECK: FunctionDecl {{.+}} foo 'void (int *__single __counted_by(len), int)'
// CHECK: FunctionDecl {{.+}} bar 'void (int *__single __terminated_by(0), int)'
// XXX: rdar://127827450
// CHECK: FunctionDecl {{.+}} bar 'void (int *__single __terminated_by(0), int)'

void test() {
  int arr[10];
  foo(arr, 10);

  int *__null_terminated ptr = 0;
  bar(ptr, 0);
}
