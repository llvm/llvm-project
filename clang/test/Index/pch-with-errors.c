#ifndef HEADER
#define HEADER

#include "blahblah.h"
void erroneous(int);
void erroneous(float);

struct bar;
struct zed {
  bar g;
};
struct baz {
  zed h;
};

void errparm(zed e);

struct S {
  {
;

#else

void foo(void) {
  erroneous(0);
}

#endif

// RUN: c-index-test -write-pch %t.h.pch %s -Xclang -detailed-preprocessing-record
// RUN: c-index-test -test-load-source local %s -include %t.h -Xclang -detailed-preprocessing-record | FileCheck -check-prefix=CHECK-PARSE %s
// RUN: c-index-test -index-file %s -include %t.h -Xclang -detailed-preprocessing-record | FileCheck -check-prefix=CHECK-INDEX %s

// CHECK-PARSE: pch-with-errors.c:{{.*}}:6: FunctionDecl=foo
// CHECK-PARSE: pch-with-errors.c:{{.*}}:3: CallExpr=erroneous

// CHECK-INDEX: [indexDeclaration]: kind: function | name: foo
// CHECK-INDEX: [indexEntityReference]: kind: function | name: erroneous

// RUN: %clang -fsyntax-only %s -include %t.h 2>&1 | FileCheck -check-prefix=PCH-ERR %s
// PCH-ERR: error: PCH file contains compiler errors

// RUN: c-index-test -write-pch %t.pch foobar.c 2>&1 | FileCheck -check-prefix=NONEXISTENT %s
// NONEXISTENT: Unable to load translation unit
