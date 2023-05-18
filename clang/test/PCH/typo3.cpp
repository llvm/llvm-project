// RUN: not %clang_cc1 -emit-pch %s -o %t.pch 2>&1 | FileCheck %s

struct S {
  // Make sure TypoExprs in default init exprs are corrected before serializing
  // in PCH.
  int y = bar;
  // CHECK: use of undeclared identifier 'bar'
};
