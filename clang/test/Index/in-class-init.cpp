struct S {
  int field = 2;
};

// RUN: c-index-test -test-load-source all -std=c++11 %s | FileCheck %s
// CHECK: 2:7: FieldDecl=field:2:7 (Definition) Extent=[2:3 - 2:16]
