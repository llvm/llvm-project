// RUN: c-index-test -test-load-source all %s | FileCheck %s

// This test case previously just crashed the frontend.

struct abc *P;
int main(

// CHECK: StructDecl=abc:5:8 Extent=[5:1 - 5:11]
// CHECK: VarDecl=P:5:13 (Definition) Extent=[5:1 - 5:14]
// CHECK: VarDecl=main:6:5 (Definition) Extent=[6:1 - 6:9]

