// Test without serialization:
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown -ast-dump %s \
// RUN: | FileCheck -strict-whitespace %s
//
// Test with serialization:
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown -x c++ \
// RUN:   -include-pch %t -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck -strict-whitespace %s

// Verify that a C++20 reversed rewritten operator does not produce inverted
// source ranges on the synthesized CXXOperatorCallExpr and UnaryOperator.
// When `A{} != B{}` is rewritten as `!(B{}.operator==(A{}))`, the arguments
// are reversed in the AST but the source ranges must stay in source order.

struct B {
  bool operator==(B);
};
struct A {
  operator B();
};
bool x = A{} != B{};

// CHECK: VarDecl {{.*}} <line:23:1, col:19> col:6 x 'bool'
// CHECK-NEXT: `-ExprWithCleanups {{.*}} <col:10, col:19>
// CHECK-NEXT:   `-CXXRewrittenBinaryOperator {{.*}} <col:10, col:19>
// CHECK-NEXT:     `-UnaryOperator {{.*}} <col:14, col:19> 'bool' prefix '!'
// CHECK-NEXT:       `-CXXOperatorCallExpr {{.*}} <col:10, col:19> 'bool' '=='
