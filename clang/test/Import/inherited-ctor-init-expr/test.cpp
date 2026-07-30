// RUN: clang-import-test -dump-ast -expression=%s -import=%S/Inputs/A.cpp | FileCheck %s
// CHECK: | | | `-CXXInheritedCtorInitExpr

void foo() {
  C c;
}
