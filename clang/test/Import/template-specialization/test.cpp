// RUN: clang-import-test -dump-ast -import %S/Inputs/T.cpp -expression %s | FileCheck %s

// CHECK: |-ClassTemplateSpecializationDecl
// CHECK-SAME: <line:4:1, line:8:1> line:4:20 struct A

void expr() {
  A<int>::B b1;
  A<bool>::B b2;
  b1.f + b2.g;
}

static_assert(f<char>() == 0, "");
static_assert(f<int>() == 4, "");
