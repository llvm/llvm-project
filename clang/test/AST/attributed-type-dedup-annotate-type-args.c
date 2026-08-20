// RUN: %clang_cc1 -emit-pch -o %t.pch %s
// RUN: llvm-bcanalyzer --dump --disable-histogram %t.pch | FileCheck %s

// annotate_type with VariadicExprArgument. The attribued types should
// dedupe not only based on the string argument, but all of the variadic
// arguments.

int *[[clang::annotate_type("foo", 1)]] a;
int *[[clang::annotate_type("foo", 1)]] b;
int *[[clang::annotate_type("foo", 2)]] c;
int *[[clang::annotate_type("foo")]] d;

// CHECK-COUNT-3: <TYPE_ATTRIBUTED
// CHECK-NOT:     <TYPE_ATTRIBUTED
