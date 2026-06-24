// RUN: split-file %s %t
// RUN: %clang_cc1 -load %llvmshlibdir/Attribute%pluginext -fsyntax-only -ast-dump -verify %t/good_attr.cpp | FileCheck %s
// RUN: %clang -fplugin=%llvmshlibdir/Attribute%pluginext -fsyntax-only -Xclang -verify %t/bad_attr.cpp
// REQUIRES: plugins, examples
//--- good_attr.cpp
// expected-no-diagnostics
void fn1a() __attribute__((example)) {
  __attribute__((example)) for (int i = 0; i < 9; ++i) {}
}
[[example]] void fn1b() {
  [[example]] for (int i = 0; i < 9; ++i) {}
}
[[plugin::example]] void fn1c() {
  [[plugin::example]] for (int i = 0; i < 9; ++i) {}
}
void fn2() __attribute__((example("somestring", 1, 2.0))) {
  __attribute__((example("abc", 3, 4.0))) for (int i = 0; i < 9; ++i) {}
}
template <int N> void template_fn() __attribute__((example("template", N))) {
  __attribute__((example("def", N + 1))) for (int i = 0; i < 9; ++i) {}
}
void fn3() { template_fn<5>(); }
// CHECK: -AttributedStmt 0x{{[0-9a-z]+}} {{<line:[0-9]+:[0-9]+(, col:[0-9]+)?>}}
// CHECK: -AnnotateAttr 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} "example"
// CHECK: -AnnotateAttr 0x{{[0-9a-z]+}} {{<line:[0-9]+:[0-9]+(, col:[0-9]+)?>}} "example"
// CHECK: -AttributedStmt 0x{{[0-9a-z]+}} {{<line:[0-9]+:[0-9]+(, col:[0-9]+)?>}}
// CHECK: -AnnotateAttr 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} "example"
// CHECK: -AnnotateAttr 0x{{[0-9a-z]+}} {{<line:[0-9]+:[0-9]+(, col:[0-9]+)?>}} "example"
// CHECK: -AttributedStmt 0x{{[0-9a-z]+}} {{<line:[0-9]+:[0-9]+(, col:[0-9]+)?>}}
// CHECK: -AnnotateAttr 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} "example"
// CHECK: -AnnotateAttr 0x{{[0-9a-z]+}} {{<line:[0-9]+:[0-9]+(, col:[0-9]+)?>}} "example"
// CHECK: -AttributedStmt 0x{{[0-9a-z]+}} {{<line:[0-9]+:[0-9]+(, col:[0-9]+)?>}}
// CHECK: -AnnotateAttr 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} "example"
// CHECK: -StringLiteral 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} 'const char[{{[0-9]+}}]' lvalue "abc"
// CHECK: -IntegerLiteral 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} 'int' 3
// CHECK: -FloatingLiteral 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} 'double' 4.000000e+00
// CHECK: -AnnotateAttr 0x{{[0-9a-z]+}} {{<line:[0-9]+:[0-9]+(, col:[0-9]+)?>}} "example"
// CHECK: -StringLiteral 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} 'const char[{{[0-9]+}}]' lvalue "somestring"
// CHECK: -IntegerLiteral 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} 'int' 1
// CHECK: -FloatingLiteral 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} 'double' 2.000000e+00
// CHECK: -AnnotateAttr 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} Implicit "example"
// CHECK: -StringLiteral 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} 'const char[{{[0-9]+}}]' lvalue "def"
// CHECK: -BinaryOperator 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} 'int' '+'
// CHECK: -IntegerLiteral 0x{{[0-9a-z]+}} {{<line:[0-9]+:[0-9]+(, col:[0-9]+)?>}} 'int' 5
// CHECK: -IntegerLiteral 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} 'int' 1
// CHECK: -AnnotateAttr 0x{{[0-9a-z]+}} {{<line:[0-9]+:[0-9]+(, col:[0-9]+)?>}} "example"
// CHECK: -StringLiteral 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} 'const char[{{[0-9]+}}]' lvalue "template"
// CHECK: -IntegerLiteral 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} 'int' 5

//--- bad_attr.cpp
int var1 __attribute__((example("otherstring"))) = 1; // expected-warning {{'example' attribute only applies to functions}}
class Example {
  void __attribute__((example)) fn3(); // expected-error {{'example' attribute only allowed at file scope}}
};
void fn4() __attribute__((example(123))) { // expected-error {{first argument to the 'example' attribute must be a string literal}}
  __attribute__((example("somestring"))) while (true); // expected-warning {{'example' attribute only applies to for loop statements}}
}
void fn5() __attribute__((example("a","b", 3, 4.0))) { // expected-error {{'example' attribute only accepts at most three arguments}}
  __attribute__((example("a","b", 3, 4.0))) for (int i = 0; i < 10; ++i) {} // expected-error {{'example' attribute only accepts at most three arguments}}
}
