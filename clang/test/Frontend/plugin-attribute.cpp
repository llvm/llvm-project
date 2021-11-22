// RUN: split-file %s %t
// RUN: %clang -cc1 -load %llvmshlibdir/Attribute%pluginext -fsyntax-only -ast-dump -verify %t/good_attr.cpp | FileCheck %s
// RUN: %clang -fplugin=%llvmshlibdir/Attribute%pluginext -fsyntax-only -Xclang -verify %t/bad_attr.cpp
// REQUIRES: plugins, examples
//--- good_attr.cpp
// expected-no-diagnostics
void fn1a() __attribute__((example)) {}
[[example]] void fn1b() {}
[[plugin::example]] void fn1c() {}
void fn2() __attribute__((example("somestring", 1, 2.0))) {}
// CHECK-COUNT-4: -AnnotateAttr 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} "example"
// CHECK: -StringLiteral 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} 'const char[{{[0-9]+}}]' lvalue "somestring"
// CHECK: -IntegerLiteral 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} 'int' 1
// CHECK: -FloatingLiteral 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} 'double' 2.000000e+00

//--- bad_attr.cpp
int var1 __attribute__((example("otherstring"))) = 1; // expected-warning {{'example' attribute only applies to functions}}
class Example {
  void __attribute__((example)) fn3(); // expected-error {{'example' attribute only allowed at file scope}}
};
void fn4() __attribute__((example(123))) { } // expected-error {{first argument to the 'example' attribute must be a string literal}}
void fn5() __attribute__((example("a","b", 3, 4.0))) { } // expected-error {{'example' attribute only accepts at most three arguments}}
