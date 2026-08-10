// RUN: %clang_cc1 -fsyntax-only -verify %s

// Member Test1 hides class Test1
class Test1 {
  static int Test1; // expected-error {{member 'Test1' has the same name as its class}}
                    // expected-note@-1 {{class 'Test1' is hidden by a non-type declaration of 'Test1' here}}
  void fn1() {
    Test1 x; // expected-error {{must use 'class' tag to refer to type 'Test1' in this scope}}
  }
  int fn2() {
    return Test1;
  }
};

// Member Test2 doesn't hide class Test2 as its declaration is invalid
class Test2 { // expected-note {{declared here}}
  static NoSuchType Test2; // expected-error {{unknown type name 'NoSuchType'}}
                           // expected-error@-1 {{member 'Test2' has the same name as its class}}
  void fn1() {
    Test2 x;
  }
  int fn2() {
    return Test2; // expected-error {{'Test2' does not refer to a value}}
  }
};

// Test3a::x doesn't hide Test3b::x as its declaration is invalid
namespace Test3a {
  NoSuchType x() { return 0; } // expected-error {{unknown type name 'NoSuchType'}}
}
namespace Test3b {
  class x; // expected-note {{declared here}}
}
using Test3a::x;
using Test3b::x;
int test3_fn() {
  return x; // expected-error {{'x' does not refer to a value}}
}

// Function Test4 hides class Test4, whose declaration is invalid
class Test4 : public NoSuchType { // expected-error {{expected class name}}

};
int Test4() { return 0; }

int test4_fn() {
  return Test4();
}

// Function Test5 doesn't hide class Test5 when both are invalid
class Test5 : public NoSuchType { // expected-error {{expected class name}}

};
NoSuchType Test5() { return 0; } // expected-error {{unknown type name 'NoSuchType'}}

Test5 test5_fn() {
  Test5 x;
  return x;
}
