// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fms-compatibility -verify \
// RUN:   -Wno-missing-declarations -Wno-unused-value %s

void foo() {
  [] {
    struct { // expected-error {{anonymous structs and classes must be class members}}
      void bar(int & = "") {} // expected-error {{non-const lvalue reference to type 'int' cannot bind}}
                              // expected-note@-1 {{passing argument to parameter here}}
                              // expected-error@-2 {{functions cannot be declared in an anonymous struct}}
    };
  };
}

// Verify __FUNCTION__ in a lambda-local struct default argument resolves to
// the lambda's operator(), not the member function or enclosing function.
void test_lambda_function_in_default_arg() {
  (void)[] {
    struct S {
      void f2(const char *s = __FUNCTION__) {}
      // expected-warning@-1 {{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
    };
    return 0;
  }();
}

void test_lambda_function_size() {
  (void)[] {
    struct S {
      char proof[sizeof(__FUNCTION__) == 11 ? 1 : -1];
      void f2(const char *s = __FUNCTION__) {}
      // expected-warning@-1 {{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
    };
    return 0;
  }();
}
