// RUN: %clang_cc1 -std=c++26 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify=cxx23 %s

// Test for C++26 [[indeterminate]] attribute (P2795R5)

// In C++23, the attribute is unknown and ignored
void test_cxx23() {
  [[indeterminate]] int x;  // cxx23-warning {{'indeterminate' attribute ignored}}
}

#if __cplusplus >= 202400L

// local variable with automatic storage duration
void test_local_var() {
  [[indeterminate]] int x;       // OK
  [[indeterminate]] int arr[10]; // OK
  [[indeterminate]] int a, b, c; // OK - multiple declarators
}

// function parameter
void test_param([[indeterminate]] int x);  // OK

// static storage duration
// expected-warning@+1 {{'indeterminate' attribute only applies to local variables or function parameters}}
[[indeterminate]] int global_var;

void test_static() {
  // expected-warning@+1 {{'indeterminate' attribute only applies to local variables or function parameters}}
  [[indeterminate]] static int x;
  // expected-warning@+1 {{'indeterminate' attribute only applies to local variables or function parameters}}
  [[indeterminate]] thread_local int y;
}

// attribute on class-type local variable
struct S {
  int x;
  S() {}
};

void test_class_type() {
  [[indeterminate]] S s;  // OK - member x has indeterminate value
}

// constexpr context should error on reading indeterminate value
constexpr int test_constexpr() {
  [[indeterminate]] int x;
  return x;  // expected-note {{read of uninitialized object is not allowed in a constant expression}}
}
constexpr int val_constexpr = test_constexpr(); // expected-error {{constexpr variable 'val_constexpr' must be initialized by a constant expression}} \
                                                // expected-note {{in call to 'test_constexpr()'}}

// declaration position
void test_decl_position() {
  int x [[indeterminate]];  // OK - attribute on declarator
  [[indeterminate]] int y;  // OK - attribute at beginning
}

// [[indeterminate]] must be on first declaration (P2795R5 [dcl.attr.indet]/p2)
void first_decl_test(int x);                                // first declaration without attribute
void first_decl_test([[indeterminate]] int x);              // expected-error {{'[[indeterminate]]' attribute on parameter 'x' must appear on the first declaration of the function}}
                                                            // expected-note@-2 {{previous declaration is here}}

void first_decl_ok([[indeterminate]] int x);                // first declaration with attribute - OK
void first_decl_ok([[indeterminate]] int x) {}              // redeclaration with attribute - OK

#endif // __cplusplus >= 202400L
