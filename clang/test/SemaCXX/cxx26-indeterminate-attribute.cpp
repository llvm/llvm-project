// RUN: %clang_cc1 -std=c++26 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -pedantic -verify=cxx23 %s

// Test for C++26 [[indeterminate]] attribute (P2795R5)

// In C++23, the attribute is accepted as an extension with a warning.
void test_cxx23() {
  [[indeterminate]] int x;  // cxx23-warning {{use of the 'indeterminate' attribute is a C++26 extension}}
}

#if __cplusplus >= 202400L

void test_local_var() {
  [[indeterminate]] int x;       // OK
  [[indeterminate]] int arr[10]; // OK
  [[indeterminate]] int a, b, c; // OK - multiple declarators
}

void test_param([[indeterminate]] int x);  // OK

// expected-warning@+1 {{'indeterminate' attribute only applies to local variables or function parameters}}
[[indeterminate]] int global_var;

void test_static() {
  // expected-warning@+1 {{'indeterminate' attribute only applies to local variables or function parameters}}
  [[indeterminate]] static int x;
  // expected-warning@+1 {{'indeterminate' attribute only applies to local variables or function parameters}}
  [[indeterminate]] thread_local int y;
}

struct S {
  int x;
  S() {}
};

void test_class_type() {
  [[indeterminate]] S s;  // OK - member x has indeterminate value
}

struct S2 {
  [[indeterminate]] int x; // expected-warning {{'indeterminate' attribute only applies to local variables or function parameters}}
  S2() {}
};

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

struct NoCtorInit {
  int a;
  int b;
};

void test_struct_no_ctor() {
  [[indeterminate]] NoCtorInit n;  // OK - members have indeterminate values
}

struct PartialInit {
  int x;
  int y;
  constexpr PartialInit() : x(0) {} // y not initialized
};

void test_partial_init() {
  [[indeterminate]] PartialInit p;  // OK - y is indeterminate
}

template<typename T>
void test_template() {
  [[indeterminate]] T x;  // OK
}
void instantiate_templates() {
  test_template<int>();
  test_template<float>();
  test_template<S>();
}

template<typename... Ts>
void test_pack() {
  [[indeterminate]] int arr[sizeof...(Ts)];  // OK
}
void instantiate_pack() {
  test_pack<int, float, double>();
}

template<typename T>
void test_template_param([[indeterminate]] T x) {} // OK

void instantiate_template_param() {
  int a;
  test_template_param(a);
  test_template_param(3.14);
}

template<typename T, typename... Rest>
void test_variadic_param([[indeterminate]] T first) {
  [[indeterminate]] T local;  // OK
}
void instantiate_variadic() {
  test_variadic_param<int, float, double>(0);
}

#endif // __cplusplus >= 202400L
