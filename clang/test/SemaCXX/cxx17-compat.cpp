// RUN: %clang_cc1 -fsyntax-only -std=c++17 -pedantic -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++2a -Wc++17-compat-pedantic -verify %s -Wno-defaulted-function-deleted

struct A {};
int (A::*pa)() const&;
int use_pa = (A().*pa)();
#if __cplusplus <= 201703L
  // expected-warning@-2 {{invoking a pointer to a 'const &' member function on an rvalue is a C++2a extension}}
#else
  // expected-warning@-4 {{invoking a pointer to a 'const &' member function on an rvalue is incompatible with C++ standards before C++2a}}
#endif

struct B {
  void b() {
    (void) [=, this] {};
#if __cplusplus <= 201703L
    // expected-warning@-2 {{explicit capture of 'this' with a capture default of '=' is a C++2a extension}}
#else
    // expected-warning@-4 {{explicit capture of 'this' with a capture default of '=' is incompatible with C++ standards before C++2a}}
#endif
  }

  int n : 5 = 0;
#if __cplusplus <= 201703L
    // expected-warning@-2 {{default member initializer for bit-field is a C++2a extension}}
#else
    // expected-warning@-4 {{default member initializer for bit-field is incompatible with C++ standards before C++2a}}
#endif
};

auto Lambda = []{};
decltype(Lambda) AnotherLambda;
#if __cplusplus <= 201703L
    // expected-error@-2 {{no matching constructor}} expected-note@-3 2{{candidate}}
#else
    // expected-warning@-4 {{default construction of lambda is incompatible with C++ standards before C++2a}}
#endif

void copy_lambda() { Lambda = Lambda; }
#if __cplusplus <= 201703L
    // expected-error@-2 {{deleted}} expected-note@-10 {{lambda}}
#else
    // expected-warning@-4 {{assignment of lambda is incompatible with C++ standards before C++2a}}
#endif

struct DefaultDeleteWrongTypeBase {
  DefaultDeleteWrongTypeBase(DefaultDeleteWrongTypeBase&);
};
struct DefaultDeleteWrongType : DefaultDeleteWrongTypeBase {
  DefaultDeleteWrongType(const DefaultDeleteWrongType&) = default;
#if __cplusplus <= 201703L
    // expected-error@-2 {{a member or base requires it to be non-const}}
#else
    // expected-warning@-4 {{explicitly defaulting this copy constructor with a type different from the implicit type is incompatible with C++ standards before C++2a}}
#endif
};

void ForRangeInit() {
  for (int arr[3] = {1, 2, 3}; int n : arr) {}
#if __cplusplus <= 201703L
    // expected-warning@-2 {{range-based for loop initialization statements are a C++2a extension}}
#else
    // expected-warning@-4 {{range-based for loop initialization statements are incompatible with C++ standards before C++2a}}
#endif
}

struct ConstexprVirtual {
  virtual constexpr void f() {}
#if __cplusplus <= 201703L
    // expected-error@-2 {{virtual function cannot be constexpr}}
#else
    // expected-warning@-4 {{virtual constexpr functions are incompatible with C++ standards before C++2a}}
#endif
};
