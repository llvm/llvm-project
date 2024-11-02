// RUN: %clang_cc1 -fsyntax-only -std=c++17 -pedantic -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++20 -Wc++17-compat-pedantic -verify %s -Wno-defaulted-function-deleted

struct A {};
int (A::*pa)() const&;
int use_pa = (A().*pa)();
#if __cplusplus <= 201703L
  // expected-warning@-2 {{invoking a pointer to a 'const &' member function on an rvalue is a C++20 extension}}
#else
  // expected-warning@-4 {{invoking a pointer to a 'const &' member function on an rvalue is incompatible with C++ standards before C++20}}
#endif

struct B {
  void b() {
    (void) [=, this] {};
#if __cplusplus <= 201703L
    // expected-warning@-2 {{explicit capture of 'this' with a capture default of '=' is a C++20 extension}}
#else
    // expected-warning@-4 {{explicit capture of 'this' with a capture default of '=' is incompatible with C++ standards before C++20}}
#endif
  }

  int n : 5 = 0;
#if __cplusplus <= 201703L
    // expected-warning@-2 {{default member initializer for bit-field is a C++20 extension}}
#else
    // expected-warning@-4 {{default member initializer for bit-field is incompatible with C++ standards before C++20}}
#endif
};

auto Lambda = []{};
decltype(Lambda) AnotherLambda;
#if __cplusplus <= 201703L
    // expected-error@-2 {{no matching constructor}} expected-note@-3 2{{candidate}}
#else
    // expected-warning@-4 {{default construction of lambda is incompatible with C++ standards before C++20}}
#endif

void copy_lambda() { Lambda = Lambda; }
#if __cplusplus <= 201703L
    // expected-error@-2 {{deleted}} expected-note@-10 {{lambda}}
#else
    // expected-warning@-4 {{assignment of lambda is incompatible with C++ standards before C++20}}
#endif

struct DefaultDeleteWrongTypeBase {
  DefaultDeleteWrongTypeBase(DefaultDeleteWrongTypeBase&);
};
struct DefaultDeleteWrongType : DefaultDeleteWrongTypeBase {
  DefaultDeleteWrongType(const DefaultDeleteWrongType&) = default;
#if __cplusplus <= 201703L
    // expected-error@-2 {{a member or base requires it to be non-const}}
#else
    // expected-warning@-4 {{explicitly defaulting this copy constructor with a type different from the implicit type is incompatible with C++ standards before C++20}}
#endif
};

void ForRangeInit() {
  for (int arr[3] = {1, 2, 3}; int n : arr) {}
#if __cplusplus <= 201703L
    // expected-warning@-2 {{range-based for loop initialization statements are a C++20 extension}}
#else
    // expected-warning@-4 {{range-based for loop initialization statements are incompatible with C++ standards before C++20}}
#endif
}

struct ConstexprVirtual {
  virtual constexpr void f() {}
#if __cplusplus <= 201703L
    // expected-error@-2 {{virtual function cannot be constexpr}}
#else
    // expected-warning@-4 {{virtual constexpr functions are incompatible with C++ standards before C++20}}
#endif
};

struct C { int x, y, z; };
static auto [cx, cy, cz] = C();
#if __cplusplus <= 201703L
    // expected-warning@-2 {{decomposition declaration declared 'static' is a C++20 extension}}
#else
    // expected-warning@-4 {{decomposition declaration declared 'static' is incompatible with C++ standards before C++20}}
#endif
void f() {
  static thread_local auto [cx, cy, cz] = C();
#if __cplusplus <= 201703L
    // expected-warning@-2 {{decomposition declaration declared with 'static thread_local' specifiers is a C++20 extension}}
#else
    // expected-warning@-4 {{decomposition declaration declared with 'static thread_local' specifiers is incompatible with C++ standards before C++20}}
#endif
}

struct DefaultedComparisons {
  bool operator==(const DefaultedComparisons&) const = default;
  bool operator!=(const DefaultedComparisons&) const = default;
#if __cplusplus <= 201703L
  // expected-warning@-3 {{defaulted comparison operators are a C++20 extension}}
  // expected-warning@-3 {{defaulted comparison operators are a C++20 extension}}
#else
  // expected-warning@-6 {{defaulted comparison operators are incompatible with C++ standards before C++20}}
  // expected-warning@-6 {{defaulted comparison operators are incompatible with C++ standards before C++20}}
#endif
  bool operator<=>(const DefaultedComparisons&) const = default;
#if __cplusplus <= 201703L
  // expected-error@-2 {{'operator<=' cannot be the name of a variable or data member}} expected-error@-2 0+{{}} expected-warning@-2 {{}}
#else
  // expected-warning@-4 {{'<=>' operator is incompatible with C++ standards before C++20}}
#endif
  bool operator<(const DefaultedComparisons&) const = default;
  bool operator<=(const DefaultedComparisons&) const = default;
  bool operator>(const DefaultedComparisons&) const = default;
  bool operator>=(const DefaultedComparisons&) const = default;
#if __cplusplus <= 201703L
  // expected-error@-5 {{only special member functions}}
  // expected-error@-5 {{only special member functions}}
  // expected-error@-5 {{only special member functions}}
  // expected-error@-5 {{only special member functions}}
#else
  // expected-warning@-10 {{defaulted comparison operators are incompatible with C++ standards before C++20}}
  // expected-warning@-10 {{defaulted comparison operators are incompatible with C++ standards before C++20}}
  // expected-warning@-10 {{defaulted comparison operators are incompatible with C++ standards before C++20}}
  // expected-warning@-10 {{defaulted comparison operators are incompatible with C++ standards before C++20}}
#endif
};

namespace NTTP {
  struct A {};
  template<A> struct Class {};
#if __cplusplus <= 201703L
  // expected-error@-2 {{non-type template parameter cannot have type 'A' before C++20}}
#else
  // expected-warning@-4 {{non-type template parameter of type 'A' is incompatible with C++ standards before C++20}}
#endif
}
