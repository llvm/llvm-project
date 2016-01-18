// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++1y -DCXX1Y
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11 -Wno-c++1y-extensions

// FIXME: This is in p11 (?) in C++1y.
void f() {
  decltype(auto) a = a; // expected-error{{variable 'a' declared with 'auto' type cannot appear in its own initializer}}
  if (decltype(auto) b = b) {} // expected-error {{variable 'b' declared with 'auto' type cannot appear in its own initializer}}
  decltype(auto) c = ({ decltype(auto) d = c; 0; }); // expected-error {{variable 'c' declared with 'auto' type cannot appear in its own initializer}}
}

void g() {
  decltype(auto) a; // expected-error{{declaration of variable 'a' with type 'decltype(auto)' requires an initializer}}
  
  decltype(auto) *b; // expected-error{{cannot form pointer to 'decltype(auto)'}} expected-error{{declaration of variable 'b' with type 'decltype(auto) *' requires an initializer}}

  if (decltype(auto) b) {} // expected-error {{must have an initializer}}
  for (;decltype(auto) b;) {} // expected-error {{must have an initializer}}
  while (decltype(auto) b) {} // expected-error {{must have an initializer}}
  if (decltype(auto) b = true) { (void)b; }
}

decltype(auto) n(1,2,3); // expected-error{{initializer for variable 'n' with type 'decltype(auto)' contains multiple expressions}}

namespace N
{
  // All of these are references, because a string literal is an lvalue.
  decltype(auto) a = "const char (&)[19]", b = a, c = (a);
}

void h() {
  decltype(auto) b = 42ULL;

  for (decltype(auto) c = 0; c < b; ++c) {
  }
}

template<typename T, typename U> struct same;
template<typename T> struct same<T, T> {};

void i() {
  decltype(auto) x = 5;
  decltype(auto) int r; // expected-error {{cannot combine with previous 'decltype(auto)' declaration specifier}} expected-error {{requires an initializer}}
}

namespace p3_example {
  template<typename T, typename U> struct is_same_impl {
    static const bool value = false;
  };
  template<typename T> struct is_same_impl<T, T> {
    static const bool value = true;
  };
  template<typename T, typename U> constexpr bool is_same() {
    return is_same_impl<T,U>::value;
  }

  auto x = 5;
  const auto *v = &x, u = 6;
  static auto y = 0.0;
  auto int r;  // expected-warning {{storage class}} expected-error {{file-scope}}

  static_assert(is_same<decltype(x), int>(), "");
  static_assert(is_same<decltype(v), const int*>(), "");
  static_assert(is_same<decltype(u), const int>(), "");
  static_assert(is_same<decltype(y), double>(), "");

#ifdef CXX1Y
  auto f() -> int;
  auto g() { return 0.0; }
  auto h();

  static_assert(is_same<decltype(f), int()>(), "");
  static_assert(is_same<decltype(g), double()>(), "");
#endif
}
