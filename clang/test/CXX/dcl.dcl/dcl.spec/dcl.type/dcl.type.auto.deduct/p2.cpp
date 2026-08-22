// RUN: %clang_cc1 -std=c++20 -verify=expected,cxx20 %s
// RUN: %clang_cc1 -std=c++23 -verify %s

// p2.3 allows only T = auto in T(x).

void test_decay() {
  int v[3];
  static_assert(__is_same(decltype(auto(v)), int *)); // cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
  static_assert(__is_same(decltype(auto{v}), int *)); // cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
  static_assert(__is_same(decltype(auto("lit")), char const *)); // cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
  static_assert(__is_same(decltype(auto{"lit"}), char const *)); // cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}

  constexpr long i = 1;
  static_assert(__is_same(decltype(i), long const));
  static_assert(__is_same(decltype(auto(1L)), long)); // cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
  static_assert(__is_same(decltype(auto{1L}), long)); // cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
  static_assert(__is_same(decltype(auto(i)), long));  // cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
  static_assert(__is_same(decltype(auto{i}), long));  // cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}

  class A {
  } a;
  A const ac;

  static_assert(__is_same(decltype(auto(a)), A));  // cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
  static_assert(__is_same(decltype(auto(ac)), A)); // cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}

  A &lr = a;
  A const &lrc = a;
  A &&rr = static_cast<A &&>(a);
  A const &&rrc = static_cast<A &&>(a);

  static_assert(__is_same(decltype(auto(lr)), A));  // cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
  static_assert(__is_same(decltype(auto(lrc)), A)); // cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
  static_assert(__is_same(decltype(auto(rr)), A));  // cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
  static_assert(__is_same(decltype(auto(rrc)), A)); // cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
}

class cmdline_parser {
public:
  cmdline_parser(char const *);
  auto add_option(char const *, char const *) && -> cmdline_parser &&;
};

void test_rvalue_fluent_interface() {
  auto cmdline = cmdline_parser("driver");
  auto internal = auto{cmdline}.add_option("--dump-full", "do not minimize dump"); // cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
}

template <class T> constexpr auto decay_copy(T &&v) { return static_cast<T &&>(v); } // expected-error {{calling a protected constructor}}

class A {
  int x;
  friend void f(A &&);

public:
  A();

  auto test_access() {
    static_assert(__is_same(decltype(auto(*this)), A));  // cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
    static_assert(__is_same(decltype(auto(this)), A *)); // cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}

    f(A(*this));          // ok
    f(auto(*this));       // ok in P0849 cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
    f(decay_copy(*this)); // expected-note {{in instantiation of function template specialization}}
  }

  auto test_access() const {
    static_assert(__is_same(decltype(auto(*this)), A)); // ditto cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
    static_assert(__is_same(decltype(auto(this)), A const *)); // cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
  }

protected:
  A(const A &); // expected-note {{declared protected here}}
};

// post-C++17 semantics
namespace auto_x {
constexpr struct Uncopyable {
  constexpr explicit Uncopyable(int) {}
  constexpr Uncopyable(Uncopyable &&) = delete;
} u = auto(Uncopyable(auto(Uncopyable(42)))); // cxx20-warning 2 {{'auto' as a functional-style cast is a C++23 extension}}
} // namespace auto_x
