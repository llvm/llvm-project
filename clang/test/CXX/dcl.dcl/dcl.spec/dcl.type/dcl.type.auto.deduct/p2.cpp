// RUN: %clang_cc1 -std=c++2b -verify %s

// p2.3 allows only T = auto in T(x).

void test_decay() {
  int v[3];
  static_assert(__is_same(decltype(auto(v)), int *));
  static_assert(__is_same(decltype(auto{v}), int *));
  static_assert(__is_same(decltype(auto("lit")), char const *));
  static_assert(__is_same(decltype(auto{"lit"}), char const *));

  constexpr long i = 1;
  static_assert(__is_same(decltype(i), long const));
  static_assert(__is_same(decltype(auto(1L)), long));
  static_assert(__is_same(decltype(auto{1L}), long));
  static_assert(__is_same(decltype(auto(i)), long));
  static_assert(__is_same(decltype(auto{i}), long));

  class A {
  } a;
  A const ac;

  static_assert(__is_same(decltype(auto(a)), A));
  static_assert(__is_same(decltype(auto(ac)), A));

  A &lr = a;
  A const &lrc = a;
  A &&rr = static_cast<A &&>(a);
  A const &&rrc = static_cast<A &&>(a);

  static_assert(__is_same(decltype(auto(lr)), A));
  static_assert(__is_same(decltype(auto(lrc)), A));
  static_assert(__is_same(decltype(auto(rr)), A));
  static_assert(__is_same(decltype(auto(rrc)), A));
}

class cmdline_parser {
public:
  cmdline_parser(char const *);
  auto add_option(char const *, char const *) && -> cmdline_parser &&;
};

void test_rvalue_fluent_interface() {
  auto cmdline = cmdline_parser("driver");
  auto internal = auto{cmdline}.add_option("--dump-full", "do not minimize dump");
}

template <class T> constexpr auto decay_copy(T &&v) { return static_cast<T &&>(v); } // expected-error {{calling a protected constructor}}

class A {
  int x;
  friend void f(A &&);

public:
  A();

  auto test_access() {
    static_assert(__is_same(decltype(auto(*this)), A));
    static_assert(__is_same(decltype(auto(this)), A *));

    f(A(*this));          // ok
    f(auto(*this));       // ok in P0849
    f(decay_copy(*this)); // expected-note {{in instantiation of function template specialization}}
  }

  auto test_access() const {
    static_assert(__is_same(decltype(auto(*this)), A)); // ditto
    static_assert(__is_same(decltype(auto(this)), A const *));
  }

protected:
  A(const A &); // expected-note {{declared protected here}}
};

// post-C++17 semantics
namespace auto_x {
constexpr struct Uncopyable {
  constexpr explicit Uncopyable(int) {}
  constexpr Uncopyable(Uncopyable &&) = delete;
} u = auto(Uncopyable(auto(Uncopyable(42))));
} // namespace auto_x
