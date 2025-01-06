// RUN: %clang_cc1 -std=c++20 -Wno-unused-value -fsyntax-only -verify %s

namespace GH49266 {
struct X {
  X() = default;
  X(X const&) = delete; // expected-note {{'X' has been explicitly marked deleted here}}
};

void take_by_copy(auto &...args) {
  [...args = args] {}(); // expected-error {{call to deleted constructor}}
}

void take_by_ref(auto &...args) {
  [&...args = args] {}(); // args is passed by reference and not copied.
}

void foo() {
  X x;
  take_by_copy(x); // expected-note {{in instantiation of function template specialization}}
  take_by_ref(x);
}
}

namespace GH48937 {

template <typename... Ts>
consteval int f(Ts... ts) {
  return ([]<Ts a = 42>(){ return a;}, ...)();
}

static_assert(f(0, 42) == 42);

template <typename Ts>
int g(Ts ts) {
  return ([]<Ts a = 42>(){ return a;}, ...)();  // expected-error {{pack expansion does not contain any unexpanded parameter packs}}
}

template <typename... Ts>
int h(Ts... ts) {
  return ([]<Ts a = 42>(){ return a;})();  // expected-error {{expression contains unexpanded parameter pack 'Ts'}}
}

}

namespace GH63677 {

template<typename>
void f() {
  []<typename... Ts>() -> void {
    [...us = Ts{}]{
      (Ts(us), ...);
    };
  }.template operator()<int, int>();
}

template void f<int>();

template <class>
inline constexpr auto fun =
  []<class... Ts>(Ts... ts) {
    return [... us = (Ts&&) ts]<class Fun>(Fun&& fn) mutable {
      return static_cast<Fun&&>(fn)(static_cast<Ts&&>(us)...);
    };
  };

void f() {
  [[maybe_unused]] auto s = fun<int>(1, 2, 3, 4);
}

}

namespace GH61460 {

template<typename... Ts>
void f1(Ts... ts);

template <typename... Ts> void g(Ts... p1s) {
  (void)[&](auto... p2s) {
    (
        [&] {
          p1s;
          f1(p1s);
          sizeof(p1s);
          p2s;
        },
        ...);
  };
}

template <typename... Ts> void g2(Ts... p1s) {
  (void)[&](auto... p2s) { [&] { p1s; p2s; }; }; // expected-error {{unexpanded parameter pack 'p2s'}}
}

void f1() { g(); }

} // namespace GH61460

namespace GH112352 {

template <class>
constexpr bool foo = false;

template <int>
constexpr bool bar = false;

template <template<class> class>
constexpr bool baz = false;

struct S {
  template <typename... Types, int... Values> void foldExpr1() {
    (void)[]<int... Is> {
      ([] {
        Is;
        // Propagate up the flag ContainsUnexpandedParameterPack from VarDecl.
        S var(foo<Types>);
        foo<Types>;
        bar<Values>;
        int a = Values;
      } &&
       ...);
    };
  }

  template <template<class> class... TTPs> void foldExpr2() {
    (void)[]<int... Is> {
      ([] {
        Is;
        baz<TTPs>;
        TTPs<int> D;
      } && ...);
    };
  }
};

void use() {
  S().foldExpr1();
  S().foldExpr2();
}

} // namespace GH112352
