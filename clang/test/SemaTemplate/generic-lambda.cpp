// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

// expected-no-diagnostics

template <class T, class U> constexpr bool is_same_v = false;
template <class T> constexpr bool is_same_v<T, T> = true;
template <class T, class U>
concept is_same = is_same_v<T, U>;

template <class T> struct X {};
template <class T, class U>
concept C1 = is_same<T, X<U>>;

template <class T1> X<X<X<T1>>> t1() {
  return []<class T2>(T2) -> X<X<T2>> {
    struct S {
      static X<X<T2>> f() {
        return []<class T3>(T3) -> X<T3> {
          static_assert(is_same<T2, X<T1>>);
          static_assert(is_same<T3, X<T2>>);
          return X<T3>();
        }(X<T2>());
      }
    };
    return S::f();
  }(X<T1>());
};
template X<X<X<int>>> t1<int>();

#if 0 // FIXME: crashes
template<class T1> auto t2() {
  return []<class T2>(T2) {
    struct S {
      static auto f() {
        return []<class T3>(T3) {
          static_assert(is_same<T2, X<T1>>);
          static_assert(is_same<T3, X<T2>>);
          return X<T3>();
        }(X<T2>());
      }
    };
    return S::f();
  }(X<T1>());
};
template auto t2<int>();
static_assert(is_same<decltype(t2<int>()), X<X<X<int>>>>);

template<class T1> C1<X<X<T1>>> auto t3() {
  return []<C1<T1> T2>(T2) -> C1<X<T2>> auto {
    struct S {
      static auto f() {
        return []<C1<T2> T3>(T3) -> C1<T3> auto {
          return X<T3>();
        }(X<T2>());
      }
    };
    return S::f();
  }(X<T1>());
};
template C1<X<X<int>>> auto t3<int>();
static_assert(is_same<decltype(t3<int>()), X<X<X<int>>>>);
#endif

namespace GH95735 {

int g(int fn) {
  return [f = fn](auto tpl) noexcept(noexcept(f)) { return f; }(0);
}

int foo(auto... fn) {
  // FIXME: This one hits the assertion "if the exception specification is dependent,
  // then the noexcept expression should be value-dependent" in the constructor of
  // FunctionProtoType.
  // One possible solution is to update Sema::canThrow() to consider expressions
  // (e.g. DeclRefExpr/FunctionParmPackExpr) involving unexpanded parameters as Dependent.
  // This would effectively add an extra value-dependent flag to the noexcept expression.
  // However, I'm afraid that would also cause ABI breakage.
  // [...f = fn](auto tpl) noexcept(noexcept(f)) { return 0; }(0);
  [...f = fn](auto tpl) noexcept(noexcept(g(fn...))) { return 0; }(0);
  return [...f = fn](auto tpl) noexcept(noexcept(g(f...))) { return 0; }(0);
}

int v = foo(42);

} // namespace GH95735
