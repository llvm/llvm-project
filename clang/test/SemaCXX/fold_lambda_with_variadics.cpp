// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify %s
// expected-no-diagnostics
struct tuple {
    int x[3];
};

template <class F>
int apply(F f, tuple v) {
    return f(v.x[0], v.x[1], v.x[2]);
}

int Cartesian1(auto x, auto y) {
    return apply([&](auto... xs) {
        return (apply([xs](auto... ys) {
            return (ys + ...);
        }, y) + ...);
    }, x);
}

int Cartesian2(auto x, auto y) {
    return apply([&](auto... xs) {
        return (apply([zs = xs](auto... ys) {
            return (ys + ...);
        }, y) + ...);
    }, x);
}

template <int ...> struct Ints{};
template <int> struct Choose {
  template<class> struct Templ;
};
template <int ...x>
int Cartesian3(auto y) {
    return [&]<int ...xs>(Ints<xs...>) {
        // check in default template arguments for
        // - type template parameters,
        (void)(apply([]<class = decltype(xs)>(auto... ys) {
          return (ys + ...);
        }, y) + ...);
        // - template template parameters.
        (void)(apply([]<template<class> class = Choose<xs>::template Templ>(auto... ys) {
          return (ys + ...);
        }, y) + ...);
        // - non-type template parameters,
        return (apply([]<int = xs>(auto... ys) {
            return (ys + ...);
        }, y) + ...);

    }(Ints<x...>());
}

template <int ...x>
int Cartesian4(auto y) {
    return [&]<int ...xs>(Ints<xs...>) {
        return (apply([]<decltype(xs) xx = 1>(auto... ys) {
            return (ys + ...);
        }, y) + ...);
    }(Ints<x...>());
}

int Cartesian5(auto x, auto y) {
    return apply([&](auto... xs) {
        return (apply([](auto... ys) __attribute__((diagnose_if(!__is_same(decltype(xs), int), "message", "error"))) {
            return (ys + ...);
        }, y) + ...);
    }, x);
}


int main() {
    auto x = tuple({1, 2, 3});
    auto y = tuple({4, 5, 6});
    Cartesian1(x, y);
    Cartesian2(x, y);
    Cartesian3<1,2,3>(y);
    Cartesian4<1,2,3>(y);
    Cartesian5(x, y);
}
