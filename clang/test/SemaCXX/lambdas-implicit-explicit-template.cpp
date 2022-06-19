// RUN: %clang_cc1 -std=c++20 -DEXPLICIT -verify %s
// RUN: %clang_cc1 -std=c++17 -DEXPLICIT -verify -Wno-c++20-extensions %s
// RUN: %clang_cc1 -std=c++14 -verify %s

// expected-no-diagnostics

#ifdef EXPLICIT

template <typename F>
void a(F &&f) {
  f.template operator()<0>();
}

template <typename F>
void b(F &&f) {
  a([=]<int i>() {
    f.template operator()<i>();
  });
}

void c() {
  b([&]<int i>() {
  });
}

#endif

template <typename F> void a1(F f) { f.operator()(0); }

template <typename F> void b1(F f) {
  a1([=](auto i) { f.operator()(i); });
}

void c1() {
  b1([&](auto i) {});
}

void c2() {
  const auto lambda = [&](auto arg1) {};
  [&](auto arg2) { lambda.operator()(arg2); }(0);
}

auto d = [](auto) {};

template <typename T>
void d1(T x) { d.operator()(x); }

void d2() { d1(0); }

template <typename T> int e1 = [](auto){ return T(); }.operator()(T());
int e2 = e1<int>;
