// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fsyntax-only %s

// REQUIRES: aarch64-registered-target || arm-registered-target

// expected-no-diagnostics

struct a {};
__SVFloat32_t b(a);
template <class c> using e = decltype(b(c()));
e<a> f(a);
template <class c> using h = decltype(f(c()));
template <class g> struct i {
  static void j() {
    a d;
    g()(d);
  }
};
struct k {
  template <class c> void operator()(c) {
    [](h<c>) {};
  }
  void l() { i<k>::j; }
};
