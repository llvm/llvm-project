// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s

// expected-no-diagnostics

// Check recursive instantiation of lambda does not cause assertion.
// lambda function `f` in `fun1` is instantiated twice: first
// as f(f, Number<1>), then as f(f, Number<0>). The
// LocalInstantiationScopes of these two instantiations both contain
// `f` and `i`. However, since they are not merged, clang should not
// assert for that.

template <unsigned v>
struct Number
{
   static constexpr unsigned value = v;
};

template <unsigned IBegin = 0,
          unsigned IEnd = 1>
constexpr auto fun1(Number<IBegin> = Number<0>{}, Number<IEnd>  = Number<1>{})
{
  constexpr unsigned a = 0;
  auto f = [&](auto fs, auto i) {
    if constexpr(i.value > 0)
    {
      (void)a;
      return fs(fs, Number<IBegin>{});
    }
    (void)a;
  };

  return f(f, Number<IEnd>{});
}


void fun2() {
  fun1();
}
