// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics

namespace ParameterPacksWithFunctions {
  template<typename ...> struct count;

  template<typename Head, typename ...Tail>
  struct count<Head, Tail...> {
    static const unsigned value = 1 + count<Tail...>::value;
  };

  template<>
  struct count<> {
    static const unsigned value = 0;
  };
  
  template<unsigned> struct unsigned_c { };

  template<typename ... Types>
  unsigned_c<count<Types...>::value> f();

  void test_f() {
    unsigned_c<0> uc0a = f(); // okay, deduced to an empty pack
    unsigned_c<0> uc0b = f<>();
    unsigned_c<1> uc1 = f<int>();
    unsigned_c<2> uc2 = f<float, double>();
  }
}
