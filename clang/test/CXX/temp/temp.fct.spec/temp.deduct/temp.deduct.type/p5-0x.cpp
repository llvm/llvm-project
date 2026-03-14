// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

// FIXME: More bullets to go!

template<typename T, typename U>
struct has_nondeduced_pack_test {
  static const bool value = false;
};

template<typename R, typename FirstType, typename ...Types>
struct has_nondeduced_pack_test<R(FirstType, Types..., int), 
                                R(FirstType, Types...)> {
  static const bool value = true;
};

// - A function parameter pack that does not occur at the end of the
//   parameter-declaration-clause.
//
// We interpret [temp.deduct.call]p1's
//
//   "When a function parameter pack appears in a non-deduced context
//   (12.9.2.5), the type of that pack is never deduced."
//
// as applying in all deduction contexts, not just [temp.deduct.call],
// so we do *not* deduce Types from the second argument here. (More
// precisely, we deduce it as <> when processing the first argument,
// and then fail because 'int' doesn't match 'double, int'.)
int check_nondeduced_pack_test0[
                   has_nondeduced_pack_test<int(float, double, int),
                                            int(float, double)>::value? -1 : 1];

template<typename ...T> void has_non_trailing_pack(T ..., int);
void (*ptr_has_non_trailing_pack)(char, int) = has_non_trailing_pack<char>;

template<typename ...T, typename U> void has_non_trailing_pack_and_more(T ..., U); // expected-note {{failed}}
void (*ptr_has_non_trailing_pack_and_more_1)(float, double, int) = &has_non_trailing_pack_and_more<float, double>;
void (*ptr_has_non_trailing_pack_and_more_2)(float, double, int) = &has_non_trailing_pack_and_more<float>; // expected-error {{does not match}}

// - A function parameter for which the associated argument is an initializer
//   list but the parameter does not have a type for which deduction from an
//   initializer list is specified.

// We interpret these "non-deduced context"s as actually deducing the arity --
// but not the contents -- of a function parameter pack appropriately for the
// number of arguments.
namespace VariadicVsInitList {
  template<typename T, typename ...> struct X { using type = typename T::error; };
  template<typename ...T, typename X<int, T...>::type = 0> void f(T ...) = delete;
  void f(long);
  void f(long, long);
  void f(long, long, long);

  // FIXME: We shouldn't say "substitution failure: " here.
  template<typename ...T> void g(T ...) = delete; // expected-note {{substitution failure: deduced incomplete pack <(no value)> for template parameter 'T'}}

  void h() {
    // These all call the non-template overloads of 'f', because of a deduction
    // failure due to incomplete deduction of the pack 'T'. If deduction
    // succeeds and deduces an empty pack instead, we would get a hard error
    // instantiating 'X'.
    f({0}); // expected-warning {{braces around scalar}}
    f({0}, {0}); // expected-warning 2{{braces around scalar}}
    f(1, {0}); // expected-warning {{braces around scalar}}
    f(1, {0}, 2); // expected-warning {{braces around scalar}}

    g({0}); // expected-error {{no matching function}}
  }
}
