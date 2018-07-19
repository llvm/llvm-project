// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics

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
int check_nondeduced_pack_test0[
                   has_nondeduced_pack_test<int(float, double, int),
                                            int(float, double)>::value? 1 : -1];


