// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics

// If type deduction cannot be done for any P/A pair, or if for any
// pair the deduction leads to more than one possible set of deduced
// values, or if different pairs yield different deduced values, or if
// any template argument remains neither deduced nor explicitly
// specified, template argument deduction fails.

template<typename ...> struct tuple;

template<typename T, typename U>
struct same_tuple {
  static const bool value = false;
};

template<typename ...Types1>
struct same_tuple<tuple<Types1...>, tuple<Types1...> > {
  static const bool value = true;
};

int same_tuple_check1[same_tuple<tuple<int, float>, tuple<int, double>>::value? -1 : 1];
int same_tuple_check2[same_tuple<tuple<float, double>, tuple<float, double>>::value? 1 : -1];

