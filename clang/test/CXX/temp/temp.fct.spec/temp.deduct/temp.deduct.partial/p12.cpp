// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics

// Note: Partial ordering of function templates containing template
// parameter packs is independent of the number of deduced arguments
// for those template parameter packs.
template<class ...> struct Tuple { }; 
template<class ... Types> int &g(Tuple<Types ...>); // #1 
template<class T1, class ... Types> float &g(Tuple<T1, Types ...>); // #2
template<class T1, class ... Types> double &g(Tuple<T1, Types& ...>); // #3

void test_g() {
  int &ir1 = g(Tuple<>()); 
  float &fr1 = g(Tuple<int, float>()); 
  double &dr1 = g(Tuple<int, float&>()); 
  double &dr2 = g(Tuple<int>());
}

template<class ... Types> int &h(int (*)(Types ...)); // #1 
template<class T1, class ... Types> float &h(int (*)(T1, Types ...)); // #2
template<class T1, class ... Types> double &h(int (*)(T1, Types& ...)); // #3

void test_h() {
  int &ir1 = h((int(*)())0); 
  float &fr1 = h((int(*)(int, float))0);
  double &dr1 = h((int(*)(int, float&))0);
  double &dr2 = h((int(*)(int))0);
}
