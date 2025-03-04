// RUN:  %clang_cc1 -std=c++20 -verify %s
// [temp.deduct.p9]
// A lambda-expression appearing in a function type or a template parameter is
// not considered part of the immediate context for the purposes of template
// argument deduction.
// [Note: The intent is to avoid requiring implementations to deal with
// substitution failure involving arbitrary statements.]
template <class T>
auto f(T) -> decltype([]() { T::invalid; } ());
void f(...);
void test_f() {
  f(0); // expected-error@-3 {{type 'int' cannot be used prior to '::'}}
        // expected-note@-1 {{while substituting deduced template arguments}}
        // expected-note@-5 {{while substituting into a lambda expression here}}
}

template <class T, unsigned = sizeof([]() { T::invalid; })>
// expected-note@-1 {{template parameter is declared here}}
void g(T);
void g(...);
void test_g() {
  g(0); // expected-error@-5 {{type 'int' cannot be used prior to '::'}}
        // expected-note@-4 {{in instantiation of default argument}}
        // expected-note@-2 {{while substituting deduced template arguments}}
        // expected-note@-8 {{while substituting into a lambda expression here}}
}

template <class T>
auto h(T) -> decltype([x = T::invalid]() { });
void h(...);
void test_h() {
  h(0);
}

template <class T>
auto i(T) -> decltype([]() -> typename T::invalid { });
void i(...);
void test_i() {
  i(0);
}


// In this example, the lambda itself is not part of an immediate context, but
// substitution to the lambda expression succeeds, producing dependent
// `decltype(x.invalid)`. The call to the lambda, however, is in the immediate context
// and it produces a SFINAE failure. Hence, we pick the second overload
// and don't produce any errors.
template <class T>
auto j(T t) -> decltype([](auto x) -> decltype(x.invalid) { } (t));   // #1
void j(...);                                                          // #2
void test_j() {
  j(0); // deduction fails on #1, calls #2.
}
