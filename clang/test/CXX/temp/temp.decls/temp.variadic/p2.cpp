// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

template<class ... Types> void f(Types ... args);

void test() {
  f(); 
  f(1); 
  f(2, 1.0);
}

// Test simple recursive variadic function template
template<typename Head, typename ...Tail>
void recurse_until_fail(const Head &, const Tail &...tail) { // expected-note{{candidate function template not viable: requires at least 1 argument, but 0 were provided}}
  recurse_until_fail(tail...); // expected-error{{no matching function for call to 'recurse_until_fail'}} \
  // expected-note{{in instantiation of function template specialization 'recurse_until_fail<char [7], >' requested here}} \
  // expected-note{{in instantiation of function template specialization 'recurse_until_fail<double, char [7]>' requested here}}
}

void test_recurse_until_fail() {
  recurse_until_fail(1, 3.14159, "string");   // expected-note{{in instantiation of function template specialization 'recurse_until_fail<int, double, char [7]>' requested here}}

}
