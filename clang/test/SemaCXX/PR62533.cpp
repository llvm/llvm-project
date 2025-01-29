// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T>
struct test {
  template<typename> using fun_diff = char; // expected-note 2{{type alias template declared here}}
};

template<typename T, typename V>
decltype(T::template fun_diff<V>) foo1() {}
// expected-note@-1 {{candidate template ignored: substitution failure [with T = test<int>, V = int]: 'test<int>::fun_diff' is expected to be a non-type template, but instantiated to a type alias template}}

template<typename T>
void foo2() {
  // expected-error@+1 {{test<int>::fun_diff' is expected to be a non-type template, but instantiated to a type alias template}}
  int a = test<T>::template fun_diff<int>;
}

template<typename T, typename V>
struct has_fun_diff {
  using type = double;
};

template<typename T>
struct has_fun_diff<T, int> {
  // expected-error@+1 {{'test<int>::fun_diff' is expected to be a non-type template, but instantiated to a type alias template}}
  using type = decltype(T::template fun_diff<int>);
};

void bar() {
  foo1<test<int>, int>(); // expected-error {{no matching function for call to 'foo1'}}
  foo2<int>(); // expected-note {{in instantiation of function template specialization}}
  has_fun_diff<test<int>, int>::type a; // expected-note {{in instantiation of template class}}
}
