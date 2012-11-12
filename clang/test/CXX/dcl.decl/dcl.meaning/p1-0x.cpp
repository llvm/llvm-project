// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// The nested-name-specifier of a qualified declarator-id shall not begin with a decltype-specifier.
class foo {
  static int i;
  void func();
};

int decltype(foo())::i; // expected-error{{'decltype' cannot be used to name a declaration}}
void decltype(foo())::func() { // expected-error{{'decltype' cannot be used to name a declaration}}
}


template<typename T>
class tfoo {
  static int i;
  void func();
};

template<typename T>
int decltype(tfoo<T>())::i; // expected-error{{nested name specifier 'decltype(tfoo<T>())::' for declaration does not refer into a class, class template or class template partial specialization}}
template<typename T>
void decltype(tfoo<T>())::func() { // expected-error{{nested name specifier 'decltype(tfoo<T>())::' for declaration does not refer into a class, class template or class template partial specialization}}
}
