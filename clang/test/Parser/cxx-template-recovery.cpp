// RUN: %clang_cc1 -fsyntax-only -verify %s

template <class T>
void Foo<T>::Bar(void* aRawPtr) { // expected-error {{no template named 'Foo'}}
    (void)(aRawPtr);
}

namespace baz {
  class klass {};
}

int *variable = 0; // ok
const baz::klass object; // ok
