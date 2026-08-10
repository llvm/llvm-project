// RUN: %clang_cc1 -fsyntax-only -verify -std=c++23 %s

namespace GH63627 {
template<class T>
void ok() {
  if  (using U = decltype([]{ return 42;}); true) {
      static_assert(U{}() == 42);
  }
  for (using U = decltype([]{ return 42;}); [[maybe_unused]] auto x : "abc") {
      static_assert(U{}() == 42);
  }
  for (using U = decltype([]{ return 42;}); false; ) {
      static_assert(U{}() == 42);
  }
}

template<class T>
void err() {
  if  (using U = decltype([]{}.foo); true) {}  // expected-error {{no member named 'foo'}}

  for (using U = decltype([]{}.foo);          // expected-error {{no member named 'foo'}}
       [[maybe_unused]] auto x : "abc") { }

  for (using U = decltype([]{}.foo);          // expected-error {{no member named 'foo'}}
       false ; ) { }
};

void test() {
  ok<int>();
  err<int>(); // expected-note {{in instantiation of function template specialization 'GH63627::err<int>'}}
}

}
