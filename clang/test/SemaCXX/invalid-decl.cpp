// RUN: %clang_cc1 -fsyntax-only -verify %s

class test1 {
  template <typename> friend int bar(bool = true) {} // expected-note {{previous declaration is here}}
  template <typename> friend int bar(bool);          // expected-error {{friend declaration specifying a default argument must be the only declaration}}
};

class test2 {
  friend int bar(bool = true) {} // expected-note {{previous declaration is here}}
  friend int bar(bool);          // expected-error{{friend declaration specifying a default argument must be the only declaration}}
};
