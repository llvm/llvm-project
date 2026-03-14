// RUN: %clang_cc1 -fsyntax-only -verify %s

#include <non-exist-header> // expected-error {{file not found}}

class S {};

template <typename T>
class E {
public:
  E(S* scope) {}
  S &getS();
};

class Z {
 private:
  static E<Z> e;
  static S& s();
};

E<Z> Z::e(&__UNKNOWN_ID__);

S& Z::s() { return Z::e.getS(); }
