// RUN: %clang_cc1 %s -verify -fsyntax-only
// PR4809
// This test is primarily checking that this doesn't crash, not the particular
// diagnostics.

const template basic_istream<char>; // expected-error {{expected unqualified-id}}

namespace S {}
template <class X> class Y {
  void x() { S::template y<char>(1); } // expected-error {{no member named 'y' in namespace 'S'}} \
                                       // expected-error {{unqualified-id}}
};
