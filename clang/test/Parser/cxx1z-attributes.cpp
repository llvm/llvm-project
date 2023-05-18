// RUN: %clang_cc1 -fsyntax-only -verify -std=c++1z %s

namespace [[]] foo {}
namespace [[]] {}
namespace [[]] bad = foo; // expected-error {{attributes cannot be specified on namespace alias}}

namespace [[]] A::B {} // expected-error {{attributes cannot be specified on a nested namespace definition}}

enum test {
  bing [[]],
  bar [[]] = 1,
  baz [[]][[]],
  quux [[]][[]] = 4
};
