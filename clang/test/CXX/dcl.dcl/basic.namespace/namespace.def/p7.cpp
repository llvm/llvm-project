// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

namespace NIL {} // expected-note {{previous definition}}
inline namespace NIL {} // expected-error {{cannot be reopened as inline}}
inline namespace IL {} // expected-note {{previous definition}}
namespace IL {} // expected-warning{{inline namespace cannot be reopened as a non-inline namespace}}

namespace {} // expected-note {{previous definition}}
inline namespace {} // expected-error {{cannot be reopened as inline}}
namespace X {
  inline namespace {} // expected-note {{previous definition}}
  namespace {} // expected-warning {{cannot be reopened as a non-inline namespace}}
}
