// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// FIXME: We should probably suppress the warning on reopening an inline
// namespace without the inline keyword if it's not the first opening of the
// namespace in the file, because there's no risk of the inlineness differing
// across TUs in that case.

namespace NIL {} // expected-note {{previous definition}}
inline namespace NIL {} // expected-error {{cannot be reopened as inline}}
inline namespace IL {} // expected-note {{previous definition}}
namespace IL {} // expected-warning{{inline namespace reopened as a non-inline namespace}}

namespace {} // expected-note {{previous definition}}
inline namespace {} // expected-error {{cannot be reopened as inline}}
namespace X {
  inline namespace {} // expected-note {{previous definition}}
  namespace {} // expected-warning {{inline namespace reopened as a non-inline namespace}}
}
