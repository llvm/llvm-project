// RUN: %clang_cc1 -fsyntax-only -Wall -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -Wall -Wno-inline-namespace-reopened-noninline -DSILENCE -verify -std=c++11 %s

namespace X {
  #ifndef SILENCE
    inline namespace {} // expected-note {{previous definition}}
    namespace {} // expected-warning {{inline namespace reopened as a non-inline namespace}}
  #else
    // expected-no-diagnostics
    inline namespace {}
    namespace {}
  #endif
}
