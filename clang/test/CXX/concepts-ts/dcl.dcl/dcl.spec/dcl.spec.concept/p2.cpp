// RUN:  %clang_cc1 -std=c++14 -fconcepts-ts -x c++ -verify %s
// REQUIRES: tls

template<typename T> concept thread_local bool VCTL = true; // expected-error {{variable concept cannot be declared 'thread_local'}}

template<typename T> concept constexpr bool VCC = true; // expected-error {{variable concept cannot be declared 'constexpr'}}

template<typename T> concept inline bool FCI() { return true; } // expected-error {{function concept cannot be declared 'inline'}}

struct X {
  template<typename T> concept friend bool FCF() { return true; } // expected-error {{function concept cannot be declared 'friend'}}
};

template<typename T> concept constexpr bool FCC() { return true; } // expected-error {{function concept cannot be declared 'constexpr'}}
