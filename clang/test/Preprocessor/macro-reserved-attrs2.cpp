// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -pedantic %s

#define noreturn // expected-warning {{noreturn is a reserved attribute identifier}}
#undef noreturn  // expected-warning {{noreturn is a reserved attribute identifier}}

#define assume // expected-warning {{assume is a reserved attribute identifier}}
#undef assume  // expected-warning {{assume is a reserved attribute identifier}}

#define carries_dependency // expected-warning {{carries_dependency is a reserved attribute identifier}}
#undef carries_dependency  // expected-warning {{carries_dependency is a reserved attribute identifier}}

#define deprecated // expected-warning {{deprecated is a reserved attribute identifier}}
#undef deprecated  // expected-warning {{deprecated is a reserved attribute identifier}}

#define fallthrough // expected-warning {{fallthrough is a reserved attribute identifier}}
#undef fallthrough  // expected-warning {{fallthrough is a reserved attribute identifier}}

#define likely // expected-warning {{likely is a reserved attribute identifier}}
#undef likely  // expected-warning {{likely is a reserved attribute identifier}}

#define no_unique_address // expected-warning {{no_unique_address is a reserved attribute identifier}}
#undef no_unique_address  // expected-warning {{no_unique_address is a reserved attribute identifier}}

#define unlikely // expected-warning {{unlikely is a reserved attribute identifier}}
#undef unlikely  // expected-warning {{unlikely is a reserved attribute identifier}}

#define maybe_unused // expected-warning {{maybe_unused is a reserved attribute identifier}}
#undef maybe_unused  // expected-warning {{maybe_unused is a reserved attribute identifier}}

#define nodiscard // expected-warning {{nodiscard is a reserved attribute identifier}}
#undef nodiscard  // expected-warning {{nodiscard is a reserved attribute identifier}}

#define likely()   // ok
#define unlikely() // ok
