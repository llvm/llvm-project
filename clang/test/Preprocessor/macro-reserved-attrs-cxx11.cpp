// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++11 -Wreserved-attribute-identifier -fsyntax-only -verify %s -DTEST1
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++11 -Wreserved-attribute-identifier -fsyntax-only -verify %s -DTEST2
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++11 -Wreserved-attribute-identifier -fsyntax-only -verify %s -DTEST3
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++11 -Wreserved-attribute-identifier -fsyntax-only -verify %s -DTEST4

#ifdef TEST1

#define assume
#undef assume

#define noreturn // expected-warning {{noreturn is a reserved attribute identifier}}
#undef noreturn  // expected-warning {{noreturn is a reserved attribute identifier}}

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

#elif TEST2

#define assume "test"
#undef assume

#define noreturn "test" // expected-warning {{noreturn is a reserved attribute identifier}}
#undef noreturn         // expected-warning {{noreturn is a reserved attribute identifier}}

#define carries_dependency "test" // expected-warning {{carries_dependency is a reserved attribute identifier}}
#undef carries_dependency         // expected-warning {{carries_dependency is a reserved attribute identifier}}

#define deprecated "test" // expected-warning {{deprecated is a reserved attribute identifier}}
#undef deprecated         // expected-warning {{deprecated is a reserved attribute identifier}}

#define fallthrough "test" // expected-warning {{fallthrough is a reserved attribute identifier}}
#undef fallthrough         // expected-warning {{fallthrough is a reserved attribute identifier}}

#define likely "test" // expected-warning {{likely is a reserved attribute identifier}}
#undef likely         // expected-warning {{likely is a reserved attribute identifier}}

#define no_unique_address "test" // expected-warning {{no_unique_address is a reserved attribute identifier}}
#undef no_unique_address         // expected-warning {{no_unique_address is a reserved attribute identifier}}

#define unlikely "test" // expected-warning {{unlikely is a reserved attribute identifier}}
#undef unlikely         // expected-warning {{unlikely is a reserved attribute identifier}}

#define maybe_unused "test" // expected-warning {{maybe_unused is a reserved attribute identifier}}
#undef maybe_unused         // expected-warning {{maybe_unused is a reserved attribute identifier}}

#define nodiscard "test" // expected-warning {{nodiscard is a reserved attribute identifier}}
#undef nodiscard         // expected-warning {{nodiscard is a reserved attribute identifier}}

#elif TEST3

#define assume() "test"     // expected-warning {{assume is a reserved attribute identifier}}
#define deprecated() "test" // expected-warning {{deprecated is a reserved attribute identifier}}
#define nodiscard() "test"  // expected-warning {{nodiscard is a reserved attribute identifier}}
#define noreturn() "test"
#define carries_dependency() "test"
#define fallthrough() "test"
#define likely() "test"
#define no_unique_address() "test"
#define unlikely() "test"
#define maybe_unused() "test"

#elif TEST4

#define assume()     // expected-warning {{assume is a reserved attribute identifier}}
#define deprecated() // expected-warning {{deprecated is a reserved attribute identifier}}
#define nodiscard()  // expected-warning {{nodiscard is a reserved attribute identifier}}
#define noreturn()
#define carries_dependency()
#define fallthrough()
#define likely()
#define no_unique_address()
#define unlikely()
#define maybe_unused()

#else

#error Unknown test

#endif
