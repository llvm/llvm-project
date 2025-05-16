// RUN: %clang_cc1 -fsyntax-only -verify %s

#ifdef BE_THE_HEADER
#pragma clang system_header

template <typename T>
struct traits;

template <>
struct [[deprecated]] traits<int> {}; // expected-note {{'traits<int>' has been explicitly marked deprecated here}}

template<typename T, typename Trait = traits<T>>  // expected-warning {{'traits<int>' is deprecated}}
struct basic_string {};

// should not warn, defined and used in system headers
using __do_what_i_say_not_what_i_do  = traits<int> ;

template<typename T, typename Trait = traits<double>>
struct should_not_warn {};

#else
#define BE_THE_HEADER
#include __FILE__

basic_string<int> test1; // expected-note {{in instantiation of default argument for 'basic_string<int>' required here}}
should_not_warn<int> test2;

#endif
