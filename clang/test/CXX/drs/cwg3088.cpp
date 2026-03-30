// RUN: %clang_cc1 -std=c++98 -fexceptions -fcxx-exceptions -pedantic-errors %s -verify-directives -verify=expected
// RUN: %clang_cc1 -std=c++11 -fexceptions -fcxx-exceptions -pedantic-errors %s -verify-directives -verify=expected,since-cxx11
// RUN: %clang_cc1 -std=c++14 -fexceptions -fcxx-exceptions -pedantic-errors %s -verify-directives -verify=expected,since-cxx11
// RUN: %clang_cc1 -std=c++17 -fexceptions -fcxx-exceptions -pedantic-errors %s -verify-directives -verify=expected,since-cxx11
// RUN: %clang_cc1 -std=c++20 -fexceptions -fcxx-exceptions -pedantic-errors %s -verify-directives -verify=expected,since-cxx11,since-cxx20
// RUN: %clang_cc1 -std=c++23 -fexceptions -fcxx-exceptions -pedantic-errors %s -verify-directives -verify=expected,since-cxx11,since-cxx20
// RUN: %clang_cc1 -std=c++2c -fexceptions -fcxx-exceptions -pedantic-errors %s -verify-directives -verify=expected,since-cxx11,since-cxx20

namespace cwg3088 { // cwg3088: partial
// Keywords introduced in C++98

#define asm
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef asm
#define asm()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define auto
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef auto
#define auto()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define bool
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef bool
#define bool()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define break
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef break
#define break()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define case
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef case
#define case()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define catch
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef catch
#define catch()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define char
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef char
#define char()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define class
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef class
#define class()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define const
// FIXME-error@-1 {{keyword is hidden by macro definition}}
#undef const
#define const()
// FIXME-error@-1 {{keyword is hidden by macro definition}}
#define const_cast
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef const_cast
#define const_cast()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define continue
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef continue
#define continue()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define default
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef default
#define default()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define delete
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef delete
#define delete()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define do
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef do
#define do()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define double
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef double
#define double()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define dynamic_cast
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef dynamic_cast
#define dynamic_cast()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define else
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef else
#define else()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define enum
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef enum
#define enum()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define explicit
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef explicit
#define explicit()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define export
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef export
#define export()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define extern
// FIXME-error@-1 {{keyword is hidden by macro definition}}
#undef extern
#define extern()
// FIXME-error@-1 {{keyword is hidden by macro definition}}
#define false
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef false
#define false()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define float
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef float
#define float()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define for
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef for
#define for()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define friend
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef friend
#define friend()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define goto
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef goto
#define goto()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define if
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef if
#define if()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define inline
// FIXME-error@-1 {{keyword is hidden by macro definition}}
#undef inline
#define inline()
// FIXME-error@-1 {{keyword is hidden by macro definition}}
#define int
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef int
#define int()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define long
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef long
#define long()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define mutable
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef mutable
#define mutable()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define namespace
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef namespace
#define namespace()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define new
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef new
#define new()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define operator
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef operator
#define operator()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define private
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef private
#define private()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define protected
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef protected
#define protected()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define public
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef public
#define public()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define register
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef register
#define register()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define reinterpret_cast
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef reinterpret_cast
#define reinterpret_cast()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define return
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef return
#define return()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define short
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef short
#define short()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define signed
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef signed
#define signed()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define sizeof
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef sizeof
#define sizeof()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define static
// FIXME-error@-1 {{keyword is hidden by macro definition}}
#undef static
#define static()
// FIXME-error@-1 {{keyword is hidden by macro definition}}
#define static_cast
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef static_cast
#define static_cast()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define struct
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef struct
#define struct()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define switch
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef switch
#define switch()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define template
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef template
#define template()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define this
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef this
#define this()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define throw
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef throw
#define throw()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define true
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef true
#define true()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define try
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef try
#define try()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define typedef
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef typedef
#define typedef()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define typeid
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef typeid
#define typeid()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define typename
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef typename
#define typename()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define union
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef union
#define union()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define unsigned
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef unsigned
#define unsigned()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define using
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef using
#define using()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define virtual
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef virtual
#define virtual()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define void
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef void
#define void()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define volatile
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef volatile
#define volatile()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define wchar_t
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef wchar_t
#define wchar_t()
// expected-error@-1 {{keyword is hidden by macro definition}}
#define while
// expected-error@-1 {{keyword is hidden by macro definition}}
#undef while
#define while()
// expected-error@-1 {{keyword is hidden by macro definition}}

// Alternative spellings introduced in C++98

#define and
// expected-error@-1 {{C++ operator 'and' (aka '&&') used as a macro name}}
#undef and
// expected-error@-1 {{C++ operator 'and' (aka '&&') used as a macro name}}
#define and()
// expected-error@-1 {{C++ operator 'and' (aka '&&') used as a macro name}}
#define and_eq
// expected-error@-1 {{C++ operator 'and_eq' (aka '&=') used as a macro name}}
#undef and_eq
// expected-error@-1 {{C++ operator 'and_eq' (aka '&=') used as a macro name}}
#define and_eq()
// expected-error@-1 {{C++ operator 'and_eq' (aka '&=') used as a macro name}}
#define bitand
// expected-error@-1 {{C++ operator 'bitand' (aka '&') used as a macro name}}
#undef bitand
// expected-error@-1 {{C++ operator 'bitand' (aka '&') used as a macro name}}
#define bitand()
// expected-error@-1 {{C++ operator 'bitand' (aka '&') used as a macro name}}
#define bitor
// expected-error@-1 {{C++ operator 'bitor' (aka '|') used as a macro name}}
#undef bitor
// expected-error@-1 {{C++ operator 'bitor' (aka '|') used as a macro name}}
#define bitor()
// expected-error@-1 {{C++ operator 'bitor' (aka '|') used as a macro name}}
#define compl
// expected-error@-1 {{C++ operator 'compl' (aka '~') used as a macro name}}
#undef compl
// expected-error@-1 {{C++ operator 'compl' (aka '~') used as a macro name}}
#define compl()
// expected-error@-1 {{C++ operator 'compl' (aka '~') used as a macro name}}
#define not
// expected-error@-1 {{C++ operator 'not' (aka '!') used as a macro name}}
#undef not
// expected-error@-1 {{C++ operator 'not' (aka '!') used as a macro name}}
#define not()
// expected-error@-1 {{C++ operator 'not' (aka '!') used as a macro name}}
#define not_eq
// expected-error@-1 {{C++ operator 'not_eq' (aka '!=') used as a macro name}}
#undef not_eq
// expected-error@-1 {{C++ operator 'not_eq' (aka '!=') used as a macro name}}
#define not_eq()
// expected-error@-1 {{C++ operator 'not_eq' (aka '!=') used as a macro name}}
#define or
// expected-error@-1 {{C++ operator 'or' (aka '||') used as a macro name}}
#undef or
// expected-error@-1 {{C++ operator 'or' (aka '||') used as a macro name}}
#define or()
// expected-error@-1 {{C++ operator 'or' (aka '||') used as a macro name}}
#define or_eq
// expected-error@-1 {{C++ operator 'or_eq' (aka '|=') used as a macro name}}
#undef or_eq
// expected-error@-1 {{C++ operator 'or_eq' (aka '|=') used as a macro name}}
#define or_eq()
// expected-error@-1 {{C++ operator 'or_eq' (aka '|=') used as a macro name}}
#define xor
// expected-error@-1 {{C++ operator 'xor' (aka '^') used as a macro name}}
#undef xor
// expected-error@-1 {{C++ operator 'xor' (aka '^') used as a macro name}}
#define xor()
// expected-error@-1 {{C++ operator 'xor' (aka '^') used as a macro name}}
#define xor_eq
// expected-error@-1 {{C++ operator 'xor_eq' (aka '^=') used as a macro name}}
#undef xor_eq
// expected-error@-1 {{C++ operator 'xor_eq' (aka '^=') used as a macro name}}
#define xor_eq
// expected-error@-1 {{C++ operator 'xor_eq' (aka '^=') used as a macro name}}

#if __cplusplus >= 201103L
// Keywords introduced in C++11

#define alignas
// since-cxx11-error@-1 {{keyword is hidden by macro definition}}
#undef alignas
#define alignas()
// since-cxx11-error@-1 {{keyword is hidden by macro definition}}
#define alignof
// since-cxx11-error@-1 {{keyword is hidden by macro definition}}
#undef alignof
#define alignof()
// since-cxx11-error@-1 {{keyword is hidden by macro definition}}
#define chat16_t
// FIXME-error@-1 {{keyword is hidden by macro definition}}
#undef char16_t
#define char16_t()
// since-cxx11-error@-1 {{keyword is hidden by macro definition}}
#define char32_t
// since-cxx11-error@-1 {{keyword is hidden by macro definition}}
#undef char32_t
#define char32_t()
// since-cxx11-error@-1 {{keyword is hidden by macro definition}}
#define constexpr
// since-cxx11-error@-1 {{keyword is hidden by macro definition}}
#undef constexpr
#define constexpr()
// since-cxx11-error@-1 {{keyword is hidden by macro definition}}
#define decltype
// since-cxx11-error@-1 {{keyword is hidden by macro definition}}
#undef decltype
#define decltype()
// since-cxx11-error@-1 {{keyword is hidden by macro definition}}
#define noexcept
// since-cxx11-error@-1 {{keyword is hidden by macro definition}}
#undef noexcept
#define noexcept()
// since-cxx11-error@-1 {{keyword is hidden by macro definition}}
#define nullptr
// since-cxx11-error@-1 {{keyword is hidden by macro definition}}
#undef nullptr
#define nullptr()
// since-cxx11-error@-1 {{keyword is hidden by macro definition}}
#define static_assert
// since-cxx11-error@-1 {{keyword is hidden by macro definition}}
#undef static_assert
#define static_assert()
// since-cxx11-error@-1 {{keyword is hidden by macro definition}}
#define thread_local
// since-cxx11-error@-1 {{keyword is hidden by macro definition}}
#undef thread_local
#define thread_local()
// since-cxx11-error@-1 {{keyword is hidden by macro definition}}

// Identifiers with special meaning introduced in C++11

#define final
// since-cxx11-error@-1 {{keyword is hidden by macro definition}}
#undef final
#define final()
// since-cxx11-error@-1 {{keyword is hidden by macro definition}}
#define override
// since-cxx11-error@-1 {{keyword is hidden by macro definition}}
#undef override
#define override()
// since-cxx11-error@-1 {{keyword is hidden by macro definition}}

// Attributes introduced in C++11

// carries_dependency was retroactively removed by P3475R2
#define noreturn
// FIXME-since-cxx11-error@-1 {{attribute is hidden by macro definition}}
#endif

#if __cplusplus >= 201402L
// Attributes introduced in C++14

#define deprecated
// FIXME-since-cxx14-error@-1 {{attribute is hidden by macro definition}}
#endif

#if __cplusplus >= 201703L
// Attributes introduced in C++17

#define fallthrough
// FIXME-since-cxx17-error@-1 {{attribute is hidden by macro definition}}
#define maybe_unused
// FIXME-since-cxx17-error@-1 {{attribute is hidden by macro definition}}
#define nodiscard
// FIXME-since-cxx17-error@-1 {{attribute is hidden by macro definition}}
#endif

#if __cplusplus >= 202002L
// Keywords introduced in C++20

#define char8_t
// since-cxx20-error@-1 {{keyword is hidden by macro definition}}
#undef char8_t
#define char8_t()
// since-cxx20-error@-1 {{keyword is hidden by macro definition}}
#define concept
// since-cxx20-error@-1 {{keyword is hidden by macro definition}}
#undef concept
#define concept()
// since-cxx20-error@-1 {{keyword is hidden by macro definition}}
#define consteval
// since-cxx20-error@-1 {{keyword is hidden by macro definition}}
#undef consteval
#define consteval()
// since-cxx20-error@-1 {{keyword is hidden by macro definition}}
#define constinit
// since-cxx20-error@-1 {{keyword is hidden by macro definition}}
#undef constinit
#define constinit()
// since-cxx20-error@-1 {{keyword is hidden by macro definition}}
#define co_await
// since-cxx20-error@-1 {{keyword is hidden by macro definition}}
#undef co_await
#define co_await()
// since-cxx20-error@-1 {{keyword is hidden by macro definition}}
#define co_return
// since-cxx20-error@-1 {{keyword is hidden by macro definition}}
#undef co_return
#define co_return()
// since-cxx20-error@-1 {{keyword is hidden by macro definition}}
#define co_yield
// since-cxx20-error@-1 {{keyword is hidden by macro definition}}
#undef co_yield
#define co_yield
// since-cxx20-error@-1 {{keyword is hidden by macro definition}}
#define requires
// since-cxx20-error@-1 {{keyword is hidden by macro definition}}
#undef requires
#define requires()
// since-cxx20-error@-1 {{keyword is hidden by macro definition}}

// Identifiers with special meaning introduced in C++20

#define import
// FIXME-error@-1 {{keyword is hidden by macro definition}}
#undef import
#define import()
// FIXME-error@-1 {{keyword is hidden by macro definition}}
#define module
// FIXME-error@-1 {{keyword is hidden by macro definition}}
#undef module
#define module()
// FIXME-error@-1 {{keyword is hidden by macro definition}}

// Attributes introduced in C++20

#define likely
// FIXME-since-cxx20-error@-1 {{attribute is hidden by macro definition}}
#undef likely
#define likely()
#define no_unique_address
// FIXME-since-cxx20-error@-1 {{attribute is hidden by macro definition}}
#define unlikely
#undef unlikely
// FIXME-since-cxx20-error@-1 {{attribute is hidden by macro definition}}
#define unlikely()
#endif

#if __cplusplus >= 202302L
// Attributes introduced in C++23

#define assume
// FIXME-since-cxx23-error@-1 {{attribute is hidden by macro definition}}
#endif

#if __cplusplus >= 202600L
// Keywords introduced in C++26

#define contract_assert
// FIXME-error@-1 {{keyword is hidden by macro definition}}
#undef contract_assert
#define contract_assert()
// FIXME-error@-1 {{keyword is hidden by macro definition}}

// Identifiers with special meaning introduced in C++26

#define post
// FIXME-error@-1 {{keyword is hidden by macro definition}}
#undef post
#define post()
// FIXME-error@-1 {{keyword is hidden by macro definition}}
#define pre
// FIXME-error@-1 {{keyword is hidden by macro definition}}
#undef pre
#define pre()
// FIXME-error@-1 {{keyword is hidden by macro definition}}

// Attributes introduced in C++26

#define indeterminate
// FIXME-since-cxx26-error@-1 {{attribute is hidden by macro definition}}
#endif
} // namespace cwg3088
