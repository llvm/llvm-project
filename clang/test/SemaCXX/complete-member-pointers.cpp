// RUN: %clang_cc1 -verify -fsyntax-only -fc++-abi=itanium -fms-extensions -fcomplete-member-pointers %s
// RUN: %clang_cc1 -verify -fsyntax-only -triple=x86_64-unknown-win32 -fc++-abi=microsoft -fms-compatibility -fcomplete-member-pointers %s
// RUN: %clang_cc1 -verify -fsyntax-only -triple=x86_64-unknown-win32 -fc++-abi=itanium -fms-compatibility -fcomplete-member-pointers %s

struct S; // expected-note {{forward declaration of 'S'}}
typedef int S::*t;
t foo; // expected-error {{member pointer has incomplete base type 'S'}}

struct S2 {
  int S2::*foo;
};
int S2::*bar;

template <typename T>
struct S3 {
  int T::*foo;
};

struct __single_inheritance S4;
int S4::* baz;

template<int I> struct Base {};
struct __single_inheritance S5 : Base<sizeof(int S5::*)> {};
struct
S6  // #S6
:
Base<sizeof(int S6::*)>
// expected-error@-1 {{member pointer has incomplete base type 'S6'}}
//   expected-note@#S6 {{this will affect the ABI of the member pointer until the bases have been specified}}
{
};

template<typename T> struct S7 {};
int S7<void>::* qux;  // #qux
template<> struct S7<void> {};
// expected-error@-1 {{explicit specialization of 'S7<void>' after instantiation}}
//   expected-note@#qux {{implicit instantiation first required here}}
