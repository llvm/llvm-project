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

template<int I> struct Base {};
struct
S5 // #S5
:
Base<sizeof(int S5::*)>
// expected-error@-1 {{member pointer has incomplete base type 'S5'}}
{
};

template<typename T> struct S6 {};
int S6<void>::* qux;  // #qux
template<> struct S6<void> {};
// expected-error@-1 {{explicit specialization of 'S6<void>' after instantiation}}
//   expected-note@#qux {{implicit instantiation first required here}}
