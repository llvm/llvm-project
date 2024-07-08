// RUN: %clang_cc1 -verify -fsyntax-only -fc++-abi=itanium -fms-extensions -fcomplete-member-pointers -Werror=microsoft-incomplete-member-pointer %s
// RUN: %clang_cc1 -verify -fsyntax-only -triple=x86_64-unknown-win32 -fc++-abi=microsoft -fms-extensions -Werror=microsoft-incomplete-member-pointer %s
// RUN: %clang_cc1 -verify -fsyntax-only -triple=x86_64-unknown-win32 -fc++-abi=itanium -fms-extensions -fcomplete-member-pointers -Werror=microsoft-incomplete-member-pointer %s
// RUN: %clang_cc1 -verify -fsyntax-only -triple=x86_64-unknown-linux -fc++-abi=itanium -fms-extensions -fcomplete-member-pointers -Werror=microsoft-incomplete-member-pointer %s

struct S; // expected-note {{forward declaration}} expected-note {{consider specifying the inheritance model}}
typedef int S::*t;
t foo; // expected-error {{this usage of a member pointer with an incomplete base type 'S' may cause ODR violations}}

struct S2 {
  int S2::*foo;
};
int S2::*bar;

template <typename T>
struct S3 {
  int T::*foo;
};

struct __single_inheritance S4;
int S4::*baz;

template<int I> struct Base {};
struct __single_inheritance S5 : Base<sizeof(int S5::*)> {};
// FIXME: Should be incomplete here (Fixed by #91990)
struct
S6 // #S6
:
Base<sizeof(int S6::*)>
{
};

template<int I> struct S7 {
  static_assert(false); // expected-error 0+ {{static assertion failed}}
};
// FIXME: S7<3> and S7<5> are not completed by MSVC but are completed with clang
int S7<1>::* completed1; // expected-note {{S7<1>}}
static_assert(sizeof(int S7<2>::**));
static_assert(sizeof(int S7<3>::*(*)[1])); // expected-note {{S7<3>}}
using completed4 = int S7<4>::*[]; // expected-note {{S7<4>}}
using completed5 = int S7<5>::*(*)[1]; // expected-note {{S7<5>}}
extern int S7<6>::* notcompleted6;
extern int S7<7>::* completed7[]; // expected-note {{S7<7>}}
extern int pass(...);
extern int S7<8>::* completed8;
int complete8 = pass(completed8); // expected-note {{S7<8>}}
template<typename T>
void notcompleted9(T S7<9>::*);
