// RUN: %clang_cc1 -std=c++11 -verify %s

alignas(4) extern int n1; // expected-note {{previous declaration}}
alignas(8) int n1; // expected-error {{redeclaration has different alignment requirement (8 vs 4)}}

alignas(8) int n2; // expected-note {{previous declaration}}
alignas(4) extern int n2; // expected-error {{different alignment requirement (4 vs 8)}}

alignas(8) extern int n3; // expected-note {{previous declaration}}
alignas(4) extern int n3; // expected-error {{different alignment requirement (4 vs 8)}}

extern int n4;
alignas(8) extern int n4;

alignas(8) extern int n5;
extern int n5;

int n6; // expected-error {{'alignas' must be specified on definition if it is specified on any declaration}}
alignas(8) extern int n6; // expected-note {{declared with 'alignas' attribute here}}

extern int n7;
alignas(8) int n7;

alignas(8) extern int n8; // expected-note {{declared with 'alignas' attribute here}}
int n8; // expected-error {{'alignas' must be specified on definition if it is specified on any declaration}}

int n9; // expected-error {{'alignas' must be specified on definition if it is specified on any declaration}}
alignas(4) extern int n9; // expected-note {{declared with 'alignas' attribute here}}

struct S;
struct alignas(16) S; // expected-note {{declared with 'alignas' attribute here}}
struct S;
struct S { int n; }; // expected-error {{'alignas' must be specified on definition if it is specified on any declaration}}

struct alignas(2) T;
struct alignas(2) T { char c; }; // expected-note {{previous declaration is here}}
struct T;
struct alignas(4) T; // expected-error {{redeclaration has different alignment requirement (4 vs 2)}}

struct U;
struct alignas(2) U {};

struct V {}; // expected-error {{'alignas' must be specified on definition if it is specified on any declaration}}
struct alignas(1) V; // expected-note {{declared with 'alignas' attribute here}}

template<int M, int N> struct alignas(M) W;
template<int M, int N> struct alignas(N) W {};
W<4,4> w44; // ok
// FIXME: We should reject this.
W<1,2> w12;
static_assert(alignof(W<4,4>) == 4, "");

template<int M, int N, int O, int P> struct X {
  alignas(M) alignas(N) static char Buffer[32]; // expected-note {{previous declaration is here}}
};
template<int M, int N, int O, int P>
alignas(O) alignas(P) char X<M, N, O, P>::Buffer[32]; // expected-error {{redeclaration has different alignment requirement (8 vs 2)}}
char *x1848 = X<1,8,4,8>::Buffer; // ok
char *x1248 = X<1,2,4,8>::Buffer; // expected-note {{in instantiation of}}

// Don't crash here.
alignas(4) struct Incomplete incomplete; // expected-error {{incomplete type}} expected-note {{forward declaration}}
