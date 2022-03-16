// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -verify %s

alignas(1) int n1; // expected-error {{requested alignment is less than minimum alignment of 4 for type 'int'}}
alignas(1) alignas(2) int n2; // expected-error {{less than minimum alignment}}
alignas(1) alignas(2) alignas(4) int n3; // ok
alignas(1) alignas(2) alignas(0) int n4; // expected-error {{less than minimum alignment}}
alignas(1) alignas(2) int n5 alignas(4); // ok
alignas(1) alignas(4) int n6 alignas(2); // ok
alignas(1) int n7 alignas(2), // expected-error {{less than minimum alignment}}
               n8 alignas(4); // ok
alignas(8) int n9 alignas(2); // ok, overaligned
alignas(1) extern int n10; // expected-error {{less than minimum alignment}}

typedef __attribute__((aligned(16))) int IntAlign16;
enum E6 : IntAlign16 {};
static_assert(alignof(E6) == 4, "");

struct S1 {
  alignas(8) int n;
};
struct alignas(2) S2 { // expected-error {{requested alignment is less than minimum alignment of 4 for type 'S2'}}
  int n;
};
struct alignas(2) S3 { // expected-error {{requested alignment is less than minimum alignment of 8 for type 'S3'}}
  S1 s1;
};
struct alignas(2) S4 : S1 { // expected-error {{requested alignment is less than minimum alignment of 8 for type 'S4'}}
};
struct S5 : S1 {
  alignas(2) S1 s1; // expected-error {{requested alignment is less than minimum alignment of 8 for type 'S1'}}
};
struct S6 {
  S1 s1;
};
struct S7 : S1 {
};
struct alignas(2) alignas(8) alignas(1) S8 : S1 {
};

S1 s1 alignas(4); // expected-error {{requested alignment is less than minimum alignment of 8 for type 'S1'}}
S6 s6 alignas(4); // expected-error {{requested alignment is less than minimum alignment of 8 for type 'S6'}}
S7 s7 alignas(4); // expected-error {{requested alignment is less than minimum alignment of 8 for type 'S7'}}

template<int N, int M, typename T>
struct alignas(N) X { // expected-error 3{{requested alignment is less than minimum}}
  alignas(M) T t; // expected-error 3{{requested alignment is less than minimum}}
};

template struct X<1, 1, char>;
template struct X<4, 1, char>;
template struct X<1, 2, char>; // expected-note {{instantiation}}
template struct X<1, 1, short>; // expected-note {{instantiation}}
template struct X<2, 1, short>; // expected-note {{instantiation}}
template struct X<2, 2, short>;
template struct X<16, 8, S1>;
template struct X<4, 4, S1>; // expected-note {{instantiation}}

template<int N, typename T>
void f() {
  alignas(N) T v; // expected-error {{requested alignment is less than minimum}}
};
template void f<1, char>();
template void f<2, char>();
template void f<1, short>(); // expected-note {{instantiation}}
template void f<2, short>();
