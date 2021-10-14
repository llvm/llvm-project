// RUN: %clang_cc1 -std=c++11 -verify %s

template<typename T, typename A, int N> struct X {
  alignas(T) alignas(A) T buffer[N];
};

static_assert(alignof(X<char, int, sizeof(int)>) == alignof(int), "");
static_assert(alignof(X<int, char, 1>) == alignof(int), "");


template<typename T, typename A, int N> struct Y {
  alignas(A) T buffer[N]; // expected-error {{requested alignment is less than minimum alignment of 4 for type 'int[1]'}}
};

static_assert(alignof(Y<char, int, sizeof(int)>) == alignof(int), "");
static_assert(alignof(Y<int, char, 1>) == alignof(int), ""); // expected-note {{in instantiation of}}

void pr16992 () {
  int x = alignof int;  // expected-error {{expected parentheses around type name in alignof expression}}
}
