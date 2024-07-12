// RUN: %clang_cc1 -Wno-unused -fsyntax-only %s -verify

struct A {
  void operator*();
  void operator+();
  void operator-();
  void operator!();
  void operator~();
  void operator&();
  void operator++();
  void operator--();
};

struct B { };

template<typename T, typename U>
void dependent(T t, T* pt, T U::* mpt, T(&ft)(), T(&at)[4]) {
  *t;
  +t;
  -t;
  !t;
  ~t;
  &t;
  ++t;
  --t;

  *pt;
  +pt;
  -pt; // expected-error {{invalid argument type 'T *' to unary expression}}
  !pt;
  ~pt; // expected-error {{invalid argument type 'T *' to unary expression}}
  &pt;
  ++pt;
  --pt;

  *mpt; // expected-error {{indirection requires pointer operand ('T U::*' invalid)}}
  +mpt; // expected-error {{invalid argument type 'T U::*' to unary expression}}
  -mpt; // expected-error {{invalid argument type 'T U::*' to unary expression}}
  !mpt;
  ~mpt; // expected-error {{invalid argument type 'T U::*' to unary expression}}
  &mpt;
  ++mpt; // expected-error {{cannot increment value of type 'T U::*'}}
  --mpt; // expected-error {{cannot decrement value of type 'T U::*'}}

  *ft;
  +ft;
  -ft; // expected-error {{invalid argument type 'T (*)()' to unary expression}}
  !ft;
  ~ft; // expected-error {{invalid argument type 'T (*)()' to unary expression}}
  &ft;
  ++ft; // expected-error {{cannot increment value of type 'T ()'}}
  --ft; // expected-error {{cannot decrement value of type 'T ()'}}

  *at;
  +at;
  -at; // expected-error {{invalid argument type 'T *' to unary expression}}
  !at;
  ~at; // expected-error {{invalid argument type 'T *' to unary expression}}
  &at;
  ++at; // expected-error {{cannot increment value of type 'T[4]'}}
  --at; // expected-error {{cannot decrement value of type 'T[4]'}}
}

// Make sure we only emit diagnostics once.
template void dependent(A t, A* pt, A B::* mpt, A(&ft)(), A(&at)[4]);
