// RUN: %clang_cc1 -std=c++20 -x c++ -Wno-constant-logical-operand -verify %s

template<typename T> concept C =
sizeof(T) == 4 && !true;      // requires atomic constraints sizeof(T) == 4 and !true

template<typename T> concept C2 = sizeof(T); // expected-error{{atomic constraint must be of type 'bool' (found }}

template<typename T> struct S {
  constexpr operator bool() const { return true; }
};

// expected-error@+3{{atomic constraint must be of type 'bool' (found 'S<int>')}}
// expected-note@#FINST{{while checking constraint satisfaction}}
// expected-note@#FINST{{while substituting deduced template arguments into function template 'f' [with T = int]}}
template<typename T> requires (S<T>{})
void f(T);
void f(long);

// Ensure this applies to operator && as well.
// expected-error@+3{{atomic constraint must be of type 'bool' (found 'S<int>')}}
// expected-note@#F2INST{{while checking constraint satisfaction}}
// expected-note@#F2INST{{while substituting deduced template arguments into function template 'f2' [with T = int]}}
template<typename T> requires (S<T>{} && true)
void f2(T);
void f2(long);

template<typename T> requires requires {
  requires S<T>{};
  // expected-error@-1{{atomic constraint must be of type 'bool' (found 'S<int>')}}
  // expected-note@-2{{while checking the satisfaction}}
  // expected-note@-3{{while checking the satisfaction of nested requirement}}
  // expected-note@-5{{while substituting template arguments}}
  // expected-note@#F3INST{{while checking constraint satisfaction}}
  // expected-note@#F3INST{{while substituting deduced template arguments into function template 'f3' [with T = int]}}
  //
}
void f3(T);
void f3(long);

// Doesn't diagnose, since this is no longer a compound requirement.
template<typename T> requires (bool(1 && 2))
void f4(T);
void f4(long);

void g() {
  f(0); // #FINST
  f2(0); // #F2INST
  f3(0); // #F3INST
  f4(0);
}

template<typename T>
auto Nullptr = nullptr;

template<typename T> concept NullTy = Nullptr<T>;
// expected-error@-1{{atomic constraint must be of type 'bool' (found }}
// expected-note@+1{{while checking the satisfaction}}
static_assert(NullTy<int>);

template<typename T>
auto Struct = S<T>{};

template<typename T> concept StructTy = Struct<T>;
// expected-error@-1{{atomic constraint must be of type 'bool' (found 'S<int>')}}
// expected-note@+1{{while checking the satisfaction}}
static_assert(StructTy<int>);
