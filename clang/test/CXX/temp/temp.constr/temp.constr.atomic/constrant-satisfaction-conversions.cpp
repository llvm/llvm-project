// RUN: %clang_cc1 -std=c++20 -x c++ -Wno-constant-logical-operand -verify %s

template<typename T> concept C =
sizeof(T) == 4 && !true;      // requires atomic constraints sizeof(T) == 4 and !true

template<typename T> concept C2 = sizeof(T); // expected-error{{atomic constraint must be of type 'bool' (found }}

template<typename T> struct S {
  constexpr operator bool() const { return true; }
};

// expected-error@+3{{atomic constraint must be of type 'bool' (found 'S<int>')}}
// expected-note@#FINST{{while checking constraint satisfaction}}
// expected-note@#FINST{{in instantiation of function template specialization}}
template<typename T> requires (S<T>{})
void f(T);
void f(int);

// Ensure this applies to operator && as well.
// expected-error@+3{{atomic constraint must be of type 'bool' (found 'S<int>')}}
// expected-note@#F2INST{{while checking constraint satisfaction}}
// expected-note@#F2INST{{in instantiation of function template specialization}}
template<typename T> requires (S<T>{} && true)
void f2(T);
void f2(int);

template<typename T> requires requires {
  requires S<T>{};
  // expected-error@-1{{atomic constraint must be of type 'bool' (found 'S<int>')}}
  // expected-note@-2{{while checking the satisfaction}}
  // expected-note@-3{{in instantiation of requirement}}
  // expected-note@-4{{while checking the satisfaction}}
  // expected-note@-6{{while substituting template arguments}}
  // expected-note@#F3INST{{while checking constraint satisfaction}}
  // expected-note@#F3INST{{in instantiation of function template specialization}}
  //
}
void f3(T);
void f3(int);

// Doesn't diagnose, since this is no longer a compound requirement.
template<typename T> requires (bool(1 && 2))
void f4(T);
void f4(int);

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
