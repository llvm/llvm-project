// RUN: %clang_cc1 -fsyntax-only -verify %s

void f();

// Test typeof(expr) canonicalization
template<typename T>
void f0(T x, __typeof__(f(x)) y) { } // expected-note{{previous}}

template<typename T>
void f0(T x, __typeof__((f)(x)) y) { }

template<typename U>
void f0(U u, __typeof__(f(u))) { } // expected-error{{redefinition}}

// Test insane typeof(expr) overload set canonicalization
void f(int);
void f(double);

template<typename T, T N>
void f0a(T x, __typeof__(f(N)) y) { } // expected-note{{previous}}

void f(int);

template<typename T, T N>
void f0a(T x, __typeof__(f(N)) y) { } // expected-error{{redefinition}}

void f(float);

// Test dependently-sized array canonicalization
template<typename T, int N, int M>
void f1(T (&array)[N + M]) { } // expected-note{{previous}}

template<typename T, int N, int M>
void f1(T (&array)[M + N]) { }

template<typename T, int M, int N>
void f1(T (&array)[M + N]) { } // expected-error{{redefinition}}

// Test dependently-sized extended vector type canonicalization
template<typename T, int N, int M>
struct X2 {
  typedef T __attribute__((ext_vector_type(N))) type1;
  typedef T __attribute__((ext_vector_type(M))) type2;
  typedef T __attribute__((ext_vector_type(N))) type3;
  
  void f0(type1); // expected-note{{previous}}
  void f0(type2);
  void f0(type3); // expected-error{{redeclared}}
};

// Test canonicalization doesn't conflate different literal suffixes.
template<typename T> void literal_suffix(int (&)[sizeof(T() + 0)]) {}
template<typename T> void literal_suffix(int (&)[sizeof(T() + 0L)]) {}
template<typename T> void literal_suffix(int (&)[sizeof(T() + 0LL)]) {}
template<typename T> void literal_suffix(int (&)[sizeof(T() + 0.f)]) {}
template<typename T> void literal_suffix(int (&)[sizeof(T() + 0.)]) {}
template<typename T> void literal_suffix(int (&)[sizeof(T() + 0.l)]) {}
