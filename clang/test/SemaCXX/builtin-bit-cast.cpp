// RUN: %clang_cc1 -verify -std=c++2a -fsyntax-only -triple x86_64-apple-macosx10.14.0 %s
// RUN: %clang_cc1 -verify -std=c++2a -fsyntax-only -triple x86_64-apple-macosx10.14.0 %s -fno-signed-char

#if !__has_builtin(__builtin_bit_cast)
#error
#endif

template <class T, T v>
T instantiate() {
  return __builtin_bit_cast(T, v);
}

int x = instantiate<int, 32>();

struct secret_ctor {
  char member;

private: secret_ctor() = default;
};

void test1() {
  secret_ctor c = __builtin_bit_cast(secret_ctor, (char)0);
}

void test2() {
  constexpr int i = 0;
  // expected-error@+1{{size of '__builtin_bit_cast' source type 'const int' does not match destination type 'char' (4 vs 1 bytes)}}
  constexpr char c = __builtin_bit_cast(char, i);
}

struct not_trivially_copyable {
  virtual void foo() {}
};

// expected-error@+1{{'__builtin_bit_cast' source type must be trivially copyable}}
constexpr unsigned long ul = __builtin_bit_cast(unsigned long, not_trivially_copyable{});

// expected-error@+1 {{'__builtin_bit_cast' destination type must be trivially copyable}}
constexpr long us = __builtin_bit_cast(unsigned long &, 0L);

namespace PR42936 {
template <class T> struct S { int m; };

extern S<int> extern_decl;

int x = __builtin_bit_cast(int, extern_decl);
S<char> y = __builtin_bit_cast(S<char>, 0);
}

template <class To, class From>
// expected-note@+1{{possible target for call}}
constexpr To bit_cast(const From &from) {
  return __builtin_bit_cast(To, bit_cast);
  // expected-error@-1{{reference to overloaded function could not be resolved; did you mean to call it?}}
} // expected-note {{control reached end of constexpr function}}
constexpr __int128_t foo = bit_cast<__int128_t>((long double)0);
// expected-error@-1{{constexpr variable 'foo' must be initialized by a constant expression}}
// expected-note@-2{{in call to 'bit_cast<__int128, long double>((long double)0)'}}

volatile int volatile_int;
int from_volatile = __builtin_bit_cast(int, volatile_int);
// expected-error@-1 {{'__builtin_bit_cast' source type 'volatile int' is volatile-qualified}}

struct volatile_class {
  int i;
};
using volatile_class_t = volatile volatile_class;
int from_volatile_prvalue = __builtin_bit_cast(int, volatile_class_t{});
// expected-error@-1 {{'__builtin_bit_cast' source type 'volatile_class_t' (aka 'volatile volatile_class') is volatile-qualified}}

volatile int volatile_array[2];
long long from_volatile_array = __builtin_bit_cast(long long, volatile_array);
// expected-error@-1 {{'__builtin_bit_cast' source type 'volatile int[2]' is volatile-qualified}}

struct volatile_bit_field {
  volatile unsigned b : 8;
};
unsigned from_volatile_bit_field(volatile_bit_field &v) {
  // expected-error@+1 {{'__builtin_bit_cast' source type 'volatile unsigned int' is volatile-qualified}}
  return __builtin_bit_cast(unsigned, v.b);
}

struct volatile_member {
  volatile int i;
};
int from_volatile_member(volatile_member &v) {
  return __builtin_bit_cast(int, v);
}
