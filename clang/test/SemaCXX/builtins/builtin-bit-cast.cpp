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
  // expected-error@+1{{__builtin_bit_cast source size does not equal destination size (4 vs 1)}}
  constexpr char c = __builtin_bit_cast(char, i);
}

struct not_trivially_copyable {
  virtual void foo() {}
};

// expected-error@+1{{__builtin_bit_cast source type must be trivially copyable}}
constexpr unsigned long ul = __builtin_bit_cast(unsigned long, not_trivially_copyable{});

// expected-error@+1 {{__builtin_bit_cast destination type must be trivially copyable}}
constexpr long us = __builtin_bit_cast(unsigned long &, 0L);

namespace PR42936 {
template <class T> struct S { int m; };

extern S<int> extern_decl;

int x = __builtin_bit_cast(int, extern_decl);
S<char> y = __builtin_bit_cast(S<char>, 0);
}
