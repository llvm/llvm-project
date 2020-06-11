// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_cc1 -include-pch %t -fsyntax-only -verify %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

template <class T, class U>
constexpr T BuiltinBitCastWrapper(const U &Arg) {
  return __builtin_bit_cast(T, Arg);
}

#else

int main() {
  return BuiltinBitCastWrapper<int>(0);
}

#endif
