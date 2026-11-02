// RUN: %clang_cc1 -std=gnu++17 -fsyntax-only -verify %s

template <typename... T>
void conditionalPack(T... Values) {
  ((void)(Values ?: 0), ...);
}

template void conditionalPack<int, int, int>(int, int, int);

template <typename... T>
void unexpandedPack(T... Values) {
  (void)(Values ?: 0); // expected-error {{expression contains unexpanded parameter pack 'Values'}}
}

template void unexpandedPack<int, int, int>(int, int, int);
