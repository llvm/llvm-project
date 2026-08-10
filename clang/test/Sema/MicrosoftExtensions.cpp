// RUN: %clang_cc1 -triple i686-windows %s -fsyntax-only -Wmicrosoft -verify -fms-extensions
// RUN: %clang_cc1 -triple x86_64-windows %s -fsyntax-only -Wmicrosoft -verify -fms-extensions

// Check that __ptr32/__ptr64 can be compared.
int test_ptr_comparison(int *__ptr32 __uptr p32u, int *__ptr32 __sptr p32s,
                        int *__ptr64 p64) {
  return (p32u == p32s) +
         (p32u == p64) +
         (p32s == p64);
}

template<typename T>
void bad(T __ptr32 a) { // expected-error {{'__ptr32' attribute only applies to pointer arguments}}`
  (*a) += 1;
}

template<int size_expected, typename T>
void f(T a) {
  (*a) += sizeof(a);
  static_assert(sizeof(a) == size_expected, "instantiated template argument has unexpected size");
}
void g(int *p) {
  // instantiate for default sized pointer
  f<sizeof(void*)>(p);
}

void h(int *__ptr32 p) {
  // instantiate for 32-bit pointer
  f<4>(p);
}
