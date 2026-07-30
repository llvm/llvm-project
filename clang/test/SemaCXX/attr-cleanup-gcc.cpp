// RUN: %clang_cc1 %s -verify -fsyntax-only -Wgcc-compat

namespace N {
  void c1(int *a) {}
}

void c2(int *a) {}

template <typename Ty>
void c3(Ty *a) {}

void t3() {
  int v1 __attribute__((cleanup(N::c1)));  // expected-warning  {{GCC does not allow the 'cleanup' attribute argument to be anything other than a simple identifier}}
  int v2 __attribute__((cleanup(c2)));
  int v3 __attribute__((cleanup(c3<int>)));  // expected-warning  {{GCC does not allow the 'cleanup' attribute argument to be anything other than a simple identifier}}
}
