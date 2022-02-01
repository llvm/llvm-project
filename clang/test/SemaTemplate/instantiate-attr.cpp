// FIXME -Wno-aix-compat added temporarily while the diagnostic is being
// refined.

// RUN: %clang_cc1 -fsyntax-only -verify -Wno-aix-compat %s
// expected-no-diagnostics
template <typename T>
struct A {
  char a __attribute__((aligned(16)));

  struct B {
    typedef T __attribute__((aligned(16))) i16;
    i16 x;
  };
};
int a[sizeof(A<int>) == 16 ? 1 : -1];
int a2[sizeof(A<int>::B) == 16 ? 1 : -1];

// rdar://problem/8243419
namespace test1 {
  template <typename T> struct A {
    int a;
    T b[0];
  } __attribute__((packed));

  typedef A<unsigned long> type;

  int test0[sizeof(type) == 4 ? 1 : -1];
  int test1[__builtin_offsetof(type, a) == 0 ? 1 : -1];
  int test2[__builtin_offsetof(type, b) == 4 ? 1 : -1];
}

namespace test2 {
  template <class type>
  struct fastscriptmember {
      type Member __attribute__ ((packed));
      char x;
  };
  int test0[sizeof(fastscriptmember<int>) == 5 ? 1 : -1];
}
