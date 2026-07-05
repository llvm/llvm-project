// RUN: %clang_cc1 -fsyntax-only -verify %s
// XFAIL: *

// Note: we fail this test because we perform template instantiation
// at the end of the translation unit, so argument-dependent lookup
// finds functions that occur after the point of instantiation. Note
// that GCC fails this test; EDG passes the test in strict mode, but
// not in relaxed mode.
namespace N {
  struct A { };
  struct B : public A { };

  int& f0(A&);
}

template<typename T, typename Result>
struct X0 {
  void test_f0(T t) {
    Result r = f0(t);
  };
};

void test_f0() {
  X0<N::A, int&> xA;
  xA.test_f0(N::A());
  X0<N::B, int&> xB;
  xB.test_f0(N::B());
}

namespace N {
  char& f0(B&);
}
