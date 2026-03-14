// RUN: %clang_cc1 -fsyntax-only %s

namespace test1 {
  template <class C>
  struct C2
  {
    static int p __attribute__((visibility("hidden")));
  };
  int f() {
    return C2<int>::p;
  }
}
