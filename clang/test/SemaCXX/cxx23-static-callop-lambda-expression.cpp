// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s -fexperimental-new-constant-interpreter

namespace ns1 {
  auto lstatic = []() static { return 3; };
  int (*f2)(void) = lstatic;

}

namespace ns1_1 {

  auto lstatic = []() static consteval  //expected-error{{cannot take address of consteval call}} \
                                          expected-note {{declared here}}
  { return 3; };

  // FIXME: the above error should indicate that it was triggered below.
  int (*f2)(void) = lstatic;

}


namespace ns2 {
  auto lstatic = []() static { return 3; };
  constexpr int (*f2)(void) = lstatic;
  static_assert(lstatic() == f2());
}

namespace ns3 {
  void main() {
    static int x = 10;
    auto L = []() static { return x; };
  }
}
