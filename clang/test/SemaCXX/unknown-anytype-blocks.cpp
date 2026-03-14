// RUN: %clang_cc1 -funknown-anytype -fblocks -fsyntax-only -verify -std=c++11 %s

namespace test1 {
  __unknown_anytype (^foo)();
  __unknown_anytype (^bar)();
  int test() {
    auto ret1 = (int)foo();
    auto ret2 = bar(); // expected-error {{'bar' has unknown return type; cast the call to its declared return type}}
    return ret1;
  }
}
