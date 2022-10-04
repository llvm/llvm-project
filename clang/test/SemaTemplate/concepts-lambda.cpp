// RUN: %clang_cc1 -std=c++20 -verify %s
// expected-no-diagnostics

namespace GH57945 {
  template<typename T>
    concept c = true;

  template<typename>
    auto f = []() requires c<void> {
    };

  void g() {
      f<int>();
  };
}
