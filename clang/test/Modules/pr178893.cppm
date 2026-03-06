// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple -xc++ -emit-llvm -o - %s -w | FileCheck %s

// CHECK-LABEL: define {{.*}}@_ZZN8PR178893W3mod6format5parseEPiENKUlvE_clEv
// CHECK-LABEL: define {{.*}}@_ZZN8PR178893W3mod6format5parseEPiENKUlvE0_clEv

export module mod;

namespace PR178893 {
  struct format {
      static inline int parse(int* i)
      {
          int number;
          number = [&]() -> int { return i[0]; }();

          volatile bool b = true;
          if (b) {
              auto identifier = [&]() -> int { return i[1]; }();
              return identifier;
          }

          return number;
      }
  };

  int test_format() {
      int n[2] = {1, 0};
      return format::parse(n);
  }
}
