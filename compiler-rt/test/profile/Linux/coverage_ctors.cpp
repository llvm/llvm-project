// RUN: %clang_profgen -x c++  -std=c++11 -fuse-ld=gold -fcoverage-mapping -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: llvm-cov show %t -instr-profile %t.profdata -filename-equivalence 2>&1 | FileCheck %s

struct Base {
  int B;
  Base() : B(2) {}
  Base(const struct Base &b2) {
    if (b2.B == 0) {
      B = b2.B + 1;
    } else
      B = b2.B;
  }
};

struct Derived : public Base {
  Derived(const Derived &) = default; // CHECK:  2| [[@LINE]]|  Derived(const Derived &) = default;
  Derived() = default;                // CHECK:  1| [[@LINE]]|  Derived() = default
  int I;
  int getI() { return I; }
};

Derived dd;
int g;
int main() {
  Derived dd2(dd);
  Derived dd3(dd);

  g = dd2.getI() + dd3.getI();
  return 0;
}
