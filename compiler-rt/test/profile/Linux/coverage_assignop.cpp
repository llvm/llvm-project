// RUN: %clang_profgen -x c++  -std=c++11 -fuse-ld=gold -fcoverage-mapping -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: llvm-cov show %t -instr-profile %t.profdata -filename-equivalence 2>&1 | FileCheck %s
struct Base1 {
  int B;
  Base1(int b) : B(b) {}
  void operator=(const struct Base1 &b) { B = b.B + 1; }
};

struct Base2 {
  int B;
  Base2(int b) : B(b) {}
  void operator=(const struct Base2 &b) { B = b.B; }
};

struct Derived1 : public Base1 {
  Derived1() : Base1(10) {}
  Derived1(int B) : Base1(B) {}
  Derived1 &operator=(const Derived1 &) = default; // CHECK: 2| [[@LINE]]|  Derived1 &operator=(const Derived1 &) = default;
};

struct Derived2 : public Derived1, public Base2 {
  Derived2() : Derived1(20), Base2(30) {}
  Derived2(int B1, int B2) : Derived1(B1), Base2(B2) {}
  Derived2 &operator=(const Derived2 &) = default; // CHECK: 1| [[@LINE]]|  Derived2 &operator=(const Derived2 &) = default;
};

Derived1 d1(1);
Derived2 d2(2, 3);

int main() {
  Derived1 ld1;
  Derived2 ld2;
  Derived2 ld22;
  ld1 = d1;
  ld2 = d2;
  if (ld1.B != 2 || ld2.Base1::B != 3 || ld2.Base2::B != 3 ||
      ld22.Base1::B != 20 || ld22.Base2::B != 30)
    return 1;
  return 0;
}
