// RUN: %clang -x c++ -fno-exceptions  -std=c++11 -fuse-ld=gold -fprofile-instr-generate -fcoverage-mapping -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: llvm-cov show %t -instr-profile %t.profdata -filename-equivalence 2>&1 | FileCheck %s

struct Base {
  int B;
  Base(int B_) : B(B_) {}
  ~Base() {}
};

struct Derived : public Base {
  Derived(int K) : Base(K), I(K), J(K) {}
  ~Derived() = default; // CHECK:  2| [[@LINE]]|  ~Derived
  int I;
  int J;
  int getI() { return I; }
};

int g;
int main() {
  Derived dd(10);
  Derived dd2(120);
  g = dd2.getI() + dd.getI();
  return 0;
}
