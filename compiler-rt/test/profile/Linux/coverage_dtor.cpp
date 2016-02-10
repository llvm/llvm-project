// RUN: %clang -x c++ -fno-exceptions  -std=c++11 -fuse-ld=gold -fprofile-instr-generate -fcoverage-mapping -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: llvm-cov show %t -instr-profile %t.profdata -filename-equivalence 2>&1 | FileCheck %s
// 32 bit testing
// RUN: %clang -x c++ -fno-exceptions  -std=c++11 -fuse-ld=gold -fprofile-instr-generate -fcoverage-mapping -m32 -o %t.32 %s
// RUN: env LLVM_PROFILE_FILE=%t.32.profraw %run %t.32
// RUN: llvm-profdata merge -o %t.32.profdata %t.32.profraw
// RUN: llvm-cov show %t.32 -instr-profile %t.32.profdata -filename-equivalence 2>&1 | FileCheck %s

int g = 100;
struct Base {
  int B;
  Base(int B_) : B(B_) {}
  ~Base() { g -= B; }
};

struct Derived : public Base {
  Derived(int K) : Base(K) {}
  ~Derived() = default; // CHECK:  2| [[@LINE]]|  ~Derived() = default;
};

int main() {
  {
    Derived dd(10);
    Derived dd2(90);
  }
  if (g != 0)
    return 1;          // CHECK:  0| [[@LINE]]|    return 1;
  return 0;
}
