// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm %s -o - | \
// RUN: FileCheck %s
// RUN: %clang_cc1 -triple i386-apple-darwin -emit-llvm %s -o - | \
// RUN: FileCheck %s

extern "C" int printf(...);

struct S {
  S() { printf("S::S()\n"); }
  int iS;
};

struct M {
  S ARR_S; 
};

int main() {
  M m1;
}

// CHECK: call void @_ZN1SC1Ev
