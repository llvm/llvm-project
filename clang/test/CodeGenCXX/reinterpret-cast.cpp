// RUN: %clang_cc1 -emit-llvm -o - %s -std=c++11
// REQUIRES: LP64

void *f1(unsigned long l) {
  return reinterpret_cast<void *>(l);
}

unsigned long f2() {
  return reinterpret_cast<unsigned long>(nullptr);
}

unsigned long f3(void *p) {
  return reinterpret_cast<unsigned long>(p);
}

void f4(int*&);
void f5(void*& u) {
  f4(reinterpret_cast<int*&>(u));
}
