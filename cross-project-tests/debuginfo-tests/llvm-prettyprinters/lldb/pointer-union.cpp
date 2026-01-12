#include <cstdio>

#include "llvm/ADT/PointerUnion.h"

int main() {
  int a = 5;
  float f = 4.0;
  struct alignas(8) Z {};
  Z z;

  struct Derived : public Z {};
  Derived derived;

  llvm::PointerUnion<Z *, float *> z_float(&f);
  llvm::PointerUnion<Z *, float *> raw_z_float(nullptr);

  llvm::PointerUnion<long long *, int *, float *> long_int_float(&a);
  llvm::PointerUnion<Z *> z_only(&z);

  llvm::PointerIntPair<llvm::PointerUnion<Z *, float *>, 1> union_int_pair(
      z_float, 1);

  puts("Break here");

  z_float = &derived;

  puts("Break here");
}
