#include <cstdio>

#include "llvm/ADT/PointerUnion.h"

struct HasVirtual {
  virtual void func() = 0;
  virtual ~HasVirtual() = default;
};
struct DerivedWithVirtual : public HasVirtual {
  virtual void func() override;
  virtual ~DerivedWithVirtual() = default;
};

void DerivedWithVirtual::func() {}

int main() {
  int a = 5;
  float f = 4.0;
  struct alignas(8) Z {};
  Z z;

  struct Derived : public Z {};
  Derived derived;

  DerivedWithVirtual dv;

  llvm::PointerUnion<Z *, float *> z_float(&f);
  llvm::PointerUnion<Z *, float *> raw_z_float(nullptr);

  llvm::PointerUnion<long long *, int *, float *> long_int_float(&a);
  llvm::PointerUnion<Z *> z_only(&z);

  llvm::PointerIntPair<llvm::PointerUnion<Z *, float *>, 1> union_int_pair(
      z_float, 1);

  puts("Break here");

  z_float = &derived;

  puts("Break here");

  llvm::PointerUnion<HasVirtual *, float *> virtual_float(&dv);

  puts("Break here");
}
