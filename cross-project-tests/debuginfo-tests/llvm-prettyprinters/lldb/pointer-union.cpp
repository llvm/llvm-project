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

struct alignas(8) Z {};
struct Derived : public Z {};

// Types for variable-width tag encoding test.
// 3 x alignof(4) + 2 x alignof(8) requires escape-coded tags because
// ceil(log2(5)) = 3 > min(NumLowBitsAvailable) = 2.
template <int I> struct alignas(4) Align4 {};
template <int I> struct alignas(8) Align8 {};

int main() {
  int a = 5;
  float f = 4.0;
  Z z;
  Derived derived;

  DerivedWithVirtual dv;

  llvm::PointerUnion<Z *, float *> z_float(&f);
  llvm::PointerUnion<Z *, float *> raw_z_float(nullptr);
  llvm::PointerUnion<Z *, float *> null_float(static_cast<float *>(nullptr));

  llvm::PointerUnion<long long *, int *, float *> long_int_float(&a);
  llvm::PointerUnion<Z *> z_only(&z);

  llvm::PointerIntPair<llvm::PointerUnion<Z *, float *>, 1> union_int_pair(
      z_float, 1);

  puts("Break here");

  z_float = &derived;

  puts("Break here");

  llvm::PointerUnion<HasVirtual *, float *> virtual_float(&dv);

  puts("Break here");

  // Function-local types stress template_argument lookup in debuggers.
  struct alignas(8) Local {};
  Local local;
  llvm::PointerUnion<Local *, float *> local_float(&local);

  puts("Break here");

  // Variable-width tag encoding: formatter should fall back to void*.
  Align4<0> a4_0;
  Align8<0> a8_0;
  llvm::PointerUnion<Align4<0> *, Align4<1> *, Align4<2> *, Align8<0> *,
                     Align8<1> *>
      varwidth(&a4_0);
  llvm::PointerUnion<Align4<0> *, Align4<1> *, Align4<2> *, Align8<0> *,
                     Align8<1> *>
      varwidth_tier1(&a8_0);

  puts("Break here");
}
