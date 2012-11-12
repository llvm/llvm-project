// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s
// rdar: // 8562966
// pr8409

// CHECK: @_ZN1CIiE11needs_guardE = weak_odr global
// CHECK: @_ZGVN1CIiE11needs_guardE = weak_odr global

struct K
{
  K();
  K(const K &);
  ~K();
  void PrintNumK();
};

template<typename T>
struct C
{
  void Go() { needs_guard.PrintNumK(); }
  static K needs_guard;
};

template<typename T> K C<T>::needs_guard;

void F()
{
  C<int>().Go();
}

