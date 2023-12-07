// RUN: %clang_cc1 %s -emit-llvm -triple %itanium_abi_triple -o - | FileCheck %s
// pr8409

// CHECK: @_ZN1CIiE11needs_guardE = linkonce_odr {{(dso_local )?}}global
// CHECK: @_ZGVN1CIiE11needs_guardE = linkonce_odr {{(dso_local )?}}global

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

