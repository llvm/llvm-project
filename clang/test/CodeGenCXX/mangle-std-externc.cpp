// RUN: %clang_cc1 %s -DNS=std -emit-llvm -triple %itanium_abi_triple -o - | FileCheck %s --check-prefix=CHECK-STD
// RUN: %clang_cc1 %s -DNS=n -emit-llvm -triple %itanium_abi_triple -o - | FileCheck %s --check-prefix=CHECK-N

// _ZNSt1DISt1CE1iE = std::D<std::C>::i
// CHECK-STD: @_ZNSt1DISt1CE1iE = 

// _ZN1n1DINS_1CEE1iE == n::D<n::C>::i
// CHECK-N: @_ZN1n1DINS_1CEE1iE = 

namespace NS {
  extern "C" {
    class C {
    };
  }

  template <class T>
  class D {
  public:
    static int i;
  };

}


int f() {
  return NS::D<NS::C>::i;
}
