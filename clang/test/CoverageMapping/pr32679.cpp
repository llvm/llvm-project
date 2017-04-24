// RUN: %clang_cc1 -cc1 -triple i686-pc-windows-msvc19.0.0 -emit-obj -fprofile-instrument=clang -std=c++14 -fdelayed-template-parsing -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name pr32679.cpp -o - %s | FileCheck %s -check-prefix=MSABI -implicit-check-not=f2
// RUN: %clang_cc1 -cc1 -triple %itanium_abi_triple -emit-obj -fprofile-instrument=clang -std=c++14 -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name pr32679.cpp -o - %s | FileCheck %s -check-prefix=ITANIUM -implicit-check-not=f2

template <typename T, int S1>
struct CreateSpecialization;

template <typename T>
struct CreateSpecialization<T, 0> {
  static constexpr T f1() { return 0; }
  static constexpr T f2() { return 0; }
};

int main() {
  CreateSpecialization<int, 0>::f1();

  // Don't emit coverage mapping info for functions in dependent contexts.
  //
  // E.g we never fully instantiate CreateSpecialization<T, 0>::f2(), so there
  // should be no mapping for it.

  return 0;
}

// MSABI: main:
// MSABI-NEXT:   File 0, [[@LINE-12]]:12 -> [[@LINE-3]]:2 = #0
// MSABI-NEXT: ?f1@?$CreateSpecialization@H$0A@@@SAHXZ:
// MSABI-NEXT:   File 0, [[@LINE-18]]:27 -> [[@LINE-18]]:40 = #0

// ITANIUM: main:
// ITANIUM-NEXT:   File 0, [[@LINE-17]]:12 -> [[@LINE-8]]:2 = #0
// ITANIUM-NEXT: _ZN20CreateSpecializationIiLi0EE2f1Ev:
// ITANIUM-NEXT:   File 0, [[@LINE-23]]:27 -> [[@LINE-23]]:40 = #0
