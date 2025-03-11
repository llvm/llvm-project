// RUN: %clang_cc1 -fopenacc -ast-print %s -o - | FileCheck %s

auto Lambda = [](){};
// CHECK: #pragma acc routine(Lambda)
#pragma acc routine(Lambda)
int function();
// CHECK: #pragma acc routine(function)
#pragma acc routine (function)

namespace NS {
  int NSFunc();
auto Lambda = [](){};
}
// CHECK: #pragma acc routine(NS::NSFunc)
#pragma acc routine(NS::NSFunc)
// CHECK: #pragma acc routine(NS::Lambda)
#pragma acc routine(NS::Lambda)

struct S {
  void MemFunc();
  static void StaticMemFunc();
  constexpr static auto Lambda = [](){};
// CHECK: #pragma acc routine(S::MemFunc)
#pragma acc routine(S::MemFunc)
// CHECK: #pragma acc routine(S::StaticMemFunc)
#pragma acc routine(S::StaticMemFunc)
// CHECK: #pragma acc routine(S::Lambda)
#pragma acc routine(S::Lambda)

// CHECK: #pragma acc routine(MemFunc)
#pragma acc routine(MemFunc)
// CHECK: #pragma acc routine(StaticMemFunc)
#pragma acc routine(StaticMemFunc)
// CHECK: #pragma acc routine(Lambda)
#pragma acc routine(Lambda)
};

// CHECK: #pragma acc routine(S::MemFunc)
#pragma acc routine(S::MemFunc)
// CHECK: #pragma acc routine(S::StaticMemFunc)
#pragma acc routine(S::StaticMemFunc)
// CHECK: #pragma acc routine(S::Lambda)
#pragma acc routine(S::Lambda)

template<typename T>
struct DepS {
  void MemFunc();
  static void StaticMemFunc();
  constexpr static auto Lambda = [](){};

// CHECK: #pragma acc routine(Lambda)
#pragma acc routine(Lambda)
// CHECK: #pragma acc routine(MemFunc)
#pragma acc routine(MemFunc)
// CHECK: #pragma acc routine(StaticMemFunc)
#pragma acc routine(StaticMemFunc)

// CHECK: #pragma acc routine(DepS<T>::Lambda)
#pragma acc routine(DepS::Lambda)
// CHECK: #pragma acc routine(DepS<T>::MemFunc)
#pragma acc routine(DepS::MemFunc)
// CHECK: #pragma acc routine(DepS<T>::StaticMemFunc)
#pragma acc routine(DepS::StaticMemFunc)

// CHECK: #pragma acc routine(DepS<T>::Lambda)
#pragma acc routine(DepS<T>::Lambda)
// CHECK: #pragma acc routine(DepS<T>::MemFunc)
#pragma acc routine(DepS<T>::MemFunc)
// CHECK: #pragma acc routine(DepS<T>::StaticMemFunc)
#pragma acc routine(DepS<T>::StaticMemFunc)
};

// CHECK: #pragma acc routine(DepS<int>::Lambda)
#pragma acc routine(DepS<int>::Lambda)
// CHECK: #pragma acc routine(DepS<int>::MemFunc)
#pragma acc routine(DepS<int>::MemFunc)
// CHECK: #pragma acc routine(DepS<int>::StaticMemFunc)
#pragma acc routine(DepS<int>::StaticMemFunc)


template<typename T>
void TemplFunc() {
// CHECK: #pragma acc routine(T::MemFunc)
#pragma acc routine(T::MemFunc)
// CHECK: #pragma acc routine(T::StaticMemFunc)
#pragma acc routine(T::StaticMemFunc)
// CHECK: #pragma acc routine(T::Lambda)
#pragma acc routine(T::Lambda)
}
