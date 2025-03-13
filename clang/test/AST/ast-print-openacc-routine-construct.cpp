// RUN: %clang_cc1 -fopenacc -ast-print %s -o - | FileCheck %s

auto Lambda = [](){};
// CHECK: #pragma acc routine(Lambda) worker bind(identifier)
#pragma acc routine(Lambda) worker bind(identifier)
int function();
// CHECK: #pragma acc routine(function) vector nohost bind("string")
#pragma acc routine (function) vector nohost bind("string")

// CHECK: #pragma acc routine(function) device_type(Something) seq
#pragma acc routine(function) device_type(Something) seq
// CHECK: #pragma acc routine(function) dtype(Something) seq
#pragma acc routine(function) dtype(Something) seq

namespace NS {
  int NSFunc();
auto Lambda = [](){};
}
// CHECK: #pragma acc routine(NS::NSFunc) seq
#pragma acc routine(NS::NSFunc) seq
// CHECK: #pragma acc routine(NS::Lambda) nohost gang
#pragma acc routine(NS::Lambda) nohost gang

constexpr int getInt() { return 1; }

struct S {
  void MemFunc();
  static void StaticMemFunc();
  constexpr static auto Lambda = [](){};
// CHECK: #pragma acc routine(S::MemFunc) gang(dim: 1)
#pragma acc routine(S::MemFunc) gang(dim:1)
// CHECK: #pragma acc routine(S::StaticMemFunc) gang(dim: getInt())
#pragma acc routine(S::StaticMemFunc) gang(dim:getInt())
// CHECK: #pragma acc routine(S::Lambda)  worker
#pragma acc routine(S::Lambda) worker

// CHECK: #pragma acc routine(MemFunc) gang(dim: 1)
#pragma acc routine(MemFunc) gang(dim:1)
// CHECK: #pragma acc routine(StaticMemFunc) gang(dim: getInt())
#pragma acc routine(StaticMemFunc) gang(dim:getInt())
// CHECK: #pragma acc routine(Lambda) nohost worker
#pragma acc routine(Lambda) nohost worker
};

// CHECK: #pragma acc routine(S::MemFunc) gang(dim: 1)
#pragma acc routine(S::MemFunc) gang(dim:1)
// CHECK: #pragma acc routine(S::StaticMemFunc) worker
#pragma acc routine(S::StaticMemFunc) worker
// CHECK: #pragma acc routine(S::Lambda) vector
#pragma acc routine(S::Lambda) vector

template<typename T>
struct DepS {
  void MemFunc();
  static void StaticMemFunc();
  constexpr static auto Lambda = [](){ return 1;};

// CHECK: #pragma acc routine(Lambda) gang(dim: Lambda())
#pragma acc routine(Lambda) gang(dim:Lambda())
// CHECK: #pragma acc routine(MemFunc) worker
#pragma acc routine(MemFunc) worker
// CHECK: #pragma acc routine(StaticMemFunc) seq
#pragma acc routine(StaticMemFunc) seq

// CHECK: #pragma acc routine(DepS<T>::Lambda) gang(dim: 1)
#pragma acc routine(DepS::Lambda) gang(dim:1)
// CHECK: #pragma acc routine(DepS<T>::MemFunc) gang
#pragma acc routine(DepS::MemFunc) gang
// CHECK: #pragma acc routine(DepS<T>::StaticMemFunc) worker
#pragma acc routine(DepS::StaticMemFunc) worker

// CHECK: #pragma acc routine(DepS<T>::Lambda) vector
#pragma acc routine(DepS<T>::Lambda) vector
// CHECK: #pragma acc routine(DepS<T>::MemFunc) seq nohost
#pragma acc routine(DepS<T>::MemFunc) seq nohost
// CHECK: #pragma acc routine(DepS<T>::StaticMemFunc) nohost worker
#pragma acc routine(DepS<T>::StaticMemFunc) nohost worker

// CHECK: #pragma acc routine(MemFunc) worker dtype(*)
#pragma acc routine (MemFunc) worker dtype(*)
// CHECK: #pragma acc routine(MemFunc) device_type(Lambda) vector
#pragma acc routine (MemFunc) device_type(Lambda) vector
};

// CHECK: #pragma acc routine(DepS<int>::Lambda) gang bind("string")
#pragma acc routine(DepS<int>::Lambda) gang bind("string")
// CHECK: #pragma acc routine(DepS<int>::MemFunc) gang(dim: 1)
#pragma acc routine(DepS<int>::MemFunc) gang(dim:1)
// CHECK: #pragma acc routine(DepS<int>::StaticMemFunc) vector bind(identifier)
#pragma acc routine(DepS<int>::StaticMemFunc) vector bind(identifier)


template<typename T>
void TemplFunc() {
// CHECK: #pragma acc routine(T::MemFunc) gang(dim: T::SomethingElse())
#pragma acc routine(T::MemFunc) gang(dim:T::SomethingElse())
// CHECK: #pragma acc routine(T::StaticMemFunc) worker nohost bind(identifier)
#pragma acc routine(T::StaticMemFunc) worker nohost bind(identifier)
// CHECK: #pragma acc routine(T::Lambda) nohost seq bind("string")
#pragma acc routine(T::Lambda) nohost seq bind("string")
}
