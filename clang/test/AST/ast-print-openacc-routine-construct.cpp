// RUN: %clang_cc1 -fopenacc -ast-print %s -o - | FileCheck %s

auto Lambda = [](){};
// CHECK: auto Lambda = []() {
#pragma acc routine(Lambda) worker bind(identifier)
// CHECK: #pragma acc routine(Lambda) worker bind(identifier)

int function();
#pragma acc routine (function) vector nohost bind("string")
// CHECK: #pragma acc routine(function) vector nohost bind("string")
#pragma acc routine(function) device_type(multicore) seq
// CHECK-NEXT: #pragma acc routine(function) device_type(multicore) seq
#pragma acc routine(function) dtype(radeon) seq
// CHECK-NEXT: #pragma acc routine(function) dtype(radeon) seq

#pragma acc routine nohost vector
int function2();
// CHECK: #pragma acc routine nohost vector
// CHECK-NEXT: int function2()

#pragma acc routine worker nohost bind("asdf")
auto Lambda2 = [](){};
// CHECK: #pragma acc routine worker nohost bind("asdf")
// CHECK-NEXT: auto Lambda2 = []() {
#pragma acc routine worker nohost bind("asdf")
auto Lambda3 = [](auto){};
// CHECK: #pragma acc routine worker nohost bind("asdf")
// CHECK-NEXT: auto Lambda3 = [](auto) {

namespace NS {
  int NSFunc();
auto Lambda = [](){};
}

#pragma acc routine(NS::NSFunc) seq
// CHECK: #pragma acc routine(NS::NSFunc) seq
#pragma acc routine(NS::Lambda) nohost gang

constexpr int getInt() { return 1; }

struct S {
  // CHECK: struct S {
  // despite being targetted by 'named' versions, we shouldn't print the
  // attribute here.
  // CHECK-NEXT: void MemFunc();
  void MemFunc();
#pragma acc routine gang(dim: 1)
  void MemFunc2();
// CHECK-NEXT: #pragma acc routine gang(dim: 1)
// CHECK-NEXT: void MemFunc2();
  static void StaticMemFunc();
// CHECK-NEXT: static void StaticMemFunc();
#pragma acc routine gang(dim: getInt())
  static void StaticMemFunc2();
// CHECK-NEXT: #pragma acc routine gang(dim: getInt())
// CHECK-NEXT: static void StaticMemFunc2();

  constexpr static auto Lambda = [](){};
// CHECK-NEXT: static constexpr auto Lambda = []() {
#pragma acc routine worker
  constexpr static auto Lambda2 = [](){ return 1; };
// CHECK: #pragma acc routine worker
// CHECK-NEXT: static constexpr auto Lambda2 = []() {

#pragma acc routine(S::MemFunc) gang(dim:1)
// CHECK: #pragma acc routine(S::MemFunc) gang(dim: 1)
#pragma acc routine(S::StaticMemFunc) gang(dim:getInt())
// CHECK-NEXT: #pragma acc routine(S::StaticMemFunc) gang(dim: getInt())
#pragma acc routine(S::Lambda) worker
// CHECK-NEXT: #pragma acc routine(S::Lambda) worker

#pragma acc routine(MemFunc) gang(dim:1)
// CHECK-NEXT: #pragma acc routine(MemFunc) gang(dim: 1)
#pragma acc routine(StaticMemFunc) gang(dim:getInt())
// CHECK-NEXT: #pragma acc routine(StaticMemFunc) gang(dim: getInt())
#pragma acc routine(Lambda) nohost worker
// CHECK-NEXT: #pragma acc routine(Lambda) nohost worker
};

#pragma acc routine(S::MemFunc) gang(dim:1)
// CHECK: #pragma acc routine(S::MemFunc) gang(dim: 1)
#pragma acc routine(S::StaticMemFunc) worker
// CHECK-NEXT: #pragma acc routine(S::StaticMemFunc) worker
#pragma acc routine(S::Lambda) vector
// CHECK-NEXT: #pragma acc routine(S::Lambda) vector

template<typename T>
struct DepS {
  void MemFunc();
// CHECK: void MemFunc();

  static void StaticMemFunc();
// CHECK-NEXT: static void StaticMemFunc();

#pragma acc routine gang(dim: T{1})
  static T StaticMemFunc2();
// CHECK-NEXT: #pragma acc routine gang(dim: T{1})
// CHECK-NEXT: static T StaticMemFunc2();

  constexpr static auto Lambda = [](){ return 1;};
// CHECK-NEXT: static constexpr auto Lambda = []() {

#pragma acc routine gang(dim: T{1})
  constexpr static auto Lambda2 = [](){return 1;};
// CHECK: #pragma acc routine gang(dim: T{1})
// CHECK-NEXT: static constexpr auto Lambda2 = []() {
#pragma acc routine gang(dim: T{1})
  constexpr static auto Lambda3 = [](auto){return 1;};
// CHECK: #pragma acc routine gang(dim: T{1})
// CHECK-NEXT: static constexpr auto Lambda3 = [](auto) {
#pragma acc routine gang(dim: Lambda())
  T MemFunc2();
// CHECK: #pragma acc routine gang(dim: Lambda())
// CHECK-NEXT: T MemFunc2();

#pragma acc routine(Lambda) gang(dim:Lambda())
// CHECK-NEXT: #pragma acc routine(Lambda) gang(dim: Lambda())
#pragma acc routine(MemFunc) worker
// CHECK-NEXT: #pragma acc routine(MemFunc) worker
#pragma acc routine(StaticMemFunc) seq
// CHECK-NEXT: #pragma acc routine(StaticMemFunc) seq

#pragma acc routine(DepS::Lambda) gang(dim:1)
// CHECK-NEXT: #pragma acc routine(DepS<T>::Lambda) gang(dim: 1)
#pragma acc routine(DepS::MemFunc) gang
// CHECK-NEXT: #pragma acc routine(DepS<T>::MemFunc) gang
#pragma acc routine(DepS::StaticMemFunc) worker
// CHECK-NEXT: #pragma acc routine(DepS<T>::StaticMemFunc) worker

#pragma acc routine(DepS<T>::Lambda) vector
// CHECK-NEXT: #pragma acc routine(DepS<T>::Lambda) vector
#pragma acc routine(DepS<T>::MemFunc) seq nohost
// CHECK-NEXT: #pragma acc routine(DepS<T>::MemFunc) seq nohost
#pragma acc routine(DepS<T>::StaticMemFunc) nohost worker
// CHECK-NEXT: #pragma acc routine(DepS<T>::StaticMemFunc) nohost worker

#pragma acc routine (MemFunc) worker dtype(*)
// CHECK-NEXT: #pragma acc routine(MemFunc) worker dtype(*)
#pragma acc routine (MemFunc) device_type(nvidia) vector
// CHECK-NEXT: #pragma acc routine(MemFunc) device_type(nvidia) vector
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

  auto Lambda1 = [](){};
#pragma acc routine(Lambda1) seq
// CHECK: #pragma acc routine(Lambda1) seq
#pragma acc routine seq
  auto Lambda2 = [](){};
// CHECK: #pragma acc routine seq
// CHECK-NEXT: auto Lambda2 = []() {
#pragma acc routine seq
  auto Lambda3 = [](auto){};
// CHECK: #pragma acc routine seq
// CHECK-NEXT: auto Lambda3 = [](auto) {
  Lambda3(T{});
}
