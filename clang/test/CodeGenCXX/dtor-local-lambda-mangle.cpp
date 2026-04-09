// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -O2 -emit-llvm -o /dev/null %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -O2 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

struct E {
  E();
  template<typename T>
  E(T t);
  ~E();
};

E::E() {
  struct {
    // CHECK-DAG: _ZTSN1EC13$_012anotherValueMUlvE_E
    int anotherValue = [x = 1] { return x; }();
  } obj;
}

template<typename T>
E::E(T t) {
  struct {
    // CHECK-DAG: _ZTSN1EC1IiEUt_UlvE_E
    int anotherValue = [x = 1] { return x; }();
  } obj;
}

E::~E() {
  struct {
    // CHECK-DAG: _ZTSN1ED13$_012anotherValueMUlvE_E
    int anotherValue = [x = 2] { return x; }();
  } obj;
}

E e(1);
