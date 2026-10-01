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
    // CHECK-DAG: _ZTSZN1EC1EvEN3$_012anotherValueMUlvE_E
    int anotherValue = [x = 1] { return x; }();
  } obj;
}

template<typename T>
E::E(T t) {
  struct {
    // CHECK-DAG: _ZTSZN1EC1IiEET_ENUt_12anotherValueMUlvE_E
    int anotherValue = [x = 1] { return x; }();
  } obj;
}

E::~E() {
  struct {
    // CHECK-DAG: _ZTSZN1ED1EvEN3$_012anotherValueMUlvE_E
    int anotherValue = [x = 2] { return x; }();
  } obj;
}

E e(1);
