// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -O2 -emit-llvm -o /dev/null %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -fclang-abi-compat=18 -emit-llvm -o - %s | FileCheck %s

struct E {
  E();
  ~E();
};

E::E() {
  struct {
    int anotherValue = [] { return 1; }();
  } obj;
}

E::~E() {
  struct {
    int anotherValue = [] { return 2; }();
  } obj;
}

// CHECK-LABEL: define{{.*}} @"_ZZN1EC1EvEN3$_0C2Ev"
// CHECK: call{{.*}} @"_ZZN1EC1EvENK3$_012anotherValueMUlvE_clEv"
// CHECK-LABEL: define{{.*}} @"_ZZN1EC1EvENK3$_012anotherValueMUlvE_clEv"

// CHECK-LABEL: define{{.*}} @"_ZZN1ED1EvEN3$_0C2Ev"
// CHECK: call{{.*}} @"_ZZN1ED1EvENK3$_012anotherValueMUlvE_clEv"
// CHECK-LABEL: define{{.*}} @"_ZZN1ED1EvENK3$_012anotherValueMUlvE_clEv"
