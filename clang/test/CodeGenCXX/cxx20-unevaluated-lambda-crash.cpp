// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 %s -emit-llvm -o - | FileCheck %s

// CHECK-LABEL: define linkonce_odr void @"_ZN10Issue579601EIiEENS_1FILNS_3$_0EEEEv"()
namespace Issue57960 {
template<auto>
class F {};

template<typename D>
F<[]{}> E() {
    return {};
}

static auto f = E<int>();
}
