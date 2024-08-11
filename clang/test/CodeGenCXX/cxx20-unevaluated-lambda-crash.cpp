// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 %s -emit-llvm -o - | FileCheck %s

// FIXME: GCC mangles this as             _ZN10Issue579601EIiEENS_1FIXtlNS_UlvE_EEEEEv
// CHECK-LABEL: define linkonce_odr void @_ZN10Issue579601EIiEENS_1FILUlvE_EEEv()
namespace Issue57960 {
template<auto>
class F {};

template<typename D>
F<[]{}> E() {
    return {};
}

static auto f = E<int>();
}
