// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 %t/M.cppm \
// RUN:    -emit-module-interface -o %t/M.pcm
// RUN: %clang_cc1 -std=c++20 %t/Use.cpp -fprebuilt-module-path=%t \
// RUN:    -triple %itanium_abi_triple \
// RUN:    -emit-llvm -o - -disable-llvm-passes \
// RUN:    | FileCheck %t/Use.cpp --check-prefix=CHECK-O0
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -O1 %t/M.cppm \
// RUN:    -emit-module-interface -o %t/M.pcm
// RUN: %clang_cc1 -std=c++20 %t/Use.cpp -fprebuilt-module-path=%t -O1 \
// RUN:    -triple %itanium_abi_triple \
// RUN:    -emit-llvm -o - -disable-llvm-passes | \
// RUN:    FileCheck %t/Use.cpp --check-prefix=CHECK-O1

//--- foo.h
int func_in_gmf() {
    return 43;
}
int func_in_gmf_not_called() {
    return 44;
}

template <class T>
class A {
public:
    __attribute__((always_inline))
    inline constexpr int getValue() {
        return 43;
    }

    inline constexpr int getValue2() {
        return 43;
    }
};

extern template class A<char>;

//--- M.cppm
module;
#include "foo.h"
export module M;
int non_exported_func() {
    return 43 + func_in_gmf();
}
export int exported_func() {
    return non_exported_func();
}

int non_exported_func_not_called() {
    return 44;
}
export int func_not_called() {
    return non_exported_func_not_called();
}

export 
__attribute__((always_inline))
int always_inline_func() {
    return 45;
}

export using ::A;

//--- Use.cpp
import M;
int use() {
    A<char> a;
    return exported_func() + always_inline_func() +
           a.getValue() + a.getValue2();
}

// CHECK-O0: define{{.*}}_Z3usev(
// CHECK-O0: declare{{.*}}_ZW1M13exported_funcv(
// CHECK-O0: declare{{.*}}_ZW1M18always_inline_funcv(
// CHECK-O0: define{{.*}}@_ZN1AIcE8getValueEv(
// CHECK-O0: declare{{.*}}@_ZN1AIcE9getValue2Ev(
// CHECK-O0-NOT: func_in_gmf
// CHECK-O0-NOT: func_in_gmf_not_called
// CHECK-O0-NOT: non_exported_func
// CHECK-O0-NOT: non_exported_func_not_called
// CHECK-O0-NOT: func_not_called

// Checks that the generated code within optimizations keep the same behavior with
// O0 to keep consistent ABI.
// CHECK-O1: define{{.*}}_Z3usev(
// CHECK-O1: declare{{.*}}_ZW1M13exported_funcv(
// CHECK-O1: declare{{.*}}_ZW1M18always_inline_funcv(
// CHECK-O1: define{{.*}}@_ZN1AIcE8getValueEv(
// CHECK-O1: declare{{.*}}@_ZN1AIcE9getValue2Ev(
// CHECK-O1-NOT: func_in_gmf
// CHECK-O1-NOT: func_in_gmf_not_called
// CHECK-O1-NOT: non_exported_func
// CHECK-O1-NOT: non_exported_func_not_called
// CHECK-O1-NOT: func_not_called
