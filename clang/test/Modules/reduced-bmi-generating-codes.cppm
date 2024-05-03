// Although the reduced BMI are not designed to be generated,
// it is helpful for testing whether we've reduced the definitions.
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/a.cppm \
// RUN:     -emit-reduced-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/b.cpp \
// RUN:     -fmodule-file=a=%t/a.pcm -S -emit-llvm -o - \
// RUN:     | FileCheck %t/b.cpp

//--- a.cppm
export module a;

export template <class T>
class A {
public:
    int member() {
        return 43;
    }
};

// Instantiate `A<int>::member()`.
export int a_member = A<int>().member();

export const int a = 43;

//--- b.cpp
import a;

static_assert(a == 43);

int b() {
    A<int> a;
    return a.member();
}

// CHECK: define{{.*}}@_ZNW1a1AIiE6memberEv
