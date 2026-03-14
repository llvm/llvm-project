// https://github.com/llvm/llvm-project/issues/59780
//
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -triple %itanium_abi_triple -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fprebuilt-module-path=%t \
// RUN:     -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %t/use.cpp
// RUN: %clang_cc1 -std=c++20 %t/a.pcm -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %t/a.cppm

// Test again with reduced BMI.
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -triple %itanium_abi_triple -emit-module-interface \
// RUN:     -o %t/a.full.pcm
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -triple %itanium_abi_triple -emit-reduced-module-interface \
// RUN:     -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fprebuilt-module-path=%t \
// RUN:     -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %t/use.cpp
// RUN: %clang_cc1 -std=c++20 %t/a.full.pcm -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %t/a.cppm


//--- a.cppm
export module a;

export template<typename T>
int x = 0;

export template<>
int x<int> = 0;

export template<typename T>
struct Y {
    static int value;
};

template <typename T>
int Y<T>::value = 0;

export template<>
struct Y<int> {
    static int value;
};

int Y<int>::value = 0;

// CHECK-NOT: @_ZW1a1xIiE = {{.*}}external{{.*}}global
// CHECK-NOT: @_ZNW1a1YIiE5valueE = {{.*}}external{{.*}}global

//--- use.cpp
import a;
int foo() {
    return x<int> + Y<int>::value;
}

// CHECK: @_ZW1a1xIiE = {{.*}}external{{.*}}global
// CHECK: @_ZNW1a1YIiE5valueE = {{.*}}external{{.*}}global
