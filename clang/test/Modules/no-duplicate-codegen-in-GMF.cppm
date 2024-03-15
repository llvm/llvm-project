// Tests that the declaration won't get emitted after being merged.
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/A.cppm -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/B.cppm -emit-module-interface -o %t/B.pcm \
// RUN:     -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/B.pcm -S -emit-llvm -o - \
// RUN:     -fprebuilt-module-path=%t | FileCheck %t/B.cppm

// Test again with reduced BMI. Note that we need to generate full BMI for B.cppm
// since it is required to generate backend codes.
// RUN: rm %t/A.pcm %t/B.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/A.cppm -emit-reduced-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/B.cppm -emit-module-interface -o %t/B.pcm \
// RUN:     -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/B.pcm -S -emit-llvm -o - \
// RUN:     -fprebuilt-module-path=%t | FileCheck %t/B.cppm


//--- foo.h

template <class T>
class foo {
public:
    T value;
    T GetValue() { return value; }
};

template class foo<int>;

//--- A.cppm
module;
#include "foo.h"
export module A;
export using ::foo;

//--- B.cppm
module;
#include "foo.h"
export module B;
import A;
export using ::foo;
export int B() {
    foo<int> f;
    return f.GetValue();
}

// CHECK-NOT: define{{.*}}@_ZN3fooIiE8GetValueEv
