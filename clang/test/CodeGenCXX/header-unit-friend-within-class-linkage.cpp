// Tests that the friend function with-in an class definition in the header unit is still implicit inline.
// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple -xc++-user-header -emit-header-unit %t/foo.h -o %t/foo.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple -fmodule-file=%t/foo.pcm %t/user.cpp \
// RUN:   -S -emit-llvm -disable-llvm-passes -o - | FileCheck %t/user.cpp

//--- foo.h
class foo {
    int value;
public:
    foo(int v) : value(v) {}

    friend int getFooValue(foo f) {
        return f.value;
    }
};

//--- user.cpp
import "foo.h";
int use() {
    foo f(43);
    return getFooValue(f);
}

// CHECK: define{{.*}}linkonce_odr{{.*}}@_Z11getFooValue3foo
