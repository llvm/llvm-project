// Tests that the member function with-in an class definition in the header unit is still implicit inline.
// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple -xc++-user-header -emit-header-unit %t/foo.h -o %t/foo.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple -fmodule-file=%t/foo.pcm %t/user.cpp \
// RUN:   -S -emit-llvm -disable-llvm-passes -o - | FileCheck %t/user.cpp

//--- foo.h
class foo {
public:
    int getValue() {
        return 43;
    }
};

//--- user.cpp
import "foo.h";
int use() {
    foo f;
    return f.getValue();
}

// CHECK: define{{.*}}linkonce_odr{{.*}}@_ZN3foo8getValueEv
