// https://github.com/llvm/llvm-project/issues/59765
// FIXME: Since the signature of the constructors/destructors is
// different in different targets. The current CHECK can't work
// well when targeting or running on AIX.
// It would be better to add the corresponding test for other test.
// UNSUPPORTED: system-aix
//
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/M.cppm -triple %itanium_abi_triple -emit-module-interface -o %t/M.pcm
// RUN: %clang_cc1 -std=c++20 %t/M.pcm -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %t/M.cppm
// RUN: %clang_cc1 -std=c++20 %t/Use.cpp -fprebuilt-module-path=%t -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %t/Use.cpp
//
// Check that the behavior of header units is normal as headers.
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple -xc++-user-header %t/foo.h -emit-header-unit -o %t/foo.pcm
// RUN: %clang_cc1 -std=c++20 %t/UseHU.cpp -fmodule-file=%t/foo.pcm -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %t/UseHU.cpp

//--- M.cppm
export module M;
export class A {
public:
    A();
    ~A();
    void use();
};
export A a;

// CHECK: @_ZW1M1a = {{.*}}global %class.A zeroinitializer
// CHECK: define{{.*}}void @__cxx_global_var_init()
// CHECK-NEXT: entry:
// CHECK-NEXT: call{{.*}}@_ZNW1M1AC1Ev({{.*}}@_ZW1M1a
// CHECK-NEXT: call{{.*}}@__cxa_atexit(ptr @_ZNW1M1AD1Ev, ptr @_ZW1M1a

//--- Use.cpp
import M;
void use() {
    a.use();
}

// CHECK-NOT: @_ZNW1M1AC1Ev
// CHECK-NOT: @_ZNW1M1AD1Ev

//--- foo.h
class A {
public:
    A();
    ~A();
    void use();
};
A a;

//--- UseHU.cpp
import "foo.h";
void use() {
    a.use();
}

// CHECK: @a = {{.*}}global %class.A zeroinitializer
// CHECK: define{{.*}}void @__cxx_global_var_init()
// CHECK-NEXT: entry:
// CHECK-NEXT: call{{.*}}@_ZN1AC1Ev({{.*}}@a
// CHECK-NEXT: call{{.*}}@__cxa_atexit(ptr @_ZN1AD1Ev, ptr @a
