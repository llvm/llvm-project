// RUN: %clang_cc1 %s -fxray-instrument -std=c++11 -x c++ -emit-llvm -o - -triple arm-unknown-linux-gnu | FileCheck %s

// Make sure that the LLVM attribute for XRay-annotated functions do show up.
[[clang::xray_always_instrument]] void foo() {
// CHECK: define void @_Z3foov() #0
};

[[clang::xray_never_instrument]] void bar() {
// CHECK: define void @_Z3barv() #1
};

// CHECK: #0 = {{.*}}"function-instrument"="xray-always"
// CHECK: #1 = {{.*}}"function-instrument"="xray-never"
