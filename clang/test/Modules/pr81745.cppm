// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/M.cppm  -triple=x86_64-linux-gnu \
// RUN:     -emit-module-interface -o %t/M.pcm
// RUN: %clang_cc1 -std=c++20 %t/foo.cpp -fprebuilt-module-path=%t \
// RUN:      -triple=x86_64-linux-gnu  -emit-llvm -o - | FileCheck %t/foo.cpp

//--- M.cppm
export module M;
export struct S1 {
    consteval S1(int) {}
};

//--- foo.cpp
import M;
void foo() {
    struct S2 { S1 s = 0; };
    S2 s;
}

// CHECK-NOT: _ZNW1M2S1C1Ei
