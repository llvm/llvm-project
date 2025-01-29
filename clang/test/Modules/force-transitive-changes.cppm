// Test that the changes from export imported modules and touched
// modules can be popullated as expected.
//
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-reduced-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 %t/A.v1.cppm -emit-reduced-module-interface -o %t/A.v1.pcm

// The BMI of B should change it export imports A, so all the change to A should be popullated
// to B.
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -emit-reduced-module-interface -fmodule-file=A=%t/A.pcm \
// RUN:     -o %t/B.pcm
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -emit-reduced-module-interface -fmodule-file=A=%t/A.v1.pcm \
// RUN:     -o %t/B.v1.pcm
// RUN: not diff %t/B.v1.pcm %t/B.pcm  &> /dev/null

//--- A.cppm
export module A;
export int funcA() {
    return 43;
}

//--- A.v1.cppm
export module A;

export int funcA() {
    return 43;
}

//--- B.cppm
export module B;
export import A;

export int funcB() {
    return funcA();
}
