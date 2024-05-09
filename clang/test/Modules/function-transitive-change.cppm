// Test that, in C++20 modules reduced BMI, the implementation detail changes
// in non-inline function may not propagate while the inline function changes
// can get propagate.
//
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-reduced-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/a.v1.cppm -emit-reduced-module-interface -o %t/a.v1.pcm
//
// The BMI of A should differ since the different implementation.
// RUN: not diff %t/a.pcm %t/a.v1.pcm &> /dev/null
//
// The BMI of B should change since the dependent inline function changes
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -emit-reduced-module-interface -fmodule-file=a=%t/a.pcm \
// RUN:     -o %t/b.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -emit-reduced-module-interface -fmodule-file=a=%t/a.v1.pcm \
// RUN:     -o %t/b.v1.pcm
// RUN: not diff %t/b.v1.pcm %t/b.pcm  &> /dev/null
//
// Test the case with unused partitions.
// RUN: %clang_cc1 -std=c++20 %t/M-A.cppm -emit-reduced-module-interface -o %t/M-A.pcm
// RUN: %clang_cc1 -std=c++20 %t/M-B.cppm -emit-reduced-module-interface -o %t/M-B.pcm
// RUN: %clang_cc1 -std=c++20 %t/M.cppm -emit-reduced-module-interface -o %t/M.pcm \
// RUN:     -fmodule-file=M:partA=%t/M-A.pcm \
// RUN:     -fmodule-file=M:partB=%t/M-B.pcm
// RUN: %clang_cc1 -std=c++20 %t/N.cppm -emit-reduced-module-interface -o %t/N.pcm \
// RUN:     -fmodule-file=M:partA=%t/M-A.pcm \
// RUN:     -fmodule-file=M:partB=%t/M-B.pcm \
// RUN:     -fmodule-file=M=%t/M.pcm
//
// Now we change `M-A.cppm` to `M-A.v1.cppm`.
// RUN: %clang_cc1 -std=c++20 %t/M-A.v1.cppm -emit-reduced-module-interface -o %t/M-A.v1.pcm
// RUN: %clang_cc1 -std=c++20 %t/M.cppm -emit-reduced-module-interface -o %t/M.v1.pcm \
// RUN:     -fmodule-file=M:partA=%t/M-A.v1.pcm \
// RUN:     -fmodule-file=M:partB=%t/M-B.pcm
// RUN: %clang_cc1 -std=c++20 %t/N.cppm -emit-reduced-module-interface -o %t/N.v1.pcm \
// RUN:     -fmodule-file=M:partA=%t/M-A.v1.pcm \
// RUN:     -fmodule-file=M:partB=%t/M-B.pcm \
// RUN:     -fmodule-file=M=%t/M.v1.pcm
//
// The BMI of N can keep unchanged since the N didn't use the changed partition unit 'M:A'.
// RUN: diff %t/N.v1.pcm %t/N.pcm  &> /dev/null

//--- a.cppm
export module a;
export inline int a() {
    return 48;
}

//--- a.v1.cppm
export module a;
export inline int a() {
    return 50;
}

//--- b.cppm
export module b;
import a;
export inline int b() {
    return a();
}

//--- M-A.cppm
export module M:partA;
export inline int a() {
    return 43;
}

//--- M-A.v1.cppm
export module M:partA;
export inline int a() {
    return 50;
}

//--- M-B.cppm
export module M:partB;
export inline int b() {
    return 44;
}

//--- M.cppm
export module M;
export import :partA;
export import :partB;

//--- N.cppm
export module N;
import M;

export inline int n() {
    return b();
}
