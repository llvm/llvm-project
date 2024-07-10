// Test that a change related to an ADL can be propagated correctly.
//
// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/Common.cppm -emit-reduced-module-interface -o %t/Common.pcm
//
// RUN: %clang_cc1 -std=c++20 %t/m-partA.cppm -emit-reduced-module-interface -o %t/m-partA.pcm \
// RUN:     -fmodule-file=Common=%t/Common.pcm
// RUN: %clang_cc1 -std=c++20 %t/m-partA.v1.cppm -emit-reduced-module-interface -o \
// RUN:     %t/m-partA.v1.pcm -fmodule-file=Common=%t/Common.pcm
// RUN: %clang_cc1 -std=c++20 %t/m-partB.cppm -emit-reduced-module-interface -o %t/m-partB.pcm
// RUN: %clang_cc1 -std=c++20 %t/m.cppm -emit-reduced-module-interface -o %t/m.pcm \
// RUN:     -fmodule-file=m:partA=%t/m-partA.pcm -fmodule-file=m:partB=%t/m-partB.pcm \
// RUN:     -fmodule-file=Common=%t/Common.pcm
// RUN: %clang_cc1 -std=c++20 %t/m.cppm -emit-reduced-module-interface -o %t/m.v1.pcm \
// RUN:     -fmodule-file=m:partA=%t/m-partA.v1.pcm -fmodule-file=m:partB=%t/m-partB.pcm \
// RUN:     -fmodule-file=Common=%t/Common.pcm
//
// Produce B.pcm and B.v1.pcm
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -emit-reduced-module-interface -o %t/B.pcm \
// RUN:     -fmodule-file=m=%t/m.pcm -fmodule-file=m:partA=%t/m-partA.pcm \
// RUN:     -fmodule-file=m:partB=%t/m-partB.pcm -fmodule-file=Common=%t/Common.pcm
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -emit-reduced-module-interface -o %t/B.v1.pcm \
// RUN:     -fmodule-file=m=%t/m.v1.pcm -fmodule-file=m:partA=%t/m-partA.v1.pcm \
// RUN:     -fmodule-file=m:partB=%t/m-partB.pcm -fmodule-file=Common=%t/Common.pcm
//
// Verify that both B.pcm and B.v1.pcm can work as expected.
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fsyntax-only -verify -fmodule-file=m=%t/m.pcm \
// RUN:     -fmodule-file=m:partA=%t/m-partA.pcm -fmodule-file=m:partB=%t/m-partB.pcm \
// RUN:     -fmodule-file=B=%t/B.pcm -fmodule-file=Common=%t/Common.pcm \
// RUN:     -DEXPECTED_VALUE=false
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fsyntax-only -verify -fmodule-file=m=%t/m.v1.pcm \
// RUN:     -fmodule-file=m:partA=%t/m-partA.v1.pcm -fmodule-file=m:partB=%t/m-partB.pcm \
// RUN:     -fmodule-file=B=%t/B.v1.pcm -fmodule-file=Common=%t/Common.pcm \
// RUN:     -DEXPECTED_VALUE=true
//
// Since we add new ADL function in m-partA.v1.cppm, B.v1.pcm is expected to not be the same with
// B.pcm.
// RUN: not diff %t/B.pcm %t/B.v1.pcm &> /dev/null
 
// Test that BMI won't differ if it doesn't refer adl.
// RUN: %clang_cc1 -std=c++20 %t/C.cppm -emit-reduced-module-interface -o %t/C.pcm \
// RUN:     -fmodule-file=m=%t/m.pcm -fmodule-file=m:partA=%t/m-partA.pcm \
// RUN:     -fmodule-file=m:partB=%t/m-partB.pcm -fmodule-file=Common=%t/Common.pcm
// RUN: %clang_cc1 -std=c++20 %t/C.cppm -emit-reduced-module-interface -o %t/C.v1.pcm \
// RUN:     -fmodule-file=m=%t/m.v1.pcm -fmodule-file=m:partA=%t/m-partA.v1.pcm \
// RUN:     -fmodule-file=m:partB=%t/m-partB.pcm -fmodule-file=Common=%t/Common.pcm
// RUN: diff %t/C.pcm %t/C.v1.pcm &> /dev/null
 
//--- Common.cppm
export module Common;
 
export namespace N {
    struct A {
        constexpr operator int() {
            return 43;
        }
    };
}
 
//--- m-partA.cppm
export module m:partA;
import Common;
 
export namespace N {}
 
//--- m-partA.v1.cppm
export module m:partA;
import Common;
 
export namespace N {
constexpr bool adl(A) { return true; }
}
 
//--- m-partB.cppm
export module m:partB;
 
export constexpr bool adl(int) { return false; }
 
//--- m.cppm
export module m;
export import :partA;
export import :partB;
 
//--- B.cppm
export module B;
import m;
 
export template <class C>
constexpr bool test_adl(C c) {
    return adl(c);
}
 
//--- use.cpp
// expected-no-diagnostics
import B;
import Common;
 
void test() {
    N::A a;
    static_assert(test_adl(a) == EXPECTED_VALUE);
}
 
//--- C.cppm
export module B;
import m;
 
export template <class C>
constexpr bool not_test_adl(C c) {
    return false;
}
