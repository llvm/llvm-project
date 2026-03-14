// Test that branch regions are generated for conditions in nested macro
// expansions.

// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name branch-macros.cpp %s | FileCheck %s

#define COND1 (a == b)
#define COND2 (a != b)
#define COND3 (COND1 && COND2)
#define COND4 (COND3 ? COND2 : COND1)
#define MACRO1 COND3
#define MACRO2 MACRO1
#define MACRO3 MACRO2


// CHECK-LABEL: _Z4funcii:
bool func(int a, int b) {
                             // CHECK: Branch,File 0, [[@LINE+15]]:12 -> [[@LINE+15]]:13 = #17, (#0 - #17)
                             // CHECK: Branch,File 0, [[@LINE+14]]:17 -> [[@LINE+14]]:18 = #18, (#17 - #18)
                             // CHECK: Branch,File 0, [[@LINE+13]]:22 -> [[@LINE+13]]:23 = #16, (#15 - #16)
                             // CHECK: Branch,File 0, [[@LINE+12]]:27 -> [[@LINE+12]]:28 = #14, (#13 - #14)
                             // CHECK: Branch,File 0, [[@LINE+11]]:32 -> [[@LINE+11]]:33 = #12, (#11 - #12)
    bool c = COND1 && COND2; // CHECK: Branch,File 1, [[@LINE-16]]:15 -> [[@LINE-16]]:23 = #1, (#0 - #1)
                             // CHECK: Branch,File 2, [[@LINE-16]]:15 -> [[@LINE-16]]:23 = #2, (#1 - #2)
    bool d = COND3;          // CHECK: Branch,File 7, [[@LINE-18]]:15 -> [[@LINE-18]]:23 = #3, (#0 - #3)
                             // CHECK: Branch,File 8, [[@LINE-18]]:15 -> [[@LINE-18]]:23 = #4, (#3 - #4)
    bool e = MACRO1;         // CHECK: Branch,File 12, [[@LINE-20]]:15 -> [[@LINE-20]]:23 = #5, (#0 - #5)
                             // CHECK: Branch,File 13, [[@LINE-20]]:15 -> [[@LINE-20]]:23 = #6, (#5 - #6)
    bool f = MACRO2;         // CHECK: Branch,File 16, [[@LINE-22]]:15 -> [[@LINE-22]]:23 = #7, (#0 - #7)
                             // CHECK: Branch,File 17, [[@LINE-22]]:15 -> [[@LINE-22]]:23 = #8, (#7 - #8)
    bool g = MACRO3;         // CHECK: Branch,File 19, [[@LINE-24]]:15 -> [[@LINE-24]]:23 = #9, (#0 - #9)
                             // CHECK: Branch,File 20, [[@LINE-24]]:15 -> [[@LINE-24]]:23 = #10, (#9 - #10)
    return c && d && e && f && g;
}

// CHECK-LABEL: _Z5func2ii:
bool func2(int a, int b) {
    bool h = MACRO3 || COND4;// CHECK: Branch,File 2, [[@LINE-28]]:15 -> [[@LINE-28]]:38 = (#1 - #2), #2
                             // CHECK: Branch,File 8, [[@LINE-32]]:15 -> [[@LINE-32]]:23 = #6, (#1 - #6)
                             // CHECK: Branch,File 9, [[@LINE-32]]:15 -> [[@LINE-32]]:23 = #7, (#6 - #7)
                             // CHECK: Branch,File 11, [[@LINE-34]]:15 -> [[@LINE-34]]:23 = #3, (#0 - #3)
                             // CHECK: Branch,File 12, [[@LINE-34]]:15 -> [[@LINE-34]]:23 = #4, (#3 - #4)
    return h;
}
