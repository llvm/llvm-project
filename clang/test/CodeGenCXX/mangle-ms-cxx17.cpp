// RUN: %clang_cc1 -std=c++1z -fms-extensions -emit-llvm %s -o - -triple=i386-pc-win32 -fms-compatibility-version=19.10 | FileCheck -allow-deprecated-dag-overlap %s --check-prefix=CHECK --check-prefix=MSVC2017
// RUN: %clang_cc1 -std=c++1z -fms-extensions -emit-llvm %s -o - -triple=i386-pc-win32 -fms-compatibility-version=19.00 | FileCheck -allow-deprecated-dag-overlap %s --check-prefix=CHECK --check-prefix=MSVC2015

struct S {
    int x;
    double y;
};
S f();

// CHECK-DAG: "?$S1@@3US@@B"
const auto [x0, y0] = f();
// CHECK-DAG: "?$S2@@3US@@B"
const auto [x1, y1] = f();

static union {
int a;
double b;
};

// CHECK-DAG: "?$S4@@3US@@B"
const auto [x2, y2] = f();
