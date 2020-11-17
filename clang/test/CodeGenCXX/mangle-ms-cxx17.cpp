// RUN: %clang_cc1 -std=c++1z -fms-extensions -emit-llvm %s -o - -triple=i386-pc-win32 -fms-compatibility-version=19.10 | FileCheck -allow-deprecated-dag-overlap %s
// RUN: %clang_cc1 -std=c++1z -fms-extensions -emit-llvm %s -o - -triple=i386-pc-win32 -fms-compatibility-version=19.00 | FileCheck -allow-deprecated-dag-overlap %s

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

// CHECK-DAG: "?i1@@3V<lambda_1>@0@B"
inline const auto i1 = [](auto x) { return 0; };
// CHECK-DAG: "?i2@@3V<lambda_1>@0@B"
inline const auto i2 = [](auto x) { return 1; };
// CHECK-DAG: "??$?RH@<lambda_1>@i1@@QBE?A?<auto>@@H@Z"
// CHECK-DAG: "??$?RH@<lambda_1>@i2@@QBE?A?<auto>@@H@Z"
int g() {return i1(1) + i2(1); }
