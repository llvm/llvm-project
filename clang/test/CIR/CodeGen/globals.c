// There seems to be some differences in how constant expressions are evaluated
// in C vs C++. This causees the code gen for C initialized globals to be a
// bit different from the C++ version. This test ensures that these differences
// are accounted for.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o - | FileCheck %s
// XFAIL: *

char string[] = "whatnow";
// CHECK: cir.global external @string = #cir.const_array<"whatnow\00" : !cir.array<i8 x 8>> : !cir.array<i8 x 8>
