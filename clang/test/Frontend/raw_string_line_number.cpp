// RUN: %clang_cc1 -E %s | FileCheck %s
// CHECK: AAA
// CHECK-NEXT: BBB
R"(
AAA)"
BBB
