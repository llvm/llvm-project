// RUN: %clang_cc1 -E %s | FileCheck %s
// CHECK: #pragma GCC diagnostic warning "-Wall"
#pragma GCC diagnostic warning "-Wall"
// CHECK: #pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wall"
// CHECK: #pragma GCC diagnostic error "-Wall"
#pragma GCC diagnostic error "-Wall"
// CHECK: #pragma GCC diagnostic fatal "-Wall"
#pragma GCC diagnostic fatal "-Wall"
// CHECK: #pragma GCC diagnostic push
#pragma GCC diagnostic push
// CHECK: #pragma GCC diagnostic pop
#pragma GCC diagnostic pop

// CHECK: #pragma clang diagnostic warning "-Wall"
#pragma clang diagnostic warning "-Wall"
// CHECK: #pragma clang diagnostic ignored "-Wall"
#pragma clang diagnostic ignored "-Wall"
// CHECK: #pragma clang diagnostic error "-Wall"
#pragma clang diagnostic error "-Wall"
// CHECK: #pragma clang diagnostic fatal "-Wall"
#pragma clang diagnostic fatal "-Wall"
// CHECK: #pragma clang diagnostic push
#pragma clang diagnostic push
// CHECK: #pragma clang diagnostic pop
#pragma clang diagnostic pop
