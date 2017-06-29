// Makes sure it doesn't crash.

// XFAIL: linux

// RUN: rm -rf %t
// RUN: %clang_cc1 %s -index-store-path %t/idx -std=c++14
// RUN: c-index-test core -print-record %t/idx | FileCheck %s

namespace crash1 {
// CHECK: [[@LINE+1]]:6 | function/C
auto getit() { return []() {}; }
}
