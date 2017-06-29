// Test to make sure we don't crash, rdar://30816887.

// RUN: rm -rf %t.idx
// RUN: %clang_cc1 %s -index-store-path %t.idx -fobjc-arc -fobjc-arc-cxxlib=libstdc++
// RUN: c-index-test core -print-record %t.idx | FileCheck %s

// XFAIL: linux

// CHECK: [[@LINE+1]]:6 | function/C
void test1(void);
