// Test to make sure we don't crash, rdar://30816887&36162712.

// RUN: mkdir -p %t/include
// RUN: echo 'module Foo { header "test.h" }' > %t/include/module.modulemap
// RUN: echo '' > %t/include/test.h

// RUN: rm -rf %t.idx %t.mcp
// RUN: %clang_cc1 %s -index-store-path %t.idx -I %t/include -fobjc-arc -fobjc-arc-cxxlib=libstdc++ -fmodules -fmodules-cache-path=%t.mcp -fimplicit-module-maps
// RUN: c-index-test core -print-record %t.idx | FileCheck %s

// RUN: rm -rf %t.idx2
// RUN: %clang_cc1 %s -index-store-path %t.idx2 -I %t/include -fobjc-arc -fobjc-arc-cxxlib=libstdc++ -fmodules -fmodules-cache-path=%t.mcp -fimplicit-module-maps
// RUN: c-index-test core -print-record %t.idx2 | FileCheck %s


// XFAIL: linux

@import Foo;

// CHECK: [[@LINE+1]]:6 | function/C
void test1(void);
