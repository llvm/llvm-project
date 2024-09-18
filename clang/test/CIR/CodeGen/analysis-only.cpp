// Check `-fclangir-analysis-only` would generate code correctly.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir-analysis-only -std=c++20 \
// RUN:     -O2 -emit-llvm %s -o - | FileCheck %s

extern "C" void foo() {}

// CHECK: define{{.*}} @foo(

