// This test verifies that we produce min-legal-vector-width attributes

// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

void __attribute((__min_vector_width__(128))) foo() {}

// CHECK: "min-legal-vector-width"="128"
