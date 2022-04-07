// Example input for update_cc_test_checks
// RUN: %clang_cc1 -no-opaque-pointers -triple=x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

long test(long a, int b) {
  return a + b;
}

// A function with a mangled name
__attribute__((overloadable)) long test(long a, int b, int c) {
  return a + b + c;
}
