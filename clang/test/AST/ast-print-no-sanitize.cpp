// RUN: %clang_cc1 -std=c++11 -ast-print %s -o - | FileCheck %s

void should_not_crash_1() __attribute__((no_sanitize_memory));
[[clang::no_sanitize_memory]] void should_not_crash_2();

// CHECK: void should_not_crash_1() __attribute__((no_sanitize("memory")));
// CHECK: void should_not_crash_2() {{\[\[}}clang::no_sanitize("memory"){{\]\]}};
