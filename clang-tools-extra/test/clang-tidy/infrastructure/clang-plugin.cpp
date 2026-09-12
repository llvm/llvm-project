// REQUIRES: clang-plugin
// UNSUPPORTED: system-windows
// RUN: %clang_cc1 -load %llvmshlibdir/clangTidyPlugin%pluginext -add-plugin clang-tidy -plugin-arg-clang-tidy -checks=-*,modernize-use-nullptr,bugprone-narrowing-conversions %s -std=c++11 -verify
// RUN: %clang -Xclang -load -Xclang %llvmshlibdir/clangTidyPlugin%pluginext -Xclang -add-plugin -Xclang clang-tidy -Xclang -plugin-arg-clang-tidy -Xclang -checks=-*,modernize-use-nullptr,bugprone-narrowing-conversions -Xclang -verify %s -std=c++11 -S -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -load %llvmshlibdir/clangTidyPlugin%pluginext -add-plugin clang-tidy -plugin-arg-clang-tidy -checks=-* %s -std=c++11 -verify=disabled
// disabled-no-diagnostics

extern "C" {
int *p = 0; // expected-warning {{use nullptr [modernize-use-nullptr]}}
}
// CHECK: @p = {{.*}}global ptr null

int narrow(double value) {
  return value; // expected-warning {{narrowing conversion from 'double' to 'int' [bugprone-narrowing-conversions]}}
}
