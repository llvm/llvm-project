// RUN: %clang_cc1 -std=c++20 -E -verify %s | FileCheck %s

// We will emit error in phase 7, so no diagnostics in -E mode.
// expected-no-diagnostics

// CHECK: export __preprocessed_module M
export module M
