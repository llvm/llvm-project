// Tests tht when we preprocess a source, the compiler won't
// require to load the named modules after import.
//
// RUN: %clang_cc1 -std=c++20 -E %s -o - | FileCheck %s
import non_exist_modules;

// CHECK: import non_exist_modules;
