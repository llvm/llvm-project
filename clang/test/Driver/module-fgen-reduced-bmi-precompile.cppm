// Test that --precompile-reduced-bmi generates -emit-reduced-module-interface
// RUN: %clang -std=c++20 %s --precompile-reduced-bmi -### 2>&1 | FileCheck %s --check-prefix=CHECK-REDUCED
//
// Test that --precompile still generates -emit-module-interface
// RUN: %clang -std=c++20 %s --precompile -### 2>&1 | FileCheck %s --check-prefix=CHECK-FULL

export module Hello;

// CHECK-REDUCED: -emit-reduced-module-interface
// CHECK-REDUCED-NOT: -emit-module-interface

// CHECK-FULL: -emit-module-interface
// CHECK-FULL-NOT: -emit-reduced-module-interface
