// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-apple-darwin10 | FileCheck %s

// This file tests the clang extension which allows initializing the components
// of a complex number individually using an initialization list.  (There is a
// extensive description and test in test/Sema/complex-init-list.c.)

_Complex float x = { 1.0f, 1.0f/0.0f };
// CHECK: @x = global { float, float } { float 1.000000e+00, float 0x7FF0000000000000 }, align 4

_Complex float f(float x, float y) { _Complex float z = { x, y }; return z; }
// CHECK: define <2 x float> @f
// CHECK: alloca { float, float }
// CHECK: alloca { float, float }

_Complex float f2(float x, float y) { return (_Complex float){ x, y }; }
// CHECK: define <2 x float> @f2
// CHECK: alloca { float, float }
// CHECK: alloca { float, float }
