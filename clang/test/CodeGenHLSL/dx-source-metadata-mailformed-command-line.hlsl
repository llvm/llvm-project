// RUN: not %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm \
// RUN:   -debug-info-kind=constructor \
// RUN:   -fdx-record-command-line "clang_dxc \\" \
// RUN:   -o - %s 2>&1 | FileCheck %s

// RUN: not %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm \
// RUN:   -debug-info-kind=constructor \
// RUN:   -fdx-record-command-line "clang_dxc \\" \
// RUN:   -fsyntax-only \
// RUN:   -o - %s 2>&1 | FileCheck %s

// CHECK: error: invalid escaped command line: only escaped backslashes and spaces are supported

float foo(float a, float b) {
  return a + b;
}
