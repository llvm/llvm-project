// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

void fn() {
  (void)__builtin_hlsl_create_handle(0);
}

// CHECK: call ptr @llvm.dx.create.handle(i8 0)
