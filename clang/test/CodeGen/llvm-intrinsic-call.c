// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

float llvm_sin_f32(float) asm("llvm.sin.f32");

float test(float a) {
  return llvm_sin_f32(a);
}

// CHECK: call float @llvm.sin.f32(float {{%.+}}){{$}}

// CHECK: declare float @llvm.sin.f32(float) [[ATTRS:#[0-9]+]]

// CHECK: attributes [[ATTRS]] = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
