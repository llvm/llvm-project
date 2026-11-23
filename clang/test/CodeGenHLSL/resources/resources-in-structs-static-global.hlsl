// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple spirv1.6-unknown-vulkan1.3-compute -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// We should not split global struct instances containing resources when they are
// static

// CHECK: @_ZL6cached = internal {{.*}}global %struct.A

struct A {
  RWBuffer<float> Buf;
};

RWBuffer<float> buffer;
static A cached = {buffer};

void write(A value) {
  value.Buf[0] = 1.0f;
}

[numthreads(1, 1, 1)]
void main() {
  write(cached);
}
