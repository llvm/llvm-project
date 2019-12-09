// RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=clc++ -O0 -emit-llvm -o - | FileCheck %s

struct Storage final {
  constexpr const float& operator[](const int index) const noexcept {
    return InternalStorage[index];
  }

  const float InternalStorage[1];
};

constexpr Storage getStorage() {
  return Storage{{1.0f}};
}

constexpr float compute() {
  constexpr auto s = getStorage();
  return 2.0f / (s[0]);
}

constexpr float FloatConstant = compute();

// CHECK-LABEL: define spir_kernel void @foo
// CHECK: store float 2.000000e+00
kernel void foo(global float *x) {
  *x = FloatConstant;
}
