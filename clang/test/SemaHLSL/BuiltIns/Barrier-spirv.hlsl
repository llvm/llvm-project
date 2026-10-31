// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   spirv-unknown-vulkan-library %s -emit-llvm-only -verify

RWBuffer<float> UAVBuffer;

void test_spirv() {
  // expected-error@+1 {{Barrier is only available for the DirectX target}}
  Barrier(UAV_MEMORY, GROUP_SYNC);

  // expected-error@+1 {{Barrier is only available for the DirectX target}}
  Barrier(UAVBuffer, DEVICE_SCOPE);
}
