// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.7-library %s -fhlsl-strict-availability \
// RUN:   -fsyntax-only -verify -verify-ignore-unexpected=note

RWBuffer<float> UAVBuffer;

void test_availability() {
  // expected-error@+1 {{'Barrier' is only available on Shader Model 6.8 or newer}}
  Barrier(UAV_MEMORY, GROUP_SYNC);

  // expected-error@+1 {{'Barrier<float>' is only available on Shader Model 6.8 or newer}}
  Barrier(UAVBuffer, DEVICE_SCOPE);
}
