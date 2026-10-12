// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.8-library %s -fsyntax-only -verify
// expected-no-diagnostics

RWBuffer<float> UAVBuffer;

[shader("vertex")]
void vertex_main() {
  Barrier(ALL_MEMORY, DEVICE_SCOPE);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void compute_main() {
  Barrier(GROUP_SHARED_MEMORY, GROUP_SYNC | GROUP_SCOPE);
  Barrier(UAVBuffer, DEVICE_SCOPE);
}
