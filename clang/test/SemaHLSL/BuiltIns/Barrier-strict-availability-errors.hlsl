// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.8-compute -hlsl-entry main \
// RUN:   -fhlsl-strict-availability -fsyntax-only -verify %s

void barrier_helper() {
  // expected-error@+1 {{GROUP_SCOPE requires a nonzero MemoryTypeFlags argument}}
  Barrier(0, GROUP_SCOPE);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
  barrier_helper();
}
