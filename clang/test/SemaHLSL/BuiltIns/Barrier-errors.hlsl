// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.8-library %s -fsyntax-only -verify \
// RUN:   -verify-ignore-unexpected=note

RWBuffer<float> UAVBuffer;
Buffer<float> ReadOnlyBuffer;

void test_flags(uint Flags) {
  // expected-error@+1 {{argument to Barrier must be a constant integer}}
  Barrier(Flags, GROUP_SYNC);

  // expected-error@+1 {{argument to Barrier must be a constant integer}}
  Barrier(UAV_MEMORY, Flags);

  // expected-error@+1 {{invalid MemoryTypeFlags for Barrier operation; expected 0, ALL_MEMORY, or some combination of UAV_MEMORY, GROUP_SHARED_MEMORY, NODE_INPUT_MEMORY, NODE_OUTPUT_MEMORY flags}}
  Barrier(ALL_MEMORY | 0x10, GROUP_SYNC);

  // expected-error@+1 {{invalid SemanticFlags for Barrier operation; expected 0 or some combination of GROUP_SYNC, GROUP_SCOPE, DEVICE_SCOPE flags}}
  Barrier(UAV_MEMORY, DEVICE_SCOPE | 0x8);
}

void test_resource() {
  // expected-error@+1 {{no matching function for call to 'Barrier'}}
  Barrier(ReadOnlyBuffer, DEVICE_SCOPE);
}
