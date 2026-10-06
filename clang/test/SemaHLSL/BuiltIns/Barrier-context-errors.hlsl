// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.8-library %s -fsyntax-only -verify \
// RUN:   -verify-ignore-unexpected=note

RWBuffer<float> UAVBuffer;

export void invalid_scope_combinations() {
  // expected-error@+1 {{GROUP_SCOPE specified for Barrier operation without applicable memory}}
  Barrier(0, GROUP_SCOPE);

  // expected-error@+1 {{DEVICE_SCOPE specified for Barrier operation without applicable memory}}
  Barrier(GROUP_SHARED_MEMORY, DEVICE_SCOPE);

  // expected-error@+1 {{DEVICE_SCOPE specified for Barrier operation without applicable memory}}
  Barrier(NODE_OUTPUT_MEMORY, DEVICE_SCOPE);

  Barrier(NODE_OUTPUT_MEMORY | NODE_INPUT_MEMORY, DEVICE_SCOPE);
  Barrier(NODE_OUTPUT_MEMORY | UAV_MEMORY, DEVICE_SCOPE);
}

void group_barriers() {
  // expected-error@+1 {{GROUP_SHARED_MEMORY specified for Barrier operation when context has no visible group}}
  Barrier(GROUP_SHARED_MEMORY, 0);

  // expected-error@+1 {{GROUP_SYNC or GROUP_SCOPE specified for Barrier operation when context has no visible group}}
  Barrier(UAV_MEMORY, GROUP_SYNC);

  // expected-error@+1 {{NODE_INPUT_MEMORY or NODE_OUTPUT_MEMORY may only be specified for Barrier operation in a node shader}}
  Barrier(NODE_INPUT_MEMORY, 0);

  // expected-error@+1 {{resource Barrier operation requires a shader stage with a visible group}}
  Barrier(UAVBuffer, DEVICE_SCOPE);
}

void invalid_scope_helper() {
  // expected-error@+1 {{DEVICE_SCOPE specified for Barrier operation without applicable memory}}
  Barrier(0, DEVICE_SCOPE);
}

[shader("vertex")]
void vertex_main() {
  group_barriers();
  invalid_scope_helper();
  Barrier(ALL_MEMORY, DEVICE_SCOPE);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void compute_main() {
  invalid_scope_helper();

  // expected-error@+1 {{GROUP_SCOPE specified for Barrier operation without applicable memory}}
  Barrier(0, GROUP_SCOPE);

  // expected-error@+1 {{DEVICE_SCOPE specified for Barrier operation without applicable memory}}
  Barrier(GROUP_SHARED_MEMORY, DEVICE_SCOPE);

  Barrier(GROUP_SHARED_MEMORY, GROUP_SYNC | GROUP_SCOPE);
  Barrier(UAVBuffer, DEVICE_SCOPE);
}
