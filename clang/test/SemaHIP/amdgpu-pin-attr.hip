// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx950 -x hip \
// RUN:   -fcuda-is-device -fsyntax-only -verify %s

typedef float float2 __attribute__((ext_vector_type(2)));

__attribute__((device)) void ok(void) {
  __attribute__((amdgpu_pin_agpr(0))) float2 a;   // fine
  __attribute__((amdgpu_pin_vgpr(8))) float2 b;   // fine
  constexpr int base = 16;
  __attribute__((amdgpu_pin_agpr(base + 4))) float2 c; // constant expr: fine
  (void)a; (void)b; (void)c;
}

__attribute__((device)) void bad(int n) { // expected-note {{declared here}}
  // expected-error@+1 {{'amdgpu_pin_agpr' attribute requires a non-negative integral compile time constant expression}}
  __attribute__((amdgpu_pin_agpr(-1))) float2 a;
  // expected-error@+2 {{expression is not an integral constant expression}}
  // expected-note@+1 {{function parameter 'n' with unknown value cannot be used in a constant expression}}
  __attribute__((amdgpu_pin_vgpr(n))) float2 b;
  (void)a; (void)b;
}

// The attribute only applies to variables.
// expected-warning@+1 {{'amdgpu_pin_agpr' attribute only applies to variables}}
__attribute__((device)) __attribute__((amdgpu_pin_agpr(0))) void func(void) {}
