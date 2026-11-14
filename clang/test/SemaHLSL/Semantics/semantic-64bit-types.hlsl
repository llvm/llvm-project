// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -x hlsl -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -finclude-default-header -x hlsl -fsyntax-only -verify=spirv %s
// spirv-no-diagnostics

// DXIL signatures cannot contain 64-bit components. Reject these types rather
// than attempting to assign extra semantic indices to wide vectors.
struct Input {
  double3 A[2] : USER0;
  // expected-error@-1 {{semantic 'USER' cannot use 64-bit component types in DXIL shader inputs or outputs (was 'double3[2]' (aka 'vector<double, 3>[2]'))}}
  float B : USER2;
};

[shader("pixel")]
float4 overlap_repro(Input I) : SV_Target { return (float4)I.B; }

[shader("pixel")]
float4 scalar_input(double P : USER) : SV_Target { return (float4)P; }
// expected-error@-1 {{semantic 'USER' cannot use 64-bit component types in DXIL shader inputs or outputs (was 'double')}}

[shader("pixel")]
float4 vector_input(double4 P : USER) : SV_Target { return (float4)P; }
// expected-error@-1 {{semantic 'USER' cannot use 64-bit component types in DXIL shader inputs or outputs (was 'double4' (aka 'vector<double, 4>'))}}

// System-value names interpreted as arbitrary semantics still obey the general
// signature type restrictions.
[shader("vertex")]
float4 position_input(double4 P : SV_Position) : SV_Position { return (float4)P; }
// expected-error@-1 {{semantic 'SV_Position' cannot use 64-bit component types in DXIL shader inputs or outputs (was 'double4' (aka 'vector<double, 4>'))}}

[shader("vertex")]
double4 return_value(float4 P : USER) : USER { return (double4)P; }
// expected-error@-1 {{semantic 'USER' cannot use 64-bit component types in DXIL shader inputs or outputs (was 'double4' (aka 'vector<double, 4>'))}}

[shader("vertex")]
void output_array(out double3 P[2][2] : USER) {}
// expected-error@-1 {{semantic 'USER' cannot use 64-bit component types in DXIL shader inputs or outputs (was 'double3[2][2]' (aka 'vector<double, 3>[2][2]'))}}

[shader("vertex")]
void matrix_input(double2x2 P : USER) {}
// expected-error@-1 {{semantic 'USER' cannot use 64-bit component types in DXIL shader inputs or outputs (was 'double2x2' (aka 'matrix<double, 2, 2>'))}}

[shader("vertex")]
void signed_input(int64_t P : USER) {}
// expected-error@-1 {{semantic 'USER' cannot use 64-bit component types in DXIL shader inputs or outputs (was 'int64_t' (aka 'long'))}}

[shader("vertex")]
uint64_t unsigned_output() : USER { return 0; }
// expected-error@-1 {{semantic 'USER' cannot use 64-bit component types in DXIL shader inputs or outputs (was 'uint64_t' (aka 'unsigned long'))}}

// A semantic on a non-entry function does not make its type a signature type.
double4 helper(double4 P : USER) : USER { return P; }

[shader("vertex")]
float4 local_double(float4 P : USER) : SV_Position {
  double4 D = helper((double4)P);
  return (float4)D;
}
