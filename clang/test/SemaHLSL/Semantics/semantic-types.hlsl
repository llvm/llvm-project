// RUN: %clang_cc1 -fnative-half-type -triple dxil-pc-shadermodel6.3-library -finclude-default-header -x hlsl -fsyntax-only -verify %s

[shader("pixel")]
double4 target_double(float4 P : SV_Position) : SV_Target { return (double4)P; }
// expected-error@-1 {{semantic 'SV_Target' must be a scalar or vector of up to 4 components of 16 or 32 bit floating-point type (was 'double4' (aka 'vector<double, 4>'))}}

[shader("pixel")]
float4 position_double(double4 P : SV_Position) : SV_Target { return (float4)P; }
// expected-error@-1 {{semantic 'SV_Position' must be a scalar or vector of up to 4 components of 16 or 32 bit floating-point type (was 'double4' (aka 'vector<double, 4>'))}}

[shader("pixel")]
float4 position_half(half4 P : SV_Position) : SV_Target { return (float4)P; }

[shader("compute")][numthreads(1,1,1)]
void group_index_vector(uint2 GI : SV_GroupIndex) {}
// expected-error@-1 {{semantic 'SV_GroupIndex' must be a scalar of 32 bit integer type (was 'uint2' (aka 'vector<uint, 2>'))}}

[shader("compute")][numthreads(1,1,1)]
void group_index_16bit(half GI : SV_GroupIndex) {}
// expected-error@-1 {{semantic 'SV_GroupIndex' must be a scalar of 32 bit integer type (was 'half')}}

[shader("compute")][numthreads(1,1,1)]
void group_index_ok(uint GI : SV_GroupIndex) {}

[shader("compute")][numthreads(1,1,1)]
void thread_id_signed(int3 ID : SV_DispatchThreadID) {}

[shader("compute")][numthreads(1,1,1)]
void thread_id_scalar(uint ID : SV_GroupThreadID) {}

[shader("compute")][numthreads(1,1,1)]
void thread_id_too_wide(uint4 ID : SV_GroupID) {}
// expected-error@-1 {{semantic 'SV_GroupID' must be a scalar or vector of up to 3 components of 16 or 32 bit integer type (was 'uint4' (aka 'vector<uint, 4>'))}}

[shader("compute")][numthreads(1,1,1)]
void thread_id_float(float3 ID : SV_GroupID) {}
// expected-error@-1 {{semantic 'SV_GroupID' must be a scalar or vector of up to 3 components of 16 or 32 bit integer type (was 'float3' (aka 'vector<float, 3>'))}}

// SV_Position on a vertex input is arbitrary, so double4 is allowed.
[shader("vertex")]
float4 position_vs_in(double4 P : SV_Position) : SV_Position { return (float4)P; }

[shader("pixel")]
float4 user_double(double4 P : USER) : SV_Target { return (float4)P; }
