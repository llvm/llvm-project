// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -x hlsl -fsyntax-only -verify %s

// Type and index checks require an entry point.

[shader("compute")][numthreads(8,8,1)]
// expected-error@+1 {{semantic 'SV_DispatchThreadID' must be a scalar or vector of up to 3 components of 16 or 32 bit integer type (was 'float')}}
void CSMain(float ID : SV_DispatchThreadID) {

}

struct ST {
  int a;
  float b;
};

// The second field gets index 1, which SV_DispatchThreadID rejects.
[shader("compute")][numthreads(8,8,1)]
void CSMain2(ST ID : SV_DispatchThreadID) {
// expected-error@-1 {{semantic 'SV_DispatchThreadID' does not allow indexing}}
// expected-error@-2 {{semantic 'SV_DispatchThreadID' must be a scalar or vector of up to 3 components of 16 or 32 bit integer type (was 'float')}}

}

[shader("compute")][numthreads(8,8,1)]
// expected-error@+1 {{semantic 'SV_DispatchThreadID' must be a scalar or vector of up to 3 components of 16 or 32 bit integer type (was 'uint4' (aka 'vector<uint, 4>'))}}
void CSMain3(uint4 ID : SV_DispatchThreadID) {

}

// Array elements also require distinct semantic indices.
[shader("compute")][numthreads(8,8,1)]
void CSMain4(uint3 ID[2] : SV_DispatchThreadID) {
// expected-error@-1 {{semantic 'SV_DispatchThreadID' does not allow indexing}}

}

void foo() {
// expected-warning@+1 {{'SV_DispatchThreadID' attribute only applies to parameters, non-static data members, and functions}}
  uint V : SV_DispatchThreadID;

}

struct ST2 {
// expected-warning@+1 {{'SV_DispatchThreadID' attribute only applies to parameters, non-static data members, and functions}}
    static uint X : SV_DispatchThreadID;
    uint s : SV_DispatchThreadID;
};

[shader("compute")][numthreads(8,8,1)]
// expected-error@+1 {{semantic 'SV_GroupID' must be a scalar or vector of up to 3 components of 16 or 32 bit integer type (was 'float')}}
void CSMain_GID(float ID : SV_GroupID) {
}

[shader("compute")][numthreads(8,8,1)]
void CSMain2_GID(ST GID : SV_GroupID) {
// expected-error@-1 {{semantic 'SV_GroupID' does not allow indexing}}
// expected-error@-2 {{semantic 'SV_GroupID' must be a scalar or vector of up to 3 components of 16 or 32 bit integer type (was 'float')}}

}

void foo_GID() {
// expected-warning@+1 {{'SV_GroupID' attribute only applies to parameters, non-static data members, and functions}}
  uint GIS : SV_GroupID;
}

struct ST2_GID {
// expected-warning@+1 {{'SV_GroupID' attribute only applies to parameters, non-static data members, and functions}}
    static uint GID : SV_GroupID;
    uint s_gid : SV_GroupID;
};

[shader("compute")][numthreads(8,8,1)]
// expected-error@+1 {{semantic 'SV_GroupThreadID' must be a scalar or vector of up to 3 components of 16 or 32 bit integer type (was 'float')}}
void CSMain_GThreadID(float ID : SV_GroupThreadID) {
}

[shader("compute")][numthreads(8,8,1)]
void CSMain2_GThreadID(ST GID : SV_GroupThreadID) {
// expected-error@-1 {{semantic 'SV_GroupThreadID' does not allow indexing}}
// expected-error@-2 {{semantic 'SV_GroupThreadID' must be a scalar or vector of up to 3 components of 16 or 32 bit integer type (was 'float')}}
}

void foo_GThreadID() {
// expected-warning@+1 {{'SV_GroupThreadID' attribute only applies to parameters, non-static data members, and functions}}
  uint GThreadIS : SV_GroupThreadID;
}

struct ST2_GThreadID {
// expected-warning@+1 {{'SV_GroupThreadID' attribute only applies to parameters, non-static data members, and functions}}
    static uint GThreadID : SV_GroupThreadID;
    uint s_gthreadid : SV_GroupThreadID;
};

[shader("compute")][numthreads(8,8,1)]
// expected-error@+1 {{semantic 'SV_GroupIndex' must be a scalar of 32 bit integer type (was 'uint2' (aka 'vector<uint, 2>'))}}
void CSMain_GIndex(uint2 GI : SV_GroupIndex) {
}

[shader("vertex")]
// expected-error@+4 {{semantic 'SV_GroupIndex' is not supported in vertex shader inputs}}
// expected-error@+3 {{semantic 'SV_DispatchThreadID' is not supported in vertex shader inputs}}
// expected-error@+2 {{semantic 'SV_GroupID' is not supported in vertex shader inputs}}
// expected-error@+1 {{semantic 'SV_GroupThreadID' is not supported in vertex shader inputs}}
void vs_main(int GI : SV_GroupIndex, uint ID : SV_DispatchThreadID, uint GID : SV_GroupID, uint GThreadID : SV_GroupThreadID) {}
