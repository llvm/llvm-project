// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -x hlsl -ast-dump -verify -o - %s

[numthreads(8,8,1)]
// expected-error@+1 {{attribute 'SV_DispatchThreadID' only applies to a field or parameter of type 'uint/uint2/uint3'}}
void CSMain(float ID : SV_DispatchThreadID) {

}

struct ST {
  int a;
  float b;
};
[numthreads(8,8,1)]
// expected-error@+1 {{attribute 'SV_DispatchThreadID' only applies to a field or parameter of type 'uint/uint2/uint3'}}
void CSMain2(ST ID : SV_DispatchThreadID) {

}

void foo() {
// expected-warning@+1 {{'SV_DispatchThreadID' attribute only applies to parameters and non-static data members}}
  uint V : SV_DispatchThreadID;

}

struct ST2 {
// expected-warning@+1 {{'SV_DispatchThreadID' attribute only applies to parameters and non-static data members}}
    static uint X : SV_DispatchThreadID;
    uint s : SV_DispatchThreadID;
};

[numthreads(8,8,1)]
// expected-error@+1 {{attribute 'SV_GroupID' only applies to a field or parameter of type 'uint/uint2/uint3'}}
void CSMain_GID(float ID : SV_GroupID) {
}

[numthreads(8,8,1)]
// expected-error@+1 {{attribute 'SV_GroupID' only applies to a field or parameter of type 'uint/uint2/uint3'}}
void CSMain2_GID(ST GID : SV_GroupID) {

}

void foo_GID() {
// expected-warning@+1 {{'SV_GroupID' attribute only applies to parameters and non-static data members}}
  uint GIS : SV_GroupID;
}

struct ST2_GID {
// expected-warning@+1 {{'SV_GroupID' attribute only applies to parameters and non-static data members}}
    static uint GID : SV_GroupID;
    uint s_gid : SV_GroupID;
};

[numthreads(8,8,1)]
// expected-error@+1 {{attribute 'SV_GroupThreadID' only applies to a field or parameter of type 'uint/uint2/uint3'}}
void CSMain_GThreadID(float ID : SV_GroupThreadID) {
}

[numthreads(8,8,1)]
// expected-error@+1 {{attribute 'SV_GroupThreadID' only applies to a field or parameter of type 'uint/uint2/uint3'}}
void CSMain2_GThreadID(ST GID : SV_GroupThreadID) {

}

void foo_GThreadID() {
// expected-warning@+1 {{'SV_GroupThreadID' attribute only applies to parameters and non-static data members}}
  uint GThreadIS : SV_GroupThreadID;
}

struct ST2_GThreadID {
// expected-warning@+1 {{'SV_GroupThreadID' attribute only applies to parameters and non-static data members}}
    static uint GThreadID : SV_GroupThreadID;
    uint s_gthreadid : SV_GroupThreadID;
};


[shader("vertex")]
// expected-error@+4 {{attribute 'SV_GroupIndex' is unsupported in 'vertex' shaders, requires compute}}
// expected-error@+3 {{attribute 'SV_DispatchThreadID' is unsupported in 'vertex' shaders, requires compute}}
// expected-error@+2 {{attribute 'SV_GroupID' is unsupported in 'vertex' shaders, requires compute}}
// expected-error@+1 {{attribute 'SV_GroupThreadID' is unsupported in 'vertex' shaders, requires compute}}
void vs_main(int GI : SV_GroupIndex, uint ID : SV_DispatchThreadID, uint GID : SV_GroupID, uint GThreadID : SV_GroupThreadID) {}
