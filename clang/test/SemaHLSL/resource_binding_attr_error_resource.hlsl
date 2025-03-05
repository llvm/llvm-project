// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - -fsyntax-only %s -verify

// This test validates the diagnostics that are emitted when a variable with a "resource" type
// is bound to a register using the register annotation


template<typename T>
struct MyTemplatedSRV {
  __hlsl_resource_t [[hlsl::resource_class(SRV)]] x;
};

struct MySRV {
  __hlsl_resource_t [[hlsl::resource_class(SRV)]] x;
};

struct MySampler {
  __hlsl_resource_t [[hlsl::resource_class(Sampler)]] x;
};

struct MyUAV {
  __hlsl_resource_t [[hlsl::resource_class(UAV)]] x;
};

struct MyCBuffer {
  __hlsl_resource_t [[hlsl::resource_class(CBuffer)]] x;
};


// expected-error@+1  {{binding type 'i' ignored. The 'integer constant' binding type is no longer supported}}
MySRV invalid : register(i2);

// expected-error@+1  {{binding type 't' only applies to SRV resources}}
MyUAV a : register(t2, space1);

// expected-error@+1  {{binding type 'u' only applies to UAV resources}}
MySampler b : register(u2, space1);

// expected-error@+1  {{binding type 'b' only applies to constant buffer resources}}
MyTemplatedSRV<int> c : register(b2);

// expected-error@+1  {{binding type 's' only applies to sampler state}}
MyUAV d : register(s2, space1);

// empty binding prefix cases:
// expected-error@+1 {{expected identifier}}
MyTemplatedSRV<int> e: register();

// expected-error@+1 {{expected identifier}}
MyTemplatedSRV<int> f: register("");
