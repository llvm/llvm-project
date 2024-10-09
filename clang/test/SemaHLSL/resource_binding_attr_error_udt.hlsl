// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - -fsyntax-only %s -verify

template<typename T>
struct MyTemplatedUAV {
  __hlsl_resource_t [[hlsl::resource_class(UAV)]] x;
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

// Valid: f is skipped, SRVBuf is bound to t0, UAVBuf is bound to u0
struct Eg1 {
  float f;
  MySRV SRVBuf;
  MyUAV UAVBuf;
  };
Eg1 e1 : register(t0) : register(u0); 

// Valid: f is skipped, SRVBuf is bound to t0, UAVBuf is bound to u0. 
// UAVBuf2 gets automatically assigned to u1 even though there is no explicit binding for u1.
struct Eg2 {
  float f;
  MySRV SRVBuf;
  MyUAV UAVBuf;
  MyUAV UAVBuf2;
  };
Eg2 e2 : register(t0) : register(u0); 

// Valid: Bar, the struct within Eg3, has a valid resource that can be bound to t0. 
struct Eg3 {
  struct Bar {
    MyUAV a;
  };
  Bar b;
};
Eg3 e3 : register(u0);

// Valid: the first sampler state object within 's' is bound to slot 5
struct Eg4 {
  MySampler s[3];
};

Eg4 e4 : register(s5);


struct Eg5 {
  float f;
}; 
// expected-warning@+1{{binding type 't' only applies to types containing SRV resources}}
Eg5 e5 : register(t0);

struct Eg6 {
  float f;
}; 
// expected-warning@+1{{binding type 'u' only applies to types containing UAV resources}}
Eg6 e6 : register(u0);

struct Eg7 {
  float f;
}; 
// expected-warning@+1{{binding type 'b' only applies to types containing constant buffer resources}}
Eg7 e7 : register(b0);

struct Eg8 {
  float f;
}; 
// expected-warning@+1{{binding type 's' only applies to types containing sampler state}}
Eg8 e8 : register(s0);

struct Eg9 {
  MySRV s;
}; 
// expected-warning@+1{{binding type 'c' only applies to types containing numeric types}}
Eg9 e9 : register(c0);

struct Eg10{
  // expected-error@+1{{'register' attribute only applies to cbuffer/tbuffer and external global variables}}
  MyTemplatedUAV<int> a : register(u9);
};
Eg10 e10;


template<typename R>
struct Eg11 {
    R b;
};
// expected-warning@+1{{binding type 'u' only applies to types containing UAV resources}}
Eg11<MySRV> e11 : register(u0);
// invalid because after template expansion, there are no valid resources inside Eg11 to bind as a UAV, only an SRV


struct Eg12{
  MySRV s1;
  MySRV s2;
};
// expected-warning@+3{{binding type 'u' only applies to types containing UAV resources}}
// expected-warning@+2{{binding type 'u' only applies to types containing UAV resources}}
// expected-error@+1{{binding type 'u' cannot be applied more than once}}
Eg12 e12 : register(u9) : register(u10);

struct Eg13{
  MySRV s1;
  MySRV s2;
};
// expected-warning@+4{{binding type 'u' only applies to types containing UAV resources}}
// expected-warning@+3{{binding type 'u' only applies to types containing UAV resources}}
// expected-warning@+2{{binding type 'u' only applies to types containing UAV resources}}
// expected-error@+1{{binding type 'u' cannot be applied more than once}}
Eg13 e13 : register(u9) : register(u10) : register(u11);

struct Eg14{
 MyTemplatedUAV<int> r1;  
};
// expected-warning@+1{{binding type 't' only applies to types containing SRV resources}}
Eg14 e14 : register(t9);

struct Eg15 {
  float f[4];
}; 
// expected no error
Eg15 e15 : register(c0);

