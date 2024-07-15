// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - -fsyntax-only %s -verify

// TODO: Implement "Buffer", we use a substitute UDT
// to test the 't' binding type for this test.

template<typename T>
struct [[hlsl::resource_class(SRV)]] Buffer {
  T x;
};

// TODO: Implement "SamplerState", we use a substitute UDT
// to test the 's' binding type for this test.
struct [[hlsl::resource_class(Sampler)]] SamplerState {
  int x;
};

// TODO: Implement "Texture2D", we use a substitute UDT
// to test a non-templated 't' binding type for this test.
struct [[hlsl::resource_class(UAV)]] Texture2D {
  int x;
};

struct Eg1 {
  float f;
  Buffer<float> Buf;
  RWBuffer<float> RWBuf;
  };
Eg1 e1 : register(t0) : register(u0); 
// Valid: f is skipped, Buf is bound to t0, RWBuf is bound to u0


struct Eg2 {
  float f;
  Buffer<float> Buf;
  RWBuffer<float> RWBuf;
  RWBuffer<float> RWBuf2;
  };
Eg2 e2 : register(t0) : register(u0); 
// Valid: f is skipped, Buf is bound to t0, RWBuf is bound to u0. 
// RWBuf2 gets automatically assigned to u1 even though there is no explicit binding for u1.

struct Eg3 {
  struct Bar {
    RWBuffer<int> a;
    };
    Bar b;
};
Eg3 e3 : register(u0);
// Valid: Bar, the struct within Eg3, has a valid resource that can be bound to t0. 

struct Eg4 {
  SamplerState s[3];
};

Eg4 e4 : register(s5);
// Valid: the first sampler state object within Eg5's s is bound to slot 5


struct Eg5 {
  float f;
}; 
// expected-warning@+1{{variable of type 'Eg5' bound to register type 't' does not contain a matching 'srv' resource}}
Eg5 e5 : register(t0);

struct Eg6 {
  struct Bar {
    float f;
  };
  Bar b;
};
// expected-warning@+1{{variable of type 'Eg6' bound to register type 't' does not contain a matching 'srv' resource}}
Eg6 e6 : register(t0);

struct Eg7 {
  RWBuffer<int> a;
}; 
// expected-warning@+1{{register 'c' used on type with no contents to allocate in a constant buffer}}
Eg7 e7 : register(c0);


struct Eg8{
  // expected-error@+1{{'register' attribute only applies to cbuffer/tbuffer and external global variables}}
  RWBuffer<int> a : register(u9);
};
Eg8 e8;


template<typename R>
struct Eg9 {
    R b;
};
// expecting warning: {{variable of type 'Eg9' bound to register type 'u' does not contain a matching 'uav' resource}}
Eg9<Texture2D> e9 : register(u0);

// invalid because after template expansion, there are no valid resources inside Eg10 to bind as a UAV.


struct Eg10{
  RWBuffer<int> a;
  RWBuffer<int> b;
};

// expected-error@+1{{conflicting register annotations: multiple register annotations detected for register type 'u'}}
Eg10 e10 : register(u9) : register(u10);
// expected-error@+1{{conflicting register annotations: multiple register annotations detected for register type 'u'}}
Eg10 e10a : register(u9, space0) : register(u9, space1);

struct Eg11{
  RWBuffer<int> a;  
};

// expected-warning@+1{{register 'c' used on type with no contents to allocate in a constant buffer}}
Eg11 e11 : register(c9);