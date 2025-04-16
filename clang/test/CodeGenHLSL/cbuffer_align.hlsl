// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -fsyntax-only -verify -verify-ignore-unexpected=warning

struct S0 {
  half a;
  half b;
  half c;
  half d;
  half e;
  half f;
  half g;
  half h;
};

cbuffer CB0Pass {
  S0 s0p : packoffset(c0.x);
  float f0p : packoffset(c1.x);
}

cbuffer CB0Fail {
  S0 s0f : packoffset(c0.x);
  float f0f : packoffset(c0.w);
  // expected-error@-1 {{packoffset overlap between 'f0f', 's0f'}}
}

struct S1 {
  float a;
  double b;
  float c;
};

cbuffer CB1Pass {
  S1 s1p : packoffset(c0.x);
  float f1p : packoffset(c1.y);
}

cbuffer CB1Fail {
  S1 s1f : packoffset(c0.x);
  float f1f : packoffset(c1.x);
  // expected-error@-1 {{packoffset overlap between 'f1f', 's1f'}}
}

struct S2 {
  float3 a;
  float2 b;
};

cbuffer CB2Pass {
  S2 s2p : packoffset(c0.x);
  float f2p : packoffset(c1.z);
}

cbuffer CB2Fail {
  S2 s2f : packoffset(c0.x);
  float f2f : packoffset(c1.y);
  // expected-error@-1 {{packoffset overlap between 'f2f', 's2f'}}
}

struct S3 {
  float3 a;
  float b;
};

cbuffer CB3Pass {
  S3 s3p : packoffset(c0.x);
  float f3p : packoffset(c1.x);
}

cbuffer CB3Fail {
  S3 s3f : packoffset(c0.x);
  float f3f : packoffset(c0.w);
  // expected-error@-1 {{packoffset overlap between 'f3f', 's3f'}}
}

struct S4 {
  float2 a;
  float2 b;
};

cbuffer CB4Pass {
  S4 s4p : packoffset(c0.x);
  float f4p : packoffset(c1.x);
}

cbuffer CB4Fail {
  S4 s4f : packoffset(c0.x);
  float f4f : packoffset(c0.w);
  // expected-error@-1 {{packoffset overlap between 'f4f', 's4f'}}
}

struct S5 {
  float a[3];
};

cbuffer CB5Pass {
  S5 s5p : packoffset(c0.x);
  float f5p : packoffset(c2.y);
}

cbuffer CB5Fail {
  S5 s5f : packoffset(c0.x);
  float f5f : packoffset(c2.x);
  // expected-error@-1 {{packoffset overlap between 'f5f', 's5f'}}
}

struct S6 {
  float a;
  float2 b;
};

cbuffer CB6Pass {
  S6 s6p : packoffset(c0.x);
  float f6p : packoffset(c0.w);
}

cbuffer CB6Fail {
  S6 s6f : packoffset(c0.x);
  float f6f : packoffset(c0.y);
  // expected-error@-1 {{packoffset overlap between 'f6f', 's6f'}}
}
