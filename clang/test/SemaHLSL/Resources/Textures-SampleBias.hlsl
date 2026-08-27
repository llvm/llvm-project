// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library \
// RUN:   -finclude-default-header -fsyntax-only -verify -DOFFSET_TYPE=int2 \
// RUN:   -DTEXTURE=Texture2D -DCOORD_TYPE=float2 %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library \
// RUN:   -finclude-default-header -fsyntax-only -verify -DOFFSET_TYPE=int2 \
// RUN:   -DTEXTURE=Texture2DArray -DCOORD_TYPE=float3 %s

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   OFFSET_TYPE        offset type, one component per resource dimension
//   TEXTURE            resource type name
//   COORD_TYPE         sample location type (DIM components plus the array
//                      slice)

TEXTURE<float4> tex;
SamplerState samp;

void main() {
  COORD_TYPE loc = (COORD_TYPE)0;
  float bias = 0;
  OFFSET_TYPE offset = (OFFSET_TYPE)0;
  float clamp = 0;

  tex.SampleBias(samp, loc, bias);
  tex.SampleBias(samp, loc, bias, offset);
  tex.SampleBias(samp, loc, bias, offset, clamp);

  // Too few arguments.
  tex.SampleBias(samp, loc); // expected-error {{no matching member function for call to 'SampleBias'}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 2 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 2 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 5 arguments, but 2 were provided}}

  // Too many arguments.
  tex.SampleBias(samp, loc, bias, offset, clamp, 0); // expected-error {{no matching member function for call to 'SampleBias'}}
  // expected-note@*:* {{candidate function not viable: requires 5 arguments, but 6 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 6 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 6 were provided}}

  // Invalid argument types.
  tex.SampleBias(samp, loc, bias, offset, "invalid"); // expected-error {{no matching member function for call to 'SampleBias'}}
  // expected-note@*:* {{no known conversion from 'const char[8]' to 'float' for 5th argument}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 5 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 5 were provided}}
}
