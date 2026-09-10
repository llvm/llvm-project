// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library \
// RUN:   -finclude-default-header -fsyntax-only -verify=expected,offset,dim2 \
// RUN:   -DHAS_OFFSET -DOFFSET_TYPE=int2 -DTEXTURE=Texture2D \
// RUN:   -DCOORD_TYPE=float2 %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library \
// RUN:   -finclude-default-header -fsyntax-only -verify=expected,offset,dim2 \
// RUN:   -DHAS_OFFSET -DOFFSET_TYPE=int2 -DTEXTURE=Texture2DArray \
// RUN:   -DCOORD_TYPE=float3 %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library \
// RUN:   -finclude-default-header -fsyntax-only \
// RUN:   -verify=expected,nooffset,dim3 -DOFFSET_TYPE=int3 \
// RUN:   -DTEXTURE=TextureCube -DCOORD_TYPE=float3 %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library \
// RUN:   -finclude-default-header -fsyntax-only \
// RUN:   -verify=expected,nooffset,dim3 -DOFFSET_TYPE=int3 \
// RUN:   -DTEXTURE=TextureCubeArray -DCOORD_TYPE=float4 %s

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   HAS_OFFSET         defined for types whose sampling and gathering methods
//                      have overloads taking an offset
//   OFFSET_TYPE        offset type, one component per resource dimension
//   TEXTURE            resource type name
//   COORD_TYPE         sample location type (DIM components plus the array
//                      slice)
//
// Check prefixes:
//   offset             diagnostics for types that have offset overloads
//   dim2               diagnostics naming a 2-component offset or location
//                      vector
//   nooffset           diagnostics for types that have no offset overloads
//   dim3               diagnostics naming a 3-component offset or location
//                      vector
//
// HAS_OFFSET, OFFSET_TYPE and OFFSET_ARG as the matching preprocessor

TEXTURE<float4> tex;
SamplerState samp;

void main() {
  COORD_TYPE loc = (COORD_TYPE)0;
  float bias = 0;
  OFFSET_TYPE offset = (OFFSET_TYPE)0;
  float clamp = 0;

  tex.SampleBias(samp, loc, bias);

#ifdef HAS_OFFSET
  tex.SampleBias(samp, loc, bias, offset);
  tex.SampleBias(samp, loc, bias, offset, clamp);
#else
  // This type has no overload that takes an offset, but it does have one that
  // takes a clamp, so the 4th parameter is the clamp.
  tex.SampleBias(samp, loc, bias, clamp);

  // Passing an offset therefore selects the clamp overload and truncates.
  // nooffset-warning@+1 {{implicit conversion turns vector to scalar}}
  tex.SampleBias(samp, loc, bias, offset);
#endif

  // Too few arguments.
  // offset-note@*:* {{candidate function not viable: requires 5 arguments, but 2 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 2 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 2 were provided}}
  // expected-error@+1 {{no matching member function for call to 'SampleBias'}}
  tex.SampleBias(samp, loc);

  // Too many arguments.
  // offset-note@*:* {{candidate function not viable: requires 5 arguments, but 6 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 6 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 6 were provided}}
  // expected-error@+1 {{no matching member function for call to 'SampleBias'}}
  tex.SampleBias(samp, loc, bias, offset, clamp, 0);

  // Invalid argument types.
  // offset-note@*:* {{no known conversion from 'const char[8]' to 'float' for 5th argument}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 5 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 5 were provided}}
  // expected-error@+1 {{no matching member function for call to 'SampleBias'}}
  tex.SampleBias(samp, loc, bias, offset, "invalid");
}
