// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library \
// RUN:   -finclude-default-header -fsyntax-only -verify=expected,offset,dim2 \
// RUN:   -DHAS_OFFSET -DOFFSET_TYPE=int2 -DTEXTURE=Texture2D \
// RUN:   -DCOORD_TYPE=float2 -DGRAD_TYPE=float2 %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library \
// RUN:   -finclude-default-header -fsyntax-only -verify=expected,offset,dim2 \
// RUN:   -DHAS_OFFSET -DOFFSET_TYPE=int2 -DTEXTURE=Texture2DArray \
// RUN:   -DCOORD_TYPE=float3 -DGRAD_TYPE=float2 %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library \
// RUN:   -finclude-default-header -fsyntax-only \
// RUN:   -verify=expected,nooffset,dim3 -DOFFSET_TYPE=int3 \
// RUN:   -DTEXTURE=TextureCube -DCOORD_TYPE=float3 -DGRAD_TYPE=float3 %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library \
// RUN:   -finclude-default-header -fsyntax-only \
// RUN:   -verify=expected,nooffset,dim3 -DOFFSET_TYPE=int3 \
// RUN:   -DTEXTURE=TextureCubeArray -DCOORD_TYPE=float4 -DGRAD_TYPE=float3 %s

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   HAS_OFFSET         defined for types whose sampling and gathering methods
//                      have overloads taking an offset
//   OFFSET_TYPE        offset type, one component per resource dimension
//   TEXTURE            resource type name
//   COORD_TYPE         sample location type (DIM components plus the array
//                      slice)
//   GRAD_TYPE          SampleGrad ddx/ddy type, one component per resource
//                      dimension
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
  GRAD_TYPE ddx = (GRAD_TYPE)0;
  GRAD_TYPE ddy = (GRAD_TYPE)0;
  OFFSET_TYPE offset = (OFFSET_TYPE)0;
  float clamp = 0;

  tex.SampleGrad(samp, loc, ddx, ddy);

#ifdef HAS_OFFSET
  tex.SampleGrad(samp, loc, ddx, ddy, offset);
  tex.SampleGrad(samp, loc, ddx, ddy, offset, clamp);
#else
  // This type has no overload that takes an offset, but it does have one that
  // takes a clamp, so the 5th parameter is the clamp.
  tex.SampleGrad(samp, loc, ddx, ddy, clamp);

  // Passing an offset therefore selects the clamp overload and truncates.
  // nooffset-warning@+1 {{implicit conversion turns vector to scalar}}
  tex.SampleGrad(samp, loc, ddx, ddy, offset);
#endif

  // Too few arguments.
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 3 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 5 arguments, but 3 were provided}}
  // offset-note@*:* {{candidate function not viable: requires 6 arguments, but 3 were provided}}
  // expected-error@+1 {{no matching member function for call to 'SampleGrad'}}
  tex.SampleGrad(samp, loc, ddx);

  // Too many arguments.
  // offset-note@*:* {{candidate function not viable: requires 6 arguments, but 7 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 5 arguments, but 7 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 7 were provided}}
  // expected-error@+1 {{no matching member function for call to 'SampleGrad'}}
  tex.SampleGrad(samp, loc, ddx, ddy, offset, clamp, 0);

  // Invalid argument types.
  // offset-note@*:* {{no known conversion from 'const char[8]' to 'float' for 6th argument}}
  // expected-note@*:* {{candidate function not viable: requires 5 arguments, but 6 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 6 were provided}}
  // expected-error@+1 {{no matching member function for call to 'SampleGrad'}}
  tex.SampleGrad(samp, loc, ddx, ddy, offset, "invalid");
}
