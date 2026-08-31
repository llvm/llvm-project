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
  float lod = 0;
  OFFSET_TYPE offset = (OFFSET_TYPE)0;

  tex.SampleLevel(samp, loc, lod);

#ifdef HAS_OFFSET
  tex.SampleLevel(samp, loc, lod, offset);
#else
  // This type has no overload that takes an offset.
  // nooffset-note@* {{'SampleLevel' declared here}}
  // nooffset-error@+1 {{too many arguments to function call, expected 3, have 4}}
  tex.SampleLevel(samp, loc, lod, offset);
#endif

  // Too few arguments.
  // offset-note@*:* {{candidate function not viable: requires 3 arguments, but 2 were provided}}
  // offset-note@*:* {{candidate function not viable: requires 4 arguments, but 2 were provided}}
  // nooffset-note@* {{'SampleLevel' declared here}}
  // offset-error@+2 {{no matching member function for call to 'SampleLevel'}}
  // nooffset-error@+1 {{too few arguments to function call, expected 3, have 2}}
  tex.SampleLevel(samp, loc);

  // Too many arguments.
  // offset-note@*:* {{candidate function not viable: requires 4 arguments, but 5 were provided}}
  // offset-note@*:* {{candidate function not viable: requires 3 arguments, but 5 were provided}}
  // nooffset-note@* {{'SampleLevel' declared here}}
  // offset-error@+2 {{no matching member function for call to 'SampleLevel'}}
  // nooffset-error@+1 {{too many arguments to function call, expected 3, have 5}}
  tex.SampleLevel(samp, loc, lod, offset, 0);

  // Invalid argument types.
  // offset-note@*:* {{no known conversion from 'const char[8]' to 'float' for 3rd argument}}
  // offset-note@*:* {{candidate function not viable: requires 4 arguments, but 3 were provided}}
  // offset-error@+2 {{no matching member function for call to 'SampleLevel'}}
  // nooffset-error@+1 {{cannot initialize a parameter of type 'float' with an lvalue of type 'const char[8]'}}
  tex.SampleLevel(samp, loc, "invalid");
}
