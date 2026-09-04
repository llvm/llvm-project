// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -finclude-default-header -DTEXTURE=Texture2D -DHAS_OFFSET \
// RUN:   -DLOAD_TYPE=int3 -DLOAD_FLOAT_TYPE=float3 -DNARROW_LOAD_TYPE=int2 \
// RUN:   -DWIDE_LOAD_TYPE=int4 -DOFFSET_TYPE=int2 -DOFFSET_FLOAT_TYPE=float2 \
// RUN:   -DWIDE_OFFSET_TYPE=int3 -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -finclude-default-header -DTEXTURE=Texture2DArray -DHAS_OFFSET \
// RUN:   -DLOAD_TYPE=int4 -DLOAD_FLOAT_TYPE=float4 -DNARROW_LOAD_TYPE=int3 \
// RUN:   -DOFFSET_TYPE=int2 -DOFFSET_FLOAT_TYPE=float2 \
// RUN:   -DWIDE_OFFSET_TYPE=int3 -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -finclude-default-header -DTEXTURE=RWTexture2D -DLOAD_TYPE=int2 \
// RUN:   -DLOAD_FLOAT_TYPE=float2 -DNARROW_LOAD_TYPE=int1 \
// RUN:   -DWIDE_LOAD_TYPE=int3 -DOFFSET_TYPE=int2 -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -finclude-default-header -DTEXTURE=RWTexture2DArray -DLOAD_TYPE=int3 \
// RUN:   -DLOAD_FLOAT_TYPE=float3 -DNARROW_LOAD_TYPE=int2 \
// RUN:   -DWIDE_LOAD_TYPE=int4 -DOFFSET_TYPE=int2 -verify %s

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   TEXTURE            resource type name
//   LOAD_TYPE          Load location type
//   LOAD_FLOAT_TYPE    Load location type, but floating point
//   NARROW_LOAD_TYPE   a Load location with one component too few
//   WIDE_LOAD_TYPE     a Load location with one component too many; only
//                      defined where a wider vector exists
//   OFFSET_TYPE        offset type, one component per resource dimension
//   OFFSET_FLOAT_TYPE  offset type, but floating point
//   WIDE_OFFSET_TYPE   an offset with one component too many
//   HAS_OFFSET         defined for read-only (SRV) textures, whose location
//                      carries a trailing mip level and which have a second
//                      Load overload taking an offset
//
// A UAV descriptor binds a single mip slice, so a RWTexture location has no mip
// component, and DXIL's TextureLoad takes no offset on a UAV. Those types
// therefore have one Load overload instead of two, which changes the
// diagnostics from overload resolution failures to plain arity errors.
//
// The diagnostics that name a vector width use `-re` directives so that the
// same assertions apply to every texture type.

TEXTURE<float4> t;

float4 test_exact_location(LOAD_TYPE loc) {
  // No diagnostics expected: the location has exactly the right width.
  return t.Load(loc);
}

float4 test_too_few_args() {
#ifdef HAS_OFFSET
  return t.Load(); // expected-error {{no matching member function for call to 'Load'}}
  // expected-note@*:* {{candidate function not viable: requires single argument 'Location', but no arguments were provided}}
  // expected-note@*:* {{candidate function not viable: requires 2 arguments, but 0 were provided}}
#else
  return t.Load(); // expected-error {{too few arguments to function call, single argument 'Location' was not specified}}
  // expected-note@*:* {{'Load' declared here}}
#endif
}

float4 test_too_many_args(LOAD_TYPE loc, OFFSET_TYPE offset) {
#ifdef HAS_OFFSET
  return t.Load(loc, offset, 1); // expected-error {{no matching member function for call to 'Load'}}
  // expected-note@*:* {{candidate function not viable: requires 2 arguments, but 3 were provided}}
  // expected-note@*:* {{candidate function not viable: requires single argument 'Location', but 3 arguments were provided}}
#else
  // There is no offset overload on a UAV, so even two arguments is too many.
  return t.Load(loc, offset); // expected-error {{too many arguments to function call, expected single argument 'Location', have 2 arguments}}
  // expected-note@*:* {{'Load' declared here}}
#endif
}

float4 test_invalid_coord_type(LOAD_FLOAT_TYPE loc) {
  return t.Load(loc); // expected-warning {{implicit conversion turns floating-point number into integer: }}
}

#ifdef HAS_OFFSET
float4 test_invalid_offset_type(LOAD_TYPE loc, OFFSET_FLOAT_TYPE offset) {
  // expected-warning-re@+1 {{implicit conversion turns floating-point number into integer: '{{float[0-9]}}' (aka 'vector<float, {{[0-9]}}>') to 'vector<int, {{[0-9]}}>' (vector of {{[0-9]}} 'int' values)}}
  return t.Load(loc, offset);
}
#endif

float4 test_invalid_location_count(NARROW_LOAD_TYPE loc) {
#ifdef HAS_OFFSET
  return t.Load(loc); // expected-error {{no matching member function for call to 'Load'}}
  // expected-note@*:* {{candidate function not viable: no known conversion from }}
  // expected-note@*:* {{candidate function not viable: requires 2 arguments, but 1 was provided}}
#else
  // expected-error-re@+1 {{cannot initialize a parameter of type 'vector<int, {{[0-9]}}>' (vector of {{[0-9]}} 'int' values) with an lvalue of type '{{int[0-9]?}}'{{( \(aka 'vector<int, [0-9]>'\))?}}}}
  return t.Load(loc);
#endif
}

#ifdef WIDE_LOAD_TYPE
float4 test_truncated_location_count(WIDE_LOAD_TYPE loc) {
  // On a UAV this is exactly a location that still carries a mip component: it
  // truncates like any other over-wide HLSL vector argument.
  // expected-warning-re@+1 {{implicit conversion truncates vector: '{{int[0-9]}}' (aka 'vector<int, {{[0-9]}}>') to 'vector<int, {{[0-9]}}>' (vector of {{[0-9]}} 'int' values)}}
  return t.Load(loc);
}
#endif

#ifdef HAS_OFFSET
float4 test_splatted_offset_count(LOAD_TYPE loc, int offset) {
  // No errors expected. The vector will be generated by splatting `offset`.
  return t.Load(loc, offset);
}

float4 test_invalid_offset_count(LOAD_TYPE loc, WIDE_OFFSET_TYPE offset) {
  // expected-warning-re@+1 {{implicit conversion truncates vector: '{{int[0-9]}}' (aka 'vector<int, {{[0-9]}}>') to 'vector<int, {{[0-9]}}>' (vector of {{[0-9]}} 'int' values)}}
  return t.Load(loc, offset);
}
#endif
