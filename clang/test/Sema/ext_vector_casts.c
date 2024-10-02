// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -fsyntax-only -verify -flax-vector-conversions=none -Wconversion %s

typedef __attribute__((ext_vector_type(8))) _Bool BoolVector;

typedef __attribute__(( ext_vector_type(2) )) float float2;
typedef __attribute__(( ext_vector_type(3) )) float float3;
typedef __attribute__(( ext_vector_type(4) )) int int4;
typedef __attribute__(( ext_vector_type(8) )) short short8;
typedef __attribute__(( ext_vector_type(4) )) float float4;
typedef float t3 __attribute__ ((vector_size (16)));
typedef __typeof__(sizeof(int)) size_t;
typedef unsigned long ulong2 __attribute__ ((ext_vector_type(2)));
typedef size_t stride4 __attribute__((ext_vector_type(4)));
typedef _Bool bool4 __attribute__(( ext_vector_type(4) ));

static void test(void) {
    float2 vec2;
    float3 vec3;
    float4 vec4, vec4_2;
    int4 ivec4;
    short8 ish8;
    t3 vec4_3;
    bool4 bvec4 = 0;
    int *ptr;
    int i;

    vec3 += vec2; // expected-error {{cannot convert between vector values of different size}}
    vec4 += vec3; // expected-error {{cannot convert between vector values of different size}}
    
    vec4 = 5.0f;
    vec4 = (float4)5.0f;
    vec4 = (float4)5;
    vec4 = (float4)vec4_3;
    
    ivec4 = (int4)5.0f;
    ivec4 = (int4)5;
    ivec4 = (int4)vec4_3;
    
    i = (int)ivec4; // expected-error {{invalid conversion between vector type 'int4' (vector of 4 'int' values) and integer type 'int' of different size}}
    i = ivec4; // expected-error {{assigning to 'int' from incompatible type 'int4' (vector of 4 'int' values)}}
    
    ivec4 = (int4)ptr; // expected-error {{invalid conversion between vector type 'int4' (vector of 4 'int' values) and scalar type 'int *'}}
    
    vec4 = (float4)vec2; // expected-error {{invalid conversion between ext-vector type 'float4' (vector of 4 'float' values) and 'float2' (vector of 2 'float' values)}}
  
    ish8 += 5;
    ivec4 *= 5;
     vec4 /= 5.2f;
     vec4 %= 4; // expected-error {{invalid operands to binary expression ('float4' (vector of 4 'float' values) and 'int')}}
    ivec4 %= 4;
    ivec4 += vec4; // expected-error {{cannot convert between vector values of different size ('int4' (vector of 4 'int' values) and 'float4' (vector of 4 'float' values))}}
    ivec4 += (int4)vec4;
    ivec4 -= ivec4;
    ivec4 |= ivec4;
    ivec4 += ptr; // expected-error {{cannot convert between vector and non-scalar values ('int4' (vector of 4 'int' values) and 'int *')}}

    bvec4 != 0; // expected-warning {{inequality comparison result unused}} \
                // expected-note {{use '|=' to turn this inequality comparison into an or-assignment}}
}

typedef __attribute__(( ext_vector_type(2) )) float2 vecfloat2; // expected-error{{invalid vector element type 'float2' (vector of 2 'float' values)}}

void inc(float2 f2) {
  f2++; // expected-error{{cannot increment value of type 'float2' (vector of 2 'float' values)}}
  __real f2; // expected-error{{invalid type 'float2' (vector of 2 'float' values) to __real operator}}
}

typedef enum
{
    uchar_stride = 1,
    uchar4_stride = 4,
    ushort4_stride = 8,
    short4_stride = 8,
    uint4_stride = 16,
    int4_stride = 16,
    float4_stride = 16,
} PixelByteStride;

stride4 RDar15091442_get_stride4(int4 x, PixelByteStride pixelByteStride);
stride4 RDar15091442_get_stride4(int4 x, PixelByteStride pixelByteStride)
{
    stride4 stride;
    // This previously caused an assertion failure.
    stride.lo = ((ulong2) x) * pixelByteStride; // no-warning
    return stride;
}

typedef __attribute__((ext_vector_type(4))) float float32x4_t;

typedef float C3DVector3 __attribute__((ext_vector_type(3)));

extern float32x4_t vabsq_f32(float32x4_t __a);

C3DVector3 Func(const C3DVector3 a) {
    return (C3DVector3)vabsq_f32((float32x4_t)a); // expected-error {{invalid conversion between ext-vector type 'float32x4_t' (vector of 4 'float' values) and 'C3DVector3' (vector of 3 'float' values)}}
}

typedef double double2 __attribute__ ((ext_vector_type(2)));

static void splats(int i, long l, __uint128_t t, float f, double d) {
  short8 vs = 0;
  int4 vi = i;
  ulong2 vl = (unsigned long)l;
  float2 vf = f;
  double2 vd = d;
  
  vs = 65536 + vs; // expected-warning {{implicit conversion from 'int' to 'short8' (vector of 8 'short' values) changes value from 65536 to 0}}
  vs = vs + i; // expected-warning {{implicit conversion loses integer precision}}
  vs = vs + 1;
  vs = vs + 1.f; // expected-error {{cannot convert between vector values of different size}}
  
  vi = l + vi; // expected-warning {{implicit conversion loses integer precision}}
  vi = 1 + vi;
  vi = vi + 2.0; // expected-error {{cannot convert between vector values of different size}}
  vi = vi + 0xffffffff; // expected-warning {{implicit conversion changes signedness}}
  
  vl = l + vl; // expected-warning {{implicit conversion changes signedness}}
  vl = vl + t; // expected-warning {{implicit conversion loses integer precision}}
  
  vf = 1 + vf;
  vf = l + vf; // expected-warning {{implicit conversion from 'long' to 'float2' (vector of 2 'float' values) may lose precision}}
  vf = 2.0 + vf;
  vf = d + vf; // expected-warning {{implicit conversion loses floating-point precision}}
  vf = vf + 0xffffffff; // expected-warning {{implicit conversion from 'unsigned int' to 'float2' (vector of 2 'float' values) changes value from 4294967295 to 4294967296}}
  vf = vf + 2.1; // expected-warning {{implicit conversion loses floating-point precision}}

  vd = l + vd; // expected-warning {{implicit conversion from 'long' to 'double2' (vector of 2 'double' values) may lose precision}}
  vd = vd + t; // expected-warning {{implicit conversion from '__uint128_t' (aka 'unsigned __int128') to 'double2' (vector of 2 'double' values) may lose precision}}
}
