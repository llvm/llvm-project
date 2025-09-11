// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -S -verify -o - -target-feature +avx

// RUN: not %clang_cc1 %s -triple=x86_64-apple-darwin -emit-obj -target-feature +avx 2> %t.err
// RUN: FileCheck < %t.err %s
// CHECK: 1 error generated

typedef unsigned short uint16_t;
typedef long long __m128i __attribute__((__vector_size__(16)));
typedef float __v8sf __attribute__ ((__vector_size__ (32)));
typedef float __m256 __attribute__ ((__vector_size__ (32)));
typedef uint16_t half;
typedef __attribute__ ((ext_vector_type( 8),__aligned__( 16))) half half8;
typedef __attribute__ ((ext_vector_type(16),__aligned__( 32))) half half16;
typedef __attribute__ ((ext_vector_type(16),__aligned__( 2))) half half16U;
typedef __attribute__ ((ext_vector_type( 8),__aligned__( 32))) float float8;
typedef __attribute__ ((ext_vector_type(16),__aligned__( 64))) float float16;
static inline half8 __attribute__((__overloadable__)) convert_half( float8 a ) {
  return __extension__ ({ __m256 __a = (a); (__m128i)__builtin_ia32_vcvtps2ph256((__v8sf)__a, (0x00)); }); // expected-error {{'__builtin_ia32_vcvtps2ph256' needs target feature f16c}}
}
static inline half16 __attribute__((__overloadable__)) convert_half( float16 a ) {
  half16 r;
  r.lo = convert_half(a.lo);
  return r;
}
void avx_test( uint16_t *destData, float16 argbF)
{
  ((half16U *)destData)[0] = convert_half(argbF);
}

half16 test( float16 a ) {
  half16 r;
  r.lo = convert_half(a.lo);
  return r;
}
void avx_test2( uint16_t *destData, float16 argbF)
{
  // expected-warning@+1{{AVX vector argument of type 'float16' (vector of 16 'float' values) without 'avx512f' enabled changes the ABI}}
  ((half16U *)destData)[0] = test(argbF);
}

__attribute__((always_inline)) half16 test2( float16 a ) {
  half16 r;
  r.lo = convert_half(a.lo);
  return r;
}
void avx_test3( uint16_t *destData, float16 argbF)
{
  ((half16U *)destData)[0] = test2(argbF);
}
