// RUN: %clang_cc1 -fsyntax-only -verify "-triple" "thumbv7-apple-ios3.0.0" -target-feature +neon %s
// rdar://9208404

typedef int MP4Err;
typedef float Float32;
typedef float float32_t;
typedef __attribute__((neon_vector_type(4))) float32_t float32x4_t;
typedef float vFloat __attribute__((__vector_size__(16)));
typedef vFloat VFLOAT;
typedef unsigned long UInt32;

extern int bar (float32x4_t const *p);

int foo (const Float32 *realBufPtr) {
  float32x4_t const *vRealPtr = (VFLOAT *)&realBufPtr[0];
  return bar(vRealPtr);
}

MP4Err autoCorrelation2nd_Neon(Float32 *alphar, Float32 *alphai,
			    const Float32 *realBufPtr,
			    const Float32 *imagBufPtr,
			    const UInt32 len)
{
  float32x4_t const *vRealPtr = (VFLOAT *)&realBufPtr[0];
  return 0;
}

namespace rdar11688587 {
  typedef float float32_t;
  typedef __attribute__((neon_vector_type(4))) float32_t float32x4_t;

  template<int I>
  float test()
  {
    extern float32x4_t vec;
    return __extension__ ({ 
        float32x4_t __a = (vec); 
        (float32_t)__builtin_neon_vgetq_lane_f32(__a, I);  // expected-error-re{{argument value {{.*}} is outside the valid range}}
      });
  }

  template float test<1>();
  template float test<4>(); // expected-note{{in instantiation of function template specialization 'rdar11688587::test<4>' requested here}}
}
