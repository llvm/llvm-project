// RUN: %clang_cc1 -triple aarch64 -target-feature +sve -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple aarch64 -target-feature +sve -fsyntax-only -verify -x c++ %s

#include <arm_sve.h>

svfloat16_t test_f16(svfloat16_t x) { return __arithmetic_fence(x); }
svfloat32_t test_f32(svfloat32_t x) { return __arithmetic_fence(x); }
svfloat64_t test_f64(svfloat64_t x) { return __arithmetic_fence(x); }
svbfloat16_t test_bf16(svbfloat16_t x) { return __arithmetic_fence(x); }

svfloat32_t test_const(const svfloat32_t x) {
  return __arithmetic_fence(x);
}

void test_invalid(svint32_t i, svuint64_t u, svbool_t p,
                  svfloat32x2_t t2, svfloat64x3_t t3, svfloat16x4_t t4,
                  svcount_t count, svmfloat8_t m) {
  __arithmetic_fence(i); // expected-error {{invalid operand of type 'svint32_t'}}
  __arithmetic_fence(u); // expected-error {{invalid operand of type 'svuint64_t'}}
  __arithmetic_fence(p); // expected-error {{invalid operand of type 'svbool_t'}}
  __arithmetic_fence(t2); // expected-error {{invalid operand of type 'svfloat32x2_t'}}
  __arithmetic_fence(t3); // expected-error {{invalid operand of type 'svfloat64x3_t'}}
  __arithmetic_fence(t4); // expected-error {{invalid operand of type 'svfloat16x4_t'}}
  __arithmetic_fence(count); // expected-error {{invalid operand of type 'svcount_t'}}
  __arithmetic_fence(m); // expected-error {{invalid operand of type 'svmfloat8_t'}}
}

#ifdef __cplusplus
template <typename T> T test_template(T x) {
  return __arithmetic_fence(x); // expected-error {{invalid operand of type '__SVInt32_t'}}
}

svfloat64_t test_dependent(svfloat64_t x) {
  return test_template(x);
}

void test_dependent_invalid(svint32_t x) {
  test_template(x); // expected-note {{in instantiation of function template specialization 'test_template<__SVInt32_t>' requested here}}
}
#endif
