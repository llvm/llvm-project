// This file should be fine for all targets
// XFAIL: target={{.*(iu|riscv).*}}
// RUN: %clang -g -S -O2 -emit-llvm %s

#include <stdint.h>

typedef _Float16 f16t64 __attribute__((__vector_size__(128)))
__attribute__((aligned(128)));
typedef float f32t32 __attribute__((__vector_size__(128)))
__attribute__((aligned(128)));
typedef double f64t16 __attribute__((__vector_size__(128)))
__attribute__((aligned(128)));
typedef long double f128t8 __attribute__((__vector_size__(128)))
__attribute__((aligned(128)));
typedef int32_t i32t64 __attribute__((__vector_size__(256)))
__attribute__((aligned(256)));
typedef int32_t i32t32 __attribute__((__vector_size__(128)))
__attribute__((aligned(128)));
typedef int32_t i32t16 __attribute__((__vector_size__(64)))
__attribute__((aligned(64)));
typedef int32_t i32t8 __attribute__((__vector_size__(32)))
__attribute__((aligned(32)));

typedef char i1t64 __attribute__((__vector_size__(64)))
__attribute__((aligned(64)));

// Element-wise (A + B) * 2
extern inline f16t64 ripple_ew_add_and_double(f16t64 A, f16t64 B) {
  return (A + B) * 2.f;
}

// Non-element-wise (A + B) / 2
extern inline f16t64 ripple_add_and_half(f16t64 A, _Float16 B) {
  return (A + B) / 2.f;
}

// Non maskable external ripple function (B is accessed inside the function)
extern inline f16t64 ripple_add_and_half_ptr(f16t64 A, _Float16 *B) {
  return (A + *B) / 2.f;
}

// Maskable external ripple function
extern inline f16t64 ripple_mask_add_and_half_ptr(f16t64 A, _Float16 *B,
                                                  i1t64 Mask) {
  (void)Mask;
  return ((A + *B) / 2.f);
}

extern inline f16t64 ripple_add_and_half_ptr_has_no_mask_version(f16t64 A,
                                                                 _Float16 *B) {
  return (A + *B) / 2.f;
}

// Make has_mask_and_non_mask_version non pure
extern _Float16 B;
extern inline f16t64 ripple_ew_non_pure_ew_separate_mask(f16t64 A) {
  return A * B;
}

extern inline f16t64 ripple_ew_mask_non_pure_ew_separate_mask(f16t64 A,
                                                              i1t64 Mask) {
  (void)Mask;
  return A * B;
}

extern inline f16t64 ripple_mysinf16(f16t64 A) { return A; }

#define CREATE_UNARY_MATH_TEST_FUN(name)                                       \
  extern inline f16t64 ripple_ew_pure_##name##f16(f16t64 Vals) {               \
    return Vals;                                                               \
  }                                                                            \
  extern inline f32t32 ripple_ew_pure_##name##f(f32t32 Vals) { return Vals; }  \
  extern inline f64t16 ripple_ew_pure_##name(f64t16 Vals) { return Vals; }     \
  extern inline f128t8 ripple_ew_pure_##name##l(f128t8 Vals) { return Vals; }

CREATE_UNARY_MATH_TEST_FUN(sqrt)
CREATE_UNARY_MATH_TEST_FUN(asin)
CREATE_UNARY_MATH_TEST_FUN(acos)
CREATE_UNARY_MATH_TEST_FUN(atan)
CREATE_UNARY_MATH_TEST_FUN(sin)
CREATE_UNARY_MATH_TEST_FUN(cos)
CREATE_UNARY_MATH_TEST_FUN(tan)
CREATE_UNARY_MATH_TEST_FUN(sinh)
CREATE_UNARY_MATH_TEST_FUN(cosh)
CREATE_UNARY_MATH_TEST_FUN(tanh)
CREATE_UNARY_MATH_TEST_FUN(log)
CREATE_UNARY_MATH_TEST_FUN(log10)
CREATE_UNARY_MATH_TEST_FUN(log2)
CREATE_UNARY_MATH_TEST_FUN(exp)
CREATE_UNARY_MATH_TEST_FUN(exp2)
CREATE_UNARY_MATH_TEST_FUN(exp10)
CREATE_UNARY_MATH_TEST_FUN(fabs)
CREATE_UNARY_MATH_TEST_FUN(floor)
CREATE_UNARY_MATH_TEST_FUN(ceil)
CREATE_UNARY_MATH_TEST_FUN(trunc)
CREATE_UNARY_MATH_TEST_FUN(rint)
CREATE_UNARY_MATH_TEST_FUN(nearbyint)
CREATE_UNARY_MATH_TEST_FUN(round)
CREATE_UNARY_MATH_TEST_FUN(roundeven)

#undef CREATE_UNARY_MATH_TEST_FUN

#define CREATE_BINARY_MATH_TEST_FUN(name)                                      \
  extern inline f16t64 ripple_ew_pure_##name##f16(f16t64 Vals, f16t64 Vals2) { \
    (void)Vals2;                                                               \
    return Vals;                                                               \
  }                                                                            \
  extern inline f32t32 ripple_ew_pure_##name##f(f32t32 Vals, f32t32 Vals2) {   \
    (void)Vals2;                                                               \
    return Vals;                                                               \
  }                                                                            \
  extern inline f64t16 ripple_ew_pure_##name(f64t16 Vals, f64t16 Vals2) {      \
    (void)Vals2;                                                               \
    return Vals;                                                               \
  }                                                                            \
  extern inline f128t8 ripple_ew_pure_##name##l(f128t8 Vals, f128t8 Vals2) {   \
    (void)Vals2;                                                               \
    return Vals;                                                               \
  }

CREATE_BINARY_MATH_TEST_FUN(atan2)
CREATE_BINARY_MATH_TEST_FUN(pow)
CREATE_BINARY_MATH_TEST_FUN(copysign)

#undef CREATE_BINARY_MATH_TEST_FUN

extern inline void ripple_ew_pure_sincosf16(f16t64 Vals, f16t64 *sin,
                                            f16t64 *cos) {
  *sin = Vals;
  *cos = Vals;
}
extern inline void ripple_ew_pure_sincosf(f32t32 Vals, f32t32 *sin,
                                          f32t32 *cos) {
  *sin = Vals;
  *cos = Vals;
}
extern inline void ripple_ew_pure_sincos(f64t16 Vals, f64t16 *sin,
                                         f64t16 *cos) {
  *sin = Vals;
  *cos = Vals;
}
extern inline void ripple_ew_pure_sincosl(f128t8 Vals, f128t8 *sin,
                                          f128t8 *cos) {
  *sin = Vals;
  *cos = Vals;
}

extern inline f16t64 ripple_ew_pure_modff16(f16t64 Vals,
                                            f16t64 *FractionalPart) {
  *FractionalPart = Vals;
  return Vals;
}
extern inline f32t32 ripple_ew_pure_modff(f32t32 Vals, f32t32 *FractionalPart) {
  *FractionalPart = Vals;
  return Vals;
}
extern inline f64t16 ripple_ew_pure_modf(f64t16 Vals, f64t16 *FractionalPart) {
  *FractionalPart = Vals;
  return Vals;
}
extern inline f128t8 ripple_ew_pure_modfl(f128t8 Vals, f128t8 *FractionalPart) {
  *FractionalPart = Vals;
  return Vals;
}

extern inline f16t64 ripple_ew_pure_ldexpf16(f16t64 Vals, i32t64 Exp) {
  (void)Exp;
  return Vals;
}
extern inline f32t32 ripple_ew_pure_ldexpf(f32t32 Vals, i32t32 Exp) {
  (void)Exp;
  return Vals;
}
extern inline f64t16 ripple_ew_pure_ldexp(f64t16 Vals, i32t16 Exp) {
  (void)Exp;
  return Vals;
}
extern inline f128t8 ripple_ew_pure_ldexpl(f128t8 Vals, i32t8 Exp) {
  (void)Exp;
  return Vals;
}

extern inline f16t64 ripple_ew_pure_frexpf16(f16t64 Vals, i32t64 *Exp) {
  (void)Exp;
  return Vals;
}
extern inline f32t32 ripple_ew_pure_frexpf(f32t32 Vals, i32t32 *Exp) {
  (void)Exp;
  return Vals;
}
extern inline f64t16 ripple_ew_pure_frexp(f64t16 Vals, i32t16 *Exp) {
  (void)Exp;
  return Vals;
}
extern inline f128t8 ripple_ew_pure_frexpl(f128t8 Vals, i32t8 *Exp) {
  (void)Exp;
  return Vals;
}

// Test for ignoring scalar external function
extern inline int ripple_ew_pure_this_is_not_really_a_ripple_function(int Val) {
  return Val;
}
extern inline int ripple_ew_this_is_not_really_a_ripple_function2(int Val) {
  return Val;
}
extern inline int ripple_this_is_not_really_a_ripple_function3(int Val) {
  return Val;
}
extern inline int ripple_mask_this_is_not_really_a_ripple_function3(int Val) {
  return Val;
}
