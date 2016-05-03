
#if defined AMD_BUILD

// Default opt controls for the library for AMD build

extern __attribute__((const)) int __have_fast_fma32(void);

#define MASK_NO_SIGNED_ZEROES 0x01
#define MASK_UNSAFE_MATH_OPTIMIZATIONS 0x02
#define MASK_FINITE_MATH_ONLY 0x04
#define MASK_FAST_RELAXED_MATH 0x08
#define MASK_UNIFORM_WORK_GROUP_SIZE 0x10
#define MASK_DENORMS_ARE_ZERO 0x20
#define MASK_FP32_CORRECTLY_ROUNDED_DIV_SQRT 0x40
extern __attribute__((const)) int __option_mask(void);

#define TESTOPT(flag) ((__option_mask() & (flag))!=0)

__attribute__((always_inline, const, weak)) int __ocml_have_fast_fma32(void) { return __have_fast_fma32(); }
__attribute__((always_inline, const, weak)) int __ocml_have_fast_fma64(void) { return 1; }
__attribute__((always_inline, const, weak)) int __ocml_finite_only_opt(void) { return TESTOPT(MASK_FINITE_MATH_ONLY); }
__attribute__((always_inline, const, weak)) int __ocml_fast_relaxed_opt(void) { return TESTOPT(MASK_FAST_RELAXED_MATH|MASK_UNSAFE_MATH_OPTIMIZATIONS); }
__attribute__((always_inline, const, weak)) int __ocml_daz_opt(void) { return TESTOPT(MASK_DENORMS_ARE_ZERO); }
__attribute__((always_inline, const, weak)) int __ocml_amd_opt(void) { return 1; }
__attribute__((always_inline, const, weak)) int __ocml_correctly_rounded_sqrt32(void) { return TESTOPT(MASK_FP32_CORRECTLY_ROUNDED_DIV_SQRT); }

#endif

