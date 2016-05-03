
extern __attribute__((const)) int __ocml_have_fast_fma32(void);
extern __attribute__((const)) int __ocml_have_fast_fma64(void);
extern __attribute__((const)) int __ocml_finite_only_opt(void);
extern __attribute__((const)) int __ocml_fast_relaxed_opt(void);
extern __attribute__((const)) int __ocml_daz_opt(void);
extern __attribute__((const)) int __ocml_amd_opt(void);
extern __attribute__((const)) int __ocml_correctly_rounded_sqrt32(void);

#define HAVE_FAST_FMA32() __ocml_have_fast_fma32()
#define HAVE_FAST_FMA64() __ocml_have_fast_fma64()
#define FINITE_ONLY_OPT() __ocml_finite_only_opt()
#define FAST_RELAXED_OPT() __ocml_fast_relaxed_opt()
#define DAZ_OPT() __ocml_daz_opt()
#define CORRECTLY_ROUNDED_SQRT32() __ocml_correctly_rounded_sqrt32()
#define AMD_OPT() __ocml_amd_opt()

