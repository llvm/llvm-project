#define MAXFLOAT        0x1.fffffep127f
#define HUGE_VALF       __builtin_huge_valf()
#define INFINITY        __builtin_inff()
#define NAN             __builtin_nanf("")

#define FLT_DIG         6
#define FLT_MANT_DIG    24
#define FLT_MAX_10_EXP  +38
#define FLT_MAX_EXP     +128
#define FLT_MIN_10_EXP  -37
#define FLT_MIN_EXP     -125
#define FLT_RADIX       2
#define FLT_MAX         MAXFLOAT
#define FLT_MIN         0x1.0p-126f
#define FLT_EPSILON     0x1.0p-23f

#define FP_ILOGB0 (-2147483647 - 1)
#define FP_ILOGBNAN (-2147483647 - 1)

#define M_E_F           0x1.5bf0a8p+1f
#define M_LOG2E_F       0x1.715476p+0f
#define M_LOG10E_F      0x1.bcb7b2p-2f
#define M_LN2_F         0x1.62e430p-1f
#define M_LN10_F        0x1.26bb1cp+1f
#define M_PI_F          0x1.921fb6p+1f
#define M_PI_2_F        0x1.921fb6p+0f
#define M_PI_4_F        0x1.921fb6p-1f
#define M_1_PI_F        0x1.45f306p-2f
#define M_2_PI_F        0x1.45f306p-1f
#define M_2_SQRTPI_F    0x1.20dd76p+0f
#define M_SQRT2_F       0x1.6a09e6p+0f
#define M_SQRT1_2_F     0x1.6a09e6p-1f

#ifdef __CLC_INTERNAL
#define M_LOG210_F      0x1.a934f0p+1f
#endif

#ifdef cl_khr_fp64

#define HUGE_VAL        __builtin_huge_val()

#define DBL_DIG         15
#define DBL_MANT_DIG    53
#define DBL_MAX_10_EXP  +308
#define DBL_MAX_EXP     +1024
#define DBL_MIN_10_EXP  -307
#define DBL_MIN_EXP     -1021
#define DBL_MAX         0x1.fffffffffffffp1023
#define DBL_MIN         0x1.0p-1022
#define DBL_EPSILON     0x1.0p-52

#define M_E             0x1.5bf0a8b145769p+1
#define M_LOG2E         0x1.71547652b82fep+0
#define M_LOG10E        0x1.bcb7b1526e50ep-2
#define M_LN2           0x1.62e42fefa39efp-1
#define M_LN10          0x1.26bb1bbb55516p+1
#define M_PI            0x1.921fb54442d18p+1
#define M_PI_2          0x1.921fb54442d18p+0
#define M_PI_4          0x1.921fb54442d18p-1
#define M_1_PI          0x1.45f306dc9c883p-2
#define M_2_PI          0x1.45f306dc9c883p-1
#define M_2_SQRTPI      0x1.20dd750429b6dp+0
#define M_SQRT2         0x1.6a09e667f3bcdp+0
#define M_SQRT1_2       0x1.6a09e667f3bcdp-1

#ifdef __CLC_INTERNAL
#define M_LOG210        0x1.a934f0979a371p+1
#endif

#endif

#ifdef cl_khr_fp16

#if __OPENCL_VERSION__ >= 120

#define HALF_DIG        3
#define HALF_MANT_DIG   11
#define HALF_MAX_10_EXP +4
#define HALF_MAX_EXP    +16
#define HALF_MIN_10_EXP -4
#define HALF_MIN_EXP    -13

#define HALF_RADIX      2
#define HALF_MAX        0x1.ffcp15h
#define HALF_MIN        0x1.0p-14h
#define HALF_EPSILON    0x1.0p-10h

#endif

#endif
