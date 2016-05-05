
// OCML prototypes
#include "../inc/ocml.h"

// Tables
#include "tables.h"

// Builtins
#include "builtins.h"

// Mangling
#define MATH_MANGLE(N) OCML_MANGLE_F16(N)
#define MATH_PRIVATE(N) MANGLE3(__ocmlpriv,N,f16)
#define MATH_UPMANGLE(N) OCML_MANGLE_F32(N)

// Optimization Controls
#include "opts.h"

// Attributes
#define ALIGNEDATTR(X) __attribute__((aligned(X)))
#define INLINEATTR __attribute__((always_inline))
#define PUREATTR __attribute__((pure))
#define CONSTATTR __attribute__((const))

// Math controls
#include "privH.h"

// Floating point patterns
#define SIGNBIT_HP16      0x8000
#define EXSIGNBIT_HP16    0x7fff
#define EXPBITS_HP16      0x7c00
#define MANTBITS_HP16     0x03ff
#define ONEEXPBITS_HP16   0x3c00
#define TWOEXPBITS_HP16   0x4000
#define HALFEXPBITS_HP16  0x3800
#define IMPBIT_HP16       0x0400
#define QNANBITPATT_HP16  0x7e00
#define PINFBITPATT_HP16  0x7c00
#define NINFBITPATT_HP16  0xfc00
#define EXPBIAS_HP16      15
#define EXPSHIFTBITS_HP16 10
#define BIASEDEMIN_HP16   1
#define EMIN_HP16         -14
#define BIASEDEMAX_HP16   30
#define EMAX_HP16         15
#define MANTLENGTH_HP16   11
#define BASEDIGITS_HP16   5

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

