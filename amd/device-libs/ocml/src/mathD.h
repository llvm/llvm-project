/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

// OCML prototypes
#include "ocml.h"

// Tables
#include "tables.h"

// Builtins
#include "builtins.h"

// Mangling
#define MATH_MANGLE(N) OCML_MANGLE_F64(N)
#define MATH_PRIVATE(N) MANGLE3(__ocmlpriv,N,f64)

// Optimization Controls
#include "opts.h"

// Attributes
#define PUREATTR __attribute__((pure))
#define CONSTATTR __attribute__((const))

// Math controls
#include "privD.h"

// Bit patterns
#define SIGNBIT_DP64      0x8000000000000000L
#define EXSIGNBIT_DP64    0x7fffffffffffffffL
#define EXPBITS_DP64      0x7ff0000000000000L
#define MANTBITS_DP64     0x000fffffffffffffL
#define ONEEXPBITS_DP64   0x3ff0000000000000L
#define TWOEXPBITS_DP64   0x4000000000000000L
#define HALFEXPBITS_DP64  0x3fe0000000000000L
#define IMPBIT_DP64       0x0010000000000000L
#define QNANBITPATT_DP64  0x7ff8000000000000L
#define INDEFBITPATT_DP64 0xfff8000000000000L
#define PINFBITPATT_DP64  0x7ff0000000000000L
#define NINFBITPATT_DP64  0xfff0000000000000L
#define EXPBIAS_DP64      1023
#define EXPSHIFTBITS_DP64 52
#define BIASEDEMIN_DP64   1
#define EMIN_DP64         -1022
#define BIASEDEMAX_DP64   2046
#define EMAX_DP64         1023
#define LAMBDA_DP64       1.0e300
#define MANTLENGTH_DP64   53
#define BASEDIGITS_DP64   15

#define QNAN_F64 __builtin_nan("")
#define PINF_F64 __builtin_inf()
#define NINF_F64 (-__builtin_inf())
