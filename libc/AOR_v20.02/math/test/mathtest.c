/*
 * mathtest.c - test rig for mathlib
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>
#include <ctype.h>
#include <math.h>
#include <errno.h>
#include <limits.h>
#include <fenv.h>
#include "mathlib.h"

#ifndef math_errhandling
# define math_errhandling 0
#endif

#ifdef __cplusplus
 #define EXTERN_C extern "C"
#else
 #define EXTERN_C extern
#endif

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

#ifdef IMPORT_SYMBOL
#define STR2(x) #x
#define STR(x) STR2(x)
_Pragma(STR(import IMPORT_SYMBOL))
#endif

int dmsd, dlsd;
int quiet = 0;
int doround = 0;
unsigned statusmask = FE_ALL_EXCEPT;

#define EXTRABITS (12)
#define ULPUNIT (1<<EXTRABITS)

typedef int (*test) (void);

/*
  struct to hold info about a function (which could actually be a macro)
*/
typedef struct {
    enum {
        t_func, t_macro
    } type;
    enum {
        at_d, at_s,      /* double or single precision float */
        at_d2, at_s2,    /* same, but taking two args */
        at_di, at_si,    /* double/single and an int */
        at_dip, at_sip,  /* double/single and an int ptr */
        at_ddp, at_ssp,  /* d/s and a d/s ptr */
        at_dc, at_sc,    /* double or single precision complex */
        at_dc2, at_sc2   /* same, but taking two args */
    } argtype;
    enum {
        rt_d, rt_s, rt_i, /* double, single, int */
        rt_dc, rt_sc,     /* double, single precision complex */
        rt_d2, rt_s2      /* also use res2 */
    } rettype;
    union {
        void* ptr;
        double (*d_d_ptr)(double);
        float (*s_s_ptr)(float);
        int (*d_i_ptr)(double);
        int (*s_i_ptr)(float);
        double (*d2_d_ptr)(double, double);
        float (*s2_s_ptr)(float, float);
        double (*di_d_ptr)(double,int);
        float (*si_s_ptr)(float,int);
        double (*dip_d_ptr)(double,int*);
        float (*sip_s_ptr)(float,int*);
        double (*ddp_d_ptr)(double,double*);
        float (*ssp_s_ptr)(float,float*);
    } func;
    enum {
        m_none,
        m_isfinite, m_isfinitef,
        m_isgreater, m_isgreaterequal,
        m_isgreaterequalf, m_isgreaterf,
        m_isinf, m_isinff,
        m_isless, m_islessequal,
        m_islessequalf, m_islessf,
        m_islessgreater, m_islessgreaterf,
        m_isnan, m_isnanf,
        m_isnormal, m_isnormalf,
        m_isunordered, m_isunorderedf,
        m_fpclassify, m_fpclassifyf,
        m_signbit, m_signbitf,
        /* not actually a macro, but makes things easier */
        m_rred, m_rredf,
        m_cadd, m_csub, m_cmul, m_cdiv,
        m_caddf, m_csubf, m_cmulf, m_cdivf
    } macro_name; /* only used if a macro/something that can't be done using func */
    long long tolerance;
    const char* name;
} test_func;

/* used in qsort */
int compare_tfuncs(const void* a, const void* b) {
    return strcmp(((test_func*)a)->name, ((test_func*)b)->name);
}

int is_double_argtype(int argtype) {
    switch(argtype) {
    case at_d:
    case at_d2:
    case at_dc:
    case at_dc2:
        return 1;
    default:
        return 0;
    }
}

int is_single_argtype(int argtype) {
    switch(argtype) {
    case at_s:
    case at_s2:
    case at_sc:
    case at_sc2:
        return 1;
    default:
        return 0;
    }
}

int is_double_rettype(int rettype) {
    switch(rettype) {
    case rt_d:
    case rt_dc:
    case rt_d2:
        return 1;
    default:
        return 0;
    }
}

int is_single_rettype(int rettype) {
    switch(rettype) {
    case rt_s:
    case rt_sc:
    case rt_s2:
        return 1;
    default:
        return 0;
    }
}

int is_complex_argtype(int argtype) {
    switch(argtype) {
    case at_dc:
    case at_sc:
    case at_dc2:
    case at_sc2:
        return 1;
    default:
        return 0;
    }
}

int is_complex_rettype(int rettype) {
    switch(rettype) {
    case rt_dc:
    case rt_sc:
        return 1;
    default:
        return 0;
    }
}

/*
 * Special-case flags indicating that some functions' error
 * tolerance handling is more complicated than a fixed relative
 * error bound.
 */
#define ABSLOWERBOUND 0x4000000000000000LL
#define PLUSMINUSPIO2 0x1000000000000000LL

#define ARM_PREFIX(x) x

#define TFUNC(arg,ret,name,tolerance) { t_func, arg, ret, (void*)&name, m_none, tolerance, #name }
#define TFUNCARM(arg,ret,name,tolerance) { t_func, arg, ret, (void*)& ARM_PREFIX(name), m_none, tolerance, #name }
#define MFUNC(arg,ret,name,tolerance) { t_macro, arg, ret, NULL, m_##name, tolerance, #name }

/* sincosf wrappers for easier testing.  */
static float sincosf_sinf(float x) { float s,c; sincosf(x, &s, &c); return s; }
static float sincosf_cosf(float x) { float s,c; sincosf(x, &s, &c); return c; }

test_func tfuncs[] = {
    /* trigonometric */
    TFUNC(at_d,rt_d, acos, 4*ULPUNIT),
    TFUNC(at_d,rt_d, asin, 4*ULPUNIT),
    TFUNC(at_d,rt_d, atan, 4*ULPUNIT),
    TFUNC(at_d2,rt_d, atan2, 4*ULPUNIT),

    TFUNC(at_d,rt_d, tan, 2*ULPUNIT),
    TFUNC(at_d,rt_d, sin, 2*ULPUNIT),
    TFUNC(at_d,rt_d, cos, 2*ULPUNIT),

    TFUNC(at_s,rt_s, acosf, 4*ULPUNIT),
    TFUNC(at_s,rt_s, asinf, 4*ULPUNIT),
    TFUNC(at_s,rt_s, atanf, 4*ULPUNIT),
    TFUNC(at_s2,rt_s, atan2f, 4*ULPUNIT),
    TFUNCARM(at_s,rt_s, tanf, 4*ULPUNIT),
    TFUNCARM(at_s,rt_s, sinf, 3*ULPUNIT/4),
    TFUNCARM(at_s,rt_s, cosf, 3*ULPUNIT/4),
    TFUNCARM(at_s,rt_s, sincosf_sinf, 3*ULPUNIT/4),
    TFUNCARM(at_s,rt_s, sincosf_cosf, 3*ULPUNIT/4),

    /* hyperbolic */
    TFUNC(at_d, rt_d, atanh, 4*ULPUNIT),
    TFUNC(at_d, rt_d, asinh, 4*ULPUNIT),
    TFUNC(at_d, rt_d, acosh, 4*ULPUNIT),
    TFUNC(at_d,rt_d, tanh, 4*ULPUNIT),
    TFUNC(at_d,rt_d, sinh, 4*ULPUNIT),
    TFUNC(at_d,rt_d, cosh, 4*ULPUNIT),

    TFUNC(at_s, rt_s, atanhf, 4*ULPUNIT),
    TFUNC(at_s, rt_s, asinhf, 4*ULPUNIT),
    TFUNC(at_s, rt_s, acoshf, 4*ULPUNIT),
    TFUNC(at_s,rt_s, tanhf, 4*ULPUNIT),
    TFUNC(at_s,rt_s, sinhf, 4*ULPUNIT),
    TFUNC(at_s,rt_s, coshf, 4*ULPUNIT),

    /* exponential and logarithmic */
    TFUNC(at_d,rt_d, log, 3*ULPUNIT/4),
    TFUNC(at_d,rt_d, log10, 3*ULPUNIT),
    TFUNC(at_d,rt_d, log2, 3*ULPUNIT/4),
    TFUNC(at_d,rt_d, log1p, 2*ULPUNIT),
    TFUNC(at_d,rt_d, exp, 3*ULPUNIT/4),
    TFUNC(at_d,rt_d, exp2, 3*ULPUNIT/4),
    TFUNC(at_d,rt_d, expm1, ULPUNIT),
    TFUNCARM(at_s,rt_s, logf, ULPUNIT),
    TFUNC(at_s,rt_s, log10f, 3*ULPUNIT),
    TFUNCARM(at_s,rt_s, log2f, ULPUNIT),
    TFUNC(at_s,rt_s, log1pf, 2*ULPUNIT),
    TFUNCARM(at_s,rt_s, expf, 3*ULPUNIT/4),
    TFUNCARM(at_s,rt_s, exp2f, 3*ULPUNIT/4),
    TFUNC(at_s,rt_s, expm1f, ULPUNIT),

    /* power */
    TFUNC(at_d2,rt_d, pow, 3*ULPUNIT/4),
    TFUNC(at_d,rt_d, sqrt, ULPUNIT/2),
    TFUNC(at_d,rt_d, cbrt, 2*ULPUNIT),
    TFUNC(at_d2, rt_d, hypot, 4*ULPUNIT),

    TFUNCARM(at_s2,rt_s, powf, ULPUNIT),
    TFUNC(at_s,rt_s, sqrtf, ULPUNIT/2),
    TFUNC(at_s,rt_s, cbrtf, 2*ULPUNIT),
    TFUNC(at_s2, rt_s, hypotf, 4*ULPUNIT),

    /* error function */
    TFUNC(at_d,rt_d, erf, 16*ULPUNIT),
    TFUNC(at_s,rt_s, erff, 16*ULPUNIT),
    TFUNC(at_d,rt_d, erfc, 16*ULPUNIT),
    TFUNC(at_s,rt_s, erfcf, 16*ULPUNIT),

    /* gamma functions */
    TFUNC(at_d,rt_d, tgamma, 16*ULPUNIT),
    TFUNC(at_s,rt_s, tgammaf, 16*ULPUNIT),
    TFUNC(at_d,rt_d, lgamma, 16*ULPUNIT | ABSLOWERBOUND),
    TFUNC(at_s,rt_s, lgammaf, 16*ULPUNIT | ABSLOWERBOUND),

    TFUNC(at_d,rt_d, ceil, 0),
    TFUNC(at_s,rt_s, ceilf, 0),
    TFUNC(at_d2,rt_d, copysign, 0),
    TFUNC(at_s2,rt_s, copysignf, 0),
    TFUNC(at_d,rt_d, floor, 0),
    TFUNC(at_s,rt_s, floorf, 0),
    TFUNC(at_d2,rt_d, fmax, 0),
    TFUNC(at_s2,rt_s, fmaxf, 0),
    TFUNC(at_d2,rt_d, fmin, 0),
    TFUNC(at_s2,rt_s, fminf, 0),
    TFUNC(at_d2,rt_d, fmod, 0),
    TFUNC(at_s2,rt_s, fmodf, 0),
    MFUNC(at_d, rt_i, fpclassify, 0),
    MFUNC(at_s, rt_i, fpclassifyf, 0),
    TFUNC(at_dip,rt_d, frexp, 0),
    TFUNC(at_sip,rt_s, frexpf, 0),
    MFUNC(at_d, rt_i, isfinite, 0),
    MFUNC(at_s, rt_i, isfinitef, 0),
    MFUNC(at_d, rt_i, isgreater, 0),
    MFUNC(at_d, rt_i, isgreaterequal, 0),
    MFUNC(at_s, rt_i, isgreaterequalf, 0),
    MFUNC(at_s, rt_i, isgreaterf, 0),
    MFUNC(at_d, rt_i, isinf, 0),
    MFUNC(at_s, rt_i, isinff, 0),
    MFUNC(at_d, rt_i, isless, 0),
    MFUNC(at_d, rt_i, islessequal, 0),
    MFUNC(at_s, rt_i, islessequalf, 0),
    MFUNC(at_s, rt_i, islessf, 0),
    MFUNC(at_d, rt_i, islessgreater, 0),
    MFUNC(at_s, rt_i, islessgreaterf, 0),
    MFUNC(at_d, rt_i, isnan, 0),
    MFUNC(at_s, rt_i, isnanf, 0),
    MFUNC(at_d, rt_i, isnormal, 0),
    MFUNC(at_s, rt_i, isnormalf, 0),
    MFUNC(at_d, rt_i, isunordered, 0),
    MFUNC(at_s, rt_i, isunorderedf, 0),
    TFUNC(at_di,rt_d, ldexp, 0),
    TFUNC(at_si,rt_s, ldexpf, 0),
    TFUNC(at_ddp,rt_d2, modf, 0),
    TFUNC(at_ssp,rt_s2, modff, 0),
#ifndef BIGRANGERED
    MFUNC(at_d, rt_d, rred, 2*ULPUNIT),
#else
    MFUNC(at_d, rt_d, m_rred, ULPUNIT),
#endif
    MFUNC(at_d, rt_i, signbit, 0),
    MFUNC(at_s, rt_i, signbitf, 0),
};

/*
 * keywords are: func size op1 op2 result res2 errno op1r op1i op2r op2i resultr resulti
 * also we ignore: wrongresult wrongres2 wrongerrno
 * op1 equivalent to op1r, same with op2 and result
 */

typedef struct {
    test_func *func;
    unsigned op1r[2]; /* real part, also used for non-complex numbers */
    unsigned op1i[2]; /* imaginary part */
    unsigned op2r[2];
    unsigned op2i[2];
    unsigned resultr[3];
    unsigned resulti[3];
    enum {
        rc_none, rc_zero, rc_infinity, rc_nan, rc_finite
    } resultc; /* special complex results, rc_none means use resultr and resulti as normal */
    unsigned res2[2];
    unsigned status;                   /* IEEE status return, if any */
    unsigned maybestatus;             /* for optional status, or allowance for spurious */
    int nresult;                       /* number of result words */
    int in_err, in_err_limit;
    int err;
    int maybeerr;
    int valid;
    int comment;
    int random;
} testdetail;

enum {                                 /* keywords */
    k_errno, k_errno_in, k_error, k_func, k_maybeerror, k_maybestatus, k_op1, k_op1i, k_op1r, k_op2, k_op2i, k_op2r,
    k_random, k_res2, k_result, k_resultc, k_resulti, k_resultr, k_status,
    k_wrongres2, k_wrongresult, k_wrongstatus, k_wrongerrno
};
char *keywords[] = {
    "errno", "errno_in", "error", "func", "maybeerror", "maybestatus", "op1", "op1i", "op1r", "op2", "op2i", "op2r",
    "random", "res2", "result", "resultc", "resulti", "resultr", "status",
    "wrongres2", "wrongresult", "wrongstatus", "wrongerrno"
};

enum {
    e_0, e_EDOM, e_ERANGE,

    /*
     * This enum makes sure that we have the right number of errnos in the
     * errno[] array
     */
    e_number_of_errnos
};
char *errnos[] = {
    "0", "EDOM", "ERANGE"
};

enum {
    e_none, e_divbyzero, e_domain, e_overflow, e_underflow
};
char *errors[] = {
    "0", "divbyzero", "domain", "overflow", "underflow"
};

static int verbose, fo, strict;

/* state toggled by random=on / random=off */
static int randomstate;

/* Canonify a double NaN: SNaNs all become 7FF00000.00000001 and QNaNs
 * all become 7FF80000.00000001 */
void canon_dNaN(unsigned a[2]) {
    if ((a[0] & 0x7FF00000) != 0x7FF00000)
        return;                        /* not Inf or NaN */
    if (!(a[0] & 0xFFFFF) && !a[1])
        return;                        /* Inf */
    a[0] &= 0x7FF80000;                /* canonify top word */
    a[1] = 0x00000001;                 /* canonify bottom word */
}

/* Canonify a single NaN: SNaNs all become 7F800001 and QNaNs
 * all become 7FC00001. Returns classification of the NaN. */
void canon_sNaN(unsigned a[1]) {
    if ((a[0] & 0x7F800000) != 0x7F800000)
        return;                        /* not Inf or NaN */
    if (!(a[0] & 0x7FFFFF))
        return;                        /* Inf */
    a[0] &= 0x7FC00000;                /* canonify most bits */
    a[0] |= 0x00000001;                /* canonify bottom bit */
}

/*
 * Detect difficult operands for FO mode.
 */
int is_dhard(unsigned a[2])
{
    if ((a[0] & 0x7FF00000) == 0x7FF00000)
        return TRUE;                   /* inf or NaN */
    if ((a[0] & 0x7FF00000) == 0 &&
        ((a[0] & 0x7FFFFFFF) | a[1]) != 0)
        return TRUE;                   /* denormal */
    return FALSE;
}
int is_shard(unsigned a[1])
{
    if ((a[0] & 0x7F800000) == 0x7F800000)
        return TRUE;                   /* inf or NaN */
    if ((a[0] & 0x7F800000) == 0 &&
        (a[0] & 0x7FFFFFFF) != 0)
        return TRUE;                   /* denormal */
    return FALSE;
}

/*
 * Normalise all zeroes into +0, for FO mode.
 */
void dnormzero(unsigned a[2])
{
    if (a[0] == 0x80000000 && a[1] == 0)
        a[0] = 0;
}
void snormzero(unsigned a[1])
{
    if (a[0] == 0x80000000)
        a[0] = 0;
}

static int find(char *word, char **array, int asize) {
    int i, j;

    asize /= sizeof(char *);

    i = -1; j = asize;                 /* strictly between i and j */
    while (j-i > 1) {
        int k = (i+j) / 2;
        int c = strcmp(word, array[k]);
        if (c > 0)
            i = k;
        else if (c < 0)
            j = k;
        else                           /* found it! */
            return k;
    }
    return -1;                         /* not found */
}

static test_func* find_testfunc(char *word) {
    int i, j, asize;

    asize = sizeof(tfuncs)/sizeof(test_func);

    i = -1; j = asize;                 /* strictly between i and j */
    while (j-i > 1) {
        int k = (i+j) / 2;
        int c = strcmp(word, tfuncs[k].name);
        if (c > 0)
            i = k;
        else if (c < 0)
            j = k;
        else                           /* found it! */
            return tfuncs + k;
    }
    return NULL;                         /* not found */
}

static long long calc_error(unsigned a[2], unsigned b[3], int shift, int rettype) {
    unsigned r0, r1, r2;
    int sign, carry;
    long long result;

    /*
     * If either number is infinite, require exact equality. If
     * either number is NaN, require that both are NaN. If either
     * of these requirements is broken, return INT_MAX.
     */
    if (is_double_rettype(rettype)) {
        if ((a[0] & 0x7FF00000) == 0x7FF00000 ||
            (b[0] & 0x7FF00000) == 0x7FF00000) {
            if (((a[0] & 0x800FFFFF) || a[1]) &&
                ((b[0] & 0x800FFFFF) || b[1]) &&
                (a[0] & 0x7FF00000) == 0x7FF00000 &&
                (b[0] & 0x7FF00000) == 0x7FF00000)
                return 0;              /* both NaN - OK */
            if (!((a[0] & 0xFFFFF) || a[1]) &&
                !((b[0] & 0xFFFFF) || b[1]) &&
                a[0] == b[0])
                return 0;              /* both same sign of Inf - OK */
            return LLONG_MAX;
        }
    } else {
        if ((a[0] & 0x7F800000) == 0x7F800000 ||
            (b[0] & 0x7F800000) == 0x7F800000) {
            if ((a[0] & 0x807FFFFF) &&
                (b[0] & 0x807FFFFF) &&
                (a[0] & 0x7F800000) == 0x7F800000 &&
                (b[0] & 0x7F800000) == 0x7F800000)
                return 0;              /* both NaN - OK */
            if (!(a[0] & 0x7FFFFF) &&
                !(b[0] & 0x7FFFFF) &&
                a[0] == b[0])
                return 0;              /* both same sign of Inf - OK */
            return LLONG_MAX;
        }
    }

    /*
     * Both finite. Return INT_MAX if the signs differ.
     */
    if ((a[0] ^ b[0]) & 0x80000000)
        return LLONG_MAX;

    /*
     * Now it's just straight multiple-word subtraction.
     */
    if (is_double_rettype(rettype)) {
        r2 = -b[2]; carry = (r2 == 0);
        r1 = a[1] + ~b[1] + carry; carry = (r1 < a[1] || (carry && r1 == a[1]));
        r0 = a[0] + ~b[0] + carry;
    } else {
        r2 = -b[1]; carry = (r2 == 0);
        r1 = a[0] + ~b[0] + carry; carry = (r1 < a[0] || (carry && r1 == a[0]));
        r0 = ~0 + carry;
    }

    /*
     * Forgive larger errors in specialised cases.
     */
    if (shift > 0) {
        if (shift > 32*3)
            return 0;                  /* all errors are forgiven! */
        while (shift >= 32) {
            r2 = r1;
            r1 = r0;
            r0 = -(r0 >> 31);
            shift -= 32;
        }

        if (shift > 0) {
            r2 = (r2 >> shift) | (r1 << (32-shift));
            r1 = (r1 >> shift) | (r0 << (32-shift));
            r0 = (r0 >> shift) | ((-(r0 >> 31)) << (32-shift));
        }
    }

    if (r0 & 0x80000000) {
        sign = 1;
        r2 = ~r2; carry = (r2 == 0);
        r1 = 0 + ~r1 + carry; carry = (carry && (r2 == 0));
        r0 = 0 + ~r0 + carry;
    } else {
        sign = 0;
    }

    if (r0 >= (1LL<<(31-EXTRABITS)))
        return LLONG_MAX;                /* many ulps out */

    result = (r2 >> (32-EXTRABITS)) & (ULPUNIT-1);
    result |= r1 << EXTRABITS;
    result |= (long long)r0 << (32+EXTRABITS);
    if (sign)
        result = -result;
    return result;
}

/* special named operands */

typedef struct {
    unsigned op1, op2;
    char* name;
} special_op;

static special_op special_ops_double[] = {
    {0x00000000,0x00000000,"0"},
    {0x3FF00000,0x00000000,"1"},
    {0x7FF00000,0x00000000,"inf"},
    {0x7FF80000,0x00000001,"qnan"},
    {0x7FF00000,0x00000001,"snan"},
    {0x3ff921fb,0x54442d18,"pi2"},
    {0x400921fb,0x54442d18,"pi"},
    {0x3fe921fb,0x54442d18,"pi4"},
    {0x4002d97c,0x7f3321d2,"3pi4"},
};

static special_op special_ops_float[] = {
    {0x00000000,0,"0"},
    {0x3f800000,0,"1"},
    {0x7f800000,0,"inf"},
    {0x7fc00000,0,"qnan"},
    {0x7f800001,0,"snan"},
    {0x3fc90fdb,0,"pi2"},
    {0x40490fdb,0,"pi"},
    {0x3f490fdb,0,"pi4"},
    {0x4016cbe4,0,"3pi4"},
};

/*
   This is what is returned by the below functions.
   We need it to handle the sign of the number
*/
static special_op tmp_op = {0,0,0};

special_op* find_special_op_from_op(unsigned op1, unsigned op2, int is_double) {
    int i;
    special_op* sop;
    if(is_double) {
        sop = special_ops_double;
    } else {
        sop = special_ops_float;
    }
    for(i = 0; i < sizeof(special_ops_double)/sizeof(special_op); i++) {
        if(sop->op1 == (op1&0x7fffffff) && sop->op2 == op2) {
            if(tmp_op.name) free(tmp_op.name);
            tmp_op.name = malloc(strlen(sop->name)+2);
            if(op1>>31) {
                sprintf(tmp_op.name,"-%s",sop->name);
            } else {
                strcpy(tmp_op.name,sop->name);
            }
            return &tmp_op;
        }
        sop++;
    }
    return NULL;
}

special_op* find_special_op_from_name(const char* name, int is_double) {
    int i, neg=0;
    special_op* sop;
    if(is_double) {
        sop = special_ops_double;
    } else {
        sop = special_ops_float;
    }
    if(*name=='-') {
        neg=1;
        name++;
    } else if(*name=='+') {
        name++;
    }
    for(i = 0; i < sizeof(special_ops_double)/sizeof(special_op); i++) {
        if(0 == strcmp(name,sop->name)) {
            tmp_op.op1 = sop->op1;
            if(neg) {
                tmp_op.op1 |= 0x80000000;
            }
            tmp_op.op2 = sop->op2;
            return &tmp_op;
        }
        sop++;
    }
    return NULL;
}

/*
   helper function for the below
   type=0 for single, 1 for double, 2 for no sop
*/
int do_op(char* q, unsigned* op, const char* name, int num, int sop_type) {
    int i;
    int n=num;
    special_op* sop = NULL;
    for(i = 0; i < num; i++) {
        op[i] = 0;
    }
    if(sop_type<2) {
        sop = find_special_op_from_name(q,sop_type);
    }
    if(sop != NULL) {
        op[0] = sop->op1;
        op[1] = sop->op2;
    } else {
        switch(num) {
        case 1: n = sscanf(q, "%x", &op[0]); break;
        case 2: n = sscanf(q, "%x.%x", &op[0], &op[1]); break;
        case 3: n = sscanf(q, "%x.%x.%x", &op[0], &op[1], &op[2]); break;
        default: return -1;
        }
    }
    if (verbose) {
        printf("%s=",name);
        for (i = 0; (i < n); ++i) printf("%x.", op[i]);
        printf(" (n=%d)\n", n);
    }
    return n;
}

testdetail parsetest(char *testbuf, testdetail oldtest) {
    char *p; /* Current part of line: Option name */
    char *q; /* Current part of line: Option value */
    testdetail ret; /* What we return */
    int k; /* Function enum from k_* */
    int n; /* Used as returns for scanfs */
    int argtype=2, rettype=2; /* for do_op */

    /* clear ret */
    memset(&ret, 0, sizeof(ret));

    if (verbose) printf("Parsing line: %s\n", testbuf);
    while (*testbuf && isspace(*testbuf)) testbuf++;
    if (testbuf[0] == ';' || testbuf[0] == '#' || testbuf[0] == '!' ||
        testbuf[0] == '>' || testbuf[0] == '\0') {
        ret.comment = 1;
        if (verbose) printf("Line is a comment\n");
        return ret;
    }
    ret.comment = 0;

    if (*testbuf == '+') {
        if (oldtest.valid) {
            ret = oldtest;             /* structure copy */
        } else {
            fprintf(stderr, "copy from invalid: ignored\n");
        }
        testbuf++;
    }

    ret.random = randomstate;

    ret.in_err = 0;
    ret.in_err_limit = e_number_of_errnos;

    p = strtok(testbuf, " \t");
    while (p != NULL) {
        q = strchr(p, '=');
        if (!q)
            goto balderdash;
        *q++ = '\0';
        k = find(p, keywords, sizeof(keywords));
        switch (k) {
        case k_random:
            randomstate = (!strcmp(q, "on"));
            ret.comment = 1;
            return ret;                /* otherwise ignore this line */
        case k_func:
            if (verbose) printf("func=%s ", q);
            //ret.func = find(q, funcs, sizeof(funcs));
            ret.func = find_testfunc(q);
            if (ret.func == NULL)
                {
                    if (verbose) printf("(id=unknown)\n");
                    goto balderdash;
                }
            if(is_single_argtype(ret.func->argtype))
                argtype = 0;
            else if(is_double_argtype(ret.func->argtype))
                argtype = 1;
            if(is_single_rettype(ret.func->rettype))
                rettype = 0;
            else if(is_double_rettype(ret.func->rettype))
                rettype = 1;
            //ret.size = sizes[ret.func];
            if (verbose) printf("(name=%s) (size=%d)\n", ret.func->name, ret.func->argtype);
            break;
        case k_op1:
        case k_op1r:
            n = do_op(q,ret.op1r,"op1r",2,argtype);
            if (n < 1)
                goto balderdash;
            break;
        case k_op1i:
            n = do_op(q,ret.op1i,"op1i",2,argtype);
            if (n < 1)
                goto balderdash;
            break;
        case k_op2:
        case k_op2r:
            n = do_op(q,ret.op2r,"op2r",2,argtype);
            if (n < 1)
                goto balderdash;
            break;
        case k_op2i:
            n = do_op(q,ret.op2i,"op2i",2,argtype);
            if (n < 1)
                goto balderdash;
            break;
        case k_resultc:
            puts(q);
            if(strncmp(q,"inf",3)==0) {
                ret.resultc = rc_infinity;
            } else if(strcmp(q,"zero")==0) {
                ret.resultc = rc_zero;
            } else if(strcmp(q,"nan")==0) {
                ret.resultc = rc_nan;
            } else if(strcmp(q,"finite")==0) {
                ret.resultc = rc_finite;
            } else {
                goto balderdash;
            }
            break;
        case k_result:
        case k_resultr:
            n = (do_op)(q,ret.resultr,"resultr",3,rettype);
            if (n < 1)
                goto balderdash;
            ret.nresult = n; /* assume real and imaginary have same no. words */
            break;
        case k_resulti:
            n = do_op(q,ret.resulti,"resulti",3,rettype);
            if (n < 1)
                goto balderdash;
            break;
        case k_res2:
            n = do_op(q,ret.res2,"res2",2,rettype);
            if (n < 1)
                goto balderdash;
            break;
        case k_status:
            while (*q) {
                if (*q == 'i') ret.status |= FE_INVALID;
                if (*q == 'z') ret.status |= FE_DIVBYZERO;
                if (*q == 'o') ret.status |= FE_OVERFLOW;
                if (*q == 'u') ret.status |= FE_UNDERFLOW;
                q++;
            }
            break;
        case k_maybeerror:
            n = find(q, errors, sizeof(errors));
            if (n < 0)
                goto balderdash;
            if(math_errhandling&MATH_ERREXCEPT) {
                switch(n) {
                case e_domain: ret.maybestatus |= FE_INVALID; break;
                case e_divbyzero: ret.maybestatus |= FE_DIVBYZERO; break;
                case e_overflow: ret.maybestatus |= FE_OVERFLOW; break;
                case e_underflow: ret.maybestatus |= FE_UNDERFLOW; break;
                }
            }
            {
                switch(n) {
                case e_domain:
                    ret.maybeerr = e_EDOM; break;
                case e_divbyzero:
                case e_overflow:
                case e_underflow:
                    ret.maybeerr = e_ERANGE; break;
                }
            }
        case k_maybestatus:
            while (*q) {
                if (*q == 'i') ret.maybestatus |= FE_INVALID;
                if (*q == 'z') ret.maybestatus |= FE_DIVBYZERO;
                if (*q == 'o') ret.maybestatus |= FE_OVERFLOW;
                if (*q == 'u') ret.maybestatus |= FE_UNDERFLOW;
                q++;
            }
            break;
        case k_error:
            n = find(q, errors, sizeof(errors));
            if (n < 0)
                goto balderdash;
            if(math_errhandling&MATH_ERREXCEPT) {
                switch(n) {
                case e_domain: ret.status |= FE_INVALID; break;
                case e_divbyzero: ret.status |= FE_DIVBYZERO; break;
                case e_overflow: ret.status |= FE_OVERFLOW; break;
                case e_underflow: ret.status |= FE_UNDERFLOW; break;
                }
            }
            if(math_errhandling&MATH_ERRNO) {
                switch(n) {
                case e_domain:
                    ret.err = e_EDOM; break;
                case e_divbyzero:
                case e_overflow:
                case e_underflow:
                    ret.err = e_ERANGE; break;
                }
            }
            if(!(math_errhandling&MATH_ERRNO)) {
                switch(n) {
                case e_domain:
                    ret.maybeerr = e_EDOM; break;
                case e_divbyzero:
                case e_overflow:
                case e_underflow:
                    ret.maybeerr = e_ERANGE; break;
                }
            }
            break;
        case k_errno:
            ret.err = find(q, errnos, sizeof(errnos));
            if (ret.err < 0)
                goto balderdash;
            break;
        case k_errno_in:
            ret.in_err = find(q, errnos, sizeof(errnos));
            if (ret.err < 0)
                goto balderdash;
            ret.in_err_limit = ret.in_err + 1;
            break;
        case k_wrongresult:
        case k_wrongstatus:
        case k_wrongres2:
        case k_wrongerrno:
            /* quietly ignore these keys */
            break;
        default:
            goto balderdash;
        }
        p = strtok(NULL, " \t");
    }
    ret.valid = 1;
    return ret;

    /* come here from almost any error */
 balderdash:
    ret.valid = 0;
    return ret;
}

typedef enum {
    test_comment,                      /* deliberately not a test */
    test_invalid,                      /* accidentally not a test */
    test_decline,                      /* was a test, and wasn't run */
    test_fail,                         /* was a test, and failed */
    test_pass                          /* was a test, and passed */
} testresult;

char failtext[512];

typedef union {
    unsigned i[2];
    double f;
    double da[2];
} dbl;

typedef union {
    unsigned i;
    float f;
    float da[2];
} sgl;

/* helper function for runtest */
void print_error(int rettype, unsigned *result, char* text, char** failp) {
    special_op *sop;
    char *str;

    if(result) {
        *failp += sprintf(*failp," %s=",text);
        sop = find_special_op_from_op(result[0],result[1],is_double_rettype(rettype));
        if(sop) {
            *failp += sprintf(*failp,"%s",sop->name);
        } else {
            if(is_double_rettype(rettype)) {
                str="%08x.%08x";
            } else {
                str="%08x";
            }
            *failp += sprintf(*failp,str,result[0],result[1]);
        }
    }
}


void print_ulps_helper(const char *name, long long ulps, char** failp) {
    if(ulps == LLONG_MAX) {
        *failp += sprintf(*failp, " %s=HUGE", name);
    } else {
        *failp += sprintf(*failp, " %s=%.3f", name, (double)ulps / ULPUNIT);
    }
}

/* for complex args make ulpsr or ulpsri = 0 to not print */
void print_ulps(int rettype, long long ulpsr, long long ulpsi, char** failp) {
    if(is_complex_rettype(rettype)) {
        if (ulpsr) print_ulps_helper("ulpsr",ulpsr,failp);
        if (ulpsi) print_ulps_helper("ulpsi",ulpsi,failp);
    } else {
        if (ulpsr) print_ulps_helper("ulps",ulpsr,failp);
    }
}

int runtest(testdetail t) {
    int err, status;

    dbl d_arg1, d_arg2, d_res, d_res2;
    sgl s_arg1, s_arg2, s_res, s_res2;

    int deferred_decline = FALSE;
    char *failp = failtext;

    unsigned int intres=0;

    int res2_adjust = 0;

    if (t.comment)
        return test_comment;
    if (!t.valid)
        return test_invalid;

    /* Set IEEE status to mathlib-normal */
    feclearexcept(FE_ALL_EXCEPT);

    /* Deal with operands */
#define DO_DOP(arg,op) arg.i[dmsd] = t.op[0]; arg.i[dlsd] = t.op[1]
    DO_DOP(d_arg1,op1r);
    DO_DOP(d_arg2,op2r);
    s_arg1.i = t.op1r[0]; s_arg2.i = t.op2r[0];

    /*
     * Detect NaNs, infinities and denormals on input, and set a
     * deferred decline flag if we're in FO mode.
     *
     * (We defer the decline rather than doing it immediately
     * because even in FO mode the operation is not permitted to
     * crash or tight-loop; so we _run_ the test, and then ignore
     * all the results.)
     */
    if (fo) {
        if (is_double_argtype(t.func->argtype) && is_dhard(t.op1r))
            deferred_decline = TRUE;
        if (t.func->argtype==at_d2 && is_dhard(t.op2r))
            deferred_decline = TRUE;
        if (is_single_argtype(t.func->argtype) && is_shard(t.op1r))
            deferred_decline = TRUE;
        if (t.func->argtype==at_s2 && is_shard(t.op2r))
            deferred_decline = TRUE;
        if (is_double_rettype(t.func->rettype) && is_dhard(t.resultr))
            deferred_decline = TRUE;
        if (t.func->rettype==rt_d2 && is_dhard(t.res2))
            deferred_decline = TRUE;
        if (is_single_argtype(t.func->rettype) && is_shard(t.resultr))
            deferred_decline = TRUE;
        if (t.func->rettype==rt_s2 && is_shard(t.res2))
            deferred_decline = TRUE;
        if (t.err == e_ERANGE)
            deferred_decline = TRUE;
    }

    /*
     * Perform the operation
     */

    errno = t.in_err == e_EDOM ? EDOM : t.in_err == e_ERANGE ? ERANGE : 0;
    if (t.err == e_0)
        t.err = t.in_err;
    if (t.maybeerr == e_0)
        t.maybeerr = t.in_err;

    if(t.func->type == t_func) {
        switch(t.func->argtype) {
        case at_d: d_res.f = t.func->func.d_d_ptr(d_arg1.f); break;
        case at_s: s_res.f = t.func->func.s_s_ptr(s_arg1.f); break;
        case at_d2: d_res.f = t.func->func.d2_d_ptr(d_arg1.f, d_arg2.f); break;
        case at_s2: s_res.f = t.func->func.s2_s_ptr(s_arg1.f, s_arg2.f); break;
        case at_di: d_res.f = t.func->func.di_d_ptr(d_arg1.f, d_arg2.i[dmsd]); break;
        case at_si: s_res.f = t.func->func.si_s_ptr(s_arg1.f, s_arg2.i); break;
        case at_dip: d_res.f = t.func->func.dip_d_ptr(d_arg1.f, (int*)&intres); break;
        case at_sip: s_res.f = t.func->func.sip_s_ptr(s_arg1.f, (int*)&intres); break;
        case at_ddp: d_res.f = t.func->func.ddp_d_ptr(d_arg1.f, &d_res2.f); break;
        case at_ssp: s_res.f = t.func->func.ssp_s_ptr(s_arg1.f, &s_res2.f); break;
        default:
            printf("unhandled function: %s\n",t.func->name);
            return test_fail;
        }
    } else {
        /* printf("macro: name=%s, num=%i, s1.i=0x%08x s1.f=%f\n",t.func->name, t.func->macro_name, s_arg1.i, (double)s_arg1.f); */
        switch(t.func->macro_name) {
        case m_isfinite: intres = isfinite(d_arg1.f); break;
        case m_isinf: intres = isinf(d_arg1.f); break;
        case m_isnan: intres = isnan(d_arg1.f); break;
        case m_isnormal: intres = isnormal(d_arg1.f); break;
        case m_signbit: intres = signbit(d_arg1.f); break;
        case m_fpclassify: intres = fpclassify(d_arg1.f); break;
        case m_isgreater: intres = isgreater(d_arg1.f, d_arg2.f); break;
        case m_isgreaterequal: intres = isgreaterequal(d_arg1.f, d_arg2.f); break;
        case m_isless: intres = isless(d_arg1.f, d_arg2.f); break;
        case m_islessequal: intres = islessequal(d_arg1.f, d_arg2.f); break;
        case m_islessgreater: intres = islessgreater(d_arg1.f, d_arg2.f); break;
        case m_isunordered: intres = isunordered(d_arg1.f, d_arg2.f); break;

        case m_isfinitef: intres = isfinite(s_arg1.f); break;
        case m_isinff: intres = isinf(s_arg1.f); break;
        case m_isnanf: intres = isnan(s_arg1.f); break;
        case m_isnormalf: intres = isnormal(s_arg1.f); break;
        case m_signbitf: intres = signbit(s_arg1.f); break;
        case m_fpclassifyf: intres = fpclassify(s_arg1.f); break;
        case m_isgreaterf: intres = isgreater(s_arg1.f, s_arg2.f); break;
        case m_isgreaterequalf: intres = isgreaterequal(s_arg1.f, s_arg2.f); break;
        case m_islessf: intres = isless(s_arg1.f, s_arg2.f); break;
        case m_islessequalf: intres = islessequal(s_arg1.f, s_arg2.f); break;
        case m_islessgreaterf: intres = islessgreater(s_arg1.f, s_arg2.f); break;
        case m_isunorderedf: intres = isunordered(s_arg1.f, s_arg2.f); break;

        default:
            printf("unhandled macro: %s\n",t.func->name);
            return test_fail;
        }
    }

    /*
     * Decline the test if the deferred decline flag was set above.
     */
    if (deferred_decline)
        return test_decline;

    /* printf("intres=%i\n",intres); */

    /* Clear the fail text (indicating a pass unless we change it) */
    failp[0] = '\0';

    /* Check the IEEE status bits (except INX, which we disregard).
     * We don't bother with this for complex numbers, because the
     * complex functions are hard to get exactly right and we don't
     * have to anyway (C99 annex G is only informative). */
    if (!(is_complex_argtype(t.func->argtype) || is_complex_rettype(t.func->rettype))) {
        status = fetestexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW);
        if ((status|t.maybestatus|~statusmask) != (t.status|t.maybestatus|~statusmask)) {
            if (quiet) failtext[0]='x';
            else {
                failp += sprintf(failp,
                                 " wrongstatus=%s%s%s%s%s",
                                 (status & FE_INVALID ? "i" : ""),
                                 (status & FE_DIVBYZERO ? "z" : ""),
                                 (status & FE_OVERFLOW ? "o" : ""),
                                 (status & FE_UNDERFLOW ? "u" : ""),
                                 (status ? "" : "OK"));
            }
        }
    }

    /* Check the result */
    {
        unsigned resultr[2], resulti[2];
        unsigned tresultr[3], tresulti[3], wres;

        switch(t.func->rettype) {
        case rt_d:
        case rt_d2:
            tresultr[0] = t.resultr[0];
            tresultr[1] = t.resultr[1];
            resultr[0] = d_res.i[dmsd]; resultr[1] = d_res.i[dlsd];
            wres = 2;
            break;
        case rt_i:
            tresultr[0] = t.resultr[0];
            resultr[0] = intres;
            wres = 1;
            break;
        case rt_s:
        case rt_s2:
            tresultr[0] = t.resultr[0];
            resultr[0] = s_res.i;
            wres = 1;
            break;
        default:
            puts("unhandled rettype in runtest");
            wres = 0;
        }
        if(t.resultc != rc_none) {
            int err = 0;
            switch(t.resultc) {
            case rc_zero:
                if(resultr[0] != 0 || resulti[0] != 0 ||
                   (wres==2 && (resultr[1] != 0 || resulti[1] != 0))) {
                    err = 1;
                }
                break;
            case rc_infinity:
                if(wres==1) {
                    if(!((resultr[0]&0x7fffffff)==0x7f800000 ||
                         (resulti[0]&0x7fffffff)==0x7f800000)) {
                        err = 1;
                    }
                } else {
                  if(!(((resultr[0]&0x7fffffff)==0x7ff00000 && resultr[1]==0) ||
                       ((resulti[0]&0x7fffffff)==0x7ff00000 && resulti[1]==0))) {
                        err = 1;
                    }
                }
                break;
            case rc_nan:
                if(wres==1) {
                    if(!((resultr[0]&0x7fffffff)>0x7f800000 ||
                         (resulti[0]&0x7fffffff)>0x7f800000)) {
                        err = 1;
                    }
                } else {
                    canon_dNaN(resultr);
                    canon_dNaN(resulti);
                    if(!(((resultr[0]&0x7fffffff)>0x7ff00000 && resultr[1]==1) ||
                         ((resulti[0]&0x7fffffff)>0x7ff00000 && resulti[1]==1))) {
                        err = 1;
                    }
                }
                break;
            case rc_finite:
                if(wres==1) {
                    if(!((resultr[0]&0x7fffffff)<0x7f800000 ||
                         (resulti[0]&0x7fffffff)<0x7f800000)) {
                        err = 1;
                    }
                } else {
                    if(!((resultr[0]&0x7fffffff)<0x7ff00000 ||
                         (resulti[0]&0x7fffffff)<0x7ff00000)) {
                        err = 1;
                    }
                }
                break;
            default:
                break;
            }
            if(err) {
                print_error(t.func->rettype,resultr,"wrongresultr",&failp);
                print_error(t.func->rettype,resulti,"wrongresulti",&failp);
            }
        } else if (t.nresult > wres) {
            /*
             * The test case data has provided the result to more
             * than double precision. Instead of testing exact
             * equality, we test against our maximum error
             * tolerance.
             */
            int rshift, ishift;
            long long ulpsr, ulpsi, ulptolerance;

            tresultr[wres] = t.resultr[wres] << (32-EXTRABITS);
            tresulti[wres] = t.resulti[wres] << (32-EXTRABITS);
            if(strict) {
                ulptolerance = 4096; /* one ulp */
            } else {
                ulptolerance = t.func->tolerance;
            }
            rshift = ishift = 0;
            if (ulptolerance & ABSLOWERBOUND) {
                /*
                 * Hack for the lgamma functions, which have an
                 * error behaviour that can't conveniently be
                 * characterised in pure ULPs. Really, we want to
                 * say that the error in lgamma is "at most N ULPs,
                 * or at most an absolute error of X, whichever is
                 * larger", for appropriately chosen N,X. But since
                 * these two functions are the only cases where it
                 * arises, I haven't bothered to do it in a nice way
                 * in the function table above.
                 *
                 * (The difficult cases arise with negative input
                 * values such that |gamma(x)| is very near to 1; in
                 * this situation implementations tend to separately
                 * compute lgamma(|x|) and the log of the correction
                 * term from the Euler reflection formula, and
                 * subtract - which catastrophically loses
                 * significance.)
                 *
                 * As far as I can tell, nobody cares about this:
                 * GNU libm doesn't get those cases right either,
                 * and OpenCL explicitly doesn't state a ULP error
                 * limit for lgamma. So my guess is that this is
                 * simply considered acceptable error behaviour for
                 * this particular function, and hence I feel free
                 * to allow for it here.
                 */
                ulptolerance &= ~ABSLOWERBOUND;
                if (t.op1r[0] & 0x80000000) {
                    if (t.func->rettype == rt_d)
                        rshift = 0x400 - ((tresultr[0] >> 20) & 0x7ff);
                    else if (t.func->rettype == rt_s)
                        rshift = 0x80 - ((tresultr[0] >> 23) & 0xff);
                    if (rshift < 0)
                        rshift = 0;
                }
            }
            if (ulptolerance & PLUSMINUSPIO2) {
                ulptolerance &= ~PLUSMINUSPIO2;
                /*
                 * Hack for range reduction, which can reduce
                 * borderline cases in the wrong direction, i.e.
                 * return a value just outside one end of the interval
                 * [-pi/4,+pi/4] when it could have returned a value
                 * just inside the other end by subtracting an
                 * adjacent multiple of pi/2.
                 *
                 * We tolerate this, up to a point, because the
                 * trigonometric functions making use of the output of
                 * rred can cope and because making the range reducer
                 * do the exactly right thing in every case would be
                 * more expensive.
                 */
                if (wres == 1) {
                    /* Upper bound of overshoot derived in rredf.h */
                    if ((resultr[0]&0x7FFFFFFF) <= 0x3f494b02 &&
                        (resultr[0]&0x7FFFFFFF) > 0x3f490fda &&
                        (resultr[0]&0x80000000) != (tresultr[0]&0x80000000)) {
                        unsigned long long val;
                        val = tresultr[0];
                        val = (val << 32) | tresultr[1];
                        /*
                         * Compute the alternative permitted result by
                         * subtracting from the sum of the extended
                         * single-precision bit patterns of +pi/4 and
                         * -pi/4. This is a horrible hack which only
                         * works because we can be confident that
                         * numbers in this range all have the same
                         * exponent!
                         */
                        val = 0xfe921fb54442d184ULL - val;
                        tresultr[0] = val >> 32;
                        tresultr[1] = (val >> (32-EXTRABITS)) << (32-EXTRABITS);
                        /*
                         * Also, expect a correspondingly different
                         * value of res2 as a result of this change.
                         * The adjustment depends on whether we just
                         * flipped the result from + to - or vice
                         * versa.
                         */
                        if (resultr[0] & 0x80000000) {
                            res2_adjust = +1;
                        } else {
                            res2_adjust = -1;
                        }
                    }
                }
            }
            ulpsr = calc_error(resultr, tresultr, rshift, t.func->rettype);
            if(is_complex_rettype(t.func->rettype)) {
                ulpsi = calc_error(resulti, tresulti, ishift, t.func->rettype);
            } else {
                ulpsi = 0;
            }
            unsigned *rr = (ulpsr > ulptolerance || ulpsr < -ulptolerance) ? resultr : NULL;
            unsigned *ri = (ulpsi > ulptolerance || ulpsi < -ulptolerance) ? resulti : NULL;
/*             printf("tolerance=%i, ulpsr=%i, ulpsi=%i, rr=%p, ri=%p\n",ulptolerance,ulpsr,ulpsi,rr,ri); */
            if (rr || ri) {
                if (quiet) failtext[0]='x';
                else {
                    print_error(t.func->rettype,rr,"wrongresultr",&failp);
                    print_error(t.func->rettype,ri,"wrongresulti",&failp);
                    print_ulps(t.func->rettype,rr ? ulpsr : 0, ri ? ulpsi : 0,&failp);
                }
            }
        } else {
            if(is_complex_rettype(t.func->rettype))
                /*
                 * Complex functions are not fully supported,
                 * this is unreachable, but prevents warnings.
                 */
                abort();
            /*
             * The test case data has provided the result in
             * exactly the output precision. Therefore we must
             * complain about _any_ violation.
             */
            switch(t.func->rettype) {
            case rt_dc:
                canon_dNaN(tresulti);
                canon_dNaN(resulti);
                if (fo) {
                    dnormzero(tresulti);
                    dnormzero(resulti);
                }
                /* deliberate fall-through */
            case rt_d:
                canon_dNaN(tresultr);
                canon_dNaN(resultr);
                if (fo) {
                    dnormzero(tresultr);
                    dnormzero(resultr);
                }
                break;
            case rt_sc:
                canon_sNaN(tresulti);
                canon_sNaN(resulti);
                if (fo) {
                    snormzero(tresulti);
                    snormzero(resulti);
                }
                /* deliberate fall-through */
            case rt_s:
                canon_sNaN(tresultr);
                canon_sNaN(resultr);
                if (fo) {
                    snormzero(tresultr);
                    snormzero(resultr);
                }
                break;
            default:
                break;
            }
            if(is_complex_rettype(t.func->rettype)) {
                unsigned *rr, *ri;
                if(resultr[0] != tresultr[0] ||
                   (wres > 1 && resultr[1] != tresultr[1])) {
                    rr = resultr;
                } else {
                    rr = NULL;
                }
                if(resulti[0] != tresulti[0] ||
                   (wres > 1 && resulti[1] != tresulti[1])) {
                    ri = resulti;
                } else {
                    ri = NULL;
                }
                if(rr || ri) {
                    if (quiet) failtext[0]='x';
                    print_error(t.func->rettype,rr,"wrongresultr",&failp);
                    print_error(t.func->rettype,ri,"wrongresulti",&failp);
                }
            } else if (resultr[0] != tresultr[0] ||
                       (wres > 1 && resultr[1] != tresultr[1])) {
                if (quiet) failtext[0]='x';
                print_error(t.func->rettype,resultr,"wrongresult",&failp);
            }
        }
        /*
         * Now test res2, for those functions (frexp, modf, rred)
         * which use it.
         */
        if (t.func->func.ptr == &frexp || t.func->func.ptr == &frexpf ||
            t.func->macro_name == m_rred || t.func->macro_name == m_rredf) {
            unsigned tres2 = t.res2[0];
            if (res2_adjust) {
                /* Fix for range reduction, propagated from further up */
                tres2 = (tres2 + res2_adjust) & 3;
            }
            if (tres2 != intres) {
                if (quiet) failtext[0]='x';
                else {
                    failp += sprintf(failp,
                                     " wrongres2=%08x", intres);
                }
            }
        } else if (t.func->func.ptr == &modf || t.func->func.ptr == &modff) {
            tresultr[0] = t.res2[0];
            tresultr[1] = t.res2[1];
            if (is_double_rettype(t.func->rettype)) {
                canon_dNaN(tresultr);
                resultr[0] = d_res2.i[dmsd];
                resultr[1] = d_res2.i[dlsd];
                canon_dNaN(resultr);
                if (fo) {
                    dnormzero(tresultr);
                    dnormzero(resultr);
                }
            } else {
                canon_sNaN(tresultr);
                resultr[0] = s_res2.i;
                resultr[1] = s_res2.i;
                canon_sNaN(resultr);
                if (fo) {
                    snormzero(tresultr);
                    snormzero(resultr);
                }
            }
            if (resultr[0] != tresultr[0] ||
                (wres > 1 && resultr[1] != tresultr[1])) {
                if (quiet) failtext[0]='x';
                else {
                    if (is_double_rettype(t.func->rettype))
                        failp += sprintf(failp, " wrongres2=%08x.%08x",
                                         resultr[0], resultr[1]);
                    else
                        failp += sprintf(failp, " wrongres2=%08x",
                                         resultr[0]);
                }
            }
        }
    }

    /* Check errno */
    err = (errno == EDOM ? e_EDOM : errno == ERANGE ? e_ERANGE : e_0);
    if (err != t.err && err != t.maybeerr) {
        if (quiet) failtext[0]='x';
        else {
            failp += sprintf(failp, " wrongerrno=%s expecterrno=%s ", errnos[err], errnos[t.err]);
        }
    }

    return *failtext ? test_fail : test_pass;
}

int passed, failed, declined;

void runtests(char *name, FILE *fp) {
    char testbuf[512], linebuf[512];
    int lineno = 1;
    testdetail test;

    test.valid = 0;

    if (verbose) printf("runtests: %s\n", name);
    while (fgets(testbuf, sizeof(testbuf), fp)) {
        int res, print_errno;
        testbuf[strcspn(testbuf, "\r\n")] = '\0';
        strcpy(linebuf, testbuf);
        test = parsetest(testbuf, test);
        print_errno = 0;
        while (test.in_err < test.in_err_limit) {
            res = runtest(test);
            if (res == test_pass) {
                if (verbose)
                    printf("%s:%d: pass\n", name, lineno);
                ++passed;
            } else if (res == test_decline) {
                if (verbose)
                    printf("%s:%d: declined\n", name, lineno);
                ++declined;
            } else if (res == test_fail) {
                if (!quiet)
                    printf("%s:%d: FAIL%s: %s%s%s%s\n", name, lineno,
                           test.random ? " (random)" : "",
                           linebuf,
                           print_errno ? " errno_in=" : "",
                           print_errno ? errnos[test.in_err] : "",
                           failtext);
                ++failed;
            } else if (res == test_invalid) {
                printf("%s:%d: malformed: %s\n", name, lineno, linebuf);
                ++failed;
            }
            test.in_err++;
            print_errno = 1;
        }
        lineno++;
    }
}

int main(int ac, char **av) {
    char **files;
    int i, nfiles = 0;
    dbl d;

#ifdef MICROLIB
    /*
     * Invent argc and argv ourselves.
     */
    char *argv[256];
    char args[256];
    {
        int sargs[2];
        char *p;

        ac = 0;

        sargs[0]=(int)args;
        sargs[1]=(int)sizeof(args);
        if (!__semihost(0x15, sargs)) {
            args[sizeof(args)-1] = '\0';   /* just in case */
            p = args;
            while (1) {
                while (*p == ' ' || *p == '\t') p++;
                if (!*p) break;
                argv[ac++] = p;
                while (*p && *p != ' ' && *p != '\t') p++;
                if (*p) *p++ = '\0';
            }
        }

        av = argv;
    }
#endif

    /* Sort tfuncs */
    qsort(tfuncs, sizeof(tfuncs)/sizeof(test_func), sizeof(test_func), &compare_tfuncs);

    /*
     * Autodetect the `double' endianness.
     */
    dmsd = 0;
    d.f = 1.0;                       /* 0x3ff00000 / 0x00000000 */
    if (d.i[dmsd] == 0) {
        dmsd = 1;
    }
    /*
     * Now dmsd denotes what the compiler thinks we're at. Let's
     * check that it agrees with what the runtime thinks.
     */
    d.i[0] = d.i[1] = 0x11111111;/* a random +ve number */
    d.f /= d.f;                    /* must now be one */
    if (d.i[dmsd] == 0) {
        fprintf(stderr, "YIKES! Compiler and runtime disagree on endianness"
                " of `double'. Bailing out\n");
        return 1;
    }
    dlsd = !dmsd;

    /* default is terse */
    verbose = 0;
    fo = 0;
    strict = 0;

    files = (char **)malloc((ac+1) * sizeof(char *));
    if (!files) {
        fprintf(stderr, "initial malloc failed!\n");
        return 1;
    }
#ifdef NOCMDLINE
    files[nfiles++] = "testfile";
#endif

    while (--ac) {
        char *p = *++av;
        if (*p == '-') {
            static char *options[] = {
                "-fo",
#if 0
                "-noinexact",
                "-noround",
#endif
                "-nostatus",
                "-quiet",
                "-strict",
                "-v",
                "-verbose",
            };
            enum {
                op_fo,
#if 0
                op_noinexact,
                op_noround,
#endif
                op_nostatus,
                op_quiet,
                op_strict,
                op_v,
                op_verbose,
            };
            switch (find(p, options, sizeof(options))) {
            case op_quiet:
                quiet = 1;
                break;
#if 0
            case op_noinexact:
                statusmask &= 0x0F;    /* remove bit 4 */
                break;
            case op_noround:
                doround = 0;
                break;
#endif
            case op_nostatus:        /* no status word => noinx,noround */
                statusmask = 0;
                doround = 0;
                break;
            case op_v:
            case op_verbose:
                verbose = 1;
                break;
            case op_fo:
                fo = 1;
                break;
            case op_strict: /* tolerance is 1 ulp */
                strict = 1;
                break;
            default:
                fprintf(stderr, "unrecognised option: %s\n", p);
                break;
            }
        } else {
            files[nfiles++] = p;
        }
    }

    passed = failed = declined = 0;

    if (nfiles) {
        for (i = 0; i < nfiles; i++) {
            FILE *fp = fopen(files[i], "r");
            if (!fp) {
                fprintf(stderr, "Couldn't open %s\n", files[i]);
            } else
                runtests(files[i], fp);
        }
    } else
        runtests("(stdin)", stdin);

    printf("Completed. Passed %d, failed %d (total %d",
           passed, failed, passed+failed);
    if (declined)
        printf(" plus %d declined", declined);
    printf(")\n");
    if (failed || passed == 0)
        return 1;
    printf("** TEST PASSED OK **\n");
    return 0;
}

void undef_func() {
    failed++;
    puts("ERROR: undefined function called");
}
