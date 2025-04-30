/* file: libm_support.h */


/*
// Copyright (c) 2000 - 2004, Intel Corporation
// All rights reserved.
//
// Contributed 2000 by the Intel Numerics Group, Intel Corporation
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// * The name of Intel Corporation may not be used to endorse or promote
// products derived from this software without specific prior written
// permission.

//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Intel Corporation is the author of this code, and requests that all
// problem reports or change requests be submitted to it directly at
// http://www.intel.com/software/products/opensource/libraries/num.htm.
//

// History: 02/02/2000 Initial version
//          2/28/2000 added tags for logb and nextafter
//          3/22/2000 Changes to support _LIB_VERSIONIMF variable
//                    and filled some enum gaps. Added support for C99.
//          5/31/2000 added prototypes for __libm_frexp_4l/8l
//          8/10/2000 Changed declaration of _LIB_VERSIONIMF to work for library
//                    builds and other application builds (precompiler directives).
//          8/11/2000 Added pointers-to-matherr-functions declarations to allow
//                    for user-defined matherr functions in the dll build.
//         12/07/2000 Added scalbn error_types values.
//          5/01/2001 Added error_types values for C99 nearest integer
//                    functions.
//          6/07/2001 Added error_types values for fdim.
//          6/18/2001 Added include of complex_support.h.
//          8/03/2001 Added error_types values for nexttoward, scalbln.
//          8/23/2001 Corrected tag numbers from 186 and higher.
//          8/27/2001 Added check for long int and long long int definitions.
//         12/10/2001 Added error_types for erfc.
//         12/27/2001 Added error_types for degree argument functions.
//         01/02/2002 Added error_types for tand, cotd.
//         01/04/2002 Delete include of complex_support.h
//         01/23/2002 Deleted prototypes for __libm_frexp*.  Added check for
//                    multiple int, long int, and long long int definitions.
//         05/20/2002 Added error_types for cot.
//         06/27/2002 Added error_types for sinhcosh.
//         12/05/2002 Added error_types for annuity and compound
//         04/10/2003 Added error_types for tgammal/tgamma/tgammaf
//         05/16/2003 FP-treatment macros copied here from IA32 libm_support.h
//         06/02/2003 Added pad into struct fp80 (12/16 bytes).
//         08/01/2003 Added struct ker80 and macros for multiprecision addition,
//                    subtraction, multiplication, division, square root.
//         08/07/2003 History section updated.
//         09/03/2003 ALIGN(n) macro added.
//         10/01/2003 LDOUBLE_ALIGN and fp80 corrected on linux to 16 bytes.
//         11/24/2004 Added ifdef around definitions of INT32/64
//         12/15/2004 Added error_types for exp10, nextafter, nexttoward
//                    underflow.  Moved error codes into libm_error_codes.h.
//
*/

#ifndef __LIBM_SUPPORT_H_INCLUDED__
#define __LIBM_SUPPORT_H_INCLUDED__

#include <math-svid-compat.h>

#ifndef _LIBC
#if !(defined(_WIN32) || defined(_WIN64))
# pragma const_seg(".rodata") /* place constant data in text (code) section */
#endif

#if defined(__ICC) || defined(__ICL) || defined(__ECC) || defined(__ECL)
# pragma warning( disable : 1682 )	/* #1682: ixplicit conversion of a 64-bit integral type to a smaller integral type (potential portability problem) */
# pragma warning( disable : 1683 )	/* #1683: explicit conversion of a 64-bit integral type to a smaller integral type (potential portability problem) */
#endif
#endif

/* macros to form a double value in hex representation (unsigned int type) */

#define DOUBLE_HEX(hi,lo) 0x##lo,0x##hi /*LITTLE_ENDIAN*/

#include "libm_cpu_defs.h"

#if !(defined (IA64))
#  include "libm_dll.h"
#  include "libm_dispatch.h"
#endif

#include "libm_error_codes.h"

struct exceptionf
{
  int type;
  char *name;
  float arg1, arg2, retval;
};

# ifdef __cplusplus
struct __exception
{
  int type;
  char *name;
  double arg1, arg2, retval;
};
# else

#  ifndef _LIBC
struct exception
{
  int type;
  char *name;
  double arg1, arg2, retval;
};
#  endif
# endif

struct exceptionl
{
  int type;
  char *name;
  long double arg1, arg2, retval;
};

#if (defined (_MS_) && defined (IA64))
#define   MATHERR_F   _matherrf
#define   MATHERR_D   _matherr
#else
#define MATHERR_F   matherrf
#define MATHERR_D   matherr
#endif

# ifdef __cplusplus
#define EXC_DECL_D  __exception
#else
// exception is a reserved name in C++
#define EXC_DECL_D  exception
#endif

extern int MATHERR_F(struct exceptionf*);
extern int matherrl(struct exceptionl*);

/* memory format definitions (LITTLE_ENDIAN only) */

#if !(defined(SIZE_INT_32) || defined(SIZE_INT_64))
# error "You need to define SIZE_INT_32 or SIZE_INT_64"
#endif

#if (defined(SIZE_INT_32) && defined(SIZE_INT_64))
#error multiple integer size definitions; define SIZE_INT_32 or SIZE_INT_64
#endif

#if !(defined(SIZE_LONG_32) || defined(SIZE_LONG_64))
# error "You need to define SIZE_LONG_32 or SIZE_LONG_64"
#endif

#if (defined(SIZE_LONG_32) && defined(SIZE_LONG_64))
#error multiple integer size definitions; define SIZE_LONG_32 or SIZE_LONG_64
#endif

#if !defined(__USE_EXTERNAL_FPMEMTYP_H__)

#define BIAS_32  0x007F
#define BIAS_64  0x03FF
#define BIAS_80  0x3FFF

#define MAXEXP_32  0x00FE
#define MAXEXP_64  0x07FE
#define MAXEXP_80  0x7FFE

#define EXPINF_32  0x00FF
#define EXPINF_64  0x07FF
#define EXPINF_80  0x7FFF

struct fp32 { /*// sign:1 exponent:8 significand:23 (implied leading 1)*/
#if defined(SIZE_INT_32)
    unsigned significand:23;
    unsigned exponent:8;
    unsigned sign:1;
#elif defined(SIZE_INT_64)
    unsigned significand:23;
    unsigned exponent:8;
    unsigned sign:1;
#endif
};

struct fp64 { /*/ sign:1 exponent:11 significand:52 (implied leading 1)*/
#if defined(SIZE_INT_32)
    unsigned lo_significand:32;
    unsigned hi_significand:20;
    unsigned exponent:11;
    unsigned sign:1;
#elif defined(SIZE_INT_64)
    unsigned significand:52;
    unsigned exponent:11;
    unsigned sign:1;
#endif
};

struct fp80 { /*/ sign:1 exponent:15 significand:64 (NO implied bits) */
#if defined(SIZE_INT_32)
    unsigned         lo_significand;
    unsigned         hi_significand;
    unsigned         exponent:15;
    unsigned         sign:1;
#elif defined(SIZE_INT_64)
    unsigned         significand;
    unsigned         exponent:15;
    unsigned         sign:1;
#endif
    unsigned         pad:16;
#if !(defined(__unix__) && defined(__i386__))
    unsigned         padwin:32;
#endif
};

#endif /*__USE_EXTERNAL_FPMEMTYP_H__*/

#if !(defined(opensource))
typedef          __int32  INT32;
typedef   signed __int32 SINT32;
typedef unsigned __int32 UINT32;

typedef          __int64  INT64;
typedef   signed __int64 SINT64;
typedef unsigned __int64 UINT64;
#else
typedef          int  INT32;
typedef   signed int SINT32;
typedef unsigned int UINT32;

typedef          long long  INT64;
typedef   signed long long SINT64;
typedef unsigned long long UINT64;
#endif

#if (defined(_WIN32) || defined(_WIN64))        /* Windows */
# define I64CONST(bits) 0x##bits##i64
# define U64CONST(bits) 0x##bits##ui64
#elif (defined(__linux__) && defined(_M_IA64))  /* Linux,64 */
# define I64CONST(bits) 0x##bits##L
# define U64CONST(bits) 0x##bits##uL
#else                                           /* Linux,32 */
# define I64CONST(bits) 0x##bits##LL
# define U64CONST(bits) 0x##bits##uLL
#endif

struct ker80 {
    union {
        long double ldhi;
        struct fp80 fphi;
    };
    union {
        long double ldlo;
        struct fp80 fplo;
    };
    int ex;
};

/* Addition: x+y                                            */
/* The result is sum rhi+rlo                                */
/* Temporary variables: t1                                  */
/* All variables are in long double precision               */
/* Correct if no overflow (algorithm by D.Knuth)           */
#define __LIBM_ADDL1_K80( rhi,rlo,x,y, t1 )                 \
    rhi = x   + y;                                          \
    rlo = rhi - x;                                          \
    t1  = rhi - rlo;                                        \
    rlo = y   - rlo;                                        \
    t1  = x   - t1;                                         \
    rlo = rlo + t1;

/* Addition: (xhi+xlo) + (yhi+ylo)                          */
/* The result is sum rhi+rlo                                */
/* Temporary variables: t1                                  */
/* All variables are in long double precision               */
/* Correct if no overflow (algorithm by T.J.Dekker)         */
#define __LIBM_ADDL2_K80( rhi,rlo,xhi,xlo,yhi,ylo, t1 )     \
    rlo = xhi+yhi;                                          \
    if ( VALUE_GT_80(FP80(xhi),FP80(yhi)) ) {               \
        t1=xhi-rlo;t1=t1+yhi;t1=t1+ylo;t1=t1+xlo;           \
    } else {                                                \
        t1=yhi-rlo;t1=t1+xhi;t1=t1+xlo;t1=t1+ylo;           \
    }                                                       \
    rhi=rlo+t1;                                             \
    rlo=rlo-rhi;rlo=rlo+t1;

/* Addition: r=x+y                                          */
/* Variables r,x,y are pointers to struct ker80,            */
/* all other variables are in long double precision         */
/* Temporary variables: t1                                  */
/* Correct if x and y belong to interval [2^-8000;2^8000],  */
/* or when one or both of them are zero                     */
#if   defined(SIZE_INT_32)
#define __LIBM_ADDL_K80(r,x,y, t1)                          \
    if ( ((y)->ex+(y)->fphi.exponent-134 <                  \
          (x)->ex+(x)->fphi.exponent)       &&              \
         ((x)->ex+(x)->fphi.exponent <                      \
          (y)->ex+(y)->fphi.exponent+134)   &&              \
         !SIGNIFICAND_ZERO_80(&((x)->fphi)) &&              \
         !SIGNIFICAND_ZERO_80(&((y)->fphi)) )               \
    {                                                       \
        /* y/2^134 < x < y*2^134,               */          \
        /* and x,y are nonzero finite numbers   */          \
        if ( (x)->ex != (y)->ex ) {                         \
            /* adjust x->ex to y->ex */                     \
            /* t1 = 2^(x->ex - y->ex) */                    \
            FP80(t1)->sign = 0;                             \
            FP80(t1)->exponent = BIAS_80 + (x)->ex-(y)->ex; \
            /*  exponent is correct because             */  \
            /*  |x->ex - y->ex| =                       */  \
            /*  = |  (x->ex + x->fphi.exponent) -       */  \
            /*      -(y->ex + y->fphi.exponent) +       */  \
            /*              + y->fphi.exponent  -       */  \
            /*              - x->fphi.exponent     | <  */  \
            /*  < |  (x->ex+x->fphi.exponent) -         */  \
            /*      -(y->ex+y->fphi.exponent)      | +  */  \
            /*   +|  y->fphi.exponent -                 */  \
            /*      -x->fphi.exponent              | <  */  \
            /*  < 134 + 16000                           */  \
            FP80(t1)->hi_significand = 0x80000000;          \
            FP80(t1)->lo_significand = 0x00000000;          \
            (x)->ex = (y)->ex;                              \
            (x)->ldhi *= t1;                                \
            (x)->ldlo *= t1;                                \
        }                                                   \
        /* r==x+y */                                        \
        (r)->ex = (y)->ex;                                  \
        __LIBM_ADDL2_K80( (r)->ldhi,(r)->ldlo,              \
            (x)->ldhi,(x)->ldlo, (y)->ldhi,(y)->ldlo, t1 ); \
    } else if ( SIGNIFICAND_ZERO_80(&((x)->fphi)) ||        \
             ((y)->ex+(y)->fphi.exponent-BIAS_80 - 134 >=   \
              (x)->ex+(x)->fphi.exponent-BIAS_80) )         \
    {                                                       \
        /* |x|<<|y| */                                      \
        *(r) = *(y);                                        \
    } else {                                                \
        /* |y|<<|x| */                                      \
        *(r) = *(x);                                        \
    }
#elif defined(SIZE_INT_64)
#define __LIBM_ADDL_K80(r,x,y, t1)                          \
    if ( ((y)->ex+(y)->fphi.exponent-134 <                  \
          (x)->ex+(x)->fphi.exponent)       &&              \
         ((x)->ex+(x)->fphi.exponent <                      \
          (y)->ex+(y)->fphi.exponent+134)   &&              \
         !SIGNIFICAND_ZERO_80(&((x)->fphi)) &&              \
         !SIGNIFICAND_ZERO_80(&((y)->fphi)) )               \
    {                                                       \
        /* y/2^134 < x < y*2^134,               */          \
        /* and x,y are nonzero finite numbers   */          \
        if ( (x)->ex != (y)->ex ) {                         \
            /* adjust x->ex to y->ex */                     \
            /* t1 = 2^(x->ex - y->ex) */                    \
            FP80(t1)->sign = 0;                             \
            FP80(t1)->exponent = BIAS_80 + (x)->ex-(y)->ex; \
            /*  exponent is correct because             */  \
            /*  |x->ex - y->ex| =                       */  \
            /*  = |  (x->ex + x->fphi.exponent) -       */  \
            /*      -(y->ex + y->fphi.exponent) +       */  \
            /*              + y->fphi.exponent  -       */  \
            /*              - x->fphi.exponent     | <  */  \
            /*  < |  (x->ex+x->fphi.exponent) -         */  \
            /*      -(y->ex+y->fphi.exponent)      | +  */  \
            /*   +|  y->fphi.exponent -                 */  \
            /*      -x->fphi.exponent              | <  */  \
            /*  < 134 + 16000                           */  \
            FP80(t1)->significand = 0x8000000000000000;     \
            (x)->ex = (y)->ex;                              \
            (x)->ldhi *= t1;                                \
            (x)->ldlo *= t1;                                \
        }                                                   \
        /* r==x+y */                                        \
        (r)->ex = (y)->ex;                                  \
        __LIBM_ADDL2_K80( (r)->ldhi,(r)->ldlo,              \
            (x)->ldhi,(x)->ldlo, (y)->ldhi,(y)->ldlo, t1 ); \
    } else if ( SIGNIFICAND_ZERO_80(&((x)->fphi)) ||        \
             ((y)->ex+(y)->fphi.exponent-BIAS_80 - 134 >=   \
              (x)->ex+(x)->fphi.exponent-BIAS_80) )         \
    {                                                       \
        /* |x|<<|y| */                                      \
        *(r) = *(y);                                        \
    } else {                                                \
        /* |y|<<|x| */                                      \
        *(r) = *(x);                                        \
    }
#endif

/* Addition: r=x+y                                          */
/* Variables r,x,y are pointers to struct ker80,            */
/* all other variables are in long double precision         */
/* Temporary variables: t1                                  */
/* Correct for any finite x and y                           */
#define __LIBM_ADDL_NORM_K80(r,x,y, t1)                     \
    if ( ((x)->fphi.exponent-BIAS_80<-8000) ||              \
         ((x)->fphi.exponent-BIAS_80>+8000) ||              \
         ((y)->fphi.exponent-BIAS_80<-8000) ||              \
         ((y)->fphi.exponent-BIAS_80>+8000) )               \
    {                                                       \
        __libm_normalizel_k80(x);                           \
        __libm_normalizel_k80(y);                           \
    }                                                       \
    __LIBM_ADDL_K80(r,x,y, t1)

/* Subtraction: x-y                                         */
/* The result is sum rhi+rlo                                */
/* Temporary variables: t1                                  */
/* All variables are in long double precision               */
/* Correct if no overflow (algorithm by D.Knuth)           */
#define __LIBM_SUBL1_K80( rhi, rlo, x, y, t1 )              \
    rhi = x   - y;                                          \
    rlo = rhi - x;                                          \
    t1  = rhi - rlo;                                        \
    rlo = y   + rlo;                                        \
    t1  = x   - t1;                                         \
    rlo = t1  - rlo;

/* Subtraction: (xhi+xlo) - (yhi+ylo)                       */
/* The result is sum rhi+rlo                                */
/* Temporary variables: t1                                  */
/* All variables are in long double precision               */
/* Correct if no overflow (algorithm by T.J.Dekker)         */
#define __LIBM_SUBL2_K80( rhi,rlo,xhi,xlo,yhi,ylo, t1 )     \
    rlo = xhi-yhi;                                          \
    if ( VALUE_GT_80(FP80(xhi),FP80(yhi)) ) {               \
        t1=xhi-rlo;t1=t1-yhi;t1=t1-ylo;t1=t1+xlo;           \
    } else {                                                \
        t1=yhi+rlo;t1=xhi-t1;t1=t1+xlo;t1=t1-ylo;           \
    }                                                       \
    rhi=rlo+t1;                                             \
    rlo=rlo-rhi;rlo=rlo+t1;

/* Subtraction: r=x-y                                       */
/* Variables r,x,y are pointers to struct ker80,            */
/* all other variables are in long double precision         */
/* Temporary variables: t1                                  */
/* Correct if x and y belong to interval [2^-8000;2^8000],  */
/* or when one or both of them are zero                     */
#if   defined(SIZE_INT_32)
#define __LIBM_SUBL_K80(r,x,y, t1)                          \
    if ( ((y)->ex+(y)->fphi.exponent-134 <                  \
          (x)->ex+(x)->fphi.exponent)       &&              \
         ((x)->ex+(x)->fphi.exponent <                      \
          (y)->ex+(y)->fphi.exponent+134)   &&              \
         !SIGNIFICAND_ZERO_80(&((x)->fphi)) &&              \
         !SIGNIFICAND_ZERO_80(&((y)->fphi)) )               \
    {                                                       \
        /* y/2^134 < x < y*2^134,               */          \
        /* and x,y are nonzero finite numbers   */          \
        if ( (x)->ex != (y)->ex ) {                         \
            /* adjust x->ex to y->ex */                     \
            /* t1 = 2^(x->ex - y->ex) */                    \
            FP80(t1)->sign = 0;                             \
            FP80(t1)->exponent = BIAS_80 + (x)->ex-(y)->ex; \
            /*  exponent is correct because             */  \
            /*  |x->ex - y->ex| =                       */  \
            /*  = |  (x->ex + x->fphi.exponent) -       */  \
            /*      -(y->ex + y->fphi.exponent) +       */  \
            /*              + y->fphi.exponent  -       */  \
            /*              - x->fphi.exponent     | <  */  \
            /*  < |  (x->ex+x->fphi.exponent) -         */  \
            /*      -(y->ex+y->fphi.exponent)      | +  */  \
            /*   +|  y->fphi.exponent -                 */  \
            /*      -x->fphi.exponent              | <  */  \
            /*  < 134 + 16000                           */  \
            FP80(t1)->hi_significand = 0x80000000;          \
            FP80(t1)->lo_significand = 0x00000000;          \
            (x)->ex = (y)->ex;                              \
            (x)->ldhi *= t1;                                \
            (x)->ldlo *= t1;                                \
        }                                                   \
        /* r==x+y */                                        \
        (r)->ex = (y)->ex;                                  \
        __LIBM_SUBL2_K80( (r)->ldhi,(r)->ldlo,              \
            (x)->ldhi,(x)->ldlo, (y)->ldhi,(y)->ldlo, t1 ); \
    } else if ( SIGNIFICAND_ZERO_80(&((x)->fphi)) ||        \
             ((y)->ex+(y)->fphi.exponent-BIAS_80 - 134 >=   \
              (x)->ex+(x)->fphi.exponent-BIAS_80) )         \
    {                                                       \
        /* |x|<<|y| */                                      \
        (r)->ex   =   (y)->ex;                              \
        (r)->ldhi = -((y)->ldhi);                           \
        (r)->ldlo = -((y)->ldlo);                           \
    } else {                                                \
        /* |y|<<|x| */                                      \
        *(r) = *(x);                                        \
    }
#elif defined(SIZE_INT_64)
#define __LIBM_SUBL_K80(r,x,y, t1)                          \
    if ( ((y)->ex+(y)->fphi.exponent-134 <                  \
          (x)->ex+(x)->fphi.exponent)       &&              \
         ((x)->ex+(x)->fphi.exponent <                      \
          (y)->ex+(y)->fphi.exponent+134)   &&              \
         !SIGNIFICAND_ZERO_80(&((x)->fphi)) &&              \
         !SIGNIFICAND_ZERO_80(&((y)->fphi)) )               \
    {                                                       \
        /* y/2^134 < x < y*2^134,               */          \
        /* and x,y are nonzero finite numbers   */          \
        if ( (x)->ex != (y)->ex ) {                         \
            /* adjust x->ex to y->ex */                     \
            /* t1 = 2^(x->ex - y->ex) */                    \
            FP80(t1)->sign = 0;                             \
            FP80(t1)->exponent = BIAS_80 + (x)->ex-(y)->ex; \
            /*  exponent is correct because             */  \
            /*  |x->ex - y->ex| =                       */  \
            /*  = |  (x->ex + x->fphi.exponent) -       */  \
            /*      -(y->ex + y->fphi.exponent) +       */  \
            /*              + y->fphi.exponent  -       */  \
            /*              - x->fphi.exponent     | <  */  \
            /*  < |  (x->ex+x->fphi.exponent) -         */  \
            /*      -(y->ex+y->fphi.exponent)      | +  */  \
            /*   +|  y->fphi.exponent -                 */  \
            /*      -x->fphi.exponent              | <  */  \
            /*  < 134 + 16000                           */  \
            FP80(t1)->significand = 0x8000000000000000;     \
            (x)->ex = (y)->ex;                              \
            (x)->ldhi *= t1;                                \
            (x)->ldlo *= t1;                                \
        }                                                   \
        /* r==x+y */                                        \
        (r)->ex = (y)->ex;                                  \
        __LIBM_SUBL2_K80( (r)->ldhi,(r)->ldlo,              \
            (x)->ldhi,(x)->ldlo, (y)->ldhi,(y)->ldlo, t1 ); \
    } else if ( SIGNIFICAND_ZERO_80(&((x)->fphi)) ||        \
             ((y)->ex+(y)->fphi.exponent-BIAS_80 - 134 >=   \
              (x)->ex+(x)->fphi.exponent-BIAS_80) )         \
    {                                                       \
        /* |x|<<|y| */                                      \
        (r)->ex   =   (y)->ex;                              \
        (r)->ldhi = -((y)->ldhi);                           \
        (r)->ldlo = -((y)->ldlo);                           \
    } else {                                                \
        /* |y|<<|x| */                                      \
        *(r) = *(x);                                        \
    }
#endif

/* Subtraction: r=x+y                                       */
/* Variables r,x,y are pointers to struct ker80,            */
/* all other variables are in long double precision         */
/* Temporary variables: t1                                  */
/* Correct for any finite x and y                           */
#define __LIBM_SUBL_NORM_K80(r,x,y, t1)                     \
    if ( ((x)->fphi.exponent-BIAS_80<-8000) ||              \
         ((x)->fphi.exponent-BIAS_80>+8000) ||              \
         ((y)->fphi.exponent-BIAS_80<-8000) ||              \
         ((y)->fphi.exponent-BIAS_80>+8000) )               \
    {                                                       \
        __libm_normalizel_k80(x);                           \
        __libm_normalizel_k80(y);                           \
    }                                                       \
    __LIBM_SUBL_K80(r,x,y, t1)

/* Multiplication: x*y                                      */
/* The result is sum rhi+rlo                                */
/* Here t32 is the constant 2^32+1                          */
/* Temporary variables: t1,t2,t3,t4,t5,t6                   */
/* All variables are in long double precision               */
/* Correct if no over/underflow (algorithm by T.J.Dekker)   */
#define __LIBM_MULL1_K80(rhi,rlo,x,y,                       \
                                     t32,t1,t2,t3,t4,t5,t6) \
    t1=(x)*(t32); t3=x-t1; t3=t3+t1; t4=x-t3;               \
    t1=(y)*(t32); t5=y-t1; t5=t5+t1; t6=y-t5;               \
    t1=(t3)*(t5);                                           \
    t2=(t3)*(t6)+(t4)*(t5);                                 \
    rhi=t1+t2;                                              \
    rlo=t1-rhi; rlo=rlo+t2; rlo=rlo+(t4*t6);

/* Multiplication: (xhi+xlo)*(yhi+ylo)                      */
/* The result is sum rhi+rlo                                */
/* Here t32 is the constant 2^32+1                          */
/* Temporary variables: t1,t2,t3,t4,t5,t6,t7,t8             */
/* All variables are in long double precision               */
/* Correct if no over/underflow (algorithm by T.J.Dekker)   */
#define __LIBM_MULL2_K80(rhi,rlo,xhi,xlo,yhi,ylo,           \
                               t32,t1,t2,t3,t4,t5,t6,t7,t8) \
    __LIBM_MULL1_K80(t7,t8,xhi,yhi, t32,t1,t2,t3,t4,t5,t6)  \
    t1=(xhi)*(ylo)+(xlo)*(yhi); t1=t1+t8;                   \
    rhi=t7+t1;                                              \
    rlo=t7-rhi; rlo=rlo+t1;

/* Multiplication: r=x*y                                    */
/* Variables r,x,y are pointers to struct ker80,            */
/* all other variables are in long double precision         */
/* Here t32 is the constant 2^32+1                          */
/* Temporary variables: t1,t2,t3,t4,t5,t6,t7,t8             */
/* Correct if x and y belong to interval [2^-8000;2^8000]   */
#define __LIBM_MULL_K80(r,x,y, t32,t1,t2,t3,t4,t5,t6,t7,t8) \
    (r)->ex = (x)->ex + (y)->ex;                            \
    __LIBM_MULL2_K80((r)->ldhi,(r)->ldlo,                   \
        (x)->ldhi,(x)->ldlo,(y)->ldhi,(y)->ldlo,            \
        t32,t1,t2,t3,t4,t5,t6,t7,t8)

/* Multiplication: r=x*y                                    */
/* Variables r,x,y are pointers to struct ker80,            */
/* all other variables are in long double precision         */
/* Here t32 is the constant 2^32+1                          */
/* Temporary variables: t1,t2,t3,t4,t5,t6,t7,t8             */
/* Correct for any finite x and y                           */
#define __LIBM_MULL_NORM_K80(r,x,y,                         \
                               t32,t1,t2,t3,t4,t5,t6,t7,t8) \
    if ( ((x)->fphi.exponent-BIAS_80<-8000) ||              \
         ((x)->fphi.exponent-BIAS_80>+8000) ||              \
         ((y)->fphi.exponent-BIAS_80<-8000) ||              \
         ((y)->fphi.exponent-BIAS_80>+8000) )               \
    {                                                       \
        __libm_normalizel_k80(x);                           \
        __libm_normalizel_k80(y);                           \
    }                                                       \
    __LIBM_MULL_K80(r,x,y, t32,t1,t2,t3,t4,t5,t6,t7,t8)

/* Division: (xhi+xlo)/(yhi+ylo)                            */
/* The result is sum rhi+rlo                                */
/* Here t32 is the constant 2^32+1                          */
/* Temporary variables: t1,t2,t3,t4,t5,t6,t7,t8,t9          */
/* All variables are in long double precision               */
/* Correct if no over/underflow (algorithm by T.J.Dekker)   */
#define __LIBM_DIVL2_K80(rhi,rlo,xhi,xlo,yhi,ylo,           \
                            t32,t1,t2,t3,t4,t5,t6,t7,t8,t9) \
    t7=(xhi)/(yhi);                                         \
    __LIBM_MULL1_K80(t8,t9,t7,yhi, t32,t1,t2,t3,t4,t5,t6)   \
    t1=xhi-t8; t1=t1-t9; t1=t1+xlo; t1=t1-(t7)*(ylo);       \
    t1=(t1)/(yhi);                                          \
    rhi=t7+t1;                                              \
    rlo=t7-rhi; rlo=rlo+t1;

/* Division: r=x/y                                          */
/* Variables r,x,y are pointers to struct ker80,            */
/* all other variables are in long double precision         */
/* Here t32 is the constant 2^32+1                          */
/* Temporary variables: t1,t2,t3,t4,t5,t6,t7,t8,t9          */
/* Correct if x and y belong to interval [2^-8000;2^8000]   */
#define __LIBM_DIVL_K80(r,x,y,                              \
                            t32,t1,t2,t3,t4,t5,t6,t7,t8,t9) \
    (r)->ex = (x)->ex - (y)->ex;                            \
    __LIBM_DIVL2_K80( (r)->ldhi,(r)->ldlo,                  \
        (x)->ldhi,(x)->ldlo,(y)->ldhi,(y)->ldlo,            \
        t32,t1,t2,t3,t4,t5,t6,t7,t8,t9)

/* Division: r=x/y                                          */
/* Variables r,x,y are pointers to struct ker80,            */
/* all other variables are in long double precision         */
/* Here t32 is the constant 2^32+1                          */
/* Temporary variables: t1,t2,t3,t4,t5,t6,t7,t8             */
/* Correct for any finite x and y                           */
#define __LIBM_DIVL_NORM_K80(r,x,y,                         \
                            t32,t1,t2,t3,t4,t5,t6,t7,t8,t9) \
    if ( ((x)->fphi.exponent-BIAS_80<-8000) ||              \
         ((x)->fphi.exponent-BIAS_80>+8000) ||              \
         ((y)->fphi.exponent-BIAS_80<-8000) ||              \
         ((y)->fphi.exponent-BIAS_80>+8000) )               \
    {                                                       \
        __libm_normalizel_k80(x);                           \
        __libm_normalizel_k80(y);                           \
    }                                                       \
    __LIBM_DIVL_K80(r,x,y, t32,t1,t2,t3,t4,t5,t6,t7,t8,t9)

/* Square root: sqrt(xhi+xlo)                               */
/* The result is sum rhi+rlo                                */
/* Here t32 is the constant 2^32+1                          */
/*      half is the constant 0.5                            */
/* Temporary variables: t1,t2,t3,t4,t5,t6,t7,t8,t9          */
/* All variables are in long double precision               */
/* Correct for positive xhi+xlo (algorithm by T.J.Dekker)   */
#define __LIBM_SQRTL2_NORM_K80(rhi,rlo,xhi,xlo,             \
                       t32,half,t1,t2,t3,t4,t5,t6,t7,t8,t9) \
    t7=sqrtl(xhi);                                          \
    __LIBM_MULL1_K80(t8,t9,t7,t7, t32,t1,t2,t3,t4,t5,t6)    \
    t1=xhi-t8; t1=t1-t9; t1=t1+xlo; t1=(t1)*(half);         \
    t1=(t1)/(t7);                                           \
    rhi=t7+t1;                                              \
    rlo=t7-rhi; rlo=rlo+t1;

/* Square root: r=sqrt(x)                                   */
/* Variables r,x,y are pointers to struct ker80,            */
/* all other variables are in long double precision         */
/* Here t32 is the constant 2^32+1                          */
/*      half is the constant 0.5                            */
/* Temporary variables: t1,t2,t3,t4,t5,t6,t7,t8,t9          */
/* Correct if x belongs to interval [2^-16000;2^16000]      */
#define __LIBM_SQRTL_K80(r,x,                               \
                       t32,half,t1,t2,t3,t4,t5,t6,t7,t8,t9) \
    if ( ((x)->ex & 1) == 1 ) {                             \
        (x)->ex    = (x)->ex + 1;                           \
        (x)->ldhi *= half;                                  \
        (x)->ldlo *= half;                                  \
    }                                                       \
    (r)->ex = (x)->ex >> 1;                                 \
    __LIBM_SQRTL2_NORM_K80( (r)->ldhi,(r)->ldlo,            \
        (x)->ldhi,(x)->ldlo,                                \
        t32,half,t1,t2,t3,t4,t5,t6,t7,t8,t9)

/* Square root: r=sqrt(x)                                   */
/* Variables r,x,y are pointers to struct ker80,            */
/* all other variables are in long double precision         */
/* Here t32 is the constant 2^32+1                          */
/*      half is the constant 0.5                            */
/* Temporary variables: t1,t2,t3,t4,t5,t6,t7,t8,t9          */
/* Correct for any positive x                               */
#define __LIBM_SQRTL_NORM_K80(r,x,                          \
                       t32,half,t1,t2,t3,t4,t5,t6,t7,t8,t9) \
    if ( ((x)->fphi.exponent-BIAS_80<-16000) ||             \
         ((x)->fphi.exponent-BIAS_80>+16000) )              \
    {                                                       \
        __libm_normalizel_k80(x);                           \
    }                                                       \
    __LIBM_SQRTL_K80(r,x, t32,half,t1,t2,t3,t4,t5,t6,t7,t8,t9)


#ifdef __INTEL_COMPILER
#define ALIGN(n) __declspec(align(n))
#else /* __INTEL_COMPILER */
#define ALIGN(n)
#endif /* __INTEL_COMPILER */

/* macros to form a long double value in hex representation (unsigned short type) */

#if (defined(__unix__) && defined(__i386__))
# define LDOUBLE_ALIGN 12	/* IA32 Linux: 12-byte alignment */
#else	/*__linux__ & IA32*/
# define LDOUBLE_ALIGN 16	/* EFI2/IA32 Win or IPF Win/Linux: 16-byte alignment */
#endif	/*__linux__ & IA32*/

#if (LDOUBLE_ALIGN == 16)
#define _XPD_ ,0x0000,0x0000,0x0000
#else /*12*/
#define _XPD_ ,0x0000
#endif

#define LDOUBLE_HEX(w4,w3,w2,w1,w0) 0x##w0,0x##w1,0x##w2,0x##w3,0x##w4 _XPD_ /*LITTLE_ENDIAN*/

/* macros to sign-expand low 'num' bits of 'val' to native integer */

#if defined(SIZE_INT_32)
# define SIGN_EXPAND(val,num)  ((int)(val) << (32-(num))) >> (32-(num)) /* sign expand of 'num' LSBs */
#elif defined(SIZE_INT_64)
# define SIGN_EXPAND(val,num)  ((int)(val) << (64-(num))) >> (64-(num)) /* sign expand of 'num' LSBs */
#endif

/* macros to form pointers to FP number on-the-fly */

#define FP32(f)  ((struct fp32 *)&f)
#define FP64(d)  ((struct fp64 *)&d)
#define FP80(ld) ((struct fp80 *)&ld)

/* macros to extract signed low and high doubleword of long double */

#if defined(SIZE_INT_32)
# define HI_DWORD_80(ld) ((((FP80(ld)->sign << 15) | FP80(ld)->exponent) << 16) | \
                          ((FP80(ld)->hi_significand >> 16) & 0xFFFF))
# define LO_DWORD_80(ld) SIGN_EXPAND(FP80(ld)->lo_significand, 32)
#elif defined(SIZE_INT_64)
# define HI_DWORD_80(ld) ((((FP80(ld)->sign << 15) | FP80(ld)->exponent) << 16) | \
                          ((FP80(ld)->significand >> 48) & 0xFFFF))
# define LO_DWORD_80(ld) SIGN_EXPAND(FP80(ld)->significand, 32)
#endif

/* macros to extract hi bits of significand.
 * note that explicit high bit do not count (returns as is)
 */

#if defined(SIZE_INT_32)
# define HI_SIGNIFICAND_80(X,NBITS) ((X)->hi_significand >> (31 - (NBITS)))
#elif defined(SIZE_INT_64)
# define HI_SIGNIFICAND_80(X,NBITS) ((X)->significand >> (63 - (NBITS)))
#endif

/* macros to check, whether a significand bits are all zero, or some of them are non-zero.
 * note that SIGNIFICAND_ZERO_80 tests high bit also, but SIGNIFICAND_NONZERO_80 does not
 */

#define SIGNIFICAND_ZERO_32(X)     ((X)->significand == 0)
#define SIGNIFICAND_NONZERO_32(X)  ((X)->significand != 0)

#if defined(SIZE_INT_32)
# define SIGNIFICAND_ZERO_64(X)    (((X)->hi_significand == 0) && ((X)->lo_significand == 0))
# define SIGNIFICAND_NONZERO_64(X) (((X)->hi_significand != 0) || ((X)->lo_significand != 0))
#elif defined(SIZE_INT_64)
# define SIGNIFICAND_ZERO_64(X)    ((X)->significand == 0)
# define SIGNIFICAND_NONZERO_64(X) ((X)->significand != 0)
#endif

#if defined(SIZE_INT_32)
# define SIGNIFICAND_ZERO_80(X)    (((X)->hi_significand == 0x00000000) && ((X)->lo_significand == 0))
# define SIGNIFICAND_NONZERO_80(X) (((X)->hi_significand != 0x80000000) || ((X)->lo_significand != 0))
#elif defined(SIZE_INT_64)
# define SIGNIFICAND_ZERO_80(X)    ((X)->significand == 0x0000000000000000)
# define SIGNIFICAND_NONZERO_80(X) ((X)->significand != 0x8000000000000000)
#endif

/* macros to compare long double with constant value, represented as hex */

#define SIGNIFICAND_EQ_HEX_32(X,BITS) ((X)->significand == 0x ## BITS)
#define SIGNIFICAND_GT_HEX_32(X,BITS) ((X)->significand >  0x ## BITS)
#define SIGNIFICAND_GE_HEX_32(X,BITS) ((X)->significand >= 0x ## BITS)
#define SIGNIFICAND_LT_HEX_32(X,BITS) ((X)->significand <  0x ## BITS)
#define SIGNIFICAND_LE_HEX_32(X,BITS) ((X)->significand <= 0x ## BITS)

#if defined(SIZE_INT_32)
# define SIGNIFICAND_EQ_HEX_64(X,HI,LO) \
    (((X)->hi_significand == 0x ## HI) && ((X)->lo_significand == 0x ## LO))
# define SIGNIFICAND_GT_HEX_64(X,HI,LO) (((X)->hi_significand > 0x ## HI) || \
    (((X)->hi_significand == 0x ## HI) && ((X)->lo_significand >  0x ## LO)))
# define SIGNIFICAND_GE_HEX_64(X,HI,LO) (((X)->hi_significand > 0x ## HI) || \
    (((X)->hi_significand == 0x ## HI) && ((X)->lo_significand >= 0x ## LO)))
# define SIGNIFICAND_LT_HEX_64(X,HI,LO) (((X)->hi_significand < 0x ## HI) || \
    (((X)->hi_significand == 0x ## HI) && ((X)->lo_significand <  0x ## LO)))
# define SIGNIFICAND_LE_HEX_64(X,HI,LO) (((X)->hi_significand < 0x ## HI) || \
    (((X)->hi_significand == 0x ## HI) && ((X)->lo_significand <= 0x ## LO)))
#elif defined(SIZE_INT_64)
# define SIGNIFICAND_EQ_HEX_64(X,HI,LO) ((X)->significand == 0x ## HI ## LO)
# define SIGNIFICAND_GT_HEX_64(X,HI,LO) ((X)->significand >  0x ## HI ## LO)
# define SIGNIFICAND_GE_HEX_64(X,HI,LO) ((X)->significand >= 0x ## HI ## LO)
# define SIGNIFICAND_LT_HEX_64(X,HI,LO) ((X)->significand <  0x ## HI ## LO)
# define SIGNIFICAND_LE_HEX_64(X,HI,LO) ((X)->significand <= 0x ## HI ## LO)
#endif

#if defined(SIZE_INT_32)
# define SIGNIFICAND_EQ_HEX_80(X,HI,LO) \
    (((X)->hi_significand == 0x ## HI) && ((X)->lo_significand == 0x ## LO))
# define SIGNIFICAND_GT_HEX_80(X,HI,LO) (((X)->hi_significand > 0x ## HI) || \
    (((X)->hi_significand == 0x ## HI) && ((X)->lo_significand >  0x ## LO)))
# define SIGNIFICAND_GE_HEX_80(X,HI,LO) (((X)->hi_significand > 0x ## HI) || \
    (((X)->hi_significand == 0x ## HI) && ((X)->lo_significand >= 0x ## LO)))
# define SIGNIFICAND_LT_HEX_80(X,HI,LO) (((X)->hi_significand < 0x ## HI) || \
    (((X)->hi_significand == 0x ## HI) && ((X)->lo_significand <  0x ## LO)))
# define SIGNIFICAND_LE_HEX_80(X,HI,LO) (((X)->hi_significand < 0x ## HI) || \
    (((X)->hi_significand == 0x ## HI) && ((X)->lo_significand <= 0x ## LO)))
#elif defined(SIZE_INT_64)
# define SIGNIFICAND_EQ_HEX_80(X,HI,LO) ((X)->significand == 0x ## HI ## LO)
# define SIGNIFICAND_GT_HEX_80(X,HI,LO) ((X)->significand >  0x ## HI ## LO)
# define SIGNIFICAND_GE_HEX_80(X,HI,LO) ((X)->significand >= 0x ## HI ## LO)
# define SIGNIFICAND_LT_HEX_80(X,HI,LO) ((X)->significand <  0x ## HI ## LO)
# define SIGNIFICAND_LE_HEX_80(X,HI,LO) ((X)->significand <= 0x ## HI ## LO)
#endif

#define VALUE_EQ_HEX_32(X,EXP,BITS) \
   (((X)->exponent == (EXP)) && (SIGNIFICAND_EQ_HEX_32(X, BITS)))
#define VALUE_GT_HEX_32(X,EXP,BITS) (((X)->exponent > (EXP)) || \
   (((X)->exponent == (EXP)) && (SIGNIFICAND_GT_HEX_32(X, BITS))))
#define VALUE_GE_HEX_32(X,EXP,BITS) (((X)->exponent > (EXP)) || \
   (((X)->exponent == (EXP)) && (SIGNIFICAND_GE_HEX_32(X, BITS))))
#define VALUE_LT_HEX_32(X,EXP,BITS) (((X)->exponent < (EXP)) || \
   (((X)->exponent == (EXP)) && (SIGNIFICAND_LT_HEX_32(X, BITS))))
#define VALUE_LE_HEX_32(X,EXP,BITS) (((X)->exponent < (EXP)) || \
   (((X)->exponent == (EXP)) && (SIGNIFICAND_LE_HEX_32(X, BITS))))

#define VALUE_EQ_HEX_64(X,EXP,HI,LO) \
   (((X)->exponent == (EXP)) && (SIGNIFICAND_EQ_HEX_64(X, HI, LO)))
#define VALUE_GT_HEX_64(X,EXP,HI,LO) (((X)->exponent > (EXP)) || \
   (((X)->exponent == (EXP)) && (SIGNIFICAND_GT_HEX_64(X, HI, LO))))
#define VALUE_GE_HEX_64(X,EXP,HI,LO) (((X)->exponent > (EXP)) || \
   (((X)->exponent == (EXP)) && (SIGNIFICAND_GE_HEX_64(X, HI, LO))))
#define VALUE_LT_HEX_64(X,EXP,HI,LO) (((X)->exponent < (EXP)) || \
   (((X)->exponent == (EXP)) && (SIGNIFICAND_LT_HEX_64(X, HI, LO))))
#define VALUE_LE_HEX_64(X,EXP,HI,LO) (((X)->exponent < (EXP)) || \
   (((X)->exponent == (EXP)) && (SIGNIFICAND_LE_HEX_64(X, HI, LO))))

#define VALUE_EQ_HEX_80(X,EXP,HI,LO) \
   (((X)->exponent == (EXP)) && (SIGNIFICAND_EQ_HEX_80(X, HI, LO)))
#define VALUE_GT_HEX_80(X,EXP,HI,LO) (((X)->exponent > (EXP)) || \
   (((X)->exponent == (EXP)) && (SIGNIFICAND_GT_HEX_80(X, HI, LO))))
#define VALUE_GE_HEX_80(X,EXP,HI,LO) (((X)->exponent > (EXP)) || \
   (((X)->exponent == (EXP)) && (SIGNIFICAND_GE_HEX_80(X, HI, LO))))
#define VALUE_LT_HEX_80(X,EXP,HI,LO) (((X)->exponent < (EXP)) || \
   (((X)->exponent == (EXP)) && (SIGNIFICAND_LT_HEX_80(X, HI, LO))))
#define VALUE_LE_HEX_80(X,EXP,HI,LO) (((X)->exponent < (EXP)) || \
   (((X)->exponent == (EXP)) && (SIGNIFICAND_LE_HEX_80(X, HI, LO))))

/* macros to compare two long doubles */

#define SIGNIFICAND_EQ_32(X,Y) ((X)->significand == (Y)->significand)
#define SIGNIFICAND_GT_32(X,Y) ((X)->significand > (Y)->significand)
#define SIGNIFICAND_GE_32(X,Y) ((X)->significand >= (Y)->significand)
#define SIGNIFICAND_LT_32(X,Y) ((X)->significand < (Y)->significand)
#define SIGNIFICAND_LE_32(X,Y) ((X)->significand <= (Y)->significand)

#if defined(SIZE_INT_32)
# define SIGNIFICAND_EQ_64(X,Y) \
  (((X)->hi_significand == (Y)->hi_significand) && ((X)->lo_significand == (Y)->lo_significand))
# define SIGNIFICAND_GT_64(X,Y) (((X)->hi_significand > (Y)->hi_significand) || \
  (((X)->hi_significand == (Y)->hi_significand) && ((X)->lo_significand >  (Y)->lo_significand)))
# define SIGNIFICAND_GE_64(X,Y) (((X)->hi_significand > (Y)->hi_significand) || \
  (((X)->hi_significand == (Y)->hi_significand) && ((X)->lo_significand >= (Y)->lo_significand)))
# define SIGNIFICAND_LT_64(X,Y) (((X)->hi_significand < (Y)->hi_significand) || \
  (((X)->hi_significand == (Y)->hi_significand) && ((X)->lo_significand <  (Y)->lo_significand)))
# define SIGNIFICAND_LE_64(X,Y) (((X)->hi_significand < (Y)->hi_significand) || \
  (((X)->hi_significand == (Y)->hi_significand) && ((X)->lo_significand <= (Y)->lo_significand)))
#elif defined(SIZE_INT_64)
# define SIGNIFICAND_EQ_64(X,Y) ((X)->significand == (Y)->significand)
# define SIGNIFICAND_GT_64(X,Y) ((X)->significand >  (Y)->significand)
# define SIGNIFICAND_GE_64(X,Y) ((X)->significand >= (Y)->significand)
# define SIGNIFICAND_LT_64(X,Y) ((X)->significand <  (Y)->significand)
# define SIGNIFICAND_LE_64(X,Y) ((X)->significand <= (Y)->significand)
#endif

#if defined(SIZE_INT_32)
# define SIGNIFICAND_EQ_80(X,Y) \
    (((X)->hi_significand == (Y)->hi_significand) && ((X)->lo_significand == (Y)->lo_significand))
# define SIGNIFICAND_GT_80(X,Y) (((X)->hi_significand > (Y)->hi_significand) || \
    (((X)->hi_significand == (Y)->hi_significand) && ((X)->lo_significand >  (Y)->lo_significand)))
# define SIGNIFICAND_GE_80(X,Y) (((X)->hi_significand > (Y)->hi_significand) || \
    (((X)->hi_significand == (Y)->hi_significand) && ((X)->lo_significand >= (Y)->lo_significand)))
# define SIGNIFICAND_LT_80(X,Y) (((X)->hi_significand < (Y)->hi_significand) || \
    (((X)->hi_significand == (Y)->hi_significand) && ((X)->lo_significand <  (Y)->lo_significand)))
# define SIGNIFICAND_LE_80(X,Y) (((X)->hi_significand < (Y)->hi_significand) || \
    (((X)->hi_significand == (Y)->hi_significand) && ((X)->lo_significand <= (Y)->lo_significand)))
#elif defined(SIZE_INT_64)
# define SIGNIFICAND_EQ_80(X,Y) ((X)->significand == (Y)->significand)
# define SIGNIFICAND_GT_80(X,Y) ((X)->significand >  (Y)->significand)
# define SIGNIFICAND_GE_80(X,Y) ((X)->significand >= (Y)->significand)
# define SIGNIFICAND_LT_80(X,Y) ((X)->significand <  (Y)->significand)
# define SIGNIFICAND_LE_80(X,Y) ((X)->significand <= (Y)->significand)
#endif

#define VALUE_EQ_32(X,Y) \
   (((X)->exponent == (Y)->exponent) && (SIGNIFICAND_EQ_32(X, Y)))
#define VALUE_GT_32(X,Y) (((X)->exponent > (Y)->exponent) || \
   (((X)->exponent == (Y)->exponent) && (SIGNIFICAND_GT_32(X, Y))))
#define VALUE_GE_32(X,Y) (((X)->exponent > (Y)->exponent) || \
   (((X)->exponent == (Y)->exponent) && (SIGNIFICAND_GE_32(X, Y))))
#define VALUE_LT_32(X,Y) (((X)->exponent < (Y)->exponent) || \
   (((X)->exponent == (Y)->exponent) && (SIGNIFICAND_LT_32(X, Y))))
#define VALUE_LE_32(X,Y) (((X)->exponent < (Y)->exponent) || \
   (((X)->exponent == (Y)->exponent) && (SIGNIFICAND_LE_32(X, Y))))

#define VALUE_EQ_64(X,Y) \
   (((X)->exponent == (Y)->exponent) && (SIGNIFICAND_EQ_64(X, Y)))
#define VALUE_GT_64(X,Y) (((X)->exponent > (Y)->exponent) || \
   (((X)->exponent == (Y)->exponent) && (SIGNIFICAND_GT_64(X, Y))))
#define VALUE_GE_64(X,Y) (((X)->exponent > (Y)->exponent) || \
   (((X)->exponent == (Y)->exponent) && (SIGNIFICAND_GE_64(X, Y))))
#define VALUE_LT_64(X,Y) (((X)->exponent < (Y)->exponent) || \
   (((X)->exponent == (Y)->exponent) && (SIGNIFICAND_LT_64(X, Y))))
#define VALUE_LE_64(X,Y) (((X)->exponent < (Y)->exponent) || \
   (((X)->exponent == (Y)->exponent) && (SIGNIFICAND_LE_64(X, Y))))

#define VALUE_EQ_80(X,Y) \
   (((X)->exponent == (Y)->exponent) && (SIGNIFICAND_EQ_80(X, Y)))
#define VALUE_GT_80(X,Y) (((X)->exponent > (Y)->exponent) || \
   (((X)->exponent == (Y)->exponent) && (SIGNIFICAND_GT_80(X, Y))))
#define VALUE_GE_80(X,Y) (((X)->exponent > (Y)->exponent) || \
   (((X)->exponent == (Y)->exponent) && (SIGNIFICAND_GE_80(X, Y))))
#define VALUE_LT_80(X,Y) (((X)->exponent < (Y)->exponent) || \
   (((X)->exponent == (Y)->exponent) && (SIGNIFICAND_LT_80(X, Y))))
#define VALUE_LE_80(X,Y) (((X)->exponent < (Y)->exponent) || \
   (((X)->exponent == (Y)->exponent) && (SIGNIFICAND_LE_80(X, Y))))

/* add/subtract 1 ulp macros */

#if defined(SIZE_INT_32)
# define ADD_ULP_80(X) \
    if ((++(X)->lo_significand == 0) && \
        (++(X)->hi_significand == (((X)->exponent == 0) ? 0x80000000 : 0))) \
    { \
        (X)->hi_significand |= 0x80000000; \
        ++(X)->exponent; \
    }
# define SUB_ULP_80(X) \
    if (--(X)->lo_significand == 0xFFFFFFFF) { \
        --(X)->hi_significand; \
        if (((X)->exponent != 0) && \
            ((X)->hi_significand == 0x7FFFFFFF) && \
            (--(X)->exponent != 0)) \
        { \
            (X)->hi_significand |= 0x80000000; \
        } \
    }
#elif defined(SIZE_INT_64)
# define ADD_ULP_80(X) \
    if (++(X)->significand == (((X)->exponent == 0) ? 0x8000000000000000 : 0))) { \
        (X)->significand |= 0x8000000000000000; \
        ++(X)->exponent; \
    }
# define SUB_ULP_80(X) \
    { \
        --(X)->significand; \
        if (((X)->exponent != 0) && \
            ((X)->significand == 0x7FFFFFFFFFFFFFFF) && \
            (--(X)->exponent != 0)) \
        { \
            (X)->significand |= 0x8000000000000000; \
        } \
    }
#endif


/* */

#define VOLATILE_32 /*volatile*/
#define VOLATILE_64 /*volatile*/
#define VOLATILE_80 /*volatile*/

#define QUAD_TYPE _Quad

#endif    /*__LIBM_SUPPORT_H_INCLUDED__*/
