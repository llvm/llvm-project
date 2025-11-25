/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ockl.h"

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define _C(X,Y) X ## Y
#define C(X,Y) _C(X,Y)

#define red_full reduce
#define scan_full scan

#define PFX __ockl_wf
#define ATTR __attribute__((overloadable))

#define i32_tn int
#define u32_tn uint
#define i64_tn long
#define u64_tn ulong
#define f32_tn float
#define f64_tn double
#define f16_tn half

#define true_inc inclusive_
#define false_inc exclusive_

#define GENROT(O,T) \
ATTR T##_tn \
C(sub_group_reduce_,O)(T##_tn x) \
{ \
    return C(PFX,C(red_,C(O,C(_,T))))(x); \
}

#define GENRO(O) \
    GENROT(O,i32) \
    GENROT(O,u32) \
    GENROT(O,i64) \
    GENROT(O,u64) \
    GENROT(O,f32) \
    GENROT(O,f64) \
    GENROT(O,f16)

GENRO(add)
GENRO(max)
GENRO(min)

#define GENSOTI(O, T, I) \
ATTR T##_tn \
C(sub_group_scan_,C(I##_inc,O))(T##_tn x) \
{ \
    return C(PFX,C(scan_,C(O,C(_,T))))(x, I); \
}

#define GENSOT(O,T) \
    GENSOTI(O,T,false) \
    GENSOTI(O,T,true)

#define GENSO(O) \
    GENSOT(O,i32) \
    GENSOT(O,u32) \
    GENSOT(O,i64) \
    GENSOT(O,u64) \
    GENSOT(O,f32) \
    GENSOT(O,f64) \
    GENSOT(O,f16)

GENSO(add)
GENSO(max)
GENSO(min)

