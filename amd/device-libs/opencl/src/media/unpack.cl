/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ockl.h"

#define ATTR __attribute__((overloadable, always_inline, const))

#define _F(N) __ockl_unpack##N##_f32
#define F(N) _F(N)

#define L2(N) F(N)(a.s0), F(N)(a.s1)
#define L3(N) L2(N), F(N)(a.s2)
#define L4(N) L3(N), F(N)(a.s3)
#define L8(N) L4(N), F(N)(a.s4), F(N)(a.s5), F(N)(a.s6), F(N)(a.s7)
#define L16(N) L8(N), F(N)(a.s8), F(N)(a.s9), F(N)(a.sa), F(N)(a.sb), F(N)(a.sc), F(N)(a.sd), F(N)(a.se), F(N)(a.sf)

#define GENN(N,B) \
ATTR float##N \
amd_unpack##B(uint##N a) \
{ \
    return (float##N)( L##N(B) ); \
}

#define GEN(B) \
    GENN(16,B) \
    GENN(8,B) \
    GENN(4,B) \
    GENN(3,B) \
    GENN(2,B)
           
GEN(0)
GEN(1)
GEN(2)
GEN(3)

ATTR float amd_unpack0(uint a) { return F(0)(a); }
ATTR float amd_unpack1(uint a) { return F(1)(a); }
ATTR float amd_unpack2(uint a) { return F(2)(a); }
ATTR float amd_unpack3(uint a) { return F(3)(a); }

