/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ockl.h"

#define ATTR __attribute__((overloadable, const))

#define F OCKL_MANGLE_U32(bfe)

#define L2 F(a.s0, b.s0, c.s0), F(a.s1, b.s1, c.s1)
#define L3 L2, F(a.s2, b.s2, c.s2)
#define L4 L3, F(a.s3, b.s3, c.s3)
#define L8 L4, F(a.s4, b.s4, c.s4), F(a.s5, b.s5, c.s5), F(a.s6, b.s6, c.s6), F(a.s7, b.s7, c.s7)
#define L16 L8, F(a.s8, b.s8, c.s8), F(a.s9, b.s9, c.s9), F(a.sa, b.sa, c.sa), F(a.sb, b.sb, c.sb), \
                F(a.sc, b.sc, c.sc), F(a.sd, b.sd, c.sd), F(a.se, b.se, c.se), F(a.sf, b.sf, c.sf)


#define GEN(N) \
ATTR uint##N \
amd_bfe(uint##N a, uint##N b, uint##N c) \
{ \
    return (uint##N)( L##N ); \
}

GEN(16)
GEN(8)
GEN(4)
GEN(3)
GEN(2)

ATTR uint amd_bfe(uint a, uint b, uint c) { return F(a, b, c); }

