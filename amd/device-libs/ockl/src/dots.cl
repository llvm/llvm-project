
/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "oclc.h"
#include "ockl.h"

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__attribute__((target("dot10-insts"), const)) static float amdgcn_fdot2(half2 a, half2 b, float c, bool s)
{ if (s) return __builtin_amdgcn_fdot2(a, b, c, true);
  else   return __builtin_amdgcn_fdot2(a, b, c, false); }

__attribute__((target("dot2-insts"), const)) static int amdgcn_sdot2(short2 a, short2 b, int c, bool s)
{ if (s) return __builtin_amdgcn_sdot2(a, b, c, true);
  else   return __builtin_amdgcn_sdot2(a, b, c, false); }

__attribute__((target("dot2-insts"), const)) static uint amdgcn_udot2(ushort2 a, ushort2 b, uint c, bool s)
{ if (s) return __builtin_amdgcn_udot2(a, b, c, true);
  else   return __builtin_amdgcn_udot2(a, b, c, false); }

__attribute__((target("dot1-insts"), const)) static int amdgcn_sdot4(int a, int b, int c, bool s)
{ if (s) return __builtin_amdgcn_sdot4(a, b, c, true);
  else   return __builtin_amdgcn_sdot4(a, b, c, false); }

__attribute__((target("dot7-insts"), const)) static uint amdgcn_udot4(uint a, uint b, uint c, bool s)
{ if (s) return __builtin_amdgcn_udot4(a, b, c, true);
  else   return __builtin_amdgcn_udot4(a, b, c, false); }

__attribute__((target("dot1-insts"), const)) static int amdgcn_sdot8(int a, int b, int c, bool s)
{ if (s) return __builtin_amdgcn_sdot8(a, b, c, true);
  else   return __builtin_amdgcn_sdot8(a, b, c, false); }

__attribute__((target("dot7-insts"), const)) static uint amdgcn_udot8(uint a, uint b, uint c, bool s)
{ if (s) return __builtin_amdgcn_udot8(a, b, c, true);
  else   return __builtin_amdgcn_udot8(a, b, c, false); }


__attribute__((target("dot8-insts"), const)) static uint amdgcn_sudot4(bool as, uint a, bool bs, uint b, uint c, bool s)
{
  if (!as && !bs && !s) return __builtin_amdgcn_sudot4(false, a, false, b, c, false);
  if (!as && !bs &&  s) return __builtin_amdgcn_sudot4(false, a, false, b, c, true );
  if (!as &&  bs && !s) return __builtin_amdgcn_sudot4(false, a, true , b, c, false);
  if (!as &&  bs &&  s) return __builtin_amdgcn_sudot4(false, a, true , b, c, true );
  if ( as && !bs && !s) return __builtin_amdgcn_sudot4(true , a, false, b, c, false);
  if ( as && !bs &&  s) return __builtin_amdgcn_sudot4(true , a, false, b, c, true );
  if ( as &&  bs && !s) return __builtin_amdgcn_sudot4(true , a, true , b, c, false);
                        return __builtin_amdgcn_sudot4(true , a, true , b, c, true );
}

__attribute__((target("dot8-insts"), const)) static uint amdgcn_sudot8(bool as, uint a, bool bs, uint b, uint c, bool s)
{
  if (!as && !bs && !s) return __builtin_amdgcn_sudot8(false, a, false, b, c, false);
  if (!as && !bs &&  s) return __builtin_amdgcn_sudot8(false, a, false, b, c, true );
  if (!as &&  bs && !s) return __builtin_amdgcn_sudot8(false, a, true , b, c, false);
  if (!as &&  bs &&  s) return __builtin_amdgcn_sudot8(false, a, true , b, c, true );
  if ( as && !bs && !s) return __builtin_amdgcn_sudot8(true , a, false, b, c, false);
  if ( as && !bs &&  s) return __builtin_amdgcn_sudot8(true , a, false, b, c, true );
  if ( as &&  bs && !s) return __builtin_amdgcn_sudot8(true , a, true , b, c, false);
                        return __builtin_amdgcn_sudot8(true , a, true , b, c, true );
}

#define SWDOT __oclc_ISA_version < 9006 || __oclc_ISA_version == 9009 || __oclc_ISA_version == 10100
#define SWIDOT2 __oclc_ISA_version < 9006 || __oclc_ISA_version == 9009 || __oclc_ISA_version == 10100 || __oclc_ISA_version >= 11000
#define SUDOT __oclc_ISA_version >= 11000

#define AS_INT(X) __builtin_astype(X, int)
#define AS_UINT(X) __builtin_astype(X, uint)
#define ATTR __attribute__((const))

ATTR static float
fmuladd(float a, float b, float c)
{
    #pragma OPENCL FP_CONTRACT ON
    return a * b + c;
}

ATTR float
__ockl_fdot2(half2 a, half2 b, float c, bool s)
{
    if (SWDOT)
        return fmuladd((float)a.s1, (float)b.s1, fmuladd((float)a.s0, (float)b.s0, c));
    else
        return amdgcn_fdot2(a, b, c, true);
}

ATTR int
__ockl_sdot2(short2 a, short2 b, int c, bool s)
{
    if (SWIDOT2) {
        int p0 = (int)a.s0 * (int)b.s0;
        int p1 = (int)a.s1 * (int)b.s1;
        long r = (long)c + (long)p0 + (long)p1;

        if (s)
            return r < -2147483648L ? -2147483648 :
                   (r > 2147483647L ? 2147483647 : (int)r);
        else
            return (int)r;
    } else {
        return amdgcn_sdot2(a, b, c, s);
    }
}

ATTR uint
__ockl_udot2(ushort2 a, ushort2 b, uint c, bool s)
{
    if (SWIDOT2) {
        uint p0 = (uint)a.s0 * (uint)b.s0;
        uint p1 = (uint)a.s1 * (uint)b.s1;
        ulong r = (ulong)c + (ulong)p0 + (ulong)p1;
        return (s & (r > (ulong)0xffffffff)) ? 0xffffffff : (uint)r;
    } else {
        return amdgcn_udot2(a, b, c, s);
    }
}


ATTR int
__ockl_sdot4(char4 a, char4 b, int c, bool s)
{
    if (SWDOT) {
        int t =
            (int)a.s0 * (int)b.s0 +
            (int)a.s1 * (int)b.s1 +
            (int)a.s2 * (int)b.s2 +
            (int)a.s3 * (int)b.s3;
        return s ? __ockl_add_sat_i32(t, c) : (t + c);
    } else {
        if (SUDOT) return amdgcn_sudot4(true, AS_INT(a), true, AS_INT(b), c, s);
        else       return amdgcn_sdot4(AS_INT(a), AS_INT(b), c, s);
    }
}

ATTR uint
__ockl_udot4(uchar4 a, uchar4 b, uint c, bool s)
{
    if (SWDOT) {
        uint t =
            (uint)a.s0 * (uint)b.s0 +
            (uint)a.s1 * (uint)b.s1 +
            (uint)a.s2 * (uint)b.s2 +
            (uint)a.s3 * (uint)b.s3;
        return s ? __ockl_add_sat_u32(t, c) : (t + c);
    } else {
        return amdgcn_udot4(AS_UINT(a), AS_UINT(b), c, s);
    }
}


ATTR int
__ockl_sdot8(int a, int b, int c, bool s)
{
    if (SWDOT) {
        int t =
            ((a << 28) >> 28) * ((b << 28) >> 28) +
            ((a << 24) >> 28) * ((b << 24) >> 28) +
            ((a << 20) >> 28) * ((b << 20) >> 28) +
            ((a << 16) >> 28) * ((b << 16) >> 28) +
            ((a << 12) >> 28) * ((b << 12) >> 28) +
            ((a <<  8) >> 28) * ((b <<  8) >> 28) +
            ((a <<  4) >> 28) * ((b <<  4) >> 28) +
            ( a        >> 28) * ( b        >> 28);
        return s ? __ockl_add_sat_i32(t, c) : (t + c);
    } else {
        if (SUDOT) return amdgcn_sudot8(true, a, true, b, c, s);
        else       return amdgcn_sdot8(a, b, c, s);
    }
}

ATTR uint
__ockl_udot8(uint a, uint b, uint c, bool s)
{
    if (SWDOT) {
        uint t =
            ( a        & 0xf) * ( b        & 0xf) +
            ((a >>  4) & 0xf) * ((b >>  4) & 0xf) +
            ((a >>  8) & 0xf) * ((b >>  8) & 0xf) +
            ((a >> 12) & 0xf) * ((b >> 12) & 0xf) +
            ((a >> 16) & 0xf) * ((b >> 16) & 0xf) +
            ((a >> 20) & 0xf) * ((b >> 20) & 0xf) +
            ((a >> 24) & 0xf) * ((b >> 24) & 0xf) +
            ((a >> 28)      ) * ((b >> 28)      );
        return s ? __ockl_add_sat_u32(t, c) : (t + c);
    } else {
        return amdgcn_udot8(a, b, c, s);
    }
}

