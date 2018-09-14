/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#define ATTR __attribute__((const))

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

//-------- T __nv_abs
ATTR int __nv_abs(int x) { return abs(x); }

//-------- T __nv_llabs
ATTR long __nv_llabs(long x) { return abs(x); }

//-------- T __nv_max
ATTR int __nv_max(int a, int b) { return MAX(a,b); }

//-------- T __nv_llmax
ATTR long __nv_llmax(long a, long b) { return MAX(a,b); }

//-------- T __nv_ullmax
ATTR ulong __nv_ullmax(ulong a, ulong b) { return MAX(a,b); }

//-------- T __nv_umax
ATTR uint __nv_umax(uint a, uint b) { return MAX(a,b); }

//-------- T __nv_min
ATTR int __nv_min(int a, int b) { return MIN(a,b); }

//-------- T __nv_llmin
ATTR long __nv_llmin(long a, long b) { return MIN(a,b); }

//-------- T __nv_ullmin
ATTR ulong __nv_ullmin(ulong a, ulong b) { return MIN(a,b); }

//-------- T __nv_umin
ATTR uint __nv_umin(uint a, uint b) { return MIN(a,b); }

//-------- T __nv_sad
ATTR uint __nv_sad(int x, int y, uint z)
{
    return (z+abs(x-y));
}

//-------- T __nv_usad
ATTR uint __nv_usad(uint x, uint y, uint z)
{
    return (z+abs(x-y));
}

