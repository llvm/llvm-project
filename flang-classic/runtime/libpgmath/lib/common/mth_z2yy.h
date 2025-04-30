/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/*
 * Common set of interface routines to convert an intrinsic math library call
 * using Intel AVX-512 vectors in to two calls of the corresponding AVX2
 * implementation.
 *
 * Note: code is common to both AVX-512 and KNL architectures.
 *       Thus, have to use Intel intrinsics that are common to both systems.
 */


static
vrs16_t
__attribute__((noinline))
__gs_z2yy_x(vrs16_t x, vrs8_t(*func)(vrs8_t))
{ 
  vrs8_t rl, ru;
  ru = func((vrs8_t) _mm512_extractf64x4_pd((__m512d)x, 1));
  rl = func((vrs8_t) _mm512_castps512_ps256(x));
  return (vrs16_t) _mm512_insertf64x4(_mm512_castpd256_pd512((__m256d)rl),
                                      (__m256d)ru, 1);
}

static
vrs16_t
__attribute__((noinline))
__gs_z2yy_xy(vrs16_t x, vrs16_t y, vrs8_t(*func)(vrs8_t, vrs8_t))
{ 
  vrs8_t rl, ru;
  ru = func((vrs8_t) _mm512_extractf64x4_pd((__m512d)x, 1),
            (vrs8_t) _mm512_extractf64x4_pd((__m512d)y, 1));
  rl = func((vrs8_t) _mm512_castps512_ps256(x),
            (vrs8_t) _mm512_castps512_ps256(y));
  return (vrs16_t) _mm512_insertf64x4(_mm512_castpd256_pd512((__m256d)rl),
                                      (__m256d)ru, 1);
}

static
vrs16_t
__attribute__((noinline))
__gs_z2yy_sincos(vrs16_t x, vrs8_t(*func)(vrs8_t))
{ 
  vrs8_t su, sl, cu;
  su = func((vrs8_t) _mm512_extractf64x4_pd((__m512d)x, 1));
  asm("vmovaps  %%ymm1, %0" : :"m"(cu) :);
  sl = func((vrs8_t) _mm512_castps512_ps256(x));
  asm("vinsertf64x4 $0x1,%0,%%zmm1,%%zmm1" : : "m"(cu) : );
  return (vrs16_t) _mm512_insertf64x4(_mm512_castpd256_pd512((__m256d)sl),
                                      (__m256d)su, 1);
}

static
vrs16_t
__attribute__((noinline))
__gs_z2yy_xk1(vrs16_t x, int64_t iy, vrs8_t(*func)(vrs8_t, int64_t))
{
  vrs8_t rl, ru;
  ru = func((vrs8_t) _mm512_extractf64x4_pd((__m512d)x, 1), iy);
  rl = func((vrs8_t) _mm512_castps512_ps256(x), iy);
  return (vrs16_t) _mm512_insertf64x4(_mm512_castpd256_pd512((__m256d)rl),
                                      (__m256d)ru, 1);
}

static
vrs16_t
__attribute__((noinline))
__gs_z2yy_xi(vrs16_t x, vis16_t iy, vrs8_t(*func)(vrs8_t, vis8_t))
{
  vrs8_t rl, ru;
  ru = func((vrs8_t) _mm512_extractf64x4_pd((__m512d)x, 1),
            (vis8_t) _mm512_extractf64x4_pd((__m512d)iy, 1));
  rl = func((vrs8_t) _mm512_castps512_ps256(x),
            (vis8_t) _mm512_castps512_ps256((__m512)iy));
  return (vrs16_t) _mm512_insertf64x4(_mm512_castpd256_pd512((__m256d)rl),
                                     (__m256d)ru, 1);
}

static
vrs16_t
__attribute__((noinline))
__gs_z2yy_xk(vrs16_t x, vid8_t iyu, vid8_t iyl, vrs8_t(*func)(vrs8_t, vid4_t, vid4_t))
{
  vrs8_t rl, ru;
  ru = func((vrs8_t) _mm512_extractf64x4_pd((__m512d)x, 1),
            (vid4_t) _mm512_extractf64x4_pd((__m512d)iyu, 1),
            (vid4_t) _mm512_extractf64x4_pd((__m512d)iyu, 0));
  rl = func((vrs8_t) _mm512_castps512_ps256(x),
            (vid4_t) _mm512_extractf64x4_pd((__m512d)iyl, 1),
            (vid4_t) _mm512_extractf64x4_pd((__m512d)iyl, 0));
  return (vrs16_t) _mm512_insertf64x4(_mm512_castpd256_pd512((__m256d)rl),
                                     (__m256d)ru, 1);
}

static
vrd8_t
__attribute__((noinline))
__gd_z2yy_x(vrd8_t x, vrd4_t(*func)(vrd4_t))
{
  vrd4_t rl, ru;
  ru = func((vrd4_t) _mm512_extractf64x4_pd((__m512d)x, 1));
  rl = func((vrd4_t) _mm512_castpd512_pd256(x));
  return (vrd8_t) _mm512_insertf64x4(_mm512_castpd256_pd512((__m256d)rl),
                                     (__m256d)ru, 1);
}

static
vrd8_t
__attribute__((noinline))
__gd_z2yy_xy(vrd8_t x, vrd8_t y, vrd4_t(*func)(vrd4_t, vrd4_t))
{
  vrd4_t rl, ru;
  ru = func((vrd4_t) _mm512_extractf64x4_pd((__m512d)x, 1),
            (vrd4_t) _mm512_extractf64x4_pd((__m512d)y, 1));
  rl = func((vrd4_t) _mm512_castpd512_pd256(x),
            (vrd4_t) _mm512_castpd512_pd256(y));
  return (vrd8_t) _mm512_insertf64x4(_mm512_castpd256_pd512((__m256d)rl),
                                     (__m256d)ru, 1);
}

static
vrd8_t
__attribute__((noinline))
__gd_z2yy_sincos(vrd8_t x, vrd4_t(*func)(vrd4_t))
{ 
  vrd4_t su, sl, cu;
  su = func((vrd4_t) _mm512_extractf64x4_pd((__m512d)x, 1));
  asm("vmovaps  %%ymm1, %0" : :"m"(cu) :);
  sl = func((vrd4_t) _mm512_castpd512_pd256(x));
  asm("vinsertf64x4 $0x1,%0,%%zmm1,%%zmm1" : : "m"(cu) : );
  return (vrd8_t) _mm512_insertf64x4(_mm512_castpd256_pd512((__m256d)sl),
                                      (__m256d)su, 1);
}

static
vrd8_t
__attribute__((noinline))
__gd_z2yy_xk1(vrd8_t x, int64_t iy, vrd4_t(*func)(vrd4_t, int64_t))
{
  vrd4_t rl, ru;
  ru = func((vrd4_t) _mm512_extractf64x4_pd((__m512d)x, 1), iy);
  rl = func((vrd4_t) _mm512_castpd512_pd256(x), iy);
  return (vrd8_t) _mm512_insertf64x4(_mm512_castpd256_pd512((__m256d)rl),
                                     (__m256d)ru, 1);
}

static
vrd8_t
__attribute__((noinline))
__gd_z2yy_xk(vrd8_t x, vid8_t iy, vrd4_t(*func)(vrd4_t, vid4_t))
{
  vrd4_t rl, ru;
  ru = func((vrd4_t) _mm512_extractf64x4_pd((__m512d)x, 1),
            (vid4_t) _mm512_extractf64x4_pd((__m512d)iy, 1));
  rl = func((vrd4_t) _mm512_castpd512_pd256(x),
            (vid4_t) _mm512_castpd512_pd256((__m512d)iy));
  return (vrd8_t) _mm512_insertf64x4(_mm512_castpd256_pd512((__m256d)rl),
                                     (__m256d)ru, 1);
}

static
vrd8_t
__attribute__((noinline))
__gd_z2yy_xi(vrd8_t x, vis8_t iy, vrd4_t(*func)(vrd4_t, vis4_t))
{
  vrd4_t rl, ru;
  ru = func((vrd4_t) _mm512_extractf64x4_pd((__m512d)x, 1),
            (vis4_t) _mm256_extractf128_si256((__m256i)iy, 1));
  rl = func((vrd4_t) _mm512_castpd512_pd256(x),
            (vis4_t) _mm256_castsi256_si128((__m256i)iy));
  return (vrd8_t) _mm512_insertf64x4(_mm512_castpd256_pd512((__m256d)rl),
                                     (__m256d)ru, 1);
}

