/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
#include "mth_intrinsics.h"

vrs4_t
__ZGVxN4v__mth_i_vr4(vrs4_t x, float func(float))
{
  int i;
  vrs4_t r;
  for (i = 0; i < 4; i++) {
    r[i] = func(x[i]);
  }
  return r;
}

vrs4_t
__ZGVxM4v__mth_i_vr4(vrs4_t x, vis4_t mask, float func(float))
{
  int i;
  vrs4_t r;
  for (i = 0; i < 4; i++) {
    if (mask[i])
      r[i] = func(x[i]);
  }
  return r;
}

vrs4_t
__ZGVxN4vv__mth_i_vr4vr4(vrs4_t x, vrs4_t y, float func(float, float))
{
  int i;
  vrs4_t r;
  for (i = 0; i < 4; i++) {
    r[i] = func(x[i], y[i]);
  }
  return r;
}

vrs4_t
__ZGVxM4vv__mth_i_vr4vr4(vrs4_t x, vrs4_t y, vis4_t mask, float func(float, float))
{
  int i;
  vrs4_t r;
  for (i = 0; i < 4; i++) {
    if (mask[i])
      r[i] = func(x[i], y[i]);
  }
  return r;
}

vrd2_t
__ZGVxN2v__mth_i_vr8(vrd2_t x, double func(double))
{
  int i;
  vrd2_t r;
  for (i = 0; i < 2; i++) {
    r[i] = func(x[i]);
  }
  return r;
}

vrd2_t
__ZGVxM2v__mth_i_vr8(vrd2_t x, vid2_t mask, double func(double))
{
  int i;
  vrd2_t r;
  for (i = 0; i < 2; i++) {
    if (mask[i])
      r[i] = func(x[i]);
  }
  return r;
}

vrd2_t
__ZGVxN2vv__mth_i_vr8vr8(vrd2_t x, vrd2_t y, double func(double, double))
{
  int i;
  vrd2_t r;
  for (i = 0; i < 2; i++) {
    r[i] = func(x[i], y[i]);
  }
  return r;
}

vrd2_t
__ZGVxM2vv__mth_i_vr8vr8(vrd2_t x, vrd2_t y, vid2_t mask, double func(double, double))
{
  int i;
  vrd2_t r;
  for (i = 0; i < 2; i++) {
    if (mask[i])
      r[i] = func(x[i], y[i]);
  }
  return r;
}

vrs4_t
__ZGVxN4v__mth_i_vr4si4(vrs4_t x, int32_t iy, float func(float, int32_t))
{
  int i;
  vrs4_t r;
  for (i = 0 ; i < 4 ; i++) {
    r[i] = func(x[i], iy);
  }
  return r;
}

vrs4_t
__ZGVxM4v__mth_i_vr4si4(vrs4_t x, int32_t iy, vis4_t mask, float func(float, int32_t))
{
  int i;
  vrs4_t r;
  for (i = 0 ; i < 4 ; i++) {
    if (mask[i])
      r[i] = func(x[i], iy);
  }
  return r;
}

vrs4_t
__ZGVxN4vv__mth_i_vr4vi4(vrs4_t x, vis4_t iy, float func(float, int32_t))
{
  int i;
  vrs4_t r;
  for (i = 0 ; i < 4 ; i++) {
    r[i] = func(x[i], iy[i]);
  }
  return r;
}

vrs4_t
__ZGVxM4vv__mth_i_vr4vi4(vrs4_t x, vis4_t iy, vis4_t mask, float func(float, int32_t))
{
  int i;
  vrs4_t r;
  for (i = 0 ; i < 4 ; i++) {
    if (mask[i])
      r[i] = func(x[i], iy[i]);
  }
  return r;
}

vrs4_t
__ZGVxN4v__mth_i_vr4si8(vrs4_t x, long long iy, float func(float, long long))
{
  int i;
  vrs4_t r;
  for (i = 0 ; i < 4 ; i++) {
    r[i] = func(x[i], iy);
  }
  return r;
}

vrs4_t
__ZGVxM4v__mth_i_vr4si8(vrs4_t x, long long iy, vis4_t mask, float func(float, long long))
{
  int i;
  vrs4_t r;
  for (i = 0 ; i < 4 ; i++) {
    if (mask[i])
      r[i] = func(x[i], iy);
  }
  return r;
}

vrs4_t
__ZGVxN4vv__mth_i_vr4vi8(vrs4_t x, vid2_t iyu, vid2_t iyl, float func(float, long long))
{
  int i;
  vrs4_t r;
  for (i = 0 ; i < 2 ; i++) {
    r[i] = func(x[i], iyu[i]);
  }
  for (i = 2 ; i < 4 ; i++) {
    r[i] = func(x[i], iyl[i-2]);
  }
  return r;
}

vrs4_t
__ZGVxM4vv__mth_i_vr4vi8(vrs4_t x, vid2_t iyu, vid2_t iyl, vis4_t mask, float func(float, long long))
{
  int i;
  vrs4_t r;
  for (i = 0 ; i < 2 ; i++) {
    if (mask[i])
      r[i] = func(x[i], iyu[i]);
  }
  for (i = 2 ; i < 4 ; i++) {
    if (mask[i])
      r[i] = func(x[i], iyl[i-2]);
  }
  return r;
}


//---------------


vrd2_t
__ZGVxN2v__mth_i_vr8si4(vrd2_t x, int32_t iy, double func(double, int32_t))
{
  int i;
  vrd2_t r;
  for (i = 0 ; i < 2 ; i++) {
    r[i] = func(x[i], iy);
  }
  return r;
}

vrd2_t
__ZGVxM2v__mth_i_vr8si4(vrd2_t x, int32_t iy, vid2_t mask, double func(double, int32_t))
{
  int i;
  vrd2_t r;
  for (i = 0 ; i < 2 ; i++) {
    if (mask[i])
      r[i] = func(x[i], iy);
  }
  return r;
}

/*
 * __ZGVxN2vv__mth_i_vr8vi4 and __ZGVxM2vv__mth_i_vr8vi4 should
 * be defined as:
 * __ZGVxN2vv__mth_i_vr8vi4(vrd2_t x, vis2_t iy, double func(double, int32_t))
 * __ZGVxM2vv__mth_i_vr8vi4(vrd2_t x, vis2_t iy, vid2_t mask, double func(double, int32_t))
 *
 * But the POWER architectures needs the 32-bit integer vectors to
 * be the full 128-bits of a vector register.
 */

vrd2_t
__ZGVxN2vv__mth_i_vr8vi4(vrd2_t x, vis4_t iy, double func(double, int32_t))
{
  int i;
  vrd2_t r;
  for (i = 0 ; i < 2 ; i++) {
    r[i] = func(x[i], iy[i]);
  }
  return r;
}

vrd2_t
__ZGVxM2vv__mth_i_vr8vi4(vrd2_t x, vis4_t iy, vid2_t mask, double func(double, int32_t))
{
  int i;
  vrd2_t r;
  for (i = 0 ; i < 2 ; i++) {
    if (mask[i])
      r[i] = func(x[i], iy[i]);
  }
  return r;
}

vrd2_t
__ZGVxN2v__mth_i_vr8si8(vrd2_t x, long long iy, double func(double, long long))
{
  int i;
  vrd2_t r;
  for (i = 0 ; i < 2 ; i++) {
    r[i] = func(x[i], iy);
  }
  return r;
}

vrd2_t
__ZGVxM2v__mth_i_vr8si8(vrd2_t x, long long iy, vid2_t mask, double func(double, long long))
{
  int i;
  vrd2_t r;
  for (i = 0 ; i < 2 ; i++) {
    if (mask[i])
      r[i] = func(x[i], iy);
  }
  return r;
}

vrd2_t
__ZGVxN2vv__mth_i_vr8vi8(vrd2_t x, vid2_t iy, double func(double, long long))
{
  int i;
  vrd2_t r;
  for (i = 0 ; i < 2 ; i++) {
    r[i] = func(x[i], iy[i]);
  }
  return r;
}

vrd2_t
__ZGVxM2vv__mth_i_vr8vi8(vrd2_t x, vid2_t iy, vid2_t mask, double func(double, long long))
{
  int i;
  vrd2_t r;
  for (i = 0 ; i < 2 ; i++) {
    if (mask[i])
      r[i] = func(x[i], iy[i]);
  }
  return r;
}


vcs1_t
__ZGVxN1v__mth_i_vc4(vcs1_t x, float _Complex func(float _Complex))
{
  int i;
  float _Complex tx;
  *(vcs1_t *)&tx = x;
  tx = func(tx);
  return *(vcs1_t *)&tx;
}

vcs1_t
__ZGVxN1vv__mth_i_vc4vc4(vcs1_t x, vcs1_t y, float _Complex func(float _Complex, float _Complex))
{
  int i;
  float _Complex tx;
  float _Complex ty;
  *(vcs1_t *)&tx = x;
  *(vcs1_t *)&ty = y;
  tx = func(tx, ty);
  return *(vcs1_t *)&tx;
}

vcs2_t
__ZGVxN2v__mth_i_vc4(vcs2_t x, float _Complex func(float _Complex))
{
  int i;
  float _Complex tx[2];
  *(vcs2_t *)&tx = x;
  for (i = 0 ; i < 2 ; i++) {
    tx[i] = func(tx[i]);
  }
  return *(vcs2_t *)&tx;
}

vcs2_t
__ZGVxN2vv__mth_i_vc4vc4(vcs2_t x, vcs2_t y, float _Complex func(float _Complex, float _Complex))
{
  int i;
  float _Complex tx[2];
  float _Complex ty[2];
  *(vcs2_t *)&tx = x;
  *(vcs2_t *)&ty = y;
  for (i = 0 ; i < 2 ; i++) {
    tx[i] = func(tx[i], ty[i]);
  }
  return *(vcs2_t *)&tx;
}

vcd1_t
__ZGVxN1v__mth_i_vc8(vcd1_t x, double _Complex func(double _Complex))
{
  int i;
  double _Complex tx;
  *(vcd1_t *)&tx = x;
  tx = func(tx);
  return *(vcd1_t *)&tx;
}

vcd1_t
__ZGVxN1vv__mth_i_vc8vc8(vcd1_t x, vcd1_t y, double _Complex func(double _Complex, double _Complex))
{
  int i;
  double _Complex tx;
  double _Complex ty;
  *(vcd1_t *)&tx = x;
  *(vcd1_t *)&ty = y;
  tx = func(tx, ty);
  return *(vcd1_t *)&tx;
}

vcs1_t
__ZGVxN1v__mth_i_vc4si4(vcs1_t x, int32_t iy, float _Complex func(float _Complex, int32_t))
{
  int i;
  float _Complex tx;
  *(vcs1_t *)&tx = x;
  tx = func(tx, iy);
  return *(vcs1_t *)&tx;
}

vcs1_t
__ZGVxN1v__mth_i_vc4si8(vcs1_t x, long long iy, float _Complex func(float _Complex, long long))
{
  int i;
  float _Complex tx;
  *(vcs1_t *)&tx = x;
  tx = func(tx, iy);
  return *(vcs1_t *)&tx;
}

vcd1_t
__ZGVxN1v__mth_i_vc8si4(vcd1_t x, int32_t iy, double _Complex func(double _Complex, int32_t))
{
  int i;
  double _Complex tx;
  *(vcd1_t *)&tx = x;
  tx = func(tx, iy);
  return *(vcd1_t *)&tx;
}

vcd1_t
__ZGVxN1v__mth_i_vc8si8(vcd1_t x, long long iy, double _Complex func(double _Complex, long long))
{
  int i;
  double _Complex tx;
  *(vcd1_t *)&tx = x;
  tx = func(tx, iy);
  return *(vcd1_t *)&tx;
}
