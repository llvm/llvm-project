/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
#include "mth_intrinsics.h"

vrs16_t
__ZGVzN16v__mth_i_vr4(vrs16_t x, float func(float))
{
  int i;
  vrs16_t r;
  for (i = 0; i < 16; i++) {
    r[i] = func(x[i]);
  }
  return r;
}

vrs16_t
__ZGVzM16v__mth_i_vr4(vrs16_t x, vis16_t mask, float func(float))
{
  int i;
  vrs16_t r;
  for (i = 0; i < 16; i++) {
    if (mask[i])
      r[i] = func(x[i]);
  }
  return r;
}

vrs16_t
__ZGVzN16vv__mth_i_vr4vr4(vrs16_t x, vrs16_t y, float func(float, float))
{
  int i;
  vrs16_t r;
  for (i = 0; i < 16; i++) {
    r[i] = func(x[i], y[i]);
  }
  return r;
}

vrs16_t
__ZGVzM16vv__mth_i_vr4vr4(vrs16_t x, vrs16_t y, vis16_t mask, float func(float, float))
{
  int i;
  vrs16_t r;
  for (i = 0; i < 16; i++) {
    if (mask[i])
      r[i] = func(x[i], y[i]);
  }
  return r;
}

vrd8_t
__ZGVzN8v__mth_i_vr8(vrd8_t x, double func(double))
{
  int i;
  vrd8_t r;
  for (i = 0; i < 8; i++) {
    r[i] = func(x[i]);
  }
  return r;
}

vrd8_t
__ZGVzM8v__mth_i_vr8(vrd8_t x, vid8_t mask, double func(double))
{
  int i;
  vrd8_t r;
  for (i = 0; i < 8; i++) {
    if (mask[i])
      r[i] = func(x[i]);
  }
  return r;
}

vrd8_t
__ZGVzN8vv__mth_i_vr8vr8(vrd8_t x, vrd8_t y, double func(double, double))
{
  int i;
  vrd8_t r;
  for (i = 0; i < 8; i++) {
    r[i] = func(x[i], y[i]);
  }
  return r;
}

vrd8_t
__ZGVzM8vv__mth_i_vr8vr8(vrd8_t x, vrd8_t y, vid8_t mask, double func(double, double))
{
  int i;
  vrd8_t r;
  for (i = 0; i < 8; i++) {
    if (mask[i])
      r[i] = func(x[i], y[i]);
  }
  return r;
}

vrs16_t
__ZGVzN16v__mth_i_vr4si4(vrs16_t x, int32_t iy, float func(float, int32_t))
{
  int i;
  vrs16_t r;
  for (i = 0 ; i < 16 ; i++) {
    r[i] = func(x[i], iy);
  }
  return r;
}

vrs16_t
__ZGVzM16v__mth_i_vr4si4(vrs16_t x, int32_t iy, vis16_t mask, float func(float, int32_t))
{
  int i;
  vrs16_t r;
  for (i = 0 ; i < 16 ; i++) {
    if (mask[i])
      r[i] = func(x[i], iy);
  }
  return r;
}

vrs16_t
__ZGVzN16vv__mth_i_vr4vi4(vrs16_t x, vis16_t iy, float func(float, int32_t))
{
  int i;
  vrs16_t r;
  for (i = 0 ; i < 16 ; i++) {
    r[i] = func(x[i], iy[i]);
  }
  return r;
}

vrs16_t
__ZGVzM16vv__mth_i_vr4vi4(vrs16_t x, vis16_t iy, vis16_t mask, float func(float, int32_t))
{
  int i;
  vrs16_t r;
  for (i = 0 ; i < 16 ; i++) {
    if (mask[i])
      r[i] = func(x[i], iy[i]);
  }
  return r;
}

vrs16_t
__ZGVzN16v__mth_i_vr4si8(vrs16_t x, long long iy, float func(float, long long))
{
  int i;
  vrs16_t r;
  for (i = 0 ; i < 16 ; i++) {
    r[i] = func(x[i], iy);
  }
  return r;
}

vrs16_t
__ZGVzM16v__mth_i_vr4si8(vrs16_t x, long long iy, vis16_t mask, float func(float, long long))
{
  int i;
  vrs16_t r;
  for (i = 0 ; i < 16 ; i++) {
    if (mask[i])
      r[i] = func(x[i], iy);
  }
  return r;
}

vrs16_t
__ZGVzN16vv__mth_i_vr4vi8(vrs16_t x, vid8_t iyu, vid8_t iyl, float func(float, long long))
{
  int i;
  vrs16_t r;
  for (i = 0 ; i < 8 ; i++) {
    r[i] = func(x[i], iyu[i]);
  }
  for (i = 8 ; i < 16 ; i++) {
    r[i] = func(x[i], iyl[i-8]);
  }
  return r;
}

vrs16_t
__ZGVzM16vv__mth_i_vr4vi8(vrs16_t x, vid8_t iyu, vid8_t iyl, vis16_t mask, float func(float, long long))
{
  int i;
  vrs16_t r;
  for (i = 0 ; i < 8 ; i++) {
    if (mask[i])
      r[i] = func(x[i], iyu[i]);
  }
  for (i = 8 ; i < 16 ; i++) {
    if (mask[i])
      r[i] = func(x[i], iyl[i-8]);
  }
  return r;
}

vrd8_t
__ZGVzN8v__mth_i_vr8si4(vrd8_t x, int32_t iy, double func(double, int32_t))
{
  int i;
  vrd8_t r;
  for (i = 0 ; i < 8 ; i++) {
    r[i] = func(x[i], iy);
  }
  return r;
}

vrd8_t
__ZGVzM8v__mth_i_vr8si4(vrd8_t x, int32_t iy, vid8_t mask, double func(double, int32_t))
{
  int i;
  vrd8_t r;
  for (i = 0 ; i < 8 ; i++) {
    if (mask[i])
      r[i] = func(x[i], iy);
  }
  return r;
}

vrd8_t
__ZGVzN8vv__mth_i_vr8vi4(vrd8_t x, vis8_t iy, double func(double, int32_t))
{
  int i;
  vrd8_t r;
  for (i = 0 ; i < 8 ; i++) {
    r[i] = func(x[i], iy[i]);
  }
  return r;
}

vrd8_t
__ZGVzM8vv__mth_i_vr8vi4(vrd8_t x, vis8_t iy, vid8_t mask, double func(double, int32_t))
{
  int i;
  vrd8_t r;
  for (i = 0 ; i < 8 ; i++) {
    if (mask[i])
      r[i] = func(x[i], iy[i]);
  }
  return r;
}

vrd8_t
__ZGVzN8v__mth_i_vr8si8(vrd8_t x, long long iy, double func(double, long long))
{
  int i;
  vrd8_t r;
  for (i = 0 ; i < 8 ; i++) {
    r[i] = func(x[i], iy);
  }
  return r;
}

vrd8_t
__ZGVzM8v__mth_i_vr8si8(vrd8_t x, long long iy, vid8_t mask, double func(double, long long))
{
  int i;
  vrd8_t r;
  for (i = 0 ; i < 8 ; i++) {
    if (mask[i])
      r[i] = func(x[i], iy);
  }
  return r;
}

vrd8_t
__ZGVzN8vv__mth_i_vr8vi8(vrd8_t x, vid8_t iy, double func(double, long long))
{
  int i;
  vrd8_t r;
  for (i = 0 ; i < 8 ; i++) {
    r[i] = func(x[i], iy[i]);
  }
  return r;
}

vrd8_t
__ZGVzM8vv__mth_i_vr8vi8(vrd8_t x, vid8_t iy, vid8_t mask, double func(double, long long))
{
  int i;
  vrd8_t r;
  for (i = 0 ; i < 8 ; i++) {
    if (mask[i])
      r[i] = func(x[i], iy[i]);
  }
  return r;
}

vcs8_t
__ZGVzN8v__mth_i_vc4(vcs8_t x, float _Complex func(float _Complex))
{
  int i;
  float _Complex tx[8];
  *(vcs8_t *)&tx = x;
  for (i = 0 ; i < 8 ; i++) {
    tx[i] = func(tx[i]);
  }
  return *(vcs8_t *)&tx;
}

vcs8_t
__ZGVzN8vv__mth_i_vc4vc4(vcs8_t x, vcs8_t y, float _Complex func(float _Complex, float _Complex))
{
  int i;
  float _Complex tx[8];
  float _Complex ty[8];
  *(vcs8_t *)&tx = x;
  *(vcs8_t *)&ty = y;
  for (i = 0 ; i < 8 ; i++) {
    tx[i] = func(tx[i], ty[i]);
  }
  return *(vcs8_t *)&tx;
}

vcd4_t
__ZGVzN4v__mth_i_vc8(vcd4_t x, double _Complex func(double _Complex))
{
  int i;
  double _Complex tx[4];
  *(vcd4_t *)&tx = x;
  for (i = 0 ; i < 4 ; i++) {
    tx[i] = func(tx[i]);
  }
  return *(vcd4_t *)&tx;
}

vcd4_t
__ZGVzN4vv__mth_i_vc8vc8(vcd4_t x, vcd4_t y, double _Complex func(double _Complex, double _Complex))
{
  int i;
  double _Complex tx[4];
  double _Complex ty[4];
  *(vcd4_t *)&tx = x;
  *(vcd4_t *)&ty = y;
  for (i = 0 ; i < 4 ; i++) {
    tx[i] = func(tx[i], ty[i]);
  }
  return *(vcd4_t *)&tx;
}
