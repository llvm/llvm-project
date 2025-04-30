/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

int
ftn_i_jishft(int x, int shift)
{
  if (shift < 0) {
    shift = -shift;
    if (shift >= 32)
      return 0;
    return (unsigned)x >> shift;
  } else {
    if (shift >= 32)
      return 0;
    return x << shift;
  }
}

int
ftn_i_shift(int x, int shift)
{
  if (shift < 0) {
    shift = -shift;
    return (unsigned)x >> shift;
  } else {
    return x << shift;
  }
}

float
ftn_i_rmin(float x, float y)
{
  if (x > y)
    return y;
  return x;
}

float
ftn_i_rmax(float x, float y)
{
  if (x < y)
    return y;
  return x;
}

double
ftn_i_dmax(double x, double y)
{
  if (x > y)
    return y;
  return x;
}

double
ftn_i_dmin(double x, double y)
{
  if (x < y)
    return y;
  return x;
}

int
ftn_i_isign(int x, int sign)
{
  if (sign >= 0) {
    if (x > 0)
      return x;
    return -x;
  }
  if (x < 0)
    return x;
  return -x;
}

float
ftn_i_sign(float x, int sign)
{
  if (sign >= 0) {
    if (x > 0.0)
      return x;
    return -x;
  }
  if (x < 0.0)
    return x;
  return -x;
}

double
ftn_i_dsign(double x, double sign)
{
  if (sign >= 0) {
    if (x > 0.0)
      return x;
    return -x;
  }
  if (x < 0.0)
    return x;
  return -x;
}

float
ftn_i_dim(float a, float b)
{
  if (a > b)
    return a - b;
  return 0.0;
}

int
ftn_i_idim(int a, int b)
{
  if (a > b)
    return a - b;
  return 0;
}

double
ftn_i_ddim(double a, double b)
{
  if (a > b)
    return a - b;
  return 0.0;
}

#ifdef TARGET_SUPPORTS_QUADFP
long double
ftn_i_qdim(long double a, long double b)
{
  return (a > b) ? (a - b) : 0.0;
}
#endif
