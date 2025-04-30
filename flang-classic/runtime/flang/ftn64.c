/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

union u {
  double d;
  int i[2];
};

/*
 * macros for accessing the endian-dependent halves of a 64-bit typeless
 * quantity when stored in a union of a double and an int[2] array):
 *   _MSH -- most significant half  ([1] - little endian, [0] - big endian)
 *   _LSH -- least significant half ([0] - little endian, [1] - big endian)
 */
#undef _MSH
#undef _LSH
#define _MSH(uu) uu.i[1]
#define _LSH(uu) uu.i[0]

double
ftn_i_not64(double op1)
{
  union u u1;

  u1.d = op1;
  u1.i[0] = ~u1.i[0];
  u1.i[1] = ~u1.i[1];
  return u1.d;
}
double
ftn_i_and64(double op1, double op2)
{
  union u u1;
  union u u2;

  u1.d = op1;
  u2.d = op2;
  u1.i[0] &= u2.i[0];
  u1.i[1] &= u2.i[1];
  return u1.d;
}
double
ftn_i_or64(double op1, double op2)
{
  union u u1;
  union u u2;

  u1.d = op1;
  u2.d = op2;
  u1.i[0] |= u2.i[0];
  u1.i[1] |= u2.i[1];
  return u1.d;
}
double
ftn_i_xor64(double op1, double op2)
{
  union u u1;
  union u u2;

  u1.d = op1;
  u2.d = op2;
  u1.i[0] ^= u2.i[0];
  u1.i[1] ^= u2.i[1];
  return u1.d;
}
double
ftn_i_xnor64(double op1, double op2)
{
  union u u1;
  union u u2;

  u1.d = op1;
  u2.d = op2;
  u1.i[0] = ~(u1.i[0] ^ u2.i[0]);
  u1.i[1] = ~(u1.i[1] ^ u2.i[1]);
  return u1.d;
}
double
ftn_i_shift64(double op1, int cnt)
{
  /*
          logical shift:
              cnt < 0 ==> shift op1 left by |cnt|
              cnt > 0 ==> shift op1 right by cnt
              |cnt| >= 64 ==> result is 0
  */
  union u u1;
  union u u2;

  u1.d = op1;
  if (cnt >= 64 || cnt <= -64) {
    u2.i[0] = u2.i[1] = 0;
  } else if (cnt == 0) { /*  0 == cnt */
    _MSH(u2) = _MSH(u1);
    _LSH(u2) = _LSH(u1);
  } else if (cnt >= 32) { /*  32 <= cnt <= 63  */
    _MSH(u2) = _LSH(u1) << (cnt - 32);
    _LSH(u2) = 0;
  } else if (cnt > 0) { /*  0 < cnt <= 31 */
    _MSH(u2) = _MSH(u1) << cnt;
    _MSH(u2) |= (unsigned)_LSH(u1) >> (32 - cnt);
    _LSH(u2) = _LSH(u1) << cnt;
  } else if (cnt <= -32) { /*  -63 <= cnt <= -32 */
    _MSH(u2) = 0;
    _LSH(u2) = (unsigned)_MSH(u1) >> ((-cnt) - 32);
  } else /* if (cnt < 0) */ { /*  -31 <= cnt < 0 */
    int acnt = -cnt;
    _MSH(u2) = (unsigned)_MSH(u1) >> acnt;
    _LSH(u2) = (unsigned)_LSH(u1) >> acnt;
    _LSH(u2) |= (unsigned)_MSH(u1) << (cnt + 32);
  }
  return u2.d;
}

int
ftn_i_dp2ir(double dp)
{
  union u u1;
  /*
      result is first element of int[2] which is union u'd with dp if little
      endian; if big endian, result is second element.
  */
  u1.d = dp;
  return _LSH(u1);
}
float
ftn_i_dp2sp(double dp)
{
  union u u1;
  int i;

  u1.d = dp;
  i = _LSH(u1);
  return *(float *)&i;
}

double
ftn_i_ir2dp(int ir)
{
  union u u1;

  u1.d = 0.0;
  _LSH(u1) = ir;
  return u1.d;
}
double
ftn_i_sp2dp(float sp)
{
  union u u1;

  u1.d = 0.0;
  _LSH(u1) = *(int *)&sp;
  return u1.d;
}
