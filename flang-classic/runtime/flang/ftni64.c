/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "ftni64.h"

_ULONGLONG_T
ftn_i_kishft(_ULONGLONG_T op, int count)
{
  /*
          logical shift:
              cnt < 0 ==> shift op1 left by |cnt|
              cnt > 0 ==> shift op1 right by cnt
              |cnt| >= 64 ==> result is 0
  */

  if (count >= 0) {
    if (count >= 64)
      return 0;
    return op << count;
  }
  if (count <= -64)
    return 0;
  return op >> -count;
}

__I8RET_T
ftn_i_xori64(int op1, int op2, int op3, int op4)
{
  DBLINT64 u1;
  DBLINT64 u2;

  u1[0] = op2;
  u1[1] = op1;
  u2[0] = op4;
  u2[1] = op3;
  u1[0] ^= u2[0];
  u1[1] ^= u2[1];
  UTL_I_I64RET(u1[0], u1[1]);
}

__I8RET_T
ftn_i_xnori64(int op1, int op2, int op3, int op4)
{
  DBLINT64 u1;
  DBLINT64 u2;

  u1[0] = op2;
  u1[1] = op1;
  u2[0] = op4;
  u2[1] = op3;
  u1[0] = ~(u1[0] ^ u2[0]);
  u1[1] = ~(u1[1] ^ u2[1]);
  UTL_I_I64RET(u1[0], u1[1]);
}

int
ftn_i_kr2ir(int op1, int op2)
{
  DBLINT64 u1;
  /*
      result is first element of int[2] which is union u'd with dp if little
      endian; if big endian, result is second element.
  */
  u1[0] = op1;
  u1[1] = op2;
  return I64_LSH(u1);
}

float
ftn_i_kr2sp(int op1, int op2)
{
  DBLINT64 u1;
  int i;

  u1[0] = op1;
  u1[1] = op2;
  i = I64_LSH(u1);
  return (float)i;
}

double
ftn_i_kr2dp(int op1, int op2)
{
  INT64D u1;

  u1.i[0] = op1;
  u1.i[1] = op2;
  return u1.d;
}
