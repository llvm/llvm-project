/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* Low level (backend) f90 merge intrinsics:
 * These functions are not in the usual hpf/f90 rte directory because
 * their calls are not generated  by the front-end.
 */

#include <stdint.h>

int
ftn_i_imerge(int tsource, int fsource, int mask)
{
  return mask ? tsource : fsource;
}

int64_t
ftn_i_kmerge(int64_t tsource, int64_t fsource, int mask)
{
  return mask ? tsource : fsource;
}

float
ftn_i_rmerge(float tsource, float fsource, int mask)
{
  return mask ? tsource : fsource;
}

double
ftn_i_dmerge(double tsource, double fsource, int mask)
{
  return mask ? tsource : fsource;
}
