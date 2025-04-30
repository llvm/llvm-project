/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	gettim3f.c - Implements gettim subroutine.  */

#if defined(_WIN64)

#include <windows.h>
#include "ent3f.h"

void ENT3F(GETTIM, gettim)(unsigned short *hr, unsigned short *min,
                           unsigned short *sec, unsigned short *hsec)
{
  SYSTEMTIME st;
  GetLocalTime(&st); /* gets current time */
  *hr = st.wHour;
  *min = st.wMinute;
  *sec = st.wSecond;
  *hsec = st.wMilliseconds / 10;
}

void ENT3F(GETTIM4, gettim4)(int *hr, int *min, int *sec, int *hsec)
{
  SYSTEMTIME st;
  GetLocalTime(&st); /* gets current time */
  *hr = st.wHour;
  *min = st.wMinute;
  *sec = st.wSecond;
  *hsec = st.wMilliseconds / 10;
}

void ENT3F(GETTIM8, gettim8)(long long *hr, long long *min, long long *sec,
                             long long *hsec)
{
  SYSTEMTIME st;
  GetLocalTime(&st); /* gets current time */
  *hr = st.wHour;
  *min = st.wMinute;
  *sec = st.wSecond;
  *hsec = st.wMilliseconds / 10;
}
#endif
