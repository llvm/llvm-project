/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	getdat3f.c - Implements getdat subroutine.  */

#if defined(_WIN64)

#include <windows.h>
#include "ent3f.h"

void ENT3F(GETDAT, getdat)(unsigned short *iyr, unsigned short *imon,
                           unsigned short *iday)
{
  SYSTEMTIME st;
  GetLocalTime(&st); /* gets current time */
  *iyr = st.wYear;
  *imon = st.wMonth;
  *iday = st.wDay;
}

void ENT3F(GETDAT4, getdat4)(int *iyr, int *imon, int *iday)
{
  SYSTEMTIME st;
  GetLocalTime(&st); /* gets current time */
  *iyr = st.wYear;
  *imon = st.wMonth;
  *iday = st.wDay;
}

void ENT3F(GETDAT8, getdat8)(long long *iyr, long long *imon, long long *iday)
{
  SYSTEMTIME st;
  GetLocalTime(&st); /* gets current time */
  *iyr = st.wYear;
  *imon = st.wMonth;
  *iday = st.wDay;
}
#endif
