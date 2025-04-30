/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
     Contributed by David Mosberger-Tang <davidm@hpl.hp.com>.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

/* Constants shared between setcontext() and getcontext().  Don't
   install this header file.  */

#define SIG_BLOCK	0
#define SIG_UNBLOCK	1
#define SIG_SETMASK	2

#include <sigcontext-offsets.h>

#define rTMP	r16
#define rPOS	r16
#define rCPOS	r17
#define rNAT	r18

#define rB5	r18
#define rB4	r19
#define rB3	r20
#define rB2	r21
#define rB1	r22
#define rB0	r23
#define rRSC	r24
#define rBSP	r25
#define rRNAT	r26
#define rUNAT	r27
#define rFPSR	r28
#define rPFS	r29
#define rLC	r30
#define rPR	r31
