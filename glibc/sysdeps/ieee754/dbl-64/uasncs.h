/*
 * IBM Accurate Mathematical Library
 * Written by International Business Machines Corp.
 * Copyright (C) 2001-2021 Free Software Foundation, Inc.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, see <https://www.gnu.org/licenses/>.
 */

/******************************************************************/
/*                                                                */
/* MODULE_NAME:uasncs.h                                           */
/*                                                                */
/* common data and variables prototype and definition             */
/******************************************************************/

#ifndef UANSNCS_H
#define UANSNCS_H

#ifdef BIG_ENDI
 static const  mynumber
/**/           a1 = {{0x3FC55580, 0x00000000 }},  /*  0.1666717529296875     */
/**/           a2 = {{0xBED55555, 0x55552330 }},  /* -5.0862630208224597e-06 */
/**/          hp0 = {{0x3FF921FB, 0x54442D18 }},  /*  1.5707963267948966     */
/**/          hp1 = {{0x3C91A626, 0x33145C07 }};  /*  6.123233995736766e-17  */

#else
#ifdef LITTLE_ENDI
 static const  mynumber
/**/           a1 = {{0x00000000, 0x3FC55580 }},  /*  0.1666717529296875     */
/**/           a2 = {{0x55552330, 0xBED55555 }},  /* -5.0862630208224597e-06 */
/**/          hp0 = {{0x54442D18, 0x3FF921FB }},  /*  1.5707963267948966     */
/**/          hp1 = {{0x33145C07, 0x3C91A626 }};  /*  6.123233995736766e-17  */

#endif
#endif

static const double
              f1 =  1.66666666666664110590506577996662E-01,
              f2 =  7.50000000026122686814431784722623E-02,
              f3 =  4.46428561421059750978517350006940E-02,
              f4 =  3.03821268582119319911193410625235E-02,
              f5 =  2.23551211026525610742786300334557E-02,
              f6 =  1.81382903404565056280372531963613E-02;
static const double
   c2 = 0.74999999999985410757087492918602258E-01,
   c3 = 0.44642857150311968932423372477866076E-01,
   c4 = 0.30381942574778615766200591683810471E-01,
   c5 = 0.22372413472984868331447708777000650E-01,
   c6 = 0.17333630246451830686009693735025490E-01,
   c7 = 0.14710362893628210269950864741085777E-01;

static const double big = 103079215104.0, t24 = 16777216.0, t27 = 134217728.0;
static const double
   rt0 = 9.99999999859990725855365213134618E-01,
   rt1 = 4.99999999495955425917856814202739E-01,
   rt2 = 3.75017500867345182581453026130850E-01,
   rt3 = 3.12523626554518656309172508769531E-01;
#endif
