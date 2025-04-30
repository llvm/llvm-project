
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

/************************************************************************/
/*  MODULE_NAME: atnat2.h                                                */
/*                                                                      */
/*                                                                      */
/* 	common data and variables definition for BIG or LITTLE ENDIAN   */
/************************************************************************/



#ifndef ATNAT2_H
#define ATNAT2_H


#define MM 5
#ifdef BIG_ENDI

  static const mynumber
  /* polynomial I */
/**/ d3             = {{0xbfd55555, 0x55555555} }, /* -0.333... */
/**/ d5             = {{0x3fc99999, 0x999997fd} }, /*  0.199... */
/**/ d7             = {{0xbfc24924, 0x923f7603} }, /* -0.142... */
/**/ d9             = {{0x3fbc71c6, 0xe5129a3b} }, /*  0.111... */
/**/ d11            = {{0xbfb74580, 0x22b13c25} }, /* -0.090... */
/**/ d13            = {{0x3fb375f0, 0x8b31cbce} }, /*  0.076... */
  /* polynomial II */
/**/ f3             = {{0xbfd55555, 0x55555555} }, /* -1/3      */
/**/ ff3            = {{0xbc755555, 0x55555555} }, /* -1/3-f3   */
/**/ f5             = {{0x3fc99999, 0x9999999a} }, /*  1/5      */
/**/ ff5            = {{0xbc699999, 0x9999999a} }, /*  1/5-f5   */
/**/ f7             = {{0xbfc24924, 0x92492492} }, /* -1/7      */
/**/ ff7            = {{0xbc624924, 0x92492492} }, /* -1/7-f7   */
/**/ f9             = {{0x3fbc71c7, 0x1c71c71c} }, /*  1/9      */
/**/ ff9            = {{0x3c5c71c7, 0x1c71c71c} }, /*  1/9-f9   */
/**/ f11            = {{0xbfb745d1, 0x745d1746} }, /* -1/11     */
/**/ f13            = {{0x3fb3b13b, 0x13b13b14} }, /*  1/13     */
/**/ f15            = {{0xbfb11111, 0x11111111} }, /* -1/15     */
/**/ f17            = {{0x3fae1e1e, 0x1e1e1e1e} }, /*  1/17     */
/**/ f19            = {{0xbfaaf286, 0xbca1af28} }, /* -1/19     */
  /* constants    */
/**/ inv16          = {{0x3fb00000, 0x00000000} }, /*  1/16         */
/**/ opi            = {{0x400921fb, 0x54442d18} }, /*  pi           */
/**/ opi1           = {{0x3ca1a626, 0x33145c07} }, /*  pi-opi       */
/**/ mopi           = {{0xc00921fb, 0x54442d18} }, /* -pi           */
/**/ hpi            = {{0x3ff921fb, 0x54442d18} }, /*  pi/2         */
/**/ hpi1           = {{0x3c91a626, 0x33145c07} }, /*  pi/2-hpi     */
/**/ mhpi           = {{0xbff921fb, 0x54442d18} }, /* -pi/2         */
/**/ qpi            = {{0x3fe921fb, 0x54442d18} }, /*  pi/4         */
/**/ mqpi           = {{0xbfe921fb, 0x54442d18} }, /* -pi/4         */
/**/ tqpi           = {{0x4002d97c, 0x7f3321d2} }, /*  3pi/4        */
/**/ mtqpi          = {{0xc002d97c, 0x7f3321d2} }, /* -3pi/4        */
/**/ u1             = {{0x3c314c2a, 0x00000000} }, /*  9.377e-19    */
/**/ u2             = {{0x3bf955e4, 0x00000000} }, /*  8.584e-20    */
/**/ u3             = {{0x3bf955e4, 0x00000000} }, /*  8.584e-20    */
/**/ u4             = {{0x3bf955e4, 0x00000000} }, /*  8.584e-20    */
/**/ u5             = {{0x3aaef2d1, 0x00000000} }, /*  5e-26        */
/**/ u6             = {{0x3a6eeb36, 0x00000000} }, /*  3.122e-27    */
/**/ u7             = {{0x3a6eeb36, 0x00000000} }, /*  3.122e-27    */
/**/ u8             = {{0x3a6eeb36, 0x00000000} }, /*  3.122e-27    */
/**/ u91            = {{0x3c6dffc0, 0x00000000} }, /*  1.301e-17    */
/**/ u92            = {{0x3c527bd0, 0x00000000} }, /*  4.008e-18    */
/**/ u93            = {{0x3c3cd057, 0x00000000} }, /*  1.562e-18    */
/**/ u94            = {{0x3c329cdf, 0x00000000} }, /*  1.009e-18    */
/**/ ua1            = {{0x3c3a1edf, 0x00000000} }, /*  1.416e-18    */
/**/ ua2            = {{0x3c33f0e1, 0x00000000} }, /*  1.081e-18    */
/**/ ub             = {{0x3a98c56d, 0x00000000} }, /*  2.001e-26    */
/**/ uc             = {{0x3a9375de, 0x00000000} }, /*  1.572e-26    */
/**/ ud[MM]         ={{{0x38c6eddf, 0x00000000} }, /*  3.450e-35    */
/**/                  {{0x35c6ef60, 0x00000000} }, /*  1.226e-49    */
/**/                  {{0x32c6ed2f, 0x00000000} }, /*  4.354e-64    */
/**/                  {{0x23c6eee8, 0x00000000} }, /*  2.465e-136   */
/**/                  {{0x11c6ed16, 0x00000000} }},/*  4.955e-223   */
/**/ ue             = {{0x38900e9d, 0x00000000} }, /*  3.02e-36     */
/**/ two500         = {{0x5f300000, 0x00000000} }, /*  2**500       */
/**/ twom500        = {{0x20b00000, 0x00000000} }; /*  2**(-500)    */

#else
#ifdef LITTLE_ENDI

  static const mynumber
  /* polynomial I */
/**/ d3             = {{0x55555555, 0xbfd55555} }, /* -0.333... */
/**/ d5             = {{0x999997fd, 0x3fc99999} }, /*  0.199... */
/**/ d7             = {{0x923f7603, 0xbfc24924} }, /* -0.142... */
/**/ d9             = {{0xe5129a3b, 0x3fbc71c6} }, /*  0.111... */
/**/ d11            = {{0x22b13c25, 0xbfb74580} }, /* -0.090... */
/**/ d13            = {{0x8b31cbce, 0x3fb375f0} }, /*  0.076... */
  /* polynomial II */
/**/ f3             = {{0x55555555, 0xbfd55555} }, /* -1/3      */
/**/ ff3            = {{0x55555555, 0xbc755555} }, /* -1/3-f3   */
/**/ f5             = {{0x9999999a, 0x3fc99999} }, /*  1/5      */
/**/ ff5            = {{0x9999999a, 0xbc699999} }, /*  1/5-f5   */
/**/ f7             = {{0x92492492, 0xbfc24924} }, /* -1/7      */
/**/ ff7            = {{0x92492492, 0xbc624924} }, /* -1/7-f7   */
/**/ f9             = {{0x1c71c71c, 0x3fbc71c7} }, /*  1/9      */
/**/ ff9            = {{0x1c71c71c, 0x3c5c71c7} }, /*  1/9-f9   */
/**/ f11            = {{0x745d1746, 0xbfb745d1} }, /* -1/11     */
/**/ f13            = {{0x13b13b14, 0x3fb3b13b} }, /*  1/13     */
/**/ f15            = {{0x11111111, 0xbfb11111} }, /* -1/15     */
/**/ f17            = {{0x1e1e1e1e, 0x3fae1e1e} }, /*  1/17     */
/**/ f19            = {{0xbca1af28, 0xbfaaf286} }, /* -1/19     */
  /* constants    */
/**/ inv16          = {{0x00000000, 0x3fb00000} }, /*  1/16         */
/**/ opi            = {{0x54442d18, 0x400921fb} }, /*  pi           */
/**/ opi1           = {{0x33145c07, 0x3ca1a626} }, /*  pi-opi       */
/**/ mopi           = {{0x54442d18, 0xc00921fb} }, /* -pi           */
/**/ hpi            = {{0x54442d18, 0x3ff921fb} }, /*  pi/2         */
/**/ hpi1           = {{0x33145c07, 0x3c91a626} }, /*  pi/2-hpi     */
/**/ mhpi           = {{0x54442d18, 0xbff921fb} }, /* -pi/2         */
/**/ qpi            = {{0x54442d18, 0x3fe921fb} }, /*  pi/4         */
/**/ mqpi           = {{0x54442d18, 0xbfe921fb} }, /* -pi/4         */
/**/ tqpi           = {{0x7f3321d2, 0x4002d97c} }, /*  3pi/4        */
/**/ mtqpi          = {{0x7f3321d2, 0xc002d97c} }, /* -3pi/4        */
/**/ u1             = {{0x00000000, 0x3c314c2a} }, /*  9.377e-19    */
/**/ u2             = {{0x00000000, 0x3bf955e4} }, /*  8.584e-20    */
/**/ u3             = {{0x00000000, 0x3bf955e4} }, /*  8.584e-20    */
/**/ u4             = {{0x00000000, 0x3bf955e4} }, /*  8.584e-20    */
/**/ u5             = {{0x00000000, 0x3aaef2d1} }, /*  5e-26        */
/**/ u6             = {{0x00000000, 0x3a6eeb36} }, /*  3.122e-27    */
/**/ u7             = {{0x00000000, 0x3a6eeb36} }, /*  3.122e-27    */
/**/ u8             = {{0x00000000, 0x3a6eeb36} }, /*  3.122e-27    */
/**/ u91            = {{0x00000000, 0x3c6dffc0} }, /*  1.301e-17    */
/**/ u92            = {{0x00000000, 0x3c527bd0} }, /*  4.008e-18    */
/**/ u93            = {{0x00000000, 0x3c3cd057} }, /*  1.562e-18    */
/**/ u94            = {{0x00000000, 0x3c329cdf} }, /*  1.009e-18    */
/**/ ua1            = {{0x00000000, 0x3c3a1edf} }, /*  1.416e-18    */
/**/ ua2            = {{0x00000000, 0x3c33f0e1} }, /*  1.081e-18    */
/**/ ub             = {{0x00000000, 0x3a98c56d} }, /*  2.001e-26    */
/**/ uc             = {{0x00000000, 0x3a9375de} }, /*  1.572e-26    */
/**/ ud[MM]         ={{{0x00000000, 0x38c6eddf} }, /*  3.450e-35    */
/**/                  {{0x00000000, 0x35c6ef60} }, /*  1.226e-49    */
/**/                  {{0x00000000, 0x32c6ed2f} }, /*  4.354e-64    */
/**/                  {{0x00000000, 0x23c6eee8} }, /*  2.465e-136   */
/**/                  {{0x00000000, 0x11c6ed16} }},/*  4.955e-223   */
/**/ ue             = {{0x00000000, 0x38900e9d} }, /*  3.02e-36     */
/**/ two500         = {{0x00000000, 0x5f300000} }, /*  2**500       */
/**/ twom500        = {{0x00000000, 0x20b00000} }; /*  2**(-500)    */

#endif
#endif

#endif
