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
/* MODULE_NAME:utan.h                                             */
/*                                                                */
/* common data and variables prototype and definition             */
/******************************************************************/

#ifndef UTAN_H
#define UTAN_H

#ifdef BIG_ENDI
  static const mynumber
  /* polynomial I */
/**/ d3             = {{0x3FD55555, 0x55555555} }, /*  0.333... */
/**/ d5             = {{0x3FC11111, 0x111107C6} }, /*  0.133... */
/**/ d7             = {{0x3FABA1BA, 0x1CDB8745} }, /*    .      */
/**/ d9             = {{0x3F9664ED, 0x49CFC666} }, /*    .      */
/**/ d11            = {{0x3F82385A, 0x3CF2E4EA} }, /*    .      */
  /* polynomial II */
  /* polynomial III */
/**/ e0             = {{0x3FD55555, 0x55554DBD} }, /*    .      */
/**/ e1             = {{0x3FC11112, 0xE0A6B45F} }, /*    .      */

  /* constants    */
/**/ mfftnhf        = {{0xc02f0000, 0x00000000} }, /*-15.5      */

/**/ g1             = {{0x3e4b096c, 0x00000000} }, /* 1.259e-8  */
/**/ g2             = {{0x3faf212d, 0x00000000} }, /* 0.0608    */
/**/ g3             = {{0x3fe92f1a, 0x00000000} }, /* 0.787     */
/**/ g4             = {{0x40390000, 0x00000000} }, /* 25.0      */
/**/ g5             = {{0x4197d784, 0x00000000} }, /* 1e8       */
/**/ gy2            = {{0x3faf212d, 0x00000000} }, /* 0.0608    */

/**/            mp1 = {{0x3FF921FB, 0x58000000} },
/**/            mp2 = {{0xBE4DDE97, 0x3C000000} },
/**/            mp3 = {{0xBC8CB3B3, 0x99D747F2} },
/**/            pp3 = {{0xBC8CB3B3, 0x98000000} },
/**/            pp4 = {{0xbacd747f, 0x23e32ed7} },
/**/          hpinv = {{0x3FE45F30, 0x6DC9C883} },
/**/          toint = {{0x43380000, 0x00000000} };

#else
#ifdef LITTLE_ENDI

  static const mynumber
  /* polynomial I */
/**/ d3             = {{0x55555555, 0x3FD55555} }, /*  0.333... */
/**/ d5             = {{0x111107C6, 0x3FC11111} }, /*  0.133... */
/**/ d7             = {{0x1CDB8745, 0x3FABA1BA} }, /*    .      */
/**/ d9             = {{0x49CFC666, 0x3F9664ED} }, /*    .      */
/**/ d11            = {{0x3CF2E4EA, 0x3F82385A} }, /*    .      */
  /* polynomial II */
  /* polynomial III */
/**/ e0             = {{0x55554DBD, 0x3FD55555} }, /*    .      */
/**/ e1             = {{0xE0A6B45F, 0x3FC11112} }, /*    .      */

  /* constants    */
/**/ mfftnhf        = {{0x00000000, 0xc02f0000} }, /*-15.5      */

/**/ g1             = {{0x00000000, 0x3e4b096c} }, /* 1.259e-8  */
/**/ g2             = {{0x00000000, 0x3faf212d} }, /* 0.0608    */
/**/ g3             = {{0x00000000, 0x3fe92f1a} }, /* 0.787     */
/**/ g4             = {{0x00000000, 0x40390000} }, /* 25.0      */
/**/ g5             = {{0x00000000, 0x4197d784} }, /* 1e8       */
/**/ gy2            = {{0x00000000, 0x3faf212d} }, /* 0.0608    */

/**/            mp1 = {{0x58000000, 0x3FF921FB} },
/**/            mp2 = {{0x3C000000, 0xBE4DDE97} },
/**/            mp3 = {{0x99D747F2, 0xBC8CB3B3} },
/**/            pp3 = {{0x98000000, 0xBC8CB3B3} },
/**/            pp4 = {{0x23e32ed7, 0xbacd747f} },
/**/          hpinv = {{0x6DC9C883, 0x3FE45F30} },
/**/          toint = {{0x00000000, 0x43380000} };

#endif
#endif

#endif
