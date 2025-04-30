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
/*  MODULE_NAME: urem.h                                                 */
/*                                                                      */
/*                                                                      */
/* 	common data and variables definition for BIG or LITTLE ENDIAN   */
/************************************************************************/

#ifndef UREM_H
#define UREM_H

#ifdef BIG_ENDI
static const mynumber big = {{0x43380000, 0}},  /* 6755399441055744 */
                     t128 = {{0x47f00000, 0}},  /*  2^ 128          */
                    tm128 = {{0x37f00000, 0}},  /*  2^-128          */
                      ZERO = {{0, 0}},          /*  0.0             */
                     nZERO = {{0x80000000, 0}}; /* -0.0             */
#else
#ifdef LITTLE_ENDI
static const mynumber big = {{0, 0x43380000}},  /* 6755399441055744 */
                     t128 = {{0, 0x47f00000}},  /*  2^ 128          */
                    tm128 = {{0, 0x37f00000}},  /*  2^-128          */
                      ZERO = {{0, 0}},          /*  0.0             */
                     nZERO = {{0, 0x80000000}}; /* -0.0             */
#endif
#endif

#endif
