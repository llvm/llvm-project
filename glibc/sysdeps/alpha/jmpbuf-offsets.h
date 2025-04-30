/* Private macros for accessing __jmp_buf contents.  Alpha version.
   Copyright (C) 2006-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#define JB_S0  0
#define JB_S1  1
#define JB_S2  2
#define JB_S3  3
#define JB_S4  4
#define JB_S5  5
#define JB_PC  6
#define JB_FP  7
#define JB_SP  8
#define JB_F2  9
#define JB_F3  10
#define JB_F4  11
#define JB_F5  12
#define JB_F6  13
#define JB_F7  14
#define JB_F8  15
#define JB_F9  16
