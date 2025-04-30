/* Conversion from and to IBM1388.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Masahide Washizawa <washi@jp.ibm.com>, 2005.

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

#define DATA_HEADER	"ibm1388.h"
#define CHARSET_NAME	"IBM1388//"
#define FROM_LOOP	from_ibm1388
#define TO_LOOP		to_ibm1388
#define SB_TO_UCS4	__ibm1388sb_to_ucs4
#define DB_TO_UCS4_IDX	__ibm1388db_to_ucs4_idx
#define DB_TO_UCS4	__ibm1388db_to_ucs4
#define UCS4_TO_SB_IDX	__ucs4_to_ibm1388sb_idx
#define UCS4_TO_SB	__ucs4_to_ibm1388sb
#define UCS4_TO_DB_IDX	__ucs4_to_ibm1388db_idx
#define UCS4_TO_DB	__ucs4_to_ibm1388db
#define UCS_LIMIT	0xffff

#include "ibm1364.c"
