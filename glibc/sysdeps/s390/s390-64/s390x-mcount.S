/* 64 bit S/390-specific implementation of profiling support.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#define MCOUNT_SYMBOL _mcount
#define MCOUNT_CALLER_OFF 232	/* Pushed by the _mcount stub.  */
#define MCOUNT_CALLEE_REG 14	/* Normal calling convention.  */
#define MCOUNT_RETURN_REG 14
#include "s390x-mcount.h"
#undef MCOUNT_SYMBOL
#undef MCOUNT_CALLER_OFF
#undef MCOUNT_CALLEE_REG
#undef MCOUNT_RETURN_REG

#define MCOUNT_SYMBOL __fentry__
#define MCOUNT_CALLER_OFF 160	/* Saved %r14.  */
#define MCOUNT_CALLEE_REG 0	/* __fentry__ calling convention.  */
#define MCOUNT_RETURN_REG 1	/* Cannot return via %r0.  */
#include "s390x-mcount.h"
#undef MCOUNT_SYMBOL
#undef MCOUNT_CALLER_OFF
#undef MCOUNT_CALLEE_REG
#undef MCOUNT_RETURN_REG

#undef mcount
weak_alias (_mcount, mcount)
