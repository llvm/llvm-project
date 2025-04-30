/* Return backtrace of current program state.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#include <execinfo.h>
#include <stdlib.h>
#include <unwind-link.h>

struct trace_arg
{
  void **array;
  struct unwind_link *unwind_link;
  int cnt, size;
  void *lastfp, *lastsp;
};

static _Unwind_Reason_Code
backtrace_helper (struct _Unwind_Context *ctx, void *a)
{
  struct trace_arg *arg = a;

  /* We are first called with address in the __backtrace function.
     Skip it.  */
  if (arg->cnt != -1)
    arg->array[arg->cnt]
      = (void *) UNWIND_LINK_PTR (arg->unwind_link, _Unwind_GetIP) (ctx);
  if (++arg->cnt == arg->size)
    return _URC_END_OF_STACK;

  /* %fp is DWARF2 register 14 on M68K.  */
  arg->lastfp
    = (void *) UNWIND_LINK_PTR (arg->unwind_link, _Unwind_GetGR) (ctx, 14);
  arg->lastsp
    = (void *) UNWIND_LINK_PTR (arg->unwind_link, _Unwind_GetCFA) (ctx);
  return _URC_NO_REASON;
}


/* This is a global variable set at program start time.  It marks the
   highest used stack address.  */
extern void *__libc_stack_end;


/* This is the stack layout we see with every stack frame
   if not compiled without frame pointer.

           +-----------------+       +-----------------+
    %fp -> | %fp last frame--------> | %fp last frame--->...
           |                 |       |                 |
           | return address  |       | return address  |
           +-----------------+       +-----------------+

   First try as far to get as far as possible using
   _Unwind_Backtrace which handles -fomit-frame-pointer
   as well, but requires .eh_frame info.  Then fall back to
   walking the stack manually.  */

struct layout
{
  struct layout *fp;
  void *ret;
};


int
__backtrace (void **array, int size)
{
  struct trace_arg arg =
    {
     .array = array,
     .unwind_link = __libc_unwind_link_get (),
     .size = size,
     .cnt = -1,
    };

  if (size <= 0 || arg.unwind_link == NULL)
    return 0;

  UNWIND_LINK_PTR (arg.unwind_link, _Unwind_Backtrace)
    (backtrace_helper, &arg);

  if (arg.cnt > 1 && arg.array[arg.cnt - 1] == NULL)
    --arg.cnt;
  else if (arg.cnt < size)
    {
      struct layout *fp = (struct layout *) arg.lastfp;

      while (arg.cnt < size)
	{
	  /* Check for out of range.  */
	  if ((void *) fp < arg.lastsp || (void *) fp > __libc_stack_end
	      || ((long) fp & 1))
	    break;

	  array[arg.cnt++] = fp->ret;
	  fp = fp->fp;
	}
    }
  return arg.cnt != -1 ? arg.cnt : 0;
}
weak_alias (__backtrace, backtrace)
libc_hidden_def (__backtrace)
