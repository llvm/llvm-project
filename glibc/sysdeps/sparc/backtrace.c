/* Return backtrace of current program state.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David S. Miller <davem@davemloft.net>

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Library General Public License as
   published by the Free Software Foundation; either version 2 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Library General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#include <execinfo.h>
#include <stddef.h>
#include <sysdep.h>
#include <sys/trap.h>
#include <backtrace.h>
#include <unwind-link.h>

struct layout
{
  unsigned long locals[8];
  unsigned long ins[6];
  unsigned long next;
  void *return_address;
};

struct trace_arg
{
  void **array;
  struct unwind_link *unwind_link;
  _Unwind_Word cfa;
  int cnt;
  int size;
};

static _Unwind_Reason_Code
backtrace_helper (struct _Unwind_Context *ctx, void *a)
{
  struct trace_arg *arg = a;
  _Unwind_Ptr ip;

  /* We are first called with address in the __backtrace function.
     Skip it.  */
  if (arg->cnt != -1)
    {
      ip = UNWIND_LINK_PTR (arg->unwind_link, _Unwind_GetIP) (ctx);
      arg->array[arg->cnt] = (void *) ip;

      /* Check whether we make any progress.  */
      _Unwind_Word cfa
	= UNWIND_LINK_PTR (arg->unwind_link, _Unwind_GetCFA) (ctx);

      if (arg->cnt > 0 && arg->array[arg->cnt - 1] == arg->array[arg->cnt]
	 && cfa == arg->cfa)
       return _URC_END_OF_STACK;
      arg->cfa = cfa;
    }
  if (++arg->cnt == arg->size)
    return _URC_END_OF_STACK;
  return _URC_NO_REASON;
}

int
__backtrace (void **array, int size)
{
  int count;
  struct trace_arg arg =
    {
     .array = array,
     .unwind_link = __libc_unwind_link_get (),
     .size = size,
     .cnt = -1,
    };

  if (size <= 0)
    return 0;

  if (arg.unwind_link == NULL)
    {
      struct layout *current;
      unsigned long fp, i7;

      asm volatile ("mov %%fp, %0" : "=r"(fp));
      asm volatile ("mov %%i7, %0" : "=r"(i7));
      current = (struct layout *) (fp + BACKTRACE_STACK_BIAS);

      array[0] = (void *) i7;

      if (size == 1)
	return 1;

      backtrace_flush_register_windows();
      for (count = 1; count < size; count++)
	{
	  array[count] = current->return_address;
	  if (!current->next)
	    break;
	  current = (struct layout *) (current->next + BACKTRACE_STACK_BIAS);
	}
    }
  else
    {
      UNWIND_LINK_PTR (arg.unwind_link, _Unwind_Backtrace)
	(backtrace_helper, &arg);

      /* _Unwind_Backtrace seems to put NULL address above
	 _start.  Fix it up here.  */
      if (arg.cnt > 1 && arg.array[arg.cnt - 1] == NULL)
	--arg.cnt;
      count = arg.cnt != -1 ? arg.cnt : 0;
    }
  return count;
}
weak_alias (__backtrace, backtrace)
libc_hidden_def (__backtrace)
