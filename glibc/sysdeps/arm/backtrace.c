/* Return backtrace of current program state.
   Copyright (C) 2008-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Kazu Hirata <kazu@codesourcery.com>, 2008.

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

#include <execinfo.h>
#include <stdlib.h>
#include <unwind-link.h>

struct trace_arg
{
  void **array;
  struct unwind_link *unwind_link;
  int cnt, size;
};

#ifdef SHARED
/* This function is identical to "_Unwind_GetGR", except that it uses
   "unwind_vrs_get" instead of "_Unwind_VRS_Get".  */
static inline _Unwind_Word
unwind_getgr (struct unwind_link *unwind_link,
	      _Unwind_Context *context, int regno)
{
  _uw val;
  UNWIND_LINK_PTR (unwind_link, _Unwind_VRS_Get)
    (context, _UVRSC_CORE, regno, _UVRSD_UINT32, &val);
  return val;
}

/* This macro is identical to the _Unwind_GetIP macro, except that it
   uses "unwind_getgr" instead of "_Unwind_GetGR".  */
#define unwind_getip(context) \
  (unwind_getgr (arg->unwind_link, context, 15) & ~(_Unwind_Word)1)

#else /* !SHARED */
# define unwind_getip _Unwind_GetIP
#endif

static _Unwind_Reason_Code
backtrace_helper (struct _Unwind_Context *ctx, void *a)
{
  struct trace_arg *arg = a;

  /* We are first called with address in the __backtrace function.
     Skip it.  */
  if (arg->cnt != -1)
    arg->array[arg->cnt] = (void *) unwind_getip (ctx);
  if (++arg->cnt == arg->size)
    return _URC_END_OF_STACK;
  return _URC_NO_REASON;
}

int
__backtrace (void **array, int size)
{
  struct trace_arg arg =
    {
     .array = array,
     .unwind_link = __libc_unwind_link_get (),
     .size = size,
     .cnt = -1
    };

  if (size <= 0 || arg.unwind_link == NULL)
    return 0;

  UNWIND_LINK_PTR (arg.unwind_link, _Unwind_Backtrace)
    (backtrace_helper, &arg);

  if (arg.cnt > 1 && arg.array[arg.cnt - 1] == NULL)
    --arg.cnt;
  return arg.cnt != -1 ? arg.cnt : 0;
}
weak_alias (__backtrace, backtrace)
libc_hidden_def (__backtrace)
