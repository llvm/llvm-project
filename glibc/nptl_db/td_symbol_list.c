/* Return list of symbols the library can request.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2001.

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

#include <assert.h>
#include "thread_dbP.h"

static const char *symbol_list_arr[] =
{
# define DB_LOOKUP_NAME(idx, name)		[idx] = STRINGIFY (name),
# define DB_LOOKUP_NAME_TH_UNIQUE(idx, name)	[idx] = STRINGIFY (name),
# include "db-symbols.h"
# undef	DB_LOOKUP_NAME
# undef	DB_LOOKUP_NAME_TH_UNIQUE

  [SYM_NUM_MESSAGES] = NULL
};


const char **
td_symbol_list (void)
{
  return symbol_list_arr;
}


ps_err_e
td_mod_lookup (struct ps_prochandle *ps, const char *mod,
	       int idx, psaddr_t *sym_addr)
{
  ps_err_e result;
  assert (idx >= 0 && idx < SYM_NUM_MESSAGES);
  result = ps_pglobal_lookup (ps, mod, symbol_list_arr[idx], sym_addr);

  return result;
}
