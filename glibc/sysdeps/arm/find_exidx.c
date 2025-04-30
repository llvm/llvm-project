/* Copyright (C) 2005-2021 Free Software Foundation, Inc.
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

#include <link.h>
#include <unwind.h>

struct unw_eh_callback_data
{
  _Unwind_Ptr pc;
  _Unwind_Ptr exidx_start;
  int exidx_len;
};


/* Callback to determins if the PC lies within an object, and remember the
   location of the exception index table if it does.  */

static int
find_exidx_callback (struct dl_phdr_info * info, size_t size, void * ptr)
{
  struct unw_eh_callback_data * data;
  const ElfW(Phdr) *phdr;
  int i;
  int match;
  _Unwind_Ptr load_base;

  data = (struct unw_eh_callback_data *) ptr;
  load_base = info->dlpi_addr;
  phdr = info->dlpi_phdr;

  match = 0;
  for (i = info->dlpi_phnum; i > 0; i--, phdr++)
    {
      if (phdr->p_type == PT_LOAD)
        {
          _Unwind_Ptr vaddr = phdr->p_vaddr + load_base;
          if (data->pc >= vaddr && data->pc < vaddr + phdr->p_memsz)
            match = 1;
        }
      else if (phdr->p_type == PT_ARM_EXIDX)
	{
	  data->exidx_start = (_Unwind_Ptr) (phdr->p_vaddr + load_base);
	  data->exidx_len = phdr->p_memsz;
	}
    }

  return match;
}


/* Find the exception index table containing PC.  */

_Unwind_Ptr
__gnu_Unwind_Find_exidx (_Unwind_Ptr pc, int * pcount)
{
  struct unw_eh_callback_data data;

  data.pc = pc;
  data.exidx_start = 0;
  if (__dl_iterate_phdr (find_exidx_callback, &data) <= 0)
    return 0;

  *pcount = data.exidx_len / 8;
  return data.exidx_start;
}
