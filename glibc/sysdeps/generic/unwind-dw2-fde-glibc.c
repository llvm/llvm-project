/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
   Contributed by Jakub Jelinek <jakub@redhat.com>.

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

/* Locate the FDE entry for a given address, using PT_GNU_EH_FRAME ELF
   segment and dl_iterate_phdr to avoid register/deregister calls at
   DSO load/unload.  */

#ifdef _LIBC
# include <shlib-compat.h>
#endif

#if !defined _LIBC || SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_2_5)

#include <link.h>
#include <stddef.h>

#define _Unwind_Find_FDE _Unwind_Find_registered_FDE

#include <unwind-dw2-fde.c>

#undef _Unwind_Find_FDE

extern fde * _Unwind_Find_registered_FDE (void *pc,
					  struct dwarf_eh_bases *bases);
extern fde * _Unwind_Find_FDE (void *, struct dwarf_eh_bases *);

struct unw_eh_callback_data
{
  _Unwind_Ptr pc;
  void *tbase;
  void *dbase;
  void *func;
  fde *ret;
};

struct unw_eh_frame_hdr
{
  unsigned char version;
  unsigned char eh_frame_ptr_enc;
  unsigned char fde_count_enc;
  unsigned char table_enc;
};

/* Like base_of_encoded_value, but take the base from a struct object
   instead of an _Unwind_Context.  */

static _Unwind_Ptr
base_from_cb_data (unsigned char encoding, struct unw_eh_callback_data *data)
{
  if (encoding == DW_EH_PE_omit)
    return 0;

  switch (encoding & 0x70)
    {
    case DW_EH_PE_absptr:
    case DW_EH_PE_pcrel:
    case DW_EH_PE_aligned:
      return 0;

    case DW_EH_PE_textrel:
      return (_Unwind_Ptr) data->tbase;
    case DW_EH_PE_datarel:
      return (_Unwind_Ptr) data->dbase;
    }
  abort ();
}

static int
_Unwind_IteratePhdrCallback (struct dl_phdr_info *info, size_t size, void *ptr)
{
  struct unw_eh_callback_data *data = (struct unw_eh_callback_data *) ptr;
  const ElfW(Phdr) *phdr, *p_eh_frame_hdr;
  const ElfW(Phdr) *p_dynamic __attribute__ ((unused));
  long n, match;
  _Unwind_Ptr load_base;
  const unsigned char *p;
  const struct unw_eh_frame_hdr *hdr;
  _Unwind_Ptr eh_frame;
  struct object ob;

  /* Make sure struct dl_phdr_info is at least as big as we need.  */
  if (size < offsetof (struct dl_phdr_info, dlpi_phnum)
	     + sizeof (info->dlpi_phnum))
    return -1;

  match = 0;
  phdr = info->dlpi_phdr;
  load_base = info->dlpi_addr;
  p_eh_frame_hdr = NULL;
  p_dynamic = NULL;

  /* See if PC falls into one of the loaded segments.  Find the eh_frame
     segment at the same time.  */
  for (n = info->dlpi_phnum; --n >= 0; phdr++)
    {
      if (phdr->p_type == PT_LOAD)
	{
	  _Unwind_Ptr vaddr = phdr->p_vaddr + load_base;
	  if (data->pc >= vaddr && data->pc < vaddr + phdr->p_memsz)
	    match = 1;
	}
      else if (phdr->p_type == PT_GNU_EH_FRAME)
	p_eh_frame_hdr = phdr;
      else if (phdr->p_type == PT_DYNAMIC)
	p_dynamic = phdr;
    }
  if (!match || !p_eh_frame_hdr)
    return 0;

  /* Read .eh_frame_hdr header.  */
  hdr = (const struct unw_eh_frame_hdr *)
	(p_eh_frame_hdr->p_vaddr + load_base);
  if (hdr->version != 1)
    return 1;

#ifdef CRT_GET_RFIB_DATA
# ifdef __i386__
  data->dbase = NULL;
  if (p_dynamic)
    {
      /* For dynamicly linked executables and shared libraries,
	 DT_PLTGOT is the gp value for that object.  */
      ElfW(Dyn) *dyn = (ElfW(Dyn) *)(p_dynamic->p_vaddr + load_base);
      for (; dyn->d_tag != DT_NULL ; dyn++)
	if (dyn->d_tag == DT_PLTGOT)
	  {
	    /* On IA-32, _DYNAMIC is writable and GLIBC has relocated it.  */
	    data->dbase = (void *) dyn->d_un.d_ptr;
	    break;
	  }
    }
# else
#  error What is DW_EH_PE_datarel base on this platform?
# endif
#endif
#ifdef CRT_GET_RFIB_TEXT
# error What is DW_EH_PE_textrel base on this platform?
#endif

  p = read_encoded_value_with_base (hdr->eh_frame_ptr_enc,
				    base_from_cb_data (hdr->eh_frame_ptr_enc,
						       data),
				    (const unsigned char *) (hdr + 1),
				    &eh_frame);

  /* We require here specific table encoding to speed things up.
     Also, DW_EH_PE_datarel here means using PT_GNU_EH_FRAME start
     as base, not the processor specific DW_EH_PE_datarel.  */
  if (hdr->fde_count_enc != DW_EH_PE_omit
      && hdr->table_enc == (DW_EH_PE_datarel | DW_EH_PE_sdata4))
    {
      _Unwind_Ptr fde_count;

      p = read_encoded_value_with_base (hdr->fde_count_enc,
					base_from_cb_data (hdr->fde_count_enc,
							   data),
					p, &fde_count);
      /* Shouldn't happen.  */
      if (fde_count == 0)
	return 1;
      if ((((_Unwind_Ptr) p) & 3) == 0)
	{
	  struct fde_table {
	    signed initial_loc __attribute__ ((mode (SI)));
	    signed fde __attribute__ ((mode (SI)));
	  };
	  const struct fde_table *table = (const struct fde_table *) p;
	  size_t lo, hi, mid;
	  _Unwind_Ptr data_base = (_Unwind_Ptr) hdr;
	  fde *f;
	  unsigned int f_enc, f_enc_size;
	  _Unwind_Ptr range;

	  mid = fde_count - 1;
	  if (data->pc < table[0].initial_loc + data_base)
	    return 1;
	  else if (data->pc < table[mid].initial_loc + data_base)
	    {
	      lo = 0;
	      hi = mid;

	      while (lo < hi)
		{
		  mid = (lo + hi) / 2;
		  if (data->pc < table[mid].initial_loc + data_base)
		    hi = mid;
		  else if (data->pc >= table[mid + 1].initial_loc + data_base)
		    lo = mid + 1;
		  else
		    break;
		}

	      if (lo >= hi)
		__gxx_abort ();
	    }

	  f = (fde *) (table[mid].fde + data_base);
	  f_enc = get_fde_encoding (f);
	  f_enc_size = size_of_encoded_value (f_enc);
	  read_encoded_value_with_base (f_enc & 0x0f, 0,
					&f->pc_begin[f_enc_size], &range);
	  if (data->pc < table[mid].initial_loc + data_base + range)
	    data->ret = f;
	  data->func = (void *) (table[mid].initial_loc + data_base);
	  return 1;
	}
    }

  /* We have no sorted search table, so need to go the slow way.
     As soon as GLIBC will provide API so to notify that a library has been
     removed, we could cache this (and thus use search_object).  */
  ob.pc_begin = NULL;
  ob.tbase = data->tbase;
  ob.dbase = data->dbase;
  ob.u.single = (fde *) eh_frame;
  ob.s.i = 0;
  ob.s.b.mixed_encoding = 1;  /* Need to assume worst case.  */
  data->ret = linear_search_fdes (&ob, (fde *) eh_frame, (void *) data->pc);
  if (data->ret != NULL)
    {
      unsigned int encoding = get_fde_encoding (data->ret);
      _Unwind_Ptr func;
      read_encoded_value_with_base (encoding,
				    base_from_cb_data (encoding, data),
				    data->ret->pc_begin, &func);
      data->func = (void *) func;
    }
  return 1;
}

# ifdef _LIBC
# define dl_iterate_phdr __dl_iterate_phdr
# endif

fde *
_Unwind_Find_FDE (void *pc, struct dwarf_eh_bases *bases)
{
  struct unw_eh_callback_data data;
  fde *ret;

  ret = _Unwind_Find_registered_FDE (pc, bases);
  if (ret != NULL)
    return ret;

  data.pc = (_Unwind_Ptr) pc;
  data.tbase = NULL;
  data.dbase = NULL;
  data.func = NULL;
  data.ret = NULL;

  if (dl_iterate_phdr (_Unwind_IteratePhdrCallback, &data) < 0)
    return NULL;

  if (data.ret)
    {
      bases->tbase = data.tbase;
      bases->dbase = data.dbase;
      bases->func = data.func;
    }
  return data.ret;
}

#endif
