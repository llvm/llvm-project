/* Set up the data structures for the system-supplied DSO.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

static inline void __attribute__ ((always_inline))
setup_vdso (struct link_map *main_map __attribute__ ((unused)),
	    struct link_map ***first_preload __attribute__ ((unused)))
{
#ifdef NEED_DL_SYSINFO_DSO
  if (GLRO(dl_sysinfo_dso) == NULL)
    return;

  /* Do an abridged version of the work _dl_map_object_from_fd would do
     to map in the object.  It's already mapped and prelinked (and
     better be, since it's read-only and so we couldn't relocate it).
     We just want our data structures to describe it as if we had just
     mapped and relocated it normally.  */
  struct link_map *l = _dl_new_object ((char *) "", "", lt_library, NULL,
				       0, LM_ID_BASE);
  if (__glibc_likely (l != NULL))
    {
      static ElfW(Dyn) dyn_temp[DL_RO_DYN_TEMP_CNT] attribute_relro;

      l->l_phdr = ((const void *) GLRO(dl_sysinfo_dso)
		   + GLRO(dl_sysinfo_dso)->e_phoff);
      l->l_phnum = GLRO(dl_sysinfo_dso)->e_phnum;
      for (uint_fast16_t i = 0; i < l->l_phnum; ++i)
	{
	  const ElfW(Phdr) *const ph = &l->l_phdr[i];
	  if (ph->p_type == PT_DYNAMIC)
	    {
	      l->l_ld = (void *) ph->p_vaddr;
	      l->l_ldnum = ph->p_memsz / sizeof (ElfW(Dyn));
	    }
	  else if (ph->p_type == PT_LOAD)
	    {
	      if (! l->l_addr)
		l->l_addr = ph->p_vaddr;
	      if (ph->p_vaddr + ph->p_memsz >= l->l_map_end)
		l->l_map_end = ph->p_vaddr + ph->p_memsz;
	      if ((ph->p_flags & PF_X)
		  && ph->p_vaddr + ph->p_memsz >= l->l_text_end)
		l->l_text_end = ph->p_vaddr + ph->p_memsz;
	    }
	  else
	    /* There must be no TLS segment.  */
	    assert (ph->p_type != PT_TLS);
	}
      l->l_map_start = (ElfW(Addr)) GLRO(dl_sysinfo_dso);
      l->l_addr = l->l_map_start - l->l_addr;
      l->l_map_end += l->l_addr;
      l->l_text_end += l->l_addr;
      l->l_ld = (void *) ((ElfW(Addr)) l->l_ld + l->l_addr);
      elf_get_dynamic_info (l, dyn_temp);
      _dl_setup_hash (l);
      l->l_relocated = 1;

      /* The vDSO is always used.  */
      l->l_used = 1;

      /* Initialize l_local_scope to contain just this map.  This allows
	 the use of dl_lookup_symbol_x to resolve symbols within the vdso.
	 So we create a single entry list pointing to l_real as its only
	 element */
      l->l_local_scope[0]->r_nlist = 1;
      l->l_local_scope[0]->r_list = &l->l_real;

      /* Now that we have the info handy, use the DSO image's soname
	 so this object can be looked up by name.  */
      if (l->l_info[DT_SONAME] != NULL)
	{
	  char *dsoname = ((char *) D_PTR (l, l_info[DT_STRTAB])
			   + l->l_info[DT_SONAME]->d_un.d_val);
	  l->l_libname->name = dsoname;
	  l->l_name = dsoname;
	}

      /* Add the vDSO to the object list.  */
      _dl_add_to_namespace_list (l, LM_ID_BASE);

# if IS_IN (rtld)
      /* Rearrange the list so this DSO appears after rtld_map.  */
      assert (l->l_next == NULL);
      assert (l->l_prev == main_map
	      || (l->l_prev != NULL && l->l_prev->l_prev == main_map));
      GL(dl_rtld_map).l_next = l;
      l->l_prev = &GL(dl_rtld_map);
      *first_preload = &l->l_next;
# else
      GL(dl_nns) = 1;
# endif

      /* We have a prelinked DSO preloaded by the system.  */
      GLRO(dl_sysinfo_map) = l;
# ifdef NEED_DL_SYSINFO
      if (GLRO(dl_sysinfo) == DL_SYSINFO_DEFAULT)
	GLRO(dl_sysinfo) = GLRO(dl_sysinfo_dso)->e_entry + l->l_addr;
# endif
    }
#endif
}
