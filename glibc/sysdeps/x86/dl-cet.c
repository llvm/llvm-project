/* x86 CET initializers function.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.

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

#include <unistd.h>
#include <errno.h>
#include <libintl.h>
#include <ldsodefs.h>
#include <dl-cet.h>

/* GNU_PROPERTY_X86_FEATURE_1_IBT and GNU_PROPERTY_X86_FEATURE_1_SHSTK
   are defined in <elf.h>, which are only available for C sources.
   X86_FEATURE_1_IBT and X86_FEATURE_1_SHSTK are defined in <sysdep.h>
   which are available for both C and asm sources.  They must match.   */
#if GNU_PROPERTY_X86_FEATURE_1_IBT != X86_FEATURE_1_IBT
# error GNU_PROPERTY_X86_FEATURE_1_IBT != X86_FEATURE_1_IBT
#endif
#if GNU_PROPERTY_X86_FEATURE_1_SHSTK != X86_FEATURE_1_SHSTK
# error GNU_PROPERTY_X86_FEATURE_1_SHSTK != X86_FEATURE_1_SHSTK
#endif

/* Check if object M is compatible with CET.  */

static void
dl_cet_check (struct link_map *m, const char *program)
{
  /* Check how IBT should be enabled.  */
  enum dl_x86_cet_control enable_ibt_type
    = GL(dl_x86_feature_control).ibt;
  /* Check how SHSTK should be enabled.  */
  enum dl_x86_cet_control enable_shstk_type
    = GL(dl_x86_feature_control).shstk;

  /* No legacy object check if both IBT and SHSTK are always on.  */
  if (enable_ibt_type == cet_always_on
      && enable_shstk_type == cet_always_on)
    {
      THREAD_SETMEM (THREAD_SELF, header.feature_1, GL(dl_x86_feature_1));
      return;
    }

  /* Check if IBT is enabled by kernel.  */
  bool ibt_enabled
    = (GL(dl_x86_feature_1) & GNU_PROPERTY_X86_FEATURE_1_IBT) != 0;
  /* Check if SHSTK is enabled by kernel.  */
  bool shstk_enabled
    = (GL(dl_x86_feature_1) & GNU_PROPERTY_X86_FEATURE_1_SHSTK) != 0;

  if (ibt_enabled || shstk_enabled)
    {
      struct link_map *l = NULL;
      unsigned int ibt_legacy = 0, shstk_legacy = 0;
      bool found_ibt_legacy = false, found_shstk_legacy = false;

      /* Check if IBT and SHSTK are enabled in object.  */
      bool enable_ibt = (ibt_enabled
			 && enable_ibt_type != cet_always_off);
      bool enable_shstk = (shstk_enabled
			   && enable_shstk_type != cet_always_off);
      if (program)
	{
	  /* Enable IBT and SHSTK only if they are enabled in executable.
	     NB: IBT and SHSTK may be disabled by environment variable:

	     GLIBC_TUNABLES=glibc.cpu.hwcaps=-IBT,-SHSTK
	   */
	  enable_ibt &= (CPU_FEATURE_USABLE (IBT)
			 && (enable_ibt_type == cet_always_on
			     || (m->l_x86_feature_1_and
				 & GNU_PROPERTY_X86_FEATURE_1_IBT) != 0));
	  enable_shstk &= (CPU_FEATURE_USABLE (SHSTK)
			   && (enable_shstk_type == cet_always_on
			       || (m->l_x86_feature_1_and
				   & GNU_PROPERTY_X86_FEATURE_1_SHSTK) != 0));
	}

      /* ld.so is CET-enabled by kernel.  But shared objects may not
	 support IBT nor SHSTK.  */
      if (enable_ibt || enable_shstk)
	{
	  unsigned int i;

	  i = m->l_searchlist.r_nlist;
	  while (i-- > 0)
	    {
	      /* Check each shared object to see if IBT and SHSTK are
		 enabled.  */
	      l = m->l_initfini[i];

	      if (l->l_init_called)
		continue;

#ifdef SHARED
	      /* Skip CET check for ld.so since ld.so is CET-enabled.
		 CET will be disabled later if CET isn't enabled in
		 executable.  */
	      if (l == &GL(dl_rtld_map)
		  ||  l->l_real == &GL(dl_rtld_map)
		  || (program && l == m))
		continue;
#endif

	      /* IBT is enabled only if it is enabled in executable as
		 well as all shared objects.  */
	      enable_ibt &= (enable_ibt_type == cet_always_on
			     || (l->l_x86_feature_1_and
				 & GNU_PROPERTY_X86_FEATURE_1_IBT) != 0);
	      if (!found_ibt_legacy && enable_ibt != ibt_enabled)
		{
		  found_ibt_legacy = true;
		  ibt_legacy = i;
		}

	      /* SHSTK is enabled only if it is enabled in executable as
		 well as all shared objects.  */
	      enable_shstk &= (enable_shstk_type == cet_always_on
			       || (l->l_x86_feature_1_and
				   & GNU_PROPERTY_X86_FEATURE_1_SHSTK) != 0);
	      if (enable_shstk != shstk_enabled)
		{
		  found_shstk_legacy = true;
		  shstk_legacy = i;
		}
	    }
	}

      bool cet_feature_changed = false;

      if (enable_ibt != ibt_enabled || enable_shstk != shstk_enabled)
	{
	  if (!program)
	    {
	      if (enable_ibt_type != cet_permissive)
		{
		  /* When IBT is enabled, we cannot dlopen a shared
		     object without IBT.  */
		  if (found_ibt_legacy)
		    _dl_signal_error (0,
				      m->l_initfini[ibt_legacy]->l_name,
				      "dlopen",
				      N_("rebuild shared object with IBT support enabled"));
		}

	      if (enable_shstk_type != cet_permissive)
		{
		  /* When SHSTK is enabled, we cannot dlopen a shared
		     object without SHSTK.  */
		  if (found_shstk_legacy)
		    _dl_signal_error (0,
				      m->l_initfini[shstk_legacy]->l_name,
				      "dlopen",
				      N_("rebuild shared object with SHSTK support enabled"));
		}

	      if (enable_ibt_type != cet_permissive
		  && enable_shstk_type != cet_permissive)
		return;
	    }

	  /* Disable IBT and/or SHSTK if they are enabled by kernel, but
	     disabled in executable or shared objects.  */
	  unsigned int cet_feature = 0;

	  if (!enable_ibt)
	    cet_feature |= GNU_PROPERTY_X86_FEATURE_1_IBT;
	  if (!enable_shstk)
	    cet_feature |= GNU_PROPERTY_X86_FEATURE_1_SHSTK;

	  int res = dl_cet_disable_cet (cet_feature);
	  if (res != 0)
	    {
	      if (program)
		_dl_fatal_printf ("%s: can't disable CET\n", program);
	      else
		{
		  if (found_ibt_legacy)
		    l = m->l_initfini[ibt_legacy];
		  else
		    l = m->l_initfini[shstk_legacy];
		  _dl_signal_error (-res, l->l_name, "dlopen",
				    N_("can't disable CET"));
		}
	    }

	  /* Clear the disabled bits in dl_x86_feature_1.  */
	  GL(dl_x86_feature_1) &= ~cet_feature;

	  cet_feature_changed = true;
	}

#ifdef SHARED
      if (program && (ibt_enabled || shstk_enabled))
	{
	  if ((!ibt_enabled
	       || enable_ibt_type != cet_permissive)
	      && (!shstk_enabled
		  || enable_shstk_type != cet_permissive))
	    {
	      /* Lock CET if IBT or SHSTK is enabled in executable unless
	         IBT or SHSTK is enabled permissively.  */
	      int res = dl_cet_lock_cet ();
	      if (res != 0)
		_dl_fatal_printf ("%s: can't lock CET\n", program);
	    }

	  /* Set feature_1 if IBT or SHSTK is enabled in executable.  */
	  cet_feature_changed = true;
	}
#endif

      if (cet_feature_changed)
	{
	  unsigned int feature_1 = 0;
	  if (enable_ibt)
	    feature_1 |= GNU_PROPERTY_X86_FEATURE_1_IBT;
	  if (enable_shstk)
	    feature_1 |= GNU_PROPERTY_X86_FEATURE_1_SHSTK;
	  struct pthread *self = THREAD_SELF;
	  THREAD_SETMEM (self, header.feature_1, feature_1);
	}
    }
}

void
_dl_cet_open_check (struct link_map *l)
{
  dl_cet_check (l, NULL);
}

#ifdef SHARED

# ifndef LINKAGE
#  define LINKAGE
# endif

LINKAGE
void
_dl_cet_check (struct link_map *main_map, const char *program)
{
  dl_cet_check (main_map, program);
}
#endif /* SHARED */
